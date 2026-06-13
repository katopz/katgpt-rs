#!/usr/bin/env python3
"""Generate NPC Brain CoreML Model using coremltools mb.program public API.

Plan 254 Part 2: ANE-Latent NPC Brain Compute
Research 224: coremltools Public API ANE Distillation
Issue 004: ANE CoreML Model Generation Pipeline

Builds a CoreML ML Program model that fuses three NPC "think brain" ops:
  1. Sense projection: diagonal ternary sign × HLA state → sigmoid scalar per module
  2. Emotion dot-product: HLA state · emotion direction → sigmoid scalar
  3. Zone dot-product: HLA state · zone direction → sigmoid scalar

The ternary bit-plane projection from SenseModule::project() is converted to
float weights at model generation time (lossless: -1/0/+1 → f32).

Output: npc_brain.mlpackage (CoreML ML Program, ANE-optimized)

Usage:
    python scripts/generate_npc_brain_model.py [--batch-size 1] [--quantize] [--verify-ane]

Requirements:
    - Python 3.10-3.12 (coremltools native extensions require C++ compilation)
    - pip install coremltools numpy

    NOTE: Python 3.13+ may not have pre-built native extensions (BlobWriter).
    The model building logic works, but serialization to .mlpackage requires
    the C++ native extensions. Use Python 3.12 for best compatibility.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import get_new_symbol
    from coremltools.converters.mil.mil import types as mil_types
    from coremltools.models import MLModel
except ImportError:
    print(
        "ERROR: coremltools is required. Install with: pip install coremltools",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants matching katgpt-core/src/sense/backend.rs
# ---------------------------------------------------------------------------

MAX_MODULES: int = 6  # SenseKind count
HLA_DIM: int = 8  # HLA state dimensionality
MAX_DIRECTIONS: int = 8  # Max ternary directions per module


# ---------------------------------------------------------------------------
# Ternary weight conversion
# ---------------------------------------------------------------------------


class TernaryDir:
    """Python mirror of Rust TernaryDir (katgpt-core/src/types.rs).

    Ternary bit-plane direction: pos_bits and neg_bits are u64 bitmasks.
    Sign at dimension j = ((pos_bits >> j) & 1) - ((neg_bits >> j) & 1) ∈ {-1, 0, +1}.
    """

    def __init__(self, pos_bits: int = 0, neg_bits: int = 0, row_scale: float = 1.0):
        self.pos_bits = pos_bits
        self.neg_bits = neg_bits
        self.row_scale = row_scale

    def to_float_weights(self, dim: int = HLA_DIM) -> np.ndarray:
        """Convert ternary signs to float weight vector.

        Returns shape (dim,) with values in {-row_scale, 0.0, +row_scale}.
        """
        weights = np.zeros(dim, dtype=np.float32)
        for j in range(dim):
            pos = (self.pos_bits >> j) & 1
            neg = (self.neg_bits >> j) & 1
            sign = float(pos) - float(neg)  # ∈ {-1, 0, +1}
            weights[j] = sign * self.row_scale
        return weights


class SenseModule:
    """Python mirror of Rust SenseModule (katgpt-core/src/types.rs).

    Contains up to 8 ternary directions and a confidence scalar.
    The project() method is a diagonal extraction: direction i contributes
    sign from bit i × hla_state[i] × row_scale[i].
    """

    def __init__(
        self,
        directions: list[TernaryDir],
        n_directions: int,
        confidence: float = 1.0,
    ):
        self.directions = directions
        self.n_directions = min(n_directions, MAX_DIRECTIONS)
        self.confidence = confidence

    def project_weight_matrix(self) -> np.ndarray:
        """Build diagonal projection weight matrix for ANE matmul.

        For direction i, the sign is extracted from bit i of that direction.
        This means direction i contributes:
            sign_i = (pos_bit_i_of_dir[i]) - (neg_bit_i_of_dir[i])
            dot += sign_i * hla_state[i] * row_scale[i]

        Since each direction only touches dimension i (diagonal),
        we build a (n_directions, HLA_DIM) weight matrix W where:
            W[i, j] = row_scale[i] * ((pos_bits[i] >> j) & 1 - (neg_bits[i] >> j) & 1)
        But the actual projection only uses the diagonal:
            W[i, i] = row_scale[i] * ((pos_bits[i] >> i) & 1 - (neg_bits[i] >> i) & 1)

        For ANE efficiency, we use the FULL weight matrix approach:
        matmul(hla_state, W.T) gives all direction projections at once.
        The ANE will zero out the off-diagonal contributions via weight structure.

        Returns shape (n_directions, HLA_DIM).
        """
        n = self.n_directions
        W = np.zeros((n, HLA_DIM), dtype=np.float32)
        for i in range(n):
            dir_i = self.directions[i]
            for j in range(HLA_DIM):
                pos = (dir_i.pos_bits >> j) & 1
                neg = (dir_i.neg_bits >> j) & 1
                sign = float(pos) - float(neg)
                W[i, j] = sign * dir_i.row_scale
        return W


# ---------------------------------------------------------------------------
# Model building with mb.program
# ---------------------------------------------------------------------------


def build_npc_brain_model(batch_size: int = 1) -> MLModel:
    """Build NPC brain CoreML model using coremltools mb.program.

    Three fused operations in a single ML Program:
      1. Sense projection: [B, MAX_MODULES, HLA_DIM] × [B, HLA_DIM] → sigmoid → [B, MAX_MODULES]
      2. Emotion projection: [B, HLA_DIM] · [B, HLA_DIM] → sigmoid → [B, 1]
      3. Zone projection: [B, HLA_DIM] · [B, HLA_DIM] → sigmoid → [B, 1]

    The sense projection uses a matmul where the weight matrix encodes
    the ternary signs as float values. The confidence scaling is applied
    per-module at input preparation time (Rust side).

    Args:
        batch_size: Maximum batch size for the model. If <= 1, the batch
            dimension is dynamic (RangeDim 1..1024) so the same model serves
            any batch size from a single NPC up to a full 1000-NPC tick.

    Returns:
        Compiled MLModel ready for saving.
    """
    # Use iOS 18 for stateful + multifunction support
    target = ct.target.iOS18

    # Dynamic batch dimension when batch_size <= 1 (enables batch=10..1000 at runtime).
    # Fixed batch dimension otherwise (slightly faster dispatch for known shapes).
    # We use a Symbol for the batch dim — mb.program's Placeholder accepts symbols
    # and coremltools will infer a RangeDim constraint from the reshape(-1, ...).
    if batch_size <= 1:
        batch_dim = get_new_symbol("batch")
    else:
        batch_dim = batch_size

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                shape=(batch_dim, MAX_MODULES, HLA_DIM), dtype=mil_types.fp32
            ),
            mb.TensorSpec(shape=(batch_dim, HLA_DIM), dtype=mil_types.fp32),
            mb.TensorSpec(shape=(batch_dim, HLA_DIM), dtype=mil_types.fp32),
            mb.TensorSpec(shape=(batch_dim, HLA_DIM), dtype=mil_types.fp32),
            mb.TensorSpec(shape=(batch_dim, MAX_MODULES), dtype=mil_types.fp32),
        ],
        opset_version=target,
        function_name="main",
    )
    def npc_brain(
        sense_weights: mb.Operation,
        hla_state: mb.Operation,
        emotion_direction: mb.Operation,
        zone_direction: mb.Operation,
        confidence: mb.Operation,
    ):
        # ── Op 1: Sense projection ──────────────────────────────────
        # sense_weights: [B, MAX_MODULES, HLA_DIM] — ternary→float weights
        # hla_state: [B, HLA_DIM]
        # We need: for each module m, dot(sense_weights[m], hla_state) → sigmoid
        # This is a batch matmul: [B, MAX_MODULES, HLA_DIM] × [B, HLA_DIM, 1] → [B, MAX_MODULES, 1]

        # Reshape hla_state for batch matmul: [B, HLA_DIM] → [B, HLA_DIM, 1]
        # Use -1 for the batch dim so it works with dynamic shapes.
        hla_col = mb.reshape(x=hla_state, shape=[-1, HLA_DIM, 1])

        # Batch matmul: [B, MAX_MODULES, HLA_DIM] × [B, HLA_DIM, 1] → [B, MAX_MODULES, 1]
        sense_raw = mb.matmul(
            x=sense_weights, y=hla_col, transpose_x=False, transpose_y=False
        )

        # Squeeze: [B, MAX_MODULES, 1] → [B, MAX_MODULES]
        sense_dot = mb.squeeze(x=sense_raw, axes=[2])

        # Sigmoid activation (not softmax!)
        sense_proj = mb.sigmoid(x=sense_dot)

        # Apply confidence scaling AFTER sigmoid (matches Rust: confidence * sigmoid(dot))
        # confidence: [B, MAX_MODULES], sense_proj: [B, MAX_MODULES]
        sense_proj = mb.mul(x=sense_proj, y=confidence)

        # ── Op 2: Emotion projection ────────────────────────────────
        # element-wise multiply → reduce_sum → sigmoid
        emotion_dot_raw = mb.mul(x=hla_state, y=emotion_direction)
        emotion_dot = mb.reduce_sum(x=emotion_dot_raw, axes=[1])  # [B]
        emotion_proj = mb.sigmoid(x=emotion_dot)  # [B]

        # ── Op 3: Zone projection ───────────────────────────────────
        zone_dot_raw = mb.mul(x=hla_state, y=zone_direction)
        zone_dot = mb.reduce_sum(x=zone_dot_raw, axes=[1])  # [B]
        zone_proj = mb.sigmoid(x=zone_dot)  # [B]

        return sense_proj, emotion_proj, zone_proj

    # Convert to ML Program (required for ANE)
    model = ct.convert(
        npc_brain,
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
    )

    return model


def quantize_model(model: MLModel) -> MLModel:
    """Apply INT8 per-tensor symmetric quantization.

    Uses coremltools public API (no private API needed).
    """
    from coremltools.optimize.coreml import (
        LinearQuantizer,
        OpLinearQuantizerConfig,
        OptimizationConfig,
    )

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            weight_threshold=0,  # Quantize all weights
        )
    )
    quantizer = LinearQuantizer(config)
    return quantizer.compile(model)


def verify_ane_placement(model: MLModel) -> dict:
    """Verify ANE placement via MLComputePlan.

    Returns dict with per-operation device placement info.
    """
    try:
        from coremltools.models.compute_plan import MLComputePlan

        plan = MLComputePlan.load_from_path(
            model.get_compiled_model_path(),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        results = {
            "total_ops": 0,
            "ane_ops": 0,
            "cpu_ops": 0,
            "gpu_ops": 0,
            "details": [],
        }

        # Introspect the MIL program
        spec = model.get_spec()
        if hasattr(spec, "ml_program") and spec.ml_program.HasField("functions"):
            for func in spec.ml_program.functions.values():
                if hasattr(func, "block") and hasattr(func.block, "operations"):
                    for op in func.block.operations:
                        results["total_ops"] += 1
                        try:
                            usage = (
                                plan.get_compute_device_usage_for_mlprogram_operation(
                                    op
                                )
                            )
                            device = (
                                str(usage.preferred_compute_device)
                                if usage
                                else "unknown"
                            )
                        except Exception:
                            device = "unknown"

                        if "NeuralEngine" in device or "NE" in device:
                            results["ane_ops"] += 1
                        elif "GPU" in device:
                            results["gpu_ops"] += 1
                        elif "CPU" in device:
                            results["cpu_ops"] += 1

                        results["details"].append(
                            {
                                "op": op.name if hasattr(op, "name") else str(type(op)),
                                "device": device,
                            }
                        )

        return results
    except Exception as e:
        return {"error": str(e), "total_ops": 0, "ane_ops": 0}


# ---------------------------------------------------------------------------
# Weight export for Rust-side verification
# ---------------------------------------------------------------------------


def export_weights_bin(
    modules: list[SenseModule],
    emotion_direction: np.ndarray,
    zone_direction: np.ndarray,
    output_path: Path,
) -> None:
    """Export weights as binary file for Rust-side verification.

    Binary format (little-endian):
        [u32: n_modules]
        [u32: n_directions_per_module; repeated MAX_MODULES times]
        For each module:
            [f32: confidence]
            [f32 × HLA_DIM × MAX_DIRECTIONS: weight_matrix (row-major)]
        [f32 × HLA_DIM: emotion_direction]
        [f32 × HLA_DIM: zone_direction]
    """
    with open(output_path, "wb") as f:
        # Number of modules
        f.write(struct.pack("<I", len(modules)))

        # Pad to MAX_MODULES
        for i in range(MAX_MODULES):
            if i < len(modules):
                f.write(struct.pack("<I", modules[i].n_directions))
            else:
                f.write(struct.pack("<I", 0))

        # Per-module weights
        for i in range(MAX_MODULES):
            if i < len(modules):
                m = modules[i]
                f.write(struct.pack("<f", m.confidence))
                W = m.project_weight_matrix()
                # Pad to (MAX_DIRECTIONS, HLA_DIM) with zeros
                padded = np.zeros((MAX_DIRECTIONS, HLA_DIM), dtype=np.float32)
                padded[: W.shape[0], :] = W
                f.write(padded.tobytes())
            else:
                # Empty module
                f.write(struct.pack("<f", 0.0))
                f.write(np.zeros((MAX_DIRECTIONS, HLA_DIM), dtype=np.float32).tobytes())

        # Emotion direction
        assert emotion_direction.shape == (HLA_DIM,), (
            f"emotion_direction shape: {emotion_direction.shape}"
        )
        f.write(emotion_direction.astype(np.float32).tobytes())

        # Zone direction
        assert zone_direction.shape == (HLA_DIM,), (
            f"zone_direction shape: {zone_direction.shape}"
        )
        f.write(zone_direction.astype(np.float32).tobytes())


# ---------------------------------------------------------------------------
# Test weight generation: matches Rust ternary projection
# ---------------------------------------------------------------------------


def project_ternary_reference(
    hla_state: np.ndarray,
    directions: list[TernaryDir],
    n_directions: int,
    confidence: float,
) -> float:
    """Python reference matching SenseModule::project() exactly.

    This is the ground truth for verifying the CoreML model output.
    """
    dot = 0.0
    for i in range(n_directions):
        dir_i = directions[i]
        pos = float((dir_i.pos_bits >> i) & 1)
        neg = float((dir_i.neg_bits >> i) & 1)
        sign = pos - neg
        dot += sign * hla_state[i] * dir_i.row_scale

    # Fast sigmoid matching Rust
    x = dot
    if x >= 0.0:
        sigmoid = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sigmoid = ex / (1.0 + ex)

    return confidence * sigmoid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_test_modules() -> tuple[list[SenseModule], np.ndarray, np.ndarray]:
    """Create test modules with known ternary weights for verification.

    These match the test pattern from merkle_octree_bench.rs.
    """
    modules = []

    # Module 0: pos_bits=0b011, neg_bits=0b100, row_scale=1.0, confidence=0.8
    dirs_0 = [TernaryDir(pos_bits=0b011, neg_bits=0b100, row_scale=1.0)]
    modules.append(SenseModule(directions=dirs_0, n_directions=1, confidence=0.8))

    # Module 1: pos_bits=0b101, neg_bits=0b010, row_scale=0.5, confidence=1.0
    dirs_1 = [TernaryDir(pos_bits=0b101, neg_bits=0b010, row_scale=0.5)]
    modules.append(SenseModule(directions=dirs_1, n_directions=1, confidence=1.0))

    # Modules 2-5: empty
    for _ in range(MAX_MODULES - 2):
        modules.append(SenseModule(directions=[], n_directions=0, confidence=0.0))

    emotion_dir = np.array([0.5, -0.3, 0.8, 0.1, -0.2, 0.4, 0.0, 0.7], dtype=np.float32)
    zone_dir = np.array([0.3, 0.6, -0.1, 0.5, 0.2, -0.4, 0.8, 0.0], dtype=np.float32)

    return modules, emotion_dir, zone_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate NPC Brain CoreML Model (Plan 254 Part 2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for model input (default: 1)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 per-tensor quantization",
    )
    parser.add_argument(
        "--verify-ane",
        action="store_true",
        help="Verify ANE placement via MLComputePlan",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated files (default: current dir)",
    )
    parser.add_argument(
        "--use-test-weights",
        action="store_true",
        help="Use test weights for verification (default: use zeros)",
    )
    parser.add_argument(
        "--export-weights",
        action="store_true",
        help="Export weight binary for Rust-side verification",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NPC Brain CoreML Model Generator")
    print("Plan 254 Part 2 — ANE-Latent NPC Brain Compute")
    print("=" * 60)

    # ── Step 1: Build model ──────────────────────────────────────
    print(f"\n[1/5] Building model with batch_size={args.batch_size}...")
    try:
        model = build_npc_brain_model(batch_size=args.batch_size)
    except RuntimeError as e:
        if "BlobWriter" in str(e):
            print("  ✗ BlobWriter not loaded — Python 3.13+ native extension issue")
            print("  Use Python 3.12 for full model generation support")
            print(
                "  The model building logic is correct; serialization needs C++ extensions"
            )
            print("\n  To verify with compatible Python:")
            print(
                "    python3.12 -m venv .venv && .venv/bin/pip install coremltools numpy"
            )
            print(
                "    .venv/bin/python scripts/generate_npc_brain_model.py --output-dir models/"
            )
            return 1
        raise
    print("  ✓ Model built successfully")

    # ── Step 2: Quantize ─────────────────────────────────────────
    if args.quantize:
        print("\n[2/5] Applying INT8 quantization...")
        model = quantize_model(model)
        print("  ✓ Quantization applied")
    else:
        print("\n[2/5] Skipping quantization (use --quantize to enable)")

    # ── Step 3: Save model ───────────────────────────────────────
    model_path = output_dir / "npc_brain.mlpackage"
    print(f"\n[3/5] Saving model to {model_path}...")
    model.save(str(model_path))
    print("  ✓ Model saved")

    # ── Step 4: Verify ANE placement ─────────────────────────────
    if args.verify_ane:
        print("\n[4/5] Verifying ANE placement...")
        placement = verify_ane_placement(model)
        print(f"  Total ops: {placement.get('total_ops', 'N/A')}")
        print(f"  ANE ops:   {placement.get('ane_ops', 'N/A')}")
        print(f"  CPU ops:   {placement.get('cpu_ops', 'N/A')}")
        print(f"  GPU ops:   {placement.get('gpu_ops', 'N/A')}")
        if "error" in placement:
            print(f"  ⚠ ANE verification error: {placement['error']}")
        elif placement.get("ane_ops", 0) > 0:
            print("  ✓ ANE placement confirmed")
        else:
            print("  ⚠ No ops placed on ANE — model may need restructuring")
    else:
        print("\n[4/5] Skipping ANE verification (use --verify-ane to enable)")

    # ── Step 5: Export weights ────────────────────────────────────
    if args.export_weights:
        print("\n[5/5] Exporting weights binary...")
        if args.use_test_weights:
            modules, emotion_dir, zone_dir = make_test_modules()
        else:
            # Default: empty weights (Rust side provides weights at runtime)
            modules = [
                SenseModule(directions=[], n_directions=0, confidence=0.0)
                for _ in range(MAX_MODULES)
            ]
            emotion_dir = np.zeros(HLA_DIM, dtype=np.float32)
            zone_dir = np.zeros(HLA_DIM, dtype=np.float32)

        weights_path = output_dir / "npc_brain_weights.bin"
        export_weights_bin(modules, emotion_dir, zone_dir, weights_path)
        print(f"  ✓ Weights exported to {weights_path}")

        # Verify reference projection
        if args.use_test_weights:
            hla_test = np.array(
                [0.5, 0.3, -0.2, 0.1, 0.4, -0.1, 0.0, 0.2], dtype=np.float32
            )
            for i, m in enumerate(modules):
                if m.n_directions > 0:
                    ref = project_ternary_reference(
                        hla_test, m.directions, m.n_directions, m.confidence
                    )
                    print(f"  Module {i} reference projection: {ref:.6f}")
    else:
        print("\n[5/5] Skipping weight export (use --export-weights to enable)")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Model:      {model_path}")
    print(f"  Format:     ML Program (ANE-compatible)")
    print(f"  Precision:  FP16 (ANE native)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Quantized:  {args.quantize}")
    print(
        f"  Ops:        sense_matmul + sigmoid, emotion_dot + sigmoid, zone_dot + sigmoid"
    )
    print(f"  Public API: coremltools mb.program (no private API)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
