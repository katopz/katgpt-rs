#!/usr/bin/env python3
"""Convert transformer weights to CoreML .mlmodelc for ANE inference (Plan 176).

Usage:
    python scripts/convert_to_coreml.py --config micro --output model.mlmodelc
    python scripts/convert_to_coreml.py --config draft --output model.mlmodelc
    python scripts/convert_to_coreml.py --weights model.gguf --output model.mlmodelc

Requirements:
    pip install coremltools numpy

Based on the ane-book patterns:
    - Conv2d(1×1) trick for ANE-friendly matmuls
    - INT8 per-tensor quantization
    - FP16 compute for ANE
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import numpy as np
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models import MLModel
except ImportError:
    print("Error: coremltools and numpy required. Install with:")
    print("  pip install coremltools numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config presets — must match katgpt-core types::Config
# ---------------------------------------------------------------------------

CONFIGS = {
    # Config::micro(): vocab=27, block=16, n_layer=1, n_head=4, n_embd=16,
    # head_dim=4, mlp_hidden=64
    "micro": {
        "vocab_size": 27,
        "block_size": 16,
        "n_embd": 16,
        "n_head": 4,
        "head_dim": 4,
        "n_layer": 1,
        "n_kv_head": 4,
        "mlp_hidden": 64,
    },
    # Config::draft(): vocab=27, block=16, n_layer=1, n_head=2, n_embd=4,
    # head_dim=2, mlp_hidden=16
    "draft": {
        "vocab_size": 27,
        "block_size": 16,
        "n_embd": 4,
        "n_head": 2,
        "head_dim": 2,
        "n_layer": 1,
        "n_kv_head": 2,
        "mlp_hidden": 16,
    },
}


# ---------------------------------------------------------------------------
# Weight generation
# ---------------------------------------------------------------------------

def generate_random_weights(cfg: dict, seed: int = 42) -> dict:
    """Generate random weights for testing the conversion pipeline.

    Produces the same weight names that the full transformer graph would need:
      wte          — token embeddings       [vocab_size, n_embd]
      wpe          — position embeddings    [block_size, n_embd]
      lm_head      — output projection      [vocab_size, n_embd]
      per-layer:
        rms_norm_1 — pre-attention norm     [n_embd]
        wq         — Q projection           [n_embd, n_embd]
        wk         — K projection           [n_embd, kv_dim]
        wv         — V projection           [n_embd, kv_dim]
        wo         — output projection      [n_embd, n_embd]
        rms_norm_2 — pre-MLP norm           [n_embd]
        w1         — MLP up                 [n_embd, mlp_hidden]
        w2         — MLP down               [mlp_hidden, n_embd]
        w3         — MLP gate               [n_embd, mlp_hidden]
    """
    rng = np.random.default_rng(seed)
    V, B, D, H, HD = cfg["vocab_size"], cfg["block_size"], cfg["n_embd"], cfg["n_head"], cfg["head_dim"]
    kv_dim = cfg["n_kv_head"] * HD
    M = cfg["mlp_hidden"]

    weights = {
        "wte": rng.standard_normal((V, D)).astype(np.float32) * 0.02,
        "wpe": rng.standard_normal((B, D)).astype(np.float32) * 0.02,
        "lm_head": rng.standard_normal((V, D)).astype(np.float32) * 0.02,
    }

    for layer in range(cfg["n_layer"]):
        prefix = f"layer{layer}"
        weights[f"{prefix}.rms_norm_1"] = np.ones(D, dtype=np.float32)
        weights[f"{prefix}.wq"] = rng.standard_normal((D, D)).astype(np.float32) * 0.02
        weights[f"{prefix}.wk"] = rng.standard_normal((D, kv_dim)).astype(np.float32) * 0.02
        weights[f"{prefix}.wv"] = rng.standard_normal((D, kv_dim)).astype(np.float32) * 0.02
        weights[f"{prefix}.wo"] = rng.standard_normal((D, D)).astype(np.float32) * 0.02
        weights[f"{prefix}.rms_norm_2"] = np.ones(D, dtype=np.float32)
        weights[f"{prefix}.w1"] = rng.standard_normal((D, M)).astype(np.float32) * 0.02
        weights[f"{prefix}.w2"] = rng.standard_normal((M, D)).astype(np.float32) * 0.02
        weights[f"{prefix}.w3"] = rng.standard_normal((D, M)).astype(np.float32) * 0.02

    return weights


# ---------------------------------------------------------------------------
# Conv2d(1×1) — ANE-friendly matmul replacement
# ---------------------------------------------------------------------------

def make_conv2d_1x1(weight: np.ndarray, name: str, input_name: str,
                     output_name: str, builder):
    """Add a Conv2d(1×1) layer — ANE-friendly matmul replacement.

    ANE hardware has dedicated Conv2d acceleration. A fully-connected (matmul)
    layer expressed as Conv2d with 1×1 kernel gets ANE placement automatically.

    Args:
        weight: Shape [out_features, in_features] weight matrix.
        name: Layer name for CoreML.
        input_name: Input blob name.
        output_name: Output blob name.
        builder: NeuralNetworkBuilder instance.
    """
    # Reshape [out, in] → [out, in, 1, 1] for Conv2d convention
    w = weight.reshape(weight.shape[0], weight.shape[1], 1, 1).astype(np.float16)
    bias = np.zeros(weight.shape[0], dtype=np.float16)

    builder.add_convolution(
        name=name,
        kernel_channels=weight.shape[1],
        output_channels=weight.shape[0],
        height=1,
        width=1,
        stride_height=1,
        stride_width=1,
        border_mode="valid",
        groups=1,
        W=w,
        b=bias,
        has_bias=True,
        input_name=input_name,
        output_name=output_name,
    )


# ---------------------------------------------------------------------------
# CoreML model builder
# ---------------------------------------------------------------------------

def build_lm_head_model(cfg: dict, weights: dict) -> MLModel:
    """Build a minimal LM-head model (embedding + linear → logits).

    This demonstrates the core conversion pattern and is sufficient to test:
    1. Conv2d(1×1) matmul replacement for ANE placement
    2. FP16 weight storage
    3. CoreML model compilation and save

    The model takes a hidden-state vector [1, n_embd] and produces logits [1, vocab_size].
    """
    D = cfg["n_embd"]
    V = cfg["vocab_size"]

    input_features = [("hidden", ct.models.datatypes.Array(1, D))]
    output_features = [("logits", ct.models.datatypes.Array(1, V))]

    builder = NeuralNetworkBuilder(
        input_features, output_features,
        disable_rank5_shape_mapping=True,
    )

    # LM head as Conv2d(1×1) — the key ANE pattern
    make_conv2d_1x1(
        weight=weights["lm_head"],  # [V, D]
        name="lm_head",
        input_name="hidden",
        output_name="logits",
        builder=builder,
    )

    return MLModel(builder.spec)


def build_full_transformer_model(cfg: dict, weights: dict) -> MLModel:
    """Build the full transformer forward pass as a CoreML neural network.

    NOTE: This is a structured placeholder. The complete transformer graph
    requires careful attention to:
    - RoPE (positional encoding) or learned position embeddings
    - Multi-head attention with KV cache state management
    - RMSNorm (no learnable gain in micro config)
    - ReLU-gated MLP (w1 gate, w3 up, w2 down)
    - Residual connections

    For now, this builds a single attention + MLP layer with the Conv2d(1×1)
    trick applied to all linear projections. The embedding lookup and KV cache
    management are handled by the Rust runtime (AneBackend).
    """
    D = cfg["n_embd"]
    V = cfg["vocab_size"]
    H = cfg["n_head"]
    HD = cfg["head_dim"]
    kv_dim = cfg["n_kv_head"] * HD
    M = cfg["mlp_hidden"]

    # Input: hidden state [1, D] from Rust-side embedding lookup
    # Output: logits [1, V]
    input_features = [("hidden", ct.models.datatypes.Array(1, D))]
    output_features = [("logits", ct.models.datatypes.Array(1, V))]

    builder = NeuralNetworkBuilder(
        input_features, output_features,
        disable_rank5_shape_mapping=True,
    )

    cur = "hidden"

    for layer in range(cfg["n_layer"]):
        prefix = f"layer{layer}"

        # --- Pre-attention RMSNorm ---
        # RMSNorm = x / sqrt(mean(x²) + eps)
        # Implemented as: multiply -> reduce_mean -> add_eps -> sqrt -> divide
        sq_name = f"{prefix}.sq"
        mean_sq_name = f"{prefix}.mean_sq"
        rms_name = f"{prefix}.rms"
        eps_name = f"{prefix}.eps_add"
        sqrt_name = f"{prefix}.sqrt"
        norm_name = f"{prefix}.norm_out"

        builder.add_unary(
            name=sq_name, input_name=cur, output_name=sq_name,
            mode="square",
        )
        builder.add_reduce_mean(
            name=mean_sq_name, input_name=sq_name, output_name=mean_sq_name,
            axes=[1], keepdims=True,
        )
        builder.add_add_broadcastable(
            name=eps_name,
            input_names=[mean_sq_name],
            output_name=eps_name,
        )
        # NOTE: Full RMSNorm needs constant epsilon addition and sqrt/div.
        # For brevity, we skip the norm and go straight to projections.
        # A production version would implement the full norm subgraph here.

        # --- Attention projections (Conv2d 1×1) ---
        q_name = f"{prefix}.q"
        k_name = f"{prefix}.k"
        v_name = f"{prefix}.v"

        make_conv2d_1x1(
            weight=weights[f"{prefix}.wq"], name=f"{prefix}.wq_proj",
            input_name=cur, output_name=q_name, builder=builder,
        )
        make_conv2d_1x1(
            weight=weights[f"{prefix}.wk"], name=f"{prefix}.wk_proj",
            input_name=cur, output_name=k_name, builder=builder,
        )
        make_conv2d_1x1(
            weight=weights[f"{prefix}.wv"], name=f"{prefix}.wv_proj",
            input_name=cur, output_name=v_name, builder=builder,
        )

        # --- Attention output projection ---
        attn_out = f"{prefix}.attn_out"
        # NOTE: In a full implementation, Q/K/V would be reshaped for
        # multi-head attention, scaled dot-product computed, then wo applied.
        # For this placeholder, we combine qkv -> wo directly.
        make_conv2d_1x1(
            weight=weights[f"{prefix}.wo"], name=f"{prefix}.wo_proj",
            input_name=q_name, output_name=attn_out, builder=builder,
        )

        # --- Residual connection ---
        res1 = f"{prefix}.res1"
        builder.add_add_broadcastable(
            name=f"{prefix}.residual1",
            input_names=[cur, attn_out],
            output_name=res1,
        )

        # --- Pre-MLP RMSNorm (skipped for placeholder, same pattern as above) ---

        # --- MLP: gate(w3(x)) * up(w1(x)) -> down(w2) ---
        gate_name = f"{prefix}.gate"
        up_name = f"{prefix}.up"
        gated_name = f"{prefix}.gated"
        mlp_out = f"{prefix}.mlp_out"

        make_conv2d_1x1(
            weight=weights[f"{prefix}.w3"], name=f"{prefix}.gate_proj",
            input_name=res1, output_name=gate_name, builder=builder,
        )
        builder.add_unary(
            name=f"{prefix}.gate_relu", input_name=gate_name,
            output_name=gate_name, mode="relu",
        )
        make_conv2d_1x1(
            weight=weights[f"{prefix}.w1"], name=f"{prefix}.up_proj",
            input_name=res1, output_name=up_name, builder=builder,
        )
        builder.add_multiply_broadcastable(
            name=f"{prefix}.gate_mul",
            input_names=[gate_name, up_name],
            output_name=gated_name,
        )
        make_conv2d_1x1(
            weight=weights[f"{prefix}.w2"], name=f"{prefix}.down_proj",
            input_name=gated_name, output_name=mlp_out, builder=builder,
        )

        # --- Residual connection ---
        res2 = f"{prefix}.res2"
        builder.add_add_broadcastable(
            name=f"{prefix}.residual2",
            input_names=[res1, mlp_out],
            output_name=res2,
        )

        cur = res2

    # --- Final LM head ---
    make_conv2d_1x1(
        weight=weights["lm_head"], name="lm_head",
        input_name=cur, output_name="logits", builder=builder,
    )

    return MLModel(builder.spec)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def apply_int8_quantization(model: MLModel) -> MLModel:
    """Apply INT8 per-tensor symmetric quantization to linear layers.

    Reduces model size by ~2× (FP16 → INT8) with minimal quality loss.
    ANE supports INT8 execution natively for quantized weights.
    """
    try:
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            linear_quantize_weights,
        )
        config = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            weight_threshold=512,
        )
        model = linear_quantize_weights(model, config)
        print("  INT8 quantization applied")
        return model
    except (ImportError, Exception) as e:
        print(f"  Quantization skipped: {e}")
        return model


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert_weights_to_coreml(
    weights_path: str | None,
    output_path: str,
    config_name: str = "micro",
    quantize: bool = False,
    full_model: bool = False,
):
    """Convert transformer weights to CoreML .mlmodelc.

    Args:
        weights_path: Path to GGUF weight file (None → random weights for testing).
        output_path: Output .mlmodelc directory path.
        config_name: Config preset name matching katgpt-core types::Config.
        quantize: Apply INT8 per-tensor quantization.
        full_model: Build full transformer graph (otherwise just LM head for testing).
    """
    cfg = CONFIGS.get(config_name)
    if cfg is None:
        print(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
        sys.exit(1)

    print(f"Config: {config_name}")
    print(f"  vocab={cfg['vocab_size']}, block={cfg['block_size']}, "
          f"n_embd={cfg['n_embd']}, n_head={cfg['n_head']}, "
          f"head_dim={cfg['head_dim']}, n_layer={cfg['n_layer']}, "
          f"mlp_hidden={cfg['mlp_hidden']}")

    # Generate or load weights
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        # TODO (Plan 176 Part 7): implement GGUF loading via gguf-python
        # from gguf import GGUFReader
        # reader = GGUFReader(weights_path)
        print("  GGUF loading not yet implemented — using random weights")
        weights = generate_random_weights(cfg)
    else:
        print("Using random weights for testing")
        weights = generate_random_weights(cfg)

    # Build CoreML model
    print("Building CoreML model...")
    if full_model:
        model = build_full_transformer_model(cfg, weights)
        print("  Built full transformer graph (placeholder)")
    else:
        model = build_lm_head_model(cfg, weights)
        print("  Built LM-head model (Conv2d 1×1 pattern)")

    # Quantize if requested
    if quantize:
        print("Applying INT8 quantization...")
        model = apply_int8_quantization(model)

    # Save .mlmodelc
    output = Path(output_path)
    model_path = output if output.suffix == ".mlmodelc" else output / "model.mlmodelc"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    print(f"✅ CoreML model saved to: {model_path}")

    # Print model info
    spec = model.get_spec()
    inputs = [str(i.name) for i in spec.description.input]
    outputs = [str(o.name) for o in spec.description.output]
    print(f"   Inputs:  {inputs}")
    print(f"   Outputs: {outputs}")

    return model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert transformer weights to CoreML .mlmodelc for ANE inference (Plan 176)",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to GGUF weight file (random weights used if omitted)",
    )
    parser.add_argument(
        "--output", type=str, default="model.mlmodelc",
        help="Output .mlmodelc directory path",
    )
    parser.add_argument(
        "--config", type=str, default="micro", choices=list(CONFIGS.keys()),
        help="Model config preset (must match katgpt-core types::Config)",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Apply INT8 per-tensor quantization",
    )
    parser.add_argument(
        "--full-model", action="store_true",
        help="Build full transformer graph (default: LM-head only for testing)",
    )
    args = parser.parse_args()

    convert_weights_to_coreml(
        weights_path=args.weights,
        output_path=args.output,
        config_name=args.config,
        quantize=args.quantize,
        full_model=args.full_model,
    )


if __name__ == "__main__":
    main()
