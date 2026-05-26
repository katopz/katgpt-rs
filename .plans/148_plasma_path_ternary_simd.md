# Plan 148: PlasmaPath — Ternary SIMD Matvec (Ciot RIIR)

**Status:** Draft
**Research:** 110 (Ciot Ternary Inference Distillation)
**Related:** Plan 022 (Sparse MLP), Plan 055 (MTP Drafter), Plan 060 (SIMD matmul), Plan 066 (TileRT), Plan 103 (CODA fusion), Plan 131 (SpecHop), Issue 014 (Four-Tier Memory)
**Feature Gate:** `plasma_path` (opt-in, no default)
**GOAT Gates:** 5 (see below)

## Summary

Distill the core technique from [Cintu07/ciot](https://github.com/Cintu07/ciot) — bit-plane ternary weight encoding with branchless SIMD conditional accumulation — into `katgpt-core`. This adds a **Plasma** compute tier: multiplication-free ternary matvec using only SIMD add/subtract, targeting 2-3× throughput over our existing FP32 FMA path for CPU-bound speculative drafting.

## Five-Tier Hierarchy (aligned with Issue 014)

```
Tier       Compute                          Memory             Latency
────────   ─────────────────────────────── ───────────────── ──────────
Plasma     Ternary SIMD (add/sub only)     1.58 bits/weight   ~0.3ms/1024²
Hot        FP16/F32 SIMD (FMA)             16-32 bits/weight  ~0.5ms/1024²
Warm       SpectralQuant eigenbasis         3-4 bits/weight   ~0.8ms/1024²
Cold       Q4_K dequantize-on-read          4 bits/weight     ~1.2ms/1024²
Freeze     Disk-backed (Turso/libSQL)       Variable          ~10ms+
```

**Naming alignment:** Plasma is the "above Hot" tier — always in L1 cache / registers, never touches DRAM for weight reads (weights are bit-packed). Cold/Freeze map directly to Issue 014's Turso encrypted storage tiers.

## Tasks

### T1: TernaryWeights Type (`katgpt-core/src/types.rs`)

Add bit-plane ternary weight storage:

```rust
/// Bit-plane packed ternary weights: each element is {-1, 0, +1}.
///
/// 64 weights per block stored as two u64 bitmasks:
/// - pos_bits[block] bit k set → weight[row][k] = +1
/// - neg_bits[block] bit k set → weight[row][k] = -1
/// - both zero → weight = 0 (implicit skip, no storage needed)
///
/// `row_scale[r]` rescales the accumulated sum back toward original float magnitudes.
/// Memory: ~1.58 bits/weight (log₂3), plus one f32 per row for scale.
#[derive(Clone, Debug)]
pub struct TernaryWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,  // (cols + 63) / 64
    pub pos_bits: Vec<u64>,  // [rows * blocks64]
    pub neg_bits: Vec<u64>,  // [rows * blocks64]
    pub row_scale: Vec<f32>, // [rows]
}
```

Methods:
- `TernaryWeights::new(rows, cols)` — allocate zeroed
- `TernaryWeights::set(row, col, value: i8)` — set ternary value
- `TernaryWeights::get(row, col) -> i8` — get ternary value
- `TernaryWeights::quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self` — row-wise error-compensated quantization (from ciot's `pack_ternary.py`)

### T2: Scalar Ternary Matvec (`katgpt-core/src/simd.rs`)

Reference implementation for correctness testing:

```rust
/// Scalar reference: y[r] = row_scale[r] * Σ(col → sign(pos_bit, neg_bit) * x[col])
pub fn ternary_matvec_scalar(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    for r in 0..w.rows {
        let mut sum = 0.0f32;
        let row_base = r * w.blocks64;
        for c in 0..w.cols {
            let block = c >> 6;
            let bit = c & 63;
            let mask = 1u64 << bit;
            let idx = row_base + block;
            let pos = (w.pos_bits[idx] & mask) != 0;
            let neg = (w.neg_bits[idx] & mask) != 0;
            let sign = pos as i32 - neg as i32;
            sum += sign as f32 * x[c];
        }
        y[r] = sum * w.row_scale[r];
    }
}
```

### T3: NEON Ternary Matvec (`katgpt-core/src/simd.rs`)

RIIR of ciot's `matvec_ternary_native` for `target_arch = "aarch64"`:

```rust
#[cfg(target_arch = "aarch64")]
unsafe fn neon_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    // Per row:
    //   acc_pos += masked_load4(x, offset, (pos_bits >> chunk) & 0xF)
    //   acc_neg += masked_load4(x, offset, (neg_bits >> chunk) & 0xF)
    //   y[r] = hsum(acc_pos - acc_neg) * row_scale[r]
    //
    // masked_load4: vbslq_f32(bit_test, vld1q_f32(x+offset), zero)
    // No FMA. No multiply. Branchless.
}
```

Follow ciot's pattern: `vdupq_n_u32(bits)` → `vandq_u32` → `vcgtq_u32` → `vbslq_f32`.

### T4: AVX2 Ternary Matvec (`katgpt-core/src/simd.rs`)

RIIR of ciot's AVX2 path for `target_arch = "x86_64"`:

```rust
#[cfg(target_arch = "x86_64")]
unsafe fn avx2_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    // Per row, per 64-element block, per 8-element chunk:
    //   _mm256_and_ps(values, castsi256_ps(cmpgt_epi32(and_si256(word, lane_bits), zero)))
    //   acc_pos = _mm256_add_ps(acc_pos, masked_values)
    //   acc_neg = _mm256_add_ps(acc_neg, masked_values)
    //   sum = hsum256_ps(_mm256_sub_ps(acc_pos, acc_neg)) * row_scale
}
```

### T5: Dispatch Wrapper (`katgpt-core/src/simd.rs`)

```rust
/// SIMD-accelerated ternary matvec: y = W_ternary × x
///
/// Dispatches to NEON, AVX2, or scalar based on `simd_level()`.
/// All paths produce bit-identical results to `ternary_matvec_scalar()`.
#[cfg(feature = "plasma_path")]
pub fn simd_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    match simd_level() {
        SimdLevel::Neon => unsafe { neon_ternary_matvec(w, x, y) },
        SimdLevel::Avx2 => unsafe { avx2_ternary_matvec(w, x, y) },
        SimdLevel::Scalar => ternary_matvec_scalar(w, x, y),
    }
}
```

### T6: Batched Ternary Matmul (`katgpt-core/src/simd.rs`)

```rust
/// Batched ternary matmul: for each row batch[i], compute y[i] = W × batch[i].
/// Used for prompt-style multi-token processing.
#[cfg(feature = "plasma_path")]
pub fn simd_ternary_matmul_batch(w: &TernaryWeights, x: &[f32], batch: usize, y: &mut [f32]) {
    for b in 0..batch {
        let x_off = b * w.cols;
        let y_off = b * w.rows;
        simd_ternary_matvec(w, &x[x_off..], &mut y[y_off..]);
    }
}
```

### T7: `.bits` File Loader (`katgpt-rs/src/weights.rs`)

Load ciot-format `.bits` binary files:

```rust
/// Load a ciot-format .bits ternary weight file.
///
/// Format (little-endian):
///   magic      8 bytes  b"CIOTBIT1"
///   rows       u32
///   cols       u32
///   blocks64   u32
///   row_scale  rows × f32
///   pos_bits   rows × blocks64 × u64
///   neg_bits   rows × blocks64 × u64
#[cfg(feature = "plasma_path")]
pub fn load_ternary_bits(path: &std::path::Path) -> std::io::Result<TernaryWeights> { ... }
```

### T8: Forward Pass Dispatch (`katgpt-rs/src/transformer.rs`)

Add ternary weight dispatch to `forward_base()`:

```rust
#[cfg(feature = "plasma_path")]
if let Some(tw) = &layer.ternary_wq {
    simd_ternary_matvec(tw, &input, &mut q_buf);
} else {
    matmul(&mut q_buf, &layer.attn_wq, &input, ...);
}
```

Each `LayerWeights` optionally carries ternary variants:
```rust
pub struct LayerWeights {
    // ... existing fields ...
    #[cfg(feature = "plasma_path")]
    pub ternary_wq: Option<TernaryWeights>,
    // ... same for wk, wv, wo, w1, w2 ...
}
```

### T9: Quantization Utility (`katgpt-core/src/types.rs`)

Row-wise error-compensated ternary quantization (from ciot's `pack_ternary.py`):

```rust
impl TernaryWeights {
    /// Quantize f32 weights to ternary with row-wise error compensation.
    ///
    /// For each row:
    ///   scale = mean(|row|)
    ///   threshold = 0.5 * scale
    ///   for each weight: adjusted = value + carry
    ///     if adjusted > threshold → +1
    ///     if adjusted < -threshold → -1
    ///     else → 0
    ///     carry = adjusted - (q * scale)
    pub fn quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self { ... }
}
```

### T10: GOAT Proof Tests

| Gate | Test | Threshold |
|------|------|-----------|
| G1 | Scalar vs NEON checksum parity | bit-exact match |
| G2 | Scalar vs AVX2 checksum parity | bit-exact match |
| G3 | Roundtrip: quantize → matvec → verify | cosine sim ≥ 0.90 vs f32 matmul |
| G4 | Throughput: 1024×1024 ternary matvec | ≥ 2× vs `simd_dot_f32` scalar |
| G5 | Graceful degradation: `plasma_path` disabled | builds + runs, no ternary access |

### T11: Benchmark Harness

Add to `katgpt-rs/src/benchmark.rs`:
- `bench_ternary_matvec_256` (256×256)
- `bench_ternary_matvec_512` (512×512)
- `bench_ternary_matvec_1024` (1024×1024 hero number)
- `bench_ternary_vs_f32` (compare ternary vs existing `simd_matvec`)

Report: median ms, p95, throughput (Gop/s), checksum.

## GOAT Proof

| Gate | Proof | Threshold |
|------|-------|-----------|
| G1 | NEON ternary == scalar ternary | Checksum exact match |
| G2 | AVX2 ternary == scalar ternary | Checksum exact match |
| G3 | Quantize fidelity | Cosine sim ≥ 0.90 vs f32 on random weights |
| G4 | Throughput gain | ≥ 1.5× vs existing FP32 SIMD dot on same data |
| G5 | Graceful without feature | Compiles + passes tests with `plasma_path` off |

## Estimated Effort

~4-5 days:
- T1-T2 (types + scalar): 0.5 day
- T3-T5 (NEON/AVX2/dispatch): 1.5 days (direct RIIR from ciot source)
- T6 (batched): 0.5 day
- T7 (loader): 0.5 day
- T8 (forward dispatch): 0.5 day
- T9 (quantization): 0.5 day
- T10-T11 (GOAT + bench): 1 day

## Feature Gate

```toml
# katgpt-core/Cargo.toml
plasma_path = []

# katgpt-rs/Cargo.toml
plasma_path = ["katgpt-core/plasma_path"]
```

No default enable. Opt-in until GOAT proof passes.

## What Stays in riir-ai (Private)

- Per-game ternary threshold tuning (which layers to quantize)
- `.bits` weight files for game-specific models
- PlasmaPath + DDTree speculative decode integration parameters
- Ternary + Sparse MLP fusion heuristics

These are the "plugs" that make the open "sockets" valuable. Without the private tuning data, the open kernel is generic. With it, game-specific inference is dramatically faster.

## Dependency Graph

```
T1 (TernaryWeights type)
├── T2 (scalar matvec)
│   ├── T3 (NEON) ──┐
│   ├── T4 (AVX2) ──┼── T5 (dispatch) ── T6 (batched)
│   └────────────────┘
├── T9 (quantize)
└── T7 (loader) ── T8 (forward dispatch) ── T10 (GOAT) ── T11 (bench)
```
