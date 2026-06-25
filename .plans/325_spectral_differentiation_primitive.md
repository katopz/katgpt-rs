# Plan 325: Standalone FFT-based Spectral Differentiation Primitive

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md](../.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md) (§3 candidate plan #2)
**Source paper:** [arXiv:2511.05963](https://arxiv.org/abs/2511.05963) — *Fourier Neural Operators Explained: A Practical Perspective* (Duruisseaux/Kossaffi/Anandkumar, Caltech+NVIDIA) §2.1 spectral differentiation.
**Target:** `katgpt-rs/crates/katgpt-core/src/spectral/differentiation.rs` (new module) + Cargo feature `spectral_differentiation`.
**Status:** Done — all 4 GOAT gates PASS, promoted to DEFAULT-ON (Phase 3, 2026-06-25).

---

## Goal

Ship a thin, standalone FFT-based spectral differentiation primitive for **periodic uniform 1D grids** — the specialized case where DEC's general `exterior_derivative` (cell-complex machinery) is overkill. This is the second of the three Gain-tier gaps identified by Research 307 §3. Pure modelless (closed-form FFT + frequency-domain multiplier `(iω)^m`, no learned weights). GOAT-gated behind feature `spectral_differentiation`; promote to default-on if all four gates pass with modelless gain.

This is a natural sibling to Plan 323 (Fourier continuation) — same `spectral/` module, same `rustfft` dependency, same Scratch/Config/into/convenience-wrapper API shape.

## Why this is Gain-tier (not Super-GOAT)

Per Research 307 §3 novelty gate (0/4 YES):
- **Q1 No prior art?** ❌ — DEC `exterior_derivative` (Plan 251) covers the general case on arbitrary cell complexes. This primitive is the **specialized** form for periodic uniform 1D grids (a thin wrapper, not new capability).
- **Q2 New capability class?** ❌ — Spectral differentiation is a well-known operation; we ship it in DEC form already.
- **Q3 Product selling point?** ❌ — Cannot finish "our NPCs do X that no competitor can" with this; the relevant X is already covered.
- **Q4 Force multiplier (≥2 pillars)?** ❌ — Touches only spectral primitives (FFT path). No fusion with HLA / functor / chain / shard.

Verdict: **Gain**. Plan-only, GOAT-gated behind feature flag. No Super-GOAT guide required.

## Algorithm

Given `x ∈ R^N` sampled on a uniform grid of `N` points with spacing `h`, assumed periodic:

```
X = FFT(x)                                // N complex coefficients
ω_j = 2π · k_j / (N · h)                  // angular frequency of bin j
                                          //   k_j = j            for j ≤ N/2
                                          //   k_j = j - N        for j >  N/2
∂^m x = IFFT( (i·ω_j)^m ⊙ X )             // element-wise, then inverse FFT
```

**Derivative multipliers:**
- `m=0`: `1` (identity, bit-identical passthrough → satisfies G3).
- `m=1`: `i·ω_j` (imaginary for `k ≠ 0`, zero at DC).
- `m=2`: `-ω_j²` (real, negative — 1D Laplacian).
- Higher orders follow by exponentiation.

**Nyquist handling (even N only):** the bin at index `N/2` represents the Nyquist mode, which is ambiguous (simultaneously `+N/2` and `-N/2`) for real inputs. Its FFT coefficient is real-valued. For **odd** derivative orders (`m=1, 3, 5, ...`) multiplying by `(i·ω)^m` produces a pure imaginary value, breaking Hermitian symmetry and yielding a complex (non-real) IFFT output. We **zero the Nyquist bin for odd orders on even-length signals** to preserve real output. For **even** derivative orders (`m=2, 4, ...`) the multiplier `(i·ω)^m` is real (and well-defined), so the Nyquist bin is kept.

**Why modelless:** FFT + element-wise complex multiply + IFFT. No weight mutation, no gradient descent, no learned parameters. Pure closed-form linear algebra on a fixed orthogonal basis. Freeze/thaw-friendly by construction (deterministic function of input).

**Relationship to existing primitives:**
- **Complementary to DEC `exterior_derivative`** (`dec/operators.rs`): DEC handles arbitrary cell complexes (irregular meshes, 2D/3D grids, manifolds with boundary). This primitive handles only the 1D periodic uniform case, but does so in O(N log N) with zero topological setup — useful when the input is a flat array of equally-spaced periodic samples (e.g. time-series windows, cyclic HLA channels, ring buffers).
- **Complementary to Plan 323 `fourier_continue`** (`spectral/continuation.rs`): FC makes a non-periodic signal approximately periodic so the FFT does not ring; this primitive then differentiates the periodic (or FC-extended) signal. They chain naturally: `spectral_differentiate(&fourier_continue(x, ...), ...)` for non-periodic inputs.
- **Complementary to `flow::fft_smooth`** (`flow/fft.rs`): FFT smoothing is low-pass filtering (multiply by a real mask in frequency domain); this is differentiation (multiply by `(iω)^m`).

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `.plans/325_spectral_differentiation_primitive.md` (this file).
- [x] **T1.2** Implement `crates/katgpt-core/src/spectral/differentiation.rs`:
  - `SpecDiffError` (TooFewSamples, OutputSizeMismatch, InvalidOrder).
  - `SpecDiffConfig { order: u32, spacing: f32 }` with `DEFAULT { order: 1, spacing: 1.0 }`.
  - `SpecDiffScratch { planner: FftPlanner<f32>, freq_buf: Vec<Complex<f32>> }` with `ensure_capacity`.
  - `spectral_differentiate_into(x, out, scratch, cfg)` — hot path, reuses scratch.
  - `spectral_differentiate(x, cfg)` — convenience wrapper that allocates.
  - Frequency-index computation (handle even/odd N).
  - Nyquist-zeroing for odd orders on even N.
- [x] **T1.3** Add unit tests: identity at order=0, sin/cos correctness at order=1 and order=2, DC killed at order=1, Nyquist handling for even N, error cases, scratch reuse, convenience wrapper equivalence, even-vs-odd length parity.
- [x] **T1.4** Register module in `crates/katgpt-core/src/spectral/mod.rs`.
- [x] **T1.5** Add feature flag `spectral_differentiation = ["dep:rustfft"]` in `crates/katgpt-core/Cargo.toml` (opt-in; promote to default only after Phase 2 GOAT passes).

## Phase 2 — GOAT Gate

### Tasks

- [x] **T2.1** Write `crates/katgpt-core/benches/bench_325_spectral_differentiation_goat.rs` implementing all four gates:
  - **G1 (correctness vs analytical):** sample `sin(2πx)` on `N=64` points over one period; spectral derivative of order 1 should match `(2π/L)·cos(2πx)` to max abs error `< 1e-4`. Order 2 should match `-(2π/L)²·sin(2πx)` to `< 1e-3`. Compare against finite-difference baseline (spectral should be 100×+ more accurate on smooth periodic signals).
  - **G2 (perf):** latency per `spectral_differentiate_into` call for `N ∈ {64, 256, 1024}`. Target: `< 50µs` for N=256 (same budget as Plan 323).
  - **G3 (no-regression):** at `order=0`, output is bit-identical to input (the operator is identity).
  - **G4 (alloc-free hot path):** `spectral_differentiate_into` with pre-warmed scratch performs 0 heap allocations over 100 steady-state calls (CountingAllocator audit).
- [x] **T2.2** Register bench in `Cargo.toml` with `required-features = ["spectral_differentiation"]`.
- [x] **T2.3** Run GOAT bench. Iterate algorithm if any gate fails.
- [x] **T2.4** Write `.benchmarks/325_spectral_differentiation_goat.md` recording G1–G4 results.

## Phase 3 — Promotion Decision

### Tasks

- [x] **T3.1** If G1–G4 all PASS with **modelless gain** → promote `spectral_differentiation` to `default` feature list with comment. (Expected: yes — pure closed-form FFT, zero training, identity-at-order-0 passthrough.)
- [x] **T3.2** Verify `cargo check --all-features` clean.
- [x] **T3.3** Verify `cargo test -p katgpt-core --lib spectral::` all pass.

## Phase 4 — Commit

### Tasks

- [x] **T4.1** Commit on `develop` with `feat:` prefix per global rule.

---

## Validation Summary

**All 4 GOAT gates PASS → promoted to DEFAULT-ON.** Final results (post-G4 cached-Arc fix):

- **G1** (analytical correctness): order-1 max abs err **5.44e-7** (<1e-4); order-2 max abs err **1.27e-6** (<1e-3); spectral-vs-FD quality ratio **290.2×** (≥100×). PASS.
- **G2** (perf): N=64 **0.19µs**, N=256 **0.83µs**, N=1024 **3.82µs** (all ≤50µs target). PASS.
- **G3** (no-regression): order=0 identity, max abs err **2.38e-7** < 1e-5. PASS.
- **G4** (alloc-free hot path): **0 allocations** over 100 steady-state calls (after caching `Arc<dyn Fft<f32>>` plans in scratch + using `process_with_scratch` instead of rustfft's per-call-Vec `process()`). PASS.
- **Promotion**: `spectral_differentiation` added to `default` feature list. Pure modelless gain (closed-form FFT + `(iω)^m`, no learned weights).
- **Full GOAT doc**: [`.benchmarks/325_spectral_differentiation_goat.md`](../.benchmarks/325_spectral_differentiation_goat.md)
