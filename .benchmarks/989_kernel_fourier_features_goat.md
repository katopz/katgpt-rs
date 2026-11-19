# Bench 989: kernel_fourier_features GOAT gate (Issue 1234, Research 682)

**Status:** DONE — G1 / G1c / G1b / G2 / G4 PASS; the open-BC eigenfunction check PASSES. Bars pre-registered in Issue 1234 before the first run.

## Provenance

- G1/G1c/G1b/G4 + open-BC: m3, release, `cargo test --release -p katgpt-core --features kernel_fourier_features --test kernel_fourier_features_goat -- --test-threads 1` at `4eb832de9`. Accuracy-only (deterministic), so no box-state bar applies.
- G2 (latency): amo, rig job `f3df720a` (Exclusive lane), lattice katgpt-rs `19fa122b7`, `-- --ignored g2_`. `PROVENANCE: power=AC Power load=2.55 swap=3905.12M canary=skipped powermode=0(auto) ane=ok` (preflight PASSED).

## Results

| gate | bar | measured | verdict |
|---|---|---|---|
| G1 round-trip | full-width ΦΦᵀ vs centered sampled kernel, N=24, max abs < 1e-5 | σ=1 1.07e-7 · σ=4.2 2.69e-8 · σ=12 9.43e-9 | PASS (cannot fail by construction — disclosed in the issue) |
| G1c width law | rel Frobenius err ≤ 1.5·sqrt(tail a² / total a²), w ∈ {1,2,4,8,11} | err 0.0116–0.306 vs bound 0.108–0.957, every (σ, w) | PASS |
| G1b phase readout | data-side (centered Gram top conjugate eigen-pair) error ≤ 1/(2w), w ∈ {2,4,8} | ≤ 1.22e-7 at every σ; pairing score (top 8) = 1.00 | PASS |
| G2 O(N·w) | ns/(N·w) spread < 3× across (N,w) ∈ {(256,8),(1024,8),(1024,32),(4096,32)} | 10.09 / 5.03 / 4.76 / 5.14 ns → 2.12× | PASS |
| G4 | spectrum build + features + readout allocate nothing | 0 | PASS |
| open BC | feature columns ≍ fine-grid (M=401) eigenfunctions, raw + centered, σ ∈ {0.5,1,4.2}, top 4 | |corr| > 0.999 every cell | PASS |

Absolute G2 cell: N=4096, w=32 builds in 674 µs (5.1 ns per cos/sin pair). The N=256 cell's 10 ns/unit is fixed per-call overhead amortized over a small N.

## Disclosed

1. **G1c is conservative by construction.** The Frobenius error weights the dropped tail by a⁴, while the bound uses a². For a decaying spectrum the a⁴ tail share is ≤ the a² share, so the bound always holds with margin (measured 3–9×). It certifies "never worse than the tail law", not the law's tightness.
2. **G1b is noise-free.** Exact circulant data makes the readout exact to f32. Robustness to noisy statistics (the paper's §4 collective-eigenvalue claim) is NOT measured here.
3. **Open-BC even quantization (spec correction):** the issue's `tan k = k/(1+σ(1+σ)k²)` is the DOUBLE-CENTERED kernel's condition. The raw kernel quantizes `tan k = 1/(σk)` (measured by fine-grid eigendecomposition). Both ship as `OpenKernel::{Raw, Centered}`.

## Verdict

Opt-in (`kernel_fourier_features`). The construction, width law (as a conservative bound), readout and fingerprint are gated. Promotion needs a wired consumer: riir-ai Issue 1245 (ring latents) is the named candidate.
