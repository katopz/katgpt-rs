# Plan 301: Runtime Subspace Phase-Gate Primitive — Generic Math for N≥d + Jacobian SVD

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md](../.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md)
**Source paper:** [arXiv:2409.02426](https://arxiv.org/abs/2409.02426) — Wang et al., *Breaking the Curse of Dimensionality*.
**Private Super-GOAT guide:** `riir-neuron-db/.research/001_Subspace_Consolidation_Quality_Gate_Guide.md`
**Target:** `katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs` (new module) + Cargo feature `subspace_phase_gate`
**Status:** Active — Phase 1 complete (skeleton shipped), Phase 2 (G1 GOAT proof) complete with [Bench 301](../.benchmarks/301_subspace_phase_gate_g1.md) **G1 PASS**, Phases 3-5 deferred.

---

## Goal

Ship a generic, modelless numeric primitive that exposes three operations, all inference-time and allocation-aware:

1. **`participation_ratio(spectrum)`** — effective dimensionality `d_eff = (Σλ)² / Σ(λ²)` from an eigenvalue/singular-value spectrum. Already shipped in `spectralquant::spectral` but re-exposed here under a feature flag for consumers that don't pull in `spectral_quant`.
2. **`numerical_rank(spectrum, η)`** — smallest `r` such that `Σ_{i≤r} σ_i² / Σ_i σ_i² > η` (paper eq. 52, η = 0.99).
3. **`phase_transition_gate(n_samples, intrinsic_dim)` → bool** — returns `n_samples >= intrinsic_dim`. The Wang et al. Theorem 4 necessary condition for subspace recovery.
4. **`jacobian_svd_at(f, x, ε, scratch)`** — estimate the Jacobian of map `f: R^n → R^m` at point `x` via forward differences, return the leading singular vectors/values. Generic over the map (closure), no game/shard semantics.

This is the **open** counterpart of the private Super-GOAT at `riir-neuron-db/.research/001`. The wrapper that applies this to `NeuronShard` lives in `riir-neuron-db` (Plan 002). The wrapper that applies this to `evolve_hla()` lives in `riir-ai` (future plan).

**GOAT gate:** G1 (phase transition reproduces on synthetic MoLRG) must pass before promoting to default.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs` with module doc referencing arXiv:2409.02426 and the open research note R279.
- [x] **T1.2** Implement `pub fn participation_ratio(spectrum: &[f32]) -> f32` — `(Σλ)² / Σ(λ²)`. Chunk-4 accumulation for SIMD auto-vectorisation. Zero-allocation. Guard against all-zero input (return 0.0).
- [x] **T1.3** Implement `pub fn numerical_rank(spectrum: &[f32], eta: f32) -> usize` — smallest `r` such that cumulative energy > η·total. Default η = 0.99 (paper eq. 52). Spectrum assumed sorted descending (caller's responsibility; document this).
- [x] **T1.4** Implement `pub fn phase_transition_gate(n_samples: usize, intrinsic_dim: usize) -> bool` — `n_samples >= intrinsic_dim`. Trivially simple; the value is the *name* and the *documentation* tying it to Theorem 4.
- [x] **T1.5** Implement `pub fn estimate_intrinsic_dim(spectrum: &[f32], method: IntrinsicDimMethod) -> usize` — dispatch between `ParticipationRatio` (round PR to nearest int) and `NumericalRank { eta }`.
- [x] **T1.6** Define `pub enum IntrinsicDimMethod { ParticipationRatio, NumericalRank { eta: f32 } }`.
- [x] **T1.7** Implement `pub struct JacobianSvdScratch { col: Vec<f32>, jac: Vec<f32>, u: Vec<f32>, s: Vec<f32>, vt: Vec<f32> }` with `with_capacity(n, m)` and `clear()` for reuse.
- [x] **T1.8** Implement `pub fn jacobian_svd_at<F>(f: F, x: &[f32], eps: f32, scratch: &mut JacobianSvdScratch) -> SvdResult where F: Fn(&[f32], &mut [f32])` — forward-difference Jacobian (column at a time), then thin SVD. Return top-r `(singular_value, right_singular_vector, left_singular_vector)` triples.
- [x] **T1.9** Define `pub struct SvdResult { singular_values: Vec<f32>, right_singular_vectors: Vec<Vec<f32>>, left_singular_vectors: Vec<Vec<f32>>, rank: usize }`.
- [x] **T1.10** Wire into `katgpt-rs/crates/katgpt-core/src/lib.rs`:
   ```rust
   #[cfg(feature = "subspace_phase_gate")]
   pub mod subspace_phase_gate;
   #[cfg(feature = "subspace_phase_gate")]
   pub use subspace_phase_gate::{
       IntrinsicDimMethod, JacobianSvdScratch, SvdResult, estimate_intrinsic_dim,
       jacobian_svd_at, numerical_rank, participation_ratio, phase_transition_gate,
   };
   ```
- [x] **T1.11** Add feature to `katgpt-rs/crates/katgpt-core/Cargo.toml`: `subspace_phase_gate = []` (no extra deps for now — pure numeric. Thin SVD on small matrices uses a portable scalar implementation; SIMD optimisation deferred to Phase 3).
- [x] **T1.12** Add feature to `katgpt-rs/Cargo.toml` (umbrella) propagating to katgpt-core. _(Initially marked done but the feature line was missing; fixed in the Phase 2 commit as `subspace_phase_gate = ["katgpt-core/subspace_phase_gate"]`.)_

**Exit:** `cargo check -p katgpt-core --features subspace_phase_gate` compiles. `cargo check -p katgpt-rs --features subspace_phase_gate` compiles.

---

## Phase 2 — G1 GOAT Proof (Synthetic MoLRG Phase Transition)

### Tasks

- [x] **T2.1** Create `katgpt-rs/crates/katgpt-core/examples/subspace_phase_gate_goat.rs` (behind feature gate).
- [x] **T2.2** Generate K=3 orthogonal subspaces in R^48, each d=6, with orthonormal bases drawn from QR of random Gaussian.
- [x] **T2.3** For each N ∈ {3, 5, 6, 7, 10, 50, 200}, sample N wake events per subspace, run PCA (via SVD), measure recovery error `‖Û Û^T − U* U*^T‖_F`.
- [x] **T2.4** Plot recovery error vs N (text-based or CSV for bench harness).
- [x] **T2.5** Verify phase transition: for N < d, error > 0.5; for N ≥ d, error < 0.1. Print PASS/FAIL.
- [x] **T2.6** Verify `phase_transition_gate(N, d)` returns `false` for N < d, `true` for N ≥ d — matches the empirical recovery.
- [x] **T2.7** Compare `participation_ratio()` vs `numerical_rank(0.99)` as intrinsic-dim estimators. Document which tracks the true d better on this synthetic.

**Exit:** example runs, prints G1 PASS. Bench CSV saved to `katgpt-rs/.benchmarks/301_subspace_phase_gate_g1.md`.

---

## Phase 3 — Jacobian SVD Validation (G3 precursor)

### Tasks

- [ ] **T3.1** Construct a known low-rank linear map `f(x) = A x` where `A = U Σ V^T` is rank-3 in R^8×8.
- [ ] **T3.2** Run `jacobian_svd_at(f, x, eps=1e-4, scratch)`. Verify recovered singular values match Σ and right singular vectors match V (up to sign).
- [ ] **T3.3** Verify on a non-linear map: `f(x) = sigmoid(W x)` for low-rank W. Jacobian is `diag(sigmoid'(Wx)) W`. SVD should reveal the row space of W.
- [ ] **T3.4** Add timing: Jacobian SVD on R^8→R^8 should be < 1µs (forward diff: 8 evaluations + thin SVD of 8×8). Document in bench.

**Exit:** example prints G3-precursor PASS.

---

## Phase 4 — SIMD Optimisation (deferred)

### Tasks

- [ ] **T4.1** SIMD-accelerate `participation_ratio` and `numerical_rank` (NEON/AVX2 dispatch via `simd.rs`).
- [ ] **T4.2** Investigate Jacobi rotation SVD (O(n²) per sweep) for small matrices vs the scalar baseline.
- [ ] **T4.3** If Jacobian SVD on R^8 is still > 1µs after scalar optimisation, mark Phase 4 as required before riir-neuron-db Plan 002 can ship.

**Exit:** bench shows `participation_ratio` on 64-element spectrum < 50ns; `jacobian_svd_at` on R^8→R^8 < 500ns.

---

## Phase 5 — Promote to Default (conditional on G1 + G3-precursor)

### Tasks

- [ ] **T5.1** If G1 passes (phase transition reproduces) AND G3-precursor passes (Jacobian SVD recovers known singular vectors): add `subspace_phase_gate` to the default feature list in `katgpt-rs/Cargo.toml`.
- [ ] **T5.2** Update `katgpt-rs/README.md` Feature Showcase section with a new entry: "Subspace Phase-Gate (Plan 301)".
- [ ] **T5.3** If G1 fails: downgrade to opt-in, document the failure mode in `katgpt-rs/.benchmarks/301_*.md`, create an issue.

---

## Open questions

1. **Which SVD implementation?** For n ≤ 16 (the common case — HLA is 8-dim, `style_weights` projects to 8-dim), a portable Jacobi-rotation SVD is sufficient and avoids native-lapack deps. If we later need n > 64 (full `style_weights` Jacobian), pull in `nalgebra` (already a transitive dep?) or write a blocked SVD.
2. **Forward differences vs central differences?** Forward diff is n evaluations; central diff is 2n. Forward is the default for speed; central diff is opt-in via `eps < 0` (treat negative eps as central-diff step).
3. **Operating point for Jacobian SVD on non-DAE maps.** Paper uses t ≈ 0.8 for diffusion DAEs. For HLA evolution, the "right" point is unknown. This plan ships the *mechanism*; the *choice* of point is the consumer's responsibility (riir-neuron-db Plan 002 / riir-ai future plan).

---

## Related

- **Research:** [R279](../.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md) (open).
- **Private Super-GOAT guide:** `riir-neuron-db/.research/001_*.md`.
- **Private execution plan:** `riir-neuron-db/.plans/002_phase_transition_consolidation_gate.md`.
- **Closest shipped cousin:** `katgpt-rs/src/spectralquant/spectral.rs` (`participation_ratio`, `calibrate_eigenbasis`).
- **Consolidation cousin:** `riir-neuron-db/src/consolidation.rs::spectral_convergence_check`.

---

## TL;DR

Generic numeric primitive for the open-source engine: participation ratio, numerical rank, N≥d phase-transition gate (Wang et al. Theorem 4), and runtime Jacobian SVD via forward differences. No game semantics, no shard semantics. G1 phase-transition reproduction is the GOAT gate. Phase 1 ships the skeleton; Phase 2 proves G1; Phase 5 promotes to default if both gates pass. The private wrappers (consolidation gate in riir-neuron-db, HLA self-discovery in riir-ai) consume this primitive.
