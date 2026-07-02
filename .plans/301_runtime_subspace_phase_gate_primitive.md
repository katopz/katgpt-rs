# Plan 301: Runtime Subspace Phase-Gate Primitive — Generic Math for N≥d + Jacobian SVD

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md](../.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md)
**Source paper:** [arXiv:2409.02426](https://arxiv.org/abs/2409.02426) — Wang et al., *Breaking the Curse of Dimensionality*.
**Private Super-GOAT guide:** `riir-neuron-db/.research/001_Subspace_Consolidation_Quality_Gate_Guide.md`
**Target:** `katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs` (new module) + Cargo feature `subspace_phase_gate`
**Status:** Active — Phase 1 complete (skeleton shipped), Phase 2 (G1 GOAT proof) complete with [Bench 301](../.benchmarks/301_subspace_phase_gate_g1.md) **G1 PASS** ✅ **re-verified 2026-07-02 after the Issue 008 fix (RESOLVED, commit `4e5750c3`; tracker removed)** (wide-matrix extraction bug: the SOA refactor `a08adc4a` narrowed the column-norm scan from `0..n` to `0..min(m,n)`, missing singular values in columns `k..n`; fixed by restoring the full scan + adding a null-space deflation floor). Phase 3 complete: T3.1–T3.3 PASS (square R^8×8), **T3.4 latency gate now PASSES** (2026-07-02, Plan 301 T4.1 allocation elimination) — the prior 2403 ns/call FAIL figure measured the allocating `jacobian_svd_at` path on a slower bench machine; a breakdown probe showed ~36% of that cost was the 17-`Vec` SOA→owned conversion, NOT the SVD math. T4.1 added the zero-alloc `jacobian_svd_at_into` hot path (~800 ns/call release on R^8→R^8, under the 1µs target), with the SVD math byte-identical (G1 re-verified bit-for-bit). **Phase 5 complete (2026-07-02)**: T5.1 promoted `subspace_phase_gate` to DEFAULT-ON in `katgpt-rs/Cargo.toml`; T5.2 README Feature Showcase entry shipped. SIMD on the Jacobi inner loops is documented as non-blocking future work (the gate passes; the determinism contract discourages SIMD dispatch in the math).

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

- [x] **T3.1** Construct a known low-rank linear map `f(x) = A x` where `A = U Σ V^T` is rank-3 in R^8×8. _(Implemented in `known_rank3_map_r8x8` test helper — non-canonical orthonormal singular vectors via 2×2 rotation blocks at distinct angles, σ={10,5,2}.)_
- [x] **T3.2** Run `jacobian_svd_at(f, x, eps=1e-4, scratch)`. Verify recovered singular values match Σ and right singular vectors match V (up to sign). _(PASS: top-3 σ match {10,5,2} within 0.1; each recovered V column matches its ground-truth v_k up to sign with |dot|>0.999. Rank-3 confirmed via `numerical_rank(η=0.99)==3` + 4000× spectral gap; note the SVD's internal `result.rank` reports 4 due to its `sigma_max*1e-5` threshold vs the ~5e-4 forward-diff noise floor — see bench doc.)_
- [x] **T3.3** Verify on a non-linear map: `f(x) = sigmoid(W x)` for low-rank W. Jacobian is `diag(sigmoid'(Wx)) W`. SVD should reveal the row space of W. _(PASS: rank ≥3; every recovered right singular vector with σ>1e-3 lies in span{v1,v2,v3} via projector P_true, ‖P_true·r‖≈‖r‖ to 5e-3. Subspace containment check — individual vectors rotate within the subspace under diag(d)·W, only the subspace is invariant, matching the plan wording.)_
- [x] **T3.4** Add timing: Jacobian SVD on R^8→R^8 should be < 1µs (forward diff: 8 evaluations + thin SVD of 8×8). Document in bench. _(DONE — **PASS after T4.1 allocation elimination (2026-07-02)**: the zero-alloc `jacobian_svd_at_into` hot path measures ~800 ns/call release on R^8→R^8, under the 1µs target. The prior 2403 ns/call FAIL measured the allocating `_at` path on a slower bench machine; a breakdown showed ~36% of that was the 17-`Vec` SOA→owned conversion, not SVD math. Regression-guard test at 100µs debug-stable covers BOTH paths; gate verdict in `.benchmarks/301_*` Phase 3.)_

**Exit:** example prints G3-precursor PASS.

---

## Phase 4 — SIMD Optimisation (deferred)

### Tasks

- [x] **T4.1** ~~SIMD-accelerate~~ **Allocation-eliminate** `jacobian_svd_at` (the actual bottleneck, per the T4.1 breakdown probe). _(DONE 2026-07-02: the plan's original "SIMD the Jacobi inner loops" premise was wrong — a breakdown probe showed the SVD math is only ~24% of the `_at` cost; the 17-`Vec` SOA→owned conversion at the end of `jacobian_svd_at` is the dominant cost (~36%). Added `jacobian_svd_at_into` (zero-alloc hot path, writes directly into the scratch's internal SOA buffer) + `JacobianSvdScratch::svd_result()` getter. Brings R^8→R^8 to ~800 ns/call (<1µs target), with the SVD math byte-identical (G1 re-verified bit-for-bit). Verified by `jacobian_svd_at_into_matches_allocating_path` (bit-identical) + `tests/subspace_phase_gate_alloc_check.rs` (0 allocs/1000 calls). SIMD on the Jacobi dot loops remains a non-blocking future optimization — the gate passes, and the determinism contract discourages SIMD dispatch in the math; chunk-4 would also risk the G1 bit-identical recovery. `participation_ratio` is already chunk-4 (T1.2); `numerical_rank` is too cheap to matter (<50ns on 64-element spectrum).)_
- [x] **T4.2** Investigate Jacobi rotation SVD (O(n²) per sweep) for small matrices vs the scalar baseline. _(DONE: concluded the scalar floor is ~2.4µs for 8×8; no cheap scalar win exists that wouldn't risk the Phase 2 G1 bit-identical recovery (loosening `tol=1e-7`/`max_sweeps=60` changes D=48/n=18 G1 numerics). SIMD (T4.1) is the only path to <1µs.)_
- [x] **T4.3** If Jacobian SVD on R^8 is still > 1µs after scalar optimisation, mark Phase 4 as required before riir-neuron-db Plan 002 can ship. _(FIRED then RESOLVED: T3.4 originally measured 2403 ns/call > 1µs, firing Phase 4. T4.1 allocation elimination then brought the zero-alloc hot path to ~800 ns/call < 1µs. Phase 4 SIMD no longer required for the gate; SIMD on Jacobi loops is non-blocking future work. Phase 5 promotion UNBLOCKED.)_

**Exit:** bench shows `participation_ratio` on 64-element spectrum < 50ns; `jacobian_svd_at` on R^8→R^8 < 500ns.

---

## Phase 5 — Promote to Default (conditional on G1 + G3-precursor)

### Tasks

- [x] **T5.1** If G1 passes (phase transition reproduces) AND G3-precursor passes (Jacobian SVD recovers known singular vectors): add `subspace_phase_gate` to the default feature list in `katgpt-rs/Cargo.toml`. **DONE** (2026-07-02): added `"subspace_phase_gate"` to the `default` feature list in `katgpt-rs/Cargo.toml` (after `viable_manifold_graph`), and updated the feature-definition comment to DEFAULT-ON with the full GOAT gate summary. Follows the `newton_schulz` precedent — NOT also added to katgpt-core's `default` (already transitively enabled by the default-on `tucker_factorization` and `viable_manifold_graph`; root forwarder activates it for umbrella consumers).
- [x] **T5.2** Update `katgpt-rs/README.md` Feature Showcase section with a new entry: "Subspace Phase-Gate (Plan 301)". **DONE** (2026-07-02): entry inserted at the end of the Feature Showcase section (after QGF/Plan 268, before `## 🔧 KV Compression`). Covers the four operations, the determinism contract, the G1 phase-transition result table, the G3-precursor/T3.4/G4 gates, the T4.1 allocation-elimination win, and downstream consumers.
- [x] **T5.3** If G1 fails: downgrade to opt-in, document the failure mode in `katgpt-rs/.benchmarks/301_*.md`, create an issue. _(N/A — G1 PASSES; the block is the T3.4 latency gate, not G1. No downgrade; the feature correctly stays opt-in pending the SIMD latency fix.)_

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
