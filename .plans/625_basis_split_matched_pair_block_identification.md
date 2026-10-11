# Plan 625: `basis_split` — Matched-Pair Block Identification (SplitJEPA distill)

**Status:** In progress — Phase 0 MEASURED 2026-10-12 (T0.1–T0.4 done, outcomes recorded below); Phase 1 next. Docs landed `1e35e7bb6019` (2026-10-11; Claude verdict session 782f282b — round 1 REVISE 6 fixes, round 2 AGREE + 2 non-blocking notes folded in). Source: `.research/618_SplitJEPA_Invariant_Variant_Block_Identifiability.md` (arXiv:2610.12349). Verdict-round-1 corrections applied: no new eigensolver; TILR-composition measured FIRST; whitening default; eigengap gate; `pair_leakage` naming. Consumer issues filed in the consumer repos (report-only first; no wiring in this plan).

## Goal

A certified block split over (frozen embedding accessor, matched pairs): variant basis (top eigenspace of the whitened pair-difference second moment), invariant projector (complement), a refuse-loud identifiability verdict, and a held-out `pair_leakage` residual. Composes existing substrate; forks nothing.

## Non-goals

- No encoder training (that is the riir-train plan, gated on this primitive).
- No consumer wiring — each consumer repo carries its own issue; this plan ships the primitive + bench + gates only.
- **No new eigensolver.** `katgpt-core::linalg::symmetric_eig` (f64, deterministic, Householder+QL, documented d ≥ 256) and `karc::jacobi_eigen` (small d) exist; `katgpt-spectral` already carries three private Jacobi copies — a fifth is a DRY violation.
- No new cross-crate dependency: self-contained in `katgpt-core`.

## Phase 0 — substrate-first confirmation + TILR-first measurement

- [x] T0.1 Grep for name-collision variants (`basis_split`, `block_split`, `pair_split`, `difference_covariance`) across `crates/` — **PASS 2026-10-12: zero hits** (the `basis_split` name is free; no re-route).
- [x] T0.2 Confirm reuse surface — **CONFIRMED 2026-10-12, one drift found**: `karc::jacobi_eigen` (f64, row-major, `eigvecs[i*r+j]`, d≤16) ✓, `linalg::symmetric_eig` + `SymmetricEigScratch` (f64, d≥256) ✓, `subspace_phase_gate::{thin_svd_into, SvdScratch::with_capacity(n_cols, m_rows), SvdResultScratch::with_capacity(m_rows, n_cols)}` (f32, σ descending) ✓, `river_valley::subspace_ratios` (katgpt-spectral, metric precedent) ✓, `subspace_intervention` + `latent_confounder_audit` (core opt-in modules, the G4 tracking-allocator harness pattern) ✓, **`newton_schulz::ns_inv_sqrt_psd_into` (f32, 7 fixed NS iterations — the IN-CRATE whitening form; composition precedent `data_probe/cca.rs`) ✓**. DRIFT: `ica_lens::whiten_into` lives in **katgpt-spectral** (private fn, not dep-able — spectral deps core). Non-blocking: T1.1's whitener-as-input design covers it (caller supplies Σ or a precomputed whitener).
- [x] T0.3 **TILR-first arm: MEASURED 2026-10-12 (`tests/basis_split_phase0_tilr_arm.rs`, required-features `subspace_phase_gate`, 5/5 green).** G1 planted recovery **PASS** — d=64/k=16/N=2048 isotropic fixture, min principal-angle cosine 1.000000 (~0 rad) between the true invariant block and `I − VVᵀ`; TILR's τ=0.99 rule lands r=48 exactly (top-r right singular vectors, σ descending = the VARIANT span; the complement is the invariant block — orientation confirmed in source). **Adversarial gap evidence (no refusal, the delta `basis_split` ships):** value-shuffled pairs still return a split; all-identical pairs still return r=1; the ONLY refusal is all-zero differences (`TilrError::ZeroVariance`). **G4 census: 228 allocations/call** at N=2048/d=64 (f32 accumulator, allocating — the measured justification for the f64/zero-alloc armed path). **VERDICT per the plan's own rule: report-only consumer issues MAY run on TILR composition alone (G1 is exact there); the new f64/zero-alloc path stays justified for ARMED read paths and committable bases by the residual gaps (f32, 228 allocs, τ-heuristic vs eigengap gate, no whitening form).**
- [x] T0.4 Feature flag `basis_split` in `katgpt-core` Cargo.toml — **DONE 2026-10-12**: `basis_split = []` (opt-in, default-off, dep-free; catalog comment names the composition surface + the reflex Issue-086 first consumer).

## Phase 1 — core primitive (`crates/katgpt-core/src/basis_split.rs`)

- [ ] T1.1 Whitening-default geometry: the primitive takes the total covariance `Σ` (or a precomputed whitener) as input, or returns `Verdict::Unwhitened`. Document the generalized-eigenproblem equivalence (`M v = λ Σ v` ≡ whiten-then-eigendecompose). Rationale: without whitening, low-total-variance directions masquerade as invariant.
- [ ] T1.2 `PairDifferenceMoment`: accumulate `M += ΔΔᵀ / N` over supplied pairs, f64 scratch `&mut [f64]` param (zero-alloc; d ≤ 1024 documented envelope), deterministic order. Eigensolve: d ≤ 16 via `karc::jacobi_eigen`, **otherwise `linalg::symmetric_eig`** (measure the crossover only if the 16 < d < 256 band ever shows up in a profile — do not add a third solver for it).
- [ ] T1.3 Rank gate (`SufficientVariation`) — eigengap form: given supplied k, check the gap ratio `λ_k / λ_{k+1} ≥ τ_gap` (a missing gap means the invariant block is not actually null — THE identifiability-breaking failure); when k is not supplied, pick k at the largest gap and report it. Report the gap ratio in every verdict. Participation ratio `(Σλ)²/Σλ²` is report-only this pass (pre-register a derived threshold before ever gating on it). Below the gate ⇒ `Verdict::Unidentified` (refuse-loud, never a silent split).
- [ ] T1.4 Split: variant basis = top-k eigenvectors past the gate; invariant projector `P = I − V Vᵀ` applied as `project_invariant_into(&basis, r, out)` (one d×d matvec, ns-class at d=64).
- [ ] T1.5 `pair_leakage` residual: `mean‖P Δᵢ‖²/‖Δᵢ‖²` over held-out pairs + the agreement residual (mean `‖Δ r_H‖/‖Δ r‖`), both in a `SplitReport`. **Naming law:** this is a generalization residual of the fitted null space, NOT purity — "purity" is reserved for ground-truth settings (G1 planted blocks; deterministic-sim evaluations). Never compare it to the paper's `‖A₁₂‖` figures.
- [ ] T1.6 Docs law in module header: certifies **subspaces and insensitivity, never individual latent axes**; publish agreement residual beside every split; pair-budget law (invariant nullspace ≈ 1k pairs; complementary block needs ≳ k + slack, head-fit sample-hungry); SFA / within-class-scatter lineage named (the estimator is classical; the theorem + gates are the contribution).

## Phase 2 — bench + GOAT gates

- [ ] T2.1 `benches/basis_split_goat.rs`: G1 planted-block recovery by principal angles (`SVD(V_trueᵀ V_est)`, all < 1e-3) at d ∈ {16, 64, 256}; adversarial arms, precisely defined: **shuffled pairs** (after whitening `M ≈ 2I`, no eigengap) must emit `Unidentified` via the T1.3 gap gate; **degenerate pairs** (all identical) must emit `Unidentified`; both canaries prove the gates fire — a gate that never reds certifies nothing. Include the T0.3 TILR-composition arm as the substrate comparison row.
- [ ] T2.2 G2: `--release`; full split (accum + eigen) < 10 ms at d=64/N=2k; quote `bench_preflight` PROVENANCE beside the number.
- [ ] T2.3 G3: `cargo test -p katgpt-core --lib` count floor unchanged (new leaf code, no consumer touched).
- [ ] T2.4 G4: counting-allocator canary over the whole split path (fixed scratch only).
- [ ] T2.5 Record verdict in `.benchmarks/` per numbering discipline; promote to default ONLY on GOAT pass + a wired consumer; otherwise stays opt-in (loser law applies to any future competitor in the same slot).

## Phase 3 — handoff

- [ ] T3.1 Cross-link consumer issues (their repos): game style/affect split audit; healer formatting-invariant retrieval; decision-server presentation certificate.
- [ ] T3.2 Update Research 618 status line with the landed commit hash.

## Validation

`cargo test -p katgpt-core --features basis_split --lib` · `cargo bench --bench basis_split_goat` · full repo gate before commit.
