# Plan 638: Conditional Geometry — `cond_id` + `cond_imbalance` (Subset-Conditional Latent-Geometry Kit)

**Date:** 2026-11-15
**Research:** [katgpt-rs/.research/675_Conditional_Geometry_ID_Imbalance.md](../.research/675_Conditional_Geometry_ID_Imbalance.md)
**Source paper:** [arXiv:2608.25166](https://arxiv.org/abs/2608.25166) — "The Changing Geometry of Grammar" (Vallisa, Ravenda et al. 2026); estimator lineage ABIDE (arXiv:2405.15132), Information Imbalance (Glielmo et al. 2022, arXiv:2104.15079)
**Target:** `katgpt-rs/crates/katgpt-core/src/data_probe/cond_geometry.rs` (new module) — **gated on the EXISTING `estimators` family umbrella feature; NO new `[features]` key** (feature_cap_gate: katgpt-core 290/300, 10 headroom; Plan 635 growth guard + verdict amendment)
**Status:** Active — unclaimed; Phase 1 not started

**Verdict provenance:** Research 675 §5 Claude verdict, 2 rounds, AGREE (`request_verdict` claude_code backend, session 28954393, 2026-11-15). Four binding amendments are folded into the gate design below.

---

## Goal

Ship the minimal closed-form conditional geometry kit over point-latent populations: `cond_id` (subset-conditional intrinsic dimension via the BIDE two-ball binomial restriction — neighbor counts computed ONCE over the FULL population around each point, likelihood sum restricted to subset G, maximizer `d̂_G = log(k̄_A/k̄_B)/log(τ)`) and `cond_imbalance` (cross-space neighborhood-rank prediction between two representation sets, subset-averaged, normalized by N/2). Both are modelless, offline-audit-cadence (the `knn_entropy` posture: brute-force O(n²·d), n ≤ a few thousand), zero-steady-state-alloc, deterministic. GOAT gate: synthetic-manifold accuracy (known-d, sphere arm included per L2N2 arXiv:2603.10493) + subset-conditioning identity + planted-reorganization detection + incumbent comparison on real data shapes; no advantage over the shipped incumbents → demote to Gain.

**Design decisions recorded HERE (implementation-readiness audit — no mid-flight design):**

- **D1 — feature gating.** `#[cfg(feature = "estimators")]` on the module. No new feature key; the umbrella grant row already exists in `scripts/feature_count_pins.txt` (line 26). `feature_cap_gate.py` re-run before commit.
- **D2 — estimand (binding amendment 1).** Neighbor counts `k_A,i`, `k_B,i` are computed over the FULL population around each point i (per-point adaptive selection may be deferred to T1.5; the static-τ form is the default). The conditional estimate restricts the AVERAGE of once-computed per-point statistics to G. G1b therefore asserts (i) identity against a restricted sum of once-computed per-point statistics, and (ii) a DOCUMENTED, EXPECTED difference from estimating on the subset in isolation (different estimand — the neighbor sets themselves change — not an error).
- **D3 — kill-switch baseline (binding amendment 2).** The incumbents to beat are the SHIPPED symbols: `data_probe::geometry::within_class_effective_rank` (Plan 415) and `subspace_phase_gate::estimate_intrinsic_dim` (Plan 301; methods `participation_ratio` / `numerical_rank`). Compared on the real data shapes: the 8-D frozen rule embed (riir-refine `assets/rule_embed_frozen_v1.bin` fixture copy) and 64-dim style-weight-like populations. TwoNN is a secondary reference only (not shipped; comparison is informational).
- **D4 — degenerate inputs (binding amendment 3).** (a) Minimum subset-support floor: `MIN_GROUP_TOKENS = 10` (the paper's own discard, Appendix A.1, verified against the PDF); below the floor → loud typed refusal (`SubsetTooSmall`), never a silent estimate. (b) Tie rule: first-index-wins on equal distances (determinism law); zero-distance duplicates counted as-is in ball counts (they are the collapse signal, mirroring `knn_entropy`'s duplicates ⇒ −∞ posture — here duplicates deflate d̂, the same signal class). (c) NO interval, quantile, or coverage is claimed or emitted — nothing UQ-bearing, so the conformal-naive floor rule does not apply; stated at the module head.
- **D5 — sync discipline (non-binding, adopted).** `cond_imbalance` is a LOCAL-ONLY instrument that never crosses the sync boundary; its tie rule serves reproducibility, not quorum. Stated at the module head.
- **D6 — no new deps.** Brute-force radius counts + neighbor ranks over reusable scratch buffers (the `knn_entropy` scratch discipline). kiddo/linfa-nn recorded as the upgrade path if populations grow beyond the offline-cadence envelope.

## Phase 1 — Core module (estimators-gated)

### Tasks

- [ ] **T1.1** `crates/katgpt-core/src/data_probe/cond_geometry.rs` — module head carries: D4(c) no-UQ statement, D5 local-only statement, estimand (D2), and the citation block (paper + ABIDE + Glielmo). Register `#[cfg(feature = "estimators")] pub mod cond_geometry;` in `data_probe/mod.rs`.
- [ ] **T1.2** `BallCountScratch` — reusable buffers for the two-ball pass: per-point `k_a: Vec<u32>`, `k_b: Vec<u32>`, distance workspace; `with_capacity(n)`, `clear()`; zero steady-state alloc.
- [ ] **T1.3** `pub fn ball_counts(points: &[f32], n: usize, dim: usize, indices: &[u32], tau: f32, scratch: &mut BallCountScratch)` — one O(n²·d) pass over the population computing per-point neighbor counts inside radius r_B and r_A = τ·r_B (radii from the τ-quantile of each point's distance distribution, pinned deterministically: r_B = the ceil(0.5·√n)-th nearest-neighbor distance, first-index tie rule). Counts over ALL points (labels never filter the neighbor sets — D2).
- [ ] **T1.4** `pub fn cond_id(ball_counts: &BallCountScratch, group: &[u32]) -> Result<f32, CondGeometryError>` — the closed form over the restricted average; `SubsetTooSmall` refusal below `MIN_GROUP_TOKENS = 10`; `NonFinite` refusal on non-finite inputs.
- [ ] **T1.5** (defer-able to a follow-up plan; NOT a Phase-1 blocker) per-point adaptive neighborhood selection (ABIDE's Wilks LRT iteration). The static-τ closed form is the filing; adaptive selection is the published method's second half and is only needed if G1 shows the static form under-performing on noisy manifolds.
- [ ] **T1.6** `pub fn cond_imbalance(from: &[f32], to: &[f32], n: usize, dim: usize, group: &[u32], scratch: &mut ImbalanceScratch) -> Result<f32, CondGeometryError>` — nearest neighbor of each i in `from` (full population), its rank in `to` (full population), normalized by n/2, averaged over `group`; first-index tie rule on both the argmin and the rank count; asymmetric by construction (`Δ(A→B) ≠ Δ(B→A)`).
- [ ] **T1.7** Unit tests: closed-form known-answer vectors (hand-computed binomial counts), tie-rule pins, refusal arms (subset too small / non-finite / empty), determinism (×3 identical).

## Phase 2 — GOAT gates (PROVES OR KILLS)

### Tasks

- [ ] **T2.1 G1 — estimator accuracy on synthetic manifolds with known d** (`tests/cond_geometry_g1_manifolds.rs`): uniform samples on d-balls (d ∈ {1,2,4,8,16}) embedded in D=64, AND the d-sphere arm (L2N2 bias class). Pre-registered bar: |d̂ − d| ≤ max(1, 0.25·d) on ≥ 5 of 6 manifold arms at n=2000; sphere arm reported separately, never pooled.
- [ ] **T2.2 G1b — subset-conditioning identity (binding amendment 1)** (`tests/cond_geometry_g1b_conditioning.rs`): (i) `cond_id` over group G is BIT-IDENTICAL to a restricted average of once-computed per-point log-count statistics (proves the estimator consumes the full-population counts); (ii) the subset-in-isolation estimate (recompute counts on G alone) DIFFERS, and the difference is disclosed in the test output as the documented estimand gap — asserted present-and-finite, never asserted small.
- [ ] **T2.3 G1c — planted-reorganization detection** (`tests/cond_geometry_g1c_reorganization.rs`): population of n=512 points in two spaces; plant a neighbor reattachment for a 64-point group in space B (each group member's nearest neighbor re-assigned across a planted boundary); bar: Δ(group) ≥ 2× Δ(control group) with Δ(control) near the preserved-neighborhood floor; shuffled-label control stays high. Falsification arm: destroying the planted boundary must flip the verdict (proves the detector sees the reorganization, not the group label).
- [ ] **T2.4 G2 — latency** (`benches/`): the offline document posture n=1218, D=64 (the paper's own ~1,218 tokens/document): ball pass + one `cond_id` + one `cond_imbalance` ≤ 50 ms release, single-shot measured, bench_preflight PROVENANCE line quoted.
- [ ] **T2.5 G3 — feature-matrix clean:** `cargo clippy --all-targets` at default AND `--features estimators` AND `--no-default-features`; `feature_cap_gate.py` re-run (no new key; count stays 290).
- [ ] **T2.6 G4 — zero steady-state alloc:** counting-allocator gate on the steady-state call path (scratch pre-sized once), mirroring the `knn_entropy` G4 shape.
- [ ] **T2.7 G5 — incumbent comparison on real shapes (binding amendment 2, the kill-switch):** on the 8-D rule-embed fixture copy and a 64-dim style-weight-like population with planted group structure: does `cond_id` separate planted group contrasts that `within_class_effective_rank` and `subspace_phase_gate::estimate_intrinsic_dim` miss (or vice versa)? **Pre-registered demotion:** if the incumbents match or beat `cond_id` on BOTH real shapes (no complementary signal), the primitive DEMOTES to Gain — diagnostic reference implementation, module stays, no promotion talk, verdict recorded here.

## Phase 3 — Verdict + routing

### Tasks

- [ ] **T3.1** Record the gate table + verdict in this plan (per-task `- [x]` with numbers); update Research 675 Status line; demote/hold per T2.7.
- [ ] **T3.2** If GOAT holds: open the consumer follow-ups per Research 675 §2.3 — riir-refine Issue 239 axes (filed 2026-11-15); riir-infer `geometry_audit` lane idea (file THERE when a consumer session takes it); riir-ai II×KG fusion idea (file when riir-ai develop is synced and clean).
- [ ] **T3.3** Consumers never gate on this module until T2 completes; the module is `estimators`-gated from birth (no default posture).

---

**Cleanup:** this plan needs no target dir (tests run in-workspace); any `/tmp/cond_geometry` scratch is removed in the landing commit's session per the teardown law.
