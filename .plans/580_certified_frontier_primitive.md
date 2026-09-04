# Plan 580: Certified Frontier — Open Primitive

**Status:** **Phases 0-3 + T4.1/T4.2 EXECUTED + PASSED 2026-08-28; NOT PROMOTED (stays opt-in)** — G1/G2/G3/G4 all PASS ([Bench 688](../.benchmarks/688_certified_frontier_goat.md), commits `e3f64479` + this one). The T3.4 floor gate SPLIT: the primitive is the only arm that holds delta (0.000 vs the floor's 0.306, 6.1x over), but LOSES the plan's stated `growth * (1 - violation_rate)` composite — which the bench measures to be **degenerate** (it expands to the true-positive count, so it carries no false-positive penalty and is maximised by certifying everything; the floor scored exactly `n_valid` on all 5 seeds). Promotion deferred to a re-gate on the corrected two-stage metric proposed in Bench 688, because redefining a gate's metric after seeing the result and promoting on the redefinition is how a gate stops being a gate. **Corrected 2026-09-03 — this sentence was stale.** It said T4.3 was "ROUTED ... and is the only remaining gate"; T4.3 **ran and PASSED on 2026-08-30** ([riir-ai Bench 822](../../riir-ai/.benchmarks/822_certified_frontier_four_arm_gate.md)) under exactly the corrected two-stage metric this paragraph asks for — 239/240 corridor coverage, 0 violations in 16/16 seeds, the lean-valid floor excluded at stage 1. **No gate remains open.** What blocks promotion is the no-default-consumer rule (riir-poc is POC-grade), which is an owner call, not a measurement. **T5.3 LANDED 2026-09-03**: `DualPosteriorBuffer` — 69–84× at n = 256, `O(1)` in n, 4 368 B at any n ([Bench 688 §T5.3](../.benchmarks/688_certified_frontier_goat.md)); it changes the cost of the primitive, not its promotion status.
**Phase 0 status (superseded, kept for the record):** **EXECUTED + PASSED 2026-08-28** ([Bench 687](../.benchmarks/687_certified_frontier_phase0_poc.md), example `crates/katgpt-core/examples/certified_frontier_01_basic.rs`) — run under the owner's standing "continue remains, make decisions for best perf/sec prod grade" delegation, because T0 is precisely the gate that informs the scheduling decision. All three stated exits PASS. **Phase 1 is justified, but with a scope amendment — see T0.3 below: the dilation half is CONDITIONAL and coarse grids make `expand_certified` a silent no-op.** Phases 1-4 remain owner-scheduled.
**Date:** 2026-08-27
**Research:** [katgpt-rs/.research/510_ActFlow_Certified_Frontier_Expansion.md](../.research/510_ActFlow_Certified_Frontier_Expansion.md)
**Source paper:** [arXiv:2606.08802](https://arxiv.org/abs/2606.08802) — De Santi et al., *Active Flow Expansion for Out-of-Distribution Discovery*, 2026 (safe-set expansion operator = SAFEOPT lineage, Sui et al. 2015/2018)
**Target:** `katgpt-rs/crates/katgpt-core/src/certified_frontier.rs` (new module) + Cargo feature `certified_frontier = []` (std-only, zero deps)
**Private consumer:** riir-ai `cgsp_runtime` (curiosity frontier) + riir-games `swarm/coverage_curiosity` (certified mask) — no game semantics here.

## Goal

Ship the modelless half of ActFlow's guarantee apparatus: a **certified frontier** primitive that grows a monotone set of provably-valid latent cells from (a) a query buffer of binary verifier outcomes, (b) a closed-form uncertainty model, and (c) a Lipschitz reachability budget — then acquires the next query target on the frontier's edge and says when to stop. Generic math, no game/chain/shard semantics: buffers are `&[[f32; D]]`, labels `&[bool]`, the "verifier" is caller-side.

The paper's entire theory (soundness, monotone growth, reachability coverage, halting) is proven about exactly these GD-free parts; the primitive makes them executable.

## What ships (7 functions + 2 metrics + 1 type)

1. `posterior_variance_linear(x, buffer_feats, lambda, scratch) -> f32` — Eq 10 exact: `σ²(x) = k(x,x) − k(x,X)(K+λI)⁻¹k(X,x)`, `k = dot`. Incremental rank-1 Cholesky of `(K+λI)⁻¹` per appended observation (never a re-solve). Companion `append_observation` maintains it.
2. `beta_mean_variance(valid: u32, invalid: u32) -> (f32, f32)` — the honest closed-form μ substitute (paper's kernel-logistic μ_t needs a convex solve; Beta-Bernoulli per-cell is exact-free and house-consistent). Alt: `ridge_mean(x, X, y, λ)` — `k(x,X)(K+λI)⁻¹y`.
3. `confidence_schedule(t, delta, lambda, kappa, b_rkhs) -> f32` — Eq 31/37: `β_t = 4·L_s·B + 2·L_s·√(2κ/λ·(γ_t + log(1/δ)))` with `L_s = 1/4` and `κ = 1/(s(B)(1−s(B)))` closed-form for the sigmoid. Monotone in t (pinned property).
4. `reachability_dilation(cells, hop_budget) -> Cells` — Eq 15 as a grid/bitmap morphological dilation: admit neighbor z iff `∃z' ∈ S: margin(z') ≥ L·d(z,z')` where `margin(z') = s(μ(z')) − β̂·σ(z') − h` and `L = L_s·L_g`. H-fold = iterate.
5. `expand_certified(frontier, buffer, cfg) -> AppendList` — Eq 32: the certified-set update (LCB + Lipschitz decay, monotone union). THE core op.
6. `acquire_frontier_target(frontier) -> Option<cell>` — safe uncertainty sampling (Eq 33 / approx Eq 14 with factor α): `argmax_{z∈S} σ(z)`. The where-to-look answer.
7. `should_advance(t_since_certified, beta, gamma, epsilon) -> bool` — the halting law: a certified hop is guaranteed once `σ ≤ ε/(2β̂)`; `T ≳ 8α²β̂²γ/ε²`. The stop-looking answer.
8. Metrics: `sphere_exclusion_coverage(samples, threshold)` (greedy, order-pinned — the coverage scoreboard) + `vendi_diversity(kernel_eigs)` (`exp(−Σλᵢ log λᵢ)` — the diversity scoreboard).
9. `CertifiedFrontier<const MAX_CELLS: usize>` — fixed-capacity cell set: latent `[f32; D]`, Beta counts `(u32, u32)`, certified bit (`covered_mask` storage pattern). Zero-alloc by construction.

## Fusion (why this and not just math helpers)

- **Grow-then-navigate**: `CertifiedFrontier` cells feed `build_safe_manifold_graph` (Plan 312) as the node source — the missing acquisition half of the VMG stack.
- **EVPI composition** (riir-ai side): straddling gate `LCB − L·δ < h ≤ UCB` prunes deep-inside + far-outside cells to zero queries — the when-to-look gate gets a certified frontier instead of a disc.
- **Prop 1 bounds** (`spherical_cap_bound`, `laurent_massart_radius`) ship beside the module as pure fns — the design law + the pre-registered G8 prediction (passive vs frontier-targeted separation factor `exp((m−1)cos²φ/2)`).
- **DEC resonance** (documented, not load-bearing): frontier = `∂S` via `exterior_derivative` on the cell cochain; expansion flow divergence via `codifferential`.

## Phase 0 — Self-contained PoC (no feature gate)

- [x] **T0.1** `examples/certified_frontier_01_basic.rs` — self-contained (std only, LCG RNG): 2D checkerboard-validity world (the paper's own illustrative setup), seed set = one valid cell, verifier = ground-truth predicate queried ONLY through the buffer. Run 500 rounds of acquire→query→expand; print ASCII map of certified vs true-valid region + violation count.
- [x] **T0.2** Verify the headline separation: passive random querying vs frontier acquisition on a *sparse-frontier* variant (valid corridor of opening angle φ) — expect the Prop-1-predicted exponential gap in coverage@budget.
- [x] **T0.3 (ADDED — the question the stated exits do not ask)** Does the Lipschitz dilation contribute anything? **Measured: CONDITIONAL, with a law.** Dilation certifies a cell only when `best_cb − h >= L·spacing`, and `L·spacing` is the largest adjacent `|Δp|` on the grid. Sweep (dense world, 6 000 queries, 0 violations throughout): 16×16 → **0 dilated** (headroom 0.2083 < cost 0.2942); 32×32 → **0 dilated** (0.1437 < 0.1496); 64×64 → 6 dilated (0.0884 > 0.0745); 96×96 → **30 of 113 = 27%** dilated (0.0833 > 0.0495). Predicted crossover and observed crossover agree on all four points. **Cause: a single GLOBAL Lipschitz constant charges plateau hops the steepest-cliff price** — and the paper's `L = L_s·L_g` is global the same way, so this is not an artifact of the Beta substitute. *First attempt at this measurement was confounded and is recorded as such: counting end-state "certified but never queried" reads 0 everywhere, because the frontier policy hands max posterior σ to any freshly-certified cell and queries it moments later. Attribution must happen at the moment `cb` crosses `h`.*
- **Exit: MET 2026-08-28** — zero violations on the dense world (30/30 certified, monotone), and 51.4× separation on the sparse world (passive 1.0 vs frontier 51.4 mean over 5 seeds, 0 violations on both arms). Passive certifies only the seed cell at every seed.
- **Phase 1 scope amendments carried by T0.3** (decide before T1.4 writes function 4): (a) `FrontierConfig`/`CertifiedFrontier` must expose a `dilation_feasible()`-style predicate — a coarse grid makes `expand_certified` allocate, relax and certify nothing, silently; (b) a local/anisotropic Lipschitz estimate is where the value is, the global constant is what makes the coarse rows dead; (c) if Phase 1 must be cut, cut the dilation side — functions 6 + 7 delivered the entire 51.4× at a grid resolution where dilation contributed zero.

## Phase 1 — Module skeleton

- [x] **T1.1** Feature `certified_frontier = []` in katgpt-core + root passthrough (no deps — pure std).
- [x] **T1.2** `certified_frontier.rs` with module doc citing R510 + arXiv:2606.08802 + SAFEOPT lineage.
- [x] **T1.3** Types: `CertifiedFrontier`, `FrontierConfig { lambda, delta, b_rkhs, h, lipschitz: f32, alpha }`, `FrontierScratch` (fixed-capacity Cholesky factor + kernel column).
- [x] **T1.4** Fns 1–8 above + `#[inline]` hot paths. Export behind cfg in lib.rs.

**Phase 1 outcome (2026-08-28, commit `e3f64479`):** shipped as planned plus
the three T0.3 amendments — `dilation_feasibility()` is a first-class predicate
(a coarse lattice makes the dilation a silent no-op and the return value cannot
distinguish "nothing left" from "nothing affordable"); `FrontierCell::lipschitz`
takes a per-cell a-priori bound so a plateau hop is not charged the
steepest-cliff global price; acquisition (which carried the whole 51.4x) was
built first. `SphereExclusion { centers, saturated }` replaced a bare count so
the alloc-free 256-center cap is reported, not silently truncating.

## Phase 2 — Core correctness

- [x] **T2.1** `posterior_variance_linear`: pinned against a dense-solve reference (nalgebra-free: hand-rolled small Cholesky, or compare rank-1 incremental vs batch solve at N=64 — must agree to 1e-6).
- [x] **T2.2** `expand_certified` soundness property test (Lemma E.2 as a test): plant a known validity fn, adversarial query sequences (order-shuffled, corrupted labels calibrated by the model), assert **zero uncertified-valid→actually-invalid admissions** at the configured δ across ≥1000 seeds.
- [x] **T2.3** Monotonicity property: certified set never shrinks across arbitrary query sequences.
- [x] **T2.4** Confidence schedule: monotone in t; β_0 sanity; κ/L_s closed-form spot-checks.
- [x] **T2.5** Halting law: once `should_advance` fires, one `reachability_dilation` hop admits no violations (the Lemma E.4/F.7 contract, executed).
- [x] **T2.6** Sphere-exclusion: order-pinned determinism (fixed order → bit-identical cluster count); Vendi on planted eigenvalues.

**Phase 2 outcome (2026-08-28, commit `e3f64479`):** 22/22 green. Incremental
Cholesky vs an independent dense f64 solve at N=64, D=8: max abs 1.161e-6, max
rel 7.252e-6. Zero unsound certifications across 1000 adversarial seeds and
under the deployed policy. Halting law fires at 185 / 728 / 2961 observations
for eps = 0.2 / 0.1 / 0.05 — the predicted 1/eps^2 scaling.

Two findings recorded rather than smoothed over. (1) The halting law is
**per-cell and expensive**: `advance_horizon` prices eps=0.05 at ~1e6 rounds, so
a run spreading queries over a grid never fires it; T2.5 prints that gap instead
of asserting it away. (2) T2.5's first parameterisation (96x96, 6000 rounds) was
**vacuous** — 0.65 observations per cell certified nothing, so the dilation
soundness check ran against an empty set. 48x48 with 200 000 rounds admits ~129
cells by dilation and is the shipped gate.

## Phase 3 — GOAT gates

- [x] **T3.1 G2 perf**: batch acquisition + expansion at crowd scale — 1000 frontier queries (one per NPC) with buffer N=256, D=8; target < 1 µs/query amortized (rank-1 updates, precomputed inverse); release-only gate.
- [x] **T3.2 G4 alloc-free**: `FrontierScratch` capacity stable across 1000 expand/acquire cycles (tracking allocator).
- [x] **T3.3 G3 no-regression**: feature-off build clean; default surface untouched until promotion.
- [x] **T3.4 UQ floor** (Report-the-Floor adaptation — this primitive claims a coverage guarantee): bench certified-growth × violation-rate against the naive floor = **adjacency-only expansion** (certify any neighbor of a valid-labeled cell, no uncertainty model). The primitive must dominate the floor on the product metric (growth ⋅ (1 − violation rate)); if it cannot, demote to documented-negative.
- [x] **T3.5 Bench doc**: [`.benchmarks/688_certified_frontier_goat.md`](../.benchmarks/688_certified_frontier_goat.md) with the floor table. (Numbered 688 per the monotonic-numbering rule, not 580 — the plan's placeholder name would have recycled a plan number as a benchmark number.)

**Phase 3 outcome (2026-08-28, [Bench 688](../.benchmarks/688_certified_frontier_goat.md)):**

- **G2 PASS** — 0.264 us/query against a 1 us budget. First measured at
  **3.428 us (FAIL)**; two fixes gave 13.2x. Cached candidacy + cached Beta sd
  took it to 1.080; splitting the four hot acquisition fields into a contiguous
  `acq_sigma` lane and replacing the scalar argmax with a branch-free 8-wide max
  reduction plus a short-circuiting `position` took it to 0.264. The lane is
  derived state maintained at every mutation point, so
  `acquisition_lane_matches_a_full_rescan` pins it against a reference argmax at
  every step of a run.
- **G4 PASS** — 0 allocs / 0 deallocs over 1000 full-operator cycles. The
  instrument was **revert-probed** (an injected `vec!` produced 1000/1000; its
  removal returned to 0), because a green alloc gate that cannot go red is not a
  gate.
- **G3 PASS** — feature-off default build clean, clippy `-D warnings` clean.
- **T3.4 SPLIT** — see the status line. The diagnostic sweep (T3.4b) answered
  the question the floor comparison raises: the shipped confidence schedule
  spends **0.000 of a 0.05 budget** while certifying 35% of the valid region,
  and a 4x narrower width still holds delta. So the growth deficit is a loose
  bound, not the query budget — the paper's Eq 31/37 is derived for a kernel
  logistic model where information pools through the RKHS norm, while this
  module's default posterior is a per-cell Beta with no pooling at all.
- **Shipped as a result:** `beta_union_bound(cells, rounds, delta)` =
  `sqrt(2 ln(cells*rounds/delta))` — **+33% certified growth (308 -> 410) at
  zero measured violations**. Offered, not defaulted; its doc states the
  sub-Gaussian approximation plainly and names the exact Clopper-Pearson variant
  as the rigorous follow-up.

## Phase 4 — Fusion surface

- [x] **T4.1** `CertifiedFrontier → SafeManifoldGraph` adapter fn (nodes from certified cells; edges via existing kNN + midpoint check): one example running grow-THEN-navigate end-to-end. **DONE 2026-08-28** — `certified_manifold_graph()` + `CertifiedManifoldGraph { graph, node_to_cell, rejected_by_volume }`, gated on BOTH features. Both filters (certified AND pullback-volume) run in the adapter rather than inside the builder, which is what makes an exact `node_to_cell` mapping possible — the builder drops nodes, so ids and cell indices would otherwise diverge silently. Example `examples/certified_frontier_02_navigate.rs`: 212/1024 certified (60 by dilation), 0 violations, graph 212 nodes / 769 edges, geodesic 8 hops with weakest certified bound 0.6250 and weakest TRUE p 0.7315 (both >= h=0.6). CI-gated by `t4_1_geodesics_over_a_certified_graph_never_leave_the_certified_set`, which walks EVERY reachable pair. *The example's first version was vacuous* — it picked the globally farthest node pair, which in a two-lobe field is guaranteed unreachable, so the geodesic assertions never ran; it now picks the farthest REACHABLE node and reports the component split (110 of 212 from node 0) as the fact about the field that it is.
- [x] **T4.2** Straddling-gate helper `query_is_decision_relevant(lcb, ucb, h, cell_diam) -> bool` (the EVPI-shape prune) — pure fn + unit tests.
- [x] **T4.3** ROUTED 2026-08-28 to riir-ai Issue 774 (commit `91f502ed2`) — **DONE + PASS 2026-08-30**: the four-arm consumer gate ran on riir-poc under Bench 688's corrected two-stage gate — **[riir-ai Bench 822](../../riir-ai/.benchmarks/822_certified_frontier_four_arm_gate.md)** (plan [riir-ai 557](../../riir-ai/.plans/557_certified_frontier_four_arm_gate.md) carries the pre-registration + the 4-run amendment ledger): A (certified frontier) pooled **239/240 = 99.6 % corridor coverage, 0 violations in 16/16 seeds**, vs B (curiosity-only) = C (passive) = D (never-look) = 0 declared — separation factor **239**; the lean-valid FLOOR excluded at stage 1 (rate 0.991 vs δ = 0.05); G1 bit-identical double-run. **T5.1 is UNBLOCKED** — but the flip itself remains a katgpt-rs/owner call: riir-poc is POC-grade consumer evidence, not a production consumer (the no-default-consumer rule; `certified_frontier` stays opt-in until that call).

## Phase 5 — Promotion

**Phase 5 decision (2026-08-28): T5.2 — stays opt-in.** G1–G4 all PASS and the
primitive is the only arm holding delta, but (a) the stated floor gate FAILS on
its literal metric, (b) the in-tree fusion consumer landed with T4.1 but the cross-repo four-arm gate (T4.3, riir-ai) has not run, and
(c) opt-in costs nothing (std-only, zero deps). Bench 688 proposes the corrected
two-stage gate — admissibility (violation rate <= delta) as a hard filter, then
growth among admissible arms only — for the re-gate once a consumer exists.

- [-] **T5.1 OWNER CALL, not an open engineering task** — every gate this row names has now passed: G1–G4 (Bench 688) and the corrected two-stage floor gate on a real consumer (T4.3 / riir-ai Bench 822, 239/240 corridor coverage, floor excluded at stage 1). What blocks the flip is the **no-default-consumer rule**: riir-poc is POC-grade consumer evidence, not a production consumer. That is a scheduling decision, so this row is left for the owner rather than taken. T5.3's 69–84× does NOT move it either — a perf gain is not a promotion criterion. Original text: if G1–G4 + floor PASS → add to `default = [...]` (both Cargo.tomls) + README showcase row + `.docs/01_orientation/overview.md` Feature Flags row.
- [x] **T5.2** If floor FAILS → keep opt-in, document the regime where adjacency-only wins (dense-frontier worlds), demote honestly in R510 footer.

### First external consumer landed — 2026-08-31 (unblocks the re-gate precondition)

Bench 688 deferred promotion to "the corrected two-stage metric **once a
consumer exists**." One now does: **riir-train Plan 357 T1.2**
(`crates/riir-train-gpu/src/dllm/actflow_gp.rs`,
[Bench 563](../../riir-train/.benchmarks/563_plan357_t1_2_gp_uncertainty.md))
consumes `certified_frontier` as the **reference oracle** for its own GP
posterior variance, on both `posterior_variance_linear` and `ridge_mean`, at
every observation count 0..48 across λ ∈ {1, 1e-1, 1e-2}. Two independently
written implementations agreeing to f32 tolerance is external correctness
evidence this plan did not previously have.

**It does NOT consume `PosteriorBuffer` as its production path, and the reason
is a regime inversion worth a follow-up here.** `PosteriorBuffer` factorises the
`n × n` Gram matrix `K + λI`, which is right for this plan's own setting
(`n < D` — observations scarce, latent wide). ActFlow inverts it: a 4096-sample
warm-up against a 32-D projected feature. Measured on the consumer's side:

| | primal (`PosteriorBuffer`) | exact dual |
|---|---|---|
| variance @ n = 256, D = 32 | 158.7 µs/query | 1.99 µs/query (**79.6×**) |
| scaling in `n` | `O(n²)` | **`O(1)`** (1.007× from n=16 to n=4096) |
| state | 291 KiB @ `MAX_OBS=256`; **64 MiB @ 4096** | **4368 B at any n** |

For a **linear** kernel the two are the same number by Woodbury, exactly:

```text
k(x,x) − k(x,X)(XXᵀ + λI)⁻¹k(X,x)  ==  λ · xᵀ(XᵀX + λI)⁻¹x
```

`A = XᵀX + λI` is `D × D`, held as a lower Cholesky factor with a rank-1
`cholupdate` per observation — and its pivots are bounded below by `√λ` for every
observation sequence, so the near-duplicate-feature floor the primal needs
(`chol[n][n] = rem.max(lambda * 1e-6).sqrt()`) has no analogue.

- [x] **T5.3 LANDED 2026-09-03** Add a regime-conditional dual path beside
      `PosteriorBuffer` — `DualPosteriorBuffer<D>` with the same
      `append_observation` / `posterior_variance_linear` / `ridge_mean` surface,
      selected when `n > D`. The primal stays: it is the correct factorisation
      for this plan's own `n < D` cells, and it is now also the *oracle* the dual
      is gated against (riir-train's gate can be mirrored in-tree). Cheap —
      ~80 LOC, std-only, and the consumer's implementation is already written and
      equivalence-tested.

      **Done.** `DualPosteriorBuffer<D>` + a `LinearPosterior<D>` trait (so one
      caller drives either arm and sizes its scratch off `scratch_len()`) +
      `prefer_dual(expected_obs, d)`. Measured in-tree on M3 Max rather than
      quoting the consumer's box: **69–84× at n = 256**, dual scaling
      **0.967–0.997× from n = 16 to 4096** (`O(1)`), primal **88.9–94.5×** over
      the same span, state **4 368 B at any n**. Equivalence to the primal
      oracle: worst relative deviation **3.229e-6** on variance / **1.614e-4**
      on `ridge_mean` against a 2e-3 bar, across every `n` in 0..=48 × three
      ridges — tolerance, not bit-identity, because two f32 expression trees for
      the same real number cannot agree bit-for-bit (bit-identity IS asserted
      where available: one arm against itself). 7 new gates, 31 green in
      `certified_frontier_correctness`; full record in
      [Bench 688 §T5.3](../.benchmarks/688_certified_frontier_goat.md).

      At **n = 16 the primal is faster** (195 vs 253 ns) — which is the point of
      `prefer_dual` and means the rule `expected_obs > d` is validated by
      measurement, not asserted.

      Arming the two `certified_frontier` test targets on the way (they were
      auto-discovered with no `[[test]]` row, so naming either without the
      feature reported a green `0 passed`) exposed a **classifier** defect in
      Issue 713's load-bearing token set — see `.docs/10_audits/cfg_gated_silent_zero_pass.md` T4c.

## Risk register

| Risk | Mitigation |
|---|---|
| Beta mean substitute breaks calibration → false certifications | Soundness test (T2.2) runs against the SUBSTITUTE, not the paper's μ; floor gate (T3.4) catches it if uncertainty adds nothing over adjacency |
| Buffer-depth O(t²) blowup at long horizons | Cap buffer (ring) + document the information-gain plateau (γ ~ d_eff·log T → uncertainty is eliminable); halting law bounds t per cell anyway |
| 8-D fine, 64-D shards slow (curse of dim) | Scope: zone-grid cells + 8-D HLA latents; document d_eff caveat (R510 caveat 6) |
| Known-art operator (SAFEOPT) | Honest framing everywhere: we ship it as we ship bandits — operator known, fusion + domain novel |
| Two selection-mode negatives (Bench 035/042) suggest acquisition layers can lose | Those modified starved per-candidate pool scoring; this is input-space epistemic variance with 8K-observation substrate (healer) or ground-truth verifier (PoC) — and the floor gate is the honest kill switch |

## Cross-references

- [R510](../.research/510_ActFlow_Certified_Frontier_Expansion.md) — source distillation + Path 0 table + signal-diffs
- [Plan 312](312_viable_manifold_graph_primitive.md) — the navigation half (grow-then-navigate fusion)
- riir-ai Issue 738 EVPI gate (when-to-look), riir-games Issue 672 `covered_mask` (storage pattern)
- Sibling tracks: [riir-train Plan 357](../../riir-train/.plans/357_actflow_discrete_expansion.md) (training half), `riir-clippy Issue 048` (healer acquisition)
