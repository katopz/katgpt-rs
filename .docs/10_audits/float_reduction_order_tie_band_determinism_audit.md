# Float reduction-order + tie-band determinism audit (Issue 940)

Point-in-time census, 2026-10-16 — which f32 reductions and argmax picks in
this repo feed committed / exported / replay-compared state, and whether
their order or tie rule can vary. Scope is katgpt-rs-side only (selector /
pruner picks + belief/kernel sums); the game-side sync-surface half (flock
centroid etc.) is riir-ai Issue 1074; the swir penalty-ledger live defects
are riir-ai Issue 1075.

## Why (the two hazards, from handwired via Research 621)

1. **Reduction-order grouping.** FP32 addition is not associative; loop
   order vs tree/parallel order gives different bits (handwired's example:
   loop `799.300048828125` vs pairwise `799.2999877929688`). Our contract:
   "Deterministic replay, quorum sync, and anti-cheat require bit-identical
   reconstruction across nodes."
2. **Argmax tie band.** A pick with tolerance-based float comparisons is
   non-transitive and can select a different candidate than a strict
   sequential scan; a pick over `HashMap`/`HashSet` iteration is
   run-to-run nondeterministic at ties.

## The law (T4 — written once, here)

> **An f32 reduction or pick whose result crosses the sync/quorum boundary
> (committed, exported, hashed, or replay-compared) pins its reduction
> order and tie rule. A pick uses one fixed strict rule (first-index /
> documented last-index), never a tolerance band; a reduction uses a
> source-pinned order — the sequential fold (`dot_f32_ordered`, `.sum()`
> folds) — never a platform-lane reassociation. Anything else must be
> reclassified as local-only, with the decision recorded at the site.**

The repo already ships every ingredient of the remedy:
`katgpt_types::simd::dot_f32_ordered` (sequential fold; "When to use: paths
whose outputs are committed or compared bit-wise across nodes", with the
`ordered_dot_differs_from_simd_dot` anti-dedup pin and the Bench-844 GOAT
gate), `katgpt_core::float_order::{cmp_for_max,cmp_for_min,desc,asc}`
(NaN-safe total orders), `simd_argmax_f32` (first-index ties, test-pinned
across NEON/wasm/x86 backends), and the certified pattern itself:
`katgpt-core/src/closure/bridge_certified.rs` consumes `dot_f32_ordered` +
`fast_sigmoid` precisely because `freeze_version_hash` must be
platform-stable. The finding of this audit is that the pattern is deployed
in ONE lane and absent from every other committed-state lane below.

## Method

Full census greps over the ten katgpt-rs decision crates
(`simd_dot|simd_sum|\.sum\(\)|\.fold\(|logsumexp|rayon|par_iter` for
reductions; `max_by|min_by|argmax|sort_by|select_nth|partial_cmp|cmp_for_max`
and epsilon patterns for picks), targeted reads of every production hit,
tests/benches excluded. The ~15 headline sites below were re-verified
line-by-line by the orchestrator; the remaining census rows carry per-site
file:line for re-check at fix time. Coverage caveat: sort-based picks in
files reached by none of the greps were not individually opened.

## Classification key

| Class | Meaning |
|---|---|
| PINNED | scalar fold / elementwise `mul_add` chain — source-pinned order (Rust never reassociates f32 adds without fast-math); `f32::max` folds are order-immune |
| PLATFORM | explicit SIMD kernels (`simd_dot_f32`/`simd_sum_f32`/`simd_sum_sq`/`simd_dist_sq`) — fixed lane order per (platform, code version), DIFFERENT bits across aarch64 / x86_64 / wasm32-simd128 |
| LOCAL | result never hashed, stored, exported, or compared across nodes — recorded, not fixed |
| TIE: FI/LI | pick ties resolve first-index / last-index (both deterministic; LI must be pinned in a comment to survive a future `max_by` → `fold` swap) |
| TIE: TOL | tolerance-band comparisons — non-transitive, order-sensitive |
| TIE: UNST | tie resolved by HashMap/HashSet iteration order — run-to-run nondeterministic |

Zero `rayon`/`par_iter` reductions and zero collection-dependent REDUCTIONS
were found; collection-dependence exists only in PICKS (below).

## Part A — reduction-order census (committed-state sites)

| Site | Form | Class | Feeds | Decision |
|---|---|---|---|---|
| **MB-1** `katgpt-micro-belief/types.rs:157-177` `project_to_scalars_bridge` | `simd_dot_f32` per direction row → `fast_sigmoid` | PLATFORM | **THE 5 synced affect scalars** (latent→raw bridge; types.rs:90-91 "cross the sync boundary raw … need bit-identical deterministic replay"; bridge.rs:20-23 claims "Bit-identical across runs (G1.1)" — per-run only, not cross-node) | **PIN — LANDED 2026-10-16**: `dot_f32_ordered` + docs upgraded to the cross-node claim; divergence test `bridge_is_the_ordered_fold_not_the_platform_kernel` (len-16 craft, mirrors the certified test) |
| **MB-2** `micro-belief/attractor.rs:155-163,198-211` `AttractorKernel` | `simd_dot_f32` per row (weights × state) | PLATFORM | belief vector `s_t` → MB-1 scalars | **PIN with MB-1 — LANDED 2026-10-16**: all three dot sites (`step`, `precompute_wx_dot`, `step_with_precomputed_wx`) → `dot_f32_ordered`; the ILP now lives across rows. Measured cost: G1.4 informational 270 → **491.7 ns/step** at dim=32 in release (the sequential fold is ~1.8×; the test was already failing its 100 ns bar before, stays informational; ~1% of one core at 1000 NPC × 20 Hz) |
| **MB-4** `micro-belief/bom_arena.rs:343,444` | `simd_dot_f32(..).to_bits() as u64` → RNG seed → K hypotheses → **mean committed back into `state`** | PLATFORM | belief state (→ MB-1) | **PIN — LANDED 2026-10-16**: both planner seeds fold the obs dot through `dot_f32_ordered(&state[..dim], &observation[..dim])`, closing the cross-platform chain seed → queries → hypotheses → mean → `state` → bridge |
| MB-3 `micro-belief/leaky.rs:155` | scalar `.sum()` | PINNED | belief → bridge | good |
| MB-5 `micro-belief/bom.rs:352-359,279` `sample_k_states` / `dot_product_scorer` | `simd_dot_f32` per hypothesis | PLATFORM | distributional steering (in-process inference; nothing hashed/committed) | **LOCAL** (recorded) — winner selection consumed in-process. **AMENDED 2026-10-16 at fix unit ①**: the census's line range conflated two things — the `sample_k_states` Phase-1 base activation is BIT-COUPLED to `step()` by the pinned σ=0 degeneracy contract (`bom_sigma_zero_matches_step_attractor` asserts exact equality), so it went ordered WITH MB-2 (a LOCAL call there was not available without breaking a pinned test); only `dot_product_scorer` + the per-hypothesis winner pick stay platform SIMD/LOCAL (site comment added) |
| SE-1 `katgpt-sense/reconstruction.rs:994-1000` `evolve_belief` | scalar fold over `[f32; 6]` | PINNED | belief → SyncBank → relation scalars (the 930 lane) | good |
| **SE-2** `reconstruction.rs:1093` `evolve_belief_simd` (`sense_composition`) | `simd_sum_f32` | PLATFORM | belief → SyncBank → relation scalars; the in-repo test pins scalar-vs-SIMD ≤ 1e-5 and states "bit-equality would assert more than the underlying belief channel provides" | **PIN — LANDED 2026-10-16**: `evolve_belief_simd` DELEGATES to `evolve_belief` (the same `leaky_step` core + the same Σ[0..6] total — byte-identical by construction; the tolerance tests strengthened to exact `to_bits` equality). Side effect, recorded: the SIMD path's old early-return skipped the no-op-tick surprise/SyncBank observation the scalar path documents as intentional — the delegation aligns it. ⚠ Audit finding: the chain also rode `simd_fused_sub_scale_inplace`, whose scalar arm is FMA-shaped (`mul_add`) while its NEON/AVX2 arms are two-op — per-cell cross-platform divergence on scalar-fallback targets (see the elementwise note below). |
| **SE-3** `reconstruction.rs:791` `expand_simd` (`sense_composition`) | `simd_dot_f32` per module | PLATFORM | kind_activations → belief → SyncBank | **PIN — LANDED 2026-10-16**: `expand_simd` DELEGATES to `expand` (the pinned `SenseModule::project` sequential-FMA fold); ⚠ CENSUS EXTENDED: the audit missed the two matrix paths — `expand_with_weights` (:852) and `expand_batch` (:393) fed the same activations through `simd_matmul_rows` (expand_with_weights is `reconstruct_inner`'s DEFAULT expand under the feature). Both now fold per row with a portable sequential FMA chain (`mul_add`, defined one-rounding semantics on every target); the retired 1.27× SIMD matvec advantage (20.4 vs 25.9 ns at 6×8) is recorded as the price. Pins: `expand_simd_is_the_scalar_expand_bitwise`, `expand_with_weights_folds_rows_in_index_order` (the len-8 craft: sequential 3.0 vs lane-grouped 6.0), `expand_batch_matches_expand_with_weights_bitwise`. Note: matrix-row values are pre-rounded `sign·scale`, so the matrix fold is NOT bit-identical to `project`'s exact-shape FMA — a deterministic feature-config-level difference (pick one path per fleet), recorded at the site. |
| **SE-4** `reconstruction.rs:925` `route_simd` (`sense_composition`) | `simd_sum_f32` mean → threshold mask | PLATFORM | which modules' evidence enters the belief (razor-edge flip at the mean threshold) | **PIN — LANDED 2026-10-16**: `route_simd` DELEGATES to `route` (the pinned `.sum()` fold — bit-identical mask by construction); pin `route_simd_is_the_scalar_route_bitwise` (512-sample seeded battery + tie cases) |
| SE-5 `sense/octree.rs:128-141` `embedding_to_ternary` | scalar fold (`scale_sum`) | PINNED | ternary dirs → `SenseModule::commit()` **BLAKE3** | good |
| SE-6 `sense/schema_centroid.rs:44-52,185` | Welford scalar folds | PINNED (given ordered input slice) | centroid → seeded `KgEmbedding` → BLAKE3/Merkle | good — fix-time note: callers must pass deterministically-ordered classes |
| **SE-7** `sense/bake.rs:92` `precision_to_confidence` | `simd_sum_f32(lambda)` | PLATFORM | confidence → **Merkle leaf bytes** (octree.rs:181,197) | **PIN — LANDED 2026-10-16**: the mean folds through `lambda.iter().copied().sum()` (the source-pinned index order); pin `precision_to_confidence_folds_in_index_order` (the craft: sequential mean 0.375 vs lane-grouped 0.75 — visibly different sigmoids) |
| SE-8 `sense/reconstruction_depth_invariance.rs:141` | delegates to SE-1 + scalar `sum_sq` folds | PINNED | RMS-rescaled belief → SyncBank | good |
| **HL-1/2/3** `katgpt-hla/kernel.rs:123,194,474` | `simd_dot_f32` (`k_mq`, `q_mk`) | PLATFORM | **mutate `hla_moments`** — the Cold-tier Pod, BLAKE3-committed on freeze | **PIN — LANDED 2026-10-16**: all FOUR state-feeding dots → `dot_f32_ordered` — the census's three lines **plus the missed fourth site** `ahla_per_head_step` :637 (same committed `q_head.n` as `ahla_step`; found by classifying every `simd_dot_f32` in the crate by consumer). G2 duty discharged (Bench 939): the fix costs one +36.7 ns dot per step; whole-step A/B -1.1% .. +2.4% (within run scatter of the predicted +0.8%); readout denoms (HL-4) stay SIMD. Pins: `k_mq_folds_in_index_order_not_platform_order` + the AHLA twin (probe k at 2^-25 keeps mK bit-identical, `n[i] = 3·2^-25` exact). ⚠ One-time committed-bits change: `hla_moments` `h`/`n` arrays move at this upgrade (SK/CQV/G/mQ untouched — no reduction). |
| HL-4 `kernel.rs:321-322,389-390,494` readout denoms | `simd_dot_f32` | PLATFORM | attention readout → logits → int token ids | **LOCAL** (recorded) — tokens, not bits, cross the wire |
| **PE-1** `katgpt-personality/kernel.rs:227-235` `drift` | `simd_dot_f32(&d_recent, &ONES)` | PLATFORM | `w` → `PersonalitySnapshot.blake3` — "The **commitment** … IS synced as an audit event when a hot-swap occurs" (snapshot.rs:12-14) | **PIN — LANDED 2026-10-16**: `dot_f32_ordered(&d_recent, &ONES)` (the mul-by-1.0 is an exact f32 identity, so this IS Σ d_recent in ascending order); pin test `drift_folds_in_ascending_index_order_not_platform_order` (cancellation craft, w lands on 3.0 bit-exact); provenance law noted in snapshot.rs |
| **SP-2** `katgpt-spectral/procrustes.rs:445` `m_norm` | `simd_sum_sq(..).sqrt()` pre-scale | PLATFORM | `M̃` → Newton-Schulz → aligned-entity match "committed as a raw KG triple with a confidence scalar" (module doc; riir-chain Issue 001) | **PIN — LANDED 2026-10-16**: `dot_f32_ordered(&m, &m).sqrt()` (Σx² IS dot(x,x) — the exact sequential sum of squares; cost one O(d²) pass against the O(d³)×15 scalar Newton-Schulz — unmeasurable), plus the census-missed **uncentered residual arm** `simd_dot_f32` → ordered (the residual feeds the confidence scalar in the module's designed deployment, and the centered arm already folded sequentially — both arms consistent now). Pin `m_norm_folds_in_index_order_not_platform_order`: anchors craft `M` onto a 16-cell ulp-boundary vector (squares can't cancel — on aarch64 divergence appears only at s = 32768, s² = 2³⁰ = 1 ulp of 1e16; the test probes a per-platform candidate list and refuses to pass vacuously). Module doc's determinism section upgraded to the end-to-end bit claim (M per-cell FMA + ordered norm + scalar polar iteration + IEEE-exact sqrt/scale); the pre-940 G3 claim cited a test file that never existed — this pin is the first executable record. |
| **SP-3** `spectral/spectral_rewire.rs:308` (`spectral_rewire`, opt-in) | `simd_dot_f32` building `M = A·V` | PLATFORM | purified weight deltas → BLAKE3-committed freeze overlays | **PIN — LANDED 2026-10-16**: `dot_f32_ordered(a_row_i, v_col_j)`; steps 2/4/5 already folded per-index through single-rounding `mul_add`, so this was the one platform hole in the projection. Pin `rewiring_matrix_folds_in_index_order_not_platform_order` (driven through the real private `project_core` with a rank-1 shape whose algebra is bit-exact: `u_col(0) = e₀`, `v_col(0) = ones`, `delta[0][:]` the certified len-16 cancellation craft → M[0][0] = 3.0 not the platform 6.0, and the single cell propagates bit-exactly through B and ΔW*). Price recorded: r² dots of d_in go scalar-fold — offline freeze-time path, correctness-first. |
| SP-1 `spectral/hla_eigenbasis.rs:571-585` | `simd_dot_f32` power iteration | PLATFORM | eigenbasis — "stay local to the NPC — never synced" (file doc :41-45); provenance hashes the INPUT window, not eigvecs | **LOCAL** (recorded, by the file's own law) |
| **CL-1** `katgpt-claim/clr/vote.rs:133,232` | `simd_sum_f32(row)/m` reliability mean | PLATFORM | CLR verdict reliability → compaction gate / jury | **PIN — LANDED 2026-10-16**: both sites fold `row.iter().sum::<f32>()` (m is small — paper default 5 — so the sequential fold is also the faster shape); pin `reliability_mean_folds_in_index_order_not_platform_order` (table verifier hands out the certified len-16 craft as M=16 verdicts: ordered mean 3/16 vs platform 6/16 → orders-of-magnitude reliability flip at the `^M` gate; expectation recomputed in-test so the powf path stays portable) |
| **CL-2** `claim/clr/verifier.rs:90` | `simd_dot_f32` → sigmoid verdict | PLATFORM | per-claim verdicts → CL-1 inputs | **PIN — LANDED 2026-10-16**: `dot_f32_ordered(embedding, d)` (also removes the unchecked-len hazard — the ordered fold asserts equal lengths where the kernel trusted the caller's `direction_dim`); pin `verdict_dot_folds_in_index_order_not_platform_order` (craft embedding vs ones direction: verdict is `fast_sigmoid(3.0)` bits, provably ≠ the platform's `fast_sigmoid(6.0)`) |
| **CL-3** `claim/clr/mgpo.rs:83` `allocate_budget` | `simd_sum_f32(weights)` | PLATFORM | proportional allocation `.floor()` — integer flip at boundary → different sampling budgets | **PIN — LANDED 2026-10-16**: `weights.iter().sum()`; pin `budget_sum_folds_in_index_order_not_platform_order` (certified len-16 craft as weights: ordered sum 3.0 → seed-0 budget 3.33e9, platform sum 6.0 → 1.67e9 — the floor flip is a 2× allocation change; weights deliberately violate the documented non-negative domain because the pin targets the sum's fold order, noted in-test) |
| **DE-2** `katgpt-dec/stokes_calculus.rs:287-292,358-363` | TWO mass paths, each a pinned scalar fold but in DIFFERENT orders; repo test pins agreement at 1e-4, naming "float accumulation-order differences between the two paths" (:810) | PINNED×2, mutually divergent | belief-region mass → stampede trigger (`zone_density.rs:431`), conservation checks | **CANONICALIZE — LANDED 2026-10-16**: `boundary_flux_mass_indexed` documented CANONICAL for committed/replay-compared masses (with the caller-order caveat: the fold is in caller region order by design — a replay consumer must construct the region deterministically); `boundary_flux_mass_only{,_scratched}` documented DIAGNOSTIC with the pinned 1e-4 cross-path envelope (an envelope contract, not bit-identity — interior ±v pairs cancel exactly, surviving terms reassociate). Workspace census: neither path has a production consumer today (indexed = re-export/bench only; scratched = the pca morphogen report-only signal; the stampede trigger at zone_density.rs:431 consumes `belief_mass_divergence` = DE-1, already pinned) — so the canonicalization is a documentation+test-annotation act, zero behavior change. Test comment re-annotated with the tier law. katgpt-dec 249/249 lib, clippy `-D warnings`. |
| DE-1 `stokes_calculus.rs:80` `belief_mass_divergence` | scalar `.sum()` of abs | PINNED | divergence scalar → curiosity / ICT branching | good |
| DE-3/DE-4 + `katgpt-dec` latent-field folds | scalar folds / `katgpt_dec::simd` source-pinned twin (NOTE: different bits than the types-crate kernel — one implementation, two spellings) | PINNED | local solver/latent state | good |

Elementwise/axpy kernels (`simd_outer_product_acc`, `simd_transpose_matvec_acc`,
`simd_add_inplace`, `SyncBank::observe`, …) are PINNED per cell (single-rounding
FMA chains, no horizontal reduce) — not hazards. ⚠ **AMENDED 2026-10-16 (SE-2
fix unit)**: the classification's premise — "single-rounding FMA chains" — has
a measured hole: `simd_fused_sub_scale_inplace`'s scalar-fallback arm is
FMA-shaped (`v.mul_add(scale, -(sub·scale))`) while its NEON and AVX2 arms are
two-op (`(v − sub) · scale`) — a scalar-fallback target (wasm32, non-AVX2
x86_64) computes DIFFERENT per-cell bits than the SIMD targets for the same
input. The one sync-boundary consumer (SE-2's `evolve_belief_simd`) is fixed
by delegation; the remaining callers (`softmax_scaled`, the tiled-attention
max-shift) are tolerance-domain (post-exp) — recorded, not fixed. A kernel-level
shape unification (two-op everywhere, or FMA everywhere) is a follow-up if any
future caller is exactness-domain.

⚠ **AMENDED 2026-10-16 (SP-2 fix unit, second instance + a REPAIR)**: the same
audit found the INVERTED shape in the sibling `simd_fused_scale_acc` (and its
f16 twin): the SIMD bodies are FMA (`vfmaq_f32`/`_mm256_fmadd_ps`) but the
NEON/AVX2 tail loops were two-op (`+= scale * src`) while the scalar arm
(wasm32 et al.) is all-`mul_add` — tail cells diverged per-cell across
platforms/lengths. Unlike the sub_scale variant this one had a sync-boundary
consumer (procrustes builds the KG-committed `M` through it at `len = d`,
arbitrary dim), so the KERNEL was repaired at the root, not worked around:
all four tails now round `scale.mul_add(s, *dst)` — single-rounding fused on
every platform, every length. ⚠ One-time committed-bits change: tail cells of
odd-length calls through `simd_fused_scale_acc{,_f16}` move (procrustes `M`,
manifold_power_iter_router accumulators; attention callers are tolerance-domain).

**Local-only candidates** (recorded, no action): `katgpt-sleep`
(`predictability.rs:79`, `consume.rs:86/:103` — sleep-time cache + wake slot
pick, consumed same-node); `katgpt-band` (retention cosines + CI statistics
— per-generation routing); `spectral/ica_lens.rs` + `stiff_anomaly/*`
(interpretability/gate diagnostics); `katgpt-dec` latent cochain folds
(two-brain latent state).

## Part B — tie-band census (picks feeding decisions)

### TOL hazards (tolerance-band picks — the issue's headline class)

1. **`katgpt-claim/clr/brevity.rs:40-77`** (`clr_vote`/`clr_vote_minimal`
   delegate): `threshold = max_reliability - eps` with `tiebreak_eps = 1e-3`
   (`clr/types.rs:54-64`), then min-tokens FI inside the band. A cluster
   0.9e-3 below max is a candidate; 1.1e-3 below is not. Feeds the **CLR
   vote winner**. Decision: keep the band ONLY if documented as by-design
   leniency with the transitivity caveat recorded at the site; otherwise
   strict `cmp_for_max` + documented tie rule. **ADJUDICATED 2026-10-16:
   KEEP + DOCUMENT** — the module doc's false "never changes which answer
   wins" claim (true only of the strict ordering) replaced with the honest
   statement: the band is by-design Research-255 leniency, well-defined per
   call (threshold against THIS call's max — not a pairwise band, so no
   non-transitivity within a call), in-band rule deterministic (min tokens,
   first-encountered). Pins `band_leniency_is_by_design` (0.9e-3 below max +
   shorter wins) + `outside_the_band_the_max_wins` (1.1e-3 below loses).
2. **`katgpt-percepta/evaluator.rs:355,362`** hard attention:
   `score > best_score + 1e-9` / `(score - best_score).abs() <= 1e-9`. Also
   CROSS-COMPONENT INCONSISTENT: `hull.rs`/`BruteAttentionHead` treat only
   bit-equal scores as ties — the two hard-attention implementations
   disagree on what a tie is. Decision: one tie definition for both.
   **ADJUDICATED 2026-10-16: BIT-EQUALITY EVERYWHERE** — evaluator adopts
   the hull's `s == best_raw` (and the Issue-940 law's "never a tolerance
   band"): the 1e-9 f64 band was ~4M ulps of leniency. ⚠ One-time behavior
   change: entries scoring within (0, 1e-9] of the best no longer merge
   into the tie accumulator. Pin `attention_tie_set_is_bit_equality_not_a_band`
   (drives the real `attention_insert_and_query` with a hand-built LookUp:
   a 5e-10 score gap must NOT merge — old code averages 15.0, pinned 20.0).
3. **`katgpt-core/induced_cwm/tournament.rs:416`**: `(wr > best_wr) ||
   ((wr - best_wr).abs() < f32::EPSILON && tb > best_tb)` — an
   `f32::EPSILON` (≈1.2e-7) win-rate band gated on head-to-head;
   non-transitive. Feeds the tournament winner (advance/commit).
   **ADJUDICATED 2026-10-16: EXACT EQUALITY** — win rates are counts over a
   common denominator, so distinct achievable rates differ by ≥ 1/total; the
   band only ever fired for exactly-equal rates. Now `wr == best_wr && tb >
   best_tb` — a (theoretical) sub-EPSILON gap resolves strictly by win rate
   (more correct), exact ties keep the head-to-head path, tertiary
   first-index rule unchanged. ⚠ Effective behavior change: none at
   achievable game counts. Gated validation: `induced_cwm_tournament` 10/10
   (the module is opt-in — the default lib run compiles it to nothing),
   `induced_cwm` full lib 2184.
4. **`katgpt-band/band_conditioner.rs:415-425`** `inverse_normal_cdf_two_sided`:
   tolerance-KEYED dispatch `(a - 0.10).abs() < 1e-3 => z` — a classifier,
   not a pick; an α between grid points (e.g. 0.005) silently gets the 95%
   z. Decision: document the grid + nearest-key rule, or interpolate.
   **ADJUDICATED 2026-10-16: DOCUMENT (no interpolation)** — the doc states
   the four exact levels, the ±1e-3 acceptance windows (float-spelling
   absorption only), and that ANY other α gets the DEFAULT 95% z (not the
   nearest grid z); callers must pass one of the four levels. Interpolation
   declined (a behavior change needing its own GOAT). Pin
   `alpha_between_grid_points_gets_the_documented_default_z` (0.005 → 1.96,
   plus all four on-grid spellings) makes the rule executable-documented.

### UNST hazards (collection-order ties)

1. **`katgpt-ruliology/bandit.rs:426-438` `promoted_winner`** — `max_by` over
   a **`HashSet`**; payoff ties pick the last element in RandomState order
   → run-to-run nondeterministic winner. Worst class. Fix: tiebreak on arm
   index. **FIXED 2026-10-16**: tie → `b.cmp(&a)` so `max_by`'s last-max
   rule lands on the LOWER arm index; pin `promoted_winner_tie_resolves_by_lowest_index_not_hash_order`
   (arms 0+1 tied at payoff 1.0, winner must be 0).
2. **`katgpt-pruners/proof/sketch_population.rs:313` `best_elo`** (+ `:359`
   `sorted_by_elo`/`top_k`) — `HashMap::values()` + `max_by`/stable sort;
   equal-Elo ties (0.0 initial Elo makes this reachable) resolve by HashMap
   order. Fix: id tiebreak. **FIXED 2026-10-16**: `best_elo` tie → lower
   `SketchId` bytes; `sorted_by_elo` fully keyed (elo desc, then id asc — a
   stable sort no longer inherits HashMap order); pins
   `best_elo_tie_resolves_by_lowest_id_not_hash_order` +
   `sorted_by_elo_ties_order_by_id_not_hash_order` (top_k agrees with the
   sorted prefix).
3. **`katgpt-pruners/proof/sketch_sampler.rs:347` `sample_fallback_ucb`** —
   `values_arbitrary()` + `max_by`; exact-score ties possible at equal
   elo+visits. **FIXED 2026-10-16**: tie → lower id; pin
   `fallback_ucb_tie_resolves_by_lowest_id_not_hash_order`.
4. **`katgpt-pruners/trial_log.rs:258-271` `summary`** — HashMap iter +
   `max_by` (report-level "best arm"; lower stakes, same class). **FIXED
   2026-10-16**: tie → lower arm id; pin
   `summary_best_arm_tie_resolves_by_lowest_arm_not_hash_order`.

### NaN-unsafe (panic class, production)

**`katgpt-core/sigmoid_calibration.rs:113,510`** —
`sort_unstable_by(|a,b| a.partial_cmp(b).unwrap())` in `resolution_w_floor`
and `auc_rank_average`. A NaN input aborts the calibration fit. Fix:
`total_cmp` (a NaN then sorts deterministically instead of panicking).
**LANDED 2026-10-16** — both sites → `total_cmp` (no non-NaN behavior
change: `dedup`'s `==` still collapses −0.0/+0.0, the AUC tie groups are
`==`-keyed regardless of sort order); NaN tests added. ⚠ the module is
OPT-IN (`sigmoid_calibration`) — the default-posture lib run compiles it
to NOTHING (the green-zero trap); validation must run at
`--features sigmoid_calibration` (22 module tests + the 2 new NaN pins
green there).

### LAST-INDEX pins owed (deterministic today, unpinned)

The `max_by(cmp_for_max)` family relies on std's last-max-wins semantics
with no pin comment outside `mcts.rs:125-135` and
`bandit/mod.rs:298-300`/`session.rs:248` (the two documented exceptions):
katgpt-pruners (`tes_loop`, `wealth_bandit`, `self_distilling_bandit`,
`mux_freeze_thaw`, the go/fft player family, `riir_feedback`,
`ropd_rubric`), katgpt-core (`cgsp/dual_pool` eviction,
`compression_drafter` final pick, `factorized_action/codebook`). A future
mechanical rewrite to `fold`/`max_by_key` silently flips tie behavior —
each site gets a one-line pin comment in the fix unit. **LANDED
2026-10-16**: 41 uniform pin comments (`Issue 940 LI pin: std max_by/min_by
is LAST-{max,min}-wins — this tie pick is deliberately last-index; a fold
rewrite silently flips ties to first.`) across the 22 named files,
production code only (test mods excluded). Note `max_by_key` sites were
pinned too — also last-max-wins, and equally vulnerable to a fold rewrite.

### Edge noted

`katgpt-pruners/screening/complexity_prior.rs:416-439,600-618` `top_k`: if
every remaining `log_prob` is `-inf`, `best_ix` stays `usize::MAX` and is
written to `out` (silent degenerate index). **FIXED 2026-10-16**: the first
unchosen candidate now always takes the lead (`lp > best_lp || best_ix ==
usize::MAX`), so an all-`-inf` round deterministically picks the LOWEST
unchosen index and the sentinel can never survive a round — in BOTH the
byte and latent variants. Pin `top_k_all_neg_inf_candidates_never_leak_the_sentinel`
(α = +∞ mock proxy drives every log_prob to -inf; out must read [0, 1, 2]).

### GOOD substrate (consume, never fork)

`float_order::{cmp_for_max,cmp_for_min,desc,asc}` (NaN-safe total orders;
twins exist in riir-ai crates) · `simd_argmax_f32` first-index ties,
test-pinned across all three backends (`simd/argmax.rs:16-17`) · the
strict-scan + reverse-index idiom (`pick_domain`, `margin_split`,
`state_option_scoring::argmax_lowest_index_tie`, `two_fidelity_bai`,
`fpcg_selector`) · `entropy_bounded_commit`'s `(NaN-last, key, index)`
total order · `dot_f32_ordered` + the `bridge_certified` certified pattern
· `isMCTS`' fully-keyed `max_by` over HashMap edges (visits → mean → hash
— HashMap order never matters; the model for fixing the UNST class).

## Part C — the fix queue (priority order, each a commit-sized unit)

1. **MB-1 + MB-2** — ordered dot in the micro-belief bridge + attractor;
   docs upgraded from "across runs" to the cross-node claim; divergence
   test mirroring `certified_differs_from_platform_on_cancellation_input`.
   **LANDED 2026-10-16** — plus the forced twin: the Attractor BoM
   `sample_k_states` Phase-1 base activation moved with `step()` (the σ=0
   degeneracy tests pin bit-identity, so the twin was not optional);
   `dot_product_scorer` stays LOCAL with a site comment. G1.4 informational
   270 → 491.7 ns/step, recorded at the MB-2 row. One-time committed-bits
   change: attractor belief states + bridge scalars move at this upgrade
   (downstream replay hashes / BLAKE3 commitments over them re-baseline).
2. **PE-1** — ordered dot in `drift` (the snapshot BLAKE3 then becomes
   platform-stable). **LANDED 2026-10-16** — one-time committed-bits
   change: personality `w`/`r_expected` trajectories and every
   `PersonalitySnapshot.blake3` minted after a drift step move at this
   upgrade (re-baseline downstream audit-event commitments).
3. **MB-4** — hash the ordered dot. **LANDED 2026-10-16** — both planner
   seeds; `nonergodic_g5_identification` (BoM-baseline quality gate) green
   at the new seeds.
4. **HL-1/2/3** — ordered dots (bench the scalar-fold cost at decode-loop
   scale before landing; G2 duty). **LANDED 2026-10-16** — Bench 939
   (`.benchmarks/939_hla_ordered_dot_g2.md`): BEFORE/AFTER A/B, whole-step
   -1.1% .. +2.4% (predicted +0.8% from the raw-dot delta; one dot per
   step), readout path untouched. The census missed a fourth state-feeding
   site (`ahla_per_head_step` :637 → `q_head.n`) — fixed in the same unit;
   divergence pins for both state families (HLA `h`, AHLA `n`) added in
   kernel.rs tests.
5. **SE-2/SE-3/SE-4/SE-7** — ordered kernels or recorded envelope demotion.
   **LANDED 2026-10-16**: SE-2/SE-3-delegation + SE-4 delegate to the pinned
   scalar cores (byte-identical by construction — the strongest form of
   "pin"); the two matrix expand paths the census missed
   (`expand_with_weights`/`expand_batch`) fold with a portable sequential
   FMA chain; SE-7's mean folds through `.sum()`. Two census corrections
   ride the unit: the missed matrix paths, and the elementwise
   classification hole (`simd_fused_sub_scale_inplace` scalar arm is
   FMA-shaped vs the SIMD arms' two-op — recorded, remaining callers are
   tolerance-domain). Tests: 5 new pins + the two tolerance tests
   strengthened to exact bit-equality. Validation: sense 25 default / 32
   `sense_composition` / 37 all-three postures, clippy `-D warnings`
   `--all-features`, katgpt-core 2157/2157 lib. ⚠ One-time committed-bits
   change: a `sense_composition` node's activations/beliefs move at this
   upgrade on every platform (the folds' values change in the last ULP on
   non-cancellation inputs); scalar-posture nodes are untouched.
6. **SP-2 / SP-3 / CL-1 / CL-2 / CL-3 — LANDED 2026-10-16** (fix unit ⑥,
   one commit): SP-2 `m_norm` → `dot_f32_ordered(&m,&m).sqrt()` **plus the
   census-missed uncentered residual arm** (:527 `simd_dot_f32` → ordered —
   confidence-scalar feeder, and it makes both center arms
   reduction-consistent); SP-3 `M = A·V_r` → ordered (steps 2/4/5 already
   sequential FMA); CL-1 both reliability means → `row.iter().sum()`; CL-2
   verdict dot → ordered (also fixes an unchecked-len hazard); CL-3 budget
   sum → `iter().sum()`. PLUS the kernel repair the census surfaced:
   `simd_fused_scale_acc{,_f16}` tails were two-op vs FMA bodies/scalar arm
   (the unit-⑤ hole class, inverted) — procrustes builds the KG-committed M
   through it, so the four tails were fixed to single-rounding `mul_add` at
   the root. Pins: `m_norm_folds_in_index_order_not_platform_order`
   (per-platform ulp-boundary craft search — squares can't cancel; refuses
   to pass vacuously), `rewiring_matrix_folds_in_index_order_not_platform_order`
   (real `project_core`, bit-exact rank-1 algebra, 3.0 vs 6.0),
   `reliability_mean_folds_in_index_order_not_platform_order`,
   `verdict_dot_folds_in_index_order_not_platform_order`,
   `budget_sum_folds_in_index_order_not_platform_order`. Validation:
   katgpt-claim 57 lib (default posture — clr is default-on), katgpt-spectral
   104 procrustes / 101 rewire / 120 combined (GATED postures — both modules
   are opt-in features; the default lib run compiles them to nothing, the
   green-zero trap), clippy `-D warnings` at all three postures, downstream:
   katgpt-canon gated tests 1+5+2, katgpt-speculative 305, katgpt-core 2157,
   katgpt-sense 25, katgpt-forward 131, katgpt-types 162. ⚠ One-time
   committed-bits change: procrustes `m_norm`/rotations/residuals,
   spectral-rewire `M`/ΔW*/overlays, CLR reliabilities + verdicts + MGPO
   budgets, and the fused-scale-acc tail cells all move at this upgrade.
7. **DE-2** — canonical mass order + envelope doc. **LANDED 2026-10-16**
   (fix unit ⑦): `boundary_flux_mass_indexed` documented CANONICAL (with the
   caller-region-order caveat), `boundary_flux_mass_only{,_scratched}`
   documented DIAGNOSTIC with the 1e-4 envelope; the agreement test
   re-annotated (envelope, not bit-identity). Workspace census: no production
   consumer of either path today (the zone_density stampede trigger consumes
   `belief_mass_divergence` = DE-1, already pinned; the pca consumer is the
   report-only morphogen family) — zero behavior change. katgpt-dec 249/249
   lib, clippy `-D warnings`.
8. **NaN-hardening** — `sigmoid_calibration.rs` ×2 → `total_cmp`.
   **LANDED 2026-10-16** — gated-posture validated (see the Part B note:
   the module is opt-in; the default run greens a zero).
9. **UNST ×4 — LANDED 2026-10-16** (fix unit ⑨, one commit): all four
   collection-order ties now resolve deterministically — ruliology
   `promoted_winner` (lower arm index), `best_elo`/`sorted_by_elo`+`top_k`
   (lower SketchId bytes; the sort fully keyed), `sample_fallback_ucb`
   (lower id), `trial_log::summary` best_arm (lower arm id) — each composed
   AFTER the existing `float_order` NaN-safe comparator, exploiting `max_by`'s
   last-max rule (tie → `b.cmp(&a)` so the smaller key wins). 4 pins (one per
   site) assert the lowest-key winner over a tied population — a reversion
   flakes with probability 1 − 1/N per run. Validation: katgpt-ruliology 94,
   katgpt-pruners 126, clippy `-D warnings` both.
10. **TOL adjudication ×4 — LANDED 2026-10-16** (fix unit ⑩, one commit):
   **brevity: KEEP + DOCUMENT** (the band is by-design Research-255 leniency;
   module doc's false "never changes which answer wins" claim corrected;
   per-call well-definedness + deterministic in-band rule documented; pins
   both sides of the band). **percepta: BIT-EQUALITY EVERYWHERE** (evaluator
   adopts hull's `s == best_raw` + the Issue-940 law — the 1e-9 f64 band was
   ~4M ulps of leniency; ⚠ one-time behavior change: sub-band entries stop
   merging; pin drives the real `attention_insert_and_query` with a
   hand-built LookUp). **tournament: EXACT EQUALITY** (win rates are counts
   over a common denominator — distinct rates differ by ≥ 1/total, so the
   f32::EPSILON band only ever fired for exact ties; `wr == best_wr` now,
   non-transitive band expression gone; gated validation at
   `induced_cwm_tournament` — the module is opt-in, the default run compiles
   it to nothing). **α-grid: DOCUMENT, no interpolation** (the four levels +
   ±1e-3 windows + the default-95% fallthrough stated at the site; between-grid
   α gets the DEFAULT z, not nearest — pinned executable; interpolation
   declined: a behavior change needing its own GOAT). Validation:
   katgpt-percepta 40, katgpt-claim 59, katgpt-band 25 (α pin), katgpt-core
   2157 default + 2184 `induced_cwm` + 10 tournament gated; clippy
   `-D warnings` all.
11. **LI pin comments + the `usize::MAX` edge guard — LANDED 2026-10-16**
    (fix unit ⑪, one commit): 41 uniform LAST-index pin comments across the
    22 named files (max_by/min_by/max_by_key — the comment names std's
    last-extreme-wins rule and the fold-rewrite hazard);
    `complexity_prior::top_k` both variants guarded (first-unchosen lead →
    the all-`-inf` round picks the lowest unchosen index deterministically,
    the sentinel never survives) + the sentinel-leak pin test. Validation:
    katgpt-pruners 126 lib (incl. the new pin), katgpt-core 2183/2184 at
    `induced_cwm` (the 1 failure a KNOWN loaded-box latency-guard artifact,
    passes alone per its own panic message), clippy `-D warnings` both.

## Honest scope

- **Threat model:** same-arch fleet = the hazard is latent (same binary,
  same bits); mixed-arch quorum/replay (M3 aarch64 ↔ 4090 x86_64 ↔
  wasm32 browser clients) = ACTIVE — three different lane orders exist in
  the wild today.
- This audit changes no code. Every fix above is behavior-preserving
  per-platform (the fold value differs from the SIMD value only in the
  last ULP on non-cancellation inputs) but CHANGES committed bits on the
  fixed lanes — any consumer carrying stored BLAKE3 commitments
  (PersonalitySnapshot, SenseModule, freeze deltas) sees a one-time
  commitment change at the upgrade boundary. Freeze/thaw versioning
  absorbs this; the fix units note it.
- Consumer confirmations for sibling-repo claims (riir-engine
  HlaCacheProxy→NeuronShard freeze, sync_relation_gate, cgsp pulse_bridge)
  are inferences from this repo's docs + the workspace AGENTS rules;
  verified against the siblings at fix time if a fix unit needs it.

## The fix commits (Part C landed 2026-10-16 — 11/11 units)

`d279c7c24` audit · `b66056fcc` unit ① (MB-1/2) · `bb0ffb339` ② (PE-1) ·
`bef1f0f2f` ③ (MB-4) · `052f7e732` ④ (HL-1/2/3, Bench 939) ·
`0419d5e52` ⑤ (SE-2/3/4/7) · `5b4f3fdf7` ⑥ (SP-2/3 + CL-1/2/3 + the
fused-scale-acc tail repair) · `fa45702f5` ⑦ (DE-2 canonicalize) ·
`4fde9c260` ⑧ (NaN-hardened sorts) · `4e3ad9a84` ⑨ (UNST ×4 tiebreaks) ·
`4ac43fba7` ⑩ (TOL ×4 adjudication) · `23a04077c` ⑪ (LI pins + the
usize::MAX sentinel guard). The tracking issue was removed per
noise-reduction after this record; its full per-unit text lives in git
history at `23a04077c~1:.issues/940_float_reduction_order_tie_band_determinism_audit.md`.

## References

- Research 621 (handwired distill) · handwired `docs/patterns.md`
  (grouping analysis, `emitArgmax` tie-band law, `emitHighestScoreExact`)
- The certified precedent: `katgpt-core/src/closure/bridge_certified.rs`,
  `benches/bench_844_exact_sigmoid_ordered_dot.rs`, riir-chain Issue 156 T1
- Sibling halves: riir-ai Issue 1074 (game sync surfaces), riir-ai Issue
  1075 (swir penalty-ledger defects)
- `.issues/940_float_reduction_order_tie_band_determinism_audit.md` (this
  audit's issue; the fix queue lives there as unchecked tasks)
