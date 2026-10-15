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
| **MB-4** `micro-belief/bom_arena.rs:343,444` | `simd_dot_f32(..).to_bits() as u64` → RNG seed → K hypotheses → **mean committed back into `state`** | PLATFORM | belief state (→ MB-1) | **PIN** — hash the ordered dot; a platform-dependent float hashed into a seed diverges the whole hypothesis set cross-platform |
| MB-3 `micro-belief/leaky.rs:155` | scalar `.sum()` | PINNED | belief → bridge | good |
| MB-5 `micro-belief/bom.rs:352-359,279` `sample_k_states` / `dot_product_scorer` | `simd_dot_f32` per hypothesis | PLATFORM | distributional steering (in-process inference; nothing hashed/committed) | **LOCAL** (recorded) — winner selection consumed in-process. **AMENDED 2026-10-16 at fix unit ①**: the census's line range conflated two things — the `sample_k_states` Phase-1 base activation is BIT-COUPLED to `step()` by the pinned σ=0 degeneracy contract (`bom_sigma_zero_matches_step_attractor` asserts exact equality), so it went ordered WITH MB-2 (a LOCAL call there was not available without breaking a pinned test); only `dot_product_scorer` + the per-hypothesis winner pick stay platform SIMD/LOCAL (site comment added) |
| SE-1 `katgpt-sense/reconstruction.rs:994-1000` `evolve_belief` | scalar fold over `[f32; 6]` | PINNED | belief → SyncBank → relation scalars (the 930 lane) | good |
| **SE-2** `reconstruction.rs:1093` `evolve_belief_simd` (`sense_composition`) | `simd_sum_f32` | PLATFORM | belief → SyncBank → relation scalars; the in-repo test pins scalar-vs-SIMD ≤ 1e-5 and states "bit-equality would assert more than the underlying belief channel provides" | **PIN** (ordered sum) or demote the feature's replay claim in its docs — the fix unit picks; today the feature silently breaks cross-node bit-identity that SE-1 preserves |
| **SE-3** `reconstruction.rs:791` `expand_simd` (`sense_composition`) | `simd_dot_f32` per module | PLATFORM | kind_activations → belief → SyncBank | **PIN with SE-2** |
| **SE-4** `reconstruction.rs:925` `route_simd` (`sense_composition`) | `simd_sum_f32` mean → threshold mask | PLATFORM | which modules' evidence enters the belief (razor-edge flip at the mean threshold) | **PIN with SE-2** (scalar `route` at :879 is the pinned twin) |
| SE-5 `sense/octree.rs:128-141` `embedding_to_ternary` | scalar fold (`scale_sum`) | PINNED | ternary dirs → `SenseModule::commit()` **BLAKE3** | good |
| SE-6 `sense/schema_centroid.rs:44-52,185` | Welford scalar folds | PINNED (given ordered input slice) | centroid → seeded `KgEmbedding` → BLAKE3/Merkle | good — fix-time note: callers must pass deterministically-ordered classes |
| **SE-7** `sense/bake.rs:92` `precision_to_confidence` | `simd_sum_f32(lambda)` | PLATFORM | confidence → **Merkle leaf bytes** (octree.rs:181,197) | **PIN** |
| SE-8 `sense/reconstruction_depth_invariance.rs:141` | delegates to SE-1 + scalar `sum_sq` folds | PINNED | RMS-rescaled belief → SyncBank | good |
| **HL-1/2/3** `katgpt-hla/kernel.rs:123,194,474` | `simd_dot_f32` (`k_mq`, `q_mk`) | PLATFORM | **mutate `hla_moments`** — the Cold-tier Pod, BLAKE3-committed on freeze | **PIN** — the only platform-dependent inputs into committed HLA state; G2 (decode-loop cost of the scalar fold) is the fix unit's bench duty |
| HL-4 `kernel.rs:321-322,389-390,494` readout denoms | `simd_dot_f32` | PLATFORM | attention readout → logits → int token ids | **LOCAL** (recorded) — tokens, not bits, cross the wire |
| **PE-1** `katgpt-personality/kernel.rs:227-235` `drift` | `simd_dot_f32(&d_recent, &ONES)` | PLATFORM | `w` → `PersonalitySnapshot.blake3` — "The **commitment** … IS synced as an audit event when a hot-swap occurs" (snapshot.rs:12-14) | **PIN** — exact structural twin of the certified `bridge_certified` lane; adopt the same pattern |
| **SP-2** `katgpt-spectral/procrustes.rs:445` `m_norm` | `simd_sum_sq(..).sqrt()` pre-scale | PLATFORM | `M̃` → Newton-Schulz → aligned-entity match "committed as a raw KG triple with a confidence scalar" (module doc; riir-chain Issue 001) | **PIN** |
| **SP-3** `spectral/spectral_rewire.rs:308` (`spectral_rewire`, opt-in) | `simd_dot_f32` building `M = A·V` | PLATFORM | purified weight deltas → BLAKE3-committed freeze/thaw overlays | **PIN** |
| SP-1 `spectral/hla_eigenbasis.rs:571-585` | `simd_dot_f32` power iteration | PLATFORM | eigenbasis — "stay local to the NPC — never synced" (file doc :41-45); provenance hashes the INPUT window, not eigvecs | **LOCAL** (recorded, by the file's own law) |
| **CL-1** `katgpt-claim/clr/vote.rs:133,232` | `simd_sum_f32(row)/m` reliability mean | PLATFORM | CLR verdict reliability → compaction gate / jury | **PIN** (verdict = decision surface) |
| **CL-2** `claim/clr/verifier.rs:90` | `simd_dot_f32` → sigmoid verdict | PLATFORM | per-claim verdicts → CL-1 inputs | **PIN with CL-1** |
| **CL-3** `claim/clr/mgpo.rs:83` `allocate_budget` | `simd_sum_f32(weights)` | PLATFORM | proportional allocation `.floor()` — integer flip at boundary → different sampling budgets | **PIN** |
| **DE-2** `katgpt-dec/stokes_calculus.rs:287-292,358-363` | TWO mass paths, each a pinned scalar fold but in DIFFERENT orders; repo test pins agreement at 1e-4, naming "float accumulation-order differences between the two paths" (:810) | PINNED×2, mutually divergent | belief-region mass → stampede trigger (`zone_density.rs:431`), conservation checks | **CANONICALIZE** — pick ONE order (the indexed form) as the replay-compared mass; the other stays diagnostic with the 1e-4 envelope documented |
| DE-1 `stokes_calculus.rs:80` `belief_mass_divergence` | scalar `.sum()` of abs | PINNED | divergence scalar → curiosity / ICT branching | good |
| DE-3/DE-4 + `katgpt-dec` latent-field folds | scalar folds / `katgpt_dec::simd` source-pinned twin (NOTE: different bits than the types-crate kernel — one implementation, two spellings) | PINNED | local solver/latent state | good |

Elementwise/axpy kernels (`simd_outer_product_acc`, `simd_transpose_matvec_acc`,
`simd_add_inplace`, `SyncBank::observe`, …) are PINNED per cell (single-rounding
FMA chains, no horizontal reduce) — not hazards.

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
   strict `cmp_for_max` + documented tie rule.
2. **`katgpt-percepta/evaluator.rs:355,362`** hard attention:
   `score > best_score + 1e-9` / `(score - best_score).abs() <= 1e-9`. Also
   CROSS-COMPONENT INCONSISTENT: `hull.rs`/`BruteAttentionHead` treat only
   bit-equal scores as ties — the two hard-attention implementations
   disagree on what a tie is. Decision: one tie definition for both.
3. **`katgpt-core/induced_cwm/tournament.rs:416`**: `(wr > best_wr) ||
   ((wr - best_wr).abs() < f32::EPSILON && tb > best_tb)` — an
   `f32::EPSILON` (≈1.2e-7) win-rate band gated on head-to-head;
   non-transitive. Feeds the tournament winner (advance/commit).
4. **`katgpt-band/band_conditioner.rs:415-425`** `inverse_normal_cdf_two_sided`:
   tolerance-KEYED dispatch `(a - 0.10).abs() < 1e-3 => z` — a classifier,
   not a pick; an α between grid points (e.g. 0.005) silently gets the 95%
   z. Decision: document the grid + nearest-key rule, or interpolate.

### UNST hazards (collection-order ties)

1. **`katgpt-ruliology/bandit.rs:426-438` `promoted_winner`** — `max_by` over
   a **`HashSet`**; payoff ties pick the last element in RandomState order
   → run-to-run nondeterministic winner. Worst class. Fix: tiebreak on arm
   index.
2. **`katgpt-pruners/proof/sketch_population.rs:313` `best_elo`** (+ `:359`
   `sorted_by_elo`/`top_k`) — `HashMap::values()` + `max_by`/stable sort;
   equal-Elo ties (0.0 initial Elo makes this reachable) resolve by HashMap
   order. Fix: id tiebreak.
3. **`katgpt-pruners/proof/sketch_sampler.rs:347` `sample_fallback_ucb`** —
   `values_arbitrary()` + `max_by`; exact-score ties possible at equal
   elo+visits.
4. **`katgpt-pruners/trial_log.rs:258-271` `summary`** — HashMap iter +
   `max_by` (report-level "best arm"; lower stakes, same class).

### NaN-unsafe (panic class, production)

**`katgpt-core/sigmoid_calibration.rs:113,510`** —
`sort_unstable_by(|a,b| a.partial_cmp(b).unwrap())` in `resolution_w_floor`
and `auc_rank_average`. A NaN input aborts the calibration fit. Fix:
`total_cmp` (a NaN then sorts deterministically instead of panicking).

### LAST-INDEX pins owed (deterministic today, unpinned)

The `max_by(cmp_for_max)` family relies on std's last-max-wins semantics
with no pin comment outside `mcts.rs:125-135` and
`bandit/mod.rs:298-300`/`session.rs:248` (the two documented exceptions):
katgpt-pruners (`tes_loop`, `wealth_bandit`, `self_distilling_bandit`,
`mux_freeze_thaw`, the go/fft player family, `riir_feedback`,
`ropd_rubric`), katgpt-core (`cgsp/dual_pool` eviction,
`compression_drafter` final pick, `factorized_action/codebook`). A future
mechanical rewrite to `fold`/`max_by_key` silently flips tie behavior —
each site gets a one-line pin comment in the fix unit.

### Edge noted

`katgpt-pruners/screening/complexity_prior.rs:416-439,600-618` `top_k`: if
every remaining `log_prob` is `-inf`, `best_ix` stays `usize::MAX` and is
written to `out` (silent degenerate index).

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
   platform-stable).
3. **MB-4** — hash the ordered dot.
4. **HL-1/2/3** — ordered dots (bench the scalar-fold cost at decode-loop
   scale before landing; G2 duty).
5. **SE-2/SE-3/SE-4/SE-7** — ordered kernels or recorded envelope demotion.
6. **SP-2 / SP-3 / CL-1 / CL-2 / CL-3** — ordered.
7. **DE-2** — canonical mass order + envelope doc.
8. **NaN-hardening** — `sigmoid_calibration.rs` ×2 → `total_cmp`.
9. **UNST ×4** — id/index tiebreaks (ruliology first).
10. **TOL adjudication ×4** — strict comparator or documented-band
    decision at each site (brevity, percepta+hull consistency,
    tournament, α-grid).
11. **LI pin comments** + the `usize::MAX` edge guard.

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

## References

- Research 621 (handwired distill) · handwired `docs/patterns.md`
  (grouping analysis, `emitArgmax` tie-band law, `emitHighestScoreExact`)
- The certified precedent: `katgpt-core/src/closure/bridge_certified.rs`,
  `benches/bench_844_exact_sigmoid_ordered_dot.rs`, riir-chain Issue 156 T1
- Sibling halves: riir-ai Issue 1074 (game sync surfaces), riir-ai Issue
  1075 (swir penalty-ledger defects)
- `.issues/940_float_reduction_order_tie_band_determinism_audit.md` (this
  audit's issue; the fix queue lives there as unchecked tasks)
