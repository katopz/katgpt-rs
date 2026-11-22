# Issue 1238 — Fused multi-branch FUNCATTN (branch-parallel composition cell)

**Status:** OPEN — unclaimed. Pre-registered GOAT cell below; no run scheduled. Verdict provenance: MBT (arXiv:2611.19800) research verdict, modelless track PASS → GAIN, reviewer AGREE round 2 (2026-11-22) with both A6 conditions folded in below.

## Pinned claim (§1.5 precondition)

"Fused multi-branch FUNCATTN — K matched-width functional-attention branches executed in parallel over a shared token set with an UNLEARNED fusion policy — is a composition family member with zero shipped implementations on the FUNCATTN host (grep-verified 2026-11-22 across plans + research + src: CHIAR blend = 2-operator input-gated ROUTING; Plan 299 T3.6 = lookup multiplexing; Plan 272 = search-graph reference sets; Plan 417 = SIMD audit; guided_width = decode-free selection), for which MBT (arXiv:2611.19800) supplies the corpus's only matched-compute branch-composition evidence. CORRECTION 2026-11-22 (LEO-surface fusion hunt): the claim is host-scoped — the stack DOES ship one composition-class member on the FLOW host: `get_or_compute_dual_postmax` (`katgpt-core/src/flow/cache.rs`, Plans 459/460) = K=2 heterogeneous frozen branches (LEO teacher + UVFA student), unlearned convex fusion after the per-action max-pool, shared linear FFT coda; its 459(pre-max, demoted)→460(post-max, promoted) history is the in-stack precedent for MBT's §4.4 placement law (see Cross-refs). This issue adjudicates select-vs-fuse-vs-wide on the funcattn host; it asserts no gain."

## Source

- arXiv:2611.19800 "Multi-Branch Transformer" (Li, Chen, Deng, Xu, Chen, De Sa, He, Wang, Tong — UIUC/Rice/Michigan/Cornell, 2026-11-15). Training code + HF checkpoints public. Distilled 2026-11-22 under the research workflow; PASS-Redirect lines committed at riir-train `b89be9b77` (Joint-KV = the branch-axis member of the shared-KV family), riir-ai `14a0b05c4`, riir-infer `513795a6`.
- Host substrate (all shipped): `katgpt-core::funcattn` (Research 257 / Plan 286; default-on `776000b9`; G1–G6 green), `katgpt-attn::funcattn_compose::{chiar_blend, freeze_thaw}`, `FuncAttnSnapshotStore` (BLAKE3 commitment, atomic Arc-swap).

## Shipped composition-family inventory (corrected 2026-11-22, doc-surface hunts)

The stack ships ≥3 composition-class members — NONE on the funcattn host (the host-scoped claim above stands):
1. **Flow dual** — `get_or_compute_dual_postmax` (`katgpt-core/src/flow/cache.rs`): K=2 heterogeneous frozen branches (LEO teacher + UVFA student), unlearned convex fusion after the per-action max-pool, shared linear FFT coda (Plans 459/460 — see Cross-refs).
2. **`GdnHolaTreeVerifier`** (`katgpt_core::gdn_tree_verify::hola_fusion`, Plan 430, feature `gdn_hola_tree_verify`) — K=2 heterogeneous paths (compressive GDN + exact-recall HOLA), parameter-free additive fusion `O[i] = O_gdn[i] + O_hola[i]`, shared committed prefix `S₀` read across all draft-tree branches (the Joint-KV shape on the specdec host).
3. **`VelocityFieldEnsemble`** (`katgpt_core::velocity_field_ensemble`, Plan 376, UNGATED — always compiled, lib.rs; GOAT G1–G4) — `b̂ = Σηᵢ·bᵢ` with η SOLVED via ridge (`ridge_solve_direct_f32`): a LEARNED combiner that MBT Table 6 directly adjudicates; the uniform-η=1/P vs solved-η ablation has never run (rider below).

**Design cites for the cell:** (a) `ScaleNormalizedFusion` (`katgpt-core/src/causal_head_importance/fusion.rs`, Plan 358) — the shipped per-head RMSNorm+γ fusion operator, γ=1.0 default = identity, currently unused: the ready-made operator for the heterogeneous-branch variant's unlearned normalized-sum policy. (b) Sum-arm ceiling failure — riir-ai `crates/riir-games-mmorpg/src/hero_routine_social_tests.rs::sum_arm_degrades_threat_response_vs_max` (Sum hunts where Max flees). Mean keeps Flee share 0.022 where Max/c=0 gives 0.000 (riir-ai `.benchmarks/799_spectral_promotion_brief.md` row 88). Together they back a bounded, normalized fusion over raw Σ. They do not rank Mean against Max for the funcattn cell.

## Guard rows (MBT vocabulary as grep bait; each restates an existing law)

- **G-1 `mbt_table6_not_an_artifact_blend_license`** — Table 6's scope is ACTIVATIONS within one model at matched params. Quoting it to blend decision ARTIFACTS (specialist heads, vessels, rulegram epochs) violates arsenal law A2 ("never blend — selection is monotonic atomic hot-swap") and the vessel monotonic-apply/digest-pin identity semantics: answers become unattributable (broken, not slow).
- **G-2 `per_npc_branch_fusion_adverse_prior`** — "K personality branches per NPC, fused" breaks runetrace attribution (which branch drove the FSM transition?), HlaCacheProxy 5-scalar sync provenance (whose valence crosses?), and block-λ per-affect-block keying (undefined for a fused mixture). The legitimate form already exists: identity-keyed unit directions (Plan 514 / Issue 1239's substrate). W1-K2 (0.93–0.99×) is the named adverse prior; any proposal must pre-register against it AND extend runetrace rows + block-λ + the sync story to name the branch.
- **G-3 `shared_cache_stays_local_compute`** — Joint-KV is in-process compute; at any sync/process seam the raw-scalars-only bridge law holds (dot+sigmoid projection, clamp; Issue-940 pins).
- **G-4 MCGS position-invariance** — `multi_branch_aggregate`'s union stays parameter-free; any weighted-upgrade proposal must prove position-invariant weights or Plan 061/065's invariance property (the pillar's sloppy-test) breaks.

## MBT→fleet translation record (canonical map, 2026-11-22 sweep)

| MBT axis | Fleet operator names |
|---|---|
| BRANCH | BranchBank/BranchRouter (Plan 329, memory branches, non-interference) · BanditPruner arms · FusionArm (riir-games-mmorpg hero_routine) · tree-verify branches |
| PRELUDE/CODA | LEO shared linear FFT coda (Plans 459/460) · OnlineCompactor prefix+suffix |
| FUSION | CHIAR blend (routing) · DualLeoMixer (convex) · ScaleNormalizedFusion (unused operator) · hola_fusion (additive) · VelocityFieldEnsemble (learned η) · sigmoid_fuse_multi_branch_into (lookup multiplexing) |
| JOINT-KV | gdn_tree_verify S₀ shared prefix · bekko shared-prefix (riir-infer 006) · mtp_shared_kv (LeviathanVerifier) · Plan 297 CrossGamePrefix (shared compact KV + per-game queries) · katgpt-kv SharedKV (within-attention K-as-V, different axis) · contextual_shared_pass |
| ALLOCATION (width-depth) | guided_width (Issue 895) · bench_gram_width_depth + issue_407_t5 ("Depth dominates width": depth +6.46% vs width +0.16% — the cross-domain counterpoint at corpus scale) |
| PLACEMENT (middle) | Plans 459 (pre-max, demoted) → 460 (post-max, promoted) — the in-stack placement precedent |
| HEAD FOLD | head-gang ladder (mq8) · linking_fold · tree-nodes-folded-into-one-masked-solve (specdec) |
| W1-K2 NEGATIVE | kernel_opt `fusion-payback-gated-on-per-dispatch-overhead` + `work-scaling-probe-launch-bound` (mined rules) · `min_npcs` gate (flow) · npc_brain router threshold invariant |
| SPEEDUP MODEL | the Amdahl family: Plan 096 `SpecCostSnapshot` + bench_051 prediction-accuracy gate + kernel_opt `amdal_ceiling_wrapped_modules` (= the 1/(1−s) bound) · ane_roofline thresholds |

## Advisory riders (doc-sync lanes own these; recorded here for greppability)

- **MoA doc-rot:** the `04_supergoat_candidates/moa_activations.md` file carries the Attention-Matching (Plan 297) content, NOT the Research-126/Plan-158 MoA mechanism the README names — fix or split at next doc-sync. Same pass: `fusion_arm.rs`'s in-source "Bench 027" pointer is stale (riir-ai Bench 027 is gguf-model-loading; the figures live in riir-ai Bench 799).
- **Two-brain guard-line:** MBT's shared-cache law is contra-indicated on think-brain caches — per-NPC belief privacy is the fog-of-war axiom, never a byte-saving opportunity.
- **Fusion-policy scale boundary:** fusion-policy freedom exists at stack scale (Table 6), not at activation/FFN-layer scale (LA ⊊ MoA; W1-K2) — guards a future "simplify MoA's gates" proposal both theories predict loses.
- **Uniform-mix-MoA falsifier** (proposer-gated, no consumer pull): uniform-mix vs learned-gate MoA at matched bytes on the Bench-049 FFN fixture.
- **Plan 376 uniform-η ablation:** at the next Plan-376 touch, run η=1/P vs solved η — a tie records the Table-6 corroboration and retires the "learned fusion" claim (one cell, not a plan).
- **Bench 027→799 corroboration leg:** at the next hero-fusion A/B rerun, record whether Spectral sits within noise of Max/Mean at K∈{2,3} (record-only).
- **Gram-family all-orthogonal** (first dated sweep 2026-11-22): topogram hosts are discrete-document and selection-shaped (Compose+Abstain is ingestion, never sum-fusion); the repair descent's best-of-K IS the correct discrete fusion analog (randomized_starts caveat keeps every repair instance single-chain); bench 013's allocation curve is L-shaped to an entropy floor, not MBT's U; carrier needs no fusion term (closed v1 + two-instance bar). Re-open conditions: fill-leaf LLM seam with a real consumer; a start-independent multi-basin repair instance with a production caller.

## Why filed (Gain bar, honestly stated)

1. **No shipped member** — the unblocks-support rests on this grep alone. CHIAR's blend is unlearned (fixed τ=0.865/β=12) but ROUTING-class: an input-dependent gate softly picks which of TWO operators dominates per token. COMPOSITION-class fusion (all K branches always execute at matched width; outputs aggregated at a fixed ratio) has no implementation anywhere in the stack.
2. **MBT supplies design laws, not a contradiction.** Its Table 6 (all six learned fusions within 0.005 loss; the parameter-free normalized sum `(h₁+h₂)/√2` within 0.013) supplies a PRIOR that a fixed unlearned sum is a sufficient fusion policy — making the unlearned policy the DEFAULT arm, not a compromise. §4.4 middle placement fixes the shape (shared stem/out-projection, parallel middle only). The matched-compute width-allocation protocol fixes the experimental design. Table 5 (shared beats private KV at matched bytes, every scale) motivates arm A4. The W1-K2 decode NEGATIVE (0.93–0.99× — composition overhead ate the depth saving at small scale) is the named overhead-risk arm.
3. **Mechanism class is textbook parallel averaging.** The novelty posture is decision-content for this stack only: the cell adjudicates select-vs-fuse-vs-wide; it never asserts a win. A null result is a documented negative (the Plan-286 T5.1 precedent).

## Feature home

- `funcattn_branch_parallel = []` — NEW standalone opt-in in `katgpt-attn/Cargo.toml` (verified absent 2026-11-22: the manifest carries only `funcattn_freeze_thaw`, `funcattn_spectral_pre_rotate`, `funcattn_chiar_blend`; the `funcattn_compose/mod.rs` doc's parent feature is aspirational, NOT a manifest feature).
- Optional sub-task (never assumed by the cell): create the `funcattn_compose = [...]` parent feature the module doc already advertises.
- Never joins `default`/`full` except per the Promotion ceiling below.

## Pre-registered arms

Trained arms are small diagnostic models under the Plan-286 §Q3 FD-SGD precedent (standard small-model diagnostic training for GOAT gates is accepted).

| Arm | Composition | Params | Fusion |
|---|---|---|---|
| A1 | single wide branch (incumbent) | P | — |
| A2 | K=2 matched-width, joint FD-SGD | ≈P | Σhᵢ/√K (joint training legitimizes 1/√K — later layers adapt to the scale) |
| A3 | = A2 | ≈P | learned concat-projection (tests Table 6's "learned adds nothing" ON THIS HOST) |
| A4 | = A2 + shared slice-basis Φ, branch-local Q/K/V | ≈P | Σhᵢ/√K (the Table-5 shared-cache axis) |
| A5 | single branch at 2× width (the paper's Local-wide-core control) | 2P | — |
| **A6** | **K=2 FROZEN independently-trained half-width snapshots loaded via `FuncAttnSnapshotStore`, ZERO joint training** | ≈P | **Σhᵢ/K — plain mean, pinned per verdict: independently-trained snapshots are complete predictors, so 1/√K inflates the output by ~√K exactly when the predictors agree (the ensemble's good case); Table 6's 1/√2 normalization holds only under joint training. 1/√K stays reserved for A2.** |
| A6b (optional named negative control) | = A6 | ≈P | Σhᵢ/√K — expected to LOSE to A6; its loss confirms the scale law |

**A6 readout posture — pin ONE before any training run (verdict condition; `FuncAttnWeightsSnapshot` stores only `w_basis`, `w_q`, `w_k`, `w_v` — no stem, no out-projection):**

- (i) **Preferred:** stem + out-projection trained ONCE with the incumbent, then FROZEN; each snapshot trains only its FUNCATTN weights against the frozen readout; fusion averages branch outputs. Alloc-equivalent to the incumbent.
- (ii) Each snapshot keeps its own readout; fusion averages FINAL predictions.

Snapshots are HALF-width so "matched total params" holds: two full-width frozen branches = 2P = A5's budget, not A1's.

## Pre-registered gates

- **Step budgets (both, always):** 150 FD-SGD steps AND a ≥500-step convergence posture. Bench 058 caveat: SDPA closes to ~2× by 500+ — a pass claimed only at 150 steps is not a pass.
- **Statistics:** ≥5 seeds per arm, paired-difference bound (house convention δ = max(1.0pp, 2.5·SE) on relative MSE). A 0.013-scale effect flips on one seed.
- **G1 determinism:** bit-identical re-run (`float_order` discipline; ties → lowest index).
- **G2 fused-vs-single:** A2 MSE ≤ A1 MSE at BOTH budgets, else demote-to-null-record.
- **G3 beyond-width:** A2 ≤ A5 at BOTH budgets. If A5 wins, the branch-composition claim FAILS on this host — recorded honestly (the paper's own Local-wide-core control class).
- **G4 alloc-free hot path** (`FuncAttnScratch` pattern).
- **G5 latency:** A2 wall ≤ 2.2× A1, with per-branch shapes PINNED (per-branch `d`, per-branch `k`, per-branch k×k Tikhonov solve — the k³ term is PER BRANCH; this fixes the denominator). Named risk: the paper's W1-K2 negative (composition overhead dominating at small scale).
- **A6 gate (the modelless arm):** fused frozen mean ≤ single frozen incumbent at matched total params, both budgets, seed-paired. This is the ONLY arm whose outcome can ever matter for default promotion.

## Promotion ceiling

A G2/G3 pass is ARCHITECTURE evidence whose real consumers need trained branch weights (riir-train lane / riir-ai Plan 318 rank-k work). The feature stays opt-in regardless of the trained arms' outcome. **DEFAULT promotion requires the modelless arm A6 to win its own gate** — never the trained diagnostics alone.

## Weakest points (named at filing)

1. A5 may win — wide often beats matched-compute branching; the cell exists to adjudicate that.
2. Mechanism class is textbook parallel averaging; the claim is decision-content for this stack, nothing more.

## Tasks

- [ ] Claim: `git fetch origin && py ../katgpt-rs/scripts/claim_lease_check.py --item 1238`; append the `**Claimed:**` line (own line, nothing after the expiry token); push BEFORE implementing (48 h expiry)
- [ ] Add `funcattn_branch_parallel = []` to `katgpt-attn/Cargo.toml` (+ optional `funcattn_compose` parent sub-task)
- [ ] Pin the A6 readout posture ((i) or (ii)) IN THIS FILE before any training run
- [ ] Harness (reuse the Plan-286 T3.2 pattern): arms A1–A5, both step budgets, 5 seeds
- [ ] A6 snapshot pipeline: train two half-width single-branch snapshots under the pinned posture; load via `FuncAttnSnapshotStore`; fuse Σhᵢ/K; optional A6b negative control
- [ ] Run the cell on amo/bmo (never m3 — job-lane law); record paired stats + box state (quote `bench_preflight`'s PROVENANCE line)
- [ ] Record the verdict in `.benchmarks/` (next free number); demote-to-null-record on G2/G3 failure in the same commit as the run
- [ ] If A6 wins: file the promotion proposal separately (Feature Flag Discipline); never auto-promote

## Cross-refs

Research 257 (FUNCATTN distill) · Plan 286 (the primitive + GOAT precedent) · Plan 299 T3.6 (`sigmoid_fuse_multi_branch_into` — the lookup-side multi-branch kernel) · Research 239 L180 (MCGS graph-branch fusion gap — ANALOGY only, different axis) · `katgpt-core::guided_width` (selection-fused rollouts — consumer candidate if the composition wins) · the MBT PASS-Redirect lines (riir-train `.research/470` — Joint-KV as the branch-axis shared-KV member; riir-infer `.research/006` — bekko shared-prefix corroboration) · riir-train Plan 477 (the TRAINED parallel-drafter sibling lane — different track; claim lease expired 2026-11-07, claimable).

**LEO-surface corroboration (2026-11-22 fusion hunt):** the flow dual (`get_or_compute_dual_postmax`, Plans 459/460) is the K=2 composition precedent on the flow host — 459 fused BEFORE the per-action max-pool and was DEMOTED (25.9% vs 30% gate; `max_a(α·QL+(1−α)·QU) ≠ α·max_a QL+(1−α)·max_a QU`), 460 fused AFTER the pool with a shared linear FFT tail and was PROMOTED (31.5% at α=0.10, synthetic-gated) — MBT §4.4's middle-placement law independently corroborates the fleet's 2026-07-18 finding. Joint-KV does not transfer to that host (frozen heads + linear tail: deeper sharing collapses to the blend; the deep form was measured and lost). K>2 multi-teacher fusion is gated behind riir-ai Research 322's civ dual-LEO closure; re-open trigger recorded: "Plan 155 lands ≥2 cross-game teachers AND a consumer needs multi-teacher navigation fusion." Issue 1239 (KARC host) is the sibling composition filing.

**Rating-surface PASS (2026-11-22 fusion hunt):** "condition Elo" is ~80% shipped — `elo_k_eff = K·min(1, n/16)` (linear ramp, pinned in-source, `riir-refine/src/elo.rs`) already conditions K on evidence count; Bench 042 measured selection-weighting fusion at no headroom ("the blend weight was never a live lever"); shared-pool branch-local K contradicts the damping law's own rationale; branch-parallel Elo replay violates the Issue-940 determinism law. The one live seam is CONDITIONAL on this issue's cell winning AND composition becoming a standing serving lane (named consumer: `guided_width`): rate standing configs via the existing `riir-scenelab` `Tournament::elo_table`, with G-bar = champion-by-Elo ordering must match this cell's pre-registered verdict (mirror-contract anchor). Until then: no rating work on this lane.
