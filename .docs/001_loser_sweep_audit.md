# Phase 0.5 — Loser-Sweep Audit (Proposal 003)

Status: **complete** (2026-07-01)
Basis: 4 parallel subagent passes over every opt-in feature in `Cargo.toml`,
each cross-checked against `.benchmarks/`, `.plans/`, `.issues/`, and source
`#[cfg]` gating. Conflicts reconciled by the coordinator.

## Methodology

For each opt-in (non-default) feature:
1. Read its `Cargo.toml` inline comment for the explicit GOAT verdict.
2. Grep `.benchmarks/`, `.plans/`, `.issues/` for the cited plan + GOAT result.
3. Verify whether the feature gates real code (`#[cfg(feature = "...")]` hits).
4. Classify into one of 3 categories per Proposal 003 §"The `katgpt-deprecated`
   loser crate" → "Membership criteria — 3 categories of opt-in".

### The 3-category rule (recap)

| Cat | Name | Criterion | Action |
|---|---|---|---|
| 1 | **PENDING** | GOAT gate hasn't run yet ("opt-in until G1–G4 pass", "GOAT candidate") | **Stays** in domain crate — exiling punishes WIP |
| 2 | **BENCH-LOSER** | Lost a head-to-head, KEPT so winner-vs-loser A/B regression works | **Stays** in domain crate behind its feature flag |
| 3 | **DEAD/FAILED** | Gate ran + FAILED, OR explicitly demoted ("🪦"/"demoted"/"FAIL"/"research-only"), OR dead stub, OR off-topic | **Exile** to `katgpt-deprecated` |

Key test: **Did the GOAT gate ever run?**
- NO → category 1 (pending).
- YES + PASS + still opt-in → category 2 (kept for A/B) UNLESS "research-only"/"demoted".
- YES + FAIL / "research-only" / "demoted" / dead / off-topic → category 3 (exile).

---

## Complete membership table

### Category 3 — DEAD/FAILED (exile to `katgpt-deprecated`)

| Feature | Code lives in | Citation | Reason |
|---|---|---|---|
| `feedback` | `src/feedback.rs` | `src/feedback.rs` L19,48 | Dead stub — `log::debug!` only, `let Some(_url)` ignores URL, never HTTP POSTs |
| `unit_distance` | `src/unit_distance/` | Cargo.toml L159; Proposal 003 L130 | Number-theory Erdős research toy (Plan 090); no inference role |
| `alien_sampler` | `src/alien_sampler/` | `.benchmarks/311_alien_sampler_goat.md` L10,19; Proposal 003 L111 | 1/4 PASS (then 2/4), explicitly DEMOTED; cited in Proposal 003 §3 as category-3 exemplar |
| `dense_mesh` | `src/dense_mesh/` | `.benchmarks/266_densemesh_goat.md` L7,21 | Gate 2 FAILED empirically; "modelless hypothesis was empirically falsified"; demoted to experimental |
| `dflare_fusion` | `katgpt-core` | Cargo.toml L228; `.plans/174` L21 | 🪦 IMPROVEMENT GOAT FAILED — no measurable acceptance gain, research-only |
| `dflare_kv_routing` | `katgpt-core` | Cargo.toml L229; `.plans/174` L21 | 🪦 IMPROVEMENT GOAT FAILED — no gain over static, research-only |
| `dflare_progressive_budget` | `katgpt-core` | Cargo.toml L230; `.plans/174` L21 | 🪦 IMPROVEMENT GOAT FAILED — no gain over uniform, research-only |
| `sdpg_bandit` | `katgpt-pruners` | `.benchmarks/011_sdpg_bandit_arena.md`; Cargo.toml L217 | "Verdict: NEGATIVE RESULT"; GOAT gate FAIL (12% vs 29.6% target) |
| `compression_drafter` | `katgpt-core` | `.benchmarks/285_compression_drafter_goat.md` L1,7,98 | "GOAT FAILED (2nd run)" — G1+G2 FAIL vs TernaryDraftModel; "Final demotion... Neither promotes. Honest negative result." |
| `delta_mem` | `katgpt-pruners` | `.plans/053` L599-600, L725-737 | "BOTTOM LINE: value proposition for DDTree is unproven"; 4/6 criteria, latency ❌ 2500% increase, 0% quality gain |
| `rmsd_distill` | `katgpt-pruners` | `.plans/125` L3 | "❌ NO GOAT — negative arena result: RMSD does not improve over SDAR. Demoted to 🪦" |
| `manifold_pruner` | `katgpt-pruners` | Cargo.toml L466; `.plans/234` L3-7 | "GOAT G1 FAIL — demoted, keep opt-in"; Plan 234: "G1 FAIL: no acceptance gain... DEMOTED" |
| `stepcode` | `katgpt-pruners` | Cargo.toml L195 | "⚠️ Plan 054 — NO GAIN proven. Infrastructure only. Off by default, not in `full`." |

**Dead Cargo.toml entries (no code to exile — remove feature line):**

| Feature | Citation | Reason |
|---|---|---|
| `embedding_router` | `src/speculative/step.rs` L1665+ | "not yet started" — only commented-out test stubs, zero active `#[cfg]` |
| `language_domain` | Cargo.toml L301 | "future" placeholder — zero `#[cfg(feature = "language_domain")]` hits in src/ |
| `gpu` | `src/distill/trd.rs` L772 | Gates ONE `#[allow(dead_code)]` TODO stub (`redraft_gpu_batched`); real GPU inference is `gpu_inference` (Metal) |
| `rest` | `src/speculative/prefill.rs` L850 | Single test stub simulating REST target; client moved to riir-ai/riir-rest (Plan 009) |

**Exile count: 13 features with code + 4 dead feature entries = 17 total.**

### Category 2 — BENCH-LOSER (stays in domain crate, kept for A/B)

| Feature | Citation | Reason |
|---|---|---|
| `stokes_calculus` | `.benchmarks/314_stokes_calculus_goat.md` L17-23,118 | G-B PASS (5.36× win), G-C STRUCTURAL FAIL, G-A FAIL — partial winner; 4 primitives correct (15 unit tests); kept for A/B |
| `bake_precision` | `.benchmarks/236_bake_precision_goat.md` L5-6,29 | GOAT 10/10 PASS but G8 marginal (4.7% vs 30% target drift); "Keep opt-in, iterate" — partial winner |
| `opus_selection` | `.benchmarks/040_opus_boltzmann_bandit.md` L77-111 | GOAT proofs P1/P2/P3 all PASS but uses softmax (violates AGENTS.md "use sigmoid not softmax") — kept opt-in, flag softmax |
| `d2f_3sr_warm_start` | `.benchmarks/291_d2f_3sr_warm_start_goat.md` L15,89 | "G1 PARTIAL (0% iteration reduction — honest null result). Stays opt-in Gain, NOT default." Primitive sound; D2F is opt-in research |
| `micro_belief` | `.benchmarks/276_micro_belief_goat.md` L22-23,200 | Trait + `LeakyIntegrator` ship (PASS); `AttractorKernel`/`LatentThoughtKernel` DEMOTED — feature gates whole family, leaky integrator is promotable |
| `bisimulation_operator_inference` | Cargo.toml L475; `.benchmarks/324` | "GOAT G1–G5 all PASS" but "Opt-in by design" — generic primitive, not a default-on capability |
| `tiled_attention` | `.benchmarks/012` L130 | GOAT 8/8 correctness PASS, no gain gate proven (correctness ≠ modelless gain); awaiting gain evidence |
| `mux_latent_wire` | `.plans/243` L282-284 | "Phase 5 GOAT proof ✅ DONE (11/11 tests)" — GOAT passed, kept opt-in |
| `ega_attn` | `.benchmarks/038_ega_attn_goat.md` L49 | GOAT 6/6 PASS but never promoted — A/B candidate |
| `rt_turbo` | `.benchmarks/035_rt_turbo_goat.md` L6 | GOAT 6/6 PASS but never promoted — A/B candidate |
| `spechop` | `.benchmarks/042_spechop_goat.md` L11 | GOAT 6/6 PASS but never promoted — A/B candidate |
| `maxsim` | `.benchmarks/014` L15-37 | T2/T4/T7 PASS but never promoted — A/B candidate |
| `convex_tok` | `.benchmarks/038_convex_tok_goat.md` L59 | GOAT 12/12 PASS but never promoted — A/B candidate |
| `toast_tokenizer` | `.benchmarks/047_toast_renyi_goat.md` L4 | GOAT proofs pass but never promoted — A/B candidate |
| `safe_bandit` | `.benchmarks/036_safe_phased_bandit_goat.md` L84 | GOAT 5/5 PASS but never promoted — A/B candidate |
| `questbench` | `.benchmarks/043_questbench_goat.md` L75 | GOAT PASS (19 tests) but never promoted — A/B candidate |
| `asymmetric_kv` | `.benchmarks/036_asymmetric_kv_goat.md` L6 | GOAT 25/25 PASS but never promoted — A/B candidate |
| `still_kv` | `.benchmarks/245_still_kv_goat_metric_fix.md` L6 | GOAT PASSES legitimately (81/81) but never promoted — A/B candidate |
| `shard_kv` | `.benchmarks/045_shard_kv_goat.md` L63 | "CONDITIONAL — not promoted"; lost cross-method test — A/B keeps it |
| `iso_quant` | `.benchmarks/023_block_diagonal_goat.md` L129 | Loses to OCTOPUS on quality; niche "4D block quality" — A/B keeps it |
| `turboquant` | Cargo.toml L182; `.benchmarks/022` L60 | "legacy baseline for bench/educate only"; "demoted legacy baseline" — A/B keeps it |
| `bandit_mcts` | `.plans/067` L3-5 | "BanditMCTS beats MCTS by +67pp" — GOAT passed, game-specific, kept opt-in |
| `randopt_weight` | `.plans/121` L3-10 | "✅ Complete, 21 GOAT proofs passing" — GOAT passed, kept opt-in |
| `vpd_em_distill` | `.plans/120` L3 | "✅ COMPLETE, GOAT proofs passing" — GOAT passed, kept opt-in |
| `proof_sketch_evolution` | `.plans/128` L3 | "✅ Complete, 46 GOAT tests + 5 arena benchmarks pass" — kept opt-in |
| `committee_boost` | `.plans/132` L3-7 | "✅ Complete (T1–T26) · GOAT 7/7 PASS" — explicitly "Opt-in, requires GOAT proof before default-on" |
| `comp_width` | `.plans/205` L38-43 | "G1/G2/G3 PASS — 3/3 PASS" — GOAT passed, kept opt-in |
| `dynamic_rank` | `.plans/232` L145-155 | "NO GAIN (-0.01pp). Demoted to diagnostic-only tool" — kept for diagnostic value |
| `sdar_gate` | `.plans/072` L26-35 | Arena showed no gain (ELO 954 ≈ Rubric 955); kept as building-block dep for vpd/rmsd/data_gate |

> **Note:** Several category-2 features (ega_attn, rt_turbo, spechop, maxsim, convex_tok, toast_tokenizer, safe_bandit, questbench, asymmetric_kv, still_kv, committee_boost, etc.) are arguably **promotion candidates** — their GOAT gate passed but they were never promoted to `default`. That's a separate cleanup pass, not a loser-sweep concern.

### Category 1 — PENDING (stays in domain crate, GOAT not yet run)

The vast majority of opt-in features (~80) are category 1 — opt-in because the GOAT gate hasn't run yet, or is partially deferred, or the feature is infrastructure awaiting integration. Representative examples:

`ac_prefix`, `action_bridge`, `adaptive_gamma_forecast`, `adaptive_cot_identifiability`, `advantage_freeze_thaw`, `ane`, `async_qdq_overlap`, `auto_constraint_synthesis`, `bckvss`, `band_conditioner`, `cache_prune`, `caddtree_budget` (transitively default), `cgsp`, `cgsp_dual_pool`, `channel_simd_align`, `closure_instrument` (now default), `coexplain_pruner`, `coexplain_riir`, `collider_consistency`, `compression_drafter` [reclassified to cat-3 above], `corr_budget` (default), `cs_kv_probe` (stale comment), `cubical_nerve`, `cubical_topology`, `data_probe`, `decision_trace`, `decode_specialize`, `delta_routing` (default), `dendritic_gate` (default), `dllm`, `domino_lora`, `d2f_3sr_warm_start` [cat-2 above], `echo_env_predictor`, `engram`, `event_log`, `faithfulness_probe`, `federation_composer` (default), `flashar_anchor`, `flashar_consensus` (default), `fol_constraints`, `fol_lnn`, `fourier_continuation`, `fpcg_selector`, `funcattn`, `funcattn_chiar_blend`, `funcattn_compose`, `funcattn_freeze_thaw`, `funcattn_spectral_pre_rotate`, `funcattn_structured_basis`, `future_probe`, `gain_cost_halt`, `game_domain`, `game_state`, `gpart_adapter`, `gpart_pruning`, `gepa_reflective` (default), `g_zero` (default), `hardware_aware_scheduler`, `hoare_pruner`, `hla_eigenbasis_recovery`, `inference_router`, `induced_cwm`, `induced_cwm_ismcts`, `induced_cwm_tournament`, `insight_explain`, `interval_pruner`, `kog_cpu_fusion` (default), `kv_share` (default), `lattice_operad` (default), `latent_trajectory_geometry`, `lclm_adaptive_lod`, `leo_all_goals` (default), `llmexec_guard` (default), `lodestar` (default), `manifold_power_iter_router` (default), `mcts_k_prior`, `bandit_k_prior`, `spec_k_prior`, `memo_reflections` (default), `memory_soup_dtree`, `memory_soup_lora`, `modal_spec`, `moa_inference` (default), `mux_bandit_width`, `mux_bfs`, `mux_ddtree`, `mux_demux`, `mux_freeze_thaw`, `mux_latent_context` (default), `mux_pruner`, `nf_flow`, `nf_flow_budget`, `nf_flow_fold`, `nf_flow_gate`, `nf_flow_mux`, `nf_flow_score`, `newton_schulz` (default), `nds_proxy` (default), `orthogonal_procrustes` (default), `outlier_guard` (default), `paired_loss_diagnostic`, `parallax_attn`, `partial_scoring` (default), `pathway_tracker` (default), `peira_distill` (default), `percepta`, `percepta_compile`, `percepta_gates`, `percepta_graph`, `percepta_wasm`, `percept_route`, `personality_composition` (default), `phrase_boost` (default), `plasma_path` (default), `posterior_evolution` (default), `ppot` (default), `problem_mutator` (default), `product_policy_sharpen`, `proof_cert`, `proof_sketch_evolution` [cat-2 above], `qgf`, `qgf_adaptive`, `qgf_drafter`, `qgf_oracle`, `qgf_projector`, `q_sample_solver`, `recfm`, `regime_transition` (default), `replaid_schedules`, `reward_calibrator` (default), `reward_mem` (default), `rim_slots` (default), `river_valley` (default), `ruliology`, `rv_bandit_pruning` (default), `rv_gated_routing` (default), `rv_gated_thinking` (default), `safe_exploration_budget`, `salience_tri_gate` (default), `schema_centroid` (default), `selectivity_router`, `self_advantage_gate` (default), `self_cond_draft`, `self_distilling_bandit` (default), `sense_composition` (default), `sense_lod` (default), `sigmoid_margin` (default), `skill_lifecycle`, `skill_opt`, `slod` (default), `smear_classifier`, `sleep_consolidation` (default), `spectral_budget`, `spectral_hierarchy` (default), `spectral_pruner` (default), `spectral_rank` (default), `spectral_threat`, `spec_compile`, `spec_cost_model`, `spec_k_prior`, `spec_reconciliation` (default), `specialist_projection`, `speculative_generator` (default), `sr2am_configurator` (default), `ss_pruner` (default), `ssc_spec_draft`, `static_cal_tables` (default), `stepcode` [cat-3 above], `stokes_calculus` [cat-2 above], `subspace_phase_gate`, `substrate_gate` (default), ` sudoku`, `sudoku_cp`, `sudoku_mrv`, `swir_switch_thinking` (default), `symbolic_distill` (default), `targeted_precision` (default), `ted_lite`, `temporal_deriv`, `tes_loop` (default), `tf_loop` (default), `thinking_cot` (default), `thinking_prune` (default), `three_mode_router` (default), `trajectory_doctor`, `triggered_injection` (default), `trust_region_spec` (default), `union_bound_confidence` (default), `unit_distance` [cat-3 above], `vpd_em_distill` [cat-2 above], `vocab_channel_pruner`, `vocab_coreset`, `wall_attention`, `wasm_proof_witness`, `weight_shared_advantage_gate`, `workflow_lattice`, `zone_density_routing` (default)

---

## Phased exile plan

### Phase 3a — exile src/ items (clean, no cross-crate deps)

These 4 src/ items have no consumers outside their own module and can be exiled
immediately to `crates/katgpt-deprecated/`:

- [x] `feedback` — `src/feedback.rs` → `katgpt-deprecated` (zero consumers) — **DONE 2026-07-01**
- [x] `unit_distance` — `src/unit_distance/` + `tests/bench_unit_distance_goat.rs` + `tests/goat_090_tower_search.rs` → `katgpt-deprecated` (tests stay in root, resolve via re-export) — **DONE 2026-07-01**
- [x] `alien_sampler` — `src/alien_sampler/` + `benches/alien_sampler_bench.rs` + `benches/alien_sampler_goat.rs` → `katgpt-deprecated` (benches stay in root, resolve via re-export) — **DONE 2026-07-01**
- [-] `dense_mesh` — `src/dense_mesh/` + `tests/dense_mesh_goat_gates.rs` + `tests/prof_dense_mesh.rs` → **DEFERRED**. `node_transformer.rs` imports `crate::transformer::{forward, ForwardContext, ...}` — transformer-bound glue that can't leave root (same as `gdn2/forward.rs`, `hla/forward.rs` per Proposal 003 §"Stays in src/"). Stays in root as retained glue; exile blocked by the forward-vs-primitive seam.

### Phase 3b — exile katgpt-core items (cross-crate, during Phase 10 absorption)

These losers live in `katgpt-core` and must be exiled when `katgpt-core`
absorption runs (Phase 10). Move the code from `crates/katgpt-core/src/` to
`crates/katgpt-deprecated/src/` and update both Cargo.tomls:

- [ ] `dflare_fusion` — 🪦 GOAT FAILED
- [ ] `dflare_kv_routing` — 🪦 GOAT FAILED
- [ ] `dflare_progressive_budget` — 🪦 GOAT FAILED
- [ ] `compression_drafter` — GOAT FAILED + explicit demotion

### Phase 3c — exile katgpt-pruners items (cross-crate, during Phase 8 absorption)

These losers live in `katgpt-pruners` and must be exiled when the pruners
absorption runs (Phase 8). Move the code from `crates/katgpt-pruners/src/` to
`crates/katgpt-deprecated/src/` and update both Cargo.tomls:

- [ ] `stepcode` — NO GAIN proven
- [ ] `delta_mem` — value proposition unproven
- [ ] `rmsd_distill` — ❌ NO GOAT, demoted
- [ ] `manifold_pruner` — G1 FAIL, demoted
- [ ] `sdpg_bandit` — NEGATIVE RESULT

### Phase 3d — remove dead Cargo.toml entries (no code)

These 4 features have zero active code. Remove the feature line from root
`Cargo.toml` and clean up any commented-out `#[cfg]` stubs:

- [ ] `embedding_router` — remove from `full` + features section
- [ ] `language_domain` — remove from features section
- [ ] `gpu` — remove from `full` + features section; update `src/distill/trd.rs` dead-code stub
- [ ] `rest` — remove from `full` + features section; update `src/speculative/prefill.rs` test stub

---

## Notable stale Cargo.toml comments (fix regardless of exile)

These should be corrected in a separate docs pass — they're not exile decisions,
but the audit surfaced them:

1. **`spectral_threat`** L200 — says "GOAT PASS 46.9%" but Plan 241 says "GOAT gate not yet passed". "46.9%" appears nowhere in benchmarks.
2. **`flow_field_nav`** L98 — same fabricated "46.9%"; Plan 242 shows no GOAT PASS.
3. **`ac_prefix`** L148 — says "Opt-in until G1–G4 pass" but `.benchmarks/313_ac_prefix_modelless.md` says "PROMOTED TO DEFAULT-ON" — not reflected in `default` array.
4. **`rv_gated_thinking`** / **`rv_bandit_pruning`** L393-394 — "default-ON via rv_gated_routing" is misleading (dependency direction is reversed; these are genuinely opt-in despite GOAT passing).
5. **`compression_drafter`** L338 — "Opt-in until GOAT gate passes" but gate already ran and FAILED (`.benchmarks/285`).
6. **`cs_kv_probe`** L470 — says "opt-in until G2 duality gate passes" but Plan 280 says "GOAT gate G1/G2/G3 green". Comment is stale.

---

## Conflict reconciliation notes

The 4 audit agents disagreed on 5 features. Coordinator decisions:

| Feature | Agent A | Agent B | Agent C | Agent D | Final | Rationale |
|---|---|---|---|---|---|---|
| `manifold_pruner` | DEAD-FAILED | — | — | BENCH-LOSER | **DEAD-FAILED** | "GOAT G1 FAIL — demoted" is explicit demotion per cat-3 rule. "keep opt-in" is a workflow note (where to keep), not a category decision. Kernel piece can be extracted separately if needed. |
| `compression_drafter` | — | DEAD-FAILED | — | BENCH-LOSER | **DEAD-FAILED** | "Final demotion... Neither promotes. Honest negative result." is explicit demotion. Primitive correctness (15 tests) doesn't override the gate verdict. |
| `stokes_calculus` | — | AMBIGUOUS | — | BENCH-LOSER | **BENCH-LOSER** | G-B genuinely won (5.36×), primitives correct (15 tests, Stokes identities hold), not "demoted"/"research-only". Lost head-to-head on G-A but kept for the G-B win = classic A/B loser. |
| `opus_selection` | AMBIGUOUS | — | — | — | **BENCH-LOSER** | GOAT passed; softmax violation is a code-style flag, not a gate failure. |
| `cs_kv_probe` | AMBIGUOUS | — | — | — | **PENDING** | Stale Cargo.toml comment; Plan 280 says gates green. No actual failure. |
| `product_policy_sharpen` | — | DEAD-FAILED (dead stub) | — | — | **PENDING** | Agent B was wrong — `#[cfg(feature = "product_policy_sharpen")]` DOES have real code in `katgpt-pruners/src/self_advantage.rs` (struct + impl + 3 tests). Not a dead stub; part of Plan 283 self_advantage family. |
| `proof_cert` | — | — | — | DEAD-FAILED (off-topic) | **PENDING** | Agent D was wrong — `src/proof_cert/` has 7 real files (certificate, chain, macros, serde, wasm). It's GOAT-gate verification infrastructure, not a failed/demoted primitive. Destination crate ambiguous (katgpt-validator or katgpt-bench). |

---

## TL;DR

**17 exile candidates** identified (13 with code + 4 dead Cargo.toml entries):

**With code (move to `katgpt-deprecated`):** `feedback`, `unit_distance`, `alien_sampler`, `dense_mesh`, `dflare_fusion`, `dflare_kv_routing`, `dflare_progressive_budget`, `sdpg_bandit`, `compression_drafter`, `delta_mem`, `rmsd_distill`, `manifold_pruner`, `stepcode`

**Dead entries (remove from Cargo.toml):** `embedding_router`, `language_domain`, `gpu`, `rest`

Of the 13 with code: 4 live in `src/` (Phase 3a), 4 live in `katgpt-core` (Phase 3b), 5 live in `katgpt-pruners` (Phase 3c).

### Phase 0.5 + 3a execution status (2026-07-01)

- **Phase 0.5 (audit): DONE.** This document.
- **Phase 3a (src/ exile): 3 of 4 done.** `feedback`, `unit_distance`, `alien_sampler` exiled to `katgpt-deprecated` with back-compat re-exports. `dense_mesh` deferred (transformer-bound glue). All workspace tests pass (122 in deprecated crate, 5266+ in workspace).
- **Phase 3b/3c (cross-crate exile): deferred** to Phases 8/10 absorption.
- **Phase 3d (dead Cargo.toml entries): deferred** — low-risk cleanup, not blocking.

The remaining ~80 opt-in features are either PENDING (GOAT not yet run) or BENCH-LOSER (passed but kept opt-in for A/B). Exiling them would destroy active WIP.
