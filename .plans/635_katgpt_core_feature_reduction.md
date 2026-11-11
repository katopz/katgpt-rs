# Plan 635: katgpt-core feature reduction to ≤300 for crates.io — classification (CUT = 0, ROAD 1 as scoped is infeasible)

**Status:** TRANCHE 1 LANDED 2026-11-10 — katgpt-core 414 → **278** features (BAKE 74 + GROUP 62), katgpt-rs `8c3349bfb`; sibling spelling rewrites pushed (riir-ai `b7431c505`, riir-train `66bad5a7`, riir-neuron-db `84c9d20`, riir-infer `63e48b98`, riir-reflex `c41aabb`, riir-dao `b2f8d88`, riir-game-sdk `9280bce`, seal-remake `9337a4b4`, seal-online-remaster `c26e2ba`, riir-instinct `1d9efff`, riir-refine `c86d8603`). Open: riir-chain/riir-dapps at the develop→main merge; Tranche 2 (headroom).
**Lease closed:** katgpt-rs-85@m3, 1794304147, expires 2026-11-12T09:49:07Z — Tranche 1 landed (`8c3349bfb`); Tranche 2 unclaimed.

**Date:** 2026-11-10
**Issue:** [`.issues/992_cratesio_feature_limit_blocks_katgpt_core_publish.md`](../.issues/992_cratesio_feature_limit_blocks_katgpt_core_publish.md) (decided 2026-11-10: ROAD 2 — limit-increase email; the ROAD 1 premise was falsified by this plan's corrected census — THIS plan is the fallback map only)
**Crate:** `crates/katgpt-core` 0.5.0 (registry refuses: 414 features > 300 cap)
**Classification session:** zed-352d06e6-m3 (this box), reads+greps only, tree at `origin/develop` `3963a7d39`
**Numbering gate:** `dual_allocation_gate.py` run at allocation time — 0 twin / 0 independent / 0 counter.

## Verdict (read this first)

ROAD 1 as recorded ("cut = never spelled AND no gain/GOAT record AND not a default member AND not transitively implied") yields **|CUT| = 0**. Every one of the 19 never-spelled non-default features is gain/GOAT-documented (README / `.docs/` / `.benchmarks/` / `HISTORY.md` hits — tiered table below), and the 5 that would additionally be implied-by-kept are likewise documented. The shortfall is NOT caused by over-eager KEEP-EXCEPTIONS alone — it is caused by the spelling census being far larger than Issue 992's: **391 of 414 rows are spelled somewhere** (internal manifests + core's own 376 `required-features` rows + downstream dep tables and forwards), leaving only 19 candidates before the gain screen. **414 − 19 = 395 > 300 even with an owner override cutting every documented candidate.** Owner decision required (options below); no execution task in this plan is unblocked until then.

## Re-derived census (2026-11-10, exact TOML parse + multi-lane grep)

| lane | distinct features | detail |
|---|---|---|
| parsed `[features]` rows | **414** | matches crates.io count exactly (hand-rolled parser; multi-line rows + comments handled; 414 key lines confirmed independently by grep) |
| `default = [...]` members | **83** | KEEP-IMMUTABLE |
| internal spellings | **352** | fwd `katgpt-core/<x>` 204 across 15 manifests · dep-table `features = [...]` 3 (`subspace_intervention` in katgpt-attn dev-dep, `subspace_phase_gate` in katgpt-canon, `rating` in katgpt-pruners) · core's own `[[bench]]/[[test]]/[[example]]` `required-features` **214 distinct across 376 rows** · CLI `--features decision_wire` (`scripts/test_gate.sh:155`) |
| downstream spellings | **250** across **13 repos** | fwd 239 / dep-table 56 (overlap); per-repo table below |
| spelled union (internal ∪ downstream) | **391** | |
| kept after closure (default ∪ spelled ∪ transitively implied) | **399** | |
| candidates0 (never spelled anywhere, not default members) | **19** | 14 fully unimplied (gain-screen pool) + 5 implied-by-kept (prune children) |
| **CUT** | **0** | all 19 have gain/GOAT documentation hits |
| resulting count | **414** | required ≤ 300 → **shortfall 114** |

Downstream per-repo (fwd / dep-table / union) vs Issue 992's numbers:

| repo | mine (fwd/tbl/union) | issue 992 | delta |
|---|---|---|---|
| riir-ai | 164/45/**182** | 173 | −9 fwd, +45 tbl lane |
| riir-train | 39/3/**40** | 40 | ✓ union |
| riir-neuron-db | 32/1/**33** | 35 | −2 fwd |
| riir-infer | 35/0/**35** | 33 | +2 fwd |
| riir-refine | 19/6/**25** | 19 | +6 tbl lane |
| riir-reflex | 20/0/**20** | 19 | +1 fwd |
| riir-instinct | 14/4/**18** | 13 | +1 fwd, +4 tbl |
| riir-chain | 7/0/**7** | 8 | −1 fwd |
| riir-game-sdk | 4/0/**4** | 4 | ✓ |
| riir-dapps | 1/1/**2** | 1 | +1 tbl |
| riir-dao | 1/2/**3** | *(absent)* | **repo missing from issue's table** |
| riir-reflexer | 0/1/**1** | *(absent)* | **repo missing** |
| seal-online-remaster | 0/1/**1** | *(absent)* | **repo missing** |

Divergence explanation: Issue 992's census grepped only `katgpt-core/<feature>` forward spellings (union 234 ≈ my fwd-only 239, small drift). My census additionally counts **dep-table `features = [...]` entries** (56 features) — those break identically on row removal (cargo hard-error) — and **core's own required-features rows** (214), plus the CLI lane. My numbers are the conservative/safe set for cutting.

## Owner decision required (pick one; execution plan differs per road)

- **ROAD 2 (recommended now): email `help@crates.io`** for a feature-limit increase — the only zero-cut road; zero churn; the crate genuinely has a 414-deep feature matrix.
- **ROAD 1′ (override tranche):** accept cutting documented-but-never-spelled features (the 19 below, ranked most-stale-first) → 395 rows; **still 95 over the cap** → must continue into a breaking spelling-collapse campaign (downstream fwd/dep-table rewrites + internal required-features scrubs + default-array edits) — a multi-repo breaking release, 0.6.0, far beyond this plan's original scope.
- **LEVER F — flagless promotion of the default closure (owner idea, measured 2026-11-10):** make every GOAT-default feature unconditional (drop the row + its `#[cfg]`s). Measured on `crates/katgpt-core/Cargo.toml`: default array 83 members, transitive default closure **96**; 21 of those activate an optional dep (`dep:`/`crate/feat`) → flagless would make the dep non-optional, so only **75 are pure**. Arithmetic: 414 − 75 = **339 (still 39 over)**; with the 19 ROAD 1′ cuts, 320 (still 20 over) — **not sufficient alone**. Cost: **73 of the 75 are spelled in some Cargo.toml** (root forwards, `required-features`, sibling dep tables — cargo hard-errors on a removed feature name), 487 `#[cfg]` sites in katgpt-rs `*.rs`, and **13 consumers opt OUT with `default-features = false`** (wasm: `katgpt-moka-wasm`, `riir-core-wasm`, `riir-wasm`; lean: riir-reflex, riir-rethink, riir-instinct, riir-refine, riir-dao, riir-chaind, …) — flagless forces all 75 modules into those builds, so each needs a wasm32 + compile-time check before it loses its flag. Verdict: right DIRECTION (a GOAT default is pillar, a flag on it is pure tax), but it is the same multi-repo breaking spelling-collapse class as ROAD 1′ and does not reach 300 on its own → it does NOT replace ROAD 2; run it as the first tranche IF ROAD 1′ is ordered, ranked: pure ∧ wasm32-clean ∧ fewest spellers first.
- **ROAD 3 (hybrid, Issue 992):** cut only mechanically-free rows — but note the 8 empty rows among the candidates are all documented too; there is no documentation-free alias pool left under this rule set.

## Execution tasks — ⛔ ALL BLOCKED on the owner decision above

- [x] Road decided (2026-11-10): ROAD 2 — crates.io limit-increase email (verdict + email draft in Issue 992); ROAD 1′ = THIS plan, fallback only, never default
- [x] **Growth guard LANDED 2026-11-11** (`a1bd4ad70`; PASS line names the tightest crate + headroom since the follow-up commit; `scripts/feature_cap_gate.py`, a docs_gate CHECK on every push + a pre-flight in both `scripts/release.sh` entry points; katgpt-core 278/300 reads WARN, 22 headroom; a 301-row copy FAILS exit 1; `--manifest PATH` for any rendered publish manifest — riir-refine's `cratesio_publish.sh` lane can call it, not wired there while riir-refine Issue 221 is under another session's lease) — was: Growth guard: feature-count ceiling check in the publish lane — warn as the count approaches the granted limit, hard-fail BEFORE the registry refuses; wire into the kat-secret publish-cascade script lane (Issue 992 verdict round 2)
- [x] LEVER F tranche — executed as Tranche 1 (below): per pure default-closure feature — wasm32 build check under `default-features = false`, rewrite its spellers (root fwd, `required-features`, sibling dep tables), drop row + `#[cfg]` sites; 75 candidates, 73 spelled (2026-11-10 census, re-derive at execution)
- [ ] Manifest row cuts: delete each CUT row from `crates/katgpt-core/Cargo.toml` `[features]` (per final CUT table — currently EMPTY)
- [ ] Implication scrubs: edit every surviving parent row's value list that names a cut child (currently EMPTY; tranche parents listed in the PRUNE table below)
- [ ] `required-features` scrubs: any `[[bench]]/[[test]]/[[example]]` row naming a cut feature (currently EMPTY — none of the 19 candidates appears in any of the 376 required-features rows; re-verify at execution time)
- [ ] Code removal for code-affected cuts: every `#[cfg(feature = "NAME")]` / `cfg!(...)` site in `crates/katgpt-core/src/` (EMPTY for CUT; per-feature files listed below for the tranche)
- [ ] Version bump `0.5.0` → `0.6.0` (feature REMOVAL is breaking) + the 8 published sibling crates' dep-version coordination per Issue 992's note
- [ ] README.md + `examples/README.md` feature-count/flag-list updates (`scripts/count_features.py` gate row)
- [ ] Gates: `cargo check` · `cargo check --all-features` · `cargo test -p katgpt-core --lib` · `cargo clippy --all-targets` · `./scripts/docs_gate.sh`
- [ ] Downstream zero-cut-spelling audit: re-grep all 13 downstream repos + katgpt-rs internal lanes for every cut name (fwd + dep-table + CLI + required-features lanes; census snapshot is 2026-11-10 — siblings move daily, re-derive at execution)
- [ ] Publish cascade re-run per Issue 992 (kat-secret Touch-ID lane; core 0.6.0 must land so riir-refine's crates.io lane unblocks)

## Tranche 1 — EXECUTED 2026-11-10 (BAKE + GROUP → 414 → 278)

Owner order 2026-11-10: "we won't mail and wait forever — find more features to bake or group". Classifier + rewriters: per-`cfg`-site AST classification (ITEM/FILE = additive, BODY/MACRO/ATTR = behavioural, `not(...)` sites tracked), predicate substitution + simplification (baked → `true`, grouped → parent), zero FALSE-predicate results by construction (a negated feature is never baked).

- **BAKE (74):** default-closure members with no optional-dep activation (`dep:` / optional `crate/feat`) and no `not(feature)` site, closed under implication. Rows deleted, `#[cfg]`s removed or simplified; their `katgpt-types/*` sub-features hoisted onto the `katgpt-types` dep (`domain_latent`, `maxsim`, `plasma_path`, `sigmoid_margin`, `sparse_mlp`). Default members that activate an optional dep (papaya, ahash, rustfft, katgpt-dec/-sense/-micro-belief/-personality, unicode-normalization) KEEP their flag — lean consumers do not gain deps.
- **GROUP (62):** non-default, additive-only (every src site an item/module/file gate), no dep activation, exactly ONE implied parent → folded into that parent (the child's code now compiles whenever the parent is on). 23 of them chain into a baked parent (effectively baked, behaviour-neutral: additive code only); 39 fold into a still-opt-in parent (opt-in discipline kept at the parent).
- Rewritten: 197 core `*.rs` (442 attributes removed, 159 simplified), 76 `--features` run-instructions, core manifest (136 rows; required-features rows), 11 katgpt-rs dependent manifests (root forwards of baked names become `[]` no-op forwards, grouped → parent), README/examples counts, 4 root comments, 2 bench labels.
- **Verified:** amo all-features workspace clippy rc=0 with the warning set IDENTICAL to `origin/develop` (25 = 25 messages); `-p katgpt-core` clippy no-default-features + all-targets rc=0; workspace default clippy rc=0; `cargo test -p katgpt-core --lib` 2487 passed / 0 failed; wasm32 `--no-default-features` check clean; docs_gate count/comment/bench-label audits clean.
- **Siblings:** path-dep spellers rewritten by the same rewriter (riir-ai, riir-train, riir-neuron-db, riir-infer, riir-refine, riir-instinct, riir-reflex, riir-dao, riir-game-sdk, seal-remake, seal-online-remaster). **Deferred:** riir-chain + riir-dapps pin `branch = "main"` (git dep) — rewrite when develop merges to main; `sprint005-vessel-cache` (feature-branch clone of seal-remake, not ours).

Baked: `ac_prefix`, `action_bridge`, `ane_fused_chain`, `ane_roofline`, `best_belief`, `bmr`, `causal_identification`, `cce_moderator`, `clr_weighted_set_attention`, `conformal_predictive_intervals`, `contrastive_scope`, `cross_resolution_transport`, `domain_latent`, `dual_gram_pca`, `dual_leo`, `event_state_windows`, `fitted_anchor_tables`, `fitted_v_reconstruct`, `fitted_value_tables`, `fourier_continuation`, `funcattn`, `funcattn_structured_basis`, `geometric_product`, `grapem_rodrigues`, `heal_validation`, `hebbian_kernel_memory`, `hmm_homeostasis`, `hope_capacity`, `indicator_probe_bank`, `indicator_similarity`, `karc_forecaster`, `kinematic_rollout`, `latent_field_steering`, `leo_all_goals`, `linking_fold_fold`, `llmexec_guard`, `local_branch_routing`, `mag_mining`, `manifold_bandit`, `manifold_erasure`, `maxsim`, `mean_field_regime`, `non_interference_branches`, `phase_rotation_coupling`, `phase_separation`, `plasma_path`, `poincare_navigator`, `position_group_action`, `product_key_memory`, `qmc_sampling`, `rating`, `region_subspace_steering`, `renoise_ce`, `salience_tri_gate`, `set_attention`, `sigmoid_margin`, `simd_lut_dequant`, `slice_tca`, `slt`, `slt_sweep`, `smooth_min_similarity`, `sparse_mlp`, `spectral_hierarchy`, `spherical_steering`, `ssd_block`, `ssmax_temperature`, `subspace_phase_gate`, `subspace_steering`, `temp_loss_fingerprint`, `tilr_invariant_subspace`, `triggered_injection`, `tucker_factorization`, `velocity_field_ensemble`, `viable_manifold_graph`

Grouped (child → parent): `adaptive_causal_calibration`→`causal_head_importance`, `bridge_certified`→`closure_instrument`, `cgsp_dual_pool`→`cgsp`, `closed_form_weight_write`→`quant_error_lora`, `cond_audit`→`stale_residual`, `cross_stage_relocation`→`causal_head_importance`, `differential_anchor`→`fitted_anchor_tables`, `differential_kv_eviction`→`kv_sink_window`, `dying`→`usage_rate_eviction`, `effective_degree`→`karc_forecaster`, `engram_privilege`→`engram`, `escalation_probe_gate`→`state_probe`, `freedom_selection`→`renoise_ce`, `gmm_support`→`factorized_action`, `grape_joint_lift`→`grapem_rodrigues`, `hint_regret`→`best_belief`, `idea_divergence`→`partial_scoring`, `imbalance_monitor`→`knn_entropy`, `indicator_cascade`→`indicator_probe_bank`, `induced_cwm_ismcts`→`induced_cwm`, `induced_cwm_tournament`→`induced_cwm`, `karc_batched_matvec`→`karc_forecaster`, `karc_hebbian_readout`→`hebbian_kernel_memory`, `karc_lod_tier`→`karc_forecaster`, `kron_chunk_wy`→`kron_memory`, `kron_tile`→`meld`, `kv_sink_window`→`usage_rate_eviction`, `lattice_memory_3d`→`lattice_memory`, `lclm_adaptive_lod`→`mux_latent_context`, `loop_alignment_probe`→`margin_gate`, `margin_gate`→`indicator_cascade`, `mi_est`→`gaussianity_probe`, `mop_path_entropy`→`cgsp`, `mux_bfs`→`mux_ddtree`, `mux_ddtree`→`mux_pruner`, `mux_freeze_thaw`→`mux_pruner`, `mux_height_gate`→`mux_bfs`, `mux_latent_wire`→`mux_latent_context`, `numeric_stability`→`mag_mining`, `operator_consistency`→`bisimulation_operator_inference`, `orthogonal_factorization`→`spectral_pencil`, `product_key_memory_episodic`→`product_key_memory_freeze`, `product_key_memory_freeze`→`product_key_memory`, `qgf_adaptive`→`qgf_drafter`, `qgf_drafter`→`qgf`, `qgf_projector`→`speculative_generator`, `renoise_ce_surprise`→`renoise_ce`, `rotary_value_embedding`→`position_group_action`, `rtdc_subtree_inclusion`→`rtdc`, `saddle_escape`→`gain_cost_halt`, `set_refine`→`set_attention`, `slod`→`spectral_hierarchy`, `ssmax_adaptive`→`ssmax_temperature`, `state_option_scoring`→`distance_abstain`, `state_probe`→`hint_regret`, `structural_cot_halt`→`ict_branching`, `subspace_intervention`→`subspace_phase_gate`, `support_regime`→`functional_substitution_gate`, `tf_loop`→`lt2_looped`, `variable_rank_domain_expert`→`committed_field_blend`, `velocity_field_disagreement`→`velocity_field_ensemble`, `weight_edit_overlay`→`gain_cost_halt`

- [x] Landed: katgpt-rs `8c3349bfb` + 11 sibling commits (status line); docs_gate 36/36 on the real checkout; test_gate PASS; consumers clean (amo `~/scratch/cws`)
- [ ] riir-chain + riir-dapps (`branch = "main"` git deps): run `rewrite_toml.py`-equivalent spelling rewrite at the develop→main merge (riir-dapps spells `best_belief` — baked)
- [x] **katgpt-core v0.5.0 PUBLISHED to crates.io 2026-11-11** (owner Touch-ID, `p1-cascade` token, from a clean detached worktree at `c116555ad`; dry-run `9b9c79a15` + `c22ac844e`; 278 features, 5.3 MiB compressed). Issue 992 CLOSED + removed (recovery: `git log --diff-filter=D -- .issues/992_*`); riir-refine Issue 221 P1 9/9 (`11e3a5e7`)

## Tranche 2 — NEXT (headroom): umbrella families + sub-crate extraction

- [ ] Umbrella-group the ~156 standalone additive non-default features into ~5 family flags (attention/KV, memory/retrieval, sampling/search, geometry/linalg, probes/gates) → ~278 → ~130; behavioural (BODY-site) features keep their own flags
- [ ] Sub-crate extraction of the top-10 families (each crate gets its own 300 budget; core exposes ONE umbrella per sub-crate) — separate plan; leaf-most families first (`linalg` coupling is the main blocker)

## Table A — CUT rows by class: EMPTY (|CUT| = 0)

No row satisfies all four cut conditions (never spelled · not default · not implied-by-kept · no gain/GOAT hit). The class buckets (row-only / code-affected / pure forward / dep-activator) are therefore all empty; the tranche classification of the 19 near-miss candidates is in Table B/E instead. Note: none of the 19 is a `dep:`-activator row (that class is empty even at tranche level).

## Table B — KEEP-EXCEPTIONS (19/19 candidates documented, ranked most-stale-first ≈ tranche order)

Evidence tiers: **BENCH** = a `.benchmarks/` GOAT/measurement record names it · **DOC** = README/`.docs`/`HISTORY` gain record · **CATALOG** = `.docs/09_feature_catalog/opt_in_features.md` row only (the catalog lists every feature by design — weakest tier). Rank is by earliest evidence anchor (approximate; bench/issue numbers are roughly chronological).

| # | feature | tier | class | citation (earliest/best) | cfg sites |
|---|---|---|---|---|---|
| 1 | `simd_sigmoid` | DOC | prune-child (implied by default member `bom_sampling`) | README.md:1367 (G1.3 **1.87×** SIMD PASS, Issues 024/025); `crates/katgpt-core/README.md:41`; manifest row `bom_sampling` (auto-enables, G3 PASS K=8 1.87×, Issue 025) | 0 (row-only) |
| 2 | `lt2_looped` | BENCH | prune-child (implied by kept `tf_loop`) | `.benchmarks/033_lt2_looped_goat.md` (bench 33; Plan 108 GOAT 8/8); README.md:343/403 | 0 (row-only) |
| 3 | `substrate_gate` | CATALOG | pool | `.docs/01_orientation/overview.md:680` (maps `katgpt-core/substrate_gate`); catalog row. ⚠ README.md:455/1094 + `substrate_gate_goat` bench document the ROOT-crate same-name feature (root default line 195), NOT this core row — core row itself has no direct bench | 0 (row-only) |
| 4 | `mux_pruner` | CATALOG | pool (mux family root) | `.docs/09_feature_catalog/opt_in_features.md:1775`; `.docs/01_orientation/overview.md:74` (MUX superposition tree search) | lib.rs, mux/mod.rs |
| 5 | `mux_ddtree` | CATALOG | pool | catalog:1776 (implies mux_pruner) | mux/mod.rs |
| 6 | `mux_freeze_thaw` | CATALOG | pool | catalog:1779 (implies mux_pruner) | mux/mod.rs |
| 7 | `mux_bandit_width` | CATALOG | pool | catalog:1777; overview.md:143 | mux/mod.rs |
| 8 | `chunked_net_fetch` | DOC (stub) | pool | catalog:1960 (Plan 272 T3.3, "NetChunkFetcher — network chunk fetcher stub"); `.docs/03_memory/chunked_content_store.md:56` | content_store/fetcher.rs, content_store/mod.rs |
| 9 | `mux_bfs` | BENCH (weak) | pool | `.benchmarks/202_comp_width_goat.md:61` (GOAT command line `--features "mux_bfs comp_width"`); catalog:1778 | mux/mod.rs |
| 10 | `product_key_memory_freeze` | BENCH | prune-child (implied by kept `product_key_memory_episodic`) | `.benchmarks/408_pkm_episodic_fusion.md:6` (Plan 408 Phase 4); `.docs/03_memory/product_key_memory.md:86` | lib.rs, product_key_memory/mod.rs |
| 11 | `cochain_point_sampler` | BENCH | pool | `.benchmarks/422_cochain_point_sampler_goat.md:78` ("Stays opt-in — substrate-completeness"); README.md:2938; manifest row (Plan 422) | 0 (row-only) |
| 12 | `position_group_action` | BENCH | prune-child (implied by default member `fitted_v_reconstruct` + kept `rotary_value_embedding`) | `.benchmarks/458_position_group_action_goat.md`; README.md:2946; `.docs/09_feature_catalog/negative_results.md:244` (7/7 GOAT PASS, application-axis failure) | lib.rs (2 sites) |
| 13 | `grape_ap_vector` | BENCH | pool | `.benchmarks/459_grape_ap_vector_goat.md:4`; README.md:2947; overview.md:736 | lib.rs |
| 14 | `grape_joint_lift` | BENCH | pool | `.benchmarks/460_grape_joint_lift_goat.md:4` (implies `grapem_rodrigues`); README.md:2948 | lib.rs |
| 15 | `karc_householder_eig` | DOC | prune-child (implied by kept `karc_householder_eig_par`) | README.md:2910 (Issue 186 Path B, 2026-07-20); `.docs/09_feature_catalog/opt_in_features.md:704` (Issues 186+187); no bench FILE found | karc/large_dh.rs (2 sites) |
| 16 | `ternary_trit_pack` | BENCH | pool | `.benchmarks/582_trit_pack_goat.md:6`; `.benchmarks/846_bitcos_goat.md:6`; `.benchmarks/586_avx2_ternary_t4_measurements.md:147`; manifest row (Issue 582, 1.75 bits/weight, −17.6% footprint) | lib.rs |
| 17 | `laprop` | BENCH | pool | `.benchmarks/678_laprop_normalized_momentum_goat.md:8`; manifest row (Issue 689, 2026-08-25) | lib.rs |
| 18 | `mux_height_gate` | BENCH | pool | `.benchmarks/679_mux_height_gate_goat.md:11`; manifest row (Issue 688, Coconut §4.4 port) | mux/mod.rs |
| 19 | `dirichlet_dist` | BENCH | pool | `.benchmarks/904_dirichlet_dist_goat.md:7`; HISTORY.md:182 (Bench 904, thinning 183 ns @ N=8); manifest row (Issue 912, Research 596) | dirichlet_dist.rs, lib.rs |

**KEEP-EXCEPTION count: 19** (all of `candidates0`). If the owner overrides the gain screen, the CATALOG-tier rows (#3–#8, six features) are the weakest evidence and the honest first tranche; cutting all 19 still leaves 395 > 300 (see Verdict).

## Table C — PRUNE-implication list (default posture: KEEP the child; every pair flagged for review, nothing decided)

| kept parent row | would-be-cut child | parent's own status | review note |
|---|---|---|---|
| `bom_sampling` | `simd_sigmoid` | **DEFAULT member** (default array) + spelled (CLI-adjacent lane; bench docs) | cutting the child edits the DEFAULT array → behavior change for every default build (bom_sampling auto-enables simd_sigmoid per Issue 025 G3). Highest-risk pair. |
| `fitted_v_reconstruct` | `position_group_action` | **DEFAULT member** + spelled | same default-array-edit class as above. |
| `rotary_value_embedding` | `position_group_action` | spelled (kept) | second parent of the same child; both parents must drop it in one commit. |
| `tf_loop` | `lt2_looped` | spelled (kept) | child is a row-only alias (0 cfg sites) — cheapest prune on the board. |
| `product_key_memory_episodic` | `product_key_memory_freeze` | spelled (kept via bench-408-era required-features lane) | child has 3 cfg sites (lib.rs:1273, product_key_memory/mod.rs:111,113). |
| `karc_householder_eig_par` | `karc_householder_eig` | spelled (kept) | child is the Issue-186 eigensolver path; parent forwards to it semantically (large_dh.rs cfg both ways incl. `not(feature)` arms — scrubbing needs BOTH arm directions). |

All 5 children are ALSO KEEP-EXCEPTIONS (Table B) — pruning any of them additionally overrides documentation.

## Table D — Manifest scrub sites (if any tranche feature is later cut)

None of the 19 appears in any `[[bench]]/[[test]]/[[example]]` `required-features` row (verified against the raw manifest AND the parsed row set) and none is CLI-spelled in scripts/workflows. The only in-manifest references are feature-row value lists (Table C) plus each row's own line:

- `karc_householder_eig` ← row `karc_householder_eig_par`
- `lt2_looped` ← row `tf_loop`
- `mux_bfs` ← row `mux_height_gate` · `mux_ddtree` ← row `mux_bfs` · `mux_pruner` ← rows `mux_ddtree`, `mux_freeze_thaw` (mux family cuts cleanly only as a whole — cut all six together)
- `position_group_action` ← rows `fitted_v_reconstruct`, `rotary_value_embedding`
- `product_key_memory_freeze` ← row `product_key_memory_episodic`
- `simd_sigmoid` ← row `bom_sampling` (DEFAULT member row)

The 12 other candidates have **zero** in-manifest references outside their own row (rows `chunked_net_fetch`, `cochain_point_sampler`, `dirichlet_dist`, `grape_ap_vector`, `grape_joint_lift`, `laprop`, `mux_bandit_width`, `mux_height_gate`, `substrate_gate`, `ternary_trit_pack` are self-contained).

## Table E — cfg-site files in `crates/katgpt-core/src/` (tranche; EMPTY for CUT)

| feature(s) | files |
|---|---|
| `mux_pruner` | `src/lib.rs`, `src/mux/mod.rs` |
| `mux_bfs`, `mux_ddtree`, `mux_freeze_thaw`, `mux_bandit_width`, `mux_height_gate` | `src/mux/mod.rs` (one file — the whole mux family is one module) |
| `chunked_net_fetch` | `src/content_store/fetcher.rs`, `src/content_store/mod.rs` |
| `dirichlet_dist` | `src/dirichlet_dist.rs`, `src/lib.rs` |
| `grape_ap_vector`, `grape_joint_lift`, `laprop`, `position_group_action`, `ternary_trit_pack` | `src/lib.rs` |
| `karc_householder_eig` | `src/karc/large_dh.rs` (incl. a `not(feature = ...)` arm — scrub both directions) |
| `product_key_memory_freeze` | `src/lib.rs`, `src/product_key_memory/mod.rs` |
| *(row-only, 0 cfg sites)* | `cochain_point_sampler`, `lt2_looped`, `simd_sigmoid`, `substrate_gate` |

Totals: 15 code-affected candidates across **8 distinct files**; 4 row-only. (Execution-phase edits — none done in this session.)

## Risk flags for the orchestrator

1. **ROAD 1 infeasible as scoped** — |CUT| = 0; shortfall 114; owner override on all 19 still leaves 395. Decision cannot be executed by cutting alone; it needs ROAD 2 (limit increase) or a breaking spelling-collapse campaign (ROAD 1′+).
2. **Downstream census divergence** — 250 distinct over 13 repos vs Issue 992's 234 over 10; three repos (`riir-dao`, `riir-reflexer`, `seal-online-remaster`) were absent from the issue's table and DO spell core features (dep-table lane). Any future cut audit must use the wider lane set (fwd + dep-table + CLI + required-features), not fwd-only.
3. **Core's own required-features wall** — 376 rows pin 214 distinct features; cargo hard-errors on any unknown name there. The current candidates happen to have zero such rows, but any larger tranche must re-run that scan first.
4. **Gain screen as specified is nearly all-keeping** — `.docs/09_feature_catalog/opt_in_features.md` lists every feature by design, so "any `.docs` hit = KEEP-EXCEPTION" keeps ~everything catalogued. I complied with the rule and tiered the evidence (Table B); only the owner can decide CATALOG-tier documentation is demotable.
5. **Empty-row census divergence (cosmetic)** — Issue 992 counted 79 empty rows; exact parse says 232 (its class buckets overlap differently). The 414 total agrees exactly.
6. **Census is a snapshot** — sibling repos move daily; the downstream zero-cut-spelling audit task must re-derive at execution time.
7. **`substrate_gate` name collision** — the documented default-ON `substrate_gate` (README:1094, `substrate_gate_goat` bench) is the ROOT-crate feature (root default line 195); the core row of the same name is a distinct, never-spelled alias. Same pattern for `lt2_looped` (root feature, root default) vs the core alias. Per-crate namespaces were kept separate throughout.

## Summary

**(1) original task** — classify katgpt-core's 414 features per Issue 992's ROAD 1 cut rule and file the execution plan as `.plans/635`.
**(2) accomplished** — exact 414-row parse; 4-lane census (internal 352 · downstream 250 over 13 repos · union 391); closure kept 399; candidates0 19 (14 pool + 5 prune children); gain screen 19/19 documented → **CUT = 0, shortfall 114**; divergence vs Issue 992 quantified (3 unlisted repos + dep-table lane); `dual_allocation_gate` clean; this doc written; `.plans/.highwater` 634 → 635. No Cargo.toml/.rs edits, no builds, no commits.
**(3) what remains** — owner decision (ROAD 2 recommended / ROAD 1′ override tranche); all execution checkboxes above stay BLOCKED until then; tranche tables B/C/D/E are execution-ready if ROAD 1′ is chosen.
**(4) active plan state** — `.plans/635_katgpt_core_feature_reduction.md` OPEN; `.issues/992` still OPEN (this plan is its classification child).
