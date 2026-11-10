# Plan 635: katgpt-core feature reduction to ≤300 for crates.io — classification (CUT = 0, ROAD 1 as scoped is infeasible)

**Status:** FALLBACK MAP ONLY (ROAD 1′) — never the default road. DECIDED 2026-11-10: ROAD 2 (crates.io limit-increase email) per the delegated-owner verdict on Issue 992 (round 1 REVISE → round 2 AGREE); this plan's execution stays BLOCKED unless the owner orders the ROAD 1′ campaign or crates.io counters with a specific partial limit. Growth-guard task added.

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
- **ROAD 3 (hybrid, Issue 992):** cut only mechanically-free rows — but note the 8 empty rows among the candidates are all documented too; there is no documentation-free alias pool left under this rule set.

## Execution tasks — ⛔ ALL BLOCKED on the owner decision above

- [x] Road decided (2026-11-10): ROAD 2 — crates.io limit-increase email (verdict + email draft in Issue 992); ROAD 1′ = THIS plan, fallback only, never default
- [ ] Growth guard: feature-count ceiling check in the publish lane — warn as the count approaches the granted limit, hard-fail BEFORE the registry refuses; wire into the kat-secret publish-cascade script lane (Issue 992 verdict round 2)
- [ ] Manifest row cuts: delete each CUT row from `crates/katgpt-core/Cargo.toml` `[features]` (per final CUT table — currently EMPTY)
- [ ] Implication scrubs: edit every surviving parent row's value list that names a cut child (currently EMPTY; tranche parents listed in the PRUNE table below)
- [ ] `required-features` scrubs: any `[[bench]]/[[test]]/[[example]]` row naming a cut feature (currently EMPTY — none of the 19 candidates appears in any of the 376 required-features rows; re-verify at execution time)
- [ ] Code removal for code-affected cuts: every `#[cfg(feature = "NAME")]` / `cfg!(...)` site in `crates/katgpt-core/src/` (EMPTY for CUT; per-feature files listed below for the tranche)
- [ ] Version bump `0.5.0` → `0.6.0` (feature REMOVAL is breaking) + the 8 published sibling crates' dep-version coordination per Issue 992's note
- [ ] README.md + `examples/README.md` feature-count/flag-list updates (`scripts/count_features.py` gate row)
- [ ] Gates: `cargo check` · `cargo check --all-features` · `cargo test -p katgpt-core --lib` · `cargo clippy --all-targets` · `./scripts/docs_gate.sh`
- [ ] Downstream zero-cut-spelling audit: re-grep all 13 downstream repos + katgpt-rs internal lanes for every cut name (fwd + dep-table + CLI + required-features lanes; census snapshot is 2026-11-10 — siblings move daily, re-derive at execution)
- [ ] Publish cascade re-run per Issue 992 (kat-secret Touch-ID lane; core 0.6.0 must land so riir-refine's crates.io lane unblocks)

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
