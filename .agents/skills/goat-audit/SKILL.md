---
name: goat-audit
description: Audit cross-repo GOAT/gain primitive cherry-pick status across the multi-repo stack (katgpt-rs upstream, riir-* consumers). Detects stalls (default-on in katgpt-rs for ≥7 days with zero runtime wiring in riir-*), DRY violations (duplicated substrate in riir-* that should consume katgpt-core), and SOLID violations. Use when auditing primitive cherry-pick coverage, before opening a plan that consumes a katgpt-rs primitive, or quarterly as a hygiene gate.
---

# goat-audit — Cross-Repo GOAT/Gain Cherry-Pick Audit

Use this skill when auditing whether the riir-* private repos have consumed the GOAT-validated primitives shipped in `katgpt-rs`. It detects four failure classes:

1. **Stalls** — primitive default-on in katgpt-rs for ≥7 days, zero runtime wiring in riir-*.
2. **DRY violations** — riir-* ships a local copy of substrate that should consume `katgpt-core` / `katgpt-transformer` / `katgpt-pruners`.
3. **SOLID violations** — primitive consumed in the wrong layer (e.g. a sync-boundary primitive wired into a latent-only runtime, or vice versa).
4. **Fork drift (the Issue 019 class)** — riir-* is a fork of katgpt-rs (e.g. `riir-engine/src/lib.rs` says "Extracted from katgpt-rs (MIT, frozen at v0.1.0)") and a Layer 2 struct-name grep hits a file that defines its own `pub struct SameName` with zero `use katgpt_*::` imports. The audit must distinguish **consumer** (true positive, primitive wired) from **duplicate** (false positive, primitive re-implemented locally — the canonical has bit-rotted since v0.1.0).

## When to use

- **Quarterly hygiene gate** — re-audit after every major katgpt-rs release.
- **Before opening a plan** that consumes a katgpt-rs primitive — confirm it isn't already wired (avoid duplicate work).
- **When a user asks** "did riir-* get every good thing from katgpt-rs yet?" (the canonical trigger).
- **After a promotion in katgpt-rs** — track that the riir-* runtime wiring plan is filed within 7 days.

## DO NOT use for

- Paper distillation (use the `research` skill).
- Single-repo refactors with no cross-repo angle.
- Bug fixes with no architectural impact.

## Repos in scope (the product set of 8 — audit focuses on the 4 cherry-pick targets)

> Canonical list: `katgpt-rs/AGENTS.md` §"Repo count". `riir-dapps` (dApp layer,
> added 2026-08-20) is the 8th and is OUT OF SCOPE here for the same reason as
> riir-game-sdk: it composes outcomes, it hosts no feature-gated katgpt
> primitive to audit.

```
katgpt-rs          ← public engine (substrate: katgpt-core + 16 leaf crates + root)
riir-ai            ← private runtime/game (cognitive, ARG, CLR, HLA, karc, cwm, etc.)
riir-chain         ← private chain (LatCal, quorum, asset lifecycle)
riir-neuron-db     ← private neuron-shard leaf (Pod, freeze, consolidation, AnyRAG)
riir-train         ← private training vault (OUT OF SCOPE — training-only methods)
riir-game-sdk      ← private game-vocabulary facade + dev-tool workspace (OUT OF SCOPE —
                      consumes vocabulary from riir-games-shared in riir-ai, not katgpt-rs
                      engine primitives directly; transitively reaches katgpt-core only via
                      the always-on path dep, no feature-flag-gated primitives to audit)
```

**Why the SDK is out of scope:** this audit tracks katgpt-rs default-on
primitives (feature-flag-gated, GOAT-validated) consumed in riir-* runtimes. The
SDK is a facade over `riir-games-shared` (in riir-ai workspace) — its primitives
are vocabulary types, not katgpt-rs engine features, so it ships no
feature-flag-gated katgpt-rs primitive that needs cherry-pick tracking.
(`riir-armageddon` was listed here on the same reasoning — product-domain
types — until it was retired 2026-09-02, owner act.)

## Workflow

### Step 0 — Pre-flight (MANDATORY before any verdict)

Run all of these in parallel; do NOT skip:

1. `read_file katgpt-rs/Cargo.toml` — extract the `default = [...]` feature list. These are the GOAT-validated primitives.
2. `read_file katgpt-rs/crates/katgpt-core/Cargo.toml` — extract substrate features.
3. `read_file riir-ai/crates/riir-engine/Cargo.toml` — extract riir-engine features + the katgpt features each one forwards to.
4. `read_file riir-neuron-db/Cargo.toml` — extract neuron-db features.
5. `read_file riir-chain/Cargo.toml` + `riir-chain/crates/riir-chaind/Cargo.toml` — extract chain features.
6. `list_directory katgpt-rs/.benchmarks` — these track the GOAT gate evidence per primitive.
7. `list_directory riir-ai/.issues` + `riir-ai/.plans` — existing wiring plans/issues to avoid duplication.

### Step 1 — Build the default-on primitive inventory

From `katgpt-rs/Cargo.toml`'s `default = [...]`, build a table:

| Primitive | Default-on since | katgpt-rs plan | Substrate location |
|---|---|---|---|

Source the "default-on since" from the inline comment in the Cargo.toml (e.g. `# zone_density_routing DEFAULT-ON (Plan 351 Phase 3): ...`). If no date, mark "unknown" and treat as fresh.

### Step 2 — For each primitive, check riir-* consumption (THREE-LAYER check)

**Critical lesson (Issue 003):** feature-name grep returns false negatives. The `salience_tri_gate` feature is consumed via `SalienceTriGate::decide_with_delegate_nudge` in `karc_bridge/anticipation.rs` — a feature-name grep missed this; struct/function-name grep caught it.

**Critical lesson (Issue 019, 2026-07-04):** struct-name grep returns false POSITIVES in fork-derived repos. `riir-engine/src/transformer/mod.rs` defined its own `KVCache`/`KVSnapshot`/`PAGE_SIZE` with zero `use katgpt_transformer::` imports even though `katgpt-transformer` was declared as a mandatory dep in `Cargo.toml` and paid for in compile time — the prior audit marked `KVCache` as WIRED when it was actually a local duplicate shadowing the canonical type. Plan 406 Phase 1/2 de-forked these. Layer 3 (consumer-vs-duplicate discrimination) is mandatory before any WIRED verdict.

**Skill correction (2026-07-09, Issue 420):** the prior version of this paragraph cited `riir-engine/src/kvarn_quality.rs` defining `KvCacheQualityReport` as a duplicate of `katgpt-kv::kvarn::KvCacheQualityReport`. That example was factually wrong — `katgpt-kv` does NOT ship `KvCacheQualityReport` (verified: 0 matches across `katgpt-rs/**/*.rs`). The riir-engine type is a legitimately local `ThinkingController` abstraction (Plan 199 T1), NOT fork drift. The `crate::transformer` shadow case above replaces it as the canonical example.

For each primitive, run ALL THREE greps in parallel across the three consumer
repos named by §"Repos in scope" above.

**This narrow set is deliberate, not drift** — it is the reasoned in-scope list
(product set of 8 minus the four documented OUT OF SCOPE entries), which is why
the block carries the marker below and `scripts/skill_repo_set_gate.py` does not
flag it. Widen it only by editing §"Repos in scope" first; the scope section is
the decision, this grep is only its implementation.

Caveat worth knowing when you read a "0 consumers" result: it is a claim about
these three repos, not about the workspace's 19. A DUPLICATE of a primitive can
live in an out-of-scope repo (the standing example was `riir-armageddon`
consuming `GenericSpatialBelief` in 3 files — that repo is retired as of
2026-09-02, but the hazard it illustrates is not) — the anti-duplication
sweep that covers all of them is
`substrate-first` Audit Step 2, not this one.

<!-- repo-set-ok: the reasoned in-scope consumer set per §"Repos in scope" -->
```bash
cd /Users/katopz/git

# Layer 1 — feature-name grep (catches Cargo.toml forwards + cfg gates)
# -r + --include, NOT `riir-ai/**/*.toml`: bash without `shopt -s globstar`
# expands `**` as a single `*`, so the old form silently checked ONE level.
grep -rn "<feature_name>" --include='*.toml' --exclude-dir=target \
  riir-ai/ riir-chain/ riir-neuron-db/

# Layer 2 — struct/function-name grep (catches actual code consumption)
grep -rnE "<CamelCaseStruct>|<snake_case_fn>" --include='*.rs' \
  --exclude-dir=target riir-ai/ riir-chain/ riir-neuron-db/

# Layer 3 — consumer-vs-duplicate check (MANDATORY for every Layer 2 hit)
# For each file returned by Layer 2, grep the FILE for katgpt imports.
# If the file has zero `use katgpt_*::` lines, the hit is a DUPLICATE, not a consumer.
grep "^use katgpt" <each-file-from-layer-2-results>
```

**Layer 3 classification rules:**
- File has `use katgpt_*::SomeType` AND uses `SomeType` in code → **CONSUMER (true wired)**
- File has zero `use katgpt` lines but defines `pub struct SameName` → **DUPLICATE (fork drift)** — file the file as a de-fork candidate per Issue 019; do NOT count the primitive as wired
- File has `use katgpt_*::SomeType` but the Layer 2 hit name is locally defined → **MERGE candidate** — the file consumes some katgpt types but re-defines the queried primitive locally; needs human review

**Vocabulary translation before grepping:** list the primitive's 3–5 exported type/function names from the source file (e.g. `katgpt-rs/crates/katgpt-pruners/src/soft_reject.rs` exports `SoftRejectVerdict`, `SoftRejectConfig`, `soft_reject_decide`, `soft_reject_with_relax`, `RelaxationStrategy`, `NoRelaxation`). Grep for ALL of them, not just the feature name.

**Derived-primitive grep (the T2 `smooth_min_similarity` lesson, Issue 532, 2026-07-18):** when a primitive is consumed as SUBSTRATE by a later promoted primitive (e.g. `recos` in `katgpt-core` is implemented on top of `smooth_min_similarity` — `#[cfg(feature = "recos")]` lives inside the `similarity` module and implies `smooth_min_similarity`), grepping for the substrate's own names misses consumers of the derivative. The riir-* consumer of `recos` calls `recos_sim_ranking`, not `smooth_min_similarity` — a Layer 2 grep for `smooth_min` returns 20+ katgpt-rs hits and the riir-neuron-db consumer hides on a later page. **Mitigation:** before auditing a substrate primitive, list the katgpt-core features that imply it (read each feature definition in `crates/katgpt-core/Cargo.toml` for `feature_x = ["feature_y"]` edges — `recos = ["smooth_min_similarity"]` is the canonical case), then grep for the derivative primitive's exported names too. This was the second occurrence of the paginated-grep-hides-consumer failure mode (the first was T9 `local_branch_routing`, where the consumer used `branch_routing::{DotProductRouter, PostCandidateRouter}` aliases while the grep was for `branching`).

### Step 3 — Classify each primitive

| Class | Criteria | Action |
|---|---|---|
| **Wired (DEFAULT-ON)** | Layer 3 CONSUMER in riir-* runtime AND promoted to default-on in riir-engine/riir-games | No action |
| **Wired (opt-in)** | Layer 3 CONSUMER in riir-* runtime, opt-in feature in riir-* | Check the opt-in reason; if "until GOAT gate passes" and the gate is overdue, flag |
| **Stall** | Default-on in katgpt-rs ≥7 days, zero Layer 3 CONSUMER (Layers 1+2 miss OR Layer 2 hits are all DUPLICATES) | **File `.issues/` in the appropriate riir-* repo** |
| **Fork drift (Issue 019 class)** | Layer 2 hit but Layer 3 says DUPLICATE — riir-* file defines its own `pub struct SameName` with no `use katgpt_*::` | **File de-fork task in `.issues/` — see Issue 019** for the canonical riir-engine substrate de-fork plan |
| **Partial** | Layer 3 CONSUMER in one riir-* repo but the natural second consumer is missing | Note in issue; defer if natural consumer doesn't exist yet |
| **Deliberate** | Has a documented reason for not being wired (e.g. `claim_rubric` is CI-only by design) | No action |
| **Too fresh** | Default-on <7 days | Defer; re-audit next quarter |

### Step 4 — DRY / SOLID check

**Two complementary checks: the `crate::*` scan (catches imports) and the zero-import scan (catches fork drift).**

#### Check A — `crate::*` substrate refs (the Plan 008 lesson)

For each katgpt-core/katgpt-transformer/katgpt-pruners substrate module, check riir-engine src for `crate::*` imports of substrate that should be `katgpt_*::`:

```
# If riir-engine has its own hla/transformer/types.rs/tokenizer.rs/dd_tree.rs that
# duplicates katgpt-core, that's a DRY violation (Plan 008 class).
grep "crate::hla|crate::transformer|crate::types|crate::tokenizer|crate::dd_tree|crate::spec_types|crate::mcts|crate::sampling|crate::delta_mem|crate::simd" riir-ai/crates/riir-engine/src/**/*.rs
```

All should be `katgpt_core::*` / `katgpt_transformer::*` / `katgpt_speculative::*` (Plan 008 Phase 2 closure). Any remaining `crate::*` is a DRY violation.

#### Check B — Zero-import substrate files (the Issue 019 lesson)

Fork drift can hide in files that have NO `crate::*` substrate refs and NO `use katgpt_*::` lines — pure standalone local impls. Check B finds them:

```
# List all .rs files in riir-engine/src that have zero katgpt imports.
# Every substrate-side file in this list is a fork-drift candidate.
for f in $(find riir-ai/crates/riir-engine/src -name '*.rs'); do
  if ! grep -q '^use katgpt' "$f"; then
    echo "ZERO-IMPORT: $f"
  fi
done
```

For each ZERO-IMPORT file, manually classify:
- **Substrate duplicate** (matches a katgpt-* leaf module) → de-fork candidate per Issue 019
- **riir-only runtime** (cognitive stack like clr/karc/cgsp with no katgpt equivalent) → KEEP, not a violation
- **Test/example file** → exempt

The cognitive stack (arg_runtime, bom_arena, cce_runtime, etc.) intentionally has riir-local files; the LLM-substrate layer (transformer, quant, sampling, kv, lora-adapters, mcts, delta_mem) is where drift hides.

For SOLID: check that primitives are consumed in the right layer:
- Sync-boundary primitives (chain commitment, Merkle root) → riir-chain or riir-neuron-db, NOT riir-ai runtime.
- Latent-only primitives (emotion projection, set attention) → riir-ai runtime, NOT riir-chain.
- Substrate (SIMD, transformer, types) → katgpt-core, NOT riir-engine local.

### Step 5 — Output

Write the audit to a new `.issues/NNN_cross_repo_goat_cherry_pick_audit.md` file in the riir-* repo that owns the largest gap (usually riir-ai). Use this format:

```markdown
# Issue NNN: Cross-Repo GOAT/Gain Cherry-Pick Audit

## TL;DR
<one paragraph: how many stalls, which is strongest, what's the fix>

## Gap inventory (default-on in katgpt-rs, NOT cherry-picked)
| # | Primitive | Default-on since | riir-* status | Class |
|---|---|---|---|---|
| T1 | ... | ... | ... | TRUE GAP — fix |
| T2 | ... | ... | ... | Stall — defer |

## What IS wired (the strong half)
<list of consumed primitives, no action>

## Tasks
### Phase 1 — T1 <strongest stall>
- [ ] T1.1 ...
```

### Step 6 — Commit + report

Per global AGENTS.md rule, commit on `develop` (no feature branches):

```bash
git add .issues/NNN_cross_repo_goat_cherry_pick_audit.md
git commit -m "docs: file issue NNN — cross-repo GOAT cherry-pick audit (T1: <strongest stall>)"
```

Report to the user:
- Total primitives audited (count of katgpt-rs default-on).
- Stalls found (with days-since-promotion).
- DRY violations (count of remaining `crate::*` substrate refs).
- Top 3 priorities (strongest stalls with clear design intent).

## Common false-positive patterns (DO NOT report as gaps)

1. **Feature-name-only grep miss** — the primitive IS wired under a different name (struct/function). Always run the struct/function grep (Layer 2).
2. **Bench-only consumer is sufficient** — some primitives (e.g. `future_probe`) ship as benches that validate the primitive; runtime wiring waits for the gate to pass. Mark "Too fresh" not "Stall".
3. **Meta-discipline validators** — `claim_rubric` is CI-time only by design. Its `claim_rubric_bridge.rs` doc-comment says so explicitly. Don't flag.
4. **Opt-in by design** — `closed_unit_compaction` is opt-in because compaction is a sleep-cycle op, not hot path. Don't flag unless the opt-in reason is stale.
5. **Cross-repo transitively** — `subspace_phase_gate` is consumed via riir-neuron-db's freeze gate even though riir-ai runtime doesn't call the diagnostic directly. Mark "Partial" not "Stall".
6. **Fork-drift false WIRED (Issue 019 class)** — Layer 2 struct-name grep hits a file, but Layer 3 reveals the file has zero `use katgpt_*::` imports and defines its own `pub struct SameName`. The primitive is NOT wired — it's locally re-implemented. Report as **Fork drift** (file de-fork task), NOT as WIRED. Without Layer 3, the audit systematically under-counts substrate-side drift in fork-derived repos. Canonical case: `riir-engine/src/transformer/mod.rs` defined local `KVCache`/`KVSnapshot`/`PAGE_SIZE` while `katgpt_transformer::{KVCache, KVSnapshot, PAGE_SIZE}` shipped upstream (the dep was even declared in `Cargo.toml` but unused — `grep "^use katgpt_transformer"` returned 0 matches). Plan 406 Phase 1/2 de-forked these bit-identical and superset types. NOTE (Issue 420, 2026-07-09): a prior version of this example cited `riir-engine/src/kvarn_quality.rs` duplicating `katgpt-kv::kvarn::KvCacheQualityReport` — that was factually wrong (katgpt-kv ships no such type); the riir-engine `KvCacheQualityReport` is a legitimately local `ThinkingController` abstraction, not fork drift.
7. **Substrate-via-derivative consumer (Issue 532 T2 class, 2026-07-18)** — a substrate primitive IS wired in riir-*, but only transitively via a derivative primitive. `smooth_min_similarity` is substrate for `recos` (`recos` implies `smooth_min_similarity`); riir-neuron-db's `recos_rerank` calls `recos_sim_ranking`, which compiles + uses `smooth_min_similarity` transitively. A Layer 2 grep for `smooth_min` returns 20+ katgpt-rs-side hits and the riir-neuron-db consumer (using the `recos_sim_ranking` alias) hides on a later page. The audit verdicted "0 consumers in riir-*" — wrong. **Mitigation:** when a substrate primitive has derivative primitives that depend on it (read the feature implications in `crates/katgpt-core/Cargo.toml`), grep for the derivatives' exported names too, not just the substrate's own names. Same failure class as #1 (feature-name-only grep miss) but harder to spot because the alias lives in katgpt-rs, not riir-*.
8. **Data-contract consumption (Issue 532 method note, 2026-07-17)** — a primitive is consumed via a sync-boundary data contract, NOT via a `use katgpt_*::` import. The primitive's output shape (e.g. η ∈ R^P weights) crosses the sync boundary (chain quorum commit) as a sidecar receipt; the consumer (e.g. riir-chain) reads the committed bytes, not the Rust type. A Layer 2 grep for `use katgpt_*::` returns 0 matches → false STALL verdict. **Mitigation:** when a primitive produces raw synced values, check whether the consumer reads the committed sidecar bytes (Issue 003 T3-T7 pattern) rather than importing the type. The primitive's output shape IS the load-bearing contract — consumption is via byte-level agreement, not Rust-level import.

## The 7-day rule (when a stall becomes actionable)

- **< 7 days default-on:** Too fresh. Note in audit but don't file an issue.
- **7–14 days default-on:** Stall. File issue with wiring plan.
- **> 14 days default-on:** Critical stall. File issue + flag for the next standup.

The 7-day window gives the open primitive time to land its bench evidence before the runtime wiring is expected. It mirrors the "promote-or-demote within one cycle" discipline.

## Cross-references

- `katgpt-rs/AGENTS.md` — Feature Flag Discipline (the GOAT gate contract).
- `katgpt-rs/.agents/skills/research/SKILL.md` — research workflow (paper → 7-repo routing).
- `riir-ai/.issues/003_cross_repo_goat_cherry_pick_audit.md` — the canonical audit (2026-07-03). **Compromised** by the Layer 3 gap — substrate-side primitives marked WIRED in this audit need re-verification per Issue 019.
- `riir-ai/.issues/019_riir_engine_substrate_de_fork.md` — the LLM-substrate de-fork plan (2026-07-04). Documents the fork-drift failure class and the Layer 3 fix.
- `katgpt-rs/.plans/008_katgpt_core_substrate_extraction.md` — the cross-repo DRY closure record (cognitive substrate: hla/types/tokenizer/dd_tree).
- `riir-ai/.issues/532_cross_repo_goat_cherry_pick_audit.md` — the 2026-07-17 audit (Phase 1 HLA de-fork + Phase 2 watch list T2-T9 all resolved). File removed per noise-reduction rule; the method-note lessons (paginated grep, data-contract consumption) are preserved as false-positive patterns #7 + #8 above.
- **2026-09-04 audit (3rd run) — 1 critical stall filed (riir-ai Issue 867); RESOLVED same day.** Scope: the 3 katgpt-core default-on promotions since the 08-15 pass (`contrastive_scope` 08-19, `rating` 08-25, `kinematic_rollout` 08-26) + the 5 too-fresh deferrals now matured. Verdict: 2 wired (`rating` via riir-games ruliology `ParadigmRanking` — production `update_scored`; `drift_segment` via riir-engine `npc_episodic` → civ salience gate, opt-in both sides), **1 critical stall (`similarity_inference`: 24 days default-on, ZERO consumers workspace-wide — coordinator-verified; wire-or-demote in Issue 867 T1)** → **T1 RESOLVED 2026-09-04: DEMOTED to opt-in (katgpt-rs `e5fdbd35`, Issue 867 T1.3)** — all three wire candidates were gameplay-emergence owner calls; the compose-partner question answered (latent_cce_moderator enables `katgpt-rs/cce_moderator`, never `similarity_inference`); default lib surface 2004 → 1981 (−23). T2.1 also done same day: the 5th Elo copy (`go_trust_region.rs`) delegated to `rating::update_scored`, bit-identical (riir-ai `567ff35c3`). T2.2 premise drift: `kg_pkm_attention` is now Issue 765's GOAT-benched opt-in-pending-consumer fix (sanctioned posture); `latent_functor/pkm_bridge` stays the one genuine zero-caller deletion candidate (owner call). 4 notes (contrastive_scope's consumer is riir-clippy only; kinematic_rollout's two designated PoCs both REFUTED their gameplay claims — non-adoption is defend-wrong-correct; self-spec acceptance is a config default exercised only via the FlashAR bench; the PkmEpisodicStore TF-IDF bridge idea for ndb's Bm25Index). Two run-premise corrections: FlashAR Eq 21 + PkmEpisodicStore TF-IDF are opt-in upstream (no stall clock). Upstream record drift found + fixed same day: the contrastive_scope lib.rs comment still said "Opt-in POC" post-promotion. Method note: the root-vs-core default-list diff must extract only up to the FIRST `]` — a greedy sed that captures into the trailing Phase-comment text (which contains `]` and commas) inflated the core default count 73→308 and hid the `kinematic_rollout`/`contrastive_scope` promotions from the first diff pass.
- **2026-08-15 clean-pass audit** (no issue filed — zero actionable findings, per the noise-reduction rule). Scope: all 10 katgpt-core primitives promoted default-on after the 2026-07-17 audit and ≥7 days old — poincare_navigator (07-18), chunked_content_store (07-18), causal_identification (07-18), conformal_predictive_intervals (07-20), karc_forecaster (07-21), hope_capacity (07-24), hebbian_kernel_memory (07-25), ane_fused_chain (07-14, borderline re-check), clr_weighted_set_attention (08-06), phase_separation (08-07). Method: Layers 1+2+3 via two parallel subagent greps across riir-ai/riir-chain/riir-neuron-db. **Verdict: 10/10 WIRED, zero stalls, zero unintentional duplicates.** Two default-on-strong (karc_forecaster via engine `karc_runtime`+`npc_sleep_time`; ane_fused_chain via `npc_brain_router`); eight opt-in consumers, all with Layer-3-verified `use katgpt_core::` imports (chain `neuron_vessel_cold_tier`, neuron-db `hope_bridge`/`hebbian_fact_store`, games swarm `tick_swarm_emotions_collective`, engine `causal_id`/`poincare_bridge`/`karc_bridge::conformal_sidecar`/`phase_separation_bridge`). Partial-surface notes (no action — natural consumers don't exist yet): the conformal metrics suite (`winkler_score`/`empirical_coverage`/`mean_crps`/`ResidualRingBuffer`) and the content-store CDC/fetcher half (`FastCdcChunker`/`TieredChunkFetcher`/`NetChunkFetcher`/`build_binary_merkle_*`) are unconsumed; riir-gpu does not consume `ane_roofline` (its "roofline" hits are local GB/s bench helpers — an open opportunity, not drift). Deferred as too fresh for the next audit: similarity_inference (Bench 579, 08-11), katgpt-forward self-spec acceptance (Issue 587/Bench 634, 08-14), mop (opt-in — promotion is a product decision), drift_segment (katgpt-kv, 08-15), FlashAR Eq 21 (08-15), PkmEpisodicStore TF-IDF gate (08-15).
