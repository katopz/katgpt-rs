---
name: substrate-first
description: Pre-implementation DRY gate + existing-code drift audit for the multi-repo workspace. Use BEFORE writing any new System impl, trait, perception/cognition/emotion pipeline, state management, spatial query, or vocabulary type — to verify you're consuming existing substrate, not duplicating it. Also use to AUDIT existing code for parallel-system DRY violations (code that re-implements substrate under different names). The canonical defense is vocabulary translation; concepts ship under operator names (`GenericSpatialBelief`, `decay_confidence`), not English names ("threat field", "spatial hash") — a single-vocabulary grep returns ZERO hits even when the substrate fully exists. Sibling to boundary-guard + goat-audit + feature-gate-audit + doc-sync.
---

# Substrate-First — DRY gate + drift audit

The workspace — **18 repos** carrying a root `BOUNDARY.md` as of 2026-09-01,
derived by the snippet in Step 2 rather than typed here — has a recurring
failure mode: an agent receives a task
("add threat perception"), jumps to implementation without checking existing
substrate, and builds a **parallel system** that duplicates functionality
already shipped under a different name. The user then has to catch it manually.

This skill prevents that. It runs in two modes:

1. **Pre-implementation gate** — run BEFORE writing code
2. **Existing-code audit** — scan for already-shipped DRY violations

## Canonical failures (the pattern this skill prevents)

### Failure 1 — ThreatField (Issue 047, riir-mmorpg-examples, 2026-08-01)

**Built:** `ThreatField` — a spatial hash grid (`HashMap<(i32,i32), u32>`)
for threat perception. Deposited monster positions into cells; NPCs sampled
the 3×3 neighborhood.

**Already existed:** `GenericSpatialBelief<T>` + `target_within_visible_radius()`
+ `decay_confidence()` — the full fog-of-war → belief → decay pipeline in
`riir-games-shared/src/game_traits/spatial.rs`.

**Why the grep missed it:** The agent searched for "threat field" /
"spatial hash". The substrate ships as `GenericSpatialBelief` /
`SpatialBelief` / `confidence_decay`. **A single-vocabulary grep returns
ZERO hits even when the substrate fully exists.**

**Resolution:** Reverted. The plain scan (`tick_swarm_emotions`) is a simpler
POC-scale simplification. The belief-based system is deferred until fog-of-war
becomes a gameplay feature.

### Failure 2 — Orchard + Motivation in SDK src/ (Issue 490 + Issue 493)

**Built:** `NpcReasonSystem`, `AppleGrowSystem`, `OrchardGoal`, `EmotionField`,
`EmotionAxis`, `tick_feeling_brain` directly in the SDK facade's `src/`.

**Already existed:** The boundary rule (riir-game-sdk/AGENTS.md) says "no game
logic in `src/`" + Proposal 019 excludes emotion from the SDK. The substrate
belongs in `riir-games` (in riir-ai).

**Why the grep missed it:** The agent didn't check whether the types violated
the domain classification rule (latent semantic emotion ≠ raw physical
vocabulary).

**Resolution:** Extracted to `riir-games` (orchard) and `riir-games::motivation`
(emotion). The SDK re-exports them.

---

## Mode 1: Pre-implementation gate (BEFORE writing code)

Run this checklist before implementing ANY of these:

- New `impl System` or tick function
- New trait + impl (perception, cognition, emotion, state management)
- New spatial query / index / hash / grid
- New vocabulary type / DTO / config struct
- New "helper" function that does math (distance, sigmoid, projection)
- New pipeline (perception → emotion → behavior, freeze → sync → thaw)

### Step 1 — Vocabulary-translate your search

The concept you're building probably already exists under a **different name**.
Before grepping, write down 3+ name variants for the concept:

| You're building... | Also search for... | Likely substrate names |
|---|---|---|
| "threat field" / "spatial hash" | belief, perception, spatial cognition, fog-of-war, visibility | `GenericSpatialBelief`, `SpatialBelief`, `confidence_decay`, `target_within_visible_radius` |
| "emotion" / "feeling" / "mood" | affect, drive, motivation, fear, desire | `EmotionField`, `EmotionAxis`, `AffectField`, `DriveSpecSet`, `tick_feeling_brain` |
| "state sync" / "delta" / "snapshot" | replication, gossip, cache, commitment | `SyncBlock`, `ZoneDelta`, `PlayerStateCache`, `GossipDelta`, `SyncRegistry` |
| "position" / "movement" / "physics" | spatial, coordinate, force, velocity | `MapPos3D`, `ForceVector`, `SpatialIndex`, `GridSpatialIndex` |
| "save" / "persist" / "freeze" | thaw, snapshot, serialize, store | `freeze_avatar_delta`, `LocalKvStore`, `ShardIndex`, `NeuronShard` |
| "validate" / "anti-cheat" / "check" | verify, guard, proof | `AvatarAntiCheatValidator`, `AdaptiveModConfig`, `make_validator_predicate` |
| "tick" / "update" / "loop" | system, schedule, game core | `System`, `TickCtx`, `World`, `GameCore`, `FrameSnapshot` |
| "knowledge" / "relationship" / "graph" | triple, semantic, KG | `KgTriple`, `KgTripleTemplate`, `DualSignalEvidence` |
| "attack" / "damage" / "combat" | hp, health, dex, fight | `Hp`, `Dex`, `CombatConfig`, `combat_tick`, `attack_interval_ticks` |
| "npc" / "swarm" / "crowd" | agent, bot, forager | `SwarmState`, `ForagerSwarmSystem`, `ForagerAi`, `BotThought` |
| "decay" / "fade" / "forget" | sigmoid, heal, baseline, confidence | `decay_confidence`, `tick_feeling_brain`, `sigmoid`, `EmotionBaseline` |
| "embedding" / "vector" / "latent" | direction, projection, HLA, shard | `NeuronShard`, `HlaCacheProxy`, `compute_animal_emotions` |

**This is the same technique as the paper→code vocabulary translation in
AGENTS.md §"Manifold Geometry".** The R296 canonical failure applies internally
too: a concept-name grep returns zero hits because the math ships under operator
names.

### Step 2 — Grep the CODEBASE (not just docs)

```bash
# Grep ALL repos for the substrate names from Step 1.
# Use multiple variants — the concept may exist under any of them.
#
# DERIVE the repo set; never type it (fixed 2026-09-01 — the hard-coded
# 7-repo brace list this replaced could not see 11 of the 18 contract repos,
# INCLUDING two product-set ones: riir-armageddon consumes
# `GenericSpatialBelief` in 2 files that the canonical DRY grep was structurally
# unable to find, and riir-dapps was equally invisible. A gate that cannot see a
# repo cannot tell you whether it consumes substrate or duplicates it.)
cd /Users/katopz/git
grep -rn 'GenericSpatialBelief\|SpatialBelief\|confidence_decay' \
    --include='*.rs' --exclude-dir=target --exclude-dir=.git \
    $(ls -d */ | while read -r d; do
        [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && printf '%s ' "$d"
      done)
```

Three deliberate details, each measured on this pattern (2026-09-01):

| detail | why |
|---|---|
| `-d "$d/.git"`, not `-e` | a `git worktree` has a `.git` **file**. `ai-perfwt` (a riir-ai worktree) otherwise contributes 43 duplicate hits — a DRY gate reporting one implementation twice is the failure it exists to prevent |
| `--exclude-dir=target` | the old form had none, and spent nearly all of its 23 s inside build dirs |
| unquoted `$( … )`, not `$VAR` | zsh does **not** word-split an unquoted parameter expansion (grep would get one giant argument and fail) but it *does* split an unquoted command substitution. Verified in this shell |

Net: **227 hits in 1.4 s** vs the old form's 221 in 23 s — strictly wider
coverage, no worktree duplicates, 16× faster.

```bash
# Also grep .research/ and .proposals/ for design rules that apply:
grep -rn 'two-brain\|fog-of-war\|domain classification\|sync boundary' \
    /Users/katopz/git/*/.{research,proposals,docs}/ 2>/dev/null
```

If you find existing substrate → **STOP**. Consume it. Do not build a parallel
system. Document why you're consuming it (in the plan/issue).

### Step 3 — Check AGENTS.md architectural rules

Before building, verify your design doesn't violate these:

| Rule | Source | What it means |
|---|---|---|
| **Domain classification** | AGENTS.md §"Latent vs Raw Space Rules" | Physical = raw exact; Semantic = latent dot-product + sigmoid; Social = KG triples |
| **Two-brain model** | AGENTS.md §"Spatial Cognition" | Info brain (synced ground truth) ≠ think brain (per-NPC beliefs, fog-of-war gated) |
| **Sync boundary** | AGENTS.md §"Sync Boundary Rule" | Through `SyncBlock` → quorum → Cold = raw + deterministic; Local = latent |
| **Bridge pattern** | AGENTS.md §"Bridge Pattern" | raw → latent = dot+sigmoid; latent → raw = clamp; zero-alloc, gateable |
| **KG triple emission** | AGENTS.md §"KG Triple Emission" | Semantic encounters → KG triple; Physical events → TxDelta with raw values |
| **Facade constraint** | riir-game-sdk/AGENTS.md | SDK = re-export facade, no engine deps; vocabulary in riir-games-shared |
| **Boundary rule** | riir-game-sdk/AGENTS.md | No game logic in consumer `src/`; game systems in `riir-games` |

If your design violates any of these → **STOP**. File an issue. Rethink.

### Step 4 — Decide: consume vs. build

| Situation | Action |
|---|---|
| Substrate EXISTS and fits | Consume it. Wire via trait/config. Zero new substrate code. |
| Substrate EXISTS but wrong shape | Extend the substrate (in the right repo). File a plan. |
| Substrate DOESN'T exist | File an issue in the right repo FIRST. Then build. |
| You're not sure | **STOP and file an issue.** Don't guess. |

**Never build new substrate inside a consumer.** The consumer provides data +
wiring only. If you're writing loops/math/constants in a consumer's `src/` →
you're building substrate in the wrong place.

### Step 5 — Record the decision

In the plan/issue, document:

```
## Substrate check (substrate-first skill)
- Searched for: [concept names + variants]
- Found: [existing substrate or "none"]
- Decision: [consume / extend / build new]
- Architectural rules checked: [list which rules apply + verdict]
```

---

## Mode 2: Existing-code audit (scan for drift)

Run this when reviewing code, when you suspect a parallel system, or quarterly
as a DRY-hygiene gate (alongside boundary-guard).

### Audit Step 1 — Inventory substrate primitives

For each domain, identify what substrate exists:

```bash
# Perception / spatial cognition
grep -rn 'GenericSpatialBelief\|SpatialBelief\|target_within_visible_radius' \
    --include='*.rs' /Users/katopz/git/*/  | grep -v '/tests/' | grep -v '/target/'

# Emotion / affect
grep -rn 'EmotionField\|EmotionAxis\|tick_feeling_brain\|AffectField\|DriveSpecSet' \
    --include='*.rs' /Users/katopz/git/*/

# State sync
grep -rn 'SyncBlock\|ZoneDelta\|PlayerStateCache\|GossipDelta\|SyncRegistry' \
    --include='*.rs' /Users/katopz/git/*/

# Spatial
grep -rn 'SpatialIndex\|GridSpatialIndex\|OctreeSpatialIndex\|MapPos3D' \
    --include='*.rs' /Users/katopz/git/*/
```

### Audit Step 2 — Grep for parallel systems

For each substrate primitive found in Step 1, grep consumer code for
reimplemented versions.

**Derive the consumer set** — this block hard-coded
`{riir-mmorpg-examples,riir-game-sdk}` until 2026-09-01 and so could see 2 of
the 18 contract repos. That is the same defect the vocabulary step above was
fixed for, in the same file, one section down: `60655c48` corrected the Step 2
*named* "Step 2" and left this one, which is the step that actually looks for
duplicate implementations. `riir-armageddon` — a product-set repo that consumes
`GenericSpatialBelief` in 2 files (6 sites) — was invisible to every grep below.

```bash
cd /Users/katopz/git
consumers() {
  ls -d */ | while read -r d; do
    [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && printf '%s ' "$d"
  done
}
# Look for inline distance math (should use MapPos3D methods):
grep -rn '(dx.*dx.*dy.*dy).*sqrt\|distance_2d.*fn\|fn.*distance' \
    --include='*.rs' --exclude-dir=target --exclude-dir=.git $(consumers)

# Look for inline sigmoid/exp (should use substrate sigmoid or tick_feeling_brain):
grep -rn '1\.0\s*/\s*(1\.0\s*\+\|sigmoid\|exp(' \
    --include='*.rs' --exclude-dir=target --exclude-dir=.git $(consumers)

# Look for HashMap-based spatial structures (should use SpatialIndex substrate):
grep -rn 'HashMap.*i32.*i32\|spatial.*hash\|cell.*grid' \
    --include='*.rs' --exclude-dir=target --exclude-dir=.git $(consumers)

# Look for parallel belief/perception types (should use GenericSpatialBelief):
grep -rn 'struct.*Belief\|struct.*Perception\|struct.*Visibility\|last_known' \
    --include='*.rs' --exclude-dir=target --exclude-dir=.git $(consumers)

# Look for parallel emotion types (should use EmotionField/AffectField):
grep -rn 'struct.*Fear\|struct.*Mood\|struct.*Emotion\|fear.*f32' \
    --include='*.rs' --exclude-dir=target --exclude-dir=.git $(consumers)
```

### Audit Step 3 — Classify findings

For each hit, classify:

| Classification | Meaning | Action |
|---|---|---|
| **False positive** | The code is legitimately consumer-specific (e.g., `MonsterThreatSource` impl) | No action — document why |
| **POC simplification** | Duplicates substrate but produces identical behavior at POC scale | Document as known debt; fix when scale changes |
| **DRY violation** | Re-implements substrate under a different name | File issue; extract to substrate |
| **Architectural violation** | Violates two-brain model / sync boundary / domain classification | File issue; redesign |

### Audit Step 4 — Report

Summarize findings:

```
## Substrate-first audit (date)
### Substrate inventory
- [domain]: [primitive] at [location]
### Findings
- [file:line] — [classification] — [description]
### Clean
- [domain] — no violations found
```

---

## The vocabulary-translation defense (why this skill exists)

The hardest failures to catch are the ones where the substrate **exists** but
ships under a name that doesn't match the concept you're searching for. This is
the R296 canonical failure (documented in AGENTS.md §"Manifold Geometry"),
applied internally:

```
You think: "I need a threat field"
You grep:   "threat field" / "spatial hash"  → 0 hits
Substrate:  GenericSpatialBelief + decay_confidence  → exists, fully functional

You think: "I need emotion decay"
You grep:   "emotion decay" / "fear fade"  → 0 hits
Substrate:  tick_feeling_brain + DecayRates + EmotionBaseline  → exists

You think: "I need state persistence"
You grep:   "save state" / "persist"  → 0 hits
Substrate:  LocalKvStore + freeze_avatar_delta + ShardIndex  → exists
```

**The defense:** always search 3+ vocabulary variants. The translation table
in Mode 1 Step 1 is the canonical reference. Extend it when you discover new
mismatches.

---

## When NOT to use this skill

- Pure refactoring that doesn't add new concepts (renaming, reorganizing)
- Bug fixes in existing code (the substrate is already consumed or not)
- Test-only code (tests can define inline helpers)
- Build/config changes (Cargo.toml, scripts)

---

## Relationship to sibling skills

| Skill | What it checks | When |
|---|---|---|
| **substrate-first** (this) | "Does the substrate already exist? Are you duplicating it?" | Before writing code + audit |
| **boundary-guard** | "Is this code in the right repo? Is the consumer too fat?" | After writing code + audit |
| **feature-gate-audit** | "Do feature-gate claims match source wiring?" | Before promoting/demoting flags |
| **goat-audit** | "Has the katgpt-rs primitive been cherry-picked to riir-*?" | Cross-repo cherry-pick tracking |
| **doc-sync** | "Do docs match git history?" | After landing plans/issues |

`substrate-first` is **upstream** of `boundary-guard`: if substrate-first
catches the drift before it ships, boundary-guard has nothing to find. They're
complementary — substrate-first is the prevention, boundary-guard is the cure.

---

## Filing violations

When the audit finds a DRY violation or parallel system:

1. **File an issue** in the repo where the violation lives
2. **Reference this skill** + the substrate it duplicates
3. **Include the vocabulary translation** (what you searched for vs. what the
   substrate is actually called)
4. **Propose the fix** (consume substrate / extract to substrate / revert)
5. **Classify** (POC simplification vs. DRY violation vs. architectural violation)

Do NOT fix in the same commit as detection — separate detection from fix so
other agents can review the violation independently.

---

## Run log

| Date | Scope | Verdict | Record |
|---|---|---|---|
| 2026-09-04 (05:0x, zero-contention unit — `digit_ood_eval` at 223% CPU + the 851 MLX ranking `--wait-quiet 600` armed + sibling Batch-104 GOAT in `/tmp/riir-clippy-b104` → grep/parse-only per the Plan 337 discipline) | **Mode 2 drift audit — the post-08-29 fresh wave**: 092 S1–S3 extraction (~10K LOC consumer→substrate moves in `riir-games-mmorpg`), mmorpg lineage surface (Issue 094), dapps KAT crank/alarm legs, dao yield-ceiling mirror, seal-remake viewer/node fresh code, chain pay_request + asset-lifecycle/forensic, and the VERIFICATION half of the three copy-class campaigns (853/861 percentile, 867 Elo, 087 triage) | **1 finding FILED (ndb Issue 611, detection-only); the three copy-class campaigns verified TERMINAL/COMPLETE; zero parallel systems. The discipline held through the largest consumer→substrate move in workspace history.** | Definition-level greps (`struct\|enum\|trait\|fn` at line start) over the derived 16-repo set, `obsolete/` excluded, plus the three copy-class censuses and body-level reads of every candidate. **[1] nearest_rank/percentile census: COMPLETE, zero unaccounted copies** — canonical `katgpt_core::stats::nearest_rank` (861 promotion); documented specializations hold (games-shared stats.rs leaf-clean note, riir-e2e `7ac2315` 2nd specialization); `speculative/types.rs::nearest_rank_p99` is a different fn (rank-INDEX computer, exact-int `div_ceil`, own doc); `tpr/als.rs:551` + mmorpg `ccu_loadgen.rs:654` + civ `prod_latency_bench.rs:391` are INTERPOLATED percentiles (round/lerp on the (n−1) scale — a different statistic, auditor-SAFE class; converting als would rewrite BLAKE3-committed artifact bytes — correctly left alone). Auditor cross-check: 0 DEGENERATE / 0 TRUNC-VAR both repos. **[2] Elo census: COMPLETE** — `katgpt-pruners/arena/types.rs` expected/update DELEGATE to `katgpt_core::rating` (Issue 686 note in-source; counted in 867-T2.1's five); dao consumes `rating::*` constants directly (declared allowlist dep). **[3] triage/Regime: HOLDING** — `pet_teaching_677.rs::triage` is the τ-binding adapter over the SDK facade (087 fix INTACT through the S3 re-host); substrate `pet_teaching::frontier_regime_of` consumes `katgpt_core::hint_regret::triage` DIRECTLY — correct for substrate (the facade route is unavailable in riir-ai; a dep on the SDK would be a back-edge; the in-source comment documents the route history). Other `Regime` enums (mean_field, kinematics, data_probe, four_regime_router, jlens_poc, RegimeBehavior) are domain-vocabulary collisions, not copies. **[4] THE FINDING — riir-neuron-db carries 4 PRODUCTION scalar sigmoid copies while katgpt-core is an UNCONDITIONAL dep there**: `neighbor_heal.rs:100` (pub, zero external consumers — grep-verified), `shard_compactor.rs:952`, `local_kv/views.rs:76` (3 lines below a `katgpt_core::simd::simd_dot_f32` call), `interpolation_geometry:427` (audit-grade, self-documented mirror of shard_compactor); the crate's own `precision_drift.md` documents ULP-tolerance for exactly these paths, and `zone_geometry`/`dec_arena` production ALREADY consume the substrate `simd_sigmoid_inplace` (their local scalars are `#[cfg(test)]` parity refs — the delegation pattern exists in-repo). Filed as **ndb Issue 611** (`ffdc38a`, detection-only, highwater 610→611) with the freeze-commitment version-boundary caveat. Deliberate keep: `substrate_geometry_audits.rs:207` clamps ±50 because it MIRRORS CommittedFieldBlend's gate formula (a different function, not `katgpt_core::sigmoid`). **[5] riir-chain sigmoid census: NOTE-LEVEL, no issue** — `forensic/recover.rs` f64 = REQUIRED (`chain_forensic = ["dep:blake3"]` only); `congestion.rs` inline = REQUIRED (`chain_congestion_control` pulls no katgpt-core); engine-bridge `lora_posterior.rs` = REQUIRED BY DESIGN (the crate exists to keep katgpt out of the chain Docker context); `consensus/curator_bridge.rs` + `curator_reward.rs` sit under `chain_curator` (dep:katgpt-core available) BUT all three consuming fns have ZERO production callers and `compute_accuracy_score` is test-only — production-dead surface carrying latent duplication; delegate-or-prune when consumers materialize. Also: `katgpt_core::sigmoid` is Cephes (~1 ULP), NOT bit-identical to the chain copies' libm-exp forms — delegation there is a semantic change even where the dep exists. `block_backoff.rs` delegates correctly (chain_guard exists to arm exactly that). **[6] 092 moved substrate code: CLEAN** — all S1–S3 modules import substrate paths (`GenericSpatialBelief`/`MapPos3D`/`social`/`swarm`/`game_sync`) + crate-internal siblings; no re-implemented substrate. **[7] lineage.rs (094): model consumer** — composes ndb `MerkleFrozenEnvelope`/`HEADER_LEN`/`GENESIS_PARENT_COMMITMENT`; blake3 only for per-envelope integrity. **False positives documented:** riir-viz `QuestLayerVisibility`/`LayerVisibility` (render-layer masks, view layer); predation `Fov*SwarmEmotionSystem` (documented model-consumer wrappers); `crowd_demo` sigmoid/`FearColorConfig` (view color mapping); `NullAffectField` (canonical substrate default); riir-auth `session.rs::sigmoid` (a DELEGATING wrapper over `katgpt_core::sigmoid` — def-grep false alarm, body read required); dapps kat/ has zero latent/sigmoid gating of money decisions (domain rule holds); seal-remake + viewbridge fresh code clean (viewbridge's 2 fresh files are xtask/FFI = boundary-guard territory). Method note: `$(var)` is command-substitution syntax and EXECUTES a command named by the variable — the repo-set helper must be a shell FUNCTION `consumers() { …; }` called as `$(consumers)`, or the grep silently falls back to scanning the whole tree (this run's first attempt did, and `obsolete/` hits had to be filtered; the corrected form inlines the set). Sibling safety: ndb's 46 dirty src files = live sibling WIP — the finding touched only `.issues/` (+highwater); riir-ai's bench/rematch dirties untouched. | **Mode 2 drift audit — the 11 repos the fixed Step 2 can now reach** (riir-armageddon, riir-dapps, riir-dao, riir-deployer, riir-viewbridge, riir-auth, riir-burner, katgpt-web, riir-clippy, seal-game-editor; riir-unity has 0 `.rs`). ~700 `.rs` files. Ran immediately after the instrument fix in the row below — a blind gate's clean history is not evidence. | **CLEAN — zero parallel systems, zero issues filed.** Two definitions looked like candidates and both survive inspection: `riir-auth/src/kg.rs:66 AuthKgTriple` CONSUMES `riir_neuron_db::vibe::KgTripleTemplate` and converts into it (a domain enum over the substrate, with the sync-boundary rule stated in its header — the model citation, not a violation); `seal-game-editor/…/model_3d_view.rs:381 SceneSpatialIndex` is a private AABB BVH over mesh triangles for viewport ray-picking — a rendering structure, not the game-world `GridSpatialIndex` grid. `riir-armageddon` consumes `GenericSpatialBelief` correctly (3 files). | Two definition-level grep families over the set: the Step 1 substrate vocabulary (belief / emotion / affect / spatial-hash / swarm / fog / sync-delta / KgTriple / NeuronShard / MapPos / ForceVector) and the game-rule + cognition family (Goal / Drive / Motivation / Reputation / Karma / Quest / Crafting / Bounty / Threat / Perception / Cognition / Memory / Consolidat). Definitions only — `struct|enum|trait|fn` at line start — because CONSUMPTION is the desired outcome and only a *definition* can be a parallel system. seal-game-editor's 28 `Quest*` types are content-AUTHORING types in the repo that owns authoring; `riir-dapps` defines zero game-rule types, which is the riir-chain Issues 096/097 failure class not recurring. |
| 2026-09-01 | **Instrument fix, not an audit** — Step 2's own repo coverage, measured after the workspace gate reported 18 contract repos while this skill said 7 | **The gate was blind to 11 of 18 repos, two of them product-set.** `riir-armageddon` consumes `GenericSpatialBelief` in 2 files (6 sites; the 2026-09-01 row said 3 files and was re-measured) the hard-coded brace list could not reach (it consumes correctly — but this skill could not have told you); `riir-dapps` equally invisible. | Fixed in `60655c48` by deriving the set instead of typing it. Three measured details, restated here because that commit's message lost them to backtick substitution: **(1)** test `-d "$d/.git"`, NOT `-e` — a `git worktree` has a `.git` *file*, and `ai-perfwt` (a riir-ai worktree) otherwise contributes 43 duplicate hits, a DRY gate reporting one implementation twice; **(2)** `--exclude-dir=target` — the old form had none and spent nearly all of its 23 s inside build dirs; **(3)** unquoted `$( … )`, NOT `$VAR` — zsh does not word-split an unquoted parameter expansion (grep receives one giant argument and fails) but does split a command substitution. Net **227 hits / 1.4 s** vs **221 / 23 s**. Same class swept in 4 sibling skills: proposal / goat-audit / feature-gate-audit / research each kept a private copy of the product set reading '7 repos', all omitting `riir-dapps` since 2026-08-20; each now cites `AGENTS.md` as the one canonical home. |
| 2026-08-17 | Mode 2 drift audit — riir-mmorpg-examples (the Plan-022/539-era fresh code: party/social_signals/monster_predation/pet_pvp/pvp_karma/shared_quest_monsters/scenario_runner) + riir-game-sdk `src/` | **CLEAN overall** — 1 new minor D1 site; 2 model consumers found | The 5 audit grep families run over `src/`+`crates/` (both consumers). Findings: (1) **pet_teaching.rs:678** — `(dx*dx+dy*dy).sqrt()` where `Position.0` IS substrate `MapPos3D` → delegatable to `distance_2d_to`; appended to mmorpg Issue 069's D1 ledger (`61899d3`, detection-only). (2) **Model consumers** (the discipline is HOLDING in fresh code): `predation.rs` wraps substrate `tick_swarm_emotions_fov` (emotion.rs:353) with scratch wiring; `pet_alarm.rs` consumes `GenericSpatialBelief<PlayerTarget>` + fog-of-war gate + `distance_2d_to`. (3) False positives documented: ~12 squared-distance comparisons (`dx*dx+dy*dy <= r*r`) are the CORRECT no-sqrt idiom (the Batch 53 `distance-sq-no-sqrt` rule) — a naive D1 "fix" there would be a pessimization; `constants.rs:796` grid-space heightfield math; `authority/karma.rs:656` wire-DTO `[f32;N]` arrays (not MapPos3D); `EntityVisibility` (render-kind mask); `QuestRestockBelief` (temporal EMA, two-brain-compliant docs); game-sdk `ai/cognition.rs:42` (doc-comment example). Clean: inline sigmoid (0 hits), HashMap spatial grids (0 hits). NOTE: audit ran against origin/develop `339f4b7` (the sibling's local develop is diverged on a line without a48af77; findings greps used the local tree — issue-filed via the temp-worktree pattern). |
| 2026-08-19 | Mode 2 drift audit — the post-08-17 fresh wave (queue item 7): mmorpg ~60 files (Plans 023/024/025/026 + Issues 063-067/070-075 era — session FSM, target coordination, remote smoothing, scenario runner stimuli, planner view) + game-sdk 3 files (`entity_sync`/`quest_log_sync`/`lib`) | **CLEAN overall — 1 new minor D1 site; the discipline is HOLDING in the fresh plan code** | The 5 grep families over the `git diff --name-only` fresh set. Findings: (1) **`game/monster_pet_debug.rs:216`** — `(mx-hero_pos.x).powi(2)+….sqrt()` where BOTH operands ARE substrate `MapPos3D` (`Position(pub MapPos3D)` derefs) → delegatable to `distance_2d_to`; debug-panel cold path (once-per-frame, cached — Plan 017); detection-only record (the pet_teaching.rs:678 class; the old Issue 069 D1 ledger is resolved+removed, so THIS log row is the record). (2) False positives documented: `authority/karma.rs:669` + `zone_sweep.rs:320/579` (wire/snapshot `[f32;2]` operands — no MapPos3D to delegate, the standing EXEMPT class); `authority/mod.rs:1021` (snapshot-POD diagnostics metric); `constants.rs:878-880` (grid-space heightfield — the documented class); `constants.rs:701` (`FovLocomotion::from_step` takes raw `(dx,dy)` params — POC-local classifier, no substrate type in scope); `constants.rs:1249` + `quest_combat.rs:10118` (test-only); `remote_smoothing.rs:403` (exponential decay factor ≠ sigmoid); quest_combat cluster-radius sqrts (radial distribution math, not distance). (3) **CLEAN families**: inline sigmoid (0 fresh), HashMap spatial grids (0), parallel beliefs (0 new — `QuestRestockBelief` re-confirmed documented), parallel emotions (0). game-sdk fully clean — and `entity_sync/mod.rs` is a MODEL extraction (the NPC batch promoted from mmorpg to SDK vocabulary with the orchard-apple contrast rationale in its header — the substrate-first direction working as designed). |
| 2026-08-26 | Mode 2 drift audit — facade-route + signature-drift audit (queue item 7, post-Issue-085): the Class A pattern (direct `katgpt_core::` reaching past `riir_game_sdk`) across every facade-bound repo on disk + the SDK's 3 katgpt-core forwards vs current katgpt-core source | **CLEAN — zero findings, zero issues filed. The Issue 085 fix held; no facade signature drift.** | Grep/parse-only (no builds — the Plan 337 carve-out under the p335 T4.2f arm). **Route half:** mmorpg (root + `crates/riir-bevy` + `crates/riir-viewbridge-node` + `wasm/Cargo.toml`) — the only `katgpt_core` token in `.rs` is pet_teaching.rs:124, the doc comment RECORDING the fix; Cargo.toml mentions are comments + the BOUNDARY.md-declared `[patch]` git-unification entries; no `[dependencies]` katgpt-core entry survives `baab368`. riir-viewbridge + riir-dapps: zero `.rs` references, comment-only manifests. Facade-bound set confirmed EXACTLY mmorpg on disk — SDK-dep greps over all 11 workspace repos hit only comment text (riir-ai's riir-ffi feature-doc + riir-games issue-ref; viewbridge layout; dapps direction-rule); seal-online-remaster absent from disk, out of scope. **Signature half:** all 3 SDK→katgpt-core forwards item-complete against current source — `gm`→`prompt_backend::InferenceBackend` (always-on module, hoisted `c083423d` 08-04); `spectral_hero_gate`→`spectral_pencil{attribution,dense,genome,init,shape,field}`+`{DensePencil,SymPacked}` with `field`'s `committed_field_blend` gating matching; `hint_regret`→`{learnable_band_gate,triage,wilson_score_ci,Regime}` (all re-exported at hint_regret/mod.rs). Feature names all exist in katgpt-core's Cargo.toml. ZERO katgpt-core commits touching these modules since the facades were written (spectral last `278c9911` 08-22; hint_regret `1b73fbf1` 08-24) → no drift possible. Consumer usage fully covered: pet_teaching consumes `Regime`+`triage`; spectral consumers' 11 items across 5 submodules (`feature_influences`, `SpectralField8x4`, `seeded_dense`, `nsd_diagonal_feature`, `rank_one_feature`, …) all verified present; `NpcDecisionAttribution` is `riir_game_sdk::motivation` (riir-games substrate — outside the katgpt-core facade surface). The riir-games/shared facade half: zero riir-ai commits since 08-26 00:00 touch `riir-games`/`riir-games-shared` → the same-day consumer compile validation (full chain 2m51s + pet_teaching 15/0/3i through the facade) is CURRENT. |
| 2026-08-28 | Mode 2 drift audit — the post-08-26 fresh wave (queue item 7, zero-contention unit under a loaded box): ~55 fresh mmorpg files (gm_server/ccu_loadgen/authority progress_anchor+port_evict/pet_teaching_677/hero_signals/npc_jobs/karma_mirror/entity_inspector/pets_ui/game systems) + FULL riir-dao src (new repo, whole tree fresh) + riir-dapps kat/ expansion (staking/lease/governance/mining/rewards/telemetry/solana/stripe) + riir-chain (token/domain registries, batch verify/settlement) + riir-neuron-db (anchor_root/local_kv) | **1 new finding (filed mmorpg Issue 087, detection-only); discipline otherwise HOLDING — model consumers dominate the fresh wave** | The 5 grep families over the fresh sets. **Finding:** `pet_teaching_677.rs` (`#[cfg(test)]` Issue-677 PoC via `#[path]`) ships an inline `enum Regime` + `triage()` (~20 LOC, τ=0.5 consts baked) under the justification "Plan 576 is NOT landed" — now FALSE: `riir_game_sdk::hint_regret::{Regime, triage}` landed (Plan 576/Bench 576) and the PARENT `pet_teaching.rs` migrated to the facade 2026-08-24 (`Regime as FrontierRegime` alias + consts passed as `tau_r`/`tau_ret`). Identical semantics (same 3-branch two-threshold partition). Classification: minor test-only DRY violation + stale doc-truth claim → **mmorpg Issue 087** (RESOLVED same-day, mmorpg `fa4063e`: inline partition deleted — `issue677::triage` is now a τ-binding adapter over `riir_game_sdk::hint_regret::triage` via the parent's `FrontierRegime` alias; stale header claim corrected; G3 count-identical 3/0/1i; issue file removed). The rig's PROBE half (shadow states, K-pair CRN) is caller-owned by contract — correctly inline. **Model consumers (the discipline working):** `hero_signals.rs` consumes `game_traits::sigmoid` + `social::{karma_badness, SocialPolicyConfig}`; `npc_jobs.rs` consumes `motivation` cadence tiers + `zone::KgTriple` + `tick::System`; `karma_mirror.rs` consumes `game_sync::karma` + `social::*`; `pet_alarm.rs` continues the `GenericSpatialBelief`-shaped `last_known_pos` pattern; riir-dao `strategy.rs` consumes `katgpt_core::rating` + `hint_regret::{beta_lcb, beta_lcb_order_into}` per its DECLARED allowlist dep (correct — not a Class A reach; dao is not an SDK facade consumer); SDK `hint_regret.rs` facade signature re-verified against current source (gate exports unchanged, zero katgpt-core commits since 08-24). **False positives (documented classes):** `authority/mod.rs:1282` mean-radius over snapshot-POD `[f32;2]` fields (wire-DTO exempt); `static_data.rs:1505` AABB-closest-point geometry over config arrays (no MapPos3D operand, no point-to-AABB substrate); `scenarios/runner.rs:404` mixed MapPos3D × `[f32;2]` wire-array operands (mixed-operand exempt); `constants.rs:877` basin_heightfield grid-space math (the 08-17 documented class); `game/tests.rs` RGB color distance; `hero_routine.rs:2070` inline sigmoids INSIDE `mod tests` (tuning-pin test assertions — test-only class); `quest_combat_tests.rs` cluster-radius distribution; riir-chain `consensus/pipeline.rs` sqrt of variance→std-dev + `programs/cpi.rs:627` `checked_sqrt` AMM LP formula (statistical/financial, no spatial substrate operand). **Persistence note:** `monster_pet_debug.rs:214` D1 site (BOTH operands MapPos3D → delegatable; debug-panel cold path) from the 08-19 audit row is STILL OPEN — that row remains its record. **Out of scope this cycle:** riir-ai (155)/riir-train (104)/katgpt-rs (35) fresh files are substrate-side (the audit targets consumers duplicating substrate, not substrate growth); riir-viewbridge codegen/derive + `csharp/RiirGameNode.cs` fresh files are FFI-wall territory (boundary-guard G5, not substrate-first). Audit ran grep/parse-only against local trees whose only divergence from origin is sibling WIP disjoint from the audited files (riir-clippy test WIP + riir-dapps `src/kat/lease.rs` M + katgpt-rs untracked `certified_frontier.rs` + riir-ai bench-doc M). |
| 2026-08-29 | Mode 1 pre-implementation gate (recorded post-hoc at the Plan 581 finish pass) — twist_smc additions: `twist_cache.rs` (ValueMemo / RidgeTwistTable / select_beta_by_budget / X0ProxyReward) + the `RewardKind::Closure` row in distributional_steering | **CLEAN — all three concepts CONSUME existing substrate; the one new row is domain-justified** | Vocabulary translation (3+ variants per concept, grep over katgpt-core src): (1) **β/KL-budget selection** — "solve_beta / entropic_tilt / budget" → CONSUMED: `entropic_tilt::solve_beta` hoisted as `select_beta_by_budget` (T3.3 as planned; zero duplication). (2) **one-shot ridge readout** — "ridge / readout / normal equations" → CONSUMED: `linalg::ridge_solve` (the Cholesky house pattern; `RidgeTwistTable` is a thin fit+value wrapper). Other ridge-bearing modules (velocity_field_ensemble, closure, mean_field) have different fit surfaces — no reuse conflict. (3) **state-keyed value memo** — "memo / cache / lookup" → primitives CONSUMED (papaya lock-free map + BLAKE3 keys, the engram/hotswap conventions); ADJACENT SUBSTRATE EXAMINED: `mcts_state_action_cache::StateActionCache` (the closest keyed cache — keyed `(blake3(state), action) → (next_state, reward)`, an MCTS transition cache with no TTL/stale-first eviction and an MCTS-domain API) + `hippocampal_cache` + `engram/cache` + `tiered_kv` (different value domains); forcing the steering loop through the MCTS cache's `InferenceAction` contract would be a force-fit → a domain-specific `ValueMemo` row is justified; TTL + stale-first eviction are the new capability (persistent-agent reuse, plan T3.1). (4) **SMC resampling** — "resample / particle / systematic" → CONSUMED: `distributional_steering::{WeightedPopulation, systematic_resample_into}` + the new `twist_after_resample` ratio-chain restart on the same module's types. **Model consumers (the discipline working):** `X0ProxyReward` reuses the per-particle marginals consumers already produce (the plan's own premise); no parallel weight/resample machinery exists anywhere in the module. | 
| 2026-08-29 | Mode 2 drift audit — riir-deployer FULL tree (the unaudited fresh consumer repo: Plan 002 ops ladder landed 08-29 + the 13:20 cf-worker provider adapter + the NEW `riir-control` control-plane contract crate; queue item 7, zero-contention unit under the sibling's 809/810 quiet-box discriminator on M3) | **CLEAN — zero findings, zero issues filed. The zero-sibling-dep boundary + RAW-domain discipline are TEST-ENFORCED in the fresh code.** | Grep families over `crates/riir-deployer/src` + `crates/riir-control/src` (~5.6K LOC deployer + control). **Model consumers (the discipline working):** `sig.rs` — Ed25519 verify through `ed25519-dalek` (the workspace-standard crate, NO hand-rolled math), verify-only by construction (the doc: "the signing key NEVER appears here"); the hand-parsed pieces (30-LOC base64 + the 51-byte `ssh-ed25519` pubkey wire format) carry the documented smaller-surface rationale ("Zero `ssh-key` dep") + RFC 4648 test vectors + structural-parse negatives + a real sign/verify/tamper roundtrip test. `riir-control/src/lib.rs` states the domain discipline IN-HEADER: "The control plane is a **RAW domain** … No latent/sigmoid gating of decisions — modelless analytics may advise, never gate" — and the greps confirm: sigmoid/exp 0, distance/sqrt 0, decay/powf 0, RNG 0, spatial 0. **Cross-repo contracts pinned, not duplicated:** the Issue 080 asset-manifest consumer format is pinned by `artifact_manifest_parses_issue080_consumer_format_g3` (the SAME serde shape parses the real mmorpg consumer manifest and the deployer's) — producer-side pin, consumer-side validator lives in `riir_net::manifest` (the by-design split); `blake3_known_vector` + `zero_sibling_deps_g4` (walks resolved path deps, asserts containment — the hard boundary rule as a test) + `plan_json_is_bit_identical_across_runs_g1`. **False positives (documented classes):** `sha2` dep — required by the Issue 080 manifest format itself ("algo RECORDED": b3sum preferred, sha256 fallback → the deployer must VERIFY rows recorded with either); `envelope.rs`'s `{key_id, payload_json, sig_hex}` — pattern-BORROWED from the dao GovernanceSigner seam (verify-only, key custody ops-side) but a SELF-CONTAINED control-plane contract (the DO mutation door), NOT a mirror of riir-mcp-client's GmEnvelope (different plane) — no external producer exists yet, so no golden-pin obligation; **forward note:** a second producer of control envelopes outside this repo must add a golden-hex pin of the signing format (the dao `ban.rs` copy-gate precedent). Sibling-activity check before audit: deployer HEAD `79985ee` 13:22, tree clean, 1.5h quiet — safe. |

## References

- AGENTS.md §"Spatial Cognition (Two-Brain Model)" — the canonical perception rules
- AGENTS.md §"Manifold Geometry (Stokes Calculus)" — the R296 vocabulary-translation
  failure pattern (this skill extends it from paper→code to codebase→codebase)
- AGENTS.md §"Latent vs Raw Space Rules" — domain classification
- `riir-games-shared/src/game_traits/spatial.rs` — `GenericSpatialBelief<T>` substrate
- `riir-games-shared/src/game_traits/` — vocabulary translation table source
- Issue 047 (riir-mmorpg-examples) — the ThreatField canonical failure
- Issue 490 + Issue 493 (riir-game-sdk) — the orchard/motivation canonical failure
- `boundary-guard` skill — sibling (boundary enforcement, post-hoc)
