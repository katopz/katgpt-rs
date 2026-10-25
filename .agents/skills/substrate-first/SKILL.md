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
# 7-repo brace list this replaced could not see 13 of the 20 contract repos,
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
the 20 contract repos. That is the same defect the vocabulary step above was
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

**Compacted 2026-09-11 (user-directed verbose-history pass; the file was 74 KB).**

**Pruned to the newest 15 rows 2026-10-12** (the 81920B skill-size ceiling; 39 older rows dropped, and two 10-12 rows that had been mis-inserted into the Mode-1 vocabulary table moved back where they belong). `git log -p -- .agents/skills/substrate-first/SKILL.md` recovers every dropped row verbatim.
Rows carried full verification narratives until this compaction; `git log -p -- .agents/skills/substrate-first/SKILL.md` recovers any of them verbatim (`git log -S '<date>' -- <this file>` for one row). New rows append ONE line each.

### Standing lessons (distilled from the compacted rows — load-bearing audit classes)

- **Body-read before filing a sigmoid/def hit.** A local `fn sigmoid` that DELEGATES to `katgpt_core` (e.g. a ±50-clamp wrapper) is the SANCTIONED delegation pattern (ndb Issue 611's own fix shape; riir-auth `session.rs`; chain `runtime.rs`) — a definition-grep hit is a lead, not a finding. Corollary: **`katgpt_core::sigmoid` is Cephes (~1 ULP), NOT bit-identical to libm-`exp` forms** — delegation is a semantic change where bit-exactness is committed (freeze-commitment paths carry a version-boundary caveat).
- **Documented false-positive classes for distance/sqrt greps:** squared-distance comparisons (`dx*dx+dy*dy <= r*r` — the CORRECT no-sqrt idiom; "fixing" it is a pessimization); wire/snapshot `[f32;N]` DTO operands (no MapPos3D to delegate); grid-space heightfield math; statistical/financial sqrt (std-dev, AMM LP `checked_sqrt`); disc radial sampling (`r*sqrt(u)`); screen-space view-plane placement; `#[cfg(test)]` sites; closed-form test oracles (PSD max-eigenvalue).
- **Copy-gate convention:** a justified copy needs in-source rationale + a divergence-failing test (chain Issue 139 — a doc-only mirror with only self-consistency tests is NOT enough). Documented copy families: `BLAKE3(pubkey)[..16]` (5 gated instances — consolidate the 6th onto `riir_wallet_signer::player_id_hash`); splitmix64 finalizer (~20 substrate-side module-locals, all bit-identical — consolidate at the 4th CONSUMER-side arrival, currently 1); cross-repo message formats single-source by IMPORT (dapps imports chain's canonical `signing_message`).
- **Feature-isolated local math is REQUIRED, not drift** — when the feature pulls no katgpt-core (chain forensic f64, congestion inline, the lora bridge keeping katgpt out of the Docker context). Production-dead surface carrying latent duplication under an available dep → delegate-or-prune when consumers materialize (curator_bridge class).
- **Translate CRATE names too, not just type/fn names** — `mmorpg` vs `riir-games-mmorpg` missed a substrate consumer the issue itself cited (Issue 870: the proposed additive `StatMods` was refuted by the EXISTING clamped `StatModifier`, Plan 411).
- **`$(var)` executes a command named by the variable** — the repo-set helper must be a shell FUNCTION called as `$(consumers)`, or the grep silently scans the whole tree.
- **Sanctioned model-consumer shapes (not drift):** injected-verifier traits (GmSignatureVerifier pattern); parameterized general form + env-toggle special case (quest_center "one filter home"); feature-neutral placement for gate-union decoupling (HeroStats); `Arc<AtomicU8>` config seams; two-brain occlusion divergence between substrate and chain verifier (AOI — symmetry pinned on the CHAIN type); plane-split crypto with a hard `pub(crate)` bar + documented nonce-tradeoff rationale (ndb CLI seal/unseal).

| Date | Scope | Verdict | Record |
|---|---|---|---|
| 2026-10-26 | Mode 2 mini-audit — the post-25b delta (amo idle lane; fresh .rs across 5 repos: dapps 6 [the kat_migrate lane: migrate.rs + migrate_sweep.rs + tests + the worker's migrate_contract/migrate_routes], auth 2 [github_pow.rs + measure example], kat 2 [kat_protocol_migrate + crosswalk test], sdk 1 [aoi_fanout_roster_goat bench]; modified: kat-service worker files, auth account_key/bridge, kat_protocol_freeze, sdk aoi_fanout_curator, reflexer vessel env/lib, katgpt-assign trio; signature greps + body-reads; zero-cargo) | **CLEAN — ZERO FILINGS**: dapps `[..16]` ×3 = the documented derivation families with the formula in each doc-comment (acct16 "one derivation, two widths"; reward_key row-key; mint_receipt_id "the same derivation the mint plane pins" — cross-plane single-source by re-derivation, pin named in-source); github_pow = pure fetch/verify module, one `blake3::hash` over canonical serialization (the digest family, formula in-doc), threshold consts not math; aoi curator/bench + katgpt-assign + reflexer vessel = zero signature hits; zero new_v4/splitmix/softmax/Belief-struct/spatial-HashMap/RNG arrivals; tests+corpus = fixture class | katgpt-rs (this commit) |
| 2026-10-25b | Mode 2 mini-audit — the post-25a delta (the 4090 idle lane's own sync window, 3 fresh .rs + 1 py across 1 repo: reflex `src/lanes/elephant.rs` NEW [412 lines], `src/harness/runner.rs` + `src/bin/harness.rs` mod, `scripts/elephant_server.py` NEW; the rethink-032-T10.2 mammoth/elephant comparison-lane wave `1e6d4db..f60923a`; signature greps + body-reads; zero-cargo, sibling measurement active) | **CLEAN — ZERO FILINGS**: elephant.rs = pure HTTP wire lane (env consts + post/get + serde parse + `determinism_view` clock-strip) with ZERO numeric-decision surface — no sigmoid/softmax/exp/hash/seed/distance; the `TeacherWireLane` impl = textbook glue over the EXISTING lane trait (the mammoth-family shape, `spend_env`/`MAX_CASES_ENV` spend-ceiling contract carried); runner.rs softmax ×3 = doc-comments DESCRIBING the elephant/bekko lanes' own readouts (never implemented here), `distance_only_*` = pre-existing gate-floor test fn names, blake3 mentions = gating doc-comments; harness.rs `&digest[..16]` = the corpus-row blake3 DISPLAY seal in a log line ("blake3 {}…" ellipsis literal — the display-seal family, not a truncation decision); elephant_server.py = zero signature hits (reference qev serve plumbing); zero new_v4/splitmix/Belief-struct/spatial-HashMap/Mutex/emotion arrivals | katgpt-rs (this commit) |
| 2026-10-25 | Mode 2 mini-audit — the post-24c delta (anchor `9d27c7c56` end, 10-24 ~23:1x +07 → 10-25; 2 fresh .rs across 2 repos: ai 1 [riir-gpu `bench_1122_site_tax_and_ktile64` — the Issue-1122 bench vehicle, sibling lane, READ-ONLY as-found], remake 1 [`canvas_draw_placed` — the plan-412 3d-iii placed-rows lane]; mod-set 17 across the same 3 repos: render 4 [scene_set/skinned/zone_scene/assemble], remake 12 [the canvas_draw family + authored_life_view/dressing_rig/head_look/meadow_ground/quest_talk_view/vessel_boot/vessel_scene], ai 1 [riir-gpu lib]; signature greps + body-reads; zero-cargo) | **CLEAN — ZERO FILINGS; the wave is a CONSOLIDATION, not duplication**: seal-scene `scene_set.rs` carries the single-home law in its header ("the editor's model viewport lays its scenes out through it today; the RUN viewer will, so both lay the same parts out alike") and is consumed by seal-render lib/assemble/zone_scene AND sge's `model_3d_view` — the plan-412 single-renderer law as shared-home birth; `skinned.rs` likewise ("one implementation of skinning… fixed operation order, so posed output cannot shift" — Issue-940-style determinism in-source); remake `canvas_draw_placed.rs` = TEXTBOOK consume (`scene_set::{step_animated, resources_worth_posing, PoseSink}` + `assemble::scene_skin` + `LandOn` wholesale; the idle "by the map page's own sampler"; the giver's gaze arrives as `PlacedGaze` from head_look's verdict, never recomputed). REST = documented classes: head_look `1−exp(−k·dt)` = view-layer ease (seal-view no-dep class) with the in-source router comment sending the placed turn to the canvas; meadow_ground `distance3` [f32;3] + skinned quaternion-normalize/segment sqrts = the geometric class; HashMaps = id/class-keyed fade/joint/land state (zero coordinate-keyed); Mutexes = the app-thread sim-state handles (papaya N/A); skinned `[..8]` = a test-array slice assert (not a hash truncation); bench_1122 = pure harness, zero RNG/seed/sigmoid surface; zero new_v4/splitmix/softmax/belief-struct arrivals. ALSO REPAIRED (this commit): the run-log table itself — the `|---|` separator had DRIFTED into the middle of the table (13 newest rows above it rendering as plain text — the append-above-the-separator drift class, 16 rows accreted since the 10-12 prune) — separator restored under the header, tail pruned to the newest 15 rows per the documented maintenance rule; pruned rows recoverable via `git log -p` | katgpt-rs (this commit) |
| 2026-10-24c | Mode 2 — the post-24b delta wave (anchor `9d27c7c56`, 10-24 16:00+07 → ~23:1x; 147 fresh .rs across 12 repos: sge 41 [the model_facing arc: editor-core probe + gizmo + seeds + store tables + run_reach/run_spawn_models + feat-map fire/lava/waterfall/relief], remake 32 [canvas_draw host/effects/water + vessel_scene/map_stream/chunk_stream + eye_order/load_lint/lod_payload/scheduler gate tests], train 14 [bench-algo reasoner + vq_codebook backprop arms — READ-ONLY, routing-restricted], infer 11 [prefill_cuda_full + gemm mma + laya agent/stability + plan622 g1], refine 11 [kernel_opt corpus entries_gpu_101-119 — fixture class], sdk 8 [riir-vessel cas/chunk_manifest/pack/codec — the ndb-632 wire lane's landed state], render 8 [skill_fx canvas + waterfall/ice + scene zone vocab], katgpt-rs 5 [kv960 T2 alloc landing + its gate tests], ai 4 [bench_1113/656/809 test lanes + gpu lib], ndb 4 [artb + vessel query_hash], reflex 5 [elephant lane + distill + harness], seal-std 4 [seal-fx effect sim/billboard/classify]; signature greps + body-reads; zero-cargo) | **CLEAN — ZERO FILINGS; the 24b row's WIP flag RESOLVED**: katgpt-rs `kv_sensitivity_alloc.rs` LANDED as substrate-side COMPOSE — a behavioral-damage weight into the SHIPPED `waterfill_bits`/`BitAllocator` allocator ("no new search, no new quantizer: one weight vector"; T1's negative-control caveat in-doc) — the extend-via-input form, not a parallel allocator. TEXTBOOK consumes: sge `run_reach.rs` delegates the reach verdict to `riir_hunt_reach::AimReach` wholesale ("the answer is not computed here", :331 sqrt = display distance); the seal-fx single-source-by-leaf law HOLDS through the wave — seal-render `bevy_canvas_skill_fx` + seal-view `{skill_fx,fx_player,buff_fx}` all consume `seal_fx::{effect,skill_fx,buff_fx,goldens}`, zero copy arrivals; sdk riir-vessel cas/chunk_manifest = blake3 content-addressing over the artb wire (the 632 wire lane posture re-verified). REST = documented classes: infer gemm_mma golden-ratio LCGs (8 sites, all inside `#[cfg(test)] mod tests` @5197) + train train.rs:1185 xorshift seed (also `#[cfg(test)]`) = the fixture-RNG class; train carry.rs:193 `momentum_beta` σ(b) = the bench reasoner's model-forward parameterization (riir-bench-algo carries NO katgpt-core dep — only optional katgpt-pruners behind carry_parity); train model.rs `softmax_inplace` = model-forward; ndb vessel `query_hash` [..8] = cache-key derivation with perf rationale in-source (~80ns vs ~1100ns hit); reflex distill/harness [..16] = display seals; sge build_lane [..8] = warning-text display; remake gate tests hash[..8] = fixture stuffing; HashMaps = id/texture/DB-row/tile-cache keys (zero coordinate-keyed perception); Mutexes = infer prefill GPU buffer/state mirrors (resource handles, papaya N/A) + laya `PACKED_ACT_BITS` = the documented parity-seam diagnostic ring (cap 4096, aliasing-hazard story in-doc); zero new_v4 / belief-struct / emotion-struct / non-model softmax arrivals | katgpt-rs (this commit) |
| 2026-10-24b | Mode 2 mini-audit — the post-anchor delta only (10-24 ~09:00 → ~16:10; 75 touched .rs across 9 repos, 12 added: katgpt-rs 2 [kv960 T0a/T0b kv_sensitivity + capture], ai 2 [hero_cast + tests], refine 3 [multivector kill-test + router_per_span_dump], train 1 [the 648 pin test — the morning filing's fix half], remake 4 [eye_order + load_lint + g1 tests]; mod-set: sge 25, remake 8, render 8, ai 8, refine 4, katgpt-rs 4, train 2, seal-std 3, reflex 1; signature greps + body-reads; zero-cargo under 5 live sibling lanes) | **CLEAN — ZERO FILINGS; 1 PRIOR MENU DISCHARGED**: reflex `harness/runner/router.rs:111` — the 10-15 NOTE-LEVEL inline f64 sigmoid now DELEGATES `exact_sigmoid_f64` with full provenance in-source (the 014 engine.rs precedent + this audit's 10-15 row + katgpt-rs `bbc28a091` cited) + the frozen-legacy-body pin test (`sigmoid_delegation_matches_frozen_legacy_body`: bit-identical x≥0, ULP-bounded below, [−120,120] sweep, the legacy body frozen in-test) — the copy-gate done end-to-end from note to fix. TEXTBOOK: kv960 `kv_sensitivity.rs:738` splitmix = the `#[cfg(test)]` FIXTURE class (in-source "No external RNG dep; same seed → same stream" + the Sterbenz-exact bimodal fixture); refine `multivector.rs` = modelless kill-test consuming `katgpt_core::simd::maxsim_score` (substrate wholesale), its `[..16]` corpus_digest = the frontier_miner short-digest family; train 648 pin test = the expected delegation-pin landing (same-day filing → same-day fix). REST = documented classes: ai constants.rs configured-point sqrt + knockback falloff (the plan-032 lane), sge feat-map/skinned_model/walk_dots geometric [f32;3]/xz math, remake viewer/lib.rs splitmix doc-comments = the landed Issue-057 consolidation verified holding ("the view layer's ONE splitmix64 draw"), render bevy_canvas/grass view-layer geometry, forward 1/√d + rms = model-forward, reflex router `[..16]` = the routed-artifact seal display id; remake eye_order/load_lint + ai hero_cast = zero signature arrivals (glue lanes over existing vocabulary); kv960 sibling T2 WIP (kv_sensitivity_alloc.rs untracked) measured as-found, unaudited per detection-only | katgpt-rs (this commit) |
| 2026-10-24 | Mode 2 — the 10-22 10:46+07 → 10-24 ~09:00 delta wave (anchor `42712e58c`; 200 fresh .rs across 13 repos: remake 65 [the 052 quest-server arc + p2p lanes + canvas_draw host], sge 23 [run_mint + body_measure/footfall/trail lanes + legacy_crit/skill seeds], render 8 [assemble/presence/landfield — post-carve arrivals], train 31 [plans 459-461 + damping_probe + elbo_boundary + hybrid_link + vela — read-only], ai 14 [sst_gate + quest_state/effects + strike_law + 1113/1114 benches], infer 15 [Issue-959 golden_nll + tokengate + plan622 keybias gates], ndb 15 [scope lanes + item_instance — the 052 mint arc], katgpt-rs 9 [sigmoid_budget + softmax_fix + sst_certified + lattice_reopen], seal-std 9 [seal-combat + seal-item + seal-fx move_fx/ui_fx/zone_music], sdk 6 [vessel lod/chunk_manifest + e2e consume_chaos], reflex 2 [clef_conf_probe + depth_probe], rethink 1 [enc_g4_alloc], instinct 2 [tetris_cont]; signature greps + body-reads; zero-cargo under 4 live sibling lanes) | **1 FILED: riir-train Issue 648** — `damping_probe/mixer.rs:135` `pub(crate) fn sigmoid(f32)` restates the plain-libm body with katgpt-core REACHABLE (hard dep, engram_privilege — the exact posture that made 636 a filing), zero rationale/pin; f32 flavor so the 636 arm-(a) delegation is NOT zero-behavior (fast_sigmoid = Cephes ~1 ULP) — menu delegate-stated | recorded-refusal + twin-pin, filed detection-only (`cc7885d5`). NOTE-LEVEL ×3: seal-item `Draws` splitmix = the seal-fx shape missing its ONE-LINE pointer comment (seal-fx sim.rs carries the copy-gate doc done right — "copied because this crate is a leaf… keep bit-identical" — the twin site needs the same line); ndb `item_instance.rs:47/:229` derivation prose omits the `INSTANCE_ID_CONTEXT` prefix the code applies (seal-item's InstanceId is OPAQUE — "128 bits its minter chose" — so no wire bug; the doc formula is just short one hash input); instinct `cont_seed` = recipe-published posture (the export bin's `piece_seed` string IS the cross-medium pin). TEXTBOOK: `sigmoid_budget.rs` = substrate-side birth delegating fast_sigmoid with measured laws in-doc; golden_nll delegates `fixed_log_softmax_target_f64` (959 law in-source); binary_kernel/elbo softmax_row = model-forward/training class; scope_counter `Mutex<()>` = textbook RMW write-latch (race named + chaos test cited, papaya rule N/A — not a map); hero_live/seal-view Mutexes = option cells + app-thread queues (papaya N/A); ndb scope/item_instance `[..16]` = lane-id/instance-id derivation family; quest-server nonce `[..16]` = the request-nonce family; body_measure/gizmo/lod/billboard sqrt = geometric vector math over raw [f32;N]; value_replay_log splitmix = the synthetic FIXTURE class ("a KNOWN answer" fixture, in-doc); softmax_fix/sst_certified = the substrate homes themselves; zero new_v4 / belief-struct / spatial-HashMap / emotion-struct arrivals | train `cc7885d5` filing + katgpt-rs (this commit) |
| 2026-10-22 | Mode 2 — the 10-21 00:34+07 → 10-22 delta wave (anchor `882e5b3a3`; 163 fresh .rs listed / 152 body-present after the corpus-fixture class exclusion, across 10 canonical repos: seal-render 87 [the plan-412 canvas carve], sge 17 [the wave's sge half: hit_sound/skill_book/zone_description lanes], ai 26 [the 639 fog lane + the 1097/1099/1102 combat splits + the composition/* reorg], katgpt-rs 7 [delta_mem/rls.rs Plan-630 exact-RLS rung + katgpt-extract bet_search], refine 11 [corpus fixtures], infer 3 [Issue-1106 stability readout], reflex 2 [modal_floor + startlux], rethink 2 [remote_lane v1 client], seal-std 5 [seal-ability/seal-skill books + seal-fx sim], remake 4 [fishing/job_skills/locomotion_config]; signature greps + body-reads; zero-cargo under the sibling sge merge gate at load ~84 on battery) | **CLEAN — ZERO FILINGS; the headline is the carve: the plan-412 move executed the 231 splitmix-home consolidation CLEANLY across the repo boundary — seal-scene `look/seed.rs` is the single production home (fire/lava/waterfall consume `salted_placement_seed` with their documented salts), sge's copies are GONE (bevy_canvas_butterfly deleted; editor-types re-exports `seal_scene::{… look …}` wholesale so `editor_types::look::seed` still resolves), and the two sanctioned singleton-locals (editor-core services.rs `sample_blake3`, editor-data random_equip) carry UPDATED doc-comments naming the new home + the keep-bit-identical copy-gate line — a copy-gate family that survived a cross-repo move intact. TEXTBOOK consumes: the 639 fog lane `hero_threat_memory` delegates `game_traits::spatial::{belief_confidence, MIN_CONFIDENCE}` (the one-law citation in-source) and `HeroItemMemory` stores raw (pos, seen_tick) with read-time substrate decay — zero restated decay formulas; stability.rs = the new calibration-only instrument stating the sequential-f32-fold Issue-940 law in-source + brute-force parity gates (its sqrt = cosine/L2 instrument math, structurally forced: riir-infer-laya carries no katgpt dep); rls.rs = SUBSTRATE-SIDE birth (the exact-RLS rung, Plan 630, naming its δ-rule + ridge_solve ladder siblings in-doc — the ladder, not duplication); modal_floor.rs = the integer-counts census (pool → global → case-order tie law, Issue-940 compliance in-source); lava inline σ + fog `1−exp(−k²)` = the documented structurally-forced / pinned-transcription classes moving with their files; cognition_steer dx·dx+dy·dy = delta-LENGTH lookahead normalization (the spin_break no-accessor class); fishing_journal radial sqrt = a view-layer display string; zero new_v4 / belief-struct / spatial-HashMap / softmax / blake3-trunc arrivals | katgpt-rs (this commit) |
| 2026-10-21 | Mode 2 — the 10-20 ~04:00 → 10-21 delta wave (117 fresh .rs across 14 repos + ~380 mod-set retouches: katgpt-rs 22/24 [kron_memory substrate birth + the NEW katgpt-extract crate], ai 13/56 [command_lane/command_grammar/fishing/spin_break/survival_stoop/skill_forces/breadcrumb_trail], sge 56/194 [plans 393-405: quest_generate/quest_search/quest_pack/shops/combat-power/level-points/sound/fish/net-selector], seal-std 12 [the seal-quest/shop/swim/stat/formula leaf wave], remake 6/102 [fish_cruise/dev_chat/capture_clock/quest_selection/skill_demo/panel_fallback], refine 5 [corpus fixture + scan_prune], train 2 [bench-algo — routing-restricted, read-only], ndb 1 [riir-rag math seam], dapps/llm 1 each; signature greps + body-reads; zero-cargo) | **1 FILED: seal-game-editor Issue 245** — editor-data carries TWO identical `pub fn sigmoid(f64)` (quest_generate/pick.rs:43 + quest_search/types.rs:289, byte-identical bodies, zero rationale, both pub) while `58b6fe90b` already built the shared `lexical.rs` seam for exactly this module pair — menu = one home in the shared module + re-export + a same-body copy-gate test, zero new edges; intra-crate DRY (the 057/231 genus), structurally forced at repo level (editor admits no katgpt package, Plan 393 T1.2) so NOT delegation. REST CLEAN — **textbook single-source-by-leaf**: seal-swim `SeededLcg`/`FishCruise`/`Water`/`MODEL_FORWARD` consumed wholesale by BOTH renderers (seal-view fish_cruise + editor-widgets swim_apply — the duplicated fish-swim CONVERTED to the shared zero-dep leaf this wave, the seal-fx strongest form, first-draw constants pinned in-leaf); ndb riir-rag/math.rs = the copy-gate done RIGHT (the katgpt_cmp OFF-arm transcription + a ~4.2M-point bit-identity mirror test compiled only when both arms exist, the embedder's model-id promise cited in-source); ai spin_break = MapPos3D::distance_2d_to consumer (its `unit()` helper outputs raw [f32;2] bearings — no accessor exists, documented class); survival_stoop sqrt = test oracle; sge llm.rs [..16] = the echo-token display family; school.rs `HashMap<(i32,i32),Vec<usize>>` = one-shot scatter-time bucket grid (build-time pool ordering, not per-tick perception); katgpt-extract temper.rs sigmoid = RECORDED-REFUSAL rationale in-source (µs-path not ns-path, platform-portable plain exp) + span_pointer.rs module-local = substrate-side same-crate, NOTE-LEVEL one-line menu (cite temper's rationale or delegate); kron_memory LCGs = the unseeded-global-RNG gate convention; train carry.rs softmax = model-forward, Box-Muller = the in-crate RNG class. MOD-SET: editor-core effect.rs:146 VERIFIES the 231 menu LANDED (delegates `editor_types::look::seed::splitmix64_finalize`); dev_vibe:485/locomotion_gait:66 = the documented seal-view no-dep class; hero_routine det_trig delegates (1057 holding); hero_progress_drive consumes the designated luck.rs home (1044 holding) | sge `245` filing + katgpt-rs (this commit) |
| 2026-10-19 | Mode 2 — the post-18d delta wave (anchor 29f37db2c, 10-18 11:28+07; 30 fresh .rs across 6 repos: seal-std 12 [the NEW seal-fx leaf, seal-std issue 001 — schema/curve/classify/quads/goldens], sge 7 [fx_export writer + warp_lint tool + skill_projectiles + look/motion + canvas motion tests + 2 examples], remake 5 [the fx reader arc: fx_player/skill_fx/clip_fx/buff_fx/audio], katgpt-rs 2 [fdr_credit Terminal-RSI + spr_order], ai 3 [buffs.rs plan-033 slice B + pin_gate_tests + wkg_inspect example], dapps 3 [sponsor + stake ingest/routes]; signature greps + body-reads; zero-cargo under 2 live sibling suites) | **CLEAN — zero filings; the wave's centerpiece is the STRONGEST single-source-by-IMPORT form on record:** seal-fx = canonical leaf (curve.rs header carries the port rationale + the bit-identical-both-players premise), editor-core::effect + editor-release::fx_export + seal-view {fx_player,skill_fx,clip_fx,buff_fx} all RE-EXPORT/import seal_fx types (editor-widgets consumes transitively via editor_core — no leaf dep needed), goldens (SKILL_FX_WIRE/CLIP_FX_WIRE/BUFF_FX_WIRE) pinned IN the leaf and parsed by BOTH halves' tests — no copy exists to diverge, stronger than a divergence pin (the 232/233 class PREVENTED, not cured). TEXTBOOK consumes: buffs.rs = `swarm::luck::seeded_hash` designated home + substrate `Hp/Position/is_dead` + determinism laws in-source (the 1044 fix holding through its first consumer wave); pin_gate_tests = MapPos `distance_2d_to` accessor; NEW substrate-side operators, not duplicates: katgpt-pruners `mm_squash` = Michaelis-Menten reciprocal form (P/(P+D') — saturating, NOT logistic; no katgpt-core restatement found, substrate-home birth, house sigmoid-rule compatibility cited), `fdr_credit.rs` cites the percentile tail-support floor in-source (audit culture compounding); dapps sponsor.rs [..16] = the 81-byte wire codec AccountId family; audio.rs linear-over-sigmoid = in-source rationale (two-anchor contract). Zero splitmix/new_v4/belief-struct/spatial-HashMap/emotion arrivals; editor-core's 1-line lerp = documented cross-reference to the seal_fx private copy, below the gate | katgpt-rs (this commit) |
| 2026-10-18d | Mode 2 — the post-18c delta wave (41 unaudited fresh .rs across 8 repos + 2 excluded-by-class: jup-tool 2 [the 4cf74d2 attribute-only posture], ai 10 [the 1091 spawn-HP + pvp-team tails], dao 2 [the landed Proposal-009 orphan], dapps 8 [the Proposal-009 TUNA airdrop trio + explorer views], mmorpg 2 [the 112/113 fixes], train 10 [plan410 stage0 + discoloop POC — READ-ONLY, routing-restricted], remake 3 + sge 3 [the plan-390 molten twin wave M9/M11 + the reach-cap fix]; katgpt-rs 223 = the 947 comment-move docs commits [zero semantic arrivals — removed non-comment lines = 3 markdown lines], riir-kat 35 = a whole-crate rustfmt round-trip [byte-identical, fixture class]; signature greps + body-reads; zero-cargo under sibling builds) | **1 FILED: seal-game-editor Issue 233** — the vessel-lava wire literal pin is ONE-SIDED: the viewer's `the_wire_lava_reads_as_the_editor_pressed_it` freezes the full WIRE JSON (uv_per_tile 0.5, warm_share 0.35, embers 36, limelight 8/20/4, glow cap 10.0) but the EDITOR side asserts no literal (its 7 tests = round-trips + const-vs-const) — an editor-types retune reds nothing anywhere while silently staling the viewer's frozen press; the plan-300 both-repos law half-satisfied; the chain-139 class third instance in this lane family (232 flow-turn bodies, remake 102 fade constants, 233 wire literals); menu = the editor twin test. VERIFIED-STILL-VALID: remake Issue 102 unchanged by the wave's re-touch (viewer FADE_GAIN 8.0 / FADE_STEPS 32.0 no-snap vs the editor's 12/16 end-snapped — the filing's table stands). REST CLEAN — textbook consumes: hero_progress_drive = `game_traits::sigmoid` delegate ×3 + the `luck::{seeded_hash, splitmix64_finalize, SPLITMIX64_GAMMA}` designated home (the 1044 fix holding); distill_tsd:1094 = a closed-form test oracle (`test_row_confidence_matches_commit_arithmetic`); discoloop Box-Muller/RMSNorm/Adam/1-√n sqrt = model-forward + POC in-crate RNG with in-source rationale; meadow_lava:283 inline σ = structurally forced (no katgpt dep, the documented seal-view class) + the 102-covered surface; the plan-390 reach cap = the CORRECT twin shape (`reach_cap twin`, asserted both sides); meadow_lava `HashMap<(LavaPart,u32)>` = a render mesh cache; dapps [..16] = the doc-comment display family; belief/emotion/new_v4 = zero arrivals | sge `67727d54` filing + katgpt-rs (this commit) |
| 2026-10-20a | Mode 2 wave audit (10-18 12:00 → 10-20 ~04:0x UTC; union-of-touched .rs via git log across the 9 workspace repos: katgpt-rs 18 / ai 1444 / train 711 / refine 55 / infer 125 / instinct 2 = ~2355 files, the bulk = the 2026-10-19 07:58 FLEET FMT SWEEP + the ai-1095 soak lane + refine B259-B264 corpus; signature greps softmax / new_v4() / fn sigmoid / SpatialBelief-family / coordinate-keyed HashMap over the touched sets, body-reads on every hit; zero-cargo, StartLux resident) | **CLEAN — ZERO FILINGS**: new_v4 hits ×3 = doc-comments CITING the law (pet_roster / motivation types / entity types); engine sigmoid defs ×3 (limelight:111 / arbitration:433 / thermal_lod:408) = the ai-1069 dispositioned NaN-guard + fast_sigmoid DELEGATION wrapper shape verbatim; katgpt-core lib.rs = the substrate home itself; infer elementwise sigmoid_f32 + clef softmax mentions = fmt-sweep-touched pre-existing + doc-comments describing the REFERENCE model's internals; SpatialBelief hits = the known ai-1051 census entry + TEXTBOOK GenericSpatialBelief<T> consumes (flee_steering/emotion); HashMap-spatial hits = id→pos value maps (u32 keys), not coordinate-keyed; corpus entries = fixture text. NOTABLE WAVE CONTENT (no drift): the fmt sweep normalized the fleet (line-ending + style only, blame-verified on the new_v4 sites), B261-B264 landed kernel_opt corpus, ai-1095/plan-638 lanes | katgpt-rs (this commit) |
| 2026-10-18c | Mode 2 mini-audit — the post-18b delta (10-17 21:00Z → 10-18 09:3x: 9 fresh .rs across 4 repos: ai 3 [crowd_state_windows T1–T4 files], refine 2 [B249 slice2/3 corpus fixture class], sge 2 [plan-390 M3b/M4 canvas lava-flow + molten-rig], remake 2 [M9 RUN viewer twins meadow_lava_flow + molten_rig]; signature greps + body-reads; zero-cargo under the plan-436 GPU chain) | **1 FILED: seal-game-editor Issue 232** — the plan-390 M3b lava flow-turn landed as a CROSS-REPO TWIN PAIR with zero divergence pins: editor `RIIR_DRIFT`/`RIIR_LAVA_FRAGMENT`/`flow_turn` body/26-line `FLOW_LAVA_VERTEX_WGSL` vs viewer `meadow_lava_flow.rs` + `shaders/flow_lava.wgsl` verified BODY-IDENTICAL this audit, but each side's tests assert only its own body (self-consistency) — the chain-139 copy-gate law (acknowledgment present, pin absent; the seal-remake 102 shape one layer up, pre-drift). Structurally forced (seal-view has no editor-widgets dep — separate deploy surfaces), so menu = per-repo byte-literal pin tests, NOT delegation. REST CLEAN: state_windows.rs = TEXTBOOK consume re-verified on the full body (katgpt_core::event_state_window + habituation_filter::settling_ticks wholesale; det_trig::ln64/exp delegates; EmotionField read-only accessor; zero alloc scratch contracts in-source; the 18b row's verdict stands); molten_rig twins (editor canvas vs viewer rig) = the WGSL-is-the-source design with the viewer's literal-quoted pinned contract tests + deliberate rig-limit divergence — sanctioned, not the 232 class; quantise ×3 private module-locals (canvas fire/molten + remake meadow_fire) = pre-231-census family, below every gate (single-copy-per-repo, distinct steps args); B249 corpus = fixture class | sge `232` filing + katgpt-rs (this commit) |
| 2026-10-18b | Mode 2 mini-audit — the post-row delta only (since 21:00Z: ai 4 [crowd_state_windows T1–T4 landing + G4 gate fix — self-audited at landing via the ai-1081 substrate-check block], sge 11 [plan-356 B2/B3/B4 motion lanes: look/fire|lava|waterfall|pulse + model_tags + motion + run_outfit], ndb 1 [art_bundle heal]; signature greps + body-reads; zero-cargo under sibling builds) | **1 FILED: seal-game-editor Issue 231** — the splitmix64 finalizer census CROSSED the per-repo threshold: 7 private module-locals across all 4 base crates (look/fire+lava+waterfall `placement_seed` ×3 with deliberate per-lane salts, editor-data random_equip `splitmix64`, editor-core services `sample_blake3` + effect `draw`, editor-widgets leaves `mix`) — each single copy structurally forced when written (no katgpt dep), the family accumulated one plan at a time past the 4th-arrival consolidation gate; menu = one editor-types home collapsing the look trio (zero new edges) + singleton crates stay local per their dep laws + copy-gate test on the landing. phases.rs Fibonacci hashing + dance.rs u32 GOLDEN = different mixer families, already single-homed. REST CLEAN: crowd_state_windows = TEXTBOOK consume (katgpt_core::event_state_window primitive wholesale + det_trig::exp test oracle + swarm math sigmoid re-export); pulse/fire look lanes = view-layer sine/dance-phase; model_tags HashMaps = counts caches; art_bundle blake3 = the substrate hash family itself; sibling's fire.rs WIP (plan-356 B2 flameless profiles) verified non-overlapping before filing (yield check) | sge `ca4ffa79` filing + katgpt-rs (this commit) |
| 2026-10-18 | Mode 2 — the 10-16 08:00 → 10-18 delta wave (128 fresh .rs across 13 repos: train 18 [vq_codebook arc + plan445 benches + drive_constants_train], editor 50 [lava/relief/sound/skill-script lanes], remake 13 [meadow_lava/relief/warp/spawn_focus], ai 12 [hero_no_gain + steer_avoid + spawn_place/target_assign + cluster_rebalance + wanderlust tests], infer 9 [riir-infer-clef lane], katgpt-rs 9 [qrr arc + approx + assign explain], dapps 3 + kat 2 + refine 2 [tuna airdrop trio], instinct 4 [eq9], shader 4 [lava gamefx], reflex 1 [gate_drift], seal-std 1; signature greps + body-reads; zero-cargo under Plan-337) | **2 FILED: riir-train Issue 636** — drive_constants_train.rs:199 `sigmoid64` restates `katgpt_core::exact_sigmoid_f64` BIT-IDENTICALLY (same two-branch stable form, binding name only differs) with the home REACHABLE (katgpt-core = hard workspace dep, engram_privilege) + house-law comment present but NO delegation refusal named + NO divergence pin = the 163 shape (copy-gate, NOT the 002 numerics class — delegation zero-behavior, verdicts replay through hard_surface); menu delegate | recorded-refusal + twin-pin. **seal-remake Issue 102** — the lava fade twins carry NO literal pin: editor ring_fade σ(12·(edge−d)/width)/steps16/end-snapped vs meadow_lava part_fade σ(8·(edge−d)/width)×presence/steps32/no-snap/presence-pre-quantise — same structural design, DRIFTED constants, WYSIWYG divergence in a lane whose premise is "the way the canvas draws it"; the header's byte-shared-shader claim covers WGSL only, the CPU fade is a duplicated transcription the pin law (plan 300) demands pins for; menu align | deliberate-divergence + pin-the-difference. REST CLEAN — TEXTBOOK substrate-first arrivals ×3: spawn_place.rs + target_assign.rs = batch alternatives composing katgpt_core::assign (Plan 620 solver) with in-source what-this-is-and-is-not sections feeding the EXISTING decentralized vocabulary (sample_spawn_point / TargetClaim::seed_claims, "NOT a wire frame and NOT server arbitration"); cluster_rebalance.rs = the game half of the chain shard_assignment_solver (Issue 164) composing katgpt-assign cross-repo, boundary respected; hero_no_gain.rs = textbook `game_traits::sigmoid` consume + Schmitt-band test legs; steer_avoid.rs consumes `swarm::math::sigmoid` = the documented fast_sigmoid delegation wrapper (pre-existing home, note-level two-homes-in-one-wave beside game_traits::sigmoid — same delegation substrate, no action); drive_constants_train = the Schmitt band FITTER (training-side f64 numerics, in-source house law) — the σ copy split out as 636, the fit lane itself sanctioned; lava sigmoids = the documented plan-356 quantised-fade design (divergence filed as 102); softmax = model-forward (clef lane, real_kl_bench instrument, encoder_train backward-mirror); sqrt/distance = statistical (gate_drift Mahalanobis-ish, bench CIs), squared-distance/tick idioms (steer_avoid), radial placement maths over [f32;2] (spawn_place — no MapPos3D in the batch API, documented class); f64 exp/cos/ln = approx.rs doc prose + clef f16 RoPE (model-forward) + gate_drift entropy; zero splitmix / new_v4 / [..16] / Belief-struct / spatial-HashMap arrivals; Schmitt/hysteresis arrivals = test legs asserting the band + spawn_place AOI doc mention — no new latch implementations | riir-train `636` filing + seal-remake `102` filing + katgpt-rs (this commit) |
| 2026-10-16 | Mode 2 - the 10-15 ~16:00 → 10-16 ~08:1x delta wave (32 fresh .rs across 6 repos: katgpt-rs 7 [plan-627 set_refine + plan-626 swir trio + plan-621 healer-lane eval suite + phase_belief], ai 4 [plan-626 probes + place_memory + wanderlust gates], refine 11 [corpus entries_28-32 + gpu_96-100 + bench_008/009 — fixture class], reflex 5 [set_refine_ab + rubric + null_stability + jeva lane], train 1 [plan437 parity example], remake 4 [079 warp/load_report/net_debug/zone_bundles/alloc_meter]; signature greps + body-reads; zero-cargo under the jeva capture) | CLEAN — zero filings; TEXTBOOK consumes throughout: set_refine.rs = thin composition over set_sigmoid_attention_wide_into + identity_projection_into (the plan-627 amendment's own framing, in-doc); phase_belief.rs carries its OWN Mode-1 substrate note in-source (signal-diff vs phase_rotation_coupling) + fast_sigmoid delegates + the sync-boundary posture stated; place_memory.rs = det_trig::sigmoid delegate + MapPos3D distance_2d_to accessor + PlaceBeliefRow domain vocabulary; NOTE-LEVEL: place_memory's freshness decay restates the sigmoid(−λ·Δt) FORMULA (documented, delegated) rather than a shared decay_confidence fn — joins the pattern-travels-by-citation family beside the Schmitt census; swir trio + view/harness files zero numeric-decision surface; test/bench/corpus = standing fixture/oracle exemptions; ALSO FOUND: 'bridge-walk-guards/' = a new sge WORKTREE (.git file, common-dir seal-game-editor) — not a repo, outside every canonical walk; the sge mainline itself landed 43 commits 10-15/16 (spawn-target + monster-mass + gate-stamping merges, all with .gate receipts — editor lanes, not audited beyond this census) | katgpt-rs (this commit) |
| 2026-10-15 | Mode 2 - the 10-13 ~16:00 -> 10-15 wave (432 fresh .rs across 13 repos: editor 198 [plan-363 equip/ambient lanes], ai 71 [the 1064/1066/1068 pvp+perf lanes + plan-635 tails], remake 61, refine 52 [corpus fixture class], dapps 13, infer 9 [the 1068 graph-walk + gdn arm], kat 9, reflex 8 [the T2.5 teacher-router + d3 lane], game-sdk 2, mmorpg/auth 3, + the behind-commit kimi_k3 files x6 read from origin; signature greps + body-reads; zero-cargo) | CLEAN - zero filings; 2 NOTE-LEVEL arrivals: (1) reflex harness/runner/router.rs:87 inline one-branch f64 sigmoid (born 10-14 0cb63ff, the T2.5 teacher-router) - katgpt-core reachable at crate level, no in-source delegation rationale; mitigated OFFLINE-instrument + the frozen-read law (no cross-build bit commitment); menu = delegate exact_sigmoid_f64 (free) or a recorded-refusal comment (the 014/suites.rs precedents); (2) editor random_equip.rs:115 splitmix64 finalizer restated (born 10-12 5a7476ed, re-touched 10-14) - structurally forced (view-layer, no katgpt dep, the seal-view 057 class), joins the splitmix census with a per-repo pub(crate) home menu. REST CLEAN: ai last_known_pos = substrate accessors in action (composition/hero_threat_lane); hero_progress_drive = textbook luck.rs consume (the 1044 fix holding post-635); kimi_k3/gpu softmax = model-forward; infer deltanet/issue046 = model-forward + the documented house-fill test fixture; refine = corpus string fixtures; seal-view splitmix/meadow-grid/belief DTOs = the documented 057/view classes (waypoint_overlay BeliefSource/Marker body-read = debug-overlay DTOs over QuestCombatState, zero parallel belief compute); dapps/kat/auth/sdk/mmorpg zero hits | katgpt-rs (this commit) |
| 2026-10-13 | Verification addendum (refine idle cycle 57, 4090) — the infer decode spin pool THREADING angle, the one class the 10-13 delta-wave row's signature greps cannot reach (thread pools are not sigmoid/splitmix/belief/distance signature items) | CLEAN — katgpt-core ships NO thread-pool/park primitive (grep: every `Pool` type is a domain pool — PoolConjecturer/DualPoolBandit/SeasonalPoolForecaster/GraphStablePool/PoolAdmissionConfig; zero ThreadPool/Condvar hits) → riir-infer's local spin pool (katgpt 936, GOAT-gated G1 bit-identity + G2 paired-perf) had no substrate to consume; riir-infer-laya's GLU condvar pool = different crate + different regime (macOS-Metal GLU ops vs CPU decode inter-op) — not intra-repo duplication. No filing | katgpt-rs (this commit) |

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

