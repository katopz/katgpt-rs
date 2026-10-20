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
| 2026-10-19 | Mode 2 — the post-18d delta wave (anchor 29f37db2c, 10-18 11:28+07; 30 fresh .rs across 6 repos: seal-std 12 [the NEW seal-fx leaf, seal-std issue 001 — schema/curve/classify/quads/goldens], sge 7 [fx_export writer + warp_lint tool + skill_projectiles + look/motion + canvas motion tests + 2 examples], remake 5 [the fx reader arc: fx_player/skill_fx/clip_fx/buff_fx/audio], katgpt-rs 2 [fdr_credit Terminal-RSI + spr_order], ai 3 [buffs.rs plan-033 slice B + pin_gate_tests + wkg_inspect example], dapps 3 [sponsor + stake ingest/routes]; signature greps + body-reads; zero-cargo under 2 live sibling suites) | **CLEAN — zero filings; the wave's centerpiece is the STRONGEST single-source-by-IMPORT form on record:** seal-fx = canonical leaf (curve.rs header carries the port rationale + the bit-identical-both-players premise), editor-core::effect + editor-release::fx_export + seal-view {fx_player,skill_fx,clip_fx,buff_fx} all RE-EXPORT/import seal_fx types (editor-widgets consumes transitively via editor_core — no leaf dep needed), goldens (SKILL_FX_WIRE/CLIP_FX_WIRE/BUFF_FX_WIRE) pinned IN the leaf and parsed by BOTH halves' tests — no copy exists to diverge, stronger than a divergence pin (the 232/233 class PREVENTED, not cured). TEXTBOOK consumes: buffs.rs = `swarm::luck::seeded_hash` designated home + substrate `Hp/Position/is_dead` + determinism laws in-source (the 1044 fix holding through its first consumer wave); pin_gate_tests = MapPos `distance_2d_to` accessor; NEW substrate-side operators, not duplicates: katgpt-pruners `mm_squash` = Michaelis-Menten reciprocal form (P/(P+D') — saturating, NOT logistic; no katgpt-core restatement found, substrate-home birth, house sigmoid-rule compatibility cited), `fdr_credit.rs` cites the percentile tail-support floor in-source (audit culture compounding); dapps sponsor.rs [..16] = the 81-byte wire codec AccountId family; audio.rs linear-over-sigmoid = in-source rationale (two-anchor contract). Zero splitmix/new_v4/belief-struct/spatial-HashMap/emotion arrivals; editor-core's 1-line lerp = documented cross-reference to the seal_fx private copy, below the gate | katgpt-rs (this commit) |
| 2026-10-18d | Mode 2 — the post-18c delta wave (41 unaudited fresh .rs across 8 repos + 2 excluded-by-class: jup-tool 2 [the 4cf74d2 attribute-only posture], ai 10 [the 1091 spawn-HP + pvp-team tails], dao 2 [the landed Proposal-009 orphan], dapps 8 [the Proposal-009 TUNA airdrop trio + explorer views], mmorpg 2 [the 112/113 fixes], train 10 [plan410 stage0 + discoloop POC — READ-ONLY, routing-restricted], remake 3 + sge 3 [the plan-390 molten twin wave M9/M11 + the reach-cap fix]; katgpt-rs 223 = the 947 comment-move docs commits [zero semantic arrivals — removed non-comment lines = 3 markdown lines], riir-kat 35 = a whole-crate rustfmt round-trip [byte-identical, fixture class]; signature greps + body-reads; zero-cargo under sibling builds) | **1 FILED: seal-game-editor Issue 233** — the vessel-lava wire literal pin is ONE-SIDED: the viewer's `the_wire_lava_reads_as_the_editor_pressed_it` freezes the full WIRE JSON (uv_per_tile 0.5, warm_share 0.35, embers 36, limelight 8/20/4, glow cap 10.0) but the EDITOR side asserts no literal (its 7 tests = round-trips + const-vs-const) — an editor-types retune reds nothing anywhere while silently staling the viewer's frozen press; the plan-300 both-repos law half-satisfied; the chain-139 class third instance in this lane family (232 flow-turn bodies, remake 102 fade constants, 233 wire literals); menu = the editor twin test. VERIFIED-STILL-VALID: remake Issue 102 unchanged by the wave's re-touch (viewer FADE_GAIN 8.0 / FADE_STEPS 32.0 no-snap vs the editor's 12/16 end-snapped — the filing's table stands). REST CLEAN — textbook consumes: hero_progress_drive = `game_traits::sigmoid` delegate ×3 + the `luck::{seeded_hash, splitmix64_finalize, SPLITMIX64_GAMMA}` designated home (the 1044 fix holding); distill_tsd:1094 = a closed-form test oracle (`test_row_confidence_matches_commit_arithmetic`); discoloop Box-Muller/RMSNorm/Adam/1-√n sqrt = model-forward + POC in-crate RNG with in-source rationale; meadow_lava:283 inline σ = structurally forced (no katgpt dep, the documented seal-view class) + the 102-covered surface; the plan-390 reach cap = the CORRECT twin shape (`reach_cap twin`, asserted both sides); meadow_lava `HashMap<(LavaPart,u32)>` = a render mesh cache; dapps [..16] = the doc-comment display family; belief/emotion/new_v4 = zero arrivals | sge `67727d54` filing + katgpt-rs (this commit) |
| 2026-10-20a | Mode 2 wave audit (10-18 12:00 → 10-20 ~04:0x UTC; union-of-touched .rs via git log across the 9 workspace repos: katgpt-rs 18 / ai 1444 / train 711 / refine 55 / infer 125 / instinct 2 = ~2355 files, the bulk = the 2026-10-19 07:58 FLEET FMT SWEEP + the ai-1095 soak lane + refine B259-B264 corpus; signature greps softmax / new_v4() / fn sigmoid / SpatialBelief-family / coordinate-keyed HashMap over the touched sets, body-reads on every hit; zero-cargo, StartLux resident) | **CLEAN — ZERO FILINGS**: new_v4 hits ×3 = doc-comments CITING the law (pet_roster / motivation types / entity types); engine sigmoid defs ×3 (limelight:111 / arbitration:433 / thermal_lod:408) = the ai-1069 dispositioned NaN-guard + fast_sigmoid DELEGATION wrapper shape verbatim; katgpt-core lib.rs = the substrate home itself; infer elementwise sigmoid_f32 + clef softmax mentions = fmt-sweep-touched pre-existing + doc-comments describing the REFERENCE model's internals; SpatialBelief hits = the known ai-1051 census entry + TEXTBOOK GenericSpatialBelief<T> consumes (flee_steering/emotion); HashMap-spatial hits = id→pos value maps (u32 keys), not coordinate-keyed; corpus entries = fixture text. NOTABLE WAVE CONTENT (no drift): the fmt sweep normalized the fleet (line-ending + style only, blame-verified on the new_v4 sites), B261-B264 landed kernel_opt corpus, ai-1095/plan-638 lanes | katgpt-rs (this commit) |
| 2026-10-18c | Mode 2 mini-audit — the post-18b delta (10-17 21:00Z → 10-18 09:3x: 9 fresh .rs across 4 repos: ai 3 [crowd_state_windows T1–T4 files], refine 2 [B249 slice2/3 corpus fixture class], sge 2 [plan-390 M3b/M4 canvas lava-flow + molten-rig], remake 2 [M9 RUN viewer twins meadow_lava_flow + molten_rig]; signature greps + body-reads; zero-cargo under the plan-436 GPU chain) | **1 FILED: seal-game-editor Issue 232** — the plan-390 M3b lava flow-turn landed as a CROSS-REPO TWIN PAIR with zero divergence pins: editor `RIIR_DRIFT`/`RIIR_LAVA_FRAGMENT`/`flow_turn` body/26-line `FLOW_LAVA_VERTEX_WGSL` vs viewer `meadow_lava_flow.rs` + `shaders/flow_lava.wgsl` verified BODY-IDENTICAL this audit, but each side's tests assert only its own body (self-consistency) — the chain-139 copy-gate law (acknowledgment present, pin absent; the seal-remake 102 shape one layer up, pre-drift). Structurally forced (seal-view has no editor-widgets dep — separate deploy surfaces), so menu = per-repo byte-literal pin tests, NOT delegation. REST CLEAN: state_windows.rs = TEXTBOOK consume re-verified on the full body (katgpt_core::event_state_window + habituation_filter::settling_ticks wholesale; det_trig::ln64/exp delegates; EmotionField read-only accessor; zero alloc scratch contracts in-source; the 18b row's verdict stands); molten_rig twins (editor canvas vs viewer rig) = the WGSL-is-the-source design with the viewer's literal-quoted pinned contract tests + deliberate rig-limit divergence — sanctioned, not the 232 class; quantise ×3 private module-locals (canvas fire/molten + remake meadow_fire) = pre-231-census family, below every gate (single-copy-per-repo, distinct steps args); B249 corpus = fixture class | sge `232` filing + katgpt-rs (this commit) |
| 2026-10-18b | Mode 2 mini-audit — the post-row delta only (since 21:00Z: ai 4 [crowd_state_windows T1–T4 landing + G4 gate fix — self-audited at landing via the ai-1081 substrate-check block], sge 11 [plan-356 B2/B3/B4 motion lanes: look/fire|lava|waterfall|pulse + model_tags + motion + run_outfit], ndb 1 [art_bundle heal]; signature greps + body-reads; zero-cargo under sibling builds) | **1 FILED: seal-game-editor Issue 231** — the splitmix64 finalizer census CROSSED the per-repo threshold: 7 private module-locals across all 4 base crates (look/fire+lava+waterfall `placement_seed` ×3 with deliberate per-lane salts, editor-data random_equip `splitmix64`, editor-core services `sample_blake3` + effect `draw`, editor-widgets leaves `mix`) — each single copy structurally forced when written (no katgpt dep), the family accumulated one plan at a time past the 4th-arrival consolidation gate; menu = one editor-types home collapsing the look trio (zero new edges) + singleton crates stay local per their dep laws + copy-gate test on the landing. phases.rs Fibonacci hashing + dance.rs u32 GOLDEN = different mixer families, already single-homed. REST CLEAN: crowd_state_windows = TEXTBOOK consume (katgpt_core::event_state_window primitive wholesale + det_trig::exp test oracle + swarm math sigmoid re-export); pulse/fire look lanes = view-layer sine/dance-phase; model_tags HashMaps = counts caches; art_bundle blake3 = the substrate hash family itself; sibling's fire.rs WIP (plan-356 B2 flameless profiles) verified non-overlapping before filing (yield check) | sge `ca4ffa79` filing + katgpt-rs (this commit) |
| 2026-10-18 | Mode 2 — the 10-16 08:00 → 10-18 delta wave (128 fresh .rs across 13 repos: train 18 [vq_codebook arc + plan445 benches + drive_constants_train], editor 50 [lava/relief/sound/skill-script lanes], remake 13 [meadow_lava/relief/warp/spawn_focus], ai 12 [hero_no_gain + steer_avoid + spawn_place/target_assign + cluster_rebalance + wanderlust tests], infer 9 [riir-infer-clef lane], katgpt-rs 9 [qrr arc + approx + assign explain], dapps 3 + kat 2 + refine 2 [tuna airdrop trio], instinct 4 [eq9], shader 4 [lava gamefx], reflex 1 [gate_drift], seal-std 1; signature greps + body-reads; zero-cargo under Plan-337) | **2 FILED: riir-train Issue 636** — drive_constants_train.rs:199 `sigmoid64` restates `katgpt_core::exact_sigmoid_f64` BIT-IDENTICALLY (same two-branch stable form, binding name only differs) with the home REACHABLE (katgpt-core = hard workspace dep, engram_privilege) + house-law comment present but NO delegation refusal named + NO divergence pin = the 163 shape (copy-gate, NOT the 002 numerics class — delegation zero-behavior, verdicts replay through hard_surface); menu delegate | recorded-refusal + twin-pin. **seal-remake Issue 102** — the lava fade twins carry NO literal pin: editor ring_fade σ(12·(edge−d)/width)/steps16/end-snapped vs meadow_lava part_fade σ(8·(edge−d)/width)×presence/steps32/no-snap/presence-pre-quantise — same structural design, DRIFTED constants, WYSIWYG divergence in a lane whose premise is "the way the canvas draws it"; the header's byte-shared-shader claim covers WGSL only, the CPU fade is a duplicated transcription the pin law (plan 300) demands pins for; menu align | deliberate-divergence + pin-the-difference. REST CLEAN — TEXTBOOK substrate-first arrivals ×3: spawn_place.rs + target_assign.rs = batch alternatives composing katgpt_core::assign (Plan 620 solver) with in-source what-this-is-and-is-not sections feeding the EXISTING decentralized vocabulary (sample_spawn_point / TargetClaim::seed_claims, "NOT a wire frame and NOT server arbitration"); cluster_rebalance.rs = the game half of the chain shard_assignment_solver (Issue 164) composing katgpt-assign cross-repo, boundary respected; hero_no_gain.rs = textbook `game_traits::sigmoid` consume + Schmitt-band test legs; steer_avoid.rs consumes `swarm::math::sigmoid` = the documented fast_sigmoid delegation wrapper (pre-existing home, note-level two-homes-in-one-wave beside game_traits::sigmoid — same delegation substrate, no action); drive_constants_train = the Schmitt band FITTER (training-side f64 numerics, in-source house law) — the σ copy split out as 636, the fit lane itself sanctioned; lava sigmoids = the documented plan-356 quantised-fade design (divergence filed as 102); softmax = model-forward (clef lane, real_kl_bench instrument, encoder_train backward-mirror); sqrt/distance = statistical (gate_drift Mahalanobis-ish, bench CIs), squared-distance/tick idioms (steer_avoid), radial placement maths over [f32;2] (spawn_place — no MapPos3D in the batch API, documented class); f64 exp/cos/ln = approx.rs doc prose + clef f16 RoPE (model-forward) + gate_drift entropy; zero splitmix / new_v4 / [..16] / Belief-struct / spatial-HashMap arrivals; Schmitt/hysteresis arrivals = test legs asserting the band + spawn_place AOI doc mention — no new latch implementations | riir-train `636` filing + seal-remake `102` filing + katgpt-rs (this commit) |
| 2026-10-16 | Mode 2 - the 10-15 ~16:00 → 10-16 ~08:1x delta wave (32 fresh .rs across 6 repos: katgpt-rs 7 [plan-627 set_refine + plan-626 swir trio + plan-621 healer-lane eval suite + phase_belief], ai 4 [plan-626 probes + place_memory + wanderlust gates], refine 11 [corpus entries_28-32 + gpu_96-100 + bench_008/009 — fixture class], reflex 5 [set_refine_ab + rubric + null_stability + jeva lane], train 1 [plan437 parity example], remake 4 [079 warp/load_report/net_debug/zone_bundles/alloc_meter]; signature greps + body-reads; zero-cargo under the jeva capture) | CLEAN — zero filings; TEXTBOOK consumes throughout: set_refine.rs = thin composition over set_sigmoid_attention_wide_into + identity_projection_into (the plan-627 amendment's own framing, in-doc); phase_belief.rs carries its OWN Mode-1 substrate note in-source (signal-diff vs phase_rotation_coupling) + fast_sigmoid delegates + the sync-boundary posture stated; place_memory.rs = det_trig::sigmoid delegate + MapPos3D distance_2d_to accessor + PlaceBeliefRow domain vocabulary; NOTE-LEVEL: place_memory's freshness decay restates the sigmoid(−λ·Δt) FORMULA (documented, delegated) rather than a shared decay_confidence fn — joins the pattern-travels-by-citation family beside the Schmitt census; swir trio + view/harness files zero numeric-decision surface; test/bench/corpus = standing fixture/oracle exemptions; ALSO FOUND: 'bridge-walk-guards/' = a new sge WORKTREE (.git file, common-dir seal-game-editor) — not a repo, outside every canonical walk; the sge mainline itself landed 43 commits 10-15/16 (spawn-target + monster-mass + gate-stamping merges, all with .gate receipts — editor lanes, not audited beyond this census) | katgpt-rs (this commit) |
| 2026-10-15 | Mode 2 - the 10-13 ~16:00 -> 10-15 wave (432 fresh .rs across 13 repos: editor 198 [plan-363 equip/ambient lanes], ai 71 [the 1064/1066/1068 pvp+perf lanes + plan-635 tails], remake 61, refine 52 [corpus fixture class], dapps 13, infer 9 [the 1068 graph-walk + gdn arm], kat 9, reflex 8 [the T2.5 teacher-router + d3 lane], game-sdk 2, mmorpg/auth 3, + the behind-commit kimi_k3 files x6 read from origin; signature greps + body-reads; zero-cargo) | CLEAN - zero filings; 2 NOTE-LEVEL arrivals: (1) reflex harness/runner/router.rs:87 inline one-branch f64 sigmoid (born 10-14 0cb63ff, the T2.5 teacher-router) - katgpt-core reachable at crate level, no in-source delegation rationale; mitigated OFFLINE-instrument + the frozen-read law (no cross-build bit commitment); menu = delegate exact_sigmoid_f64 (free) or a recorded-refusal comment (the 014/suites.rs precedents); (2) editor random_equip.rs:115 splitmix64 finalizer restated (born 10-12 5a7476ed, re-touched 10-14) - structurally forced (view-layer, no katgpt dep, the seal-view 057 class), joins the splitmix census with a per-repo pub(crate) home menu. REST CLEAN: ai last_known_pos = substrate accessors in action (composition/hero_threat_lane); hero_progress_drive = textbook luck.rs consume (the 1044 fix holding post-635); kimi_k3/gpu softmax = model-forward; infer deltanet/issue046 = model-forward + the documented house-fill test fixture; refine = corpus string fixtures; seal-view splitmix/meadow-grid/belief DTOs = the documented 057/view classes (waypoint_overlay BeliefSource/Marker body-read = debug-overlay DTOs over QuestCombatState, zero parallel belief compute); dapps/kat/auth/sdk/mmorpg zero hits | katgpt-rs (this commit) |
|---|---|---|---|
| 2026-10-13 | Verification addendum (refine idle cycle 57, 4090) — the infer decode spin pool THREADING angle, the one class the 10-13 delta-wave row's signature greps cannot reach (thread pools are not sigmoid/splitmix/belief/distance signature items) | CLEAN — katgpt-core ships NO thread-pool/park primitive (grep: every `Pool` type is a domain pool — PoolConjecturer/DualPoolBandit/SeasonalPoolForecaster/GraphStablePool/PoolAdmissionConfig; zero ThreadPool/Condvar hits) → riir-infer's local spin pool (katgpt 936, GOAT-gated G1 bit-identity + G2 paired-perf) had no substrate to consume; riir-infer-laya's GLU condvar pool = different crate + different regime (macOS-Metal GLU ops vs CPU decode inter-op) — not intra-repo duplication. No filing | katgpt-rs (this commit) |
| 2026-10-13 | Mode 2 — the 10-12 03:4x → 10-13 ~16:00 +07 delta wave (378 fresh .rs across 11 repos: ai 128 [the 1056 basis_purity_audit lane + editor-spawn fix + duel/combo tails], seal-game-editor 69, seal-remake 45, refine 33 [Batch 228 corpus — fixture class], infer 32 [GDN cudarc + issue933 probes], katgpt-rs 30 [lattice_memory], dapps 21 [the D2 pay lane], reflex 11, deployer 2, kat 3, seal-std 4; signature greps + body-reads; zero-cargo) | **CLEAN — zero filings; the 1057-close VERIFIED LANDED at all 3 sigmoid sites** (hayek_npc.rs:121 → det_trig::sigmoid64, turnin_drive.rs:107 → det_trig::sigmoid, both in-source-cited; NEW arrival integrity/injection.rs:35 = textbook sanctioned fast_sigmoid delegation with the Cephes/±40 rationale, born with the 1056 lane) + splitmix = 2 textbook consumes of the designated katgpt_core home (player_style.rs module docs + basis_purity_audit tests) + the cudarc device-RNG constant (fixture class). REST: dapps [..16] = the KV-key-derivation + wire-DOC families (D2 lane); infer softmax = model-forward domain (sanctioned); combat sim dx²+dy² = the documented squared-distance/tick idioms over plain fields; belief-struct hits ALL = the known ai-1051 census members (ns_csg/crowd_mcgs/civ SpatialBelief + NpcResidualBelief 06-13 + MonsterBeliefs vocabulary) — zero NEW arrivals; new_v4 = 0, spatial HashMap = 0, emotion structs = 0; editor/remake fresh sets = view-layer classes (no sigmoid/splitmix/belief arrivals) | katgpt-rs (this commit) |
| 2026-10-12 | Mode 2 — the 10-11 09:10 → 10-12 03:4x +07 delta wave (on-box: riir-ai 147 fresh .rs [the duel substrate game_sync/duel_engine+karma_engine, quest_turnin_drive sigmoid drives, hero_stats/peer_boot social seam, hero_combo/combo-depth, loot tables + damage_popups rolls, the `02962d6b6d` E6 det_trig rebase], riir-infer 16 [the Plan-622 Phase-7 si lane attention_si.rs + si bins + the 929 loop probe + device-lost callback], refine 1 [own heal.rs cfg-gate, known by authorship]; signature greps + body-reads + blame; Plan-337 zero-cargo under the 623 pplx capture) | **1 FILED: riir-ai Issue 1057** — the E6 platform-split commit misrouted `hayek_npc.rs:119`'s f64 sigmoid into `det_trig::exp(f32)` = E0308: riir-games does NOT compile at the non-default `npc-market` posture since 10-11 16:32 (the feature-matrix-hole class; the only f64 misroute in the 18-site det_trig census; fix = `exp64`, queued for the cargo window) + SECONDARY the det_trig sigmoid copy family crossed the copy-gate (3 identical production bodies + the two_brain 2·cos variant across 2 crates, `478972f0f5`'s turnin_drive the newest arrival; menu: `det_trig::sigmoid`/`sigmoid64`, NOT merging the katgpt_core Cephes house — the two-house E6 design recorded). REST CLEAN: infer attention_si = TEXTBOOK consume (`katgpt_core::scale_invariant::{ScaleInvariantLut, HeadCalibration, SpreadAccumulator}` wholesale); the duel/karma engines zero transcendental/crypto surface; Family A = 7 sanctioned fast_sigmoid delegation wrappers + pet_roster's documented clamp wrapper; NpcResidualBelief pre-existing 06-13 (outside fresh scope); QuestRestockBelief = documented R12; fastrand/sqrt/blake3[..16]/spatial-HashMap = documented FP classes (zero blake3-identity, zero spatial-HashMap arrivals) | riir-ai `6b37d3a960` |
| 2026-10-12 | CLOSE-OUT of the row above (refine idle cycle 20; the compiler was the census — scope grew 3× under validation): 2 MORE production misroutes from the same sweep (skill_opt/schedule.rs f64 cos → NEW det_trig::cos64; mb_personality f64 NLL ln → ln64) + 8 TEST misroutes (blast radius corrected: cargo test -p riir-games at ANY posture was red since E6); the menu LANDED (sigmoid/sigmoid64 born, bit-identical + test-pinned, 4 production sites delegated) + 16 sweep parens residue healed (cargo fix, all-features posture) | 1057 CLOSED at riir-ai 693e19e629 — all-features/all-targets -p riir-games GREEN 0 warnings, 37/40 restored test targets green; UNMASKED + FILED: ai 1058 (qsg t6_k3 — E6 libm swap reshuffled the seeded realization to a ~2.4σ draw vs the 2σ single-realization bench-931 tolerance; owner re-pin) + ai 1059 (mmorpg --all-features PRE-EXISTING combo-only breakage, hero_combo × runetrace × sync field inits) | riir-ai 693e19e629 |
| 2026-10-11 | Mode 2 mini-audit — the post-09:10-anchor delta only (the 09:10 row below already carried the day's wave; anchor = 342c9273ff's commit time; ~41 live files across 7 repos: dapps plan-223/050 lanes [routes.rs head-meta/env-view/capture + contract.rs family + the stage-0 attribution files], seal-remake 052/054 probe arc [authored_life.rs/pool.rs/quest_sim.rs/vessel_boot.rs retouches + player_style_save.rs] + the cash_shop files that appear in the log only because 792c3fe DELETED them (issue 059's dead-parallel-family drop — the drift self-corrected), editor plan-025 combo-gauge animation files [live sibling lane, read-only as-found], refine journey.rs [link-fix doc-comment only] + fsearch bench sidecars [fixture class], ai quest_grammar grammar_training.rs [contended sibling lane, as-found], infer/train examples [bench fixture class]; signature greps + body-reads; Plan-337 zero-cargo under 2 live sibling builds) | **CLEAN — zero filings.** dapps blake3[..16] hits = wire-contract DOC prose (the documented FP class, canonical in riir-kat); seal-view sqrt/distance = view-layer squared-distance idiom over plain arrays, no MapPos3D in repo (documented class); authored_life/pool.rs body-read = the POOLED hero-spin measurement gate (issue 052 wall re-anchor), an instrument not a parallel ambient implementation; no sigmoid/splitmix/belief-struct/spatial-HashMap arrivals in any fresh production file | katgpt-rs (this commit) |
| 2026-10-11 | Mode 2 — the 10-10 18:00 → 10-11 wave (241 unique fresh .rs across 10 repos after vendor/target exclusion: riir-ai 37 [1049 T2/T3 sync_relation_gate/ltm/lane + quest_combat hero_manual/stance_governor/threat_lane/manual_order + g1 step-dump tests], dapps 112 = the T15 rustfmt-restyled tree (ZERO --diff-filter=A file arrivals — pre-existing lines, the restyle class), refine 33 corpus entries [fixture class] + build_stamp, katgpt-rs 13 [the 930/bench-621 lane: scale_invariant + parallax_attn tests + state_probe_eval + escalation bench; the lane's still-uncommitted WIP files excluded], train 17 [READ-ONLY, routing-restricted], seal-remake 10, editor 10, reflex 7 [probe_delta/corpus_ab/distill], infer 1 [rope], ndb 1 [art_bundle]; signature greps + body-reads; Plan-337 zero-cargo under the plan437 eval stage + 2 live sibling cargos) | **1 FILED: riir-ai Issue 1051** — the SpatialBelief family census, surfaced by the crowd_mcgs re-touch: FOUR parallel think-brain belief structs — the layer-0 canonical `GenericSpatialBelief<T: TargetKind>` (games-shared, born 06-13) + civ `SpatialBelief` (06-07, NO rationale, canonical UNCONDITIONALLY reachable — games-civ deps games-shared, spatial is core-always-compiled) + crowd_mcgs `SpatialBelief` (06-15, re-touched 10-11; FALSE-DILEMMA rationale — "rather than depending on civ_spatial_cognition::SpatialBelief (no need to enable the feature)" argued only the feature-gated civ variant, the layer-0 name never in the candidate set = the vocabulary-translation miss class, third consumer-side arrival past the copy-gate) + engine ns_csg `SpatialBelief` (STRUCTURALLY FORCED — riir-engine does NOT dep riir-games-shared, its Cargo.toml:1809 documents the layering; note-level EXCEPT the no-+4.6-offset decay divergence [fast_sigmoid(-(λ·dt)) → 0.5 at dt=0 vs the family's ≈0.99] + the intra-engine decay duplicate vs entity_cognition/layers::decay_confidence); 4 decay_confidence impls, zero divergence pins; the 06-13 layer-0 extraction never went back to consume its June siblings (the 1027 born-before-first-caught-now class). REST CLEAN: crowd_mcgs:66 sigmoid = textbook fast_sigmoid delegate with the Never-softmax law in-source; limelight:108 = the sanctioned NaN-guard wrapper still delegating fast_sigmoid; katgpt si_probes = feature-gated (scale_invariant_attn) diagnostics CONSUMING the substrate — ScaleInvariantLut::pair + gaussian_expectation "never re-derived here"; the inline 1/(1+e^-x) closures are the integrand-under-measurement (instrument class); scale_invariant 0.327_591_1 = the documented A&S-erf constant; step-dump/scale-invariant test sigmoids = test-oracle exemption; state_probe_eval splitmix = the documented T1.2 fixture convention; reflex probe_delta CONSUMES katgpt_core::state_probe; reflex/dapps [..16] = the display/evt-key families; seal-view splitmix = the documented 057 id64 (OPEN, build-gated). NOTE-LEVEL: the Schmitt-latch census grows 6 → ~8 sites / 2 repos (NEW: seal quest_sim's 052 deflector side-latch + run_lane's talk-cone enter/exit band TODO; ai ambient's spawn/despawn band = the documented AOI hysteresis) — the game_traits::schmitt consolidation menu stands | riir-ai 342c9273ff + katgpt-rs (this commit) |
| 2026-10-10 | Mode 2 — the post-day-row delta wave (178 fresh .rs across 10 repos in the 7h window — the tail the day row's 10-09→10-10 sweep predated: katgpt-rs SyncBank issue-930 + scale_invariant_attn Plan-622 + parallax_attn + katgpt-sense re-export wiring, riir-ai sync_relation_gate 1049-T1 + hero_manual plan-024 + the 052 quarry latch, seal-remake manual_control + G3 digest probes, editor creature-page merge, infer desc_len, train mdl_collapse_probe [routing-restricted, read-only], dapps rustfmt restyles; signature greps + body-reads; Plan-337 zero-cargo under 5 live sibling lanes) | **CLEAN — zero filings; 1 new substrate landing verified textbook consume** (sync_relation_gate delegates `katgpt_core::simd::fast_sigmoid` + consumes the 930 `ReconstructionState::sync_bank()` accessor, house laws cited in-source, the limelight reduce pattern named). Schmitt-latch census 6 sites / 2 repos (limelight FSM, sync_relation_gate fold, quarry re-latch, stance governor, progress drive, field-rest band) = NOTE-LEVEL: no shared type, the pattern travels by in-source citation, each binds different domain state (FSM events / stance byte / re-latch margin / rest band) — consolidation menu (a `game_traits::schmitt` two-threshold latch) recorded for a 4th consumer-side arrival. train probe splitmix+Box-Muller = in-source no-dep POC-reproducibility rationale (fixture family). G3 FNV-1a digest = probe fixture ("std-only, not a commitment hash" in-source). REST = documented FP classes (attention online-softmax over the substrate's softmax_scaled, 1/√d scales, [8..16] wire-codec offsets, R12/HeroThreatLane/MonsterThreatSource vocabulary, editor DB-row keys, fastrand seeded test fixtures, log2 entropy definitions) | katgpt-rs (this commit) |
| 2026-10-09 | Mode 2 — the 10-07 03:51→10-09 12:2x wave (689 fresh .rs across 18 repos, vendor/ excluded: editor 139 [ambient-VFX plans 297/299/300/301/311], ai 115 [wand lane 1039-1043 + deliberation_chord 1037], refine 86 [wasm-validator + build-stamp + gpu_70-72 corpus], katgpt-rs 60 [katgpt-assign Plan 620 + lattice/lsh tails], dapps 51 [121/118 lanes], seal-remake 44 [runetrace + locomotion_gait/drop_visual/combat_fx], kat 37, infer 33 [svd_lbit], ndb 31 [riir-rag carve], reflex 26 [081 Jev-Mem], train 24 [deltanet backward], chain 18, instinct 8, dao 6, deployer 5, rethink 3, mmorpg 2, seal-std 1; signature greps + body-reads; Plan-337 zero-cargo under load ~58) | **3 FILED: riir-ai Issue 1044** (chord_drive `deliberation_chord.rs:140` re-states the exact_sigmoid body BIT-IDENTICALLY (same two-branch libm form, same order) with katgpt-core a HARD dep of riir-games — the 1035 miss class again, born the day after 1035; + `hero_progress_drive.rs:202 pub const fn splitmix` while the repo's splitmix home `luck.rs` is wrong-shape (f32 roll, no u64 hash) and quest_combat's own mod.rs:240 comment cites luck as "the precedent" — the extend-the-substrate menu). **katgpt-rs Issue 927** — `katgpt-assign::rng::SplitMix64` (65a6b1b71, 10-07+) duplicates the DESIGNATED home `katgpt-types::rng::SplitMix64` (514989664, 10-06, the 071 export) — same name, same stream core, `below(n)` DIFFERS (64-bit vs 53-bit Lemire — not drop-in), rationale argues vs `rand` only (the miss class), zero-dep law the real defense → cross-pin-in-core menu (both crates visible under `assignment`). **seal-remake Issue 057** — seal-view ×3 inline splitmix64 (viewer id64 documented 09-21 + drop_visual `phase_of` + combat_fx `unit`, both this wave) — intra-crate DRY, the 071 class, pub(crate) consolidation menu (no direct katgpt dep; wasm32 posture). REST CLEAN: reflex engine.rs exact_sigmoid delegation + the :3169 bit-parity pin HOLD (014/071 verified); reflex suites.rs delegates katgpt_core::types::rng::SplitMix64 with the 071 record in-source; traj_store sigmoid_dot_8 delegates; rule_embed unchanged note-level arrival #1; locomotion_gait.rs:66 one-branch libm σ = STRUCTURALLY FORCED (seal-view has NO direct katgpt dep — the 3 manifest mentions are comments; the seal-core precedent) with a thin comment — note-level; riir-rag retriever:1006 inverse-sigmoid = §Bridge-Pattern form with the law cited in-source; hero_routine 1/(1+exp(-const)) = compile-time const folds (FP); svd_lbit:119 seed mixer = single post-export arrival, substrate reachable but the quant lane SIBLING-HOT (rethink-025 lbit arms mid-measurement) — note-level, revisit when the lane cools; ternary_deltanet cudarc splitmix = test-fixture RNG closure (f16 edge vectors); dapps/chain/refine/ndb/editor `[..16]` = the documented row-key/evt/display families; editor feat-pet `HashMap<(i64,i32)>` = DB row keys; belief hits = substrate vocabulary in action (MonsterBeliefs, last_known_pos accessor); quest_combat sqrt = the lane's documented squared-distance/tick idioms | riir-ai Issue 1044 · katgpt-rs Issue 927 + this commit · seal-remake Issue 057 |
| 2026-10-10 | Mode 2 — the 10-09 12:00→10-10 wave (122 fresh .rs across 13 repos, vendor/ excluded: editor 32, dapps 19 [contended ×2 siblings — committed material read-only], chain 12, seal-remake 12, katgpt-rs 11, ai 11, ndb 8, train 6 [routing-restricted, read-only], infer 5 +2 vendor, kat 2, instinct 2, reflex 1, rethink 1; second-root seal-online-remaster 0; signature greps + body-reads; Plan-337 zero-cargo under the Issue-920 census) | CLEAN — zero filings; **2 prior filings VERIFIED RESOLVED**: riir-ai Issue 1044 (`9a61db9ab`, file removed same commit — deliberation_chord.rs:144 delegates `exact_sigmoid`; luck.rs:53 re-exports `katgpt_core::types::rng::{splitmix64_finalize, SPLITMIX64_GAMMA}` + the seeded_hash one-shot fold; hero_progress_drive.rs:341 consumes) + katgpt-rs Issue 927 (`46aded23d` closed; cross-pin `36823de3f` — katgpt-assign/rng.rs full in-source rationale + `crates/katgpt-core/tests/splitmix64_twin_pin.rs` the copy-gate test half). NOTE-LEVEL: katgpt-core `loop_alignment_probe.rs:120` module-local `pub fn splitmix64` step — bit-identical to `types::rng::{SPLITMIX64_GAMMA + splitmix64_finalize}` (substrate-side, below the copy-gate; one-line delegation menu noted). REST CLEAN: instinct scene_loop.rs = textbook-consume showcase (7-stage substrate map in-source: `rating`/KarcForecaster/LeoHead/entropic_tilt/gain_cost_halt/`squeezed_sigmoid`/decision_wire; the elo HashMap = the consumed `rating` book, law-4 loop-local); infer elementwise_cubecl = GPU device kernels with the `katgpt_core::sigmoid` bit-parity doc; moe_backward + kimi_k3 backward/delay σ+rms sqrt = backward-mirror class; dapps rails.rs HMAC-SHA512 = SLIP-0010 spec-mandated KDF (not the blake3 identity family); ane.rs [..16] = error-message hex truncation (display family); dapps [..16] = the documented evt-key/row-decoder families; Mutex hits = daemon/view-layer server-handle shapes (domain_anchor_commit.rs:565 itself cites the papaya rule); zero softmax/rand/new_v4/belief-struct/emotion-struct arrivals; seal-remake Issue 057 remains OPEN (standing build-gated consolidation; viewer.rs hits = the documented id64) | katgpt-rs (this commit) |
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

