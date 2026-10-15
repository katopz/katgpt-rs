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
| 2026-10-07 | Mode 2 — the post-03:51 same-day delta wave (~95 unique fresh .rs across 10 repos: katgpt-rs 18 [the 917 T3 EB lane + lattice_memory/lsh arcs], refine 28 [wasm-validator + build-stamp + gpu_70-72 corpus], ai 17, infer 14, dapps 13, editor 8, chain 10 [the Plan-066 deposit/subscription programs], train 10 [deltanet backward], kat 5, rethink 1; signature greps over the fresh-file sets via temp-list xargs, body-reads on every hit; Plan-337 zero-cargo) | **1 FILED: riir-chain Issue 163** (`9d0ba168`) — `subscription.rs:110 account_id_from_pubkey` re-declares the bare `BLAKE3(pubkey)[..16]` identity copy WITH the rationale half (`:23` cites player_id_hash + the delegation authoriser) but NO divergence pin: its own tests + e2e build fixtures WITH the fn (circular), delegation's `:2884` pin is formula-restatement class, and no chain test joins the in-lib forms to `riir_wallet_signer::player_id_hash` despite the dev-dep existing for exactly that (the wallet-signer runtime dep is dev-only — copy justified, pin missing; fix = one dev-dep-consuming test). REST CLEAN: train deltanet backward σ = the backward-must-mirror-forward numerics class; katgpt lattice cdf 0.3275911 = the A&S-erf constant (not a sigmoid), `sigmoid_only_law_grep_gate` = a gate test, `lib.rs:38` = the substrate home, lsh hamming = hash-bucket domain; chain dispatch/delegation blake3 hits = cfg(test) fixtures + the domain-separated PDA family (documented idiom); ai hits = the morning audit's already-classified set (hero_routine σ-const oracle, QuestRestockBelief R12, squared-distance/sqrt tick idioms) re-touched without new vocabulary; refine hits = corpus string-literal fixtures + display `[..16]` truncations + heal-bin bookkeeping maps; editor rainbow_distance + (u32,Option<&str>) map = view/DB-key classes; infer/kat/rethink zero hits | chain `9d0ba168` (this commit) |
| 2026-10-07 | Mode 2 — the 10-06 09:00→10-07 03:51 wave (324 fresh .rs across 16 repos, vendor/ excluded: editor 97, deployer 50, dapps 35, ai 28, infer 28, seal-remake 25, refine 16 [contended, read-only], kat 11, instinct 10, dao 8, katgpt-rs 7, sdk 3, reflex 2, rethink 2 [contended], ndb 2, mmorpg 1; signature greps + body-reads; Plan-337 zero-cargo under the T1 hyperthink capture + the seal-game-editor release build) | 1 FILED: riir-ai Issue 1035 — `quest_combat/shot_flight.rs:181` `lead_of_int` re-states the `game_traits::sigmoid` body INLINE (born 10-06 23:50 `a9528d0e1`, AFTER 1027's same-day close `e8735a83b` — the miss class again) while the SAME plan-317 landing's `hero_threat_memory.rs:37` delegates the canonical; classified the COPY-GATE class, NOT the 1027 numerics class — the canonical home (`game_traits/mod.rs:104`) carries the IDENTICAL branch-free expression, so delegation is bit-identical/zero-behavior (residual risk = silent future divergence; menu: delegate | recorded-refusal + divergence-failing pin). REST CLEAN: `hero_threat_memory` = textbook consume (Sighting + game_traits::sigmoid + MapPos3D, two-brain documented in-source); hero_routine σ-constants = cfg(test) monotonicity oracle (FP); deliberation/plan_repair inline sqrt on MapPos3D fields = note-level (Plan-1032 DEFAULT-DISABLED lane, the sim.rs:469 POC bar; compass-const sites = grid math); traj_store `sigmoid_dot_8` DELEGATES katgpt_core::sigmoid with "never softmax" in-source (sanctioned); dao guard.rs integer cosine = documented deterministic integer-Newton arm (G1, in-source); dapps `[..16]` = the documented evt-key/row-decoder families + wire-contract DOC prose; editor/seal sqrt = view-layer classes + the leaves `1-exp(-rate·dt)` literal-pose twin pair; belief-struct hits = `HeroThreatLane`/`MonsterMemoryRow.last_known_pos` (the memory substrate's own accessor) + MonsterThreatSource (the skill's own FP) + QuestRestockBelief R12; infer exp = model-forward (softmax/perplexity/deltanet decay). **Prior-filing loop VERIFIED CLOSED: reflex Issue 071 (SplitMix64 dual stream) closed same-day-as-filing 10-06 09:36 `b94d21d` — katgpt-types substrate export `514989664` + reflex consumer delegation `147309c`; the 071 close postdates the 10-06 audit row (09:02) by 34 min** | katgpt-rs (this commit) |
| 2026-10-06 | Mode 2 — the 10-03 16:00→10-06 wave (506 fresh .rs across 19 repos, vendor/ excluded: dapps 85 [the Issue-118 deposit-cursor fix landed — file itself still on disk, repo claimed by a live pack/pay lane], katgpt-rs 65, editor 64, ai 37, infer 36, train 30, seal-remake 24, rethink 23 [birth-seed carried — the 10-03 moat-split sanction holds; examples = trainer-mirror class], clippy 23, reflex 22, instinct 21, dao 16, ndb 15, reflexer 10, chain 10, kat 8, sdk 8, deployer 5, shader 4; signature greps + body-reads; Plan-337 zero-cargo under two live sibling lanes) | 1 FILED: riir-reflex Issue 071 — `src/harness/` carries TWO structurally-identical SplitMix64 streams (`suites.rs:475` pub(crate) + `runner/echo_gates.rs:56` private, the LATER `0b4a351` re-implementing a `pub(crate)` type one module over) — intra-crate DRY primary; substrate exports no reusable form (`katgpt_types::rng::Rng` is XorShift64 with splitmix as ONE-SHOT seed-mixing only — the stream is NOT splitmix; `katgpt_core::types` re-export makes a future `splitmix64_finalize`/`SplitMix64` export zero-dep consumable), the workspace strict-family copy-gate context named in-issue (seal-view viewer.rs + reflex ×2; odd-multiplier-only forms — chain `mcp/supervisor.rs:1466` placeholder, train `sil_replay.rs:133` Knuth stride — classified looser-family note-level). REST CLEAN: katgpt-core `dllm_solver.rs:377` + ai limelight/mod.rs:108 sigmoids DELEGATE `fast_sigmoid` with rationale (the 09-30 sanction re-verified); rethink trainer examples seed-carried (`--diff-filter=A` = birth/d0f41df, the frozen-mirror class); distance fns = TV-distribution oracle + harness posture tests + view-layer classes; spatial-HashMap hits = editor DB-row keys + ambient scatter grids (view) + FA-minimization state pairs + substrate-side linking_detector voxel buckets (topology domain, substrate home); `[..16]` = dapps deposit/pay evt-key + row-decoder families (documented) + display truncations; belief-struct hits = `MonsterThreatSource` (the skill's own named FP) + substrate itself | reflex `50ddbb7` (this commit) |
| 2026-10-03 | Mode 2 — the 10-01 ~16:00→10-03 wave (649 fresh .rs across 18 repos + second-root seal-online-remaster 0; census `git log --since` derived, vendor/ excluded; signature greps + body-reads; Plan-337 zero-cargo under the sibling Issue-915 two_fidelity BAI measurement + the Plan-614 4090 lane) | 1 FILED: riir-ai Issue 1027 — `vessel_attestation.rs:510` one-branch libm `sig()`, born 09-11 `6bd404350` (miss class, first caught now), native-only module under an UNCONDITIONAL katgpt-core dep (Cargo.toml:770), no rationale/pin, consumed by the Plan-618 T2 classified-capture lane — while the SAME arc's `abstraction_producer.rs:128` DELEGATES `katgpt_core::sigmoid`; no cross-build bit commitment binds the current form (classified-vs-plain + within-boot determinism pins are intra-build, anchor asserts carry 1e-6/1e-5 tolerances); secondary note-level instance `riir-games/combat/sim.rs:469` trial_log heuristic (born 06-05, riir-games hard katgpt-core dep :1605, POC feature). REST CLEAN: thermal_lod/mod.rs delegates `katgpt_core::simd::fast_sigmoid` + NaN convention (textbook); engram_runtime_store = documented single-accessor design choice over papaya + fixed-8 unrolled dot (perf rationale, note-level); abstraction_producer's inline dot+norm = fused single-pass (sanctioned shape); splitmix census UNCHANGED (0 new consumer-side production arrivals: scatter.rs consumes `luck::seeded_unit_roll` single-home; covariability_boot det_noise + pav_data u01 + dflash2 recomposition + editor pseudo-BLAKE3 = test/display-fixture classes; seal-view viewer.rs stays the lone note-level); moat split carried the encoder_arm frozen-mirror sanction into riir-rethink unchanged; distance/sqrt hits = view-layer + heightfield-occlusion (vision.rs IS substrate, documented AOI mirror) + POC tuple-pos sim (squared-distance idiom correct); exp hits = softmax/device/corpus-fixture/IRLS/softplus classes; belief-struct hits = substrate vocabulary in action (ThermalBeliefScalars over the synced 5, QuestRestockBelief R12, MonsterBeliefs domain) | riir-ai `b1a820e52` (this commit) |
| 2026-10-01 | Mode 2 — the 09-30→10-01 ~16:00 wave (439 fresh .rs across 17 repos, vendor/ excluded: editor 99, ai 82, clippy 78, dapps 35, seal-remake 33, train 22, infer 22, instinct 18, katgpt-rs 13, reflex 11, chain 11, sdk 9, ndb 5, reflexer 1; signature greps + body-reads; Plan-337 zero-cargo under the sibling's bench_1004 e2e measurement — the exact slot the 16:0x pass deferred this unit FOR) | CLEAN — zero filings; **2 prior filings VERIFIED RESOLVED**: reflex Issue 014 (engine.rs now `use katgpt_core::exact_sigmoid`, :60/:1090) + reflexer Issue 002 (`exact_sigmoid_f64 as sigmoid` + the `sigmoid_delegation_matches_frozen_legacy_body` pin). New-substrate consume-loop closed on all three katgpt-rs arrivals: perturbation_ensemble → reflex ×5 files; grouped_evidence/filter_bias_bound → clippy frontier_report.rs (139 T3); dirichlet_dist → zero consumers AND zero reimplementations (the recorded unscheduled-consumers posture, kill clock 10-31). Notable adjudications: instinct encoder_arm.rs:168 + train instinct_encoder_lane.rs:258 identical local one-branch sigmoids = the bit-faithful frozen-artifact mirror class (NLEH codec, in-source accumulation-order rationale + the Bench-029 316/600 serve-parity pin; Cephes delegation = semantic change); ai game_traits/alarm.rs = substrate MOVE (riir-stealth → games-shared layer-0, "one implementation, two consumers"; `last_known` documented raw-detector-report, NOT belief reconstruction); dapps decstat.rs splitmix shuffle = consumer-side arrival #3 (replay.rs twin, below the 4-arrival copy-gate, note-level); riir-ai "PSSA" = Research 598 docs-only (NO-FUSE) — no code substrate to drift from; exp/distance classes all documented FP (model-forward, view-layer easing, grid-space game math, doc prose, corpus fixtures, hero_routine const-folds). **Method lesson (3rd instance of the flag-misuse class): `xargs -a` is GNU-only — macOS BSD xargs rejects it (`invalid option -- a`) and a stderr redirect silences the tell, yielding a confident ALL-ZERO census until the path-existence check caught it; portable form `xargs < file`** | katgpt-rs (this commit) |
| 2026-09-30 | Mode 2 — the 09-24 03:00→09-30 wave (1656 fresh .rs across 19 repos, vendor/ excluded: katgpt-rs 750 substrate-side, infer 260 carve/GPU, editor 136, ai 128, reflex 88, clippy 78, train 48, seal-remake 38, instinct 31, …; signature greps + body-reads, blame where fresh-line isolation mattered; Plan-337 zero-cargo under the sibling Tetris measurement) | 1 FILED: riir-reflexer Issue 002 — `engine.rs:311` module-local ONE-BRANCH f64 `1/(1+exp(-x))`, no rationale/pin, beside `katgpt_core::exact_sigmoid_f64` (added FOR that caller class, Issue 156); args `(v−vmax)/scale` ≤ 0 ALWAYS — the branch where one-branch vs two-branch differ by rounding on EVERY call, and the probs feed `Answer::choice` (G1 champion-replay bit-identity pinned) → adjudication-gated (the chain-156 menu: delegate+pin OR recorded-refusal+to_bits), detection-only. REST CLEAN: riir-ai limelight/mod.rs:101 + mind/arbitration.rs:433 sigmoids DELEGATE fast_sigmoid with NaN-convention rationale (sanctioned); orchard systems.rs delegates `crate::motivation::sigmoid` (single crate home); infer dspark_drafter = bit-faithful llama.cpp reference port (delegation = semantic change); reflex nli_m1 = f64 IRLS logistic-fit instrument; train examples/tests = backward-mirror + cfg(test) classes; dapps fixstats/staking `[..16]` = row DECODERS (canonical `account_for_pubkey` lives in riir-kat, consumed via `pub use riir_kat::kat_protocol_*`); editor/seal `HashMap<(i64,i64)>` = DB row-key joins (feat-map `(i32,i32)` buckets = editor-specific viewport, no substrate in repo); belief-struct hits = substrate vocabulary in action (predator_fsm `last_known_pos` consumes it) + documented R12 `QuestRestockBelief`; splitmix = bench fixtures + the 09-21-documented seal-view viewer.rs id; hero_routine `1/(1+exp(-c))` = compile-time constant folds; distance fns = emotion/semantic domain math, not MapPos3D re-impls | reflexer `221c321` |
| 2026-09-24 | Mode 2 — the 09-23 20:00→09-24 ~02:4x overnight wave (93 fresh .rs across 7 repos: shader 47 view-layer effects/lava/dispersion, reflex 22 laya-Metal+serve [sibling-HOT lane, as-found], dapps 14 Plan-038 card/role-flows, clippy 6 corpus+B174, seal-remake 2 smell-hunt doc markers, mmorpg 1, ai 1) | 1 FILED: riir-reflex Issue 014 — `engine.rs:624` module-local one-branch sigmoid, no rationale/pin, in the file whose import block (47-54) consumes FIVE katgpt_core substrate items; delegation to `exact_sigmoid` bit-identical on the reachable domain (dot·route_scale ≤ ~8, s/temperature small; tail x<−88.7 diverges 0.0-via-inf vs tiny), the ndb-611/chain-156 class — engine.rs shipped at birth 31207ec, the 09-22→23 wave read reflex CLEAN (miss class); detection-only, sibling-active lane. REST CLEAN: shader radial-falloff hit = view-layer FP (no substrate in repo); dapps routes.rs hits = wire-contract DOC prose; clippy kat_sync_payload [..16] = Issue-082 canonical mirror docs; laya Metal kernels = model-forward math in the live lane (not audited beyond signature scan) | reflex `64a7525` |
| 2026-09-23 | Mode 2 — the 09-23 04:30→20:00 window (committed-.rs census: riir-ai 132 touched = 121 carve deletions + 11 present; riir-infer 283 fresh [132 vendor/ cubecl-runtime+wgpu-hal excluded per 738 T3]; katgpt-rs 6; clippy 9; shader 8; train 9 read-only; mmorpg 1) | CLEAN — zero filings. The carve P3 slices 3/4a/4b/5 verified DRY-correct MOVES end-to-end: 121/121 riir-ai deletions accounted (117 same-name → riir-infer-gpu: 75 byte-identical + 42 mechanical-edit [riir_engine→riir_infer_core import renames, pub(crate)→pub with documented visibility notes, documented cfg posture splits, CRLF noise]; wall_config re-home → riir-infer-core w/ same-path re-export at riir-engine/lib.rs:87; backward.rs cross-repo move → riir-train `2d0eab54` [0-line non-comment diff — training code to the training repo, the modelless boundary working]; build.rs documented orphan removal `892dec017d`; bench_663 test move) — zero dropped, zero rewritten, zero parallel copies. Plan-611 split-rung kernel (142 new PTX lines in gemv_ternary_cuda_raw @ riir-infer) = device code, FP by construction, lane measured NOT promoted. goal_salience soak lane (Issue 1002) = textbook consume: SalienceSoakParams re-export + Option boot field (None = bit-identical) + install_swarm_optionals wiring + soak readout; zero new math — every sqrt/dx² hit in the touched files PRE-EXISTING by window-diff isolation. katgpt-rs Issue-875 T3 horizon_weights.rs = NEW substrate carrying its own in-source Mode-1 vocabulary check (future-looking remaining-horizon vs tether::horizon_decay past-looking; wires INTO renoise_ce; riir-train pfd_toy is the first consumer — the consume loop closing same-day). clippy 9 (Issue-102 active_set/gate_calibration/necessity + P38 orphan_report) = instrument/CLI over existing stores, lone exp hit a corpus-rule comment; shader 8 = view-layer leaf (vec3 dot, digest-id [..16]); train sigmoids = backward-must-mirror-forward numerics (the 09-13 DFlash class) + documented-deterministic toy RNG. Method lessons: basename-collision cross-matched gemma2↔gemma4 dispatch/kv_cache/weight_buffers (4 false BIG diffs → 0-diff under full-relative-path matching — verify moves by PATH, never basename); CRLF inflated one move-diff to 2103 lines (2 real under --strip-trailing-cr — a Windows-box law) | katgpt-rs (this commit) |
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
