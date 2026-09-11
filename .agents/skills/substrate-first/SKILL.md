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

**Compacted 2026-09-11 (user-directed verbose-history pass; the file was 74 KB).**
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
| 2026-09-11 | Mode 2 — fresh wave: mop_homeostasis arc (Plans 579/580/581), Plan 582 flip leg, demo-feel quest_combat, 912 T3/T4 PoCs, Plan 583 fact-edit, Issue 915 hygiene | CLEAN — model-consumer showcase; katgpt_core hmm/mop/saddle/mag consumed wholesale; ndb freeze + hebbian_bridge consumed; splitmix census unchanged (substrate-side arrival only) | — |
| 2026-09-10 | Mode 2 ×4 — attestation commit seam e2e; qsg_crowd (Plan 577); quest_gfn MDP (Plan 368 P1); attack-reasoning/hero-FSM/HeroStats arc (Plans 575/576 + seal 005-007) | All CLEAN — zero parallel systems; qsg kernel consumption TOTAL (inline tilt = the kernel's documented recipe); quest_gfn's own Mode 1 record verified accurate; 2 documented twins (copy-gate class, below threshold) | — |
| 2026-09-10 | Mode 2 — chain attestation-commit module + deployer faucet sidecar | 1 FILED: chain Issue 139 (TxDelta composition mirrors `build_tx` with no equality pin); papaya + single-sourced replay commitment verified | chain `d773915c` |
| 2026-09-09 | Mode 2 ×2 — ndb CLI + auth account_key + dapps burn-ack v3 + clippy client half; dapps burn plane + chain AOI occlusion | CLEAN — CLI seal/unseal justified plane split; burn golden-pin chain INTACT across 3 repos; AOI occlusion mirror sanctioned (D6 symmetry on the chain type) | — |
| 2026-09-08 | Mode 2 — 173 fresh consumer files across 7 repos (092 fallout, pay_request, kat-service, item_nft, viewbridge entities) | CLEAN; splitmix64 census NOTE-level (see lessons); ndb 611 sigmoid delegation HOLDING | — |
| 2026-09-05 | Mode 2 — chain NFT plane (Issues 122-125), dapps item_nft sidecar, deployer ceremony T2.0, seal 092 adoption seam | CLEAN — zero parallel systems; `BLAKE3(pubkey)[..16]` family census 5 gated instances | — |
| 2026-09-05 | Mode 1 gate on sdk Issue 870 (`StatMods` + additive stat totals) | FALSE POSITIVE caught — `StatModifier` substrate EXISTS (clamped `resolve` [0, 2·base] refutes the additive proposal semantically); issue resolved SUBSTRATE-EXISTS; crate-name translation lesson born | ai `ebe8fc224` · sdk `b8c4a50` |
| 2026-09-04 | Mode 2 — post-08-29 wave (092 S1-S3 ~10K-LOC consumer→substrate extraction, dao yield mirror, seal viewer/node) + the 3 copy-class campaigns verified TERMINAL (861 percentile, 867 Elo, 087 triage) + the instrument-fix companion audit of the 11 newly-visible repos | 1 FILED: ndb Issue 611 — 4 production sigmoid copies vs unconditional katgpt-core dep; RESOLVED same day, all 4 delegate; rest CLEAN (`AuthKgTriple` = domain enum over substrate; editor `SceneSpatialIndex` = viewport BVH) | ndb `ffdc38a` + `6ae5435` |
| 2026-09-01 | Instrument fix, not an audit — Step 2's hard-coded brace list saw 7 of 18 repos | gate blind to 11 repos (2 product-set); set now DERIVED; 227 hits/1.4s vs 221/23s; same class fixed in 4 sibling skills | `60655c48` |
| 2026-08-29 | Mode 1 — twist_smc additions (β-budget selection, ridge readout, ValueMemo, SMC resample) | CLEAN — all four consume substrate (`entropic_tilt::solve_beta`, `linalg::ridge_solve`, papaya+BLAKE3, `systematic_resample_into`); domain-specific ValueMemo justified vs MCTS-cache force-fit | — |
| 2026-08-29 | Mode 2 — riir-deployer FULL tree (ops ladder + control-plane crate) | CLEAN — zero-sibling-dep + RAW-domain discipline TEST-ENFORCED (`zero_sibling_deps_g4`) | — |
| 2026-08-28 | Mode 2 — ~55 fresh mmorpg files + dao FULL src + dapps kat/ expansion + chain registries + ndb anchor_root/local_kv | 1 FILED: mmorpg Issue 087 (cfg(test) inline `Regime`+`triage` after Plan 576 landed) — RESOLVED same day; model consumers dominate | mmorpg `fa4063e` |
| 2026-08-26 | Mode 2 — facade-route + signature-drift audit (Class A `katgpt_core::` reaches; SDK's 3 katgpt-core forwards vs source) | CLEAN — Issue 085 fix held; zero facade signature drift | — |
| 2026-08-19 | Mode 2 — post-08-17 mmorpg wave + sdk entity_sync | CLEAN + 1 minor D1 (monster_pet_debug.rs:216, debug cold path — this row is the record); entity_sync = model EXTRACTION (the direction working) | — |
| 2026-08-17 | Mode 2 — mmorpg Plan-022/539-era fresh code + sdk src | CLEAN overall — 1 minor D1 (pet_teaching.rs:678 → Issue 069 ledger); predation/pet_alarm verified model consumers | mmorpg `61899d3` |
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
