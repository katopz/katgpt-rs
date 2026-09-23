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
| 2026-09-24 | Mode 2 — the 09-23 20:00→09-24 ~02:4x overnight wave (93 fresh .rs across 7 repos: shader 47 view-layer effects/lava/dispersion, reflex 22 laya-Metal+serve [sibling-HOT lane, as-found], dapps 14 Plan-038 card/role-flows, clippy 6 corpus+B174, seal-remake 2 smell-hunt doc markers, mmorpg 1, ai 1) | 1 FILED: riir-reflex Issue 014 — `engine.rs:624` module-local one-branch sigmoid, no rationale/pin, in the file whose import block (47-54) consumes FIVE katgpt_core substrate items; delegation to `exact_sigmoid` bit-identical on the reachable domain (dot·route_scale ≤ ~8, s/temperature small; tail x<−88.7 diverges 0.0-via-inf vs tiny), the ndb-611/chain-156 class — engine.rs shipped at birth 31207ec, the 09-22→23 wave read reflex CLEAN (miss class); detection-only, sibling-active lane. REST CLEAN: shader radial-falloff hit = view-layer FP (no substrate in repo); dapps routes.rs hits = wire-contract DOC prose; clippy kat_sync_payload [..16] = Issue-082 canonical mirror docs; laya Metal kernels = model-forward math in the live lane (not audited beyond signature scan) | reflex `64a7525` |
| 2026-09-23 | Mode 2 — the 09-23 04:30→20:00 window (committed-.rs census: riir-ai 132 touched = 121 carve deletions + 11 present; riir-infer 283 fresh [132 vendor/ cubecl-runtime+wgpu-hal excluded per 738 T3]; katgpt-rs 6; clippy 9; shader 8; train 9 read-only; mmorpg 1) | CLEAN — zero filings. The carve P3 slices 3/4a/4b/5 verified DRY-correct MOVES end-to-end: 121/121 riir-ai deletions accounted (117 same-name → riir-infer-gpu: 75 byte-identical + 42 mechanical-edit [riir_engine→riir_infer_core import renames, pub(crate)→pub with documented visibility notes, documented cfg posture splits, CRLF noise]; wall_config re-home → riir-infer-core w/ same-path re-export at riir-engine/lib.rs:87; backward.rs cross-repo move → riir-train `2d0eab54` [0-line non-comment diff — training code to the training repo, the modelless boundary working]; build.rs documented orphan removal `892dec017d`; bench_663 test move) — zero dropped, zero rewritten, zero parallel copies. Plan-611 split-rung kernel (142 new PTX lines in gemv_ternary_cuda_raw @ riir-infer) = device code, FP by construction, lane measured NOT promoted. goal_salience soak lane (Issue 1002) = textbook consume: SalienceSoakParams re-export + Option boot field (None = bit-identical) + install_swarm_optionals wiring + soak readout; zero new math — every sqrt/dx² hit in the touched files PRE-EXISTING by window-diff isolation. katgpt-rs Issue-875 T3 horizon_weights.rs = NEW substrate carrying its own in-source Mode-1 vocabulary check (future-looking remaining-horizon vs tether::horizon_decay past-looking; wires INTO renoise_ce; riir-train pfd_toy is the first consumer — the consume loop closing same-day). clippy 9 (Issue-102 active_set/gate_calibration/necessity + P38 orphan_report) = instrument/CLI over existing stores, lone exp hit a corpus-rule comment; shader 8 = view-layer leaf (vec3 dot, digest-id [..16]); train sigmoids = backward-must-mirror-forward numerics (the 09-13 DFlash class) + documented-deterministic toy RNG. Method lessons: basename-collision cross-matched gemma2↔gemma4 dispatch/kv_cache/weight_buffers (4 false BIG diffs → 0-diff under full-relative-path matching — verify moves by PATH, never basename); CRLF inflated one move-diff to 2103 lines (2 real under --strip-trailing-cr — a Windows-box law) | katgpt-rs (this commit) |
| 2026-09-23 | Mode 2 — the 09-22 14:00→09-23 window, committed-file census (511 .rs across 10 repos: riir-infer carve 272 + riir-ai carve-coordination 68 + ndb 330 release 89 + katgpt-rs 607/875/876 micro-arena lanes 52 + reflex 33 + shader 24 + editor 17 + dapps 10 + clippy 6 + kat 2 read-only sibling-hot; train 0 committed — mtime churn only; vendor/ excluded per 738 T3) | CLEAN — zero filings; one prior filing VERIFIED RESOLVED: katgpt-rs Issue 870 (`9b09783d9` 09-22 14:30 — distance_abstain delegates `crate::exact_sigmoid` + `sigmoid_delegation_matches_frozen_legacy_body` envelope pin + bench_845 re-run). riir-infer carve = DRY-correct MOVE, not a rewrite: root crate consumes katgpt-core/transformer/speculative/forward/quant/attn path deps with in-source rationale, gpu crate workspace-deps katgpt-core, origin crates emptied at the Plan-610 commits, all signature families zero over its 272 files. katgpt-rs σ/exp hits = substrate homes + test/bench fixtures + softmax/LSE model math — blame-isolated ALL pre-existing (mtime churn from FF pulls); ndb `value_tag` = content-commitment family (not identity derivation); dapps/kat blake3 = wire-contract DOC lines; splitmix = fixture + substrate-side dual.rs (below copy-gate). Method lesson: stat-mtime census over-counted 800 vs 511 — `git log --name-only --since` is the truth set | katgpt-rs (this commit) |
| 2026-09-22 | Mode 2 — the 09-21 18:00→09-22 14:00 fresh wave (katgpt-rs reflex arc source_adapter/source_features/decision_wire/compression_drafter/distance_abstain + bitcos/ternary SIMD + closure bridge + speculative/forward/moka lanes; ai riir-gpu pool_poison/vram_budget/readback taps; clippy proposers ×3 + rule_embed/fixseq/horizon; sdk vessel pack/reauthor; mmorpg one test; vendor/wgpu-hal excluded per 738 T3; dapps/kat/auth/chain/ndb — no fresh .rs) | 1 FILED: katgpt-rs Issue 870 — `distance_abstain.rs:47` single-branch libm σ, no rationale, no pin, in-crate beside THREE sanctioned patterns (closure/bridge.rs delegates fast_sigmoid; d2f delegates; 6 modules consume exact_sigmoid); divergence envelope bounded (bit-identical x≥0, ≤1 ULP in the gate domain [−10.8, 5.2]) — fix = exact_sigmoid delegation + the Issue-156 permanent pin + bench_845 re-run. REST CLEAN: rule_embed.rs:71 two-branch σ = bit-identical exact_sigmoid copy WITH rationale (consumer-side arrival #1, below the 4-arrival gate — note-level); BeliefDraft* = speculative-decoding vocabulary FP (not GenericSpatialBelief); exp/sqrt hits all documented FP classes (softmax priors, RMSNorm, 1/√hd ×5, Box-Muller, Cholesky/cosine residuals, PUCT UCB, Xavier init); corpus-string sigmoid greps in kernel_opt entries = fixture class; zero spatial-HashMap/belief-emotion-struct/BLAKE3[..16]/splitmix arrivals in any fresh production file | katgpt-rs (this commit) |
| 2026-09-19 | Mode 2 — delegated subagent pass over the 7 quiet repos' math/crypto/concurrency primitive shapes (sigmoid / Beta-LCB / Elo / argmax / RNG / dot families; 1,274 files, vocabulary-shape greps + BOUNDARY-charter adjudication) | 4 CANDIDATE-DRY, ALL in riir-chain's consensus/curator layer (`curator_bridge.rs:14/:27/:38` + `curator_reward.rs:100` exact-form sigmoids + dot; `congestion.rs:62` + `forensic/recover.rs:147` weak-f64) — FILED chain Issue 156, detection-only, adjudication-gated on the Cephes-vs-libm numerics caveat (the standing-lesson class). Six other repos CLEAN; notable positive delegation: ndb (`sigmoid`/`simd_dot_f32`/`cmp_for_max`), dao (`beta_lcb`/`rating`), clippy (`elo.rs`/`drafter.rs`) | chain `ec712ed5` |
| 2026-09-18 | Mode 2 — the 09-17/18 fresh waves (katgpt-rs FlashMemory NIAH bench_685 + dash_attn wave + 819 prior/sink lanes + data_probe sink_classify; riir-train bonsai_teacher/cvrr_boundary/plan403; riir-ai tokenizer whitespace-parity + hero_routine R12 + item_catalog + behavior_gate_poc; riir-shader editor preview/profile waves; riir-kat kat_units MicroKat/WholeKat + kat_meter/kat_account; riir-clippy OaiClient provider ladder + llm_spend + kat_footer; riir-dao envelope_gate µ-typed pins; riir-deployer genesis vest; dapps+editor fresh sets re-grepped read-only) | CLEAN — zero parallel systems, zero filings: both fresh production sigmoids DELEGATE (block_topk → `katgpt_core::simd::fast_sigmoid`; sink_classify → `crate::simd::fast_sigmoid`; 2 test-local = FP class); `QuestRestockBelief` = documented think-brain R12 (NOT SleepTimeAnticipator — wall-clock EMA vs latent catalog, local/not-synced, translated in-source); riir-train CONSUMES `riir_engine::tokenizer` wholesale (bonsai_teacher T1.1 pin in-source; the fresh parity test gates the canonical); riir-clippy provider lanes = ONE `OaiClient` over `EndpointConf` (groq→infron→CF), DRY-correct; riir-kat `blake3(pubkey)[..16]` = doc mentions of the Issue-082 single definition, zero new derivation sites; dapps blake3::hash = content commitments (different family); bench_256 splitmix = substrate-side +1 (below the consumer-side threshold); sqrt hits all FP classes (attn 1/√hd, Xavier, RMSNorm, cosine oracles) | — |
| 2026-09-17 | Mode 2 — the post-100th-boundary-run fresh wave (83 .rs files since 04:00: ndb Plan 599 T4.1 `retrieve_diverse_counter_anchored` + bench_599; riir-clippy corpus_coverage/Issue-119 Bench-094 waves + doc_lazy_continuation/cast_lossless proposers; riir-shader frame_time/capture_mode T2.1; katgpt-rs set_admission_freeze + argmax-dispatch benches; dapps kat/ledger.rs Plan-034 deposit rows; riir-ai gpu prefill/deltanet/tree-verify) | CLEAN — zero parallel systems, zero filings: ndb 599 = textbook consume (`use katgpt_core::set_admission::{SetAdmissionConfig, certify_set}` wholesale; bench-local fns = fixture class); katgpt-rs hits substrate-side + test-only; riir-ai gpu hits all `from_raw_parts` buffer plumbing (std API, not a drift signature — pattern over-broad, noted); dapps ledger `blake3(..)[..16]` ×3 = the row-KEY family (`PREFIX + blake3(composite)[..16]`, the kat:*evt: exactly-once idiom) NOT the BLAKE3(pubkey) identity-derivation family — the ×2 same-body mirror pair (MINTEVT/DEP_EVT) is documented in-source as the cross-rail mirror check, note-level; clippy [..16] = display truncations | — |
| 2026-09-17 | Mode 2 — the post-previous-row waves (Plan 599 set-admission Phases 0–3 + bench 811; 813/816 set-causal denoiser arc; ai 965 prefill CUDA graphs; dapps Plan 034 T5 SEAL deposit plane + worker routes + chaos; riir-kat deposit wire + SIWR client + render_units; seal-remake wallet panel/QR/keychain) | CLEAN — zero parallel systems, zero filings: Plan 599 consumes `vendi_diversity` + `systematic_resample_into` + `jacobi_eigen` wholesale (exp-tilt documented as "the deterministic arm over the substrate", line-1507 call verified), Sherman–Morrison = documented incremental-vs-full-inverse design choice, snap argmax = fused cosine pass (not dllm argmax misuse); 813/816 reuses incumbent kernels ("no kernel/ctx/backward change", `fast_exp` consumed; 1/√hd = model-forward FP); ai 965 zero standing-class hits; dapps `evt_key` COMPOSES the canonical `signature_key` (textbook), `derive_reference` = documented Solana-Pay ephemeral on-curve keypair (new domain; the 3-line on-curve idiom vs `new_payment_request` = note-level, below copy-gate); kat SIWR client = pure wire shapes with the sign half REFUSED in-source ("a second format to drift"), `render_units` = parameterized-general-form over `render_kat`; seal wallet QR = `qrcode` crate, custody = `keyring`, SIWR via `kat_siwr_client` — no derivation-family (BLAKE3[..16]) hits anywhere | — |
| 2026-09-17 | Mode 2 — the 09-16/17 fresh waves (seal Plan 046 brain-gen 9-state FSM — surface now in seal-remake after the seal-core/seal-edge-worker migration; ai 952 §C desperation_dual + HlaScalarFeed consumer; chain Issues 152–155 drain/overflow fix; katgpt-rs Plans 600/601 flashar confidence-commit + real-text eval; riir-clippy doc_markdown + P31 manual_range_patterns corpus wave) | CLEAN — zero parallel systems: Plan 046 is the textbook consume (substrate `riir-games-shared::npc_brain` owns NpcAction + the deterministic factory; consumer `npc_inference.rs` is a re-export shim, the FSM mapping is ONE contract-documented free fn `npc_action_to_ai_state` — orphan-barred inherent impls named in-source); 952 §C consumed `katgpt_core::dual` (not re-derived), `projection_to_affect` is the single struct→`[f32;5]` site (bridge stays raw-float decoupled by documented contract; zero production `set_hla_scalars` callers yet = the issue's own dormant record); 600/601 delegate fast_exp/argmax/sampling/pruner to katgpt_core with bit-parity rationale; chain 152–155 = pipeline correctness, no standing-class surface | — |
| 2026-09-16 | Mode 2 — the 09-15/16 fresh waves (riir-kat fix-stats wire/push + /history + replay clients + the verdict-ack arm; ai 886 T5 device-gate harness; riir-shader bevy-0.19 migration ×27 files; katgpt-rs bevy_ecs 0.19 bump monopoly+bombers; esp32 satellite-probe; editor item-placements wave ×19 files; dapps 930949a test gate) | CLEAN — zero parallel systems: riir-kat `account_for_pubkey` verified as the Issue-082 single definition (doc: "TWO-SIDED by construction… now THIS is the only definition"; mirrors retired IN) + the `the_service_account_id_is_the_payreq_derivation` cross-pin with an explicit non-vacuity guard (local 64-hex ≠ service 32-hex); fixstat run-id blake3 = idempotency key, NOT the derivation family; dapps Issue 086 golden-vector finding verified RESOLVED (file removed per noise rule); esp32 probe IMPORTS `riir_wallet_signer::identity::player_id_hash` — the exact canonical the consolidation lesson named, textbook consume; T5 sqrt = cosine-similarity test oracle (FP class); shader/feat-map exp+sqrt = the view-layer FP classes (golden-ratio const, vec3 normalize, `1-exp(-rate·dt)` easing, editor click-radius needing true distance for its take(8) sort — the SceneSpatialIndex consumer-specific precedent); placement.rs tile→world = documented linear transform, data-layer (no substrate in scope) | — |
| 2026-09-15 | Mode 2 — the 09-15 fresh waves (dapps Proposal 004 arc: fixstats/telemetry/mint_receipt v2/sankey + worker contract files; dao 25d7d2d telemetry v2 consumer; ndb free_energy_ledger + norm_matched_noise; mmorpg-remake visual_polish/lighting_panel) | CLEAN — zero parallel systems: fixstats verify CONSUMES `submitter_for` (the riir-kat leaf single-sourcing); the stats fold RE-EXPORTS `riir_neuron_db::MerkleTree` ("no mirror, no drift", in-source) and rides the existing anchor_root settlement; ndb free-energy consumes `katgpt_core::slt::rlct_reduced_rank` (Issue 620's purpose); mmorpg-view zero standing-class hits (view layer). 1 sanctioned dual-pin verified, not filed: dao `telemetry_row_hash` ↔ dapps `row_hash_of` — byte-identical domain string + canonical bytes, golden-pinned BOTH sides (`dapps_contract_pins.rs` ↔ `kat_telemetry.rs`), the Plan 003 committed-DATA pattern (dao cannot dep dapps); divergence fails both goldens | — |
| 2026-09-14 | Mode 1 — Issue 941 `mb_personality_shard` (the mb_value snapshot → NeuronShard materialization arm, the 940 follow-on) | CLEAN CONSUME of the PATTERN — the KarcShard `Wout` precedent (learned per-NPC matrix → dedicated sibling Pod) applied to `w`; vocabulary translation was the whole task: "materialize into a NeuronShard entry" reads as base-shard until the dims (1,536 f32 vs `style_weights[64]`) force the sibling-Pod reading the repo already shipped twice (KarcShard, ArchetypeBlendShard). `zone_hash = BLAKE3(npc_uuid)` identity + prefix commitment + batch Merkle + explicit pad ALL consumed verbatim from the sibling (zero new discipline invented); the "dendritic-branch-compatible" phrasing decoded to "w bytes Pod-ready `[f32]`" (overlay-applicable), NOT a structural 16/24/24 mapping. Chunked base-shards rejected (zone-registry abuse, 24 rows/NPC) | ndb `ae641f0` · ai `52a60e191` |
| 2026-09-14 | Mode 2 — the 09-13/14 riir-shader close-out wave (sessions 30+31 `5bbbfe8`/`e2dd6d6`/`d23f770` air-painting + three-tsl final ports, `1bbe1df` hot-reload `shader_src!` lane (92 sites + hot.rs), `8973960`+`5a4f242` showreel owner follow-ups; 12 fresh .rs files) + the mmorpg-remake/ai micro-wave (mmorpg-remake `90e34e30` one-boolean harness boot flip, ai `8e748e936` re-export deletion) | CLEAN — zero parallel systems; every exp/sqrt hit in the fresh set is the documented view-layer false-positive class (frame-rate-independent easing `1-exp(-rate·dt)` in showcase smoothing; quaternion/vec3 normalize + screen-space SDF triangle geometry + golden-ratio constants in render math) — riir-shader is a zero-riir-dep wasm effects leaf with NO MapPos3D/sigmoid substrate to delegate to and no semantic domain; zero hits on `fn sigmoid`, spatial-HashMap, belief-struct, BLAKE3[..16], splitmix64; the micro-wave is a config flip + a deletion (nothing to audit); companion doc fix: shader AGENTS.md "Active plan" line updated to COMPLETE 28/28 | shader (doc line) |
| 2026-09-13 | Mode 1 — Issue 940 `mb_personality` (the mb_value first consumer: per-archetype frozen circuit + per-NPC learned `w` overlay at the swarm adoption gate) | CLEAN CONSUME — `katgpt_core::mb_value` wholesale (MbCircuit/MbScratch/code_into/value/dopamine_update/saturation/set_w; toy config consumed VERBATIM, zero new circuit config); the `temperament_ladder` consumer pattern mirrored (think-brain-only, one gated scalar into an existing decision slot, bit-identical degeneration, Option-field kill-switch); sigmoid gate via `katgpt_core::sigmoid` (the house bridge); freeze = BLAKE3 over `to_bits` (bytemuck NOT pulled — stays optional); zero parallel systems (no existing LEARNED per-NPC value substrate — temperament is seeded-static, motivation utility is computed, KARC readout frozen-at-freeze — the learning axis is the issue's own slot). Harness lesson recorded: the action code needs a ZONE axis — (elev,dist) bands smear credit across reward neighborhoods (the vocabulary lesson inverted: the ENCODING must carry the axis the claim varies on) | ai `28d557137` |
| 2026-09-13 | Mode 2 — the 09-13 evening wave: dapps `21d6524` (Plan 026 corpus view + worker wiring) + ai `cd8b4e78c` (Issue 886 T1–T4 ANE headroom) + clippy `c4dcff13` (docs/scripts-only lane retirement — no substrate surface) | CLEAN — zero parallel systems: corpus_view = instance #4 of the materialized-view discipline (pulse/mine/mint precedents named in-source), CONSUMES `CommitBatch`/`CommitLevel` + `MiningQueue`/`QueueRow` + the crank-token `bearer_gate`/internal-route family; the monotone max-upsert matches the established per-instance inline pattern (mining.rs:1515); headroom.rs = the workspace's ONLY sysctl/host_statistics64 probe (no parallel memory substrate to consume), DownLadder/down_admits defined once in-lane (ane_prefill/mod.rs:416/481), SDK-verified constants + recorded deliberate divergences; zero standing-class hits (sigmoid/exp/distance/spatial-HashMap/BLAKE3[..16]/splitmix64) in both fresh files | — |
| 2026-09-13 | Mode 2 — the item-deploy emit-side wave (dapps `e025ec8` + editor e2e_item `9fb2fb3b` + deployer `00c05c9` + mmorpg-remake `53c8ae6e`; Plan 192 T4.2/T4.3) | 1 FILED: dapps Issue 086 — the attribution derivation (`account = blake3(pubkey).to_hex()`, the riir-auth mirror) has NO fixed golden vector anywhere: dapps' `account_id_matches_the_riir_auth_derivation` is tautology-class (formula restated inline — catches local edits, not substrate drift), the `GOLDEN_MANIFEST_JSON` pair (3333/4444) is derivation-INVALID by construction (the fixture pins SHAPE only — the substrate-table row overstates), deployer's test fixtures are self-consistent; the real pins are the editor's substrate-consuming `debug_assert_eq!` (deploy.rs:360-361) + live-wire 400-attribution refusals + riir-auth `cross_layer_parity` — substrate drift surfaces as a wire incident, not a red test. REST CLEAN: wire shape tri-pinned (worker `item_contract.rs:52` ↔ editor pinned-shape test ↔ deployer probe + live devnet 7-deploy proof), `build_parts` shared by both editor lanes (no parallel bundle construction), editor consumes `account.account_id()` (substrate path), deployer copy justified (zero-sibling-deps hard rule, the `lineage.rs` precedent), zero standing-class hits (sigmoid/exp/distance/splitmix/BLAKE3[..16]/HashMap-spatial) across all fresh files; mmorpg-remake deploy.yaml config-only | dapps `915a159` |
| 2026-09-13 | Mode 2 — the riir-ai 938/DFlash wave (961a5ff25 KVCA weight-epoch arc + 34f0730a7 token-path fix + 0e33b14a6 DFlash saved_mlp_hidden + 6bea6d201 per-position activation saves + e3615bd6b 515 guards + c22001269 heal; 16 files) | CLEAN — zero parallel systems: every exp/sigmoid form is MODEL-FORWARD math (GDN recurrence gates β=sigmoid(b_raw)/softplus decay at dispatch.rs:1046-62, SiLU(z) at :1123, SwiGLU at :1201, softmax components in gemma2/transformer reference oracles + kernels — the bit-faithfulness class, Cephes-vs-libm delegation would be a semantic change); `EmotionRouter` = model-consumer (argmax over EmotionAxis::ALL via profile.get_axis, 6 pre-loaded KV caches, zero emotion math); quest_style_bridge L2-normalize has NO exported substrate to consume (composition.rs `l2_normalize` is a private nested fn — arrival #2 in private contexts, below the copy-gate 4-consumer threshold, note-level); `WeightEpoch` CONSUMED from katgpt_types as committed; no spatial-HashMap/belief-struct/distance-math arrivals | — |
| 2026-09-13 | Mode 1 — Issue 763 `lif_graph` (signed-graph LIF reservoir POC) | CLEAN BUILD-NEW (no substrate exists) — LIF/membrane/refractory/spike-propagation greps across 4 on-box repos: every "spike" hit metaphorical (curiosity/loss/transient-spike prose), zero dynamics substrate; ridge readout CONSUMED (`linalg::ridge_solve_direct_f64`, the KARC precedent — `lif_graph` joined the linalg cfg any-list at birth); CSR = new weighted signed home (`EngramKgCsr` downstream KG-specific wrong-direction, `dirichlet` unweighted pairs); fixture RNG = module-local pub pattern (the `interpolation_geometry::FixtureRng` precedent) | lif `74fe08f1` |
| 2026-09-13 | Mode 2 — post-audit wave in free repos: sdk `f81b4b9` market_gold soak compaction (riir-ai Issue 928 validation over substrate cold tier) + `7f0bed2` demo TICK_FIELD re-seed + `b643587` viz ResourceLow label | CLEAN — zero parallel systems: the soak harness CONSUMES `riir_games_mmorpg::secondhand_market` (`compact_terminal` = substrate method call; compact_every/horizon = harness config, the sanctioned model-consumer shape); the demo fix lives in the documented inline demo file; the viz fix is a match arm on a substrate enum; no local sigmoid/exp/distance/sqrt, spatial-HashMap, belief/emotion-struct, BLAKE3[..16], or splitmix64 arrivals | — |
| 2026-09-20 | Mode 2 — the 09-13→09-20 week wave (seal-core/seal-edge-worker migration `b0bd57c19208`, npc_brain extraction landing, dapps TUNA/deposit lanes 09-17/18, shader gamefx graph_host; 1158 fresh-touched files across 11 repos, signature greps + per-hit blame to isolate FRESH lines only — 18 fn-signature + 106 [..16]/rng hits, 13 fresh, all classified) | CLEAN — zero parallel systems: seal-core `SealAffect::sigmoid` + games-shared `npc_brain::observe` sigmoid both STRUCTURALLY FORCED (katgpt-core reachable only via opt-in features — seal-core `sdk_consumer`, npc_brain's vocabulary half is the unconditional zero-dep layer-0 per the Issue-682 dep-weight law; both carry house-rule comments); riir-gpu parity-test cosine = cfg(test) Check-D exemption; dapps ledger/deposit `[..16]` = domain key CONSTRUCTION over the canonical riir-kat derivation (routes.rs hit is the wire-contract DOC); seal-edge-worker id_derivation = the test-enforced cross-layer parity twin; treasure-box LCG = game-content determinism RNG (domain); 15 older hits = pre-existing lines in touched files, outside wave scope | — |
| 2026-09-21 | Mode 2 — the post-116th-boundary-run FF'd fresh material (chain `b7e1ad8b` curator_bridge lint heal, dapps `ce8b428` non-unix buf silencer, train `0dfdfc71` bench interleaved-timing + docs-hygiene commits, riir-ai 887-T2 test/docs + `e4e17b4be` clippy sweep, clippy upstream ×3 docs) | CLEAN — zero parallel systems, zero filings: chain's heal touches only doc backticks + `0.013_7` + a test range-contains — the Issue-156 `exact_sigmoid`/`dot_f32_ordered` delegation and its bit-identity pin untouched; train's timing repair is test-only inline (the skill's stated exemption) citing the canonical 723/831/855 treatment with black_box on both arms; riir-ai sweep mechanical (needless_range_loop enumerate, excessive_precision round-trip f16 spellings, doc continuation), zero standing-class surface — no sigmoid/exp/distance/BLAKE3[..16]/spatial-HashMap arrivals in any fresh hunk | katgpt-rs (this commit) |
| 2026-09-21 | Mode 2 — the 09-20→09-21 fresh waves (242 fresh .rs across 11 repos: katgpt-rs Issue 861 exact_sigmoid delegation arc + bench 844 + set_diffusion/structured_read; chain Issue 156 resolution commits `f7eb85e4`+`b0b710e0`; dapps explorer_p2/p3 + corpus-days sign + tuna; game-sdk riir-vessel CAS lane; ndb dense_embed leakage-audit; deployer manifest/verify; seal-online-remaster gm-tools wave; seal-remake assigned_shaders/vessel_boot; riir-ai goal_salience + attn_fa/dflash2 and riir-clippy retrieval/tier-guard lanes read-only — contended) | CLEAN — zero filings, one prior filing VERIFIED RESOLVED: chain Issue 156 closed by the fresh commits — curator_bridge/recover delegate `katgpt_core::exact_sigmoid`/`dot_f32_ordered` with PERMANENT bit-identity tests vs frozen legacy bodies, `congestion::inclusion_probability` keeps its inline form behind the recorded-refusal marker + to_bits pin (negative-x reachable — the Cephes-vs-libm standing lesson applied as designed); riir-ai goal_salience = textbook consume (`katgpt_core::cgsp::types::sigmoid` + `successor_density_critic` wholesale); ndb dense_embed `dot_product` = 1-line wrapper over `simd_dot_f32`; katgpt-rs Issue-861 copies all delegate, the successor_density link-identity test oracle deliberately INLINE (in-source: independent oracle, assert would be circular). FP-class hits: editor `sample_blake3` splitmix = display fixture ("NOT a real commitment" in-source); seal-remake `view_id64` splitmix finalizer = note-level below the 4-consumer copy-gate (first in crate, in-source documented); dapps tuna `blake3(label)[..16]` = the SEAL plane-column domain-key idiom (not the identity family); vessel CAS blake3 = content-commitment family consuming the artb hash discipline; dapps routes/explorer blake3 hits = wire-contract DOC lines; gm-tools/inspector sqrt = view layer (no substrate in repo); bench_950 FxHashMap spatial = test fixture. Also: the grep tooling lesson re-hit — `grep --from-file` reads PATTERNS not file lists; the first zero-hit pass was flag misuse, redone via xargs before trusting a zero | katgpt-rs (this commit) |
| 2026-09-13 | Mode 2 — the 09-12/13 gallery-playground wave (dapps `f77f13f` template v2 + write-CORS + `417b805` closeout; riir-shader `1f01c79` playground.rs + pattern_demo TRACK_NAMES/arrange_recipe + web panel + harnesses) | CLEAN — zero parallel systems: parse/validate single-sourced in the `riir-shader-pattern` leaf BOTH sides import (the protocol-leaf argument, documented in gallery.rs's substrate table); playground `check_spec` = sanctioned client mirror of the same leaf gates (server authoritative, in-source doc) with the uniform check derived from `pattern_demo::TRACK_NAMES` (the demo's own bindable table — `arrange_recipe` + the gate + the test all read the one const); service `MINTABLE_TARGETS` = documented POLICY PIN (bevy boundary forces the mirror — effects can't enter the worker), template GENERATED from the table + `template_lists_every_mintable_uniform` pins template↔gate, cross-repo drift mode documented (new pattern-demo param = the playground's visible bug); no local exp/sigmoid, distance-math, spatial-HashMap, belief/emotion-struct, BLAKE3[..16], or splitmix arrivals in any fresh file | — |
| 2026-09-12 | Mode 2 — the 09-12 landing wave: Plan 586 secondhand market (ai `679cca71e` substrate + mmorpg-remake `70174c0a` front + `276b3ec5` G4 arm), Plan 587 budget caps (`2ad5ef389`), Issue 925 EVPI gate + model sleep (T1-T4), clippy 089 T5-T8 + 91 T1.5 ring + 092/093 PoC harnesses, chain `7afabe4e` precomposed-tx, dapps item-sync DO | CLEAN — zero parallel systems: market custody-at-list built ON HeroInventory + the documented shop_buy discipline (ZoneMarketBridge = NPC-auction domain, a vocabulary false positive; shop "escrow" = the NFT routing refusal, not an implementation); 587 flip-EMA = consumed `katgpt_core::saddle_escape::FlipDetector` + `convergence_cadence`; 925 delegates `bmr::efe_model_gain` + house sigmoid with a delegation-parity test; frontier consumes `best_belief_score` (T7 half-fold divergence documented in-source); fixseq span_hash = blake3 + discriminating test; gw/nonergodic harnesses import their kernels; chain composes process_tx with C4-parity/C1-refusal pins; no local exp/sigmoid forms in any fresh file | — |
| 2026-09-11 | Mode 2 — Plan 584 KARC-Hebbian three-repo arc + gw_alignment (kat `e92d19f6`/`ae04a98b`, ai `91e31d967`, ndb `dc7ae09`) | CLEAN — zero parallel systems: HebbianKarcReadout = the documented ridge unification (delegates to HebbianKernelMemory::construct; pad64 adapter with bit-identity test); KarcPairJournal HOLDS katgpt_core::DelayRing + spec-mirror divergence test (copy-gate satisfied); ndb two-block envelope composes the generic MerkleFrozenEnvelope::freeze(data_blocks) with the why documented; gw = the only quotient-alignment impl, self-differentiated vs Wasserstein1d/RSA in module docs; score_from_loss local sigmoid sanctioned (f64 + clamp + floor semantics, house-rule comment); splitmix census unchanged (substrate-side only) | — |
| 2026-09-11 | Mode 2 — fresh wave: dapps+dao kat protocol adoption (082: `f2498d2`/`7823daa`), mmorpg `943dce3` consumer follow, chain 141/142 (no substrate) | CLEAN — zero stragglers: no local StagedSwap/LeasedEpoch/account_for_pubkey defs, `kat:burn:v1:` built at exactly 2 canonical sites (riir-kat), dao units = re-export seam + live FREE_TIER_GRANT pin replacing the format twin; mmorpg verified thin via S1-72nd; lesson re-learned: a grep against a nonexistent path returns EMPTY (dao is a workspace — `crates/riir-dao/src`, not `src/`) — verify the path exists before reading a zero as clean | — |
| 2026-09-11 | Mode 2 — fresh wave: mop_homeostasis arc (Plans 579/580/581), Plan 582 flip leg, demo-feel quest_combat, 912 T3/T4 PoCs, Plan 583 fact-edit, Issue 915 hygiene | CLEAN — model-consumer showcase; katgpt_core hmm/mop/saddle/mag consumed wholesale; ndb freeze + hebbian_bridge consumed; splitmix census unchanged (substrate-side arrival only) | — |
| 2026-09-10 | Mode 2 ×4 — attestation commit seam e2e; qsg_crowd (Plan 577); quest_gfn MDP (Plan 368 P1); attack-reasoning/hero-FSM/HeroStats arc (Plans 575/576 + mmorpg-remake 005-007) | All CLEAN — zero parallel systems; qsg kernel consumption TOTAL (inline tilt = the kernel's documented recipe); quest_gfn's own Mode 1 record verified accurate; 2 documented twins (copy-gate class, below threshold) | — |
| 2026-09-10 | Mode 2 — chain attestation-commit module + deployer faucet sidecar | 1 FILED: chain Issue 139 (TxDelta composition mirrors `build_tx` with no equality pin); papaya + single-sourced replay commitment verified | chain `d773915c` |
| 2026-09-09 | Mode 2 ×2 — ndb CLI + auth account_key + dapps burn-ack v3 + clippy client half; dapps burn plane + chain AOI occlusion | CLEAN — CLI seal/unseal justified plane split; burn golden-pin chain INTACT across 3 repos; AOI occlusion mirror sanctioned (D6 symmetry on the chain type) | — |
| 2026-09-08 | Mode 2 — 173 fresh consumer files across 7 repos (092 fallout, pay_request, kat-service, item_nft, viewbridge entities) | CLEAN; splitmix64 census NOTE-level (see lessons); ndb 611 sigmoid delegation HOLDING | — |
| 2026-09-05 | Mode 2 — chain NFT plane (Issues 122-125), dapps item_nft sidecar, deployer ceremony T2.0, mmorpg-remake 092 adoption seam | CLEAN — zero parallel systems; `BLAKE3(pubkey)[..16]` family census 5 gated instances | — |
| 2026-09-05 | Mode 1 gate on sdk Issue 870 (`StatMods` + additive stat totals) | FALSE POSITIVE caught — `StatModifier` substrate EXISTS (clamped `resolve` [0, 2·base] refutes the additive proposal semantically); issue resolved SUBSTRATE-EXISTS; crate-name translation lesson born | ai `ebe8fc224` · sdk `b8c4a50` |
| 2026-09-04 | Mode 2 — post-08-29 wave (092 S1-S3 ~10K-LOC consumer→substrate extraction, dao yield mirror, mmorpg-remake viewer/node) + the 3 copy-class campaigns verified TERMINAL (861 percentile, 867 Elo, 087 triage) + the instrument-fix companion audit of the 11 newly-visible repos | 1 FILED: ndb Issue 611 — 4 production sigmoid copies vs unconditional katgpt-core dep; RESOLVED same day, all 4 delegate; rest CLEAN (`AuthKgTriple` = domain enum over substrate; editor `SceneSpatialIndex` = viewport BVH) | ndb `ffdc38a` + `6ae5435` |
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
