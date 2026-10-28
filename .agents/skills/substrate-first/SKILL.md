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
| 2026-10-28 | Mode 2 mini-audit — the post-102-discharge delta (amo idle-lane continuation; the gap window this cycle's Protocol-H ff opened past the two 07:0xZ rows' coverage: the seal plan-446-step-2 effect-soft-particles wave, 8 fresh .rs across 3 repos [sge feat-map lib.rs +41−2, seal-render bevy_canvas +48−16 / skill_fx trio +169, seal-std seal-fx classify/quads ±8]; signature greps over every delta file + body-reads on every hit; zero-cargo under the 641-T5 soak, Plan-337 posture) | **CLEAN — ZERO FILINGS; the wave is a CONSOLIDATION**: all skill-fx surfaces consume `seal_fx::effect::{flight_frame, …}` wholesale (the single-source-by-leaf law holding — the canvas scales a returned `[f32;16]` frame's rotation block `frame[..12]`, never re-derives it); every `sqrt` = the geometric class over raw `[f32;3]`/matrix components (feat-map placement distance/scale/length, bevy_canvas vector len, apply column norm); the one `Mutex` = feat-map TEST-fixture `ResolveRecording(Arc<Mutex<Vec<u32>>>)` (fixture-handle class); zero new_v4 / fn-sigmoid / softmax / splitmix / OsRng / Belief-struct / blake3-truncation / coordinate-keyed-HashMap arrivals | katgpt-rs (this commit) |
| 2026-10-28 | Mode 2 mini-audit — ISSUE-102 DISCHARGE (amo idle lane, riir-refine cycle; the 970 lane filed reflex `robust_ab.rs:278` as the router.rs copy-gate class site 2, fix deferred to a quiet window; this lane claimed (`597dc0e`) and discharged it same cycle: delegation to `katgpt_core::exact_sigmoid_f64` + the `sigmoid_delegation_matches_frozen_legacy_body` pin (−200,200 sweep + [−26,14,0,40,200] anchors, bit-identical x≥0 / ≤3 ULP below, far tail < −710 scoped OUT in-test — the one-branch underflow-to-0 vs two-branch-subnormal gap no reachable arg approaches); the 10-27 wave audit's interim "admissible local form" verdict superseded by the filed issue + pin-enforced discharge; lib 487/0, clippy all-targets all-features −D clean, rustfmt clean; reflex HISTORY 102 row) | **SITE-2 COPY-GATE CLOSED — the class is now at ZERO known instances workspace-side** (site 1 router.rs:111 discharged 10-24b `sigmoid_delegation_matches_frozen_legacy_body`; site 2 robust_ab.rs:278 discharged here; engine.rs + nli_m1 pins predate); no other inline `fn sigmoid` f64 body outside the documented false-positive classes (test-gate names, delegation-pinning tests, the `exact_sigmoid_f64` wrapper itself) | katgpt-rs (this commit) |
| 2026-10-28 | Mode 2 mini-audit — the post-10-27 delta (amo idle-lane continuation [the katgpt-handoff session]; union re-grep over 307 fresh .rs across 19 repos since the 10-27 row's coverage [dapps 66, infer 65, train 28, refine 28, ai 21, remake 18, sge 16, ndb 13, reflex 9, chain 9, katgpt-rs 9, auth 7, kat 5, rethink 4, sdk 4, render 2, instinct 2, seal-std 1; vendor/block ×4 = vendor class]; NEW surface = the reflex harness-runner split [robust_ab/lanes_hosted/lanes_wire/report/run_phases/router_stats_probe/engine_tests], the seal-harness-evolution crate [family/lib/adapter+drive gates — zero signature arrivals], sge capture lanes + treasure-box trio + skill_effect_follow, dapps recovery/migrate day-2 [recovery rows, epoch_slice failsafe row, self_path.rs via git show], infer dist landed surface [stage/planner/handshake/multi/runner — the sibling 623 lane's unstaged WIP observed as-found, unaudited per detection-only]; signature greps + body-reads on every hit; zero-cargo under the league gate2 driver + bench_809, Plan-337 posture) | **1 FILED → DISCHARGED SAME-CYCLE**: reflex `robust_ab.rs:278` inline f64 sigmoid `1/(1+exp(−x))` = the router.rs:111 copy-gate class site 2, substrate REACHABLE — THIS lane filed it detection-only (`c2017a1`, the delegate menu, fix deferred for build contention; ⚠ the discharge row ABOVE reads the filing as "the 970 lane" — the filing commit is this lane's, the attribution likely rides the `190f244` toolchain-bump adjacency); the sibling lane then claimed `597dc0e` → fixed `b1fc3fe` (exact_sigmoid_f64 + frozen-body pin) → closed `5546cea` within the hour — class at ZERO, verdict = the row above's. REST CLEAN: train l4_margin_probe `fast_sigmoid` = thin delegation wrapper; train/dapps/ndb/infer RNG hits = the seeded test/probe fixture class (in-doc "no rand dep"/"pinned"); dapps mining_rewards reward_key = blake3(epoch‖account‖record)[..16] formula in-doc; epoch_slice `acct[..8]` = the credited-µKAT LE PAYLOAD into the 16-byte failsafe row (row encoding, never a truncation); recovery `[..32]` nonce / `[..16]` acct rows = the acct16 wire family; auth totp `k[..32]` = the HMAC pad (26g holding); sdk chain_seed + refine campaign_digest = the 26g families re-verified; infer msl_field `Mutex<HashMap>` output_cache = the device-scoped ZeroCopyState resource handle (papaya N/A); zero new_v4 / fn-softmax / belief-struct / coordinate-keyed-HashMap / unseeded-global-RNG arrivals in the union | reflex `c2017a1` + katgpt-rs (this commit) |
| 2026-10-27 | Mode 2 mini-audit — the post-26h delta (amo idle lane, fmt-landing-wait cycle; UTC anchor 26h 19:06Z — methodology note: a naive `git log --since` parses LOCAL +07 and double-counted the 26h wave's own eae0f05/2a680dc; anchors are UTC; TRUE fresh .rs surface = 3 files in 3 repos: dapps `915b39e` + infer `1f99c7e` both deny.toml + scripts/deny_gate.sh ONLY (zero .rs, zero dep-graph edges — the b294 T2.4 fan-out's own Research-332 record covers them) + infer `c60b34d` (i1124-t1.5, the ONE audited .rs, `prefill_cuda_full.rs` +64/−8) + train `caa334c7` margin-probe compile fix (READ-ONLY, routing-restricted); signature greps + body-reads; zero-cargo under the 645/469 arms + i1117 measurement + elephant window-3 capture, Plan-337 posture) | **CLEAN — ZERO FILINGS**: c60b34d = pure kernel-dispatch wiring behind `prefill_mmq_v2` cfg + env-gated `RIIR_PREFILL_FFN_GLU` — failure degrades bit-safely to the incumbent pair path (the 1124-T1 contract in-source), once-only AtomicBool REACH log (the standing reach-evidence pattern), zero state/map/truncation/RNG/sigmoid arrivals; train example = dead-GPU-context removal + doc-comment T0a verdict with re-open trigger (trainer-math sanctioned class per the 26f row) | katgpt-rs (this commit) |
| 2026-10-26h | Mode 2 mini-audit — the post-26g delta (amo idle lane; anchor = the 26g carrier `de2b006aa` 19:06Z; ONE fresh wave on the audited surface: dapps `eae0f05` issue-139 totp gate + its docs close `2a680dc` — 32 fresh .rs [src/kat/totp.rs 576-line STORE half, cloudflare totp_routes.rs ROUTE host + totp_contract.rs vectors, ~25 gate-wiring touches across the kat-service contract/route files, migrate_sweep class-Z row families]; ai's two origin commits predate the anchor — already inside 26g's HEAD-state read; sibling WIP observed as-found and untouched [dapps `M tests/kat_mining.rs`; this repo kv960 M-files + untracked deny.toml/deny_gate.sh = the b294-fanout lane] — the COMMITTED state audited via git show; signature greps + body-reads on every hit; zero-cargo under the 645 base arm + the 469/plan341 margin probes, 3 compute consumers ~600–850% CPU, Plan-337 posture held) | **CLEAN — ZERO FILINGS; the wave is the textbook consume-don't-copy form**: `riir_auth::{verify_totp, totp_sha256, TotpVerdict}` named in-source as the ONLY verify primitive ("consumed, never copied") — totp_contract's oracle calls `totp_sha256`, totp.rs carries ZERO math (its sole hmac mention names the primitive in a doc-comment), totp_routes zero crypto hits; the worker adds only what the primitive cannot (persistence: the accepted step IN the action's CommitBatch with early-persist ledger/cross-plane variants in-doc; the pending→confirm enrollment machine; the 5-fail/15-min cooldown where replays-never-count); disarm = zeroing-never-delete (class-Z law in-source); the store UNGATED with the burn/credit precedent cited; signature sweep zero new_v4 / fn-sigmoid / softmax / splitmix / RNG arrivals; `[..N]` hits all documented families (blake3(pk)[..16] pinned wire contract, [..12] shape-ids, migrate formulas in-doc, one test fixture slice); Mutex hits = the pre-existing worker resource-handle classes; `base32_of` = the 20-line RFC-4648 encoder with vectors (zero-new-deps posture, codec not substrate); SystemTime::now ×1 new = the worker clock seam. RUN-LOG MAINTENANCE: oldest row (10-19) dropped, tail held at 15; `git log -p` recovers it | katgpt-rs (this commit) |
| 2026-10-26g | Mode 2 mini-audit — the post-26f delta (amo idle lane; anchor = the 26f carrier `71ae535a7` 16:45Z — NOT the 26e time-anchor, which double-counts the 26f wave; 15 fresh .rs across 6 repos [ai 2 {plan-642 harness_evolution runtime consumer + overlay gates — audited at HEAD state via git show; the worktree carries UNSTAGED deletions of both + dense quest_combat M-files = the in-flight plan-642 lane rework, as-found, unaudited per detection-only; origin/develop ahead of local HEAD}, sge 1 {editor-core run_browser}, refine 3 {blueprint lane}, ndb 3 {moedern falsifier test lanes}, sdk 3 {scenelab arena_generations + goat bench}, auth 6 {totp/rebind/provider_identity/provider_login/signing_session + warm bench}]; signature greps + readability witnesses (`grep -c 'fn '`) on every zero-hit file — the silent-zero trap; body-reads on every hit; zero-cargo under the 645 softmax arms — 3 compute consumers at ~900–1000% CPU, Plan-337 posture held) | **CLEAN — ZERO FILINGS**: refine `campaign_digest` = the documented derivation family (formula + truncation in-doc: BLAKE3 over mode‖posture‖sorted relative paths, 16-byte hex, BTreeMap determinism); sdk `chain_seed` = the documented seed-derivation family (BLAKE3 over fixed-width LE, first 8 bytes as u64, T1.2 generation-chaining law in-doc) + `corpus_hash` full digest; ai runtime = full-digest AuthoringExport integrity (no truncation), the overlay lane carrying native-only/wasm32 posture + apply-path law + freeze-custody in-source; sge run_browser `Arc<Mutex<…>>` pair = the app-thread BundleState/opened-URL cell class (papaya N/A, not maps); auth totp SHA-256 = RFC-6238 interop-mandated (blake3-impossible by protocol, rationale in-doc; `k[..32]` = the HMAC block pad, never a truncation); gates fixture goldens = full-digest pins; zero new_v4 / fn-sigmoid / softmax-impl / belief-struct / coordinate-keyed-HashMap / unseeded-RNG arrivals anywhere in the wave. RUN-LOG MAINTENANCE: tail pruned to the newest 15 (dropped 10-18d + the misordered 10-20a; `git log -p` recovers both) | katgpt-rs (this commit) |
| 2026-10-26f | Mode 2 mini-audit — the post-26e delta (amo idle lane, lease-reshape session; fresh .rs committed after the 26e anchor commit `d6bf72238` 11:53Z: ~302 files across 16 repos [ai 29 {quest_combat ambient/attack splits + bench_1113/1129/1130 lanes + constants/static_data}, infer 30 {cudarc/GLU-MMA kernels + bench903 arms + plan623 G1 + rollout_history + flashmemory gate}, remake 57 {the 008-closure wave + duel/mint/presence + vessel_zstd bench + authored_life}, dapps 26 {kat-service contract/mining splits + kat_migrate sweep}, refine 40 {B288-292 corpus + sec proposers + CLI test lanes}, sge 70 {the 433-435 editor-data/run-lane waves, locally synced post-26e}, train 9 {cooloff_mixture + pool_reband + elbo — READ-ONLY routing restriction}, chain 15, reflex 5, rethink 4 {hosted_metering}, kat 5, auth 3, sdk 3 {riir-vessel codec + G4 alloc test}, katgpt-rs 3 {harness_evolution extraction}, seal-std 4, render 11, ndb 3]; signature greps + body-reads; zero-cargo under elephant window-3 + i1117 measurement) | **CLEAN — ZERO FILINGS**: katgpt-core `harness_evolution.rs` = the VERBATIM-EXTRACTION form (spec-match discipline in the header: the scenelab arena rule is the SPEC, byte-identical adoption pinned by the spec-match grid, first-index champion law cited, the opt-in third tier excluded from the arena consumer); ai `hero_progress_drive` splitmix = the luck designated home holding (Issue-1044, delegation + doc provenance); infer `PromptKey[..16]` + rethink `arsenal_digest[..16]`/`commitment16` = the documented derivation/display-seal families (formulas in-doc; commitment_hex full form rides VesselFacts beside it); dapps `[..12]` shape-id + `[..16]` acct-width families = the pinned wire contract prose (routes.rs documents blake3(pubkey)[..16] verbatim); train elbo softmax ×2 = model-forward trainer math (riir-train routing-restricted, read-only); ssh-keygen OsRng ×7 (auth key GENERATION — correct-by-nature; sge test throwaway pairs) + refine corpus OsRng = fixture-string data; Mutexes = the quest_combat handle class + seal-view app-thread handles (mint KIT static, duel mirror, meter projection — papaya N/A, not maps); sqrt = geometric distance over raw [f32;2] + closed-form area→radius + bm25 fixture-string builder; zero new_v4 / fn-sigmoid / softmax-impl / belief-struct / coordinate-keyed-HashMap / unseeded-RNG arrivals | katgpt-rs (this commit) |
| 2026-10-26e | Mode 2 mini-audit — the post-26d delta (amo idle lane, riir-refine cycle; fresh .rs since the 26d anchor `662e90516` 07:08 UTC: 115 across 7 repos [ai 25 {loot tables 323950da6 + attack_tick/respawn_tick consumer seams + swarm-emotion split + progress_reheat bench fix}, remake 43 {the 433/434/435 slices + the P3 vessel-zstd checkpoint landing 97ef2af + pointer cues + balloons/fps-cap 60}, sge 21 {433/435 + feat-map tint/parity lanes}, refine 13 {issues 173/174 test-infra closures}, infer 8 {plan623 T0/T1 multi-host wire + issue-1122 probe + issue028 decode gate}, seal-std 4 {seal-anim attack_speed + seal-fx pointer cues}, render 1; + ndb 2 late-catch {12b1af4 dense-ordinal BM25 + 748b10f debug_assertions cfg}]; chain/reflex/dapps/train/kat/auth commits all pre-anchor (26b/26c/26d coverage); signature greps + body-reads; zero-cargo under sibling lanes) | **CLEAN — ZERO FILINGS; the wave's centerpieces verified**: the loot/hit rolls consume `riir_games::swarm::luck::seeded_unit_roll` (the designated home; quest_combat's three new salts = distinct streams, "none of these reuse SALT_CRIT" in-source, the authority-replay law in-doc); `magic_speed`/`skill_rate` = the seal-std `seal-anim/attack_speed.rs` SINGLE HOME with 5 seal-view consumers (combat_fx/skill_fx/character_sheet/quick_slots/inventory) — single-source-by-leaf holding through the 434 wave; combat_fx `unit(seq, salt)` delegates `crate::fx_rng::{GAMMA, finalize, unit01}` (the Issue-057 home holding through its newest consumer); splitmix = 1 note-level inline seed-mix (dev_spawn `monster_offset`: GAMMA-mul + rotate_left(31) partial mix feeding a seeded Rng, measured angle-collision rationale in-doc — "seq 6 and 7 drew bit-identical angles" — single site, below the 4th-arrival consolidation gate); infer `PromptKey` = documented blake3[..16] derivation (issue-1122 instrument, rationale in-doc); softmax ×4 = doc-comments only (online-softmax tile prose in the deltanet forward); feat-map `(u32,u32)` HashMap = model-keyed render-mesh cache; Mutexes = the established quest_combat `Arc<Mutex<Roster>>` handle class; sqrt/exp = geometric + model-forward (gemma2 embed_scale) + doc-comment config classes; sge OsRng ×2 = test-only throwaway ssh keygen (same-file twin pair, test-only inline-helper class); zero new_v4 / fn-sigmoid / belief-struct / coordinate-keyed-perception arrivals | katgpt-rs (this commit) |
| 2026-10-26d | Mode 2 mini-audit — the session-delta wave (M3 idle lane, overflow-continuation; 74 fresh .rs across 3 repos in the merge ranges not covered by the 26c row: seal-remake 35 [the settings_window/attach-retirement/waypoint-overlay/viewer-id wave + plan-052 quest lanes], ndb 38 [bm25 foldhash ef8599a + scope/item_instance/CLI row + bench 334/335], infer 1 [stability.rs]; katgpt-rs/refine/rethink/sge ranges = 0 fresh .rs (docs/issues/glue); signature greps + body-reads; zero-cargo) | **CLEAN — ZERO FILINGS**: viewer `view_id64` splitmix64 = delegates `crate::fx_rng::finalize` — the landed Issue-057 single-home consolidation VERIFIED HOLDING through the wave (lib.rs:117 header intact); waypoint_overlay `BeliefSource`/`BeliefMarker` = FALSE-POSITIVE view-glue (an `Arc<Mutex<QuestCombatState>>` handle + frame snapshot feeding a debug overlay; the frustration/stuck_drive/breadcrumb values are CONSUMED read-only from `riir_games_mmorpg::quest_combat` — the FSM substrate computes, the view draws); the cfg-twin resource structs = the standard feature-key pattern; vessel_scene `by_key` HashMap = id-keyed facing join (source+id, not coordinate-keyed); ndb `lane_cache` = papaya with rationale in-doc; blake3 `[..16]` sites = the documented ndb derivation families (item_instance InstanceId + CLI row integrity tag/nonce, formulas in-doc — the 10-24 row's verdict holding); bench `[..8]` = nonce fixture stuffing; Mutexes = the established seal-view app-thread handle class; zero new_v4 / fn-sigmoid / fn-softmax / softmax-impl / belief-STRUCT-parallel arrivals | katgpt-rs (this commit) |
| 2026-10-26c | Mode 2 mini-audit — the post-26b delta (amo idle lane, riir-refine cycle 3; fresh .rs since 22:55Z 10-25 across 4 repos not covered by the 26/26b rows: ai 2 [settlement_queue + warm_tier — the `770dff81c` perf pair, ndb Issue 636 T4], ndb 10 [hull.rs key column — Issue 633 T5, Bench 509 — + index/local_kv test lanes], sge 9 [run_vessel/run_items/static_export/quest g3 tests/feat-map], dapps 76 [the oversized-file-split lane MID-FLIGHT — detection-only signature pass, measured as-found]; signature greps + body-reads; zero-cargo under the elephant capture) | **CLEAN — ZERO FILINGS**: ai perf pair = borrowing-scan refactor over EXISTING substrate, zero signature arrivals (no new_v4/sigmoid/softmax/splitmix/Belief/blake3/Mutex); ndb hull = substrate-side index perf (sorted support points + first-dim key column, Deref-read-only invariant in-doc), Mutex hits = `Mutex<WalWriter>` + test `Vec<u8>` buffers (resource handles, papaya N/A), tests `[..8]` = fixture vector stuffing; sge 9 = glue/test/export lanes, zero signature hits; dapps split = pure code moves — zero new-math arrivals, explorer `[..8]` = the display-seal family (query-prefix/URL strings) in moved files carrying their doc-comment families intact; zero coordinate-keyed HashMap / emotion-struct / RNG arrivals anywhere. **CORROBORATED + EXTENDED 10-26 ~10:3x (zed-amo-idle, independent re-pass):** the dapps split lane COMPLETED (the "verbatim move" commits landed 09:18–09:31) — duplicate-fn-name census across the whole split surface returns ZERO collisions (no helper copied by any move), and the auth `38a803c` github_pow split (landed 09:05, post-dating this row's pass) is CLEAN by body-read: verify surface = pure compute under `github_pow`, transport = `github_pow_fetch` (implies `github_pow` + `dep:ureq`) keeping ureq/rustls/ring out of the kat-service worker's wasm32 graph, rationale in-source at the cfg + Cargo.toml, the `github_pow_measure` example carrying its `required-features` row same-commit, one blake3 digest with the formula in-doc; the mining.rs `[..12]` shape_ids = fixture-class test-payload builders (no derivation decision) | katgpt-rs (this commit) |
| 2026-10-26b | Mode 2 mini-audit — the post-422db349e delta (the amo lane's 01:10+07 anchor → ~05:5x; 56 fresh .rs across 7 repos: katgpt-rs 13 [the katgpt-core substrate wave: anchored_reach/metabolic_gate/stale_residual/template_decode/analytic_lattice/functional_substitution/lattice_memory/leakage_probe/hint_regret/spectral_pencil + bench_951 + kv-903], ai 16 [the 1127/1128 arc: anchored_spawns + hero_threat_lane + cognition reorg + karma_engine/p2p_gossip + bench_1125 trio + walk_grid], chain 5 [commitment_cache i166 + dispatch/server + 2 test lanes], infer 6 [BPE tokenizer + G1/G2/G4 gates + kv_evict + spike_census bin], remake 8 [the plan-043 P10 native p2p lanes: seeder/p2p_client/p2p_session + sawtooth/goodput/vessel probes], refine 7 [kernel_opt corpus — fixture class], sdk 1 [scenelab trace_index]; signature greps + body-reads; zero-cargo) | **CLEAN — ZERO FILINGS**: TEXTBOOK consumes ×3 — 1128 `anchored_spawns` seats `patrol_home` through the EXISTING home_patrol/leash_frustration/chasing_hero substrate ("no new system, no thresholds" in-source; the unanchored genesis + quest-ring fallback named as the LIVE 1127 producers), hero_threat_lane delegates `game_traits::{sigmoid, approach::*}` (the shout falloff rides the route-stamp sigmoid family), walk_grid delegates `game_traits::sigmoid` + geometric grid sqrts (documented class); SUBSTRATE-SIDE births with lineage headers (the ladder, not duplication): stale_residual `kl_logits` = stable softmax-KL with the ln(1e-30) floor rationale in-doc, input domain distinct from ugc_schedule's probability-space `kl_discrete` (not a dup), metabolic_gate delegates `crate::sigmoid`, lattice_memory SHIPS ITS OWN sigmoid-only grep-gate test (comment-stripped needle over every module file — the law as an in-crate gate), leakage_probe/bench LCGs = seeded fixtures per the gate convention, spike_census = `SplitMix64(args.seed)` seeded; chain `node_hash_from_player` = the injective zero-pad EXPANSION consuming player_id_hash (chain→guard identity bridge, never serialized, boundary posture in-doc), dispatch/delegation blake3(pubkey)[..16] sites = #[cfg(test)] fixtures carrying the byte-for-byte mirror comment (the documented copy-gate family); remake p2p lanes = a DISTINCT chunk-distribution concern (hunks/slips/verify over str0m datachannels — not game-state transport; `str0m`/`bao` already BOUNDARY-declared seal-view-only behind `p2p_vessel`, rows 166/167), karma_engine pair-key HashMap = the 1064 ordered-pair O(1) law (not coordinate-keyed), commitment_cache = id-keyed papaya over the existing MatrixAccount map, authority_sim's new std-Mutex handles = copies of the established boot-rig pattern (no papaya in-crate), tokenizer merge_ranks = BPE-internal; zero new_v4 / softmax-impl / belief-struct / coordinate-keyed / Mutex-policy arrivals | katgpt-rs (this commit)
| 2026-10-26 | Mode 2 mini-audit — the post-25b delta (amo idle lane; fresh .rs across 5 repos: dapps 6 [the kat_migrate lane: migrate.rs + migrate_sweep.rs + tests + the worker's migrate_contract/migrate_routes], auth 2 [github_pow.rs + measure example], kat 2 [kat_protocol_migrate + crosswalk test], sdk 1 [aoi_fanout_roster_goat bench]; modified: kat-service worker files, auth account_key/bridge, kat_protocol_freeze, sdk aoi_fanout_curator, reflexer vessel env/lib, katgpt-assign trio; signature greps + body-reads; zero-cargo) | **CLEAN — ZERO FILINGS**: dapps `[..16]` ×3 = the documented derivation families with the formula in each doc-comment (acct16 "one derivation, two widths"; reward_key row-key; mint_receipt_id "the same derivation the mint plane pins" — cross-plane single-source by re-derivation, pin named in-source); github_pow = pure fetch/verify module, one `blake3::hash` over canonical serialization (the digest family, formula in-doc), threshold consts not math; aoi curator/bench + katgpt-assign + reflexer vessel = zero signature hits; zero new_v4/splitmix/softmax/Belief-struct/spatial-HashMap/RNG arrivals; tests+corpus = fixture class | katgpt-rs (this commit) |
| 2026-10-25b | Mode 2 mini-audit — the post-25a delta (the 4090 idle lane's own sync window, 3 fresh .rs + 1 py across 1 repo: reflex `src/lanes/elephant.rs` NEW [412 lines], `src/harness/runner.rs` + `src/bin/harness.rs` mod, `scripts/elephant_server.py` NEW; the rethink-032-T10.2 mammoth/elephant comparison-lane wave `1e6d4db..f60923a`; signature greps + body-reads; zero-cargo, sibling measurement active) | **CLEAN — ZERO FILINGS**: elephant.rs = pure HTTP wire lane (env consts + post/get + serde parse + `determinism_view` clock-strip) with ZERO numeric-decision surface — no sigmoid/softmax/exp/hash/seed/distance; the `TeacherWireLane` impl = textbook glue over the EXISTING lane trait (the mammoth-family shape, `spend_env`/`MAX_CASES_ENV` spend-ceiling contract carried); runner.rs softmax ×3 = doc-comments DESCRIBING the elephant/bekko lanes' own readouts (never implemented here), `distance_only_*` = pre-existing gate-floor test fn names, blake3 mentions = gating doc-comments; harness.rs `&digest[..16]` = the corpus-row blake3 DISPLAY seal in a log line ("blake3 {}…" ellipsis literal — the display-seal family, not a truncation decision); elephant_server.py = zero signature hits (reference qev serve plumbing); zero new_v4/splitmix/Belief-struct/spatial-HashMap/Mutex/emotion arrivals | katgpt-rs (this commit) |
| 2026-10-25 | Mode 2 mini-audit — the post-24c delta (anchor `9d27c7c56` end, 10-24 ~23:1x +07 → 10-25; 2 fresh .rs across 2 repos: ai 1 [riir-gpu `bench_1122_site_tax_and_ktile64` — the Issue-1122 bench vehicle, sibling lane, READ-ONLY as-found], remake 1 [`canvas_draw_placed` — the plan-412 3d-iii placed-rows lane]; mod-set 17 across the same 3 repos: render 4 [scene_set/skinned/zone_scene/assemble], remake 12 [the canvas_draw family + authored_life_view/dressing_rig/head_look/meadow_ground/quest_talk_view/vessel_boot/vessel_scene], ai 1 [riir-gpu lib]; signature greps + body-reads; zero-cargo) | **CLEAN — ZERO FILINGS; the wave is a CONSOLIDATION, not duplication**: seal-scene `scene_set.rs` carries the single-home law in its header ("the editor's model viewport lays its scenes out through it today; the RUN viewer will, so both lay the same parts out alike") and is consumed by seal-render lib/assemble/zone_scene AND sge's `model_3d_view` — the plan-412 single-renderer law as shared-home birth; `skinned.rs` likewise ("one implementation of skinning… fixed operation order, so posed output cannot shift" — Issue-940-style determinism in-source); remake `canvas_draw_placed.rs` = TEXTBOOK consume (`scene_set::{step_animated, resources_worth_posing, PoseSink}` + `assemble::scene_skin` + `LandOn` wholesale; the idle "by the map page's own sampler"; the giver's gaze arrives as `PlacedGaze` from head_look's verdict, never recomputed). REST = documented classes: head_look `1−exp(−k·dt)` = view-layer ease (seal-view no-dep class) with the in-source router comment sending the placed turn to the canvas; meadow_ground `distance3` [f32;3] + skinned quaternion-normalize/segment sqrts = the geometric class; HashMaps = id/class-keyed fade/joint/land state (zero coordinate-keyed); Mutexes = the app-thread sim-state handles (papaya N/A); skinned `[..8]` = a test-array slice assert (not a hash truncation); bench_1122 = pure harness, zero RNG/seed/sigmoid surface; zero new_v4/splitmix/softmax/belief-struct arrivals. ALSO REPAIRED (this commit): the run-log table itself — the `|---|` separator had DRIFTED into the middle of the table (13 newest rows above it rendering as plain text — the append-above-the-separator drift class, 16 rows accreted since the 10-12 prune) — separator restored under the header, tail pruned to the newest 15 rows per the documented maintenance rule; pruned rows recoverable via `git log -p` | katgpt-rs (this commit) |
| 2026-10-24c | Mode 2 — the post-24b delta wave (anchor `9d27c7c56`, 10-24 16:00+07 → ~23:1x; 147 fresh .rs across 12 repos: sge 41 [the model_facing arc: editor-core probe + gizmo + seeds + store tables + run_reach/run_spawn_models + feat-map fire/lava/waterfall/relief], remake 32 [canvas_draw host/effects/water + vessel_scene/map_stream/chunk_stream + eye_order/load_lint/lod_payload/scheduler gate tests], train 14 [bench-algo reasoner + vq_codebook backprop arms — READ-ONLY, routing-restricted], infer 11 [prefill_cuda_full + gemm mma + laya agent/stability + plan622 g1], refine 11 [kernel_opt corpus entries_gpu_101-119 — fixture class], sdk 8 [riir-vessel cas/chunk_manifest/pack/codec — the ndb-632 wire lane's landed state], render 8 [skill_fx canvas + waterfall/ice + scene zone vocab], katgpt-rs 5 [kv960 T2 alloc landing + its gate tests], ai 4 [bench_1113/656/809 test lanes + gpu lib], ndb 4 [artb + vessel query_hash], reflex 5 [elephant lane + distill + harness], seal-std 4 [seal-fx effect sim/billboard/classify]; signature greps + body-reads; zero-cargo) | **CLEAN — ZERO FILINGS; the 24b row's WIP flag RESOLVED**: katgpt-rs `kv_sensitivity_alloc.rs` LANDED as substrate-side COMPOSE — a behavioral-damage weight into the SHIPPED `waterfill_bits`/`BitAllocator` allocator ("no new search, no new quantizer: one weight vector"; T1's negative-control caveat in-doc) — the extend-via-input form, not a parallel allocator. TEXTBOOK consumes: sge `run_reach.rs` delegates the reach verdict to `riir_hunt_reach::AimReach` wholesale ("the answer is not computed here", :331 sqrt = display distance); the seal-fx single-source-by-leaf law HOLDS through the wave — seal-render `bevy_canvas_skill_fx` + seal-view `{skill_fx,fx_player,buff_fx}` all consume `seal_fx::{effect,skill_fx,buff_fx,goldens}`, zero copy arrivals; sdk riir-vessel cas/chunk_manifest = blake3 content-addressing over the artb wire (the 632 wire lane posture re-verified). REST = documented classes: infer gemm_mma golden-ratio LCGs (8 sites, all inside `#[cfg(test)] mod tests` @5197) + train train.rs:1185 xorshift seed (also `#[cfg(test)]`) = the fixture-RNG class; train carry.rs:193 `momentum_beta` σ(b) = the bench reasoner's model-forward parameterization (riir-bench-algo carries NO katgpt-core dep — only optional katgpt-pruners behind carry_parity); train model.rs `softmax_inplace` = model-forward; ndb vessel `query_hash` [..8] = cache-key derivation with perf rationale in-source (~80ns vs ~1100ns hit); reflex distill/harness [..16] = display seals; sge build_lane [..8] = warning-text display; remake gate tests hash[..8] = fixture stuffing; HashMaps = id/texture/DB-row/tile-cache keys (zero coordinate-keyed perception); Mutexes = infer prefill GPU buffer/state mirrors (resource handles, papaya N/A) + laya `PACKED_ACT_BITS` = the documented parity-seam diagnostic ring (cap 4096, aliasing-hazard story in-doc); zero new_v4 / belief-struct / emotion-struct / non-model softmax arrivals | katgpt-rs (this commit) |
| 2026-10-24b | Mode 2 mini-audit — the post-anchor delta only (10-24 ~09:00 → ~16:10; 75 touched .rs across 9 repos, 12 added: katgpt-rs 2 [kv960 T0a/T0b kv_sensitivity + capture], ai 2 [hero_cast + tests], refine 3 [multivector kill-test + router_per_span_dump], train 1 [the 648 pin test — the morning filing's fix half], remake 4 [eye_order + load_lint + g1 tests]; mod-set: sge 25, remake 8, render 8, ai 8, refine 4, katgpt-rs 4, train 2, seal-std 3, reflex 1; signature greps + body-reads; zero-cargo under 5 live sibling lanes) | **CLEAN — ZERO FILINGS; 1 PRIOR MENU DISCHARGED**: reflex `harness/runner/router.rs:111` — the 10-15 NOTE-LEVEL inline f64 sigmoid now DELEGATES `exact_sigmoid_f64` with full provenance in-source (the 014 engine.rs precedent + this audit's 10-15 row + katgpt-rs `bbc28a091` cited) + the frozen-legacy-body pin test (`sigmoid_delegation_matches_frozen_legacy_body`: bit-identical x≥0, ULP-bounded below, [−120,120] sweep, the legacy body frozen in-test) — the copy-gate done end-to-end from note to fix. TEXTBOOK: kv960 `kv_sensitivity.rs:738` splitmix = the `#[cfg(test)]` FIXTURE class (in-source "No external RNG dep; same seed → same stream" + the Sterbenz-exact bimodal fixture); refine `multivector.rs` = modelless kill-test consuming `katgpt_core::simd::maxsim_score` (substrate wholesale), its `[..16]` corpus_digest = the frontier_miner short-digest family; train 648 pin test = the expected delegation-pin landing (same-day filing → same-day fix). REST = documented classes: ai constants.rs configured-point sqrt + knockback falloff (the plan-032 lane), sge feat-map/skinned_model/walk_dots geometric [f32;3]/xz math, remake viewer/lib.rs splitmix doc-comments = the landed Issue-057 consolidation verified holding ("the view layer's ONE splitmix64 draw"), render bevy_canvas/grass view-layer geometry, forward 1/√d + rms = model-forward, reflex router `[..16]` = the routed-artifact seal display id; remake eye_order/load_lint + ai hero_cast = zero signature arrivals (glue lanes over existing vocabulary); kv960 sibling T2 WIP (kv_sensitivity_alloc.rs untracked) measured as-found, unaudited per detection-only | katgpt-rs (this commit) |

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

