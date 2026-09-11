---
name: feature-gate-audit
description: Audit feature-gate status claims across the multi-repo stack. Use when a doc/plan/issue/commit-message claims a "promotion discrepancy", "not yet wired", or "Default-off until..." state, when fixing stale feature-gate comments, before promoting or demoting any feature flag, or quarterly as a feature-gate-hygiene gate. Enforces four defenses (1) source-code verification of every wiring claim — grep the production tick path, don't trust the doc; (2) multi-surface grep for stale comments across 5 documentation surfaces (source .rs, lib.rs module doc, Cargo.toml default block, downstream Cargo.toml, .benchmarks/*_promotion_review.md); (3) layer-split awareness — engine-layer DEFAULT-ON + game-layer OPT-IN is a deliberate pattern when each layer gates a different concern (generic runtime vs IP-bearing content), NOT a missed propagation; (4) gate-chain resolution — own cfg plus every ancestor mod plus the crate default list, settled by a build not a read. Sibling to goat-audit + doc-sync.
---

# feature-gate-audit — Verify feature-gate claims against production code

This skill exists because of a caught error chain: session `f92a7ffe` in
`riir-ai` (2026-07-18) framed `npc_sleep_time`'s engine-DEFAULT-ON /
civ-OPT-IN state as a "promotion discrepancy" and claimed the catalog
was "wired but not yet exercised in production civ tick". Session
`f865913e` (same day) verified both claims against production code and
found them **factually wrong**:

- The split is a **deliberate layer split** (engine = generic runtime;
  civ = IP-bearing catalog + graceful-degradation hook into
  `SleepConsolidateNode::tick` Gate 3).
- The catalog IS exercised in production via
  `crates/riir-games-civ/src/civ/daily_loop/mod.rs:1071-1078`, which
  calls `ctx.sleep_runtime.sleep_cycle_tick(&hla, &self.dirs)`.

The three defenses below would have caught both errors before they
landed. Apply them to every feature-gate claim you encounter.

## When to use

- A doc, plan, issue, commit message, or session summary claims a
  "promotion discrepancy" between two crates' feature-gate status
- A doc claims a feature is "wired but not yet exercised in production"
- You encounter a `// Default-off until G1–GN GOAT gate passes` comment
  in source code
- Before promoting a feature flag to default-on (or demoting to opt-in)
  in any product-set repo (**7**; canonical list in `katgpt-rs/AGENTS.md`
  §"Repo count" — read the LIST, not the count: this cell said 7-omitting
  `riir-dapps`, then 8-counting-retired-`riir-armageddon`. A count that drifts
  from the canonical list is the exact defect this skill audits in others)
- Quarterly as a feature-gate-hygiene gate (alongside `doc-sync` and
  `goat-audit`)

## The four defenses

### Defense 1 — Verify every "discrepancy" / "not yet wired" claim against production code

**NEVER** propagate a doc/plan/session-summary's framing of feature-gate
state without first grepping the actual production wiring. Plausible-but-
wrong framings are the canonical failure mode — they read as reasoned
verdicts, get propagated through summaries, and solidify into "facts"
that downstream sessions trust.

**Protocol:**

1. Identify the production host(s) — the function that would invoke the
   gated code path on the hot tick. Common hosts in this stack:
   - `tick_proactive_salience` (salience gate)
   - `tick_cgsp_curiosity` (CGSP runtime)
   - `tick_motivation` (motivation system)
   - `SleepConsolidateNode::tick` (sleep-time, daily loop)
   - `evolve_feeling_brain` (emotion systems)
2. `grep` for the feature name across **all** `.rs` files in the crate
   tree, not just the file the doc cited:
   ```sh
   grep -rn "FEATURE_NAME" crates/ --include="*.rs"
   ```
3. Read the body of every host function whose signature imports a type
   from the gated module.
4. If a `#[cfg(feature = "FEATURE_NAME")]` block is reached along the
   tick path, the feature IS exercised in production — regardless of
   what the doc says.
5. If no host reaches the gated block, **then** the "not yet wired"
   claim may be accurate — but verify with a call-graph grep, not just
   a single-file read.

**Anti-pattern — trust the summary:** "the prior session said X, and
the prior session is in git history, so X must be true." Prior sessions
make mistakes. Git commits are not authority — production code is.

### Defense 2 — Stale gate-status comments live in MULTIPLE locations; fix all of them

When a feature is promoted from opt-in to DEFAULT-ON, the promotion
must propagate through **five documentation surfaces**. Missing any one
leaves a stale comment that misleads future readers.

| # | Surface | What to check |
|---|---|---|
| 1 | The feature's source `.rs` file (top-of-module `//!` doc) | "Default-off until..." → "DEFAULT-ON since..." |
| 2 | The crate's `lib.rs` `pub mod` declaration comment | The `// Default-OFF until...` line above `#[cfg(feature = "...")]` |
| 3 | The crate's `Cargo.toml` `[features] default = [...]` block | Feature name should appear, OR have an explicit sibling comment explaining intentional opt-in |
| 4 | Every downstream consumer's `Cargo.toml` `[features]` block | Forwarding features don't need to be default-on, but their comments must not cite a pending gate that has already passed |
| 5 | `.benchmarks/NNN_<feature>_promotion_review.md` | Must exist + reference the GOAT gate evidence (G1–GN) |

**Canonical failure (riir-ai, 2026-07-18):** the `npc_sleep_time` comment
`// Default-off until G1–G5 GOAT gate passes (Plan 341 Phase 7)` existed
in **both** `crates/riir-engine/src/lib.rs:629` **and**
`crates/riir-games-civ/src/npc/sleep_time_catalog.rs:48`. A prior session
only caught the latter; the engine `lib.rs` surface went unfixed.

**Protocol — the 5-surface grep (run for every feature under audit):**

```sh
# In each repo that touches the feature:
grep -rn "Default-off until\|Default-OFF until" crates/ --include="*.rs"
grep -n "FEATURE_NAME" Cargo.toml crates/*/Cargo.toml
grep -n "FEATURE_NAME" crates/*/src/lib.rs
ls .benchmarks/*FEATURE_NAME* 2>/dev/null
```

**Fix discipline:** all stale surfaces go in **one commit**, not
piecemeal. The commit message should enumerate which surfaces were
fixed and cite the canonical benchmark (`Promotion Review` NNN) as
authority.

### Defense 3 — Layer splits are deliberate, not bugs

**Asymmetric feature-gate status across crates is OFTEN a deliberate
design, not a missed propagation.** Before claiming a "discrepancy",
verify that each layer is gating the SAME concern. If they're gating
**different** concerns, the asymmetry is a layer split — update the
comments to explain the split; do NOT propagate the promotion.

**Established layer-split patterns in this codebase:**

| Feature | Engine layer | Game layer | Why split |
|---|---|---|---|
| `npc_sleep_time` | DEFAULT-ON (generic runtime: `HlaSleepTimeOp`, codec, BLAKE3 commitment) | OPT-IN (`riir-games-civ`, `riir-games-shared`, `riir-games`) | Civ layer gates the IP-bearing catalog (per-NPC-type direction vectors) + graceful-degradation hook into `SleepConsolidateNode` |
| `cognitive_branches_runtime` | DEFAULT-ON (plasma-tier core) | OPT-IN sub-features (`_engram`, `_closure`, `_replay`, `_entity_cognition`) | Engine ships the core; heterogeneous subsystems stay opt-in |
| `committed_personality_runtime` | DEFAULT-ON (plasma-tier core) | OPT-IN `committed_blend_freeze` (ArchetypeBlendShard persistence layer) | Engine ships the core; persistence layer stays opt-in |
| `arg_runtime` | DEFAULT-ON (plasma-tier core) | OPT-IN `_manifold`, `_action`, `_offline`, `_replay` | Engine ships the core; heterogeneous Steps 4/5/7/8 stay opt-in |
| `cwm_runtime` | DEFAULT-ON (IP-free core with mock LLM) | OPT-IN `cwm_gemini` / `cwm_local_llm` / `cwm_wasm_loader` | Engine ships the modelless core; concrete LLM backends stay opt-in |
| `conformal_predictive_intervals` | DEFAULT-ON (Plan 468, 2026-07-20): generic `ConformalIntervalCalibrator` substrate — pure empirical-quantile math, zero-cost-when-uninvoked | OPT-IN — **7 consumer surfaces** across 2 crates: `karc_conformal_width` (riir-engine, per-NPC state field on `NpcKarcState`), `salience_conformal_width` (riir-games-civ, Delegate-nudge routing — IP-bearing), 5 probe features (`conformal_{curiosity,sleep_time,mcts_collapse,salience_tri_gate,per_channel}_probe`, riir-engine, standalone measurement) | Engine ships the math; consumers gate **three distinct concerns**: (a) per-NPC residual-pool overhead (Bench 567 measured +113.9% G2 — sidecar attachment is consumer choice); (b) IP-bearing nudge policy (Delegate vs Speak — `DelegateNudgeSource::ConformalWidth`); (c) standalone measurement corpora (`#[ignore]`-d probes sweep 11×100×100+ configurations). **The most granular layer split in the codebase** — three opt-in concerns at two consumer crates. |

**Three-layer split pattern (the `conformal_predictive_intervals` extension):**
The existing rows above are all two-layer splits (engine + game). The conformal
primitive introduced a three-layer split — engine substrate (katgpt-rs) →
engine consumer state (riir-engine `karc_conformal_width`) → civ routing
(riir-games-civ `salience_conformal_width`). The middle layer exists because
the per-NPC `Option<KarcConformalSidecar>` field has measurable overhead
(Bench 567) that the consumer should opt into explicitly, distinct from the
IP-bearing routing policy at the civ layer. When auditing a UQ primitive
with a similar shape, look for this three-layer pattern: the engine ships
the math, the engine consumer gates the per-NPC state cost, the game layer
gates the IP-bearing policy. Each layer's opt-in is a separate concern and
should NOT be propagated.

**Established propagate-the-promotion patterns (NOT splits):**

| Feature | Engine | Game | Why propagated |
|---|---|---|---|
| `karc_runtime` | DEFAULT-ON | DEFAULT-ON (`riir-games-civ`) | Both layers gate generic machinery; no IP split |
| `cgsp_runtime` | DEFAULT-ON | DEFAULT-ON | Generic machinery; no IP split |
| `proactive_salience` | n/a (no engine dep) | DEFAULT-ON | Self-contained at the civ layer |

**Decision rule:**

- **Both layers gate generic, IP-free machinery** → propagate the
  promotion to all 5 surfaces per Defense 2.
- **One layer gates IP-bearing content** (catalogs, direction vectors,
  game-design data, trained weights) → keep the IP layer opt-in,
  document the split with a comment explaining the rationale.
- **One layer gates heavy optional deps** (`riir-neuron-db`,
  `riir-chain`, GPU crates) → keep that layer opt-in, document the
  dep-cost rationale.

**Protocol — before claiming a discrepancy:**

1. Read both crates' `Cargo.toml` comments. Do they describe the SAME
   concern or DIFFERENT concerns?
2. If different concerns → it's a layer split. Update the comments on
   both sides to explain the split; do NOT propagate the promotion.
3. If same concern → it's a real discrepancy. Propagate the promotion
   and update all 5 surfaces per Defense 2.

### Defense 4 — "Is it gated?" is a question about a CHAIN, not a line

A gating claim sourced from a bare `grep 'pub mod X'` is **inadmissible**. The
matched line almost never carries the whole answer, and each missing link has
already produced a wrong verdict in this workspace:

| link | what it looks like when missed | real case |
|---|---|---|
| the item's own `#[cfg]` | you read line N (`pub mod X;`) and miss line N-1 (`#[cfg(...)]`) | riir-ai Issue 741 / Proposal 041 called `transformer/gemma2_train` **UNGATED** and used "largest AND ungated" to set eviction priority. Line 41 was the `pub mod`; line **40** was `#[cfg(feature = "gemma_lora")]`. It was gated at the exact commit measured. |
| every **ancestor** `mod` up to `lib.rs` | the item is bare, so you call it ungated — but its parent module is gated | riir-ai Issue 744 §4: `deltanet/{lm_head_lora_train,backward}` are bare, so a first pass claimed "compiled into every build, default included". The parent is `#[cfg(feature = "deltanet_inference")] pub mod deltanet;` (`lib.rs:90`). Never in a default build. |
| the crate's `default = [...]` | the list is read with the wrong instrument, and "no match" is mistaken for "no features" | A Python regex `^(\w+)\s*=\s*\[(.*?)\]\s*$` over `riir-engine/Cargo.toml` returned nothing, and that was written up as *"riir-engine declares no default features at all"* — in a riir-ai issue, a proposal, and **this table**. Measured with `cargo metadata`: **59 default features on a single 14,288-character line**, closure **76**, including `deltanet_inference` and `lora_still`. The wrong reading survived two rounds of "correction" and inverted a real finding (1,507 LOC of BPTT+AdamW *were* in every default build). **`cargo metadata --no-deps` is the authority; a regex over Cargo.toml is not.** |
| a consumer's `required-features` / dep-declaration features | you walk the `[features]` table and conclude a feature is unreachable | riir-ai Issue 744 §5: a closure walk of `riir-examples`' `[features]` "proved" `deltanet_ternary_inference` unreachable from `default`. It arrives via the `bonsai-*` rows, and the `riir-engine` dep is `default-features = false`. |

Note the first two are the **same mistake at different depths** — and the second
one was made *while writing the correction for the first*. That is the signature
of this defense being skipped rather than applied once.

`grep -B2` is **necessary but not sufficient**: it fixes link 1 and is blind to
links 2-4.

```bash
# The admissible form. All four links, for one item.
F=crates/riir-engine/src/deltanet/lm_head_lora_train.rs
grep -rnB2 "pub mod lm_head_lora_train;" crates/riir-engine/src/deltanet/mod.rs  # link 1
grep -rnB2 "pub mod deltanet;"           crates/riir-engine/src/lib.rs           # link 2 (repeat per ancestor)
python3 -c "import re,io;print(re.search(r'(?m)^default\s*=\s*\[(.*?)\]',io.open('crates/riir-engine/Cargo.toml').read()))"  # link 3
```

**Resolve the closure with the tool that owns it.** Manifests defeat regexes:
arrays span lines, or are one 14 kB line, or nest.

```bash
# The authority. Never a regex over Cargo.toml.
cargo metadata --no-deps --format-version 1 | python3 -c "
import json,sys
p=[x for x in json.load(sys.stdin)['packages'] if x['name']=='CRATE'][0]
f=p['features']; seen=set(); stack=list(f.get('default',[]))
while stack:
    k=stack.pop()
    if k in seen: continue
    seen.add(k); stack += [t for t in f.get(k,[]) if '/' not in t]
print(len(f.get('default',[])),'direct /',len(seen),'in closure'); print(sorted(seen))"
```

**And the closing rule: settle it with a build, not a read.** Every one of the
four cases above was a *reading* that survived review and died on first contact
with `cargo`. A feature matrix is cheap and decisive:

```bash
# rc=0 on every arm you claim works; the load-bearing arm is the one where the
# gate is supposed to EXCLUDE something and the crate must still compile.
for f in "" deltanet_inference deltanet_ternary_inference; do ...; done
```

Two traps when you do run it, both of which have reported a **false verdict**
here: `cargo ... 2>&1 | tail` makes `$?` the *tail's* exit, so a failed build
reports success; and in zsh `args="--features foo"; cargo check $args` passes one
argv entry (`error: unexpected argument '--features foo'`), so every arm of a
matrix loop "fails" while the code is fine. Redirect to a file and read `rc`
from cargo directly; pass flags via `"$@"`.

## Output format

After running an audit, produce a per-feature verdict table:

| Feature | Surface 1 (src .rs) | Surface 2 (lib.rs) | Surface 3 (Cargo.toml default) | Surface 4 (downstream) | Surface 5 (.benchmarks) | Verdict |
|---|---|---|---|---|---|---|
| `npc_sleep_time` | ✅ L44-66 | ✅ L629-642 | ✅ engine default-on | ✅ civ opt-in by design | ✅ bench 341 | deliberate-split |
| `<feature>` | ✅/❌ + line | ✅/❌ + line | ✅/❌ + line | ✅/❌ + line | ✅/❌ + file | see verdicts |
| `conformal_predictive_intervals` (post-Plan-468 audit, 2026-07-20) | ✅ no stale `//!` claims in `crates/katgpt-core/src/conformal/` | ✅ `crates/katgpt-core/src/lib.rs` L75-97 — promotion comment accurate (DEFAULT-ON since Plan 468, consumer gates STAY opt-in) | ✅ in `default = [...]` with Phase 21 comment | ✅ 7 consumer gates stay opt-in (deliberate three-layer split — see Defense 3 table) | ✅ Bench 340 + 560/562/563/564/565/567/568 all accurate | deliberate-split (clean after re-audit caught 3 missed surfaces: Bench 565 L7 header, `conformal_uq.md` §1.3 table rows 1/4/5, Plan 508 L8 Feature Gate line — all exhibited the append-only anti-pattern: header was updated, body was not) |
| **2026-08-15 quarterly audit** — 11 katgpt-core promotions since 2026-07-17 goat-audit window (focus set: poincare/chunked/causal/conformal/karc/hope/hebbian/ane_fused/clr/phase_separation/similarity_inference) | — | — | — | — | — | **7 fix-1-surface + 4 clean** — see the 2026-08-15 rows below |
| `similarity_inference` (2026-08-15) | ✅ mod.rs clean | ❌ lib.rs "Opt-in — Phases 2–7 pending" → FIXED (DEFAULT-ON 2026-08-11, Bench 579) | ⚠️ in default but NO phase-chain comment → added Phase 26 | ✅ no downstream | ✅ Bench 579 accurate | fix-2-surfaces |
| `phase_separation` (2026-08-15) | ✅ mod.rs clean | ❌ lib.rs "Opt-in until G1–G4 PASS — Phase 1 skeleton ships now" → FIXED (DEFAULT-ON 2026-08-07) | ✅ default + Phase 25 | ✅ riir-engine `phase_separation_salience` notes DEFAULT-ON | ✅ bench_571 accurate | fix-1-surface |
| `causal_identification` (2026-08-15) | ❌ mod.rs "G4 DEFERRED" → FIXED in place (closed 2026-07-18, Issue 183 / Bench 465 informational PASS) | ✅ | ✅ default + Phase 20 | ✅ `causal_id_consumer` clean | ✅ Bench 464/465 | fix-1-surface |
| `hebbian_kernel_memory` (2026-08-15) | ✅ lib.rs accurate (DEFAULT-ON + Defense-3 split) | ✅ | ✅ default + Phase 24 | ✅ neuron-db `hebbian_fact_store` "(now default-on)" accurate | ❌ bench_559 .rs G5 "BLOCKED/.issues/027" + "stays opt-in" println → FIXED (G5 PASS Bench 462 2026-07-25; .issues/027 resolved+removed) | fix-1-surface |
| `poincare_navigator` (2026-08-15) | ✅ | ✅ | ✅ default + Phase 19 | ✅ riir-engine `poincare_imagination` STAYS-OPT-IN-PERMANENTLY documented (Plan 497 quality refute) | ❌ Bench 449 L52 in-text "ships **opt-in**" — append-only anti-pattern (banner exists, body not fixed) → FIXED in place w/ post-promotion annotation | fix-1-surface |
| `karc_forecaster` (2026-08-15) | ✅ | ✅ | ✅ default + Phase 22 | ✅ `karc_runtime` default-on, probes opt-in by design | ❌ Bench 308 Phase-1 TL;DR "Feature stays opt-in" — append-only anti-pattern (§Phase 5.3 update exists) → FIXED in place w/ Post-Phase-5.3 annotation | fix-1-surface |
| `clr_weighted_set_attention` (2026-08-15) | ✅ no lib.rs status text | ✅ | ⚠️ in default, feature-def comment records promotion, but NO phase-chain comment → added Phase 24b (the 19b precedent) | ✅ Bench 354 update accurate | ✅ | fix-1-surface (convention) |
| `chunked_content_store`, `conformal_predictive_intervals`, `hope_capacity`, `ane_fused_chain` + 14 lib.rs STAYS-OPT-IN claims + fresh set (`multistep` pending-gate, `drift_segment`, `product_key_memory_episodic` "unchanged by Issue 650", FlashAR Eq21/`greedy_draft`, SoftmaxArgmax) + `.docs/01_orientation/overview.md` rows (2026-08-15) | ✅ | ✅ | ✅ | ✅ | ✅ | clean |

**Verdicts:**

- **clean** — all 5 surfaces accurate; no action.
- **fix-N-surfaces** — N surfaces have stale comments; fix in one commit
  per Defense 2.
- **deliberate-split** — asymmetric gate status is intentional per
  Defense 3; update comments on both sides to explain the split.
- **real-discrepancy** — same concern, different gate status across
  layers; propagate the promotion per Defense 2.
- **pending-gate** — gate has not yet passed; comment is accurate, no
  action (verify the gate is still pending by checking the latest plan
  phase status).

## Scope — every contract repo, derived

This section used to be a typed 8-row table headed "the 7 workspace repos".
It disagreed with its own header, omitted 11 repos that carry a `BOUNDARY.md`,
and listed `seal-online-remaster/`, **which does not exist** (the directory is
`seal-online-remaster-unity/`; the editor lives in `seal-game-editor/`). A
feature-gate audit scoped by that table audits a workspace that is not this one.

Derive it. The canonical count and the two axes (product set vs workspace) live
in `AGENTS.md` §"Repo count" — do not copy either number here.

```bash
cd /Users/katopz/git && for d in */; do
  [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
done
```

`-d "$d/.git"`, not `-e`: a `git worktree` has a `.git` **file**, and counting
one audits the same crate twice.

For each repo, **read its `AGENTS.md` first** — it documents the
canonical feature-gate layout and the layer-split conventions for that
repo. The repo-local rules override the general guidance here when they
conflict.

## Common failure patterns

### "Plausible but wrong"

The most common failure: a doc/plan/session-summary claims state X
based on a partial read of the code, and the claim is propagated
through downstream summaries without re-verification. **Always** grep
the production wiring before propagating the claim. The cost of
verification is ~30 seconds; the cost of a wrong claim landing in a
commit is a future audit to undo it.

### "Single-surface fix"

Fixing one stale comment but missing the other four. A reader who finds
a stale comment in `lib.rs` and a correct comment in the source file
concludes "the comment is just stale" — eroding trust in the whole
documentation surface. **Always** do the 5-surface grep; fix all stale
surfaces in one commit.

### "Append-only anti-pattern"

Touching the right surface but in the WRONG WAY: appending a footnote /
"Post-X update" section at the bottom of a doc while leaving the original
in-text claim untouched. A future reader who hits the in-text claim (which
appears first, in the TL;DR or body) trusts it without scrolling 100+
lines down to find the contradicting footnote. This is more subtle than
the single-surface fix because the surface IS touched — a naive `git
diff` audit concludes "Bench NNN was updated" — but the claim a reader
actually sees is still stale.

**Canonical failure (conformal primitive, 2026-07-20):** Plan 468's
promotion commit appended a "Plan 468 update" section to Bench 565 but
left the in-text "stays opt-in" claim at line 196 untouched. A follow-up
feature-gate-audit cycle (commits `97f32789` + `9d0716e22`) had to fix
the in-text claims across 7 surfaces — exactly because the append-only
approach had left the doc surface internally inconsistent.

**Fix discipline:** when correcting a stale claim, EDIT the original
sentence in place — convert present-tense "stays opt-in" to past-tense
"was opt-in at the time of this probe/bench" with an inline
`(Post-X update, YYYY-MM-DD, Plan NNN): ...` annotation. Footnotes are
acceptable ONLY for additional context that didn't exist in the original
claim's scope — never as a substitute for correcting the original claim.

### "Reflexive discrepancy framing"

Seeing asymmetric gate status and assuming it's a bug. This is the
single most common framing error in this codebase because the layer
split is the **dominant** pattern (6 of the 10 major runtime/primitive
rows in the Defense 3 table ship as engine-default-on +
consumer-opt-in). **Always** check whether
each layer gates the same concern before claiming a discrepancy.

### "Authoritative-sounding stale comment"

A `// Default-off until G1–G5 GOAT gate passes` comment in source code
**looks** authoritative — it cites the plan, the phase, the gates.
Readers trust it. But the comment was written **before** the gate
passed, and nobody updated it when the promotion landed. Treat any
"until... passes" comment as a candidate for staleness, and verify
against the `.benchmarks/NNN_*_promotion_review.md` (which is the
authoritative record of the gate's status).

## See also

- `~/.agents/skills/doc-sync/SKILL.md` — quarterly doc hygiene gate
  (sibling skill; feature-gate-audit focuses specifically on gate-status
  accuracy, doc-sync covers the broader `.docs/` + `README.md` sync)
- `katgpt-rs/.agents/skills/goat-audit/SKILL.md` — cross-repo GOAT
  cherry-pick audit (sibling skill; focuses on primitive propagation,
  this skill focuses on feature-flag status accuracy)
- `katgpt-rs/AGENTS.md` §"Feature Flag Discipline" — the GOAT promotion
  rule (the policy this skill enforces)
- `riir-ai/.benchmarks/341_npc_sleep_time_promotion_review.md` —
  canonical example of a complete promotion review (G1–G5 + modelless
  + latent boundary + β-tuning finding)
- Commit `f865913e` on `riir-ai/develop` (2026-07-18) — the canonical
  example of this skill's three defenses catching a prior session's
  errors
