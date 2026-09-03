---
name: proposal
description: Write a reasoned architectural proposal (.proposals/NNN_*.md) grounded in focused codebase grep + prior-art paper search. Use when arguing for a design change, new primitive, or architectural decision — the question is "should we do X?" not "how do we do X?" (that's a plan) or "what does paper Y distill to?" (that's research). Enforces honest caveats, fusion lineage, and the existing proposal format. Searches arxiv + codebase + sibling-repo proposals/research/plans before writing.
---

# Proposal — Reasoned architectural argument with codebase grounding + prior art

A **proposal** is the layer between `research` (paper distillation) and `plan`
(task execution). It argues for an architectural change, a new primitive, or a
design decision — grounded in (a) what the codebase already ships, (b) what
sibling repos have already proposed, and (c) what the literature says. It is
**not** a paper summary, **not** a task list, and **not** an audit.

The deliverable is a `.proposals/NNN_*.md` file in the repo that owns the
affected surface, written in the established proposal format (see §Output
format). It MUST ship with honest caveats and a fusion lineage — these are
non-negotiable per the canonical proposals in `katgpt-rs/.proposals/`.

## When to use

- The user asks "should we do X?" / "what if we Y?" / "propose a design for Z".
- A design question spans multiple primitives, repos, or feature flags and
  needs a reasoned argument before any plan is opened.
- A new feature wants GOAT-gate discipline but the *design* is still open —
  the proposal settles the design, the plan executes it.
- A cross-repo change touches the sync boundary, freeze/thaw, or raw↔latent
  bridge and needs the boundary rule reasoned through explicitly.

## DO NOT use for

- **Paper distillation** → use the `research` skill. Proposal may *cite* a
  paper as prior art but does not distill it.
- **Task execution** → write a `.plans/NNN_*.md`. Proposals sketch a phased
  rollout; plans own the `- [ ]` task list.
- **POC / refactor / optimization tracking** → file `.issues/NNN_*.md` per
  the global rule ("Create issue at .issues for poc, proof, optimization or
  refactor task, do not create plan").
- **Cross-repo GOAT cherry-pick audit** → use the `goat-audit` skill.
- **Bug fixes with no architectural angle.**

## Repos in scope (the product/distillation set — **8**)

Same layout as the `research` and `goat-audit` skills. **Canonical home for this
list: `katgpt-rs/AGENTS.md` §"Repo count"** (and Research 003) — it is copied
here for reading convenience, so when the two disagree, AGENTS.md wins. This
copy said "7-repo stack" and omitted `riir-dapps` from 2026-08-20 until
2026-09-01, which is what a duplicated count does:

```
katgpt-rs          ← public engine (default target for generic primitives)
riir-ai            ← private runtime/game (cognitive, freeze/thaw, HLA, ...)
riir-chain         ← private chain (LatCal, quorum, sync-boundary bridge)
riir-neuron-db     ← private neuron-shard leaf (Pod, freeze, consolidation, AnyRAG)
riir-train         ← private training vault (training-only methods — research-only routing)
riir-game-sdk      ← private game-vocabulary facade + dev-tool workspace
                      (consumers: riir-mmorpg-examples, seal-online-remaster; vocabulary source is
                      riir-games-shared in riir-ai workspace, re-exported via facade)
riir-dapps         ← private dApp layer (game outcome → generic chain settlement;
                      added 2026-08-20 — route settlement COMPOSITION here, not
                      riir-chain, which owns only value/authority primitives)
```

> The 8 above are the *product/distillation* set, not the workspace. The
> workspace is 18 repos with a root `BOUNDARY.md` — see the `substrate-first`
> skill's Step 2 for the derived enumeration. Routing targets a product repo;
> **searching** must cover all 18.

**Target-repo routing rule:** pick the repo that *owns the surface being
changed*. A primitive that ships in `katgpt-core` → proposal in
`katgpt-rs/.proposals/`. A runtime composition change → `riir-ai/.proposals/`.
A sync-boundary / LatCal / commitment bridge → `riir-chain/.proposals/`. A
shard / freeze / consolidation / AnyRAG mechanism → `riir-neuron-db/.proposals/`.
A game-vocabulary / SDK facade / backend abstraction change →
`riir-game-sdk/.proposals/`. If the proposal spans repos, file it in the repo
that owns the primary surface and cross-reference from the others.

## Pre-flight (MANDATORY before any grep)

Run all of these in parallel:

1. **`read_file <target_repo>/.proposals/.highwater`** — get the next number.
   If the file does not exist, `list_directory <target_repo>/.proposals/` and
   use `max(existing NNN) + 1`. **Always write the new highwater back** after
   creating the proposal (per global AGENTS.md numbering discipline — numbers
   are monotonic and never reused).
2. **`list_directory` `.proposals/` in ALL SEVEN repos.** These are the
   existing proposals you must not duplicate and must reason about.
3. **`read_file` the 1–2 closest existing proposals** (by filename match to
   the topic) — these set the bar for prose style, caveats, and rigor. Match
   their format.
4. If the proposal touches the sync boundary, freeze/thaw, or raw↔latent
   bridge — `read_file` the relevant section of `katgpt-rs/AGENTS.md` (the
   "Latent vs Raw Space Rules" and "Sync Boundary Rule" blocks) so the
   proposal reasons about domain classification correctly.

## Workflow

### Step 1 — Topic decomposition (one paragraph)

Write (in your own working, not into the file yet):

- **The core question** in one sentence. "Should we ship X to achieve Y?" not
  "Implement X." If you can't phrase it as a should-question, it's a plan, not
  a proposal.
- **3–5 key mechanism terms** from the topic.
- **Vocabulary translation** — for each key term, brainstorm ≥2 codebase-
  equivalent names by asking "if we already shipped this, what would we call
  it?" (see §Vocabulary translation tips below). This is the same discipline
  as the `research` skill §Workflow 1.2, but **scoped to the proposal topic**
  — do NOT pull the whole standing list.

### Step 2 — Focused codebase grep (the "more focus" step)

This is where the proposal skill differs from `research`: the grep is
**scoped to the proposal topic**, not the full corpus. Run these in parallel,
every contract repo (derived below), all four document layers + code:

DERIVE the repo set; never type it. The four-repo list this replaced
(`katgpt-rs riir-ai riir-chain riir-neuron-db`) could not see **784 of the
2,509 documents** in these layers — 31%, and `.issues` was the worst at 21 of
145 (85% invisible). A prior-art grep exists to stop a duplicate proposal, so a
layer it cannot read is a layer it cannot clear. Measured 2026-09-01; canonical
repo set lives in `AGENTS.md` §"Repo count", derived, never copied here.

```bash
cd /Users/katopz/git

# Layers A-D — one derived set per document layer. Substitute the layer name.
# (.proposals = strongest precedent and MUST be reasoned about per hit;
#  .research = prior art already distilled; .plans = in flight; .issues = tracked)
for layer in proposals research plans issues; do
  echo "=== .$layer ==="
  grep -rnE "<topic terms>|<codebase-equivalent terms>" \
    $(ls -d */ | while read -r d; do
        [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && [ -d "$d.$layer" ] \
          && printf '%s.%s ' "$d" "$layer"
      done)
done

# Layer E — shipped code (what actually exists today)
grep -rnE "<CamelCaseStruct>|<snake_case_fn>" \
  --include='*.rs' --exclude-dir=target --exclude-dir=.git \
  $(ls -d */ | while read -r d; do
      [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && printf '%s ' "$d"
    done)
```

`-d "$d/.git"`, not `-e`: a `git worktree` has a `.git` **file**, and including
one reports a single document twice. `--exclude-dir=target` is not cosmetic —
without it Layer E spends nearly all its time inside build directories.

**MANDATORY reasoning per hit** (this is the user's explicit ask: "more focus
and reasoning on related proposal topic when grep"). For every hit, classify
it in one line:

| Classification | Meaning | Action in proposal |
|---|---|---|
| **Precedent** | Already proposed / shipped the same mechanism | Proposal MUST cite it; explain what's new |
| **Contradiction** | Existing proposal/plan argues the opposite | Proposal MUST address the contradiction explicitly |
| **Complement** | Adjacent mechanism that the proposal would combine with | Cite in "Fusion lineage" section |
| **Substrate** | The primitive the proposal would build on | Cite in "What ships now" / "Proposed design" |
| **Duplicate** | The proposal would re-ship existing work | **STOP — downgrade to issue or cancel** |

If Layer A returns a hit that is a **precedent** or **duplicate**, the proposal
is likely redundant — say so to the user before writing. Do not silently
re-propose.

If Layers A–D all return zero hits, **re-run with at least one more semantic
angle** (grep for the *output behavior* — "swap when X" — instead of the
*mechanism name* — "tightness monitor"). Zero hits across all five layers is
rare; the prior cause is usually vocabulary mismatch, not novelty.

### Step 3 — Prior-art paper search (online)

The user's explicit second ask: "also find paper online for it too."

1. **arxiv keyword search** using the standing URL from global AGENTS.md:
   Use web search mcp to run **2–3 keyword variants** (paper vocabulary AND codebase vocabulary
   from Step 1). One search is rarely enough — the right keyword often lives
   in the codebase-equivalent term set, not the user's phrasing.

2. **`web_search_prime` for non-arxiv prior art** when the topic is
   engineering (lock-free, deterministic replay, anti-cheat, commitment
   schemes) rather than ML — the relevant prior art may be in engineering
   blogs, RFCs, or database literature, not arxiv.

3. **Fetch the 2–3 most promising hits** via the jina PDF reader
   (`https://r.jina.ai/https://arxiv.org/pdf/{ID}`) or `fetch` for blog/RFC
   content. Do not fetch more than 3 — the point is grounding, not survey.

4. **Distill in one paragraph each**: what is the transferable insight? What
   does the paper do that we should NOT copy (because it's training-only,
   softmax-based, or violates the latent/raw boundary)? Cite these in the
   proposal's References section.

**Honesty rule:** if the prior-art search finds a paper that already proposes
the exact mechanism, the proposal MUST say so in §Honest caveats. Do not
re-attribute a paper's idea as our invention. The canonical example of doing
this right is `katgpt-rs/.proposals/004_adaptive_causal_calibration.md`
caveat 1: "The adaptive scheme is our invention. HydraHead supplies the
causal scorer; the escalate-on-suspects mode is our design."

### Step 4 — Reasoning (the argument)

Synthesize before writing. Answer each in one paragraph of working prose:

1. **The gap.** What does the codebase (Layer E) + existing proposals
   (Layer A) + plans (Layer C) + literature (Step 3) NOT cover that this
   proposal fills? If you can't state the gap in one sentence, the proposal
   isn't ready.
2. **The proposed design.** The mechanism, in pseudocode or a diagram if
   helpful. Be concrete — abstract designs invite ambiguity.
3. **Domain classification** (if the proposal touches state that crosses a
   boundary). For each piece of state, classify:
   - Physical domain (position, HP, wallet) → MUST stay raw, deterministic,
     synced. Bridge functions must be zero-allocation.
   - Semantic domain (emotion, curiosity, style) → SHOULD operate in latent
     space, project to scalars at the boundary.
   - Sync boundary crossing → raw→latent projection via dot-product + sigmoid
     (never softmax); latent→raw via clamp.
4. **Honest caveats** (MANDATORY — no proposal ships without this section).
   What's unvalidated? What's the proposal's inventor's-regret risk? What
   would cause promotion to be rejected? See proposal 004 §"Honest caveats"
   for the bar.
5. **Fusion lineage.** What 2–3 existing primitives / proposals / research
   notes does this combine? Name them with file paths.

### Step 5 — Write the proposal

Create `.proposals/NNN_<short_title_with_underscores>.md` in the target repo
using the format in §Output format below. Then:

1. **Write the new highwater back**: `write_file <repo>/.proposals/.highwater`
   with the new NNN (zero-padded, e.g. `005`). If the file did not previously
   exist, create it.
2. **Cross-reference**: if the proposal cites research notes, plans, or
   sibling-repo proposals, link them with relative paths
   (`../.research/NNN_*.md`, `../../../riir-ai/.plans/NNN_*.md`).

### Step 6 — Commit

Per global AGENTS.md rule (which OVERRIDES the Zed default of "do not commit
unless asked" — see the user's personal AGENTS.md):

```bash
cd <target_repo>
git add .proposals/NNN_*.md .proposals/.highwater
git commit -m "docs: file proposal NNN — <one-line title> (target: <repo>)"
```

Commit on `develop` (or `main` for riir-train, which has no develop branch).
No feature branches. Do not push.

## Output format

Match the style of `katgpt-rs/.proposals/004_adaptive_causal_calibration.md`.
Use this template — sections with **(MANDATORY)** must be present; others are
optional depending on the proposal.

```markdown
# Proposal NNN — <Title>

Status: **draft | shipped Phase N | REJECTED Phase N (<reason>) | deferred**
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Fusion of: <Plan/Research/Proposal NNN × NNN × NNN>
Related: [Research NNN](../.research/NNN_*.md), [Plan NNN](../.plans/NNN_*.md)

## TL;DR

<2–4 sentences: what's proposed, the win, the cost. State explicitly whether
this is a katgpt-rs invention or a distillation of prior art. If the latter,
name the paper.>

## The problem this solves

<One paragraph: the concrete gap in the codebase today. Cite the file paths
that would benefit. State what goes wrong (or is wasted, or is unsafe) without
the proposed mechanism.>

## The proposed design

<The mechanism in concrete form — pseudocode, diagram, or struct definitions.
This is the contract a future plan would implement against.>

## Honest caveats — READ BEFORE IMPLEMENTING  (MANDATORY)

<Numbered list of unvalidated assumptions, inventor's-regret risks, and
conditions under which the proposal should be rejected. No proposal ships
without this section. See proposal 004 for the bar — 4 caveats is typical.>

## Fusion lineage

<Which 2–3 existing primitives / research notes / proposals this combines, and
what the combination produces that none of them alone can.>

## GOAT gate

<If the proposal touches a feature flag, define the gate it must pass before
promoting to default-on. Cover G1 correctness, G2 perf, G3 no-regression, and
G4 (alloc-free or equivalent). For UQ-bearing primitives, mandate the
conformal-naive floor per the "Report the Floor" rule in katgpt-rs/AGENTS.md.>

## What ships now (<repo>) vs deferred (<repo>)

### Ships now — <scope>
<concrete: which crate, which module, which feature flag>

### Deferred — <scope>
<what waits for validation, and where it lands when it passes>

### Explicitly NOT shipped by this proposal
<boundary statement — what a reader might assume is included but isn't>

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — <scope>
- [ ] T1.1 ...

### Phase 2 — ...

## Risks

<Numbered list. Distinguish perf risks, correctness risks, and architectural
risks (e.g. "this couples repo A to repo B's sync layer").>

## Out of scope  (RECOMMENDED)

<Explicit boundary — what a reader might expect this proposal to cover but
doesn't. Prevents scope creep at plan time.>

## References

<Numbered list of papers / RFCs / blog posts fetched in Step 3, with arxiv
links. Mark which are distilled vs cited-only.>

## TL;DR

<One-sentence closer — repeat the verdict (ship / defer / reject) and the
next action (open Plan NNN, wait for G1, etc.).>
```

## Vocabulary translation tips

Scoped to the proposal topic — do NOT pull the full standing list from the
`research` skill. For each key term in the topic, brainstorm ≥2 codebase
equivalents using these common patterns:

| Paper / user phrasing | Codebase equivalent (check via grep) |
|---|---|
| "speed hack" / "teleport" | `anti_cheat`, `validate_movement`, `v_max`, `tick_replay` |
| "memory tamper" | `MerkleFrozenEnvelope`, `architecture_root`, `blake3`, `merkle_root` |
| "adaptive" / "dynamic" | `adaptive_k`, `AdaptiveKRouter`, `sigmoid_gate`, `dynamic_pair` |
| "client / server / authoritative" | `pillar`, `quorum`, `SyncBlock`, `ChainConsensus` |
| "snapshot" / "checkpoint" | `freeze`, `thaw`, `KarcShard`, `ArchetypeBlendShard`, `BranchBank` |
| "direction vector" / "embedding" | `hla`, `style_weights`, `SenseModule`, `project` |
| "bridge" / "boundary crossing" | `bridge`, `exterior_derivative`, `codifferential`, `LatCal` |
| "validate" / "verify" / "audit" | `ConstraintPruner`, `claim_rubric`, `validator`, `forensic` |

The grep is the source of truth — the table above is a starting hint, not a
dictionary. Always grep both sets (paper vocab AND codebase vocab).

## Cross-references

- `katgpt-rs/.proposals/004_adaptive_causal_calibration.md` — the canonical
  proposal example. Match its prose style, caveats, and section ordering.
- `katgpt-rs/.agents/skills/research/SKILL.md` — paper distillation workflow.
  Use when a proposal cites a paper that needs deeper distillation than a
  References entry.
- `katgpt-rs/.agents/skills/goat-audit/SKILL.md` — cross-repo cherry-pick
  audit. Run before any proposal that consumes a katgpt-rs primitive into
  riir-*, to avoid re-proposing already-wired work.
- `katgpt-rs/AGENTS.md` — "Latent vs Raw Space Rules", "Sync Boundary Rule",
  feature-flag discipline, GOAT gate, "Report the Floor" UQ extension.
- Global `~/.agents/` rules — numbering discipline, commit convention
  (`docs:`/`feat:`/`fix:` prefix, on `develop`, no feature branches, no push).

## TL;DR

**Pre-flight (mandatory):** `read_file` the target repo's
`.proposals/.highwater` (or scan for max NNN); `list_directory` `.proposals/`
across all 7 repos; `read_file` the 1–2 closest existing proposals to match
style; if the topic touches sync/freeze/bridge, `read_file` the AGENTS.md
boundary rules.

**Workflow:** topic decomposition (one-sentence should-question + 3–5 terms +
scoped vocabulary translation) → **focused 5-layer grep** (proposals /
research / plans / issues / code, across all 7 repos, with per-hit reasoning:
precedent / contradiction / complement / substrate / duplicate) → prior-art
search (2–3 arxiv keyword variants + web for non-ML topics + fetch ≤3 papers)
→ reasoning (gap, design, domain classification, caveats, fusion lineage) →
write `.proposals/NNN_*.md` in the established format → write highwater back →
commit with `docs:` prefix on `develop`.

**Hard rules:** honest caveats section is MANDATORY (no proposal ships without
it); fusion lineage MUST cite 2–3 existing primitives; domain classification
MUST reason raw-vs-latent when state crosses a boundary; if Layer A grep
returns a precedent or duplicate, STOP and tell the user before writing;
scoped vocabulary translation only (not the full research standing list).
