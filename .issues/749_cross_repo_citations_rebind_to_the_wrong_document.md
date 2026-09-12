# Issue 749 — a cross-repo `Issue N` citation rebinds to the WRONG document

STATUS: RESOLVED (gate landed + 8 citations qualified) — 2026-09-12

## The defect

`numbering_gate.py` protects a number from being allocated **twice in one
repo**. It structurally cannot see the failure one level up: a citation whose
referent lives in a **different** repo.

AGENTS.md §Feature Flag Discipline cited `Issue 750 T3` for the lossy-surface
promotion rule. This repo's `.issues/.highwater` read **748**. The referent is
`riir-ai/.issues/750_behavior_first_quantization_promotion_gate.md`, RESOLVED
and removed in riir-ai `b559d2da3` — so a worktree-only search for it finds
nothing at all.

Today that citation dangles. Two allocations from now — 749, then 750 — it
stops dangling and starts resolving **silently, to the wrong document**. A
dangling reference is an inconvenience a reader notices. A reference that
rebinds to a real but unrelated issue is a wrong answer delivered with a
straight face, and nothing in the repo could have caught it.

Surfaced as an unexplained observation in Issue 748 ("`.highwater` reads 748
while AGENTS.md cites Issue 750 — structurally invisible to the numbering
gate"). This issue is that observation measured and closed.

## Measured (2026-09-12)

Over AGENTS.md + HISTORY.md, 138 citations, five kinds, 19 contract repos:

| | |
|---|---|
| unqualified cross-repo citations | **8**, over **3** numbers |
| `Issue 513` (x3) | `riir-train/.issues/513_required_features_rows_are_unverified.md` |
| `Issue 750` (x3) | `riir-ai/.issues/750_behavior_first_quantization_promotion_gate.md` |
| `Issues 490/493` (x2) | `riir-ai/.issues/490_orchard_extraction_sdk_to_riir_games.md`, `493_sdk_motivation_move_to_riir_games.md` |
| Bench / Proposal cross-repo cites | 6, **all already qualified** — the class is not uniform |

Every one resolved **uniquely by title match** — `required-features` → 513,
orchard → 490, motivation → 493, quantization → 750 — so the repair is
mechanical, not a judgement call. The convention was already in the same file
four lines away: "riir-ai `.issues/892`", "894 resolved same day in riir-ai".

## ⛔ The stated blind spot — 63 AMBIGUOUS

**63** cited numbers exist BOTH locally and in a sibling. The referent is
undecidable from the number, and the gate is green over every one of them by
construction (its predicate is "does not resolve locally"). A bare `Issue 47`
meaning riir-ai's 47 reads as this repo's 47 and always will.

It is **reported every run, deliberately not gated**. A ceiling on it reds
whenever somebody writes a perfectly *correct* citation to a new local number
a sibling also happens to have — a nuisance red for a right action, and a gate
that reds on right actions is a gate that gets bypassed
(`staged_set_audit.py`'s own rationale, one axis over). It has the standing of
tail support in the percentile audit: a quantity that sizes the blind spot and
orders the rows, never a verdict.

## Resolution

- `scripts/issue_citation_gate.py` + `scripts/issue_citation_floors.txt`, in
  docs_gate's CHECKS. Exit 0 clean / 1 drift / **2 instrument untrustworthy**.
- Population **derived**, not re-derived: it imports
  `numbering_drift_sweep.contract_repos` — the seventh predicate
  `population_sync_gate.py` already keeps in sync. An eighth derivation would
  be an eighth silent divergence.
- History is walked, not just the worktree: the noise-reduction rule REMOVES a
  resolved issue file, so a worktree-only walk calls a live citation dangling.
  Issue 750 is exactly that shape.
- Two floors (`min_repos`, `min_citations_scanned`) — a ceiling is green over
  whatever the instrument can see, so the finding count needs the population
  AND the walk size under it.
- The 8 rows qualified in AGENTS.md + HISTORY.md.

### Verified by revert probe — the pin FIRES

| probe | result |
|---|---|
| un-qualify `AGENTS.md:610` | exit 1, names `riir-ai` |
| un-qualify the **list tail** `Issues 490/493` | exit 1, **2** rows — 490 *and* 493 |
| `min_citations_scanned` floor breached | exit **2** ("the regex went blind, not the prose clean") |
| `min_repos` floor breached | exit **2** |
| restored | exit 0 |

The list-tail probe is the one that earned its keep: the first version of the
scan read only the head number and reported `Issues 490/493` as **one**
citation, hiding 493 entirely. A plural kind word can head a list
(`Issues 724, 725`, `Issues 490/493`, `Plans 596 and 597`) and reading only
the head under-reports the class.

The gate also caught **its own registration**: adding the row citing
`Issue 749` to AGENTS.md reddened it, because 749 was not yet allocated
locally and riir-ai/riir-train both have one. Filing this file cleared it.

## Found alongside — `docs_gate.sh` vs AGENTS.md, ungated

`docs_gate.sh`'s CHECKS description for `population_sync_gate.py` said "the
**six** independent contract-repo predicates"; AGENTS.md's table and the
script's own docstring both say **seven** (Issue 734 added the seventh).
Fixed in the same commit.

Nothing gates the CHECKS array against the AGENTS.md table that documents it —
a row can be added to one and not the other, and a description can drift in
either. `docs_gate_paths_sync.py` already exists for exactly this shape one
axis over (docs_gate.yml's two hand-duplicated `paths:` lists). Follow-up:
`.issues/750` (this repo).
