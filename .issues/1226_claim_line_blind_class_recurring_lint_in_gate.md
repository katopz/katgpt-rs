# Issue 1226 — claim-line blind class keeps recurring across repos: lint the claim lines inside a gate sessions already run

**Status:** Open (2026-11-16) — the write-time/discipline gap, not a script bug; `claim_lease_check.py --lint` itself works.
**Claimed:** zed-a90003f9@amo, 1794870527, expires 2026-11-18T23:08:00Z

## The measured class

The repo-wide read-time lease check goes BLIND whenever any file carries a claim-shaped line the canonical regex cannot parse — one malformed line blinds EVERY lane's adjudication in that repo. Repaired twice, recurred within days:

1. **riir-infer, repaired `e7dea61e`** (2026-11-16, the i1186 housekeeping cycle): 3 lines — a parenthetical+scope tail, a released-claim residue standing above its own close, a stale claim + a LIVE lease malformed.
2. **riir-ai, repaired `943551e2e`** (2026-11-16, the i1176 pre-registration lane — same day, different repo): 2 lines — a fresh re-claim carrying a `— scope: …` tail (`.issues/1153`), an ms-epoch (13-digit) with no comma (`.issues/1177`).

Root cause: agents write claim lines with scope tails or misformatted epochs and never run `--lint` before push. The AGENTS.md instruction ("Lint first: … --lint <file>") is discipline, not enforcement — and both repair sessions were OTHER agents' housekeeping passes finding the blindness en route, not the offending sessions.

## The fix direction (the minimal enforcement point)

Fold claim-line linting into a gate that EVERY numbered-file session already runs post-landing: `scripts/numbering_gate.py` (the bump-rides-the-allocation-commit discipline makes it the one gate touched in every claim/issue/plan commit). Cheap predicate: reuse `claim_lease_check`'s claim-shaped-line detector per `.issues/`/`.plans/` walk — a MALFORMED line is a wall (exit 1 naming file+line), EXPIRED stays advisory (re-claim is a normal re-read). Two directions pinned: lint-finds-planted-malformed (canary) and lint-clean-on-current-trees (population floor = the per-dir file counts the gate already carries).

Alternatives considered and rejected: a git pre-commit hook (needs per-repo `.githooks` wiring in 20+ repos — the enforcement surface the workspace deliberately keeps out of hooks), and docs-only re-wording (both recurrences happened DESPITE the AGENTS.md instruction — this is the same lesson as every "docs didn't hold, gate it" arc in HISTORY.md).

Scope: katgpt-rs `scripts/` only (the gate + the shared detector import); the sibling repos need zero changes (the gate runs against their paths the way `numbering_gate.py .` already does).

## Evidence refs

- `e7dea61e` (riir-infer) + the boundary-guard 491st row · `943551e2e` (riir-ai) + the 493rd row (both rows name the recurrence + the lint-at-write note).
- The detector: `scripts/claim_lease_check.py` `CANONICAL_CLAIM` shape (`**Claimed:** <session>, <epoch digits>, expires <ISO-UTC>`).
