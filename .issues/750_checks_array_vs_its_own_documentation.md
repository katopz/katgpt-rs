# Issue 750 — docs_gate.sh's CHECKS array vs the AGENTS.md table documenting it

STATUS: RESOLVED (gate landed, revert-probed 5 ways) — 2026-09-12

## The defect

`docs_gate.sh`'s `CHECKS` array and AGENTS.md's "one line per check" table are
hand-duplicated lists of the same thing, and **nothing compared them**.

Found already drifted, while fixing riir-ai Issue 749: `docs_gate.sh` described
`population_sync_gate.py` as "the **six** independent contract-repo
predicates". The script's own docstring says SEVEN, AGENTS.md's table says
seven, and Issue 734 is the commit that added the seventh. The wrong number sat
in the text a human reads on **every single docs-gate run** and nothing could
see it — the array's descriptions are printed, never checked.

This is the second instance of a class `docs_gate_paths_sync.py` already gates
one axis over (docs_gate.yml's two hand-duplicated trigger `paths:` lists). The
second instance is where it stops being a one-off.

## Resolution — `scripts/docs_gate_checks_sync.py`, in CHECKS

Two assertions, deliberately different in strictness:

1. **MEMBERSHIP** of script names, both directions — never cardinality. A count
   that MATCHES is not a checksum over a set. A check in CHECKS but not the
   table is one nobody reading the contract knows runs; a check in the table
   but not CHECKS is one nothing runs.
2. **QUANTITY WORDS** only, after issue references are stripped — not the
   prose. The two lists legitimately differ in emphasis (`by membership` vs
   `by MEMBERSHIP`, `set -u abort` vs `abort`), and a gate demanding
   byte-identity is a gate people route around. A NUMBER may not differ,
   because a quantity is a claim. Issue numbers are addresses, not quantities,
   and are stripped first — measured: **1 of 15** rows cites one in the array
   and omits it in the table, with no contradiction.

Floor `MIN_ROWS = 10` on **both** parses: a parser that silently reads zero
rows reports perfect agreement between two empty sets. Exit 0 / 1 / **2** — an
unparseable list is an untrustworthy instrument, not drift.

### Revert-probed five ways — every verdict distinguishable

| probe | result |
|---|---|
| reinstate the real drift (seven → six in docs_gate.sh) | exit 1, `['six']` vs `['seven']` |
| row in CHECKS, not in the table | exit 1, "a check nobody knows runs" |
| row in the table, not in CHECKS | exit 1, "a documented check that nothing runs" |
| AGENTS.md section retitled | exit **2**, table cannot be located |
| `CHECKS=(` renamed | exit **2**, array cannot be read |
| restored | exit 0 |

The gate documents itself: registering it added a row to both lists, so it
checks its own registration on the next run.

## Note — this number is the one riir-ai Issue 749 was about

`750` is precisely the number whose local allocation would have silently
rebound AGENTS.md's dangling `Issue 750` citation to this file instead of
`riir-ai/.issues/750_behavior_first_quantization_promotion_gate.md`. Allocating
it one commit after the citation was qualified (`e258fdaa`) is the intended
demonstration, not a coincidence: the hazard is closed, so the number is now
safe to use. `issue_citation_gate.py` keeps it that way.
