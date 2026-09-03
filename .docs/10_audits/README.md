# Audits — One-Off Consolidation / Rubric Audits

> **What you find here.** Point-in-time audits that informed a structural
> decision. These are historical records, not living API docs — kept here so the
> reasoning behind a refactor or rubric stays traceable.

## Docs

| Doc | Role |
|---|---|
| [`loser_sweep_audit.md`](loser_sweep_audit.md) | Phase 0.5 — loser-sweep audit (Proposal 003) |
| [`claim_rubric_audit.md`](claim_rubric_audit.md) | Claim-rubric audit — research notes vs `Claim` fixtures (Plan 307 T4.2) |
| [`cross_repo_consolidation_audit.md`](cross_repo_consolidation_audit.md) | Cross-repo consolidation audit — riir-ai / riir-chain / riir-neuron-db (2026-07-06) |
| [`code_smell_file_size_audit.md`](code_smell_file_size_audit.md) | Code-smell + file-size audit (Issue 162 + sub-issues 164–178, 2026-07-17) — split outcomes, KEEP verdicts (weaver / tree_builder), softmax audit, stale-TODO audit |
| [`doc_status_auditors.md`](doc_status_auditors.md) | Doc-status auditors — `scripts/bench_doc_audit.py` (`.md` labels) + `scripts/cargo_comment_audit.py` (Cargo.toml inline comments); run-after-promotion discipline (Issue 180, 16-session audit cycle 2026-07-18) |
| [`sibling_doc_drift_auditors.md`](sibling_doc_drift_auditors.md) | Design record of the doc-drift auditors' current shape — dialect blindness, `(package, feature)` reachability closure, `on by transitive default`, canaried pins, the three cadence tiers (Issue 702, closed 2026-09-01) |
| [`cfg_gated_silent_zero_pass.md`](cfg_gated_silent_zero_pass.md) | `#![cfg]`-gated targets that print a green `0 passed` — corrected 19-repo measurement, the release-profile retraction, the load-bearing token table, and the open sibling T3 table (Issue 713, closed 2026-09-03) |
| [`alloc_gate_per_thread_counter.md`](alloc_gate_per_thread_counter.md) | Alloc gates counted sibling tests' allocations — per-thread `ThreadCounter`, two-sided verification, provenance of `62911111` (Issue 714, closed 2026-09-03) |
| [`percentile_index_tail_support.md`](percentile_index_tail_support.md) | The 12→0 repair campaign that took its own sites out of the population — the `TRUNC-VAR` variable-p class, the `.trunc()` hole, the exact floor-vs-nearest-rank boundary, and why the floor is not ratcheted (2026-09-03) |
| [`debug_release_profile_axis.md`](debug_release_profile_axis.md) | The profile is part of the claim — orphaned `#[cfg]` across a blank line (Issue 715) and the full gate's release Layer 6 (Issue 716), both closed 2026-09-03 |

## Note

These are kept under `.docs/` rather than `.research/` because each documents a
**structural decision inside this repo** (what to delete / how claims must be
evidenced / where consolidation ends), not an academic distillation. If an
audit's decision is fully absorbed into code + plans and no longer needs a
standalone record, it can be deleted; until then it lives here.
