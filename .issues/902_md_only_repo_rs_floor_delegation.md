# Issue 902 — md-only contract repos can never pass `len_derived` / `shared_temp_path` (the delegated `.rs` floor demands non-zero)

**Status:** OPEN — filed 2026-09-26 at the riir-instinct registration (`54edbd097`). Measured; not fixed.

## Finding

`len_derived_drift_sweep.py` and `shared_temp_path_drift_sweep.py` carry no
`min_rs_files` column of their own. They DELEGATE the walk floor to
`scripts/orphaned_attr_drift_floors.txt` and assert that every pinned repo has
a **non-zero** row there (AGENTS.md §len_derived: "the sweep reds if any repo
it pins loses its non-zero row there"). A repo born md-only (no tracked `.rs`)
has a truthful `0` row, so both sweeps red on it forever, with zero content
findings.

- riir-reflexer (md-only registration 2026-09-25) has red `len_derived`
  since birth.
- riir-instinct (md-only registration 2026-09-26) reds both sweeps.

This is a sweep that always reds on a correct repo, the cries-wolf state
Issue 793 exists to prevent.

## Plan

- [ ] **T1** — the delegation assertion distinguishes "row lost / zeroed by a
      walk regression" from "repo has no Rust by construction". Candidate: a
      zero row is accepted when the repo's `git ls-files '*.rs'` is empty at
      HEAD (measured, not declared), and reds the moment a `.rs` file lands
      without the row being raised.
- [ ] **T2** — arms: md-only repo → green; md-only repo gains a `.rs` file with
      its row still 0 → red; code repo with a zeroed row → red (the original
      protection, kept).
- [ ] **T3** — re-run both sweeps; riir-reflexer and riir-instinct go green
      with no pin edits.

## References

katgpt-rs `54edbd097` (riir-instinct registration), `65869615d`
(riir-reflexer registration); AGENTS.md §len_derived + §shared_temp_path
sweep notes.
