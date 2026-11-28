# Issue 1252 — numbering gates are blind to directory-class allocations (reflex bench-208 incident class)

**Status:** OPEN — filed 2026-11-28 out of the live reflex incident this same session (zed-a12b3f7d@amo).
**Claimed:** zed-a12b3f7d@amo, 1795857342, expires 2026-11-30T09:15:42+00:00

## The incident (measured live, 2026-11-28)

reflex landed `.benchmarks/208_i135_posture_root_cause/` (the i135 closure evidence dir,
`1bbedde`) **without bumping `.benchmarks/.highwater`** — the counter stayed 207 while a
number-208 holder existed. Nothing caught it: `numbering_gate.py` PASSED, and the drift
sweep's same census is equally blind. The bump only landed because a later housekeeping pass
happened to `cat` the counter (`ec75b04`). One directory later the class fires for real: the
next `.benchmarks` allocation reads 207, allocates 208, and collides with the landed dir —
the exact "value + 1 is ALREADY TAKEN" incident the gate exists to wall (gate docstring,
Issues 724/725).

## Root cause

`NUMBERED = re.compile(r"^(\d+)_.+\.md$")` — the allocated-number census in BOTH instruments
sees numbered `.md` files only:

- `numbering_gate.py` `scan()` → `group_numbered()` drops every non-md entry; the
  above-highwater check (`if hw is not None and by_num and max(by_num) > hw`, ~L1546) reads
  that census, so a numbered DIRECTORY above the counter is invisible.
- `numbering_drift_sweep.py` has its own copy of the same census (`audit()` ~L305,
  `head_audit()` ~L466) — same blind spot, HEAD side included (head_listing's direct-child
  projection already yields directory names — `.benchmarks/806_x/fig.png` contributes
  `806_x` per its own docstring — they are just filtered out by NUMBERED).

Family-wide-mechanism law applies: the repair lands in both instruments in one change, not
one copy now and the twin later.

## The fix (scoped by measurement, not by guesswork)

A 34-repo census of numbered non-md entries ran this session before any code (the shape of
each decision below is measured, with the false-positive repo named):

1. **Add the DIRECTORY class to the allocator-staleness census** (direct children matching
   `^\d+_` that are directories): worktree side via `e.is_dir()`, HEAD side via the
   ls-tree projection (a first component that always appears with a trailing path is a
   directory). The duplicate classifier stays `.md`-only — a legitimate `040_doc.md` +
   `040_datadir/` pair is not a COLLISION, and folding dirs into `classify_stems` would
   manufacture findings.
2. **Exclude `.docs` from the dir-class census.** Measured false-positive: riir-instinct
   `.docs` carries `05_resources/` (the docs book's numbered-folder sort scheme) against a
   `.docs/.highwater` of 1 — folder numbers are sort order, not allocations. Every repo's
   `.docs/` book uses the same scheme, so the exclusion is structural, not per-repo.
3. **The non-md FILE class is deliberately OUT.** Measured holders: riir-ai
   `.plans/932_10k10_sb.sh` (hw 662), riir-refine `.benchmarks/234_arm_s_4090_ledger.jsonl`
   (hw 169) — plan-number-tagged working artifacts whose numbers ride an owning plan, not
   local allocations; folding them in manufactures findings. Filed here as a disclosed
   non-goal; if that class ever needs a gate it is its own adjudicated campaign.
4. **The one live dir-class finding adjudicates with the landing.** reflex
   `.benchmarks/617_openthai_exl3_4090/` (617 > hw 208) is a RAW RESULTS HOME named for the
   owning plan (riir-infer Plan 617 A5), self-documented in its own landing commit
   `3cad008`: "results dirs are raw homes named for the owning plan, not allocations". It
   pins (membership + reason), it does not bump — the number was never a reflex allocation.

## Acceptance

- [ ] `numbering_gate.py`: dir-class census in `scan()` (worktree) with `.docs` excluded;
      above-highwater check over the md ∪ dir union naming the holder; pin mechanism for
      known-raw-home rows (membership + reason, both directions like every pin file);
      `--self-test` arms: dir above hw → finding, dir at/below hw → clean, pinned dir →
      clean, `.docs` dir → never a finding, non-md file → never a finding.
- [ ] `numbering_drift_sweep.py`: the same census in `audit()` (worktree) AND `head_audit()`
      (HEAD, via the trailing-slash projection); floors/pins re-measured fleet-wide.
- [ ] Fleet re-run: gate + sweep over all 34 contract repos — 0 stale findings beyond the
      one pinned reflex row.
- [ ] The incident replays as a positive test: a planted `NNN_dir` above a stale counter is
      CAUGHT (the reflex 208 shape), not green.
- [ ] Reflex `617_openthai_exl3_4090` pinned with the `3cad008` reason.

## Provenance

- Incident: reflex `1bbedde` (un-bumped 208 dir) + `ec75b04` (the manual late bump).
- 34-repo measurement: this session, one-off script (removed at close; the selftests carry
  the coverage).
- Raw-home convention: reflex `3cad008` (bench 110 openthai EXL3 board).
- Sweep's own docstring already promises the direct-child projection keeps directory
  names ("the directory `806_x` must not vanish from the count") — the census just never
  used them.
