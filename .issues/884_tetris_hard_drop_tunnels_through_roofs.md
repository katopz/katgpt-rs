# Issue 884 — tetris_sim `hard_drop` tunnels through roofs (scans bottom-up)

**Status:** OPEN — finding recorded; fix path owner-gated (recommendation: site-side live-play filter, corpus untouched per Issue 878).

## Trigger (owner, 2026-09-25)

Arena screenshot (reflex.gist.rs/arena): "why partial 1 block not fall to the bottom? bug?"

## Two answers — one is Tetris, one is a bug

1. **The floating single block is standard Tetris ("naive gravity"), NOT a bug.**
   `clearRows` shifts every row above a cleared row down by exactly one per cleared row;
   individual cells never cascade into gaps. A vertical piece whose lower cells sat in
   cleared rows leaves its top cell hanging over whatever hole was already under it. Every
   guideline Tetris behaves this way (cascade gravity is a variant, e.g. Puyo-style).

2. **A real defect found while checking: `hard_drop` picks the DEEPEST collision-free row**
   (`for row in (0..HEIGHT).rev()` — `examples/common/tetris_sim.rs:249`, JS port
   reflex-site `assets/games/tetris.js` `hardDrop`). A real hard drop descends from the top
   and stops at the first collision. So a piece passes through a solid roof and lands in
   the cave underneath.
   - Probe (JS port): roof at row 17 cols 0..3, `O` at col 0 → lands rows **18–19**
     (under the roof); a real hard drop lands rows 15–16.
   - Pinned corpus exposure: **3 of 2660 options** (3 of 120 states) in
     `tetris_oracle_laya_en_v2.jsonl` are tunnelled placements — measured by re-dropping
     each option from row 0 and comparing rest rows.

## Fix paths

| Path | Cost | Verdict |
|---|---|---|
| A. Site-side: live arena play filters options not reachable from the top (golden parity untouched — golden runs `landingOptions` on fixture boards) | small, reflex-site only | **recommended** |
| B. Fix `hard_drop` here + regen fixture v3 → re-fit riir-reflex head, bump `TETRIS_FIXTURE_BLAKE3`, rebuild wasm head, re-record demo walk, republish tables | full chain (Issue 878 table) | only as a NEW lane per 878's rule — never mutate v2 |

- [ ] Owner call A vs B
- [ ] (A) reachable-from-top filter in reflex-site live play + unit test (the O-under-roof probe)
- [ ] (A) note on arena explain that the recorded demo/corpus keep v2 semantics (3/2660 tunnelled)
