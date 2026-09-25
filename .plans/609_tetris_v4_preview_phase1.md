# Plan 609 — Tetris next-piece preview, Phase 1 (Proposal 015; `laya-tetris-v4`)

Status: **READY — Phase 1 UNBLOCKED 2026-09-25 (T1.1 gate OPEN: 885's oracle + v3 fixture landed); owner confirmed Option B sequencing**
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Proposal: [015 — Tetris next-piece preview](../.proposals/015_tetris_next_piece_preview_lane.md) — Claude verdict AGREE (round 3, session `7cd12b60`); this plan expands its Phase-1 sketch
Depends: [Issue 885](../.issues/885_laya_tetris_v3_real_hard_drop_lane.md) — katgpt-rs-side items ONLY (oracle + v3 fixture + `_meta` parity read); the reflex/reflex-site consumer tail of 885 is NOT a precondition for this plan (v4's own consumer pass is Phase 2, separate lanes)

## Scope

katgpt-rs ONLY: the paired corpus, the v4 grammar, the two-arm oracle campaign, the go/no-go, the fixture + pins, and the katgpt-core fitter's board-grouped holdout path. Consumer work (riir-reflex serving + head integration, reflex-site render/wasm/demo) is Phase 2 and files as its own issues in those repos — hand-off notes at the end of this plan, not tasks here.

## Verified start state (2026-09-25)

- 885: `DropRule {DeepestFit(v2), FromTop(v3)}` + enumerator landed at `1a05a9764`; `tetris_01_state_enum --grammar/--join/--carry-from` proven; site JS follows the fixture's grammar (`fixture_rule`); **v3 oracle + fixture NOT landed** (only `tetris_oracle_laya_en_v2.jsonl` in `tests/fixtures/`); reflex-side serve/fixture_pins sits uncommitted in that repo's working tree (another session's lane — do not touch).
- v2 pinned surface untouched throughout: fixture BLAKE3 `f32c8577…`, λ=1 anchors, all tables (Issue 878 rule).
- Proposal 015's verified facts carry into this plan: the v2 oracle is board-blind (`laya_oracle_batch.rs` forwards the option sentence only); `loo_select`'s LOO unit is the state (leaks on a paired corpus — `game_heads.rs` L786-799).

## Tasks

### Phase 1 — katgpt-rs (this plan's execution surface)

- [x] T1.1 **Precondition gate (blocking):** Issue 885's oracle step + `tests/fixtures/tetris_oracle_laya_en_v3.jsonl` + the `_meta` numerics-parity read are LANDED (check the file exists + the 885 checklist flipped). Record the v3 fixture digest in this plan's completion note — it is v4's drift baseline (the parity read is quoted, never re-derived).
  - DONE 2026-09-25: fixture landed; **v3 fixture BLAKE3 `12035ebf43d0293c7ec00e716e72ee6a21686cc41a222938a81d0abd9316e804`, sha256 `eb67bc16c2c6b3732e7a2797d5344a22c3025ea5d3ade1b9e5f87f60da202bb3` — v4's drift baseline.** Parity read (quoted): 0/99 bit-exact, max |Δp| = 4e-6 (numerics moved + posture CPU→Metal); drift check PASS 120/2660; arena agreement unchanged vs v2 (13/120, ties 33→35); oracle 102 forwards / 2.2 s Metal. Provenance: `tests/fixtures/tetris_oracle_v3_README.md`.
- [ ] T1.2 `examples/common/tetris_sim.rs`: paired next-piece authoring — for each of the 120 v3 boards, all 7 next pieces (840 states); `render_state_sentence` gains the v4 preview sentence ("The next piece is the {TETRIS_PIECE} piece." appended to both state templates, spread + flat). Next-piece assignment is the seeded-bag walk (deterministic, disclosed in `_meta`).
- [ ] T1.3 `examples/common/grammar_tables.rs`: the `laya-tetris-v4` state grammar (5th slot reusing `TETRIS_PIECE`; option grammar UNCHANGED) + `verify_all_closed()` + the v4 decoders. Spot sentences stay byte-compatible with v3's renders (the preview lives in the state line only).
- [ ] T1.4 `tetris_01_state_enum`: the v4 dump mode — two-arm manifest. **Arm A:** 120 boards, preview masked (two-line envelope, no preview line) + ONE duplicated board as the determinism check. **Arm B:** the paired ~840 states, preview line present. `_meta` discloses: bag policy, envelope arms, inherited drop rule (FromTop), baseline fixture digest.
- [ ] T1.5 katgpt-core `state_option_scoring::head`: **board-grouped LOO path** — group id = board (all 7 preview states of a board hold out together) for λ selection and every held-out metric. The state-level path is UNCHANGED and pinned byte-identical for v2/v3 corpora (G3 test: existing head digests `7d3f1d8e…`/`c93d36dc…`/`00aa6221…` reproduce exactly).
- [ ] T1.6 Oracle campaigns via riir-reflex `laya_oracle_batch` (M3 Metal; envelope extended to accept the two-line payload — coordinate with that repo's fixture_pins lane BEFORE touching the example): Arm A (~2.7k forwards) then Arm B (~18.6k). Cost measured and recorded BEFORE the consumer-facing commit. Arm A labels land under `.benchmarks/` with a **BLAKE3 pin** (attribution-drift control; three copies not needed).
- [ ] T1.7 **Go/no-go (gates ALL consumer work):** the fraction of boards where the Arm B oracle's argmax CHANGES across the 7 previews. ≈ 0 → the preview ships in the sentence only (fidelity), the crossed-head/consumer/G1 work STOPS, the negative is recorded in the bench record + proposal 015. The raw MI readout is NOT decision-reliable (picks up calibration shifts) — the flip fraction is.
- [ ] T1.8 v4 fixture + digest pins: `tests/fixtures/tetris_oracle_laya_en_v4.jsonl` + BLAKE3/sha256 in the fixture README; the katgpt-rs copy pinned in the same commit as the fixture (the three-copy cross-repo hashing happens at Phase 2 — 885's lesson: never length-only).
- [ ] T1.9 Baseline + candidate head fits, recorded: the **spot-only head on the same Arm B labels** (the G1 preview-blind comparator — ranks all 7 variants of a board identically by construction) and the **crossed head** (one-hot 6 × spot features, ~40 columns); board-grouped metrics only (argmax agreement per state over ~840 states with board-grouped folds, pairwise ranking within board, board-centered MSE). Numbers land in the bench record regardless of verdict.
- [ ] T1.10 Bench record `.benchmarks/NNN_tetris_v4_preview_phase1/` (highwater bump) + proposal 015 status update + this plan's checkboxes.

### Phase 2 — consumers (hand-offs; separate issues in those repos, NOT this plan)

- [ ] riir-reflex: two-line tetris serving, crossed-head integration behind a config knob (default off = byte-identical), G1 within-board gate, live-lane parity test (Rust + python laya lanes receive the identical two-line payload), `fixture_pins()` four-hash (v2/v3/v4/flappy-v3), wasm head regen. **Must NOT touch the `Embedder`** (issue 030's blast-radius law).
- [ ] reflex-site: next-piece render + UI box, re-recorded walks, wasm regen, golden sha256 pins, deploy.
- [ ] Arena republish: batch with Issue 030 lever 4's pending clean-window rerun if timing overlaps (the publisher refuses drifted modelless accuracy — one merged host).

### Explicitly out of scope (per Proposal 015)

Planner lane / multi-piece lookahead; hold piece; SRS/wall-kick; flappy/lanes grammars; any score-side narrative.

## GOAT gates (bind the Phase-2 promotion — mirrored from Proposal 015)

- **G1:** crossed head beats the spot-only same-labels comparator on within-board metrics (argmax per state / pairwise ranking / board-centered MSE) under the board-grouped holdout. Pooled MSE is not a gate. Arm-A head is disclosure, never the comparator.
- **G2:** serving stays µs-tier (~40-column head; two-line state).
- **G3:** v2 AND v3 lanes byte-identical end to end — including the katgpt-core fitter's state-level path (T1.5's digest pins).
- **G4:** zero-alloc serving law holds.
- **Abort path:** T1.7 ≈ 0 flips → sentence-only fidelity, no head promotion, negative recorded (the honest outcome is a valid landing).

## Risks

1. **Campaign cost:** Arm A + Arm B ≈ 21.3k laya forwards on M3 Metal — measured at T1.6 before any consumer-facing landing; run in a preflight-clean window (the harness box-state law applies to any quoted latency; oracle labels are accuracy claims and box-independent).
2. **Coordination:** Issue 885's consumer tail is LIVE in riir-reflex's working tree (another session) — T1.6's envelope extension to `laya_oracle_batch` coordinates with it before editing; never `git checkout .` over that tree (the recovery incident in that repo's bench 040 record is the precedent).
3. **Numbering:** v4 touches the same fixture directory as 885's pending v3 fixture — sequence T1.8 strictly after T1.1; dual-allocation gate re-run at plan execution (it ran clean at filing: 0 twin, 0 independent).

## Completion

Flip this plan's `- [ ]` rows as they land; ref commit hashes per task in the proposal 015 verdict trail + the bench record. The plan closes when T1.1–T1.10 are checked and the Phase-2 hand-off issues exist in their repos (their execution is those repos' lanes, not this plan's).
