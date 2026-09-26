# Plan 609 — Tetris next-piece preview, Phase 1 (Proposal 015; `laya-tetris-v4`)

Status: **Phase 1 EXECUTED 2026-09-25 (Bench 890): T1.7 flip 75/120 (62.5% — signal present, abort path NOT fired); G1 FAIL (crossed head 332/840 vs spot-only 383/840 argmax under the board-grouped holdout) → no crossed-head promotion; grammar+fixture landed; Phase-2 crossed-head integration CLOSED on this evidence (consumer render/fidelity is the consumers' lane)**
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Proposal: [015 — Tetris next-piece preview](../.proposals/015_tetris_next_piece_preview_lane.md) — Claude verdict AGREE (round 3, session `7cd12b60`); this plan expands its Phase-1 sketch
Depends: Issue 885 (closed 2026-09-25, HISTORY.md § Issue 885) — katgpt-rs-side items ONLY (oracle + v3 fixture + `_meta` parity read); the reflex/reflex-site consumer tail of 885 is NOT a precondition for this plan (v4's own consumer pass is Phase 2, separate lanes)

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
- [x] T1.2 `examples/common/tetris_sim.rs`: paired next-piece authoring — for each of the 120 v3 boards, all 7 next pieces (840 states); `render_state_sentence` gains the v4 preview sentence ("The next piece is the {TETRIS_PIECE} piece." appended to both state templates, spread + flat). Next-piece assignment is the seeded-bag walk (deterministic, disclosed in `_meta`).
  - DONE 2026-09-25: `render_state_sentence_with_preview` composes the v3 render + the preview sentence (v2/v3 pins untouched by construction, test-pinned); bag = one full 7-piece bag per board (fastrand, the dump seed's independent stream), disclosed in `_meta.bag_policy`.
- [x] T1.3 `examples/common/grammar_tables.rs`: the `laya-tetris-v4` state grammar (5th slot reusing `TETRIS_PIECE`; option grammar UNCHANGED) + `verify_all_closed()` + the v4 decoders. Spot sentences stay byte-compatible with v3's renders (the preview lives in the state line only).
  - DONE: `tetris_state_v4()` (spread/flat + preview tail), `decode_tetris_state_v4` + `tetris_state_forward_v4`, verify_all_closed 7 tables; round-trip + archetype drift tests PASS.
- [x] T1.4 `tetris_01_state_enum`: the v4 dump mode — two-arm manifest. **Arm A:** 120 boards, preview masked (two-line envelope, no preview line) + ONE duplicated board as the determinism check. **Arm B:** the paired ~840 states, preview line present. `_meta` discloses: bag policy, envelope arms, inherited drop rule (FromTop), baseline fixture digest.
  - DONE: `--grammar laya-tetris-v4` emits the full arm-B dump + `states_arm_a.jsonl` (121 recs, blake3 `5c36fccd…`) + `states_arm_b.jsonl` (840, `e8044925…`); `V4FixtureMeta` carries bag_policy/envelope/baseline_blake3 (verified `12035ebf…` at join time); dup determinism PASSED bit-exact.
- [x] T1.5 katgpt-core `state_option_scoring::head`: **board-grouped LOO path** — group id = board (all 7 preview states of a board hold out together) for λ selection and every held-out metric. The state-level path is UNCHANGED and pinned byte-identical for v2/v3 corpora (G3 test: existing head digests `7d3f1d8e…`/`c93d36dc…`/`00aa6221…` reproduce exactly).
  - DONE: `head::loo_group_select` (+ 3 unit tests: complement-refit equality, λ-tie first-wins, group-bound refusal); state-level path untouched. ⛔ G3 MEASURED A REAL CATCH: decode_01 was RED at the parent — the Issue-884 dev-dep `serde_json/float_roundtrip` (exact parsing) moved the tetris structured head bytes (`65409c14…` → `b3c91ee0…`); agreement numbers never moved (36/35); anchor re-pinned with the cause recorded; flappy v2/v3 + lanes anchors measured identical under both parsers (`00aa6221…` is reflex's own serving serialization — their lane, noted for fixture_pins).
- [x] T1.6 Oracle campaigns via riir-reflex `laya_oracle_batch` (M3 Metal; envelope extended to accept the two-line payload — coordinate with that repo's fixture_pins lane BEFORE touching the example): Arm A (~2.7k forwards) then Arm B (~18.6k). Cost measured and recorded BEFORE the consumer-facing commit. Arm A labels land under `.benchmarks/` with a **BLAKE3 pin** (attribution-drift control; three copies not needed).
  - DONE: envelope = optional `state_sentence` per manifest record (backward-compatible by construction: absent → the identical option-only forward); `examples/laya_oracle_batch.rs` is OUTSIDE the fixture_pins lane's two dirty files (verified — no trample, no stash). Campaign: 21,297 forwards / ~11 min Metal (A 75.5 s, B 589.1 s); Arm A labels at `.benchmarks/890_arm_a_oracle.jsonl`.
- [x] T1.7 **Go/no-go (gates ALL consumer work):** the fraction of boards where the Arm B oracle's argmax CHANGES across the 7 previews. ≈ 0 → the preview ships in the sentence only (fidelity), the crossed-head/consumer/G1 work STOPS, the negative is recorded in the bench record + proposal 015. The raw MI readout is NOT decision-reliable (picks up calibration shifts) — the flip fraction is.
  - MEASURED: **75/120 (62.5%)** — NOT ≈ 0; the abort path does NOT fire; structured movement (duplicate-piece + specific-successor effects). Proceeded to T1.9 per plan.
- [x] T1.8 v4 fixture + digest pins: `tests/fixtures/tetris_oracle_laya_en_v4.jsonl` + BLAKE3/sha256 in the fixture README; the katgpt-rs copy pinned in the same commit as the fixture (the three-copy cross-repo hashing happens at Phase 2 — 885's lesson: never length-only).
  - DONE: fixture blake3 `18e6b260…`, sha256 `caee3293…`; `tests/fixtures/tetris_oracle_v4_README.md` (full provenance incl. both arms + bag policy).
- [x] T1.9 Baseline + candidate head fits, recorded: the **spot-only head on the same Arm B labels** (the G1 preview-blind comparator — ranks all 7 variants of a board identically by construction) and the **crossed head** (one-hot 6 × spot features, ~40 columns); board-grouped metrics only (argmax agreement per state over ~840 states with board-grouped folds, pairwise ranking within board, board-centered MSE). Numbers land in the bench record regardless of verdict.
  - MEASURED (Bench 890): spot-only λ=1 383/840 argmax · 77.5% concord · board-MSE 0.00740 · board-flips 0 (asserted); crossed λ=1 332/840 · 76.3% · 0.00682 · 80. Head digests in the record.
- [x] T1.10 Bench record `.benchmarks/NNN_tetris_v4_preview_phase1/` (highwater bump) + proposal 015 status update + this plan's checkboxes.
  - DONE: `.benchmarks/890_tetris_v4_preview_phase1.md` (+ `890_arm_a_oracle.jsonl`); highwater 889→890; proposal 015 updated; **G1 FAIL recorded — no crossed-head promotion**.

### Phase 2 — consumers (hand-offs; separate issues in those repos, NOT this plan)

**G1 verdict 2026-09-25: FAIL — the crossed-head integration below is CLOSED on this evidence. What remains consumer-side is the FIDELITY surface only: the next-piece render/UI (the v4 sentence) and, if wanted, the two-line serving path (the envelope is proven; the head behind it is not promoted). The fixture_pins four-hash pin should include the v4 fixture digest.**

- [x] riir-reflex (FIDELITY ONLY — no crossed head): optional two-line tetris serving path + `fixture_pins()` four-hash (v2/v3/v4/flappy-v3); parser note: fitting from these fixtures requires `serde_json/float_roundtrip` to land on the katgpt-rs-side head bytes (Bench 890 G3 finding). **Must NOT touch the `Embedder`** (issue 030's blast-radius law). The crossed-head config knob + G1 within-board gate are DROPPED (G1 failed).
  - DONE (riir-reflex Issue 031, `a56d850`): `fixture_pins()` hashes every embedded fixture against its pin; the unserved tetris v3/v4 fixtures are pinned test-side (`12035ebf…` / `18e6b260…`, missing file FAILS); the `serde_json/float_roundtrip` note is in the `game_heads` module doc. The optional two-line serving path was NOT built: with the crossed head refused, it would serve the spot head, which cannot consume the preview.
- [-] reflex-site: next-piece render + UI box (fidelity) — re-recorded walks + wasm regen only if the serve path changes; golden sha256 pins; deploy.
  - DEFERRED 2026-09-26 (Claude verdict): riir-reflex serves v3 only and v4 is unserved by design, so the serve path is unchanged and neither trigger fires (no walk re-record, no wasm regen). A next-piece UI box would show the player a preview that no served head reads, which misrepresents what the modelless lane decides on. Reopen if a head that consumes the preview is ever promoted (a new G1 PASS on the board-grouped holdout).
- [x] Arena republish: batch with Issue 030 lever 4's pending clean-window rerun if timing overlaps (the publisher refuses drifted modelless accuracy — one merged host).
  - DONE (riir-reflex Bench 041, `ae80abc`): the table was republished after Issue 030 lever 4 (Bench 040), with the m3 laya lanes and both hosts' modelless rows in one run. No v4 row, since v4 is not served.

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

**CLOSED 2026-09-25 — Phase 1 executed in full (T1.1–T1.10, Bench 890). Outcome: the preview is a real oracle input (flip 62.5%) but the linear crossed head fails G1 (332 vs 383 argmax under the board-grouped holdout) — no promotion; the v4 grammar + fixture land; Phase-2 consumer work narrows to the fidelity surface.**
