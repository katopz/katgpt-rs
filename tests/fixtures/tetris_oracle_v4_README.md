# Tetris laya oracle fixture v4 (katgpt-rs Plan 609 T1.4/T1.8 — the next-piece preview lane)

The **paired** corpus: for each of the 120 v3 states (board + falling piece,
`FromTop` drop rule unchanged), all **7 next pieces** — 840 states, 18,620
options, same pinned option order and question/checkpoint as v2/v3. The ONLY
sentence delta vs the v3 parent is the appended preview line:

> `… The {piece} piece is falling. The next piece is the {next} piece.`

Option sentences stay byte-compatible with v3's renders (the preview lives in
the state line only — asserted per state by the join: every parent's option
set is byte-identical to the v3 baseline, 840/120 parents verified).

## The two-arm envelope (plan 609 T1.4/T1.6)

The oracle forwards a **two-line payload**: the state line + the option line,
joined by the oracle tool's optional `state_sentence` field
(riir-reflex `laya_oracle_batch`). Two arms:

| arm | state line | records | forwards | oracle blake3 |
|---|---|---|---|---|
| A (masked) | the v3 state sentence (no preview) + one duplicated state (`arch:empty:I\|dup`, the determinism check — PASSED bit-exact) | 121 | 2,677 (75.5 s) | `51728cdf3e8902c5688982ba7d7a0411d9cc0b65ba8c365c8d58019e97ca941c` |
| B (preview) | the v4 state sentence (preview present) | 840 | 18,620 (589.1 s) | `e0ada1e1986413721e03a645a7631e643333ab4788e58ea3ce4d9714d6f5279c` |

## Provenance

| field | value |
|---|---|
| dump generator | `cargo run --release --example tetris_01_state_enum -- --grammar laya-tetris-v4` (katgpt-rs, seed 607) |
| dump blake3 (arm-B full dump, pre-join) | `052680b02bcfe5b9c5d2be2bbe46180ed0e64c27fda6dbcc31075a43b83f5fa0` |
| arm-A manifest blake3 | `5c36fccd082e01db5e477d41bb66af10aaffdb018305d712002329b7bb33c3f2` |
| arm-B manifest blake3 | `e80449252303e99f61cb04b5b06cf77b4e2359ed1d54a693ed19e0f78b8d95d8` |
| oracle generator | `riir-reflex examples/laya_oracle_batch @ develop (2026-09-25, the Plan-609 envelope: optional state_sentence)`, **M3 Metal posture**, no concurrent GPU compute |
| checkpoint | `english` |
| baseline (v3 fixture) blake3 | `12035ebf43d0293c7ec00e716e72ee6a21686cc41a222938a81d0abd9316e804` (the drift baseline — verified at join time) |
| forwards | 21,297 total (one `noul` forward per two-line payload) |
| fixture blake3 (this file) | `18e6b2604a2f01433a5f7d860b7c98009fa1f41ad75be5c35ee251717ac52903` |
| fixture sha256 | `caee3293674365be76977c512108ffe0c4fc5d805d96300ef1553f98f4bc664c` |
| bag policy | one full 7-piece bag per board in dump order (fastrand, the dump seed's independent stream); variants emitted in dealt order; all 7 next pieces per board |

## The arm-A attribution control (recorded — it is a finding)

Adding the state line to the forward ALONE (masked, no preview) moves laya's
answers substantially vs the v3 option-only labels: **argmax 24/120**,
0/2660 `p_clean` bit-exact, max |Δp| 0.713. The state context is a
first-order input to the oracle — the v2/v3 option-only labels measured a
context-blind model. Within arm B the state line is constant per board, so
the preview's incremental effect is measured against THIS baseline (see
Bench 890 for the flip fraction and the head fits).

## Regeneration

The dump side is fully deterministic (`tetris_01_state_enum --grammar
laya-tetris-v4`, seed 607 → byte-identical manifest blake3s). The oracle side
needs the laya weights (`~/.cache/riir-reflex/laya`, or `LAYA_WEIGHTS_DIR`/
`LAYA_HOME`) and the envelope-capable oracle tool; the full 21.3k-forward
re-run is the ONLY regeneration path (there is no carry: every forward's
payload differs from v3's). Copying the laya forward INTO katgpt-rs is
forbidden by the lane split (Plan 607) — the committed fixture is the drift
detector.
