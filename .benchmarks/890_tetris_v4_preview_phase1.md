# Bench 890 — Plan 609 Phase 1: the Tetris v4 next-piece preview (go/no-go + the paired-corpus head fits)

**Status: Phase 1 COMPLETE — T1.7 flip fraction 62.5% (signal present, not the abort path); G1 FAIL (the crossed head does not beat the preview-blind comparator) → no crossed-head promotion; the v4 grammar + fixture land as fidelity substrate.**

Proposal: [015 — Tetris next-piece preview](../.proposals/015_tetris_next_piece_preview_lane.md) · Plan: [609](../.plans/609_tetris_v4_preview_phase1.md) · Fixture: [tetris_oracle_laya_en_v4.jsonl](../tests/fixtures/tetris_oracle_laya_en_v4.jsonl) ([README](../tests/fixtures/tetris_oracle_v4_README.md)).

## The campaign (T1.6 — cost measured before any consumer landing)

M3 Metal, `english` checkpoint, no concurrent GPU compute; 21,297 two-line
forwards in ~11 min wall (Arm A 2,677 / 75.5 s; Arm B 18,620 / 589.1 s).
Oracle blake3s: Arm A `51728cdf…`, Arm B `e0ada1e1…` (full digests in the
fixture README + `_meta`). The dup determinism check PASSED (bit-exact).
Arm-A labels: [890_arm_a_oracle.jsonl](890_arm_a_oracle.jsonl) (blake3
`51728cdf3e8902c5688982ba7d7a0411d9cc0b65ba8c365c8d58019e97ca941c`).

## The envelope finding (attribution control — recorded before the go/no-go)

Arm A (masked state line, no preview) vs the v3 option-only labels: **argmax
24/120, 0/2660 p_clean bit-exact, max |Δp| 0.713.** The state line alone is a
first-order oracle input — v2/v3's option-only labels measured a
context-blind model (Proposal 015's board-blind premise, now quantified).
The go/no-go below is therefore the WITHIN-arm-B comparison (state line
constant per board, only the preview sentence varies) — the correct
isolation.

## T1.7 — the go/no-go flip fraction

**75/120 boards (62.5%)** have an arm-B oracle argmax that changes across the
7 previews; 660/840 states (78.6%) sit off the preview-blind (v3) pick.
Decidedly NOT ≈ 0 → the abort path does NOT fire: the preview is a real
input to the oracle, not sentence-only fidelity. Structure is not uniform
noise: e.g. `arch:floor_low:I` holds ONE pick (option 16) under six previews
and moves only when the next piece is `O` (option 3); `arch:empty:O` holds
option 0 under five previews and moves for `O`/`L` — duplicate-piece and
specific-successor effects, not independent shuffles.

## T1.9 — the head fits (board-grouped holdout, katgpt-core `head::loo_group_select`)

Both heads fit the SAME arm-B labels (18,620 rows / 840 states); the hold-out
unit is the BOARD (all 7 preview states together). λ selected by grouped-LOO
MSE over the frozen grid; all readings at the chosen λ under the grouped
holdout.

| head | design | λ | argmax agree | distinct | pairwise concord | board-MSE | board-flips | head blake3 |
|---|---|---|---|---|---|---|---|---|
| spot-only (comparator) | 5 spot fills + intercept | 1 | **383/840 (45.6%)** | 24 | **171,722/221,662 (77.5%)** | 0.00740 | 0 (by construction — asserted) | `295c107f…315f0` |
| crossed | spot(5) + one-hot next(6) + one-hot×spot(30) + intercept | 1 | 332/840 (39.5%) | 26 | 169,057/221,662 (76.3%) | **0.00682** | 80 | `bbd1bb23…a041814` |

Baselines: constant-pick 174/840 (20.7%) · mean 1/K chance 5.5%.

## G1 verdict — FAIL (no crossed-head promotion)

The crossed head wins ONLY board-centered MSE (it tracks the preview-dependent
level shifts) and LOSES the two decision metrics: argmax agreement 332 vs 383,
pairwise concordance 76.3% vs 77.5%. Pooled/board-centered MSE is explicitly
not the gate — the decision metrics are. **The preview-dependent part of the
oracle's behavior does not transfer across boards through a ~40-column linear
crossed head over the decoded spot bands; the spot-only head at 45.6% stays
the best linear decision rule on this corpus.** Per the plan's GOAT gates the
Phase-2 crossed-head integration (config knob, within-board serving gate, G2/G4
serving) does NOT proceed on this evidence.

The honest landing: the preview grammar + the paired fixture are the
substrate (the render is Phase-2-fidelity work for the consumer repos); any
future promotion candidate needs a different capture of the interaction
(e.g. successor-conditional features rather than raw one-hot products) — a NEW
proposal, not a re-run of this design.

## The G3 finding (T1.5's byte-identical check — a real catch)

`decode_01_losslessness` was RED at the parent commit: the tetris structured
head digest drifted from the Bench 878 anchor (`65409c14…` → `b3c91ee0…`).
Root cause MEASURED (not reasoned): the Issue-884 commit (`1a05a9764`) added
the dev-dep `serde_json/float_roundtrip` (exact float parsing — simply
correct), which changed every example's parsed `p_clean` targets by ~1 ULP,
and the tetris structured fit is the one whose weight bytes moved. The
agreement numbers never moved (36/36 in-corpus · 35/35 LOO — the digest is
byte-level sensitive, the decisions are not). The anchor is re-pinned in
`decode_01_losslessness.rs` with the full cause recorded; the flappy v2/v3 +
lanes anchors MEASURED identical under both parsers. Old digest = the
parser-bug-era record; new digest = the exact-parse anchor. Cross-repo
implication for the fixture_pins lane: a consumer fitting from these fixtures
must parse with `float_roundtrip` to land on the same head bytes (reflex's
copies are self-consistent under their own parser — their pins are their
lane).

## Validation

- `cargo test -p katgpt-core --lib --features state_option_scoring` — 3 new grouped-LOO tests PASS (complement-refit equality, λ-tie first-wins, group-bound refusal); full katgpt-core lib suite green.
- `cargo test --example tetris_01_state_enum` — 11 PASS (incl. the v4 preview-render test).
- `cargo test --example decode_01_losslessness` — 47 PASS (incl. the v4 grammar round-trip + archetype drift detector).
- `decode_01_losslessness` main — closed-space 7/7, all anchors green under exact parsing.
- `tetris_04_preview_fit` — the readings above; clippy clean on all touched examples + katgpt-core.
