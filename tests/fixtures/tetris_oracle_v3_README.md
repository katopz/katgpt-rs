# Tetris laya oracle fixture v3 (katgpt-rs Issue 885 — the real-hard-drop lane)

Same corpus shape as [v2](tetris_oracle_v2_README.md) (120 states, 2660 options,
same pinned option order, same question/checkpoint), generated under the
**`laya-tetris-v3`** grammar: drop rule `FromTop` (pieces fall onto the highest
occupied cell in their columns) instead of v2's `DeepestFit` (tunnel-through).
Per Issue 878's rule it ships as a NEW fixture; the pinned v2 fixture stays
committed and byte-identical.

## The measured v2 → v3 delta (why only 102 forwards)

Same 120 boards, same pieces, same option count — the seeded play ladder never
picked a tunnelled spot, so no play-state changed. Exactly **3 options move**
(one per affected state, all on the authored `holes` archetype, rot 3 col 8):

| state | moved option | v2 landing | v3 landing |
|---|---|---|---|
| `arch:holes:T` | option[33] rot 3 col 8 | row 15 (through the covered hole at (16,8)) | row 10 (on the plateau) |
| `arch:holes:J` | option[33] rot 3 col 8 | row 14 | row 9 |
| `arch:holes:L` | option[33] rot 3 col 8 | row 16 | row 11 |

Each moved option changes `row`/`cells`/`features`/`sentence` together; the
other 2657 options are byte-identical. The join therefore re-ran the oracle on
the **3 changed states only** (102 forwards) and carried the other 117 states
verbatim from v2 — allowed only because every carried option sentence is
byte-identical (asserted by the join tool; the oracle answer is a function of
the sentence). The 99 sentence-identical options inside the 3 fresh states are
the numerics-parity read, recorded in `_meta`.

## Provenance

| field | value |
|---|---|
| dump generator | `cargo run --release --example tetris_01_state_enum -- --grammar laya-tetris-v3` (katgpt-rs, seed 607) |
| dump blake3 (pre-join) | `6c181e96520131ceb2e177e4d1c51e21e6442be65743e9ce679eb7301ccc53f3` |
| oracle generator | `riir-reflex examples/laya_oracle_batch @ git 69a6eae` (origin/develop 2026-09-25; **M3 Metal posture** — v2 was CPU pre-Metal-lane) |
| checkpoint | `english` (ModernBERT-large, max_len 512) |
| oracle blake3 (raw decisions, 3-state manifest) | `252e0378e0b675199a4d81b8dfcc5837639b7f25dfddbd12e6c9100753520826` |
| forwards | 102 (one `noul` forward per option sentence of the 3 changed states, 2.2 s wall on Metal) |
| fixture blake3 (this file) | `12035ebf43d0293c7ec00e716e72ee6a21686cc41a222938a81d0abd9316e804` |
| fixture sha256 | `eb67bc16c2c6b3732e7a2797d5344a22c3025ea5d3ade1b9e5f87f60da202bb3` |
| question | `Does the stack look clean?` (world-anchored, never "what to do") |

Working-tree caveat: the reflex checkout carried unrelated uncommitted edits
(`src/game_heads.rs`, `tests/game_heads_serve.rs` — the v3-head serving lane)
when the oracle ran; neither is in the oracle's forward path (the laya lane).

## The parity read (honest, recorded — never asserted)

**0/99** sentence-identical options reproduce the carried v2 `p_clean`
bit-exactly, but the drift magnitude is negligible: **max |Δp| = 4e-6** across
all 99. Source: riir-reflex numerics moved 164 commits since the v2 generator
(`76f4b92`), plus the posture differs (v2 CPU, v3 Metal). The fixture's labels
stay self-consistent — each fresh state's 34 labels come from one forward pass.

## Oracle decision movement (the fix working as intended)

- `arch:holes:T`: argmax moved **off** the (formerly tunnelling) rot 3 col 8
  option → rot 0 col 1 row 10.
- `arch:holes:J`: same — off rot 3 col 8 → rot 0 col 1 row 10.
- `arch:holes:L`: stays at rot 3 col 8, which now lands on the plateau (row 11).

In v2 the argmax of T and J WAS the through-the-hole option; with `FromTop`
that option no longer exists (it lands harmlessly on the plateau instead) and
laya re-ranks. This is the lane's purpose: v3 sentences describe honest
landings.

## Regeneration

Identical shape to v2: the dump side is fully deterministic (`tetris_01_state_enum
--grammar laya-tetris-v3`, seed 607 → byte-identical blake3). The oracle side
needs the laya weights (`~/.cache/riir-reflex/laya`, or `LAYA_WEIGHTS_DIR`/
`LAYA_HOME`) and the pinned generator commit; the full-2660 re-run is NOT the
regeneration path (numerics drift would perturb the 117 carried states and
destroy the drop-fix isolation) — the join tool + `--carry-from` is. Copying
the laya forward INTO katgpt-rs is forbidden by the lane split (Plan 607) —
the committed fixture is the drift detector (the katgpt-device-verify rule).
