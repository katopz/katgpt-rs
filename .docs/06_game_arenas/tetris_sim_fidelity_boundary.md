# Tetris sim fidelity boundary: a placement-enumeration oracle, not a Tetris engine

**Status:** RECORD (a standing boundary rule). The decision is closed: no swap. Moved
here from Issue 878 on 2026-09-25 when the issue file was removed. The rule is still
live, and Issue 885 (`laya-tetris-v3`) and Plan 609 (`laya-tetris-v4`) both followed it.

## The question (owner, 2026-09-24)

"Did we use tetris from original code? Don't write it yourself — better use original for
bit-exact, and/or search for a real tetris (Rust or JS) and convert; currently looks buggy
and no tests?"

## What the sim is (grep-verified, not remembered)

The sim is `crates/katgpt-tetris/src/sim.rs`. Issue 893 moved it there byte-identical
from `examples/common/tetris_sim.rs`. The file contains zero occurrences of `bag`,
`srs`, `wall kick`, `gravity` or `hold`. Its public surface is `hard_drop`,
`landing_options`, `outcome_features`, `dellacherie_score`, `render_spot_sentence` and
`render_state_sentence`, over a 10×20 board and the 7 tetrominoes. `landing_options_with`
and `DropRule` were added in Issue 885.

It is a **hard-drop placement-enumeration oracle that happens to use tetromino
geometry**. It was built to feed the laya sentence protocol: enumerate the landing spots,
describe each one in words, and ask P(clean). No shipping Tetris exposes that unit of
work, so "bit-exact vs original" has nothing to compare against. The head also never
observes engine internals, only sentences.

The piece stream was the one guideline-fidelity gap that affects how play feels. It was
closed site-side for LIVE play on 2026-09-24: reflex-site `1cc76de` deals live pieces
from a guideline 7-bag, and recorded demo replays are untouched.

## The pins any "swap" would invalidate (check the digests, don't trust this file)

| Artifact | Value |
|---|---|
| v2 oracle fixture BLAKE3 | `f32c8577bca50726618d2bb4fb27c904148161d650f16a59c01676a97fa540bb` (pinned in `../riir-reflex/src/game_heads.rs`, "the cross-repo data contract") |
| v2 oracle fixture sha256 | `12c46035a7f6b5416c4ce34c86bc9b0b890975aff57944739a24c5869b8fc259` |
| v2 fixture shape | 120 states / 2660 options, grammar `laya-tetris-v2` |
| v3 fixture | `tests/fixtures/tetris_oracle_laya_en_v3.jsonl`, blake3 `12035ebf…` / sha256 `eb67bc16…` (Issue 885, `6a35cde32`; see `tests/fixtures/tetris_oracle_v3_README.md`) |
| Byte-identical copies | THIS repo `tests/fixtures/` · `../riir-reflex/assets/game_heads/` · `gist-rs/reflex-site` `tests/fixtures/` (sha256-pinned by its golden test) |
| Downstream | riir-reflex fitted tetris head (boot-fitted; now the v3 refit, riir-reflex `ca1483c`) · reflex-site wasm head artifact + recorded demo walks · the bench page tables |

Swapping the substrate means all of this: regenerate the fixture here, then re-fit the
head, bump the digests and re-run the serve tests (riir-reflex), then rebuild the wasm
head, re-record the demo and republish the tables (reflex-site). Every published
agreement number resets, and the head cannot see the change.

## Verdict (owner-gated; Claude verdict AGREE 2026-09-24)

**Do NOT swap.** The bit-exactness that matters already exists and is enforced. The JS
port reproduces the Rust sim **2660/2660 options byte-identically**
(reflex-site `assets/games/tetris_golden.test.mjs`, self-contained since `1cc76de`).

The "looks buggy" report was about the RENDER layer, and it was fixed the same day,
site-side, using the original Tetris piece palette. Three render defects were involved:

- green marked the chosen spot, not a falling piece;
- orange candidate ghosts muddied the board;
- the overlay went stale after a piece landed.

## The new-lane rule (load-bearing)

**Guideline fidelity, if ever wanted (SRS kicks, gravity, hold, preview), ships as a NEW
lane with its OWN grammar, fixture and head. Never mutate a pinned corpus.** It has been
applied twice:

- `laya-tetris-v3` implements the real hard drop (Issue 885, record in HISTORY.md).
  `DropRule::FromTop` is the v3 rule; `DeepestFit` remains the v2 default.
- `laya-tetris-v4` adds the next-piece preview (Proposal 015 / Plan 609, Bench 890).

## The "no tests" half: the class, recorded

The golden test existed but only ran with a sibling checkout or the `TETRIS_FIXTURE` env
set, so its status was **unknown, not passing**. Self-containment is what converted it
into evidence: the fixtures were copied into reflex-site `tests/fixtures/`, sha256-pinned
and excluded from the deploy via `.assetsignore`. The result was 16/16 site tests,
120/120 states and 2660/2660 options.

## Comparability caveat

Live arena play deals from a 7-bag stream, while the pinned corpus has NO randomizer: its
states are fixture rows. So live-play agreement is a DIFFERENT distribution from the
in-corpus anchor count (44/120 on v2, 42/120 on the v3 refit). The arena explanation has
carried this caveat since reflex-site `e9973c0`.

## Second fidelity gap: roof tunnelling (Issue 884 → 885)

v2's `hard_drop` rests a piece at the deepest collision-free row, so a piece can tunnel
through a roof (3 of 2660 v2 options). Path A (2026-09-25) filtered tunnelled spots in
live play (reflex-site `c1fcf88`) and left the corpus untouched. Path B shipped the fix as
the new v3 lane (Issue 885). Records: HISTORY.md § Issue 884, § Issue 885.
