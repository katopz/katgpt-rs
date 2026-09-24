# Issue 878 — tetris_sim fidelity boundary: a placement-enumeration oracle, NOT a Tetris engine

**Status:** RECORD — boundary documentation; decision CLOSED (no swap). Re-open only via the new-lane rule below.

## The question (owner, 2026-09-24)

"Did we use tetris from original code? Don't write it yourself — better use original for
bit-exact, and/or search for a real tetris (Rust or JS) and convert; currently looks buggy
and no tests?"

## What `examples/common/tetris_sim.rs` is (grep-verified, not remembered)

Zero occurrences of `bag`, `srs`, `wall kick`, `gravity`, `hold`. The public surface is
`hard_drop` / `landing_options` / `outcome_features` / `dellacherie_score` /
`render_spot_sentence` / `render_state_sentence` over a 10×20 board and the 7 tetrominoes.
It is a **hard-drop placement-enumeration oracle that happens to use tetromino geometry**,
built to feed the laya sentence protocol ("enumerate landing spots, describe each in
words, ask P(clean)"). No shipping Tetris exposes that unit of work, so "bit-exact vs
original" has no referent here — and the head never observes engine internals, only
sentences. The one true guideline-fidelity gap that affects play feel (the piece stream)
was closed site-side for LIVE play on 2026-09-24: reflex-site `1cc76de` deals live pieces
from a guideline 7-bag (recorded demo replays untouched).

## The pins any "swap" would invalidate (digests inline — check, don't trust this file)

| Artifact | Value |
|---|---|
| Oracle fixture BLAKE3 | `f32c8577bca50726618d2bb4fb27c904148161d650f16a59c01676a97fa540bb` (pinned in `../riir-reflex/src/game_heads.rs` as `TETRIS_FIXTURE_BLAKE3`, "the cross-repo data contract") |
| Oracle fixture sha256 | `12c46035a7f6b5416c4ce34c86bc9b0b890975aff57944739a24c5869b8fc259` |
| Fixture shape | 120 states / 2660 options, grammar `laya-tetris-v2` |
| Byte-identical copies | THIS repo `tests/fixtures/tetris_oracle_laya_en_v2.jsonl` · `../riir-reflex/assets/game_heads/` · `gist-rs/reflex-site` `tests/fixtures/` (sha256-pinned by its golden test) |
| Downstream | riir-reflex fitted tetris head (boot-fitted; published λ=1, in-corpus 44/120, Bench 881) · reflex-site wasm head artifact + the recorded 70-turn demo walk · the bench page tables |

A substrate swap = regen fixture here → re-fit head + bump digests + serve tests
(riir-reflex) → rebuild wasm head + re-record demo + republish tables (reflex-site). Every
published agreement number resets, for a change the head cannot see.

## Verdict (owner-gated; Claude verdict AGREE 2026-09-24)

**Do NOT swap.** Bit-exactness that matters already exists and is enforced: the JS port
reproduces the Rust sim **2660/2660 options byte-identically** (reflex-site
`assets/games/tetris_golden.test.mjs`, self-contained since `1cc76de`). "The current one
looks buggy" was the RENDER layer (green = chosen spot, not a falling-piece indicator;
orange candidate ghosts muddying the board; stale overlay after landing) — fixed the same
day, site-side, with the original Tetris piece palette.

**If guideline fidelity is ever wanted** (SRS kicks / gravity / hold): add it as a NEW
lane with its OWN fixture and OWN head. Never mutate the pinned corpus.

## The "no tests" half — the class, recorded

The golden test existed but only ran with a sibling checkout or `TETRIS_FIXTURE` env —
i.e. its status was **unknown, not passing**. Self-containment (fixtures copied into
reflex-site `tests/fixtures/`, sha256-pinned, `.assetsignore`d out of the deploy) is what
converted it into evidence: 16/16 site tests, 120/120 states, 2660/2660 options.

## Comparability caveat (verdict amendment)

Live arena play now deals from a 7-bag stream; the pinned corpus has NO randomizer (its
states are fixture rows). Live-play agreement is therefore a DIFFERENT distribution from
the in-corpus 44/120 — the arena explain carries this sentence since reflex-site
`e9973c0`. The generated bench table should carry the same caveat at its next
regeneration (not done this session — the tables are riir-reflex harness output).
