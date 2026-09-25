# Issue 893 — promote the tetris rulebook/lookahead substrate from `examples/` into a feature-gated module

Status: **OPEN (filed 2026-09-25; P0 of the agreed reflexer baseline — design verdict closed AGREE, rounds 1–3)**

## Why

The Bench-891/892 lane substrate — `examples/common/tetris_sim.rs` (the
bitmask board sim), `examples/common/tetris_lookahead.rs` (bag/apply/fitter),
and `examples/common/tetris_rulebook.rs` (16 rules as data, the
Build/Downstack/Survive FSM, the `Genome` line + BLAKE3 id, eval) — is
example-scoped code. Examples cannot be depended on: a consumer that needs to
run the evaluator bit-identically (a private engine repo measuring through a
bin-only lane) has nothing to path-dep, and the only alternative is a copied,
diverging second implementation — which would also break the bit-identity
anchors themselves.

katgpt-rs's own contract already accepts this class:
`BOUNDARY.md` (arena infrastructure row) admits the bomber/monopoly arenas as
"GOAT evidence infrastructure … an evaluation harness with no riir dep, upstream
of everything". A Tetris board sim + rulebook evaluator is the same class.
`rayon` is already a direct katgpt-core dependency, so the parallel `decide`
root moves at no new manifest cost.

## Scope

- [ ] T1 Move `tetris_sim.rs` + `tetris_lookahead.rs` + `tetris_rulebook.rs`
      (minus example-only harness glue) into a feature-gated module —
      placement (a `katgpt-core` feature module vs a small own crate)
      decided at plan time; opt-in, NOT default-on at landing. All three are
      one substrate: `tetris_lookahead.rs` and `tetris_rulebook.rs` import
      `crate::tetris_sim::{Board, DropRule, BoardScan, …}`, and the
      column-bitmask `Board` of `3d13f072b` (the ~0.35 ms figure's home)
      lives in `tetris_sim.rs`. `tetris_fixture.rs` STAYS example-side
      (serde + fixture path); `tetris_puct.rs` stays out (`chance_puct` is
      already the opt-in katgpt-core home).
- [ ] T2 **Bit-identity GOAT (G1)**: the Bench-892 h2h per-seed fingerprints
      reproduce to the digit; the fixture replay (v2/v3/v4: 120 states / 2660
      options) stays byte-identical; `tetris_04_preview_fit` +
      `decode_01_losslessness` outputs identical; the `tetris_09_site_walk`
      walk rows (sans ms) + summary identical (20600/115/15);
      `tetris_06_rulebook_arena` anchor 10/10; `eval ≡ eval_with` bit-level
      pin keeps holding. (Everything `3d13f072b` itself verified — the move
      touches all of their imports.)
- [ ] T3 **Perf gate (G2)**: no regression vs the recorded reference
      (~0.35 ms/decision post-`3d13f072b`; the 8× pass is `6e02c72b9`) —
      re-pinned under `--release` WITH box state recorded. Default-on
      promotion only if G2 passes clean.
- [ ] T4 **One copy from day one**: EVERY current importer switches to the
      module — `tetris_sim` consumers (`tetris_01`–`tetris_04`,
      `decode_01_losslessness`) and `tetris_lookahead` consumers
      (`tetris_05_lookahead_poc`, `tetris_07_laya_h2h` [the h2h harness],
      `tetris_06`, `tetris_08`, `tetris_09`) — no local copies remain under
      `examples/common/`. Anything left behind breaks T2's byte-identical
      checks by divergence.
- [ ] T5 **Forward freeze**: the public module ships the Bench-892 champion
      genome `68cae9d382014662` as its pinned REFERENCE genome. Evaluator
      *capabilities* (new rule types, features, FSM changes) may keep landing
      here; evolved genome *values* beyond the reference land only in the
      private consumer — the public surface must not become a drip-feed leak
      channel for the loop's output.

## Non-goals

The self-evolve search loop (enumeration / delta-gated climb driver),
certification pipeline, evolved artifacts, vessels, and any tokenomics wiring
are NOT this issue — they live in the private consumer and are deliberately
out of scope here. This issue moves already-public code into a dependable
shape and freezes it there.
