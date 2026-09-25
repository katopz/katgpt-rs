# Bench 891 — the depth-2 lookahead player POC (owner directive 2026-09-25: "1 step ahead of laya")

**Status: POC COMPLETE — under hostile garbage (18 rows @ 85% fill, n=20 games): ply2-shaped survives 18/20 games at 901 pieces/g vs ply1-classic 10/20 at 511 — depth-2 next-piece lookahead ~doubles survival; the owner's deep-well shaping adds +1 survival on its own (11/20); on empty boards every player saturates (eval quality dominates, search depth is second-order when nothing threatens).**

Example: `examples/tetris_05_lookahead_poc.rs` (`cargo run --release --example tetris_05_lookahead_poc -- <games> <cap> <garbage_rows>`).

## The three players (ONE shared terminal-board evaluation — every delta is isolated)

| player | design |
|---|---|
| ply1-classic | 1-ply greedy, Lee-style Dellacherie-class board weights (lines +5, row_trans −3.2, col_trans −9.3, holes −7.9, wells −3.4, max_h −0.1) |
| ply1-shaped | 1-ply + the owner's shaping: a well deeper than 2 left open costs −2 × (depth−2)² — *fill it with whatever piece fits now; the I may never come* |
| ply2-shaped ("the reflexer candidate") | depth-2 exhaustive: every current placement × every next-piece placement (~900 evaluated continuations per decision, µs-tier), + shaping. "Not blocking the next-next piece" is native — a blocking placement scores through its own worst forced continuation |

Common: `FromTop` physics (real hard drop, the v3 lane), guideline 7-bag (seeded, same sequence per seed for every player), guideline scoring, `garbage_board(seed, rows, 75..85%)` start.

## Results

**Empty board (cap 2000–5000): saturated.** All players survive every game at ~0.4 lines/piece — the classic eval alone never dies; search depth and shaping have nothing to bite on. points/g ordering (classic 91.5k > shaped 91.3k > 2-ply 88.2k per 5 games @5000) shows the shaping trades a little immediate scoring for safety it doesn't need there.

**Hostile garbage — 18 rows @ 85% fill, cap 1000:**

| player | survived (n=20) | pieces/g | lines/g |
|---|---|---|---|
| ply1-classic | 10/20 | 511 | 209.2 |
| ply1-shaped | 11/20 | 559 | 229.1 |
| ply2-shaped | **18/20** | **901** | **370.5** |

At n=10 same configuration: 5/10 · 6/10 · 9/10 (517/612/901 pieces/g) — stable.

**Death wall:** 20 rows @ 85% is unwinnable for all three (0/10 — the field reaches the spawn zone before any landing matters). The informative stress band is 14–18 rows.

## Context (the arena the user pointed at — reflex-site T12, engine `00aa6221`)

fitted modelless head **140 pts / 3 lines / 46 pieces** · laya **660 pts / 11 lines / 70 pieces @ 394 ms/piece**. The classic 1-ply substrate that already lives in this lane outplays laya even under hostile garbage (~0.41+ lines/piece vs 0.16), and the depth-2 candidate sustains 0.37–0.9+ lines/piece with 90% survival in the regime that kills 1-ply half the time — at µs latency (no model forward at all).

## Honest readings

- **Latency/accuracy framing (the user's "katgpt low acc"):** the 45%-accurate fitted head is a laya-IMITATOR; the lane's classic eval substrate is a far stronger player than the thing it imitates. Different lanes, different jobs — the POC makes the split visible.
- **Lookahead pays exactly where the owner predicted**: under threat (deep wells, garbage), not on empty boards. "Try every next-next combination so you don't block the next next one" = +80% pieces survived at 18 rows.
- **Shaping pays, small**: +1 survival alone (11/20 vs 10/20), and it composes with lookahead.
- **MCTS not needed at depth 2**: exhaustive is stronger (no sampling error) at ~900 evals/decision. riir-ai's MCTS runtime becomes relevant only at depth ≥3 — and the legal direction would be riir-ai consuming a promoted katgpt-core primitive, never the reverse (boundary).

## Next steps (offered, not landed)

1. Lay head-to-head on identical seeds (expensive: ~394 ms/piece via the reflex oracle; 1–3 games feasible).
2. Arena lane: expose the lookahead player on reflex-site (serving-side decision — the reflex tree is sibling-hot).
3. If the owner wants depth-3+ search or real MCTS: file the plan; the substrate boundary above applies.
