# Bench 892 — chance_puct GOAT (Issue 892 T3: the moka PUCT trick for single-player stochastic games)

**Status: COMPLETE — G1 PASS (6/6 unit arms; a Q-sign-flip perturbation reds 5/6, determinism cannot red by construction) · G2 PASS (search overhead 478–668 ns/sim on a heap-free game, b1600/b100 per-sim ratio 1.40 ≤ 3.0) · G4 PASS (0 allocs in a warm search) · G3 PASS in the one regime that discriminates — garbage 19 rows @ 75%, n=60: depth-2 exhaustive survives 21/60 vs PUCT b100 26 · b400 27 · b1600 30 (b1600 per-seed W/T/L 9/50/1, sign p≈0.02), monotone in budget; every easier regime saturates both players (ties) — at 5–150× the per-decision latency. Feature `chance_puct` stays OPT-IN (sole consumer is the tetris example; promotion is Issue 892 T6's call).**

**Box (every row, disclosed — the G2 law):** M3 Max 16 cores, macOS 26.6.2, AC power 100%, release profile, isolated target dir `/tmp/katgpt-892-puct`. The box was **heavily shared**: loadavg 22–28 during the G2 bench and the n=20 arena runs, ~10–17 during the n=60 run; free pages 153k–278k (16 KiB pages ≈ 2.4–4.3 GiB free), swap 1.1 GiB used of 2 GiB throughout. Arena ms/decision is per-worker wall time under 8 concurrent workers on that box — read it as a ratio between rows of one run, never as an absolute (the same ply2 player measured 0.36 ms in the n=60 run and 0.86–0.96 ms in the loaded n=20 runs).

## What shipped

- `crates/katgpt-core/src/chance_puct.rs` (feature `chance_puct`, root forward `chance_puct = ["katgpt-core/chance_puct"]`):
  `trait ChanceGame: Clone { type Action: Copy; type Outcome: Copy; is_terminal, value, actions(&mut Vec<(A, f32)>), apply, outcomes(&mut Vec<(O, f32)>), resolve }`,
  `ChancePuctConfig { budget: 400, c_puct: 1.5, top_k: 8, prior_temp: 0 (auto = std of kept scores), value_scale: 0 (auto = std of the root's kept prior scores) }`,
  `ChancePuct<G>::search(&root, &mut fastrand::Rng) -> Option<ChancePuctPick { action, index, visits, q }>`, `root_stats()`.
  - Prior: `σ((s_i − mean)/T)` normalised over the kept `top_k` (sigmoid, never softmax).
  - Value: `σ((v − root.value())/scale)`, terminal loss = 0, **no sign flip**, `Q` = mean backup; FPU = parent's mean Q.
  - Chance nodes: **sampled ∝ p per simulation** from the caller-seeded Rng (full expansion would multiply every simulation by the chance branching factor — 7 for a bag — while the mean backup already converges to the expectation). A chance node expands for free and the same simulation continues, so each simulation costs exactly one decision expansion (moka's cost model).
  - Root: most-visited; ties → higher Q → lower original index.
- `examples/common/tetris_puct.rs`: the Tetris adapter + `pick_puct(board, cur, next, bag_remaining, &PuctCfg, &mut Rng, eval) -> Option<usize>` (index into `landing_options_with(board, cur, FromTop)`). Information structure: a decision state is `(board, cur, preview, bag)`; placing `cur` yields a chance state whose single event is the NEW preview drawn uniformly from the 7-bag remainder (empty ⇒ full bag) — so the root's known `next` rides the state rather than being a `p = 1` chance node, and every deeper decision sees its own preview as the real game does. Prior = the 1-ply `eval` of each placement; leaf = `eval` of the node's board.
- `examples/tetris_08_puct_goat.rs` (`required-features = ["chance_puct"]`) — the G3 arena.
- `crates/katgpt-core/tests/bench_892_chance_puct_goat.rs` (`required-features = ["chance_puct"]`) — G2 + G4.

## G1 — unit arms (`cargo test -p katgpt-core --features chance_puct --lib chance_puct`)

| arm | pins |
|---|---|
| `gamble_is_judged_by_expectation_not_max` | safe 0 (p=1) vs gamble max +6 / EV −3.6 with the PRIOR favouring the gamble → safe, 5 seeds |
| `favourable_gamble_is_taken` | two-sided: flip the odds (EV +3.6) → gamble |
| `terminal_is_a_loss` | an all-dead action backs up exactly 0 and loses to a −2 leaf; terminal / leaf root ⇒ `None` |
| `top_k_prunes_low_prior_actions` | 20 actions, top_k 8 ⇒ 8 children; best-value/worst-prior unreachable; priors normalised + rank-ordered; top_k 20 ⇒ found |
| `most_visited_root_with_tie_break` | pick = most visited; visit tie → higher Q; full tie → lower index; budget 0 ⇒ top prior unsearched |
| `deterministic_under_seed_and_reuse` | same seed ⇒ identical pick + visit vector, across a dirty reused arena |

Perturbation (`n.total += 1.0 − v`, the Q-sign bug class): **5/6 red** — every arm but determinism.

## G2 + G4 — `bench_892_chance_puct_goat` (release, loadavg 27)

| budget | µs/decision | ns/sim | tree nodes |
|---|---|---|---|
| 100 | 47.8 | 478 | 1,045 |
| 400 | 227.2 | 568 | 4,053 |
| 1600 | 1,068.2 | 668 | 16,421 |

Per-sim scaling b1600/b100 = **1.40** (bar ≤ 3.0 — the walk is O(depth), not O(budget)). G4: warm repeat search **0 allocs**, identical pick. Synthetic game: 16 actions, 4 outcomes, depth ≤ 12, 1-in-29 terminal afterstates, heap-free. The search overhead is negligible against the Tetris game cost: a Tetris decision expansion (landing enumeration + ~34 applies + board features, all allocating in the sim) is ~18–50 µs derived from the G3 rows (ms/decision ÷ budget: b100 1.85 ms → 18.5 µs/sim, b1600 52.6 ms → 33 µs/sim at the lighter load), so the ~0.5 µs search overhead is ≤3% of PUCT's Tetris cost — a zero-alloc landing enumerator in the adapter, not the search, is the latency lever.

## G3 — vs `Player::Ply2Shaped` (Bench 891 champion), identical seeds, same eval (`eval_board(board_features(b, l), shaped)`), cap 1000, FromTop, seeded 7-bag

W/T/L = per-seed pieces placed vs the ply2 row.

| regime | n | player | survived | pieces/g | lines/g | points/g | ms/decision | W/T/L |
|---|---|---|---|---|---|---|---|---|
| 16 rows @ 75% | 20 | ply2 | 20/20 | 1000.0 | 410.1 | 18241 | 0.91 | — |
| | | puct b100 | 20/20 | 1000.0 | 409.7 | 18002 | 7.36 | 0/20/0 |
| | | puct b400 | 20/20 | 1000.0 | 410.4 | 17842 | 22.42 | 0/20/0 |
| | | puct b1600 | 20/20 | 1000.0 | 410.3 | 17784 | 79.65 | 0/20/0 |
| 18 rows @ 75% | 20 | ply2 | 18/20 | 900.5 | 370.5 | 16339 | 0.96 | — |
| | | puct b100 | 19/20 | 950.2 | 391.1 | 17115 | 4.74 | 1/19/0 |
| | | puct b400 | 20/20 | 1000.0 | 411.9 | 18053 | 18.82 | 2/18/0 |
| | | puct b1600 | 18/20 | 900.7 | 371.1 | 16136 | 78.80 | 1/18/1 |
| 18 rows @ 85% | 20 | ply2 | 20/20 | 1000.0 | 413.6 | 18780 | 0.96 | — |
| | | puct b100 | 20/20 | 1000.0 | 413.4 | 18566 | 4.84 | 0/20/0 |
| | | puct b400 | 19/20 | 950.5 | 393.2 | 17559 | 19.08 | 0/19/1 |
| | | puct b1600 | 20/20 | 1000.0 | 413.4 | 18526 | 65.49 | 0/20/0 |
| 19 rows @ 75% | 20 | ply2 | 6/20 | 302.9 | 124.8 | 5589 | 0.86 | — |
| | | puct b100 | 8/20 | 401.7 | 165.4 | 7271 | 3.97 | 3/17/0 |
| | | puct b400 | 8/20 | 401.2 | 165.5 | 7207 | 20.38 | 2/17/1 |
| | | puct b1600 | 10/20 | 500.1 | 206.6 | 9022 | 59.85 | 4/16/0 |
| **19 rows @ 75%** | **60** | ply2 | **21/60** | 352.7 | 145.3 | 6435 | 0.36 | — |
| | | puct b100 | **26/60** | 434.9 | 179.2 | 7863 | 1.85 | **6/53/1** |
| | | puct b400 | **27/60** | 451.5 | 186.0 | 8154 | 13.71 | **8/51/1** |
| | | puct b1600 | **30/60** | 500.6 | 206.3 | 8991 | 52.57 | **9/50/1** |

Sign test over the decided seeds at 19@75 n=60 (two-sided): b100 6–1 p≈0.13, b400 8–1 p≈0.04, **b1600 9–1 p≈0.02**. The large tie counts are mostly forced: one free row leaves many seeds dead in the first few pieces whatever the player does, and ≥21 seeds survive under both — only the ~10 divergent seeds carry the comparison, which is why n=20 could not separate the players.

## Findings

1. **PUCT beats depth-2 exhaustive where the board is genuinely threatened**, and the gain is monotone in budget at n=60 (21 → 26 → 27 → 30 survivors of 60). Depth-2 takes the MAX over the known next piece and never looks past it; PUCT averages over the bag's next draw and goes deeper on the lines it likes — the expectation-vs-max property G1 pins, measured on the real game.
2. **Every easier regime saturates both players** (16@75, 18@85 all survive; 18@75 differs by 1–2 seeds with a non-monotone budget curve = noise). Search depth is second-order when the eval alone can hold the board — Bench 891's finding, one level up.
3. **Cost: 5–150× the latency** of depth-2 (ply2 0.36–0.96 ms vs PUCT 1.9–80 ms per decision, loaded box). All budgets stay far inside the laya arena's 394 ms/piece.
4. **Bench 891's fill, settled by reproduction:** 18@75 here gives ply2 **18/20 at 900.5 pieces** — Bench 891's published "18/20 at 901 pieces", which its prose attributes to 85% fill. At 85% ply2 survives 20/20. The 891 row was measured at 75% (the committed code's default), confirming Issue 892 T0's note.
5. The PUCT defaults (`c_puct` 1.5, `top_k` 8, auto temperature/scale) were NOT tuned — the moka row transplanted as-is.

## Verdict

G1 / G2 / G4 PASS; G3 PASS (no regression in any regime — 27 seed wins vs 5 single-seed losses pooled over 12 comparisons, the n=20 19@75 rows excluded as a subset of n=60 — and a significant survival gain at b400/b1600 in the one discriminating regime). **Stays OPT-IN:** the only consumer is the tetris example, the no-default-consumer rule applies, and the gain costs 5–150× latency; promotion / which budget the rulebook player adopts is Issue 892 T4/T6's call.

## Reproduce

```bash
CARGO_TARGET_DIR=/tmp/katgpt-892-puct cargo test -p katgpt-core --features chance_puct --lib chance_puct
CARGO_TARGET_DIR=/tmp/katgpt-892-puct cargo test -p katgpt-core --features chance_puct --release \
  --test bench_892_chance_puct_goat -- --nocapture
CARGO_TARGET_DIR=/tmp/katgpt-892-puct cargo run --release --features chance_puct \
  --example tetris_08_puct_goat -- 20 1000 8 100,400,1600 16:75,18:75
# the discriminating regime
cargo run --release --features chance_puct --example tetris_08_puct_goat -- 60 1000 8 100,400,1600 19:75
```
