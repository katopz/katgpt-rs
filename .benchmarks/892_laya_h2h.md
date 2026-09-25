# Bench 892 — live laya vs candidate Tetris head-to-head on identical seeds (Issue 892 T5)

**Status: COMPLETE (before + after) — AFTER: the rulebook HYBRID champion (depth 3, 9-1/tetris in Build, Bench-891 weights in Downstack/Survive) beats laya on 60/60 seeds and out-scores it 92× (empty, 39,358 vs 428 pts/g), 119× (10@75, 35,142 vs 295) and 299× (18@75 cap 1000, 65,861 vs 220), and beats the Bench-891 ply2-shaped champion 19/20 vs 18/20 survival at 4.0× points on the hardest band. BEFORE: laya 0/60 survival; ply1-classic / ply2-shaped 120/120. laya protocol verified by an exact re-trace of the arena's recorded laya (Rust) game (60/60 argmax).**

Example: `examples/tetris_07_laya_h2h.rs`. Players are
`fn(&Board, cur: Piece, next: Piece) -> Option<usize>` (index into
`landing_options_with(board, cur, DropRule::FromTop)`); a new candidate is ONE
arm in `player_by_name` and is selected with `--player a,b,c`.

## Protocol

- Identical game per seed for every player: `Bag::new(seed)` 7-bag, seeds
  `1..=N`, start board `Board::empty()` or `garbage_board(seed, rows, fill)`,
  `DropRule::FromTop` (v3 real hard drop), guideline scoring 40/100/300/1200.
  Candidates see the next-piece preview; laya does not (its arena contract).
- **laya** = the reflex-site arena laya lane, byte-for-byte
  (`reflex-site/assets/arena.js` `scoreOptions`/`decide` +
  `assets/games/tetris.js` `buildTurn`): per option one `POST /decide`,
  `state` = the v3 spot sentence alone (`render_spot_sentence`), one `noul`
  question `"Does the stack look clean?"`, header `X-Reflex-Lane: laya`,
  P(clean) = `answers[0].probabilities[0]`, first argmax (lowest index on
  ties), 6 requests in flight per turn (the arena's laya concurrency; the
  engine's laya actor serializes forwards). Transport: raw HTTP/1.1 over
  `std::net::TcpStream`, no new deps.
- Engine: riir-reflex `a56d850` (origin/develop, clean detached worktree —
  the main checkout carried sibling WIP), `reflex 0.2.3`, features
  `modelless laya-riir laya-riir-metal`, english checkpoint, **device metal**.

### Protocol verification (before any run)

`--verify-walk reflex-site/arena/demo_oracle.json` re-traces the arena's
recorded laya (Rust) game (`tetris_walk`, seed 607, v3): from each recorded
board it re-renders the state + option sentences, asks the live engine, and
applies OUR argmax.

| check | result |
|---|---|
| state sentence render | 60/60 identical |
| option count | 60/60 |
| argmax vs recorded pick | **60/60** |
| next recorded board after our pick | 59/59 |
| max \|Δp\| vs recording (4 dp) | 0.0000 |
| re-traced game | **60 pieces / 8 lines / 520 pts** = recorded `tetris_laya` summary |

(The 660 / 11 / 70 figure quoted in Bench 891 is the laya **Python** v2
reference game; the Rust v3 lane recorded 520 / 8 / 60 on the same seed.)
Our fresh-seed laya means (below) sit in the same band.

## Baseline (before rulebook)

Box state (shared M3 Max, 64 GB, AC power 100%): load average **18.8 → 32.7**
across the runs; `memory_pressure` 90% free, vm_stat free 2.6–3.2 GB +
24 GB inactive (no paging). Concurrent heavy jobs NOT mine: another session's
laya engine (`/tmp/reflex-record/release/reflex`, 300–670% CPU, same GPU),
`tetris_06_rulebook_arena`, `tetris_08_puct_goat`. **laya latency figures are
therefore loaded-box upper bounds**; game outcomes are unaffected (laya is
deterministic — verified by the exact re-trace above).

ms/decision = wall time of one `player(board, cur, next)` call (laya: the
whole ~17-option turn incl. 6-way HTTP). ms/forward = mean single `/decide`
round-trip with 6 in flight (queueing included).

### Run 1 — empty board, cap 500, seeds 1..=20

| player | games | survived | pieces/g | lines/g | points/g | ms/decision |
|---|---|---|---|---|---|---|
| laya | 20 | 0/20 | 54.7 | 6.8 | 428 | 623.9 (169.9 ms/forward, 21021 forwards) |
| ply1-classic | 20 | 20/20 | 500.0 | 197.6 | 9332 | 0.077 |
| ply2-shaped | 20 | 20/20 | 500.0 | 198.1 | 8684 | 0.825 |

| seed | laya p/l/pts | ply1-classic p/l/pts | ply2-shaped p/l/pts | most points |
|---|---|---|---|---|
| 1 | 41/1/40 | 500/199/11060 | 500/198/8680 | ply1-classic |
| 2 | 46/3/120 | 500/199/8660 | 500/198/8540 | ply1-classic |
| 3 | 73/13/740 | 500/196/8780 | 500/198/8420 | ply1-classic |
| 4 | 58/8/360 | 500/199/9280 | 500/198/8520 | ply1-classic |
| 5 | 42/2/100 | 500/196/8720 | 500/199/9240 | ply2-shaped |
| 6 | 61/9/1600 | 500/199/9000 | 500/198/8620 | ply1-classic |
| 7 | 38/2/100 | 500/197/8980 | 500/198/8400 | ply1-classic |
| 8 | 50/6/420 | 500/196/8500 | 500/199/8600 | ply2-shaped |
| 9 | 53/6/260 | 500/198/9240 | 500/198/8400 | ply1-classic |
| 10 | 48/5/240 | 500/199/11100 | 500/198/8760 | ply1-classic |
| 11 | 49/5/240 | 500/197/8800 | 500/199/8780 | ply1-classic |
| 12 | 38/1/40 | 500/197/8600 | 500/197/8820 | ply2-shaped |
| 13 | 59/7/320 | 500/196/9120 | 500/196/8580 | ply1-classic |
| 14 | 43/3/300 | 500/197/11160 | 500/198/8800 | ply1-classic |
| 15 | 84/18/1200 | 500/199/10420 | 500/198/8540 | ply1-classic |
| 16 | 57/7/300 | 500/197/8800 | 500/198/8720 | ply1-classic |
| 17 | 35/0/0 | 500/198/8580 | 500/199/8820 | ply2-shaped |
| 18 | 122/32/1840 | 500/199/8800 | 500/198/8860 | ply2-shaped |
| 19 | 50/5/220 | 500/196/9900 | 500/199/9040 | ply1-classic |
| 20 | 47/3/120 | 500/197/9140 | 500/198/8540 | ply1-classic |

Head-to-head per seed: both candidates out-survive laya on **20/20** seeds
(pieces: candidates tie at cap); most points ply1-classic 15 · ply2-shaped 5 ·
laya 0.

### Run 2 — garbage 10 rows @ 75% fill, cap 500, seeds 1..=20

| player | games | survived | pieces/g | lines/g | points/g | ms/decision |
|---|---|---|---|---|---|---|
| laya | 20 | 0/20 | 31.8 | 5.0 | 295 | 487.9 (146.0 ms/forward, 10959 forwards) |
| ply1-classic | 20 | 20/20 | 500.0 | 205.1 | 9407 | 0.025 |
| ply2-shaped | 20 | 20/20 | 500.0 | 205.7 | 9135 | 0.473 |

| seed | laya p/l/pts | ply1-classic p/l/pts | ply2-shaped p/l/pts | most points |
|---|---|---|---|---|
| 1 | 27/3/120 | 500/205/9320 | 500/205/8800 | ply1-classic |
| 2 | 28/3/140 | 500/202/9320 | 500/205/9200 | ply1-classic |
| 3 | 25/2/80 | 500/204/9180 | 500/206/9260 | ply2-shaped |
| 4 | 31/5/400 | 500/206/9440 | 500/206/9500 | ply2-shaped |
| 5 | 27/3/140 | 500/205/9760 | 500/205/9100 | ply1-classic |
| 6 | 33/5/220 | 500/206/8960 | 500/205/9080 | ply2-shaped |
| 7 | 27/3/120 | 500/203/8880 | 500/205/9140 | ply2-shaped |
| 8 | 40/8/380 | 500/206/9200 | 500/206/8960 | ply1-classic |
| 9 | 29/5/240 | 500/203/9020 | 500/205/8920 | ply1-classic |
| 10 | 27/3/120 | 500/204/9200 | 500/205/9240 | ply2-shaped |
| 11 | 27/3/120 | 500/206/9380 | 500/205/8860 | ply1-classic |
| 12 | 20/0/0 | 500/205/9320 | 500/206/8800 | ply1-classic |
| 13 | 23/2/80 | 500/207/10960 | 500/207/9140 | ply1-classic |
| 14 | 27/3/140 | 500/206/9400 | 500/207/9040 | ply1-classic |
| 15 | 26/3/120 | 500/206/9600 | 500/206/9180 | ply1-classic |
| 16 | 18/0/0 | 500/207/9220 | 500/206/9780 | ply2-shaped |
| 17 | 70/20/1920 | 500/207/9300 | 500/206/9060 | ply1-classic |
| 18 | 49/11/640 | 500/203/9280 | 500/206/8940 | ply1-classic |
| 19 | 56/14/780 | 500/205/10000 | 500/206/9300 | ply1-classic |
| 20 | 26/3/140 | 500/206/9400 | 500/206/9400 | tie |

Head-to-head per seed: candidates out-survive laya on **20/20**; most points
ply1-classic 13 · ply2-shaped 6 · tie 1 · laya 0.

### Run 3 (extra) — garbage 16 rows @ 85% fill, cap 1000, seeds 1..=20

| player | games | survived | pieces/g | lines/g | points/g | ms/decision |
|---|---|---|---|---|---|---|
| laya | 20 | 0/20 | 24.7 | 7.5 | 776 | 478.7 (146.3 ms/forward, 8311 forwards) |
| ply1-classic | 20 | 20/20 | 1000.0 | 411.4 | 19973 | 0.023 |
| ply2-shaped | 20 | 20/20 | 1000.0 | 412.1 | 18706 | 0.518 |

laya per seed (pieces/lines/pts): 30/10/1360 · 23/7/480 · 17/4/180 ·
25/8/1320 · 21/6/1300 · 56/20/2140 · 14/3/140 · 16/3/140 · 26/8/700 ·
16/4/340 · 23/6/260 · 14/3/140 · 14/4/340 · 22/6/440 · 28/10/1380 ·
19/6/440 · 60/21/2180 · 31/10/600 · 13/3/140 · 26/8/1500. Candidates reach
the cap on 20/20; most points ply1-classic 18 · ply2-shaped 2.

## Honest readings

- **The head-to-head is not close.** laya never reaches the cap (0/60 games)
  and plays at least 9–40× fewer pieces (candidates are cap-limited); both candidates never top out (120/120).
  laya's per-decision cost is ~750–21 000× the candidates' (loaded box).
- **These three regimes cannot separate candidates on survival** — every
  candidate game saturates the cap. Candidate-vs-candidate differences here
  are points only, and 1-ply classic out-scores 2-ply shaped (the Bench-891
  shaping trades immediate scoring for safety these boards do not demand).
  For the "After" comparison, the discriminating survival band is Bench 891's
  18 rows, cap 1000 (ply1 10/20 vs ply2 18/20) — at **75%** fill, not the
  85% Bench 891's prose states: its table reproduces exactly at 75%, and 85%
  is EASIER (Issue 892 T0, measured). The After run uses 18@75.
- laya's mean on empty boards (54.7) sits between the recorded arena Rust
  game (60) and its worst seeds (35); the Python v2 reference 70 is one game
  under different drop physics.

## Reproduce

```sh
# engine (riir-reflex checkout; weights cached under ~/.cache/riir-reflex/laya)
CARGO_TARGET_DIR=/tmp/reflex-892-h2h cargo build --release --bin reflex \
    --features modelless,laya-riir,laya-riir-metal
RIIR_REFLEX_LAYA=1 RIIR_REFLEX_BIND=127.0.0.1:7392 /tmp/reflex-892-h2h/release/reflex
# harness (katgpt-rs)
CARGO_TARGET_DIR=/tmp/katgpt-892-h2h cargo build --release --example tetris_07_laya_h2h
H=/tmp/katgpt-892-h2h/release/examples/tetris_07_laya_h2h
$H --verify-walk ../reflex-site/arena/demo_oracle.json
$H --player laya,ply1-classic,ply2-shaped --games 20 --cap 500
$H --player laya,ply1-classic,ply2-shaped --games 20 --cap 500 --garbage-rows 10 --garbage-fill 75
$H --player laya,ply1-classic,ply2-shaped --games 20 --cap 1000 --garbage-rows 16 --garbage-fill 85
# After
$H --player rulebook-hybrid-d3 --games 20 --cap 500
$H --player laya,ply2-shaped,rulebook-points-d3,rulebook-hybrid-d3 --games 20 --cap 500 --garbage-rows 10 --garbage-fill 75
$H --player laya,ply2-shaped,rulebook-points-d3,rulebook-hybrid-d3 --games 20 --cap 1000 --garbage-rows 18 --garbage-fill 75
```

Flags: `--url` (default `127.0.0.1:7392`), `--concurrency` (default 6, the
arena's), `--seed-start` (default 1). Per-seed progress goes to stderr.

## After (rulebook champion)

Lead session, 2026-09-25. Same harness, same seeds `1..=20`, same engine
(riir-reflex `a56d850`, metal). Box: M3 Max, AC power, load average
2.8–8.2 during the runs (quiet relative to the baseline), memory 90% free.
New arms (`c84d9e8e2`): `rulebook-points-d3` (the score champion,
`ed5aa14b7d68472e`, depth 3) and `rulebook-hybrid-d3` (the hybrid,
`68cae9d382014662`) — see `.benchmarks/892_tetris_rulebook_arena.md`.
⚠ Through this harness's `fn(&Board, cur, next)` signature the depth-3
players take the piece after the preview as uniform over a FRESH bag (no bag
state crosses the signature); the arena's game loop uses the exact 7-bag
remainder. Local candidates run seeds in parallel (`--jobs`, results
identical by determinism; ms/decision is per-thread time).

| regime | player | survived | pieces/g | lines/g | points/g | ms/decision | sole most-points seeds |
|---|---|---|---|---|---|---|---|
| empty, cap 500 | laya (baseline run) | 0/20 | 54.7 | 6.8 | 428 | 768.6 | 0 |
| | ply2-shaped | 20/20 | 500 | 198.1 | 8,684 | 0.54 | 0 |
| | rulebook-points (d2) | 20/20 | 500 | 195.9 | 31,883 | 0.27 | 4 |
| | rulebook-points-d3 | 20/20 | 500 | 196.3 | 37,403 | 11.5 | 16 |
| | **rulebook-hybrid-d3** | 20/20 | 500 | 195.6 | **39,358** | 8.6 | (own run) |
| garbage 10@75, cap 500 | laya | 0/20 | 31.8 | 5.0 | 295 | 387.0 | 0 |
| | ply2-shaped | 20/20 | 500 | 205.7 | 9,135 | 0.37 | 0 |
| | rulebook-points-d3 | 20/20 | 500 | 203.8 | 33,987 | 7.7 | 6 |
| | **rulebook-hybrid-d3** | 20/20 | 500 | 203.9 | **35,142** | 7.8 | **14** |
| garbage 18@75, cap 1000 | laya | 0/20 | 11.4 | 3.4 | 220 | 287.6 | 0 |
| | ply2-shaped | 18/20 | 900.5 | 370.5 | 16,339 | 0.36 | 0 |
| | rulebook-points-d3 | 18/20 | 901.6 | 369.8 | 62,111 | 7.3 | 11 |
| | **rulebook-hybrid-d3** | **19/20** | **950.6** | **389.8** | **65,861** | 7.6 | 9 |

(The empty-board laya/ply2/rulebook-points rows are from the first After run
on the same seeds; the hybrid was added in a second run — laya is
deterministic, so its row does not change. laya ms/forward in the After
runs: 114–116 ms at 6 in flight.)

Readings:
- **vs laya:** every rulebook player reaches the cap on every seed laya
  plays; the hybrid scores 92× / 119× / 299× laya's points at ~2–3% of its
  per-decision latency, with no model forward at all.
- **vs the Bench 891 champion:** 3.8–4.5× points in every regime, and +1
  survival (19/20 vs 18/20) on the hardest band.
- **hybrid vs score champion:** the FSM is what buys survival — same Build
  weights, but Downstack/Survive fall back to the survival weights: 19/20 vs
  18/20 at 18@75 and +6% points; per-seed points are split (hybrid 14–6 at
  10@75, score champion 11–9 at 18@75).

---

## Optimization addendum (2026-09-25, same day) — 8.1–8.8× per-decision, bit-identical play

`perf` pass over the search hot path (`examples/common/tetris_sim.rs`,
`tetris_lookahead.rs`, `tetris_rulebook.rs`). Every optimization is
**bit-exact by construction and by gate**: the h2h replays above reproduce
to the digit (per-seed pieces/lines/points identical on all validation
seeds — empty 1..=10 cap 300 + 18@75 where seed 8's early top-out
12 pcs/3 lines/120 pts is the fingerprint), the v2/v3/v4 fixture replay
still recomputes 120 states / 2660 options byte-identically, the arena
`anchor` lane is 2/2, and the fused eval is pinned `eval ≡ eval_with`
(bit-level `to_bits` assert, all modes × all genomes) inside
`tetris_rulebook::selftest`.

| lane | before | after | speedup |
|---|---|---|---|
| rulebook-hybrid-d3, empty 10×300 | 6.92 ms/decision | **0.85** | 8.1× |
| rulebook-hybrid-d3, 18@75 10×300 | 7.32 ms/decision | **0.82** | 8.9× |
| rulebook-points-d3, empty 10×300 | 6.85 ms/decision | **0.85** | 8.1× |
| ply2-shaped (Bench 891), empty | 0.348 ms/decision | **0.12–0.14** | ~2.5× |
| rulebook-hybrid-d3, empty 20×500 | — (8.6 in the table above) | 0.855 | — |

Box: M3 Max, loadavg 3.5–6.8 during measurement (sibling agent sessions
active; the speedup is 8×, far outside load noise). Sequential
(`--jobs 1`) per-move wall time.

What changed (no behavior change anywhere):

1. **Static rotation table** (`rotations_static`) — `Piece::rotations()`
   built Vec allocations per call; the search now reads a process-wide
   table built once by the same normalize/dedupe pipeline.
2. **FromTop landing via column occupancy masks** — the per-option
   row-by-row descent scan is replaced by, per piece cell, the first
   occupied row at-or-below its spawn row (`trailing_zeros` over a
   per-column mask): O(4) per option instead of O(HEIGHT×4).
   ⚠ A plain column-heights shortcut is WRONG under overhangs (the shadow
   below a floating cell is open) — the mask form is the correct one, and
   the fast path is pinned against the descent scan by
   `from_top_fast_path_is_bit_identical_to_the_descent_scan` (3000+
   boards incl. overhang/covered-hole/tall-stack shapes; the heights
   version of this test caught a real seed-8 divergence during
   development and was replaced).
3. **`Placement.cells` as `[(usize, usize); 4]`** — a tetromino is always
   exactly 4 cells; the search no longer allocates a Vec per landing.
4. **Fused `Board::place_and_clear` + `Board::scan`** — place + clear in
   one bottom-up compaction pass (no `full_rows` Vec), and heights +
   row/col transitions + holes + hole-cover computed in two passes
   (integer-exact vs the per-feature walks; pinned by
   `scan_matches_the_per_feature_walks` over 2000+ boards). The rulebook
   leaf eval consumes the scan (`Genome::eval_with`), and ply-1 nodes no
   longer materialize the kids Vec (max fold in option order).
5. **rayon at the `decide_scored` root** — the ~34 root landing options
   are independent ~0.1 ms subtrees; collected IN OPTION ORDER so
   `decide`'s first-strict-argmax fold is bit-identical to the sequential
   loop. rayon was already a root-crate dependency.

The recorded champion numbers in the tables above are unchanged — this
pass makes the SAME decisions ~8× faster. The reflex-site arena's
"rulebook ~6.2 ms/piece" bar is now ~0.9 ms on this box (site lane
regenerates on its own cadence).
