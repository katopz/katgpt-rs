# Bench 892 — the tetris strategy RULEBOOK arena: ruliology enumeration, delta-gated self-evolve, the hybrid FSM champion (Issue 892 T1/T2/T4/T6)

**Status: COMPLETE — GOAT: the HYBRID FSM champion (`68cae9d382014662`) is PROMOTED to the lane's champion; on FRESH seeds it matches the best survival of the depth-3 family exactly (22/60 · 35/40 · 20/20, identical piece counts) at 4.0× / 4.0× / 4.7× the points of Bench-891 weights, and beats laya 60/60 seeds (92–299× points). Bench 891 ply2-shaped demoted to reference; the score-only champion kept as the Build-mode parent; chance-PUCT stays opt-in.**

Examples: `examples/tetris_06_rulebook_arena.rs` (anchor / enum / climb /
eval), rulebook `examples/common/tetris_rulebook.rs`, shared substrate
`examples/common/tetris_lookahead.rs`. Sibling records: laya head-to-head
`.benchmarks/892_laya_h2h.md`, chance-node PUCT `.benchmarks/892_chance_puct_goat.md`.

Common protocol: `DropRule::FromTop` (the laya arena's real hard drop),
seeded guideline 7-bag with a one-piece preview, scoring 40/100/300/1200,
`garbage_board(seed, rows, 75)` starts (75% — Bench 891's prose "85%" was
MEASURED to be a typo: its table reproduces at 75%, and 85% is easier),
release build, games in parallel on the shared M3 Max (load 12–30 during
runs; outcomes are deterministic, only wall times move).

## The rulebook (the ruliology surface)

16 rules as DATA — id, KG triple, source clause, physics precondition,
kind, per-mode weights `[build, downstack, survive]`, feature fn. A
`Build / Downstack / Survive` FSM picks the mode from the ROOT board by data
predicates (`survive_h`, `downstack_holes`); a `Genome` (enable mask +
weight table + depth/beam + thresholds) round-trips through one text line
and is identified by BLAKE3 of that line — the self-evolve surface.

| rule | KG triple | kind under FromTop |
|---|---|---|
| lines, row_trans, col_trans, holes, wells, max_h | classic Dellacherie/Lee terms | board weight |
| deep_well | (deep_well, gets_filled_by, any_fitting_piece) — owner, Bench 891 | board weight |
| nine_one | (stack, reserves, edge_well) — 9-1 stack | board weight |
| tetris | (I_piece, clears, tetris) | board weight |
| flat | (top, stays, flat) — bumpiness over the 9 stack columns | board weight |
| cover | (covered_hole, triggers, downstack) — fix misdrops | board weight + FSM gate |
| hold_i | (hold, reserves, I_piece) | board weight |
| preview | (player, plans_with, next_preview) — depth 2/3 | search |
| hold | (player, uses, hold_queue) | search |
| rotate_both | (player, rotates, both_directions) | **native** — the enumerator reaches every rotation in one step |
| tspin | (T_piece, spins_into, covered_slot) | **inapplicable** — requires `SoftDropRotate`; a top hard drop never enters a covered slot |

**Anchor:** `Genome::bench891_ply2_shaped()` reproduces Bench 891's
ply2-shaped per seed (pieces + lines) **20/20 @ 18 rows cap 1000** — every
delta below is on a verified engine.

## Enumeration (2^7 owner subsets + classic leave-one-out, depth 2, n=20)

Marginal effect per owner rule (mean over the 64 contexts of the others):

| rule | 18@75 pieces/g | 19@75 pieces/g | empty board |
|---|---|---|---|
| hold | +165.8 | **+390.1** | saturated |
| preview (depth 2) | **+115.1** | +24.1 | saturated |
| flat | +4.4 | +12.6 | ≈0 |
| cover (downstack FSM) | −5.6 | **−36.5** | ≈0 |
| nine_one, tetris, hold_i | 0.0 | 0.0 | 0.0 (tetris barely fires at default weights: 0.1–0.2/g) |

Readings: search (preview) and hold carry survival; the owner's scoring
rules do NOTHING at default weights because the per-line reward and the
deep-well urgency pay for singles and punish the reserved well. "Fix
misdrops immediately" as an always-on downstack mode is a measured
NEGATIVE under hostile garbage (it abandons good shape to dig).

## Delta-gated self-evolve (`climb`, no hold — the laya arena has none)

- **Survival climb** (19@75 train seeds 1..=20): 4/150 accepted, train 5→8/20,
  **held-out 7/20 vs Bench 891 8/20 — did NOT generalise** (20-seed noise at
  a survival cliff). Bench 891 weights stay the no-hold survival reference.
- **Score climb** (empty board, fitness points, train 1..=10): 22/150
  accepted → `CHAMPION_POINTS_LINE` (id `ed5aa14b7d68472e`). It DISABLED
  `lines`, `row_trans`, `deep_well` and kept `nine_one` + `tetris` + `flat` —
  the 9-1 stack emerged from the genome, it was not hand-coded.

Held-out seeds 101..=120, cap 1000:

| genome | empty pts/g | tetrises/g | 10@75 | 16@75 | 18@75 surv / pts |
|---|---|---|---|---|---|
| Bench 891 ply2-shaped (d2) | 17,412 | 0.10 | 20/20 · 17,873 | 20/20 · 18,054 | 16/20 · 14,512 |
| score champion (d2) | **64,682** | **44.7** | 20/20 · 61,002 | 19/20 · 53,281 | 15/20 · 43,567 |
| Bench-891 weights, depth 3 beam 6 | 17,240 | 0.05 | — | — | 17/20 · 15,387 |
| score champion, depth 3 beam 6 | **79,208** | **59.4** | — | — | **19/20 · 67,985** |

## The hybrid (FSM doing its job) — hardest board 19@75, n=60, seeds 1..=60

`CHAMPION_HYBRID_LINE` (id `68cae9d382014662`): the score champion's weights
in `Build`, Bench 891's weights VERBATIM in `Downstack`/`Survive` (a
per-mode weight of 0 = rule off in that mode), depth 3 beam 6.

| player | survived | pieces/g | pts/g |
|---|---|---|---|
| Bench 891 ply2-shaped (d2) | 21/60 | 353 | 6,435 |
| Bench-891 weights d3 | 28/60 | 468 | 8,440 |
| score champion d2 | 20/60 | 337 | 18,388 |
| score champion d3 | 23/60 | 386 | 27,351 |
| chance-PUCT b1600 (891 eval, sibling bench) | 30/60 | — | — |
| **hybrid d3 (sh 12, dh 3)** | **28/60** | 468 | **34,023** |

(sh, dh) sweep: survival is 28/60 at every point (the garbage start is
played in Downstack/Survive = Bench-891 weights, where the deaths happen);
points rise with sh: 29,252 (8,1) · 29,428 (8,3) · 32,275 (10,1) · 33,342
(10,3) · 32,935 (12,1) · **34,023 (12,3)**; empty board 67,513 … **79,652**.
⚠ The point was CHOSEN on these seeds — see the fresh-seed table.

## Fresh-seed validation (seeds 201.., never used for selection)

| regime | Bench-891 weights d3 | score champion d3 | **hybrid d3** |
|---|---|---|---|
| 19@75, n=60 | 22/60 · 368 pcs · 6,648 pts | 21/60 · 352 · 25,243 | **22/60 · 368 · 26,632** |
| 18@75, n=40 | 35/40 · 876 · 15,840 | 33/40 · 826 · 59,096 | **35/40 · 876 · 64,000** |
| empty, n=20 | 20/20 · 17,279 · 0.00 tetr | 20/20 · 79,223 · 59.4 tetr | **20/20 · 80,461 · 60.6 tetr** |

The hybrid's survival equals the survival parent's EXACTLY (same survived
count and same mean pieces in both garbage regimes — the deaths happen in
the Downstack/Survive phase, which plays Bench-891 weights verbatim), while
its points beat even the score-only parent in every regime. The score-only
champion loses survival on fresh seeds (−1 / −2 games). ⚠ Survival counts on
60 seeds move with the seed set: the Bench-891-d3 vs score-d3 gap read 28 vs
23 on seeds 1..=60 and 22 vs 21 on 201..=260 — the hybrid's claim rests on
EQUALITY with its survival parent on both sets, not on the gap.

## Laya head-to-head (after) — `.benchmarks/892_laya_h2h.md`

| regime | laya | Bench 891 ply2-shaped | hybrid d3 |
|---|---|---|---|
| empty, cap 500 | 0/20 · 54.7 pcs · 428 pts | 20/20 · 8,684 | 20/20 · **39,358** |
| 10@75, cap 500 | 0/20 · 31.8 · 295 | 20/20 · 9,135 | 20/20 · **35,142** |
| 18@75, cap 1000 | 0/20 · 11.4 · 220 | 18/20 · 16,339 | **19/20 · 65,861** |

ms/decision: laya 288–769 (metal, 6 in flight) · hybrid 7.6–8.6 · ply2 0.4.

## Verdict (GOAT gate)

- **G1 correctness:** anchor reproduces Bench 891 per seed 20/20; genome
  line round-trip + id pins + FSM/feature arms in `selftest()` (run before
  every arena command).
- **G2 perf:** hybrid 7.6–8.6 ms/decision (depth 3, beam 6 — the moka
  prior-pruning trick) vs laya 288–769 ms; depth 2 at 0.3–0.5 ms if latency
  matters more than the last survival.
- **G3 no-regression:** survival equal to the best depth-3 survival player
  on two fresh seed sets and +1/20 vs Bench 891 in the laya h2h; points
  4–4.7×.
- **Promote:** `CHAMPION_HYBRID_LINE` is the lane's champion (the default
  rulebook player to cite and to serve). **Demote:** Bench 891 ply2-shaped
  to the anchor/reference role. **Keep opt-in:** `chance_puct` (wins
  survival only on 19@75 at 5–150× latency). **Negative results:**
  always-on downstack (−37 pieces/g at 19@75), no-hold survival climb
  (did not generalise), 9-1/tetris at default weights (inert until the
  per-line reward and deep-well urgency are switched off in Build).
- **Hold** (+390 pieces/g at 19@75) is the biggest single lever and is NOT
  legal in the laya arena — it stays a sim-only rule until an arena admits it.

## Self-evolve hand-off

The surface is the `Genome` line: `tetris-rulebook-v1 en=… d=… b=… sh=…
dh=… w=key:build/downstack/survive;…`, BLAKE3-identified. The riir-clippy
lifecycle maps directly: a candidate genome is Promoted when it is
not-worse than the incumbent on a FROZEN seed bench (here: seeds 201..,
19@75 n=60 + 18@75 n=40 + empty n=20), Rejected otherwise, Withdrawn if it
later regresses. `climb` is the delta-gated proposer; `enum` is the
enumerative (ruliology) census; `ladder_gate` (katgpt-core, Issue 887) is
the natural curriculum gate over garbage depth when a loop consumes this.

## Reproduce

```sh
B="cargo run --release --example tetris_06_rulebook_arena --"
$B anchor --games 20 --cap 1000 --rows 18
$B enum   --games 20 --cap 1000 --rows 19        # also --rows 18, --rows 0
$B climb  --no-hold --fitness points --rows 0 --games 10 --cap 1000 --iters 150 --delta10 500
$B eval "<genome line>" --rows 19 --games 60 --cap 1000 --seed0 201
```
