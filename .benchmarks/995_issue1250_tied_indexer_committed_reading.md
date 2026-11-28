# Bench 995 — Issue 1250 Arm A: the YASA tied-indexer recall cell (committed reading)

**Lane:** katgpt-attn `tied_indexer` (Issue 1250 Arm A, Research 698 YASA distill) — the
zero-parameter tied indexer under the Bench-612 block contract (C=64, top-8, NSA forced
leaves, the exact 612 reference mass) over the REAL committed pyramid_612 capture.
**Date:** 2026-11-28 (amo, contended — load 4–6 through the run; recall is a COUNT, the
cell's own law makes it load-invariant; the release sweep was a 0.24 s re-run of cached
fixtures, disclosed).
**Session:** zed-d6400e07@amo, epoch 1795833000 (claim lease exp 2026-11-30T01:30Z).

## What ran

- `tests/bench_1250_tied_indexer.rs` — 4/4 green (canary, committed cell, full-set
  skip-loud, cost skip-loud): the harness landed by the claim lane (zed-2e856b94@amo,
  in flight at claim time), completed by this session (three dead lib-item references
  repaired onto the landed API, the FamilyCtx refactor the sibling had staged wired
  through both call sites, and the cross-group column's vacuous-loop defect fixed —
  the original loop could never fire its 0→g read).
- `tests/issue_1250_tied_indexer_cell.rs` — 2/2 green (committed columns +
  bit-identical determinism replay): this session's second harness, independent
  loader, per-position families (5 q positions per length vs the bench file's
  deepest-position-only), cross arm measured BOTH directions (0→3 and 3→0).
- `dash_attn::tied_indexer` lib unit tests — 6/6 green (ReLU mechanism pin, head-mean
  hand case, naive tie-out, reuse capacity, two panic guards).
- Postures: clippy `-D warnings` ×3 (pyramid+tied all-targets, dash_attn-only,
  no-default-features) — all clean; `cargo fmt --check -p katgpt-attn` clean;
  feature-off builds compile the module to nothing (the cfg gate).

## The committed reading (L ≤ 4096-row stride-subsampled bins — DIAGNOSTIC, never the T4 verdict)

64 non-trivial families (2 layers × 2 kv heads × 4 lengths × 4 q positions ≥ MIN_LEAVES;
the bench file's deepest-position-only protocol gives the 16-family table; the cell's
per-position protocol gives 240 rows over 4 length bins):

| arm | recall@8 (pooled) | mass-ratio |
|---|---|---|
| tied_same | 0.7695 | 0.8750 |
| single_lse | 0.8906 | 0.9483 |
| single_mean | 0.7695 | 0.8753 |
| tied_cross (0→g) | 0.5859 | — |

- **tied_same − single_lse: −0.1211 ± 0.2142 (n 64)** — the −0.01 bar FAILS at the
  committed lengths. The tied scorer lands EXACTLY on single_mean (tied − single_mean
  = +0.0000 ± 0.0221, sd≈0 on three of four bins): the mean-of-ReLU over block means
  is the BSA-class arm, not the exact-LSE class — the Jensen chain's needle term
  (which single_lse's max keeps and every block-mean arm dilutes) is exactly what the
  tied scorer lacks. The dilution failure mode the 612 gate documented for Mean/PISA-1
  is the tied arm's OWN failure mode at this contract.
- **tied_cross − that family's single_lse: −0.2695 ± 0.2483 (n 32)** — group-0's
  selection shared across groups (YASA's actual deployment shape) loses ~0.27 recall
  vs the family's own exact scan. The YASA transfer mechanism does NOT hold on this
  capture: one group's block ranking is a materially worse predictor of another
  group's truth than the family's own BSA arm.
- The cell's reverse-direction read (3→0 indexing group-0's truth) corroborates:
  cross ≠ same on 114/240 cell families — the two indexers disagree materially in
  both directions; the transfer loss is symmetric, not a head-0 quirk.
- Verdict lines (computed, never asserted — the bench_397 convention): SAME-GROUP
  FAIL · CROSS-GROUP FAIL (paired-diff form after this session's fix; the pre-fix
  "PASS" read n 0 as vacuous-pass) · plan-filing leg NOT MET (per-bin LB95 ≤ 0).

## Verdict posture

**The T4 bar adjudication stays FULL-SET-GATED** (`PYRAMID_612_FULL_DIR` — the true
contiguous L ≥ 32K bins; the drive is not mounted on this box this session). The
committed reading is the DIAGNOSTIC the full-set verdict diffs against. What the
committed data already says honestly:

1. The tied scorer at the 612 block contract is a BSA-class arm (ties single_mean
   everywhere), and the exact-LSE scan beats it by −0.12 pooled — the same gap class
   as the landed 612 mean-vs-LSE deficit. For the tied arm to clear −0.01 at 32K+,
   the length axis would have to reverse a deficit that the committed axis shows
   flat across 4K→65K source lengths (per-bin: −0.133/−0.140/−0.119/−0.127 — no
   length trend). **Pre-registered falsifier position: the committed evidence points
   NEGATIVE; the full-set cell confirms or overturns.**
2. The cross-group (deployment) shape is materially worse still (−0.27) — and YASA's
   own deployment claim rides that shape.

## Demotion law (pre-committed)

Per the issue's ungoat discipline: if the full-set cell confirms FAIL, the grade in
Research 698 is struck, the issue closes negative, and `tied_indexer` stays opt-in
forever (re-open = a class change in the indexer primitive, never a retune). The
cost leg (T3) never runs on a confirmed recall FAIL — a cheaper wrong answer is not
a win (the Bench-1011 class).

## Provenance

- Capture: `tests/fixtures/pyramid_612/` manifest `pyramid_612_capture_v1` — 4090,
  RTX 4090 exclusive, qwen38-27B Q4_K_M whole-model decode, f16-KV posture, BLAKE3
  verified at every load (both harnesses assert).
- Box: amo (M5 Ultra, macOS 27.2), AC power, load 4–6 (leg-A econ cell + sibling
  lanes contending) — counts load-invariant; no latency number is quoted anywhere
  in this record (T3 skipped: full-set-gated + the box was BUSY — the quiet-window
  law binds).
- Runs: `/tmp/i1250a` target dir (isolated, per the cargo-lock law), debug + one
  release re-run (0.24 s, fixture-cached).
