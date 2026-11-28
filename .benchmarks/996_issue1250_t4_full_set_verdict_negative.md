# Bench 996 — Issue 1250 Arm A: the registered full-set T4 verdict — CLOSED NEGATIVE

**Lane:** katgpt-attn `tied_indexer` (Issue 1250 Arm A, Research 698 YASA distill) — the
registered T4 adjudication over the TRUE contiguous full set (`PYRAMID_612_FULL_DIR`),
the pre-registered bars from the issue text + Bench 995's demotion law.
**Date:** 2026-11-28 (4090, shared lane — recall is a COUNT, load-invariant by the
cell's own law; the T3 latency table is a release-build ratio read at load ~20-28/30,
disclosed: the 83.7× margin is a complexity claim, not a tuning figure).
**Session:** zed-2e856b94@amo, epoch 1795828774 (the claim lease; folded in the
closure commit). Run: rig job `c7246fa8` (`01a21fd6-1c65…`), release, exit 0, 4/4
tests green — the v1 false-green class (a `;`-chained PowerShell job masking the
first command's failure behind the last command's exit) caught by reading the LOG,
never by trusting exit 0; v2 carries the explicit `$LASTEXITCODE` propagation.

## The registered verdict — ALL THREE T4 LEGS FAIL

128 non-trivial families (2 FA layers × 4 kv heads × 4 true lengths 4K/16K/32K/64K ×
deepest q position; the harness's `MIN_LEAVES` floor):

| arm | recall@8 (pooled, L≥32K non-trivial) | mass-ratio |
|---|---|---|
| tied_same | 0.6406 | 0.8019 |
| single_lse | **0.7793** | 0.8988 |
| single_mean | 0.6475 | 0.8030 |
| tied_cross (0→g, n 96) | 0.4258 | — |

- **T4 SAME-GROUP (the PASS bar): FAIL.** paired diff (tied_same − single_lse) at
  L≥32K = **−0.1582 ± 0.2900 (n 64)** vs the ≥ −0.01 bar.
- **T4 CROSS-GROUP (YASA's actual deployment shape): FAIL.** paired diff
  (tied_cross − that family's single_lse, same truth) = **−0.3672 ± 0.2776 (n 48)**.
  One group's block ranking is a materially worse predictor of another group's
  truth than the family's own exact scan — the transfer mechanism does not hold on
  a model not trained for it.
- **T4 PLAN-FILING leg: NOT MET.** tied_same − single_mean = −0.0117 ± 0.0930 (n 64)
  (it does not even hold the BSA floor at 32K+ — −0.0156/−0.0078 per bin, LB95
  −0.0425/−0.0350).
- **T3 COST leg: PASS** (reported, never a save): tied 71.7 µs vs single_lse
  6003.4 µs at N=65536 (83.7×; the analytic H×(N/C)/(1×N) = 9.4% dot-work ratio).
  Per Bench 995's pre-stated demotion law: **a cheaper wrong answer is not a win**
  (the Bench-1011 class) — the cost leg does not rescue a recall FAIL.

## The closure (the pre-committed demotion law, executed)

- Research 698 Track-A grade **STRUCK**: ~~Gain~~ → NOT-A-GOAT (measured negative),
  recorded in the research note's status + verdict rows in the same commit.
- Issue 1250 **arm A CLOSED NEGATIVE** — the issue file is REMOVED per the
  noise-reduction law; this record + Research 698 carry the verdict. Arm B
  (cross-layer KV-sharing init) is UNTOUCHED and stays filed — a separate lane
  (its consumer-gated T5–T7 were never claimed).
- `tied_indexer` **stays opt-in forever** (the code is correct, fast, and
  gate-tested; it measured the wrong quality class). Re-open = a CLASS change in
  the indexer primitive (per-track-trained indexer, token-level fine step, or a
  model actually trained with tied selection — YASA's own CPT context; also the
  league watch row's opponent trigger: an opponent shipping tied-indexer paths for
  models we score), **never a g/b/scale retune**.

## Why it failed (the honest mechanism read)

The committed reading (Bench 995) predicted this exactly: the mean-of-ReLU over
BLOCK MEANS cannot keep the within-block needle that single_lse's max term keeps —
the tied scorer reproduces the BSA dilution failure mode (tied ≈ single_mean
everywhere), and the exact-LSE scan beats every block-mean arm by the same gap
class Bench 612 documented for the pyramid walk (−0.133). YASA's blog number rides
their 16-token blocks + a model TRAINED with the tied scorer (the softmax backward
shapes the group's heads into usable indexers); at OUR 64-token contract on a model
never trained for it, the two priors Research 698 §5 named (the coarse-first
deficit AND the cross-group tie) both fired. The falsifier did its job.

## Provenance

- Full set: `E:/git/_sync/pycap/full` (274 files) + the root `manifest.json`
  (`pyramid_612_capture_v1`, BLAKE3-verified at every load) — the same capture
  Bench 612 adjudicated on; the committed subsets are the in-repo fixtures.
- Box: 4090 (24 cores), shared lane, GPU 100% by an unrelated census job (CPU-side
  recall arithmetic — unaffected); release build; `git pull --ff-only` landed
  `0ba9acb3b`+`f689aa513`+`8bb6de67a` before the run (the log carries the
  Fast-forward).
- Runs: v1 `df474d7f` (the path-bug false green, disclosed above), v2 `c7246fa8`
  (the record of this file). Worktree torn down after the run (the 4090 lane
  leaves no target dir behind — the job ran in-repo with the default target).
