# Bench 943: Issue 929 Phase 2 — the decode-reinject adjudication (paired vs Bench 942)

**Status:** MEASURED 2026-10-17 (M3 Max, same box posture as Bench 942,
release build riir-infer `8cbc2b5`, ~93 min wall) — **the reinject arm is
NEGATIVE as an accuracy lever: acc 0.0778 vs the stock posture's 0.3667 on
the paired fixture. The loop-refined READOUT carries the value; the
loop-refined STATE does not.** Session: zed-m3-idle-1017c (929's 2nd lease).

## What ran

The Phase-2 arm the stock posture's comment reserved (`B929_REINJECT=1`,
riir-infer `8cbc2b5`): the post-prefill snapshot/restore is skipped, so the
accuracy continuation decodes FROM the loop-refined state — the GDN
recurrent/conv state and the answer position's KV rows carry the K passes'
drift into every generated token. Paired against Bench 942's stock grid:
same binary lineage, same fixture seed, same chat/384tok/standard posture,
same think-aware scoring — the ONLY delta is the decode basis.

```sh
B929_REINJECT=1 B929_PHRASE=standard B929_CHAT=1 B929_ANSWER_TOKENS=384 \
issue929_bonsai2_loop_probe   # 180 queries, 4 loops
```

Behavior-preservation check first: the refactored stock n=8 sanity
reproduced Bench 942's smoke grid exactly (loop-3 two_hop row
0.1429/0.0000, same verdict line), then the reinject n=8 smoke read 0/8
(stock 1/8) before the full spend.

## Paired grid (all/180; stock = Bench 942)

| metric | stock | reinject |
|---|---|---|
| **acc** | **0.3667** (66/180) | **0.0778** (14/180) |
| auroc_margin peak | **0.9327/0.8879** @loop2 ood | 0.8761/0.8025 @loop2 two_hop |
| auroc_margin @loop1 all | 0.8013/0.7358 | 0.6743/0.5688 |
| auroc_cos_argmax peak | **0.7562/0.6979** @loop1 | 0.6971/0.5525 @loop1 ood |
| auroc_cos_bridge | 0.26–0.43 | 0.18–0.52 |
| verdict line | SIGNAL | SIGNAL |

## Reads

1. **Reinjection collapses accuracy 66 → 14 correct (−78%)** at identical
   probe conditions. The loop passes' state drift poisons the decode basis:
   the model declines (im_end at token 0) far more often, and its rare
   answers are worse. The first DiscoLoop question is answered on this
   model: the loop-refined state is NOT a better decode basis — the value of
   looping lives in the READOUT (the probed answer-position signal), which
   is exactly what the stock posture's snapshot/restore discipline protects.
2. **The label axis survives both postures** (SIGNAL verdicts on both) —
   14 positives still stratify a usable margin signal (LB 0.80 @loop2
   two_hop), but with 4× less support the CIs are wider everywhere. The
   stock posture is the better instrument on both axes (accuracy AND signal
   support), so it stays THE instrument; `B929_REINJECT` remains in-tree as
   the recorded arm with its measured negative.
3. **The loop-decay shape differs** — stock decays sharply at loop 3
   (cos_argmax 0.76 → 0.10); reinject decays flatly (0.70 → 0.19). A clean
   peak that decays (stock) vs no clean peak (reinject) is consistent with
   the poisoned-basis read: the reinject signals never had a sharp maximum
   to lose.

## Disposition

- **T1.4 (the reinject leg): CLOSED NEGATIVE** on the league model — the
  arm's GOAT question ("does the loop-refined state improve the
  continuation?") is measured NO, decisively (−78% acc, paired fixture).
  No promotion; the knob stays opt-in/inert by default, the negative is the
  record. Re-open class: a loop-TRAINED checkpoint (Plan 451's standing
  trigger) — a model trained to keep its loop-refined state decode-consistent
  is the only mechanism that could flip this.
- **Phase 2's GOAT adjudication is discharged**: the re-open Bench 942's
  verdict line announced is now answered — the probe's value is readout-side.
- **The stock posture's K=1 snapshot/restore law is now MEASURED-LOAD-BEARING**,
  not just gate-pinned: removing it (this bench's arm) costs 78% of the
  answers.

Raw artifact: `/tmp/issue929_full_reinject.json` (per-query rows;
box-local throwaway — the grids above are the record).
