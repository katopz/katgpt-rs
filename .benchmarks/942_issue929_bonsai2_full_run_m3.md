# Bench 942: Issue 929 — the Bonsai-2-27B full run (n=180) on M3, standard-phrasing arm, think-aware scoring

**Status:** MEASURED 2026-10-17 (M3 Max, 16 cores, ~75 min wall, release build
riir-infer `f0ae27c`) — **verdict line: SIGNAL** (`auroc_margin` LB ≥ 0.6 at
loops 1–2 on every split; T1.3 G1 gate PASSES; Phase 2 GOAT adjudication
re-opens). Session: zed-m3-idle-1017c (issue 929's claim lease).

## What ran

The issue's "one command when the box frees", at the full-run protocol the
933 record specified — plus the two instrument repairs the protocol named as
prerequisites (landed riir-infer `f0ae27c`, see § Scoring):

```sh
B929_PHRASE=standard B929_CHAT=1 B929_ANSWER_TOKENS=384 \
B929_OUT=/tmp/issue929_full_std.json \
issue929_bonsai2_loop_probe   # 180 queries (120 two-hop + 60 one-hop), 4 loops
```

Box state: M3 Max idle lane (Zed exempt per GPU policy; one sibling cargo
test lane on CPU), 64 GiB RAM, RSS peak 7.4 GB (the packed ternary weights),
no GPU involvement (CPU substrate).

## Headline grid (point/lb95, stratified bootstrap 400 iters)

| loop | split | n | acc | auroc_cos_argmax | auroc_margin | auroc_cos_bridge |
|---|---|---|---|---|---|---|
| 0 | all | 180 | 0.3667 | 0.6898/0.6213 | 0.6793/0.6172 | 0.3101/0.2541 |
| 1 | all | 180 | 0.3667 | **0.7562/0.6979** | **0.8013/0.7358** | 0.3127/0.2537 |
| 2 | all | 180 | 0.3667 | 0.5902/0.5326 | **0.9173/0.8670** | 0.3301/0.2683 |
| 3 | all | 180 | 0.3667 | 0.1030/0.0910 | 0.6426/0.5723 | 0.3194/0.2587 |
| 2 | ood | 90 | 0.3556 | 0.5059/0.4256 | **0.9327/0.8879** | 0.2861/0.2101 |
| 2 | two_hop | 120 | 0.2667 | 0.5210/0.4411 | 0.8938/0.8207 | 0.4297/0.3256 |

Reads:

- **The label axis is MEASURED-ALIVE on the league model at real support** —
  66/180 correct answers stratify the AUROC (the smoke's n=1 was fragile;
  this is the adjudication-class population). `auroc_margin` is the live
  signal (peak LB 0.89 at loop 2/ood), `cos_argmax` secondary; `cos_bridge`
  sits at 0.26–0.43 — the smoke's 1.0 was small-n exactly as the 934 record
  warned, and `bridge_top1` stays near-chance (0.05–0.18).
- **The loop structure is real and non-monotone**: every signal peaks at
  loops 1–2 and DECAYS at loop 3 (cos_argmax 0.76 → 0.10 from loop 1 to 3).
  Over-looping degrades the readout — the K sweep is a real axis for the
  Phase 2 adjudication, not a fixed 4.
- **acc is phrasing-bound, not loop-bound**: 0.3667 flat across loops (the
  probe never touches the continuation — K=1 bit-identity preserved).

## The phrasing A/B (the n=8 paired smoke that preceded the full run)

Same binary, same fixture seed, chat/384tok, n=8:

| arm | acc | auroc cells | verdict |
|---|---|---|---|
| possessive (default) | 0/8 | every cell empty (0 correct) | UNDECIDABLE — dead |
| standard | 1/8 | cos_bridge 1.0/1.0 loops 0–2 | SIGNAL (fragile n) |

This reproduces the 933 appendix cross-platform (4090 box: possessive 0/16
dead vs standard 5/16 SIGNAL). **The possessive default's label axis is dead
on both boxes; the standard arm is the live instrument.** The full run ran
the standard arm as an env-knob MEASUREMENT — the instrument default is
unchanged (`B929_PHRASE` default stays possessive); flipping the default
redefines the shared instrument and is the lane/owner's call, now priced:
without the flip, a full possessive run measures an empty-class axis
(0/8 smoke; 0/16 on the 4090).

## Scoring repairs the run depended on (riir-infer `f0ae27c`)

The 933 protocol named both gaps; both are now closed in the bin:

1. **Think-aware scoring**: correctness reads the scored answer window only
   — tokens after the LAST `</think>`, cut back at a `<think>` re-open; a
   `<think>` that never closes scores nothing. A gold-fact quotation
   mid-doubt no longer counts (the old scorer's inflation class).
2. **`<|im_end|>` early stop**: generation halts at the chat turn's end —
   `ANSWER_TOKENS` is an upper bound. Measured effect: 6/8 possessive-arm
   queries and most standard-arm queries emit im_end at token 0 (the model
   declines), costing ~1 token instead of 384 — the full run took 75 min
   where the fixed-384 spend would have been ~4× that.

`B929_OUT` rows carry `gen_scored` / `stopped_im_end` / `n_gen_tokens`
beside the unchanged raw `gen`, so paired re-analysis never guesses which
surface was scored. Machinery gates 3/3 at `f0ae27c` (K=1 bit-identity —
the scoring change cannot move the probe signals, only their labels).

## Cross-platform note

The possessive-arm n=8 smoke on M3 reproduced the 4090's dead-axis read
(0/8 vs 0/16) and the standard arm reproduced its SIGNAL — the instrument's
verdicts are portable across the two boxes at smoke scale. Per-query
generations may still differ at the argmax margins (aarch64 vs x86_64
reduction order); the bench does not pool per-query rows across boxes, only
the grid verdicts.

## What this re-arms

- **T1.3 G1: PASSED** (the verdict line's own gate). T1.4 (the decode-reinject
  half) stays closed-unbuilt — Phase 2's first question is whether the loop
  signal survives reinjection, now with a measured axis to adjudicate.
- **Phase 2 GOAT adjudication re-opens** with two priced design inputs: the
  K sweep (loop 3 decays — the peak is 1–2, not "more is better") and the
  phrasing decision above.
- **The phrasing decision is queued for the lane/owner** (issue 929): flip
  `B929_PHRASE`'s default to standard (the live axis) or keep possessive
  (dead axis) — one string, redefines the shared instrument.
  **→ RESOLVED 2026-10-18 (Claude verdict AGREE, round 1): DEFAULT FLIPPED
  to standard** — riir-infer `c43d36b`. Conditions honored in the same
  commit: the mid-adjudication freeze comment retired (Bench 943 closed the
  reinject leg — nothing was mid-adjudication), the header prints the phrase
  on BOTH arms (old records' empty suffix = possessive), and the env matches
  `standard`/`possessive` exactly with a loud refusal for anything else
  (parsed pre-model-load, so a typo costs nothing). Possessive stays
  reachable via `B929_PHRASE=possessive` for replicating 927–934. Scope:
  bin-only — the katgpt-core `FixtureSpec` library default is unchanged
  (its consumers set the arm explicitly).

Raw artifacts: `/tmp/issue929_smoke_chat.json` (possessive n=8),
`/tmp/issue929_smoke_chat_std.json` (standard n=8),
`/tmp/issue929_full_std.json` (standard n=180 per-query rows) — box-local
throwaways; the grids above are the record.
