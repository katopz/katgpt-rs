# Bench 934 — Issue 929 T1.2b Bonsai-2-27B (PQ2_0) loop-alignment probe, the FULL adjudication run — M3, chat no-think posture

**Numbering note:** allocated 934 after a same-day INDEPENDENT dual allocation of 933 (`933_…_smokes.md`, the 4090-idle session's three smoke postures, `c9c9c0f3e`) — two sessions, two stems, one number; the unpushed side (this file) renumbered per the renumber-never-reuse law. This record is the FULL-RUN companion to [Bench 933](933_issue929_bonsai2_loop_probe_smokes.md): their smokes prove the axis alive and root-cause the OOM; this run is the T1.3/T1.4 adjudication attempt.

**Status:** **COMPLETED 2026-10-12 01:31 +07** — 45,916 s (12.75 h) wall incl. the 164 s load; 180/180 queries at ~255 s/q steady-state. Log `/tmp/b929_full.log` + JSON `/tmp/b929_full.json` (volatile /tmp — the grid below is the verbatim copy, the preservation). Verdict filled 2026-10-14 (idle-loop record completion; the record skeleton + instrument-repair disclosures were committed at launch to pin the number). **Posture vs Bench 933's protocol note:** theirs recommends CHAT + 384-tok think budgets (~21 h); THIS run is the no-think posture (system + empty-think prefix, 8-tok answers, ~8 h) — both measured, cross-referenced; the phrasing-inversion caveat rides BOTH records.

**Concurrent-landing reconciliation (Issue-825 class, resolved by union):** while this run was in flight, the 4090-idle session landed `61fc36f` (bounded HybridCache KV — a measured >100 GiB OOM on the unbounded ctor — plus their own B929_CHAT framing variant, bare scaffold, think-then-answer at ~384-tok budgets) and this session's `d633fbb` (clean-state continuation + no-think chat posture + B929_SKIP) stranded on a diverged local develop. Reconciled as **`69f520e`** (the union, pushed fast-forward from a detached `61fc36f` worktree): bounded KV + clean-state continuation + the no-think posture as `B929_CHAT=1`. **The in-flight run's instrument == the union's chat posture byte-for-byte in behavior** (same template string; the KV bound is allocation-only, zero numerical effect) — its numbers record for the union instrument. The bare-scaffold/think-budget variant stays recorded in the issue as the alternative elicitation (~420 s/q), not this run's posture.

**Date:** 2026-10-11 · **Issue:** [929](../.issues/929_loop_alignment_probe_decode_reinject_poc.md) · **Research:** 614 (DiscoLoop, arXiv:2607.00341)
**Runner:** `riir-infer` bin `issue929_bonsai2_loop_probe` (`--features issue929_bonsai2_probe`), detached worktree at riir-infer `32fd838` + two instrument repairs recorded below (sibling WIP in the main checkout excluded by construction — the measurement tree is HEAD + the bin edit only)
**Box:** M3 Max, AC power, 16 cores; concurrent sibling cargo sessions present (load 4–8 of 16) — decode-QUALITY measurement (argmax identities + cosine readouts + AUROC), latency-free by construction; wall time disclosed anyway.

## Verdict

**SIGNAL — the pre-registered T1.4 bar is formally met, entirely on the `auroc_margin` axis — but the read is phrasing-limited and fragile, and is NOT the T1.3 adjudication of record.** Six (loop, split, signal) cells carry AUROC 95% CI LB ≥ 0.6, every one on margin: **loop 2/all 0.8897/0.8300** (peak), loop 2/two_hop 0.8661/0.7920, loop 1/all 0.8020/0.7192, loop 1/ood 0.8123/0.6705, loop 2/ood 0.8123/0.6858, loop 0/ood 0.7893/0.6092. The margin signal peaks at k=2 and COLLAPSES at k=3 (all-split 0.2996/0.1350) — the loop-refinement window the paper and the kimi legs (Bench 930, peak k=2–3) both show. `auroc_cos_bridge` — the axis the Bench-933 n=4 smoke read at 1.0000/1.0000 every loop — is FLAT at 0.14–0.24 across the whole grid: the smoke's cos_bridge signal did not survive n=180 (small-n artifact, disclosed). KILL not fired (T1.4's negative branch stays unfired).

Three caveats bound the claim (all pre-registered in this record + Bench 933 before launch):
1. **Possessive posture** — the run rides the as-landed possessive phrasing that Bench 933's A/B shows kills two-hop ANSWER accuracy at the no-think posture (0/16). Measured here: acc two_hop 0.0250 (3/120), all 0.0611 (11/180) — the correctness axis near-dead exactly as predicted; phrasing-limited by construction (the issue's own standing note).
2. **Fragile positives** — the AUROC positive class is 3–11 items; the 400-iter stratified bootstrap resamples that class whole. The bar is formally met; the read of record remains the owner-gated `B929_PHRASE=standard` full run.
3. **Instrument vintage** — the run's tree is the pre-union bin (worktree at `32fd838` + the chat/clean-state edits; reconciled upstream as `69f520e` per the paragraph above — behaviorally identical: same template string, KV binding allocation-only). The post-launch evolutions (`B929_PHRASE`, `B929_OUT` per-query rows, bounded KV) are NOT in this run; the log carries no `phrase=` field because the knob did not exist yet — the posture is possessive by the default it predated.

## Instrument repairs landed with this run (disclosed, both binary in effect)

1. **Chat-template elicitation (`B929_CHAT=1`)** — the as-landed completion-style prompt reads 0% on Bonsai-2: the chat/reasoning-tuned checkpoint closes the document past `<|im_end|>` and opens its own `<think>` block instead of answering (8-query smoke: `gen="<|im_end|> <think> Let me work through"`). The chat wrap (qwen3 template + empty-think prefix `<think>\n\n</think>\n\n`) makes it answer IN the answer window; specials ride the GGUF control-token matcher (one id each). Contrast Bench 930 provenance item 6: kimi's chat ids sat ABOVE vocab_size — Bonsai-2's are IN-vocab, so the chat posture is valid HERE.
2. **Clean-state continuation (label-axis integrity)** — the as-landed bin fed the greedy continuation from the K-pass looped state: each pass ADVANCES the GDN recurrent/conv state and rewrites the answer position's KV ("that IS the loop", loop_probe.rs), so the continuation decoded from a K×-advanced recurrent state — output degenerated to one near-miss character + `<|im_end|>` (`gen="z<|im_end|>"` for gold "zesty falcon"). The repair: snapshot GDN+KV post-prefill → probe K passes → restore → re-embed the ORIGINAL token → ONE stock stack pass (K=1 gate-pinned bit-identical to stock) → continuation. The loop intervention stays measurement-only at the answer position (reinject is Phase 2, closed unbuilt); the loop-refined first token is still the decode's first token.
   ⚠ **Confound disclosure:** the kimi legs (Benches 927+930) ran K=2 through the SAME polluted continuation — their 0% accuracy labels carry this instrument defect as a confound. Their UNDECIDABLE verdict is unaffected (0% ⇒ empty class either way), but the "content-untrained" attribution should be read with the confound noted.

Smoke (8 queries, one-hop controls, chat posture, post-repair): acc 4/8, AUROC cells decidable, gold answer reproduced verbatim (`gen="zesty falcon<|im_end|>"`), verdict line SIGNAL at n=8 (wide CIs — smoke only).

## Run posture

- Model: `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.2 GB, PACKED ternary load ≈ 7.3 GB resident, never f32-dequantized; untied head verified at load).
- Fixture: paper §2 shape, defaults — **180 queries (120 two-hop + 60 one-hop controls; two pools × (60+30), the Bench-930 shape)**, 12-fact windows, K=4 loops, 8-token greedy continuation, 400-iteration stratified bootstrap (seed 929). Single-threaded CPU ternary matvecs: ≈8 h wall disclosed (the box stays shared; the measurement is decode-quality, latency-free by construction).
- T1.4 kill bar (pre-registered): SIGNAL iff ≥1 decidable (loop, split, signal) AUROC 95% CI LB ≥ 0.6; KILL iff every decidable cell LB < 0.6; UNDECIDABLE only on an empty class.

## Results

Verbatim from `/tmp/b929_full.log` (score cells are point/lb95, stratified bootstrap 400 iters, seed 929):

```
# issue929 Bonsai-2-27B (PQ2_0) loop-alignment probe
# model=/Users/katopz/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf | head=untied | fixture: 180 queries, facts=12, loops=4 | elicitation=chat
# score cells are point/lb95 (stratified bootstrap, 400 iters)
loop,split,n,acc,bridge_top1,auroc_cos_argmax,auroc_margin,auroc_cos_bridge
0,all,180,0.0611,0.5500,0.6466/0.4744,0.6084/0.4024,0.2361/0.1573
0,two_hop,120,0.0250,0.8250,0.3191/0.1054,0.0997/0.0427,0.2137/0.0142
0,ood,90,0.0333,0.5778,0.4521/0.2031,0.7893/0.6092,0.1552/0.1264
1,all,180,0.0611,0.5056,0.7370/0.5966,0.8020/0.7192,0.2378/0.1573
1,two_hop,120,0.0250,0.7583,0.6296/0.4587,0.6011/0.4302,0.2222/0.0427
1,ood,90,0.0333,0.5111,0.4866/0.2184,0.8123/0.6705,0.1552/0.1264
2,all,180,0.0611,0.2889,0.3588/0.2754,0.8897/0.8300,0.2378/0.1573
2,two_hop,120,0.0250,0.4333,0.2963/0.2080,0.8661/0.7920,0.2222/0.0427
2,ood,90,0.0333,0.2556,0.1533/0.0536,0.8123/0.6858,0.1552/0.1264
3,all,180,0.0611,0.1167,0.0920/0.0694,0.2996/0.1350,0.2399/0.1573
3,two_hop,120,0.0250,0.1750,0.1282/0.0883,0.3219/0.1225,0.2336/0.0655
3,ood,90,0.0333,0.0889,0.0383/0.0000,0.1801/0.0498,0.1552/0.1264
verdict: SIGNAL — at least one (loop, split, signal) AUROC 95% CI LB >= 0.6 — T1.3 G1 gate PASSES; Phase 2 GOAT adjudication re-opens
json:{"model":"/Users/katopz/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf","untied":true,"queries":180,"loops":4,"acc_last":0.0611,"elapsed_s":45916.3}
```

acc columns are loop-invariant by design (the clean-state continuation decodes from the restored post-prefill state, so accuracy cannot depend on K — the instrument repair working as specified). `ood` = the bin's second-pool unseen-composition split (two pools × (60 two-hop + 30 one-hop), the Bench-930 shape; splits overlap).

## Interpretation

- **The DiscoLoop sign reproduces on the league model**: at a posture where the model can barely answer (2.5% two-hop), the k≤2 margin-alignment readout still ranks the rare correct answers above the wrong ones (LB 0.79–0.83) — alignment tracks correctness BEFORE the answer is emitted, the paper's core claim, in sign.
- **The two probe signals diverge at scale.** The n=4 smoke's cos_bridge 1.0000 was small-n noise; margin is the live axis on this checkpoint (cos_bridge flat 0.14–0.24 over 180 queries). Any future GOAT adjudication on this instrument should pre-register `auroc_margin` at k=2 as the primary signal, cos_bridge demoted to secondary.
- **Loops help the readout, not the model.** bridge_top1 decays monotonically with K (0.55 → 0.12 all-split) while margin-AUROC peaks at k=2 — past k=2 the stack iteration moves the head AWAY from the gold token. The loop intervention stays measurement-only (reinject is Phase 2, closed unbuilt).
- **Phase 2 re-open NOT taken on this record.** The verdict line's "Phase 2 GOAT adjudication re-opens" is the bar's formal branch, but a reinject GOAT over a 2.5%-accurate two-hop axis would measure the phrasing, not the loops. The lane's next read of record is the owner-gated `B929_PHRASE=standard` full run (issue T1.3); Phase 2 adjudicates after that lands SIGNAL.
- **Housekeeping:** the run worktree `/private/tmp/riir-infer-929` (the exact instrument source) was verified a strict subset of upstream HEAD (diff 2026-10-14 — the union `69f520e` + post-launch knobs supersede every line) and pruned after this record landed; the raw log/JSON above are the volatile-path preservation.
