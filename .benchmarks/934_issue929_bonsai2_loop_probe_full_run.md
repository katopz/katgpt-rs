# Bench 934 — Issue 929 T1.2b Bonsai-2-27B (PQ2_0) loop-alignment probe, the FULL adjudication run — M3, chat no-think posture

**Numbering note:** allocated 934 after a same-day INDEPENDENT dual allocation of 933 (`933_…_smokes.md`, the 4090-idle session's three smoke postures, `c9c9c0f3e`) — two sessions, two stems, one number; the unpushed side (this file) renumbered per the renumber-never-reuse law. This record is the FULL-RUN companion to [Bench 933](933_issue929_bonsai2_loop_probe_smokes.md): their smokes prove the axis alive and root-cause the OOM; this run is the T1.3/T1.4 adjudication attempt.

**Status:** RUN EXECUTING (launched 2026-10-11 12:46 +07, detached; log `/tmp/b929_full.log`, JSON `/tmp/b929_full.json`) — verdict + tables land at completion; the record skeleton + instrument-repair disclosures are committed NOW to pin the number (highwater bumped same commit). Early pace ~280 s/query incl. the 164 s load (q0–q2 observed) → ETA slides past the 8 h estimate; actuals land with the results. **Posture vs Bench 933's protocol note:** theirs recommends CHAT + 384-tok think budgets (~21 h); THIS run is the no-think posture (system + empty-think prefix, 8-tok answers, ~8 h) — both measured, cross-referenced; the phrasing-inversion caveat rides BOTH records.

**Concurrent-landing reconciliation (Issue-825 class, resolved by union):** while this run was in flight, the 4090-idle session landed `61fc36f` (bounded HybridCache KV — a measured >100 GiB OOM on the unbounded ctor — plus their own B929_CHAT framing variant, bare scaffold, think-then-answer at ~384-tok budgets) and this session's `d633fbb` (clean-state continuation + no-think chat posture + B929_SKIP) stranded on a diverged local develop. Reconciled as **`69f520e`** (the union, pushed fast-forward from a detached `61fc36f` worktree): bounded KV + clean-state continuation + the no-think posture as `B929_CHAT=1`. **The in-flight run's instrument == the union's chat posture byte-for-byte in behavior** (same template string; the KV bound is allocation-only, zero numerical effect) — its numbers record for the union instrument. The bare-scaffold/think-budget variant stays recorded in the issue as the alternative elicitation (~420 s/q), not this run's posture.

**Date:** 2026-10-11 · **Issue:** [929](../.issues/929_loop_alignment_probe_decode_reinject_poc.md) · **Research:** 614 (DiscoLoop, arXiv:2607.00341)
**Runner:** `riir-infer` bin `issue929_bonsai2_loop_probe` (`--features issue929_bonsai2_probe`), detached worktree at riir-infer `32fd838` + two instrument repairs recorded below (sibling WIP in the main checkout excluded by construction — the measurement tree is HEAD + the bin edit only)
**Box:** M3 Max, AC power, 16 cores; concurrent sibling cargo sessions present (load 4–8 of 16) — decode-QUALITY measurement (argmax identities + cosine readouts + AUROC), latency-free by construction; wall time disclosed anyway.

## Verdict

<!-- FILLED AT RUN COMPLETION -->

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

<!-- CSV grid + verdict line, pasted at run completion -->

## Interpretation

<!-- FILLED AT RUN COMPLETION -->
