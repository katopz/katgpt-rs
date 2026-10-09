# Bench 930 — Issue 929 T1.2 second independent checkpoint run (n=180 fixture, two α arms) — CONCURS with Bench 927: UNDECIDABLE

**Date:** 2026-10-10 · **Issue:** [929](../.issues/929_loop_alignment_probe_decode_reinject_poc.md) · **Research:** 614 (DiscoLoop, arXiv:2607.00341)
**Runner:** `examples/issue929_kimi_loop_alignment_probe.rs` (`--features kimi_k3_loader,loop_alignment_probe`)
**Companion record:** [`.benchmarks/927_issue929_loop_probe_checkpoint_run.md`](927_issue929_loop_probe_checkpoint_run.md) — an INDEPENDENT session's runner on the same checkpoint, same day, same UNDECIDABLE verdict. Two instruments, one answer: the label axis is dead on kimi-k3-0.40B.
**Box:** M3 Max, AC power, census load present (Issue 920 T1 capture, nice-19) — decode-QUALITY measurement (argmax identities + cosine readouts), latency-free by construction; wall time disclosed anyway.

## Verdict

**UNDECIDABLE — concurs with Bench 927.** 180/180 fixture queries answered WRONG in both loop arms; `bridge_top1 = 0` everywhere; **every AUROC is NaN** because the correctness class is empty (AUROC undefined on a single class — the honest reading, never folded into a number). The T1.4 kill bar presupposes both classes; it never computes. What this run ADDS to 927: n=180 at fixture scale (vs 16-item smoke postures), the α=0.5 reinject arm, and the provenance suite pinning the root cause.

| arm | loop | split | n | acc | bridge_top1 | auroc cos-argmax | auroc margin | auroc cos-bridge |
|---|---|---|---|---|---|---|---|---|
| 0 reentry (α=0) | 0 | all | 180 | 0.0000 | 0.0000 | nan | nan | nan |
| 0 | 0 | two_hop | 120 | 0.0000 | 0.0000 | nan | nan | nan |
| 0 | 0 | ood | 90 | 0.0000 | 0.0000 | nan | nan | nan |
| 0 | 1 | all | 180 | 0.0000 | 0.0000 | nan | nan | nan |
| 0 | 1 | two_hop | 120 | 0.0000 | 0.0000 | nan | nan | nan |
| 0 | 1 | ood | 90 | 0.0000 | 0.0000 | nan | nan | nan |
| 1 reinject (α=0.5) | 0 | all | 180 | 0.0000 | 0.0000 | nan | nan | nan |
| 1 | 0 | two_hop | 120 | 0.0000 | 0.0000 | nan | nan | nan |
| 1 | 0 | ood | 90 | 0.0000 | 0.0000 | nan | nan | nan |
| 1 | 1 | all | 180 | 0.0000 | 0.0000 | nan | nan | nan |
| 1 | 1 | two_hop | 120 | 0.0000 | 0.0000 | nan | nan | nan |
| 1 | 1 | ood | 90 | 0.0000 | 0.0000 | nan | nan | nan |

Fixture: paper §2 shape (250/pool entities, |R|=50 shared roles, out-degree 10, 12-fact windows, 120 unique-answer two-hop + 60 one-hop controls). K=2 caller-side loop at the answer position on fresh runtimes per (query, arm); greedy 8-token scoring. Wall 989.7 s.

## Runner-validity provenance (checked BEFORE recording the negative)

1. **Forward port byte-faithful** — `g1_logits_match_pytorch_reference` PASS on this box: our BOS top-8 (420/289/108263/559/39058/1581/799/8629 = `ith/is/ Wise/ as/ Prepare/ inter/ all/ It's`) is byte-identical to the committed python reference `ref_logits_bos.npy` (logit values match to 1e-3).
2. **Checkpoint = the published artifact** — `data/kimi-k3-0.40b/model.safetensors` sha256 `a39b7ed2769ee2f9891a19cef6f1ca7986a3f295c09759b5b3135285e5d7a678`, exactly the Bench-889 pin.
3. **Tokenizer round-trips** — encode→decode identity verified on every probe prompt shape.
4. **NLL probe** — teacher-forced NLL per token: 12.389 / 12.164 / 12.266 on three generic sentences vs **uniform log(163840) = 12.007**. The text head carries ≈ zero likelihood structure. Continuations are fluent-style but content-free (`"The meaning of life is" → " Inigo are too. Prepare to die."`).
5. **Untied head verified** — same-token cos(embed_row, lm_head_row) ≈ −0.007 mean over a 31-row sample (structurally independent tables): the probe's head-row reads are the true readout operator (the issue's tied-embeddings precondition resolved).
6. **Chat framing also tried** — the `<|im_system|>`/`<|im_assistant|>` special ids sit ABOVE the text `vocab_size` (163840); plain completion is the intended mode, and chat-style 3-shot prompts change nothing.
7. **In-repo corroboration** — Issue 584 / Bench 025 recorded the same fixture-class finding from a different instrument: "Kimi-K3-0.40B attention is too uniform for meaningful quality results" (the 395M model's retrieval patterns are near-uniform).

## Root cause & disposition

**The checkpoint's text pathway is content-untrained** (style fluent, semantics chance-level). This is a FIXTURE problem, not a mechanism problem: the probe mechanism (T1.1, `5a8c190b4`) and the fixture generator are landed, tested, and stay; Issue 929's checkpoint-era verdict is **UNDECIDABLE** (the probe question is unanswerable here, not answered-no). **Reopen trigger: riir-train Plan 451's loop-trained artifact** — any checkpoint with a trained text head re-arms the probe immediately (one command per runner). Phase 2/3 (injection, halt-on-readiness) closed UNBUILT — no injection on a substrate with no correctness axis (the 0-for-3 lesson, honored).
