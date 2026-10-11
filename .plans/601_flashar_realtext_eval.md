# Plan 601 — Real-Text D2F Eval: the Plan 600 promotion precondition (text-trained D2F + token-level NLL/exact-match gates)

**Status:** COMPLETE + GOAT GREEN 2026-09-17 — all gates pass on real text ([Bench 601](../.benchmarks/601_flashar_realtext_goat.md)); Plan 600's promotion bar (T8 all-green + T9 on real text) is MET → **confidence-commit (κ=0.9 + floor) PROMOTED as the anchor-then-fill seam's decode default** via `ConfidenceAnchorConfig::default()`; the strided entry stays as the no-floor comparator.

## Why (one paragraph)

Plan 600's acceptance demanded "T8 all-green + T9 on real text" and its
2026-09-17 resolution recorded the blocker honestly: this repo's dllm lane
was pattern-only (`generate_pattern_dataset`'s analytic [a, b, a, b] family),
so the bar could not be met and promotion was deferred. This plan grows the
missing instrument: a committed public-domain English corpus fixture, a
char-level tokenizer, a text-trained D2F via the SAME `train_mini_dllm` the
pattern lane uses, and the Plan-600 gate list re-measured on held-out real
text — plus a real-text T9 (the analytic pattern law has no counterpart; the
law is estimated nonparametrically from held-out corpus counts).

## Tasks

- [x] T1 Corpus fixture — Jane Austen, *Pride and Prejudice* tail (~107k chars,
      novel body only, chapter-aligned; Project Gutenberg eBook #1342, public
      domain in the US) at `tests/data/austen_pride_prejudice_tail.txt`, with
      provenance header.
- [x] T2 `src/dllm/text_corpus.rs` (policy-free substrate): `TEXT_CORPUS`,
      31-symbol compact-alphabet char tokenizer (`encode_text` — lowercased,
      apostrophes kept incl. U+2019, everything else folds to space),
      `slice_blocks` (panics short — a silent short eval set would shrink the
      paired-Δ resolution G1 asserts), `bigram_counts` / `bigram_law_smoothed`
      / `unigram_entropy_nats`. 4 unit tests.
- [x] T3 `Config::micro_dllm_text()` (katgpt-types) — 32-token vocab, 2 layers
      of d=32, bidirectional, mask 31, same 8-token D2F block (protocol parity
      with the pattern lane).
- [x] T4 `evaluate_masked_nll` (dllm) — teacher-forced masked NLL in
      nats/token; deliberately re-implemented against the forward logits
      rather than reusing the training loss (the eval instrument must not
      silently inherit the training loss's averaging conventions).
- [x] T5 `tests/bench_601_flashar_realtext_goat.rs` (feature `flashar_anchor`,
      release posture documented) — T1 corpus honesty (held-out masked NLL
      2.436 < unigram 2.886 nats, margin 0.45 > 0.15 bar; argmax acc 0.300 vs
      chance 0.032), T2 G1/G2/liveness (below), T3 T9′ KL vs the empirical
      bigram joint + the model-side UGC certificate printed (Ĉ = 5.93).
- [x] T6 Measured verdict — ALL GREEN (512 held-out blocks, paired seeds,
      production entries only):
      - G1 paired Δ(acc) = +0.0344 / +0.0686 / +0.1106 at κ 0.5 / 0.9 / 0.99
        (SE ≤ 0.005 ≤ 0.05 bar) — not merely non-inferior: the incumbent's
        no-floor path leaves its first slot masked on ~100% of text blocks
        (leak ≈ 1.0) and collapses to 0.007–0.013 acc, while the floor arms
        commit and land 0.047–0.117.
      - G2 fill steps 2.87 / 5.18 / 6.79 vs 7.94 / 8.00 / 8.00
        (1.2–2.8× better); wall ratio 0.72× / 0.87× / 0.91× (≤ 1.25× bar).
      - Liveness: all-mask baseline 0.000 acc — anchors carry the signal.
      - T9′ realized KL ratio 0.872 / 0.669 / 0.483 (κ 0.5 / 0.9 / 0.99,
        mean of 2×4096 MC seeds vs the add-0.5 empirical bigram joint) — the
        greedy reveal distorts the real-text block law LESS than the
        incumbent at every cell; Caveat #1 empirically benign on text.
      - G3 parity + G4 alloc-free: corpus-independent seam properties, pinned
        by `src/speculative/flashar_anchor.rs` parity tests + Bench 600's
        alloc gate (both green in the same lane, this session).
- [x] T7 PROMOTION (Plan 600 acceptance pre-authorizes on green): add
      `impl Default for ConfidenceAnchorConfig` (κ 0.9 + floor, the balanced
      GOAT cell) in katgpt-forward; update the Cargo feature comment,
      `.docs/02_inference/speculative_decoding.md` §Confidence-Commit, and the
      `.docs/09_feature_catalog` row. `AnchorConfig::default()` (stride 2) is
      unchanged; `anchor_then_fill` keeps its signature (zero caller churn).
- [x] T8 Housekeeping: `.plans/.highwater` 600→601; validate with
      `numbering_gate` + `dual_allocation_gate`; clippy 0 warnings on touched
      targets (cargo heal first, 1 manual remainder); bench_600 + alloc gate +
      parity green; default-features check green.

## Acceptance

The Plan 600 promotion precondition is CLOSED: the gates ran on real text
(real English prose, text-trained model, held-out windows) and passed. Any
future demotion follows the standing demote-on-loss rule — a re-gate on a
harder corpus (multi-sentence context, larger vocab/BPE) that shows a G1 loss
demotes the default back to stride.

## Honest scope notes

- The text model is micro (~26k params, 31-symbol alphabet, 8-token blocks,
  18k-char train window). "Real text" here = real prose with a real
  non-analytic conditional law — NOT a production-LM-scale claim. The gate's
  meaning is rule-vs-rule under a matched corpus, which is what transfers.
- The G1 win axis on text is termination-dominated (masked slots score zero
  exact-match). That is the same axis the DBTM floor exists for and is the
  honest reading; it is NOT a claim that confidence selection finds better
  tokens than the AR walk's own proposals.
- The strided incumbent's text collapse (leak ≈ 1.0 at τ ≥ 0.5) was never
  visible on the pattern corpus because toy confidences are high. Recorded in
  Bench 601 — it is itself a finding about the incumbent.
- `bench_e_stability_profile` (TileRT, unrelated) failed once under the
  concurrent-cargo load of the full `--features flashar_anchor` test sweep on
  this box and passed alone immediately after — the AGENTS-documented
  transient class, not adjudicated as a regression.
