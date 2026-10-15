# Bench: Plan 626 T0.2 — anchor over-firing telemetry (Gemma 2 2B IT, MATH-500, greedy)

**Date:** 2026-10-15 (UTC; M3 session, Plan 626 claim `116564ec1`)
**Plan:** [`626_swir_anchor_exit.md`](../.plans/626_swir_anchor_exit.md) (Phase 0 kill gate, T0.2)
**Instrument:** `riir-ai/crates/riir-engine/tests/probe_626_anchor_overfire.rs` (ignored probe; runs where the model lives — the Plan 313 bench precedent: instrument riir-ai-side, record katgpt-rs-side)
**Box state:** M3 Max, AC power, release build, isolated `CARGO_TARGET_DIR=/tmp/riirai-626`; GPU busy at 99% on the 4090 (plan435 chain) — this probe is M3-CPU-only, no GPU consumer. Latency figures are throughput-side notes, not gates.
**Model/posture:** `gemma-2-2b-it-f16.gguf` (riir-train/data), greedy (temperature 0 path = argmax), explicit-CoT only (no SwiR controller active — T0.2 asks for the ORDINARY stream), MATH-500 `data/math-500/test.jsonl` first 12 problems, max 320 steps/problem.

## Run parameters

| axis | value |
|---|---|
| problems | 12 (MATH-500 subset, indices 0–11) |
| decode steps total | 3438 |
| early terminations | 6/12 problems (all via `<end_of_turn>` 107 at steps 189–305); 6/12 hit the 320-step cap without terminating (the Gemma 2 2B rambling ceiling, consistent with Plan 313's G1 capability block) |
| runtime | ~17.7 min wall (1062.8 s) release |

## Token resolution (the arm's vocabulary facts, resolved by encode — never assumed)

| literal | resolution | note |
|---|---|---|
| EOS | id 1 (`tokenizer.eos_id()`) | |
| `<end_of_turn>` | id 107, single token | the natural Gemma 2 IT terminator |
| `</think>` | **UNRESOLVED — splits into [727, 27197, 235313]** | Gemma 2 IT carries no think-block vocabulary; the plan's DEFAULT minimal arm degrades on this model to `{EOS, <end_of_turn>}` |
| `\n` | id 108, single token | punctuation arm |
| `\n\n` | id 109, single token | punctuation arm (added after the first run showed the whitespace class dominating) |
| `.` | id 235265, single token | punctuation arm — NOT a small-id common piece; resolving by assumption would have been wrong (the founding run's report had period's count mislabeled under newline via an arm/candidate index misalignment, fixed with an arm-aligned read + `debug_assert`) |

## T0.2 telemetry — per-candidate argmax frequency (mid-stream = excluding the run's natural terminator)

| candidate | id | total hits | mid-stream | mid-rate | problems hit (of 12) |
|---|---|---|---|---|---|
| EOS | 1 | 0 | **0** | 0.000 | 0 |
| `<end_of_turn>` | 107 | 6 | **0** | 0.000 | 0 — all 6 hits WERE the terminators |
| `\n` | 108 | 54 | 54 | 1.57e-2 | 10/12 |
| `\n\n` | 109 | 121 | 121 | 3.52e-2 | 12/12 |
| `.` | 235265 | 148 | 148 | 4.31e-2 | 12/12 |

Top ambient argmaxes overall (share of 3438 steps): `" the"` 5.7%, `" "` 4.7%, `"."` 4.3%, `"\n\n"` 3.5%, `"  "` 3.2%, `"**"` 3.2% — the stream is dominated by whitespace/markdown formatting, as the plan's over-firing risk predicted.

## T0.2 read (numbers, per the plan's own wording)

1. **The minimal arm does not over-fire on this model/domain: ZERO mid-stream argmaxes for `{EOS, <end_of_turn>}` across 3438 ordinary explicit-CoT steps.** Both tokens appear only as a run's natural terminator.
2. **The punctuation arm is DISQUALIFIED with numbers**: `.` 4.31e-2, `\n\n` 3.52e-2, `\n` 1.57e-2 mid-rates — a boundary set carrying any of them collapses the anchor arm into a fixed min-dwell cap (the plan's second failure mode, now measured, not guessed). The T0.1 DEFAULT minimal arm and the exclusion of newline/period-class tokens are CONFIRMED as the right shape.
3. `</think>`'s absence on Gemma 2 is a MODEL fact, not a plan defect — on a thinking-format model the three-token arm applies; here the arm is two tokens and stays minimal.

## Scope limits (honest)

- T0.2 measures the ORDINARY explicit-CoT stream, exactly as the task words it. The anchor's live firing surface is the LATENT span (soft-embedding steps, different argmax distribution) — that surface is T0.3/T0.4's business, driven through the controller with simulated anchor exits. Under-firing IN LATENT MODE is not measured here and is NOT settled by this record.
- 6/12 problems never terminated (capability ceiling) — their steps are censored at 320, which if anything UNDERCOUNTS boundary hits near natural ends; the 0-mid-stream minimal-arm result is conservative in the arm's favor... (a censored run cannot produce a late terminator hit, but it also cannot produce MORE mid-stream minimal-arm hits — the count stands at 0 regardless).
- n=12 problems / 3438 steps is a probe sample, not a population — the kill gate (T0.5) weighs this beside T0.3/T0.4.

## Verdict for T0.2

**PASS — minimal arm survives over-firing telemetry on this model/domain (0/3438 mid-stream); punctuation arm disqualified with measured rates.** Phase 0 proceeds to T0.3 (intent agreement, non-self-answering oracle) + T0.4 (outcome-based kill arm), which measure the LATENT-span surface the anchor actually consults.
