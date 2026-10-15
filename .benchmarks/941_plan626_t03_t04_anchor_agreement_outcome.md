# Bench 941: Plan 626 T0.3 + T0.4 — anchor agreement + outcome kill arms (Gemma 2 2B IT, MATH-500, greedy, LATENT surface)

**Date:** 2026-10-16 (UTC; M3 session, Plan 626 claim `116564ec1`)
**Plan:** [`626_swir_anchor_exit.md`](../.plans/626_swir_anchor_exit.md) (Phase 0 kill gate, T0.3 + T0.4 + T0.5)
**Status:** PRE-REGISTRATION LANDED — probes not yet run (this section is the binding bar record)
**Instruments:** `riir-ai/crates/riir-engine/tests/probe_626_t03_anchor_agreement.rs` + `probe_626_t04_anchor_outcome.rs` (ignored probes; the Plan 313 split — instrument riir-ai-side, record katgpt-rs-side)
**Box state:** M3 Max, AC power, release build, isolated `CARGO_TARGET_DIR=/tmp/riirai-626`; 4090 GPU busy (plan435/436 chain, 100%) — M3-CPU-only probes, no GPU consumer. Latency figures are throughput-side notes, not gates.
**Model/posture:** `gemma-2-2b-it-f16.gguf` (riir-train/data), greedy (temperature 0 = argmax, repetition penalty 1.0 = off — Issue 1075 Defect A does not bite this posture), MATH-500 first 12 problems (same subset as Bench 940), explicit-CoT prompt identical to T0.2.

## Pre-registration (WRITTEN BEFORE THE RUN — binding)

The arms, semantics, and bars below were committed to the plan and this record **before any T0.3/T0.4 execution** (commit lands before the probe binaries are ever run). Bars once run are adjudicated as written; no post-hoc re-barring.

### T0.3 semantics (intent agreement, non-self-answering oracle)

- **Surface:** PURE-LATENT trace per problem — soft embeddings fed at every step (controller inactive), cap `PROBE626_TRACE_STEPS=256`. The anchor's live firing surface is the latent span (T0.2's scope note); the latent readout sequence is the implicit readout.
- **Anchor exit:** first step `t ≥ 4` (min_dwell) whose readout ∈ `B_min`. All boundary-hit steps recorded (fire multiplicity is under-firing evidence).
- **B_min on Gemma 2:** `{EOS(1), <end_of_turn>(107)}` — `</think>` unresolved (splits to 3 subwords; Bench 940's model fact).
- **Oracle (non-self-answering):** materialize at the exit by feeding the boundary token explicitly (the T1.6 production semantic), then greedy-decode window `W=40`. Agreement = a `B_min` member among the decoded tokens (steps exit+1..exit+W). The exit step's own readout is excluded by construction.
- **Three materializations per anchor fire:**
  1. `anchor` — feed the boundary token. (Production materialization semantics.)
  2. `masked` — feed the argmax over probs with `B_min` masked out (the second-choice token). Same window check. Context-intent evidence: if the model emits a boundary within W even when NOT fed one, the end-intent lives in the context, not the conditioning.
  3. `shuffled` ×2 — feed the plain argmax at seeded random steps (`fastrand` seeded per (problem, trial); steps in [min_dwell, trace_len), collisions with t_a shift +1). Chance baseline; each shuffled window is checked against BOTH `B_min` and the punctuation set, giving the punctuation arm its baseline for free.
- **Punctuation arm (negative control):** `B_punct = {`.`, `\n`, `\n\n`}` (the T0.2-disqualified set, ids 235265/108/109). First punct fire at dwell ≥ 4 → materialize the punct token, window check against `B_punct`. Must FAIL B2's ratio form against its own shuffled baseline — else the instrument is blind (B4).

### T0.4 semantics (outcome arms)

- Three arms per problem, sequential fresh runs (deterministic greedy → the latent prefix is shared by construction until each arm's exit):
  1. `anchor` — pure latent to fire-or-cap(320); fire → materialize the boundary token; no fire → forced exit at cap feeding the plain argmax (recorded `ForcedCap`). Explicit decode ≤192 to natural terminator (EOS or `<end_of_turn>`) or cap.
  2. `entropy` — the REAL `SwiRController` at the Plan 313 tuned posture (`w_e_to_l=32, c_max=64, w_l_to_e=0`), driven from its Latent init; exit = the FIRST Latent→Explicit transition (entropy dips below the locked reference); the transition step's argmax is fed explicitly (production semantics), then explicit decode to natural end or cap. Post-exit the probe takes over the decode loop (controller guards/inject queue not exercised) — disclosed divergence from the full Plan 313 harness, whose role here is exit-timing reference.
  3. `random` — matched-step control: pure latent; exit at a seeded random step in [8, 200); feed the plain argmax there; explicit decode to natural end or cap.
- **Scoring = EXPLICIT EMISSIONS ONLY** (riir-ai Issue 1075 Defect B: latent pseudo-argmaxes never enter the scored text). The fed exit token + every explicitly decoded argmax form the response text; latent-step readouts never do.
- **Metrics:** correctness (`extract_boxed` + `is_equivalent`), parseable rate (a `\boxed{}` exists), natural-termination rate, total steps (latent + explicit), exit step + kind.

### Bars (T0.5)

| bar | statement | FAIL meaning |
|---|---|---|
| **B1 under-firing** | minimal arm fires on ≥ 3 of 12 problems | proxy empty in latent mode on this model |
| **B2 agreement** | fed-boundary agreement ≥ 0.50 AND ≥ 2× shuffled-control agreement | materialized exits are not coherent terminations above chance |
| **B3 outcome** | anchor correctness ≥ entropy correctness AND anchor mean total steps ≤ entropy mean total steps | anchor exits at worse-than-entropy moments |
| **B4 instrument** | punctuation arm agreement < 2× its own shuffled baseline | instrument blind → whole probe VOID, re-runs after repair |

**B3 caveat (pre-registered):** the correctness axis is capability-blocked on Gemma 2 2B (Plan 313 G1 = 0%; Bench 940 saw 6/12 rambles to cap). If both arms score 0, B3's correctness leg is vacuous-informative and any PASS carries `correctness-axis: PROVISIONAL` — decisive weight on B1+B2; the correctness half re-gates on a thinking-format model (Qwen3-class).

**Verdict rule:** T0.5 PASS iff B1 ∧ B2 ∧ B3 ∧ B4. Any FAIL → plan CLOSED-NEGATIVE naming the bar; entropy-exit remains the only exit law.

## Run parameters

(filled after the run)

## Results

(filled after the run)

## Verdict for T0.5

(filled after the run)
