# Bench 941: Plan 626 T0.3 + T0.4 — anchor agreement + outcome kill arms (Gemma 2 2B IT, MATH-500, greedy, LATENT surface)

**Date:** 2026-10-16 (UTC; M3 session, Plan 626 claim `116564ec1`)
**Plan:** [`626_swir_anchor_exit.md`](../.plans/626_swir_anchor_exit.md) (Phase 0 kill gate, T0.3 + T0.4 + T0.5)
**Status:** T0.3 MEASURED — **KILL GATE CLOSED-NEGATIVE on B1 (under-firing)**; T0.4 not run (dispositive per the pre-registered verdict rule)
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

| axis | value |
|---|---|
| problems | 12 (MATH-500 indices 0–11, the Bench 940 subset) |
| latent trace | 256 steps/problem, pure soft-embedding feed (controller inactive), 3072 latent steps total |
| oracle window | W = 40 explicit greedy steps after materialization |
| materializations | ≤5/problem: anchor + masked (on fire) + shuffled ×2 + punct ×1 |
| runtime | 5434 s wall (90.6 min) release, M3-CPU |
| instrument fix mid-session | the soft-embedding scratch must be ZEROED before each `soft_embedding` call (it ACCUMULATES — the mod.rs host contract); the first launch's trace was poisoned past step 0 and the replay assert caught it (`riir-ai f2dc96b28`) — the validity check worked as designed |

## Per-problem latent-trace results

| problem | anchor fires | first fire step | punct fires | punct first step |
|---|---|---|---|---|
| 0 | 0 | — | 24 | 13 |
| 1 | 0 | — | 35 | 15 |
| 2 | 0 | — | 15 | 14 |
| 3 | 0 | — | 11 | 20 |
| **4** | **4** | **212** | 28 | 19 |
| 5 | 0 | — | 10 | 14 |
| 6 | 0 | — | 10 | 10 |
| 7 | 0 | — | 32 | 9 |
| 8 | 0 | — | 7 | 13 |
| 9 | 0 | — | 20 | 9 |
| 10–11 | 0 | — | (punct fires early, as every problem) | 9–13 |

Punct fired on **12/12 problems** (first fire step 9–20); the minimal arm fired on **1/12**.

## The one anchor fire (problem 4) — the signal is REAL where it exists

- Fire at latent step 212, readout `<end_of_turn>`(107), entropy 1.78 vs trace mean 1.50.
- **anchor materialization** (fed 107): the oracle's FIRST window token was a `B_min` member — agreement at position 1.
- **masked materialization** (fed the second-choice token 235248 with `B_min` masked out): the model emitted `<end_of_turn>` within the window ANYWAY (position 1) — the end-intent was in the CONTEXT, not the conditioning. The strongest agreement form the probe can express.
- Shuffled windows on the same problem: no `B_min` member in either (their punct hits ride the ambient punctuation rate).

So the ONE latent-space fire this model produced was a genuine end-moment. The verdict is **signal absent, not signal fake** — the plan's first designed failure mode (the proxy is empty; decline is a correct answer).

## T0.3 aggregates (the bar inputs)

| bar | measured | bar | verdict |
|---|---|---|---|
| **B1 under-firing** | **1/12 problems fire** | ≥ 3/12 | **FAIL** |
| **B2 agreement** | anchor 1.00 (n=1) · shuffled baseline 0.00 (n=24) | ≥ 0.50 AND ≥ 2× baseline | PASS-shaped at n=1 — uninformative at this fire rate |
| (reported) masked | 1.00 (n=1) | no bar | context-intent confirmed on the single fire |
| **B4 instrument** | punct arm 1.00 (n=12) vs its shuffled baseline 0.75 (n=24) → ratio **1.33×** | must be < 2× | **PASS — the instrument discriminates** (the disqualified arm fails the ratio bar as designed; the probe is not blind) |
| context | punct fires 12/12 problems · mean entropy 1.0991 overall vs 1.7825 at the anchor fire | — | the fire sat at an elevated-entropy moment |

## Qualitative observations (for the next attempt, if any)

1. **The latent readout degenerates fast.** First punct fire lands at step 9–20 on every problem; shuffled materializations from mid-trace steps decode coherent continuations (the context is intact) while LATE-trace latent steps drift into token loops (`" а а а а …"` — the SwiR paper's documented pure-latent degradation, present here). The readout stream is dominated by markdown/whitespace even more than the explicit stream (Bench 940's top-argmax finding).
2. **The fire that existed was deep and real** (step 212, elevated entropy, masked-arm-confirmed intent) — consistent with the trained-form hypothesis (format-activation SFT is what makes this signal frequent; HybridAgent's anchor works because it TRAINS the projection). A frozen IT model without think-block vocabulary has essentially no termination vocabulary in its latent readout.
3. **Robustness of the verdict to the surface choice:** production anchor exits would consult readouts inside CONTROLLER-composed latent spans (bounded by c_max=64, entropy-driven switches). All observed minimal-arm activity sat ≥ step 212 — deeper than any production span. The under-firing conclusion is therefore STRONGER on production surfaces, not weaker.

## T0.4 — not run (dispositive)

The pre-registered verdict rule (PASS iff B1 ∧ B2 ∧ B3 ∧ B4) is already false at B1. With a 1/12 fire rate, the anchor arm of T0.4 would be 11/12 `ForcedCap` exits + 1 anchor fire — the outcome comparison would measure forced-cap vs entropy behavior, not anchor-vs-entropy moments. Running it would spend ~70 min of M3 CPU on near-zero information about the anchor. The instrument is LANDED and PRE-REGISTERED (`probe_626_t04_anchor_outcome.rs`) and re-runs unchanged on a thinking-format model, where the fire rate is the first thing to re-measure.

## Verdict for T0.5

**CLOSED-NEGATIVE on B1 (under-firing: 1/12 vs the ≥3/12 bar).** B4 passed (the instrument discriminates — the negative result is trusted); B2 was PASS-shaped but uninformative at n=1; the masked arm's single data point shows the rare signal is genuine. **The anchor ARM demotes inert; entropy-exit remains the only exit law** (the plan's own demote path: 0%-fire ⇒ demote). Phase 1/2 never open on this model class. Re-open conditions (recorded, not scheduled): a thinking-format model (Qwen3-class, native `</think>`) re-runs T0.2→T0.3 unchanged — the instruments and bars are reusable as landed; a positive fire rate there opens the controller-extension phase with the same bars.
