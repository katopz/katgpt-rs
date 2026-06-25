# Issue 009: AC-Prefix × Engram × Latent Field Steering — Compute-Substrate Fusion Gap

**Date:** 2026-06-26
**Status:** **CLOSED — negative Super-GOAT verdict** (2026-06-26). The design decision is resolved by evidence, not by user choice: the fusion is not realizable without negative-ROI infrastructure investment. AC-Prefix stays shipped (default-on, GOAT-passed, modelless-G1-corrected per Issue 003); the fusion is deferred *sine die*. See Resolution below.
**Origin:** Issue 002 (AC-Prefix Super-GOAT gate) — surfaced during integration-surface audit
**Severity:** Blocking for Issue 002 (cannot write an honest implementation plan without resolving this)
**Related:** `katgpt-rs/.issues/002_ac_prefix_super_goat_gate.md`, `katgpt-rs/.research/295` §2.4 (fusion table), `katgpt-rs/.plans/313` (AC-Prefix), `katgpt-rs/.plans/299` (Engram), `katgpt-rs/.plans/309` (Latent Field Steering), `riir-ai/.plans/314` (BoM G2 arena precedent), `riir-ai/.plans/329` (QuestFunctor — the real Engram integration)

## The problem

Issue 002 asks: *does the AC-Prefix × Engram × Latent Field Steering fusion deliver a measurable quality win over Engram × Latent Field Steering at iso-compute on a real game-AI workload?*

An integration-surface audit (2026-06-26) reveals that the three primitives operate on **incompatible compute substrates**. There is no shared forward pass into which all three can be composed without first making a non-trivial design decision. The §2.4 fusion table in Research 295 framed the fusion at the *conceptual* level ("three conditioning signals, one forward pass"); at the *implementation* level the three signals live on three different compute graphs.

## Evidence — what each primitive actually consumes/produces

| Primitive | Input | Output | Compute substrate | Integration point (verified) |
|---|---|---|---|---|
| **AC-Prefix** (`AcPrefix`) | `&[u32]` token sequence + `&[usize]` conditioning positions | per-position logprobs via `ForwardForAcPrefix::forward_for_ac_prefix` returning `Vec<f32>` of length `augmented_tokens.len()` | **Causal Transformer forward pass** (token → embedding → attention with three-region mask → lm_head log-softmax). The primitive is a mask builder + sequence augmenter; it does nothing without a Transformer to apply the mask. | **Not wired anywhere in riir-ai** (grep `AcPrefix\|ac_prefix\|ForwardForAcPrefix` in `riir-ai/**/*.rs` → 0 matches). The trait `ForwardForAcPrefix` has zero implementors outside katgpt-core tests. |
| **Engram** (`fuse_into_hidden_state`) | `&[CanonicalId]` token-ids → hash keys → `K_MAX × D` latent slot vectors | additive residual into hidden state `[f32; D]` | **Latent-to-latent fusion kernel** (RMSNorm · dot · sigmoid gate). No Transformer, no token decode. | **Wired in `riir-games/quest_grammar/quest_functor.rs` (Plan 329)** as part of the QuestFunctor. `QUEST_FUNCTOR_D = 8` (HLA-matching). The QuestFunctor's docstring is explicit: *"no token decode, no softmax, sigmoid basis. The entire `propose()` call is `f32 → f32`"*. The compute is `katgpt_core::funcattn::funcattn_forward` — a **closed-form Tikhonov solve**, NOT a Transformer. |
| **Latent Field Steering** (`apply_latent_steering`) | unit-norm direction `Vec<f32>` (d≤64, HLA d=8) + strength α | additive overlay `s' = s + α·v` on `&mut [f32; 8]` HLA slice | **Element-wise SAXPY** on the post-evolve HLA slice. No Transformer, no tokens. | **Wired in `riir-engine/src/latent_field_wiring.rs` (Plan 309 T5.1/T5.3)** via `FieldRegistry` + `FactionStanceRegistry` + `apply_all`. Operates on `ReconstructionState::hla_mut()` after `evolve_hla`. |

**The incompatibility:** AC-Prefix's value proposition (single-pass `p(xe | xc)` over an augmented token sequence with a leakage-free three-region mask) **requires a causal Transformer forward pass**. The other two primitives operate on `f32` hidden-state slices with no Transformer in the loop. There is no shared forward pass.

## The three honest design directions

Resolving this requires picking one of three directions. Each has real costs.

### Direction A — Add a Transformer to the QuestFunctor path (scope creep)

Fuse at the QuestFunctor: replace/augment the FUNCATTN closed-form solve with a causal Transformer forward that takes the Engram-retrieved pattern as an AC-Prefix conditioning set, with Latent Field Steering injected as a direction vector.

- **Cost:** Plan 329's headline contract is *"latent-to-latent, no token decode, no softmax"*. Adding a Transformer + lm_head log-softmax violates this. The QuestFunctor's selling point (10× over SDPA via closed-form solve per Bench 058 G2) is replaced by an iterative Transformer — likely a perf regression at crowd scale.
- **Verdict:** rejected on first principles. This is scope creep that breaks Plan 329's design.

### Direction B — Build a new game-AI workload with a Transformer in the loop

Build a Plan-314-style arena where a causal Transformer IS in the game-AI hot path, and where "conditioning on a known future outcome" is semantically meaningful. This is the issue's own prerequisite #3. Candidate workloads:
- **Hindsight policy evaluation** (offline replay): "given the NPC died at tick T, what's the conditional likelihood of the action sequence that led there?" — AC-Prefix conditions on the known death.
- **Counterfactual curiosity** (online): "what would the NPC have done if the player had gone the other way?" — AC-Prefix conditions on the counterfactual future.
- **Dreamer-style rollout conditioning** (planning): "given a hypothesized future reward trajectory, sample behavior." — AC-Prefix conditions on the hypothesized trajectory.

- **Cost:** This is essentially "design and build a new benchmark harness from scratch." Plan 314 took 9 tasks + 5 design iterations to find the winning regime for BoM (the LeakyIntegrator winner-take-all observation encoding). AC-Prefix's "known future outcome" semantics are *harder* to design a winning regime for — the workload must be one where conditioning on the future is both meaningful AND where the conditional likelihood/sampling actually changes downstream behavior measurably.
- **Verdict:** honest, but this is a multi-phase research-and-build effort, not a single plan. Comparable to or larger than Plan 314.

### Direction C — Reframe AC-Prefix as a latent-space operator (new primitive)

Extend `AcPrefix` to operate on hidden-state sequences rather than token sequences — a "latent AC-Prefix" that builds the three-region mask over latent vectors and applies it inside an attention layer's Q/K/V compute, not at the token level.

- **Cost:** This is a **new primitive**, not the shipped `AcPrefix` (Plan 313). The shipped primitive's GOAT gate (G1 buffer bit-identical, G2 27.258× speedup vs iterative-MLM, G3 empty-prefix bit-identical, G4 alloc-free) was measured on the *token-level* primitive. A latent-space variant would need its own plan, its own GOAT gate, and its own Super-GOAT question. It is arguably not "AC-GPT" anymore — AC-GPT's load-bearing insight is the *token-copy-with-original-position* discipline that prevents multi-layer leakage through RoPE rotations; latent vectors don't have positions in the same sense.
- **Verdict:** this is a legitimate research direction but it's a *different* Super-GOAT question than Issue 002 asks. Issue 002 asks about the *shipped* AC-Prefix primitive; Direction C asks about a hypothetical latent variant.

## Why this is not an implementation gap I can unblock

The §3.5 modelless-unblock protocol (research SKILL.md) covers the case where a gate *appears* to need training but might be passable modellessly via freeze/thaw, raw/lora hot-swap, or latent-space correction. **This is not that case.** The gap here is not "the gate needs training" — it's "the three primitives have no shared compute graph, and choosing which graph to fuse them on is a design decision with substantive tradeoffs."

Each direction (A/B/C) is a legitimate path, but they lead to *different* Super-GOAT questions:
- A: "Does AC-Prefix improve QuestFunctor quest-proposal quality?" (probably no — perf regression, contract violation)
- B: "Does AC-Prefix × Engram × Steering improve a Transformer-in-the-loop game-AI workload?" (the original Issue 002 question, but requires building the workload)
- C: "Does a *latent* AC-Prefix variant improve hidden-state conditioning?" (a new question, not Issue 002)

## Recommendation

**Direction B is the only honest path that answers Issue 002's actual question**, but it requires:
1. A workload design phase (pick hindsight-eval vs counterfactual-curiosity vs dreamer-rollout).
2. A baseline harness (Engram × Latent Field Steering on the chosen workload, WITHOUT AC-Prefix) — this is itself a Plan-314-scale effort because no such harness exists today.
3. The treatment (add AC-Prefix via a `ForwardForAcPrefix` impl over the workload's Transformer).
4. An iso-compute comparison with a quality metric.

This is **two plans minimum**: one for the baseline harness (Engram × Latent Field Steering on a Transformer-in-the-loop game-AI workload), one for the AC-Prefix treatment + Super-GOAT gate. Possibly three if the workload design itself needs a research note.

**Direction A should be rejected** (breaks Plan 329's contract).
**Direction C should be filed as a separate issue** if pursued — it's a different Super-GOAT question about a hypothetical primitive, not Issue 002.

## Open question for the user

Before I write any implementation plan, I need direction on:

1. **Which workload?** Hindsight policy evaluation (offline, replay-based), counterfactual curiosity (online, during play), or dreamer-style rollout (planning-time)? Each has different infrastructure requirements and different "known future outcome" semantics.

2. **Is the scope acceptable?** Direction B is ~2 plans of work (baseline harness + treatment), comparable to Plan 314's effort but harder. The alternative is to **defer Issue 002** until a Transformer-in-the-loop game-AI workload naturally emerges from other work, and pick a different open target now (Issue 007 WASM SIMD, or Issue 001 Apollonian exploration).

3. **Or pivot to Direction C?** If the latent-space AC-Prefix reframing is more interesting than the token-level Super-GOAT gate, that's a different (new) issue and a different plan — but it doesn't answer Issue 002 as written.

## What I did NOT do (honest disclosure)

I did **not** write an implementation plan. Writing a 7-phase plan that assumes Direction B without user confirmation on the workload choice would be hallucinating scope — exactly the failure mode AGENTS.md warns against ("Do summary and stop when low confident to prevent hallucination"). The integration-surface evidence is clear that the three primitives don't share a compute graph; pretending they do would produce a plan that fails on first integration attempt.

## Cross-references

- **Issue 002:** `katgpt-rs/.issues/002_ac_prefix_super_goat_gate.md` (the Super-GOAT question this blocks)
- **Research 295 §2.4:** `katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md` (the fusion table — conceptual level)
- **Plan 313:** `katgpt-rs/.plans/313_AC_GPT_Prefix_Primitive.md` (AC-Prefix primitive, token-level)
- **Plan 299:** `katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md` (Engram, latent-level)
- **Plan 309:** `katgpt-rs/.plans/309_*` (Latent Field Steering, latent-level)
- **Plan 314:** `riir-ai/.plans/314_bom_g2_arena.md` (BoM G2 arena precedent — the template for Direction B)
- **Plan 329:** `riir-ai/.plans/329_*` (QuestFunctor — the real Engram integration, latent-only)
- **Verified integration points:**
  - AC-Prefix: 0 implementors of `ForwardForAcPrefix` outside katgpt-core tests.
  - Engram: `riir-games/quest_grammar/quest_functor.rs` (QuestFunctor, FUNCATTN closed-form solve, d=8, latent-to-latent).
  - Latent Field Steering: `riir-engine/src/latent_field_wiring.rs` (FieldRegistry + FactionStanceRegistry, additive overlay on HLA slice).

## Resolution (2026-06-26) — VERDICT: CLOSE WITH NEGATIVE SUPER-GOAT RESULT

The prior session filed this issue as "design decision required" and offered three directions (A/B/C) for the user to choose. A deeper audit (this session) resolves the decision **by evidence, not by preference**: all three directions are either rejected (A), negative-ROI (B), or a different question (C). The fusion is not realizable.

### Five verified structural facts

1. **No shared compute graph** (the original finding, re-confirmed). AC-Prefix needs a local causal Transformer forward over `&[u32]` tokens via `ForwardForAcPrefix`. Engram (`fuse_into_hidden_state`, FUNCATTN closed-form solve) and Latent Field Steering (`apply_latent_steering`, SAXPY) operate on `f32` hidden-state slices with no Transformer. The substrates are incompatible by construction.

2. **No Transformer-in-the-loop host workload exists in riir-ai** (verified by grep + API audit). Every LLM/forward path in riir-ai was checked:
   - `CwmSynthesisLlm` trait: `fn synthesize(&self, prompt: &SynthesisPrompt) -> Result<String, LlmError>` — a **remote REST call to Gemini 2.5 Pro** (or `MockCwmLlm` canned responses). No local Transformer, no attention mask access, no per-position logprobs. AC-Prefix cannot plug in.
   - QuestFunctor (Plan 329): FUNCATTN closed-form Tikhonov solve, `f32 → f32`, d=8. Not a Transformer.
   - `DominoGRU::forward`: recurrent GRU. Not a causal Transformer.
   - `AneInferenceBackend::forward`: generic ANE hardware dispatch. Not a token-sequence causal Transformer.
   - `ChannelPredictionHead::forward`: a prediction head, not a full Transformer.
   - `DominoAdapter`: adapter forward. Not a causal Transformer.
   - **Zero implementors** of `ForwardForAcPrefix` outside katgpt-core tests/examples (re-confirmed by grep).
   - Plan 314's BoM G2 arena (the Super-GOAT precedent) uses **no Transformer** — it's `BoMSampler` (latent micro-belief kernel) + `MultiThreatArena` (16-dim obs) + minimax-over-K planner.

3. **Compute economics are catastrophic**. At iso-compute, AC-Prefix is 100× (micro-GPT: ~44K FLOPs) to 377,000× (production 4-layer model: ~151M FLOPs) more expensive than additive latent fusion (Engram+Steering ≈ 400 FLOPs at HLA d=8, K_MAX=16). AC-Prefix's GOAT speedup (27× over iterative MLM unmasking) **does not apply** — the baseline (Engram+Steering) is already single-pass additive, not iterative MLM. There is no speedup to claim.

4. **Multi-layer correctness gap** (Issue 003, non-blocking for G1 but blocking for Super-GOAT). AC-Prefix G1 equivalence (`|dedup − iterative| = 0.0`) only holds for **single-layer** models. Multi-layer representations diverge (r0 copies evolve through layers attending only to other r0 copies). A real game-AI workload needs a multi-layer model (single-layer lacks capacity for meaningful reasoning). Closing the multi-layer gap requires riir-train (LoRA fine-tuning). So even if a host workload existed, the Super-GOAT quality measurement would be on an unproven-correct multi-layer forward.

5. **Research 295 §2.4 rates the novelty gate as borderline-to-negative**. Q2 (new capability class): "Latent Field Steering × Engram already gets you ~70% there additively. Borderline." Q3 (selling point): "Engram × Latent Steering already gives a weaker version of this sentence." The research note itself downgraded from Super-GOAT to GOAT for this reason. The Super-GOAT re-opening (Issue 002) was contingent on a quality win that the substrate analysis now shows is not achievable.

### Why Direction B (build a Transformer-in-the-loop arena) is negative-ROI

Direction B would require: (a) designing a game-AI workload where "conditioning on a known future outcome" is semantically meaningful (hindsight-eval / counterfactual-curiosity / dreamer-rollout), (b) building a Plan-314-scale arena with a causal Transformer in the hot path, (c) implementing `ForwardForAcPrefix` over that Transformer, (d) getting riir-train to close the multi-layer correctness gap, then (e) running the iso-compute quality comparison. This is **2–3 plans minimum** for a result that facts 1–5 strongly indicate will be **negative** (the additive baseline already covers ~70% at 100×–377,000× lower compute cost). The ROI is deeply negative.

### Constructive framing — what AC-Prefix IS good for

This verdict is **not** a verdict against the AC-Prefix primitive. The primitive is:
- Shipped, default-on, GOAT-passed (G1 modelless-corrected, G2 27× speedup, G3 no-regression, G4 alloc-free).
- The **only** primitive in katgpt-core that provides token-level arbitrary-conditional evaluation in a single forward pass.
- Valuable for **standalone** conditional evaluation queries (e.g., offline analysis: "what's the conditional likelihood of this token sequence given this conditioning set?").

The negative verdict is specifically about the **fusion** with latent-domain primitives (Engram × Latent Field Steering). The fusion requires a host workload that spans both the token domain and the latent domain in a single forward pass. No such workload exists, and building one just for this fusion is not justified.

### Re-open condition

Re-open Issue 002 only if **all three** prerequisites emerge naturally from other work (do NOT build them just for this fusion):
1. A Transformer-in-the-loop game-AI workload lands in riir-ai for an independent reason (e.g., NPC dialogue, quest text generation, replay narration).
2. That workload's Transformer is local (not a remote API), with attention-mask access.
3. riir-train closes the multi-layer correctness gap (Issue 003 follow-up).

Until then, AC-Prefix is a shipped tool awaiting its host workload. Don't build the host just for AC-Prefix; add AC-Prefix when a host emerges naturally.

## TL;DR

Issue 002 cannot be implemented without first resolving a compute-substrate incompatibility: AC-Prefix needs a causal Transformer forward over tokens; Engram and Latent Field Steering operate on f32 hidden-state slices with no Transformer. Three design directions exist (A: add Transformer to QuestFunctor — rejected; B: build a new Transformer-in-the-loop game-AI workload — honest but ~2 plans of work; C: reframe AC-Prefix as latent-space — a different Super-GOAT question). Direction B is the only path that answers Issue 002 as written, but it needs user direction on workload choice (hindsight-eval vs counterfactual-curiosity vs dreamer-rollout) before any plan can be drafted.
