# Research: G-Zero — Self-Play for Open-Ended Generation from Zero Data (21)

> Source: [G-Zero](https://arxiv.org/pdf/2605.09959) by Chengsong Huang, Haolin Liu, Tong Zheng, Runpeng Dai, Langlin Huang, Jinyuan Li, Zongxia Li, Zhepei Wei, Yu Meng, Jiaxin Huang (WashU · UVA · UMD)
> Date: 2026-05, distilled 2026-05-13
> Raw code: not yet released
> **Verdict: HIGH VALUE — Verifier-Free Reward Signal for Plan 042 / Plan 048 Self-Improving Loop**

## Summary

G-Zero is a verifier-free self-play framework where a single base model bootstraps itself on open-ended tasks (writing, advice, explanation) without external judges, ground-truth labels, or reward models. Prior self-play work (R-Zero, Absolute Zero, SPIN) hinged on a verifier — a math grader, code executor, or external judge — which created a *capability ceiling* (model can't outgrow the verifier) and invited *reward hacking* (model exploits verifier quirks).

The core trick is **Hint-δ**: an intrinsic reward measuring how much a self-generated hint shifts the Generator's own output distribution. If hint h makes the Generator more confident in a better response, then h is informative *and* the underlying query q is challenging — both are useful training signal, with no external judge required.

**Two application paths (Plan 049 modelless-first):**

1. **Phase 1 — Modelless (our primary path):** δ is architecture-agnostic — a scalar like `ScreeningPruner::relevance()`. Feed it directly into existing `AbsorbCompress` (gates heuristic promotion by blind-spot density) and `BanditPruner` (δ as dense reward). No DPO, no GRPO, no gradient updates. A `TemplateProposer` generates (query, hint) pairs from rules + bandit history — 0 GPU cost.

2. **Phase 2 — Model-Based (paper's approach, opt-in):** Only when modelless plateaus. Add GRPO-trained Proposer and length-normalized DPO Generator. Two co-evolving models from the same base:
   - **Proposer** (GRPO-trained) generates `(query, hint)` pairs that maximize Hint-δ on the frozen Generator.
   - **Generator** (DPO-trained) is fine-tuned on `(hint-conditioned response > unconditioned response)` preference pairs, internalizing the hint-guided improvement.

Critically, **>70% of the final DPO pool comes from non-verifiable tasks** (writing, advice), yet reasoning benchmarks (AIME25 +5.2 pp on Qwen3-8B-Base) improve — showing reasoning capability transfers *out* of open-ended training, not the other way around.

---

## Core Concepts

### Hint-δ Reward (the central object)

For a query `q`, a hint `h`, and an unassisted response `a_hard` sampled from the Solver on `q` only. δ is computed via **teacher-forced log-probs** — the same `a_hard` tokens are scored under both prompt contexts:

```
δ(q, h, a_hard) = (1/T) · Σ_{t=1..T} [ log π_S(a_hard_t | q, a_hard_<t)  -  log π_S(a_hard_t | q, h, a_hard_<t) ]
```

Both terms score the **same** `a_hard` tokens — the difference is whether `h` is in the prompt. Positive δ ⇒ the hint shifts the Solver away from its own unassisted response ⇒ hint carries structural signal. The paper retains the **lower half** of the empirical δ distribution (`bot50` filter) — low-δ pairs distill style shifts that generalize without hints; high-δ pairs indicate answer leakage that hurts no-hint test performance.

**Source:** `.raw/G-Zero/g_zero/hint_delta.py` — `delta = logp_q - logp_qh` via Tinker `compute_logprobs`.

Two properties make this work as supervision:
- **Intrinsic** — uses only `π_S`'s own log-probs. No verifier, no reward model, no labels.
- **Joint difficulty + informativeness** — large δ requires both a hard query (Solver uncertain) and a good hint (Solver pivots on it). Hacking δ on a trivial query is hard because the unassisted response is already near-optimal, leaving no room for the hint to shift the distribution.

### Proposer Training (GRPO)

**Paper phases (per round):**
1. **Phase 1** (optional GRPO Challenger): Proposer is GRPO-trained against δ reward. Ablation shows skipping this (`--run_phase1 false`) matches Phase-1-on within noise on Qwen3-8B-Base.
2. **Phase 2** (Build DPO pool): Challenger generates `(q, h)` pairs; Solver samples `a_hard ~ π(·|q)` and `a_assisted ~ π(·|q,h)`; compute δ; filter by percentile + quality.
3. **Phase 3** (DPO Solver): DPO-train Solver on δ-filtered pairs. Prompt = `q` only (no hint), so trained Solver internalizes hint-assisted style without needing hints at test time.

The Proposer (Challenger) is a policy that emits `(q, h)` pairs. Reward:

```
r(q, h) = δ(q, h, a_hard)  -  P_length  -  P_BLEU
```

where `a_hard` is sampled from the *frozen* Solver on `q`. Structural penalties prevent two failure modes:
- `P_length` — penalizes hints exceeding ~200 chars (else the Proposer dumps the answer into the hint).
- `P_BLEU` — penalizes near-duplicate queries within a batch (else the Proposer collapses to one easy mode).

Trained with GRPO (group-relative policy optimization, à la DeepSeek-R1) on groups of sampled `(q, h)` per batch.

### Generator Training (length-normalized DPO)

Given Proposer-curated `(q, h)`, sample two responses from the current Generator:
- `a_chosen` ~ `π_G(· | q, h)` — hint-conditioned
- `a_rejected` ~ `π_G(· | q)` — unconditioned

The hypothesis: hint-conditioned responses are systematically better when δ > 0. DPO loss is length-normalized to remove length bias.

**δ-filter (lower-half band)** — only pairs with δ in the **[0, 50] percentile** are kept. The ablation shows this matters:
- `[0, 50]` (used) → balanced gains across math + IFEval + AlpacaEval.
- `[50, 100]` → out-of-distribution: high-δ pairs are too far from the Generator's current policy; DPO destabilizes.
- `[20, 80]` → middling.

So the system trains on the *most learnable* shifts, not the most dramatic ones.

### Algorithm 1 — Co-evolutionary Loop

```
Initialize: π_G ← π_base,  π_P ← π_base
For round r = 1..R:
    # Phase 1: Proposer step (Generator frozen)
    Freeze π_G
    For each batch:
        Sample (q, h) ~ π_P
        Sample a_hard ~ π_G(· | q)
        Compute δ(q, h, a_hard) via π_G log-probs
        r = δ - P_length(h) - P_BLEU(q)
    Update π_P ← GRPO(r)

    # Phase 2: Generator step (Proposer frozen)
    Freeze π_P
    Sample many (q, h) ~ π_P
    For each (q, h):
        a_chosen   ~ π_G(· | q, h)
        a_rejected ~ π_G(· | q)
        δ ← Hint-δ(q, h, a_rejected)
    Filter to δ ∈ [0, 50] percentile
    Update π_G ← DPO_lengthnorm(chosen ≻ rejected)
```

### Theoretical Guarantee

Theorem 1 (best-iterate suboptimality):
```
J(π*) − J(π_{t₀}) ≤ Õ( ε + √η_δ )
```
where:
- `η_δ` — pseudo-label noise after δ-filtering (smaller δ-band → smaller η_δ).
- `ε` — depends on Proposer-induced exploration coverage `α_S` and distribution mismatch `C_Q` between Proposer queries and the evaluation distribution.

In plain English: as long as the Proposer explores enough (`α_S`) and the δ-filter keeps the chosen preferences clean (`η_δ`), the Generator's best round comes within bounded distance of optimal. This is the first formal convergence result for verifier-free open-ended self-play.

---

## Experimental Results

### Models

- **Qwen3-8B-Base** (no instruction tuning) — clean cold-start setting.
- **Llama-3.1-8B-Instruct** — starts from a strong instruct model; tests whether G-Zero can still push it forward.

### Benchmarks

| Suite | What it measures |
|---|---|
| **AIME24 / AIME25** (mean@32) | Mathematical reasoning |
| **IFEval** (strict + loose, prompt + instruction) | Instruction following |
| **AlpacaEval 2.0 LC** | Open-ended chat quality vs GPT-4-Turbo |

### Headline numbers (Round 2)

**Qwen3-8B-Base → +G-Zero:**
| Metric | Base | R2 | Δ |
|---|---|---|---|
| AIME25 | 7.19% | **12.40%** | +5.21 pp |
| IFEval (strict) | 43.07% | 43.81% | +0.74 pp |
| AlpacaEval LC | 8.94% | 8.47% | −0.47 pp |

**Llama-3.1-8B-Instruct → +G-Zero:**
| Metric | Base | R2 | Δ |
|---|---|---|---|
| AlpacaEval LC | 24.12% | **27.86%** | +3.74 pp |
| (no regressions on math / IFEval — unlike R-Zero) | | | |

### Key Findings

1. **Capability transfer is asymmetric.** Training on >70% non-verifiable tasks (writing, advice, explanation) *improves* verifiable benchmarks (AIME). The reverse — training on math and hoping chat improves — is the standard R-Zero failure mode.
2. **R-Zero trade-off avoided.** R-Zero gains on math at the cost of AlpacaEval; G-Zero gains AlpacaEval at no cost to math.
3. **Proposer exploration matters more than Proposer accuracy.** The Proposer's job is to surface blind spots, not solve them. BLEU penalty + δ-band filter together prevent mode collapse.
4. **2 rounds is enough.** R3 shows diminishing returns; the bottleneck becomes Generator capacity, not signal quality.

### Comparison to Prior Work

| Method | Verifier needed? | Open-ended tasks? | Capability ceiling? |
|---|---|---|---|
| **SPIN** (2024) | SFT data (human refs) | No | Yes (= SFT ref quality) |
| **Absolute Zero** (2025) | Code executor | No | Yes (= verifier domain) |
| **R-Zero** (2025) | External LLM judge | Limited | Yes (= judge capability) |
| **G-Zero** (this) | **None** | **Yes** | **No external ceiling** |

---

## What Maps to Our System

### Where Plan 042 (TTT Feedback Loop) and Plan 048 (Self-Improving Loop) currently stop

Plan 042 wires `microgpt-rs/src/feedback.rs` → anyrag `/cache/export` → `riir-gpu/feedback_consumer.rs` → retraining. The **shape** of the loop is in place. The **reward signal**, however, is currently:
- Game-domain: win/loss (Bomberman, Monopoly).
- Code-domain: compile success, validator pass.
- Generic: `InferenceResult.reward` = max relevance from the screening pruner.

None of these work for **open-ended generation** (write a doc, explain a concept, suggest a refactor). G-Zero plugs that gap: Hint-δ is a reward signal that requires *nothing but the current model's own log-probs*.

**Key insight from Plan 049 analysis:** Hint-δ is architecture-agnostic — it's a scalar like `ScreeningPruner::relevance()`. The paper uses it for gradient-based training (DPO/GRPO), but it fits equally well into our **gradient-free HL infrastructure**. Modelless comes first because our `BanditPruner` is already 80% of a Proposer (UCB1 exploration ≈ BLEU penalty + δ-coverage), and `AbsorbCompress` already promotes heuristics — δ just makes both smarter.

### What Actually Applies

#### Phase 1: Modelless (Primary Path — δ → existing HL infrastructure)

##### 1a. Hint-δ as Foundation (Highest Value, Shared by Both Phases)

Hint-δ needs two log-prob evaluations per token: `log π_G(a_t | q, a_<t)` and `log π_G(a_t | q, h, a_<t)`. Both are already computed during normal decoding:

- `riir-gpu/src/loss.rs` already emits a `log_probs_buf` for cross-entropy. That's the unconditioned term.
- The hint-conditioned term is a second forward pass with `h` prepended — *or*, if we're already running with the **EmbeddingRouter + KV cache priming** (`riir-router/embedding.rs`, Plan 024), the hint is *already* in the KV prefix. We just need to also run an unconditioned pass at training data collection time.

Implementation surface:
- New helper `compute_hint_delta(q, h, a) -> f32` in `riir-gpu` using two passes through `loss.rs::log_probs_buf`.
- New field `InferenceResult.hint_delta: Option<f32>` in `microgpt-rs/src/types.rs` (alongside existing `reward`).
- Pipe it through the existing `feedback.rs` → anyrag flow.

##### 1b. DeltaGatedAbsorbCompress (High Value, Smart Modelless)

Current `AbsorbCompress` promotes heuristics based on raw environment reward (did the game say "good"?). G-Zero's insight: promote heuristics where the model has **blind spots** (high δ), not just where the environment was positive.

- `DeltaGatedAbsorbCompress` wraps existing `AbsorbCompressLayer<P>`.
- Absorb gate: only promote arms where `δ ≥ delta_threshold` (default: 0.02).
- Dual gate with existing `ReviewMetrics` benefit-ratio: δ must be meaningful AND reviewer must be net-positive.
- Why smarter: blind spots = high-δ = model doesn't already know this → promote to constraint. Current system can't distinguish "environment was nice" from "model learned something new."

##### 1c. DeltaBanditPruner (High Value, Dense Reward)

`microgpt-rs/src/pruners/bandit.rs` (UCB1 / Thompson / ε-greedy) already does exploration. G-Zero's δ gives it a **denser, more informative reward**:

- Arm = (domain, hint-template).
- Reward = Hint-δ (immediate, per-token, no episode completion needed).
- Standard UCB1 exploration gives the `α_S` coverage Theorem 1 requires.
- `blind_spot_arms(top_k)` returns arms with highest accumulated δ — targets for next query-hint generation.

This is a much cheaper Proposer than full GRPO and likely sufficient for narrow-domain agents (Bomberman, py2rs). Plan 025 already proved model-based bandit gets +12.1% reward over modelless — δ should improve both.

##### 1d. TemplateProposer (Medium Value, Zero GPU Cost)

Rule-based query-hint generator replacing the neural Proposer for Phase 1:
- 6 categories from G-Zero Appendix A: Writing, Explanation, Advice, Analysis, Coding, Reasoning (capped ≤1/6).
- Bandit-weighted template selection: UCB1 over template categories, biased toward arms with high historical δ.
- Targets known blind spots from `DeltaBanditPruner.blind_spot_arms()`.
- 0 GPU cost, instant generation, fully deterministic.

#### Phase 2: Model-Based (Opt-in — δ → DPO/GRPO weight updates)

##### 2a. Prompt Router as Proposer (High Value, Architectural Fit)

The **Proposer** in G-Zero generates `(query, hint)` pairs. Our `riir-router` is *almost* this object today:
- `riir-router/keyword.rs` and `embedding.rs` already map `query → (domain, hint-via-KV-prime)`.
- `riir-router/registry.rs` maps domain → expert pruner + LoRA path.

The router emits hints as KV-cache primes; G-Zero wants explicit hint *text* fed into the Generator. The gap is small: have the router additionally emit a textual hint (a routed example, a doc snippet, a domain prompt-prefix) into the Generator's context — which is what RAG already does. **Plan 023 (Prompt Router) + Plan 024 (Embedding Router KV Prime) together = a Proposer prototype.** What's missing is the **training** of the Proposer to *maximize* Hint-δ, rather than just retrieve nearest neighbors.

##### 2b. DPO Training in riir-gpu (High Value, New)

`riir-gpu/training_loop.rs` currently has cross-entropy via `loss.rs`. DPO requires:
- A *pairwise* loss: `−log σ(β · (log π_G(chosen|q) − log π_G(rejected|q) − log π_ref(chosen|q) + log π_ref(rejected|q)))`.
- A frozen reference policy `π_ref` (= the Generator at the start of the round; can be the LoRA base before the round's delta).
- Length normalization (divide log-probs by token count).

The infrastructure to do this on GPU is mostly there: `lora.rs` gives us the policy delta, `loss.rs` gives us log-probs. We need a new `dpo_loss.rs` ~100 LOC and a new `training_loop::train_dpo()` entrypoint. The data side (`dataloader.rs`) already eats JSONL; the pair format `{q, chosen, rejected, delta}` is a small schema add.

##### 2c. riir-burner Pipeline for Round Cadence (Medium Value, Already Built)

`riir-burner/pipeline.rs` already does corpus → train → pack → verify. G-Zero rounds map cleanly:
- Round start: snapshot current LoRA as `π_ref`.
- Phase 1 (Proposer GRPO): collect `(q, h, δ)` triples.
- Phase 2 (Generator DPO): emit `(q, chosen, rejected)` JSONL → `riir-burner pipeline --backend rust` → new `output/adapter.bin`.
- Hot-swap via `HotSwapPruner` (existing Plan 048 mechanism).

This means we already have **3/4 of the wiring**. The new pieces are Hint-δ computation, DPO loss, and a Proposer policy.

##### 2d. Bomberman / Monopoly as Verifier-Free Domains (High Value, Showcase)

Our existing arenas have explicit verifiers (game outcome). But G-Zero's premise is that **verifier-free domains also improve when trained alongside**. Adding open-ended Bomber-Tech *explanations* ("why is this strategy good?") as a non-verifiable G-Zero domain could improve the verifiable strategy-selection metric — mirroring the paper's AIME-improves-from-AlpacaEval finding. This is a cheap, falsifiable experiment on top of our existing arenas.

**Modelless showcase:** Run Bomberman arena with `DeltaGatedAbsorbCompress` + `DeltaBanditPruner` (Phase 1). Compare win rate vs existing HL. Hypothesis: δ-gated promotion converges faster because it targets blind spots, not just low-reward arms.

### What Does NOT Map

| G-Zero Concept | Why It Doesn't Apply (Yet) | Revisit When |
|---|---|---|
| **8B base model scale** | Our draft model is tiny (head_dim=4). Hint-δ may be noisier at small scale; needs the target model in the speculative-decoding pair, not the draft. | Larger models |
| **AlpacaEval / AIME benchmarks** | Not our evaluation surface. We measure on py2rs / Bomberman / Monopoly. Need domain-appropriate analogues. | — |
| **GRPO Proposer** | Full GRPO is heavy. **Bandit pruner IS the Proposer at our scale** (Phase 1 modelless). Revisit GRPO only if bandit-Proposer plateaus (Phase 2 opt-in). | Bandit plateaus |
| **General-purpose hint text** | We use *structured* hints (domain routing, KV primes, validator outputs). `TemplateProposer` uses rule-based hints first; neural hints are Phase 2. | — |
| **Length-normalized DPO at full sequence scale** | We generate short outputs (single Bomber move, single Rust function). Length normalization matters less; standard DPO likely fine. | Longer outputs |

---

## Comparison: G-Zero vs Our Existing Feedback Loops

| Aspect | Plan 042 TTT Feedback | Plan 048 Self-Improving | Bandit Pruner | G-Zero Phase 1 (Modelless) | G-Zero Phase 2 (Model-Based) |
|---|---|---|---|---|---|
| **Reward source** | Task-specific (validator, game) | Task-specific + heuristic | Per-arm relevance | **Intrinsic (Hint-δ)** | **Intrinsic (Hint-δ)** |
| **Needs verifier?** | Yes | Yes | Yes (relevance) | **No** | **No** |
| **Open-ended tasks?** | No | No | Limited | **Yes** | **Yes** |
| **Update mechanism** | LoRA retrain (Python/Rust) | Hot-swap LoRA | Online Q-values | **AbsorbCompress + bandit Q** | DPO LoRA retrain |
| **Gradient updates?** | Yes (LoRA) | Yes (LoRA) | No | **No** | Yes (LoRA) |
| **GPU cost** | High | High | Zero | **Near-zero** (2 forward passes) | High |
| **Composable with others?** | — | — | Yes | **Yes** (δ feeds bandit + absorb) | **Yes** |

**Key insight:** G-Zero doesn't *replace* our existing feedback loops — it provides the **missing intrinsic reward** for tasks where no verifier exists. Phase 1 (modelless) makes the existing bandit + AbsorbCompress smarter without adding complexity. Phase 2 (model-based) adds neural self-play only when modelless plateaus.

---

## Application to Our System

### Direct Mappings

| Paper Concept | Our Equivalent | Phase | Status |
|---|---|---|---|
| **Hint-δ reward** | New helper in `riir-gpu` using existing `loss.rs::log_probs_buf` | Both | ❌ Need to build |
| **δ as absorb gate** | `DeltaGatedAbsorbCompress` wrapping existing `AbsorbCompressLayer` | 1 | ❌ Need to build |
| **δ as bandit reward** | `DeltaBanditPruner` wrapping existing `BanditPruner` | 1 | ❌ Need to build |
| **Template-based Proposer** | `TemplateProposer` (rule-based, bandit-weighted) | 1 | ❌ Need to build |
| **Generator model** | Main inference model in `microgpt-rs` (draft + target) | Both | ✅ Exists |
| **Bandit as Proposer** | `pruners/bandit.rs` UCB1/Thompson (80% of GRPO at our scale) | 1 | ✅ Exists |
| **Episode history** | `TrialLog` (JSONL) | Both | ✅ Direct reuse |
| **Reward hacking defense** | `ReviewMetrics` benefit-ratio gate | Both | ✅ Similar philosophy |
| **Hot-swap updated model** | `HotSwapPruner` | Both | ✅ Direct reuse |
| **Regression safety** | `RegressionSuite` | Both | ✅ Direct reuse |
| **LoRA training** | `riir-burner` pipeline (rank 32) | 2 | ✅ Direct reuse |
| **Round cadence** | `riir-burner/pipeline.rs` + `HotSwapPruner` | 2 | ✅ Exists |
| **Theorem 1 (α_S coverage)** | Bandit exploration bonus | Both | ✅ Conceptually present |
| **Frozen π_ref** | LoRA-zero baseline (= base model without adapter delta) | 2 | ✅ Exists (just freeze before round) |
| **GRPO** | Bandit pruner UCB1 (Phase 1 substitute) → full GRPO (Phase 2) | 1→2 | ⚠️ Substitute → build |
| **DPO loss** | New `riir-gpu/src/dpo_loss.rs` next to `loss.rs` | 2 | ❌ Need to build |
| **δ-filter (lower-half band)** | New filter in `feedback_consumer.rs` corpus export | 2 | ❌ Need to build |
| **Length-normalized DPO** | Variant of DPO; per-token mean | 2 | ❌ Need to build |
| **BLEU duplication penalty** | New helper; reuse `riir-validator-sdk` similarity? | 2 | ❌ Need to build |

### What to Build (Gap Analysis)

#### Phase 1: Modelless (T1–T5, ~600 LOC total)

##### Priority 1: Hint-δ Helper (Foundation, ~150 LOC)

`riir-gpu/src/hint_delta.rs`:
- Two forward passes through the Generator (with/without hint).
- Return per-token δ and mean δ.
- Reuse existing `loss.rs::log_probs_buf` plumbing.
- Pure addition; no breaking changes.
- **Shared by both phases.**

##### Priority 2: DeltaGatedAbsorbCompress (~100 LOC)

`microgpt-rs/src/pruners/delta_absorb_compress.rs`:
- Wraps existing `AbsorbCompressLayer<P>`.
- Absorb gate: `δ ≥ delta_threshold` (default: 0.02).
- Dual gate with `ReviewMetrics` benefit-ratio.
- Replaces `should_compress_gated()` with δ-aware version.

##### Priority 3: DeltaBanditPruner (~100 LOC)

`microgpt-rs/src/pruners/delta_bandit.rs`:
- Wraps existing `BanditPruner<P>`.
- `observe_delta(arm, δ)` — feed δ as dense reward.
- `blind_spot_arms(top_k)` — arms with highest accumulated δ.
- Implements `ScreeningPruner` — drop-in replacement.

##### Priority 4: TemplateProposer (~150 LOC)

`microgpt-rs/src/pruners/template_proposer.rs`:
- 6 categories from G-Zero Appendix A (Writing, Explanation, Advice, Analysis, Coding, Reasoning ≤1/6).
- UCB1-weighted template selection biased toward `blind_spot_arms()`.
- Emits `(query, hint)` pairs — no neural model needed.

##### Priority 5: Modelless Benchmark (~100 LOC)

- Compare modelless G-Zero vs existing HL on Bomberman/Monopoly arenas.
- Metrics: win rate, score, survival, episodes to convergence, blind-spot discovery rate.
- Hypothesis: δ-gated promotion converges faster (denser signal than raw reward).

#### Phase 2: Model-Based (T6–T9, ~800 LOC total, opt-in)

##### Priority 6: DPO Loss in riir-gpu (~200 LOC)

`riir-gpu/src/dpo_loss.rs`:
- Pairwise loss with reference policy.
- Length normalization.
- Wire into `training_loop.rs` as alternate path next to cross-entropy.
- JSONL schema in `dataloader.rs`: `{q, chosen, rejected, delta}`.

##### Priority 7: δ-Filtered Corpus Export (~100 LOC)

`riir-gpu/src/feedback_consumer.rs`:
- Augment polling logic to compute δ-percentile of incoming `InferenceResult`s.
- Keep only `[0, 50]` band before triggering retrain.
- Existing BLAKE3 dedup + hot-swap unchanged.

##### Priority 8: GRPO Proposer (~200 LOC)

`microgpt-rs/src/pruners/grpo_proposer.rs`:
- Replace `TemplateProposer` when modelless plateaus.
- Group of K rollouts, advantage standardization, clipped surrogate.
- Full GRPO as described in paper §2.

##### Priority 9: Plan 049 Round Driver + Benchmark (~300 LOC)

- Phase 1: bandit-Proposer rollout → collect `(q, h, δ)` to anyrag cache.
- Phase 2: DPO LoRA training via `riir-burner pipeline`.
- Hot-swap via existing mechanism.
- Eval gate: AlpacaEval-analogue on our domain (Rust doc quality? Bomber strategy explanation?).
- Sanity check: verify Theorem 1 empirically — does best round track √η_δ?

---

## Key Takeaways

1. **Verifier-free reward is the missing piece.** Our self-improving loop (Plan 042 / Plan 048) has the wiring but not the signal for open-ended tasks. Hint-δ closes that gap with a single intrinsic quantity computable from the model's own log-probs.

2. **Modelless first, model-based second.** δ is architecture-agnostic — a scalar like `ScreeningPruner::relevance()`. Feed it into existing `AbsorbCompress` + `BanditPruner` first (Phase 1, zero gradient updates, ~600 LOC). Add DPO/GRPO only when modelless plateaus (Phase 2).

3. **70% non-verifiable → verifiable gains.** The paper's most surprising finding is that training on writing/advice improves AIME math. This is *highly* relevant for us: we have many open-ended sub-tasks (explain a refactor, suggest a Bomber strategy in words) that we currently can't train on because there's no reward. G-Zero makes them training-eligible — even in modelless mode via δ-gated AbsorbCompress.

4. **The bandit pruner IS the Proposer at our scale.** UCB1's exploration bonus = G-Zero's BLEU penalty + δ-coverage. We don't need GRPO for Phase 1; we need to feed Hint-δ as the bandit reward via `DeltaBanditPruner`. Plan 025 already proved model-based bandit gets +12.1% reward — δ should improve both model-based and modelless.

5. **δ-gated AbsorbCompress is smarter than reward-gated.** Current system promotes heuristics based on "did the environment say good?" δ-gating promotes based on "did the model learn something new?" — blind spots, not just low-reward arms.

6. **Lower-half δ filter is non-obvious and load-bearing.** Don't train on the most dramatic shifts — train on the most learnable ones. This contradicts the intuition that "harder examples = more signal" and is worth a dedicated ablation in our setting. (Phase 2 only.)

7. **Theorem 1 gives us a falsifiable gate.** Suboptimality bounded by `Õ(ε + √η_δ)` means we can monitor `η_δ` (post-filter pseudo-label noise) per round and stop when it stops shrinking. Concrete halting criterion for the self-play loop. Bandit exploration provides the `α_S` coverage.

8. **Two rounds is the sweet spot.** Saves us from over-engineering an infinite loop. Run R1, R2, eval, decide whether to ship. Paper reports R3 collapse on Llama — circuit breaker needed.

9. **Composable with Plan 048's existing flow.** No architectural rewrite needed. Phase 1: add Hint-δ to `InferenceResult`, wrap `BanditPruner` and `AbsorbCompress`. Phase 2: add DPO to `training_loop`, add δ-filter to `feedback_consumer`. Incremental, not rewrite.

10. **Cheap experiment is available today.** Bomber-tech *explanation* using `DeltaBanditPruner` + `TemplateProposer` (Phase 1 modelless) would validate the framework on infrastructure we already have, without needing GRPO, DPO, or 8B-scale models.

11. **Open question — does Hint-δ work on speculative decoding?** Our draft/target pair makes δ ambiguous: do we measure on the draft or the target? Worth thinking through before Priority 1. Likely answer: target model, since the draft is conditioned on it via verification.

---

## Citation

```bibtex
@article{huang2026gzero,
  title   = {G-Zero: Self-Play for Open-Ended Generation from Zero Data},
  author  = {Huang, Chengsong and Liu, Haolin and Zheng, Tong and Dai, Runpeng and
             Huang, Langlin and Li, Jinyuan and Li, Zongxia and Wei, Zhepei and
             Meng, Yu and Huang, Jiaxin},
  journal = {arXiv preprint arXiv:2605.09959},
  year    = {2026}
}
```
