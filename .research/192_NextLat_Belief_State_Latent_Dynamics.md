# Research 192: NextLat — Belief-State Latent Dynamics for Inference-Time Prediction

**Date:** 2026-06
**Source:** arXiv:2511.05963 — Next-Latent Prediction Transformers Learn Compact World Models (Teoh, Tomar, Ahn, Hu et al., Microsoft Research, NeurIPS 2025 Keynote)
**Verdict:** GOAT — Default-On for inference, high-gain for speculative decoding quality
**Target:** Modelless (katgpt-rs) primary, Model-based (riir-ai) secondary

---

## Executive Summary

NextLat proves that training a lightweight MLP to predict the **next hidden state** from `(h_t, x_{t+1})` forces the transformer's representations to converge to **belief states** — compressed sufficient statistics of history that predict the future. The key insight for our stack: this latent dynamics model enables **variable-length self-speculative decoding** at 3.3× speedup without a separate draft model.

**Our fusion opportunity is NOT replicating NextLat's training loss** (that's model-based). Our opportunity is **distilling the inference-time artifact** — the belief-state transition dynamics — into our existing DDTree speculative pipeline at zero training cost.

---

## Paper Core Ideas

### 1. Belief States from Latent Prediction
- **Theorem:** If next-token prediction + next-latent transition prediction both converge, then `h_t` is a belief state (sufficient statistic for history).
- Standard next-token prediction alone does NOT guarantee belief states — self-attention provides "cheat" lookups.
- The latent dynamics model `p_ψ(h_{t+1} | h_t, x_{t+1})` is a residual MLP: `ĥ_{t+1} = f_ψ(h_t, x_{t+1}) + h_t`.

### 2. Variable-Length Self-Speculative Decoding
- The latent dynamics model composes recursively: `h_t → x_{t+1} → h_{t+1} → x_{t+2} → ...`
- Unlike MTP (fixed draft length = training horizon), NextLat can draft **arbitrarily long** because each latent transition is independent.
- 1.3B model on 100B tokens: up to 3.3× inference speedup, drafts remain valid to length 10+.

### 3. Compact World Models
- NextLat transformers learn **3× more compressed** latent representations (effective rank 52.7 vs GPT's 169.5).
- On Manhattan taxi rides: NextLat reconstructs coherent city maps; GPT produces incoherent "flying roads."
- On Countdown math reasoning: NextLat at d=1 beats MTP/JTP at d=8 (>35.7% improvement).

### 4. Transformer-RNN Co-Training
- NextLat effectively co-trains an RNN (the latent dynamics MLP) alongside the transformer.
- On A5 word problem: the co-trained RNN generalizes to 3× the training sequence length, even though the transformer itself cannot.
- The RNN has only 2.6M params vs 6.5M transformer — **the latent dynamics captures reusable computation that outlives the transformer's context window**.

---

## Distillation: Modelless vs Model-Based

### What's Training-Time Only (riir-ai territory)
- The auxiliary loss `L_next-h` (Smooth L1 between predicted and true hidden states)
- The KL loss `L_KL` (token distribution alignment between predicted and true states)
- Stop-gradient operators to prevent representational collapse
- Multi-step rollout supervision (`d` steps)

### What's Inference-Time (katgpt-rs territory)
- The **latent dynamics model** `p_ψ` as a lightweight draft engine
- **Belief-state quality** in existing hidden states (if trained with NextLat)
- Variable-length speculative drafting using recursive latent composition
- Compact representation → better KV cache compression, better pruning

---

## Fusion Ideas: Creative, Not Direct Mapping

### Fusion 1: Belief-State Speculative Drafter (Primary — katgpt-rs)

**Core Insight:** Our DDTree already has a speculative decoding pipeline. Instead of a separate draft model, use the **transition MLP** to draft from the target model's own hidden states.

**Architecture:**
```
Target forward(h_t, x_t) → logits, h_{t+1}
Belief Drafter: ĥ_{t+1} = f_ψ(h_t, x_{t+1}) + h_t  [residual MLP, ~2% of target params]
                logits_draft = output_head(ĥ_{t+1})
                Recursive: ĥ_{t+2} = f_ψ(ĥ_{t+1}, x_{t+2}) + ĥ_{t+1}
```

**Integration with DDTree:**
- Replace current draft model with belief-state drafter
- The drafter is a single MLP applied recursively — zero KV cache needed
- DDTree's ConstraintPruner still validates branch candidates
- Variable draft length: keep drafting until entropy exceeds threshold

**Expected gain:** 2-3× speedup on single-layer models, near-zero overhead (MLP is tiny).

**Perf concern:** MLP forward is O(d²) where d = embd dim. For embd=32, this is ~1K FLOPs per draft step vs ~50K for full forward. **Negligible overhead.**

### Fusion 2: Belief-State Pruner — Collapse Detection (katgpt-rs)

**Core Insight:** NextLat's belief states have low effective rank. If our hidden states' effective rank spikes, the model has "lost the plot" — the belief state has collapsed.

**Architecture:**
- Compute running effective rank of hidden states during decode
- Use rank as a ScreeningPruner signal: high rank → model is uncertain → invoke deeper search
- Low rank → model is confident → accept draft tokens
- This is a **modelless proxy for confidence** — no training needed, just SVD on the hidden state buffer

**Integration:** Add to `ScreeningPruner` trait. The `relevance()` method returns 1.0 - normalized_rank.

### Fusion 3: Latent Transition Cache (katgpt-rs)

**Core Insight:** NextLat's MLP predicts `h_{t+1}` from `(h_t, x_{t+1})`. For our speculative pipeline, we can **cache the transition output** and skip re-computation when the same `(h, x)` pair appears.

**Architecture:**
- Hash `(h_t, x_{t+1})` → `ĥ_{t+1}` in a fixed-size LRU cache
- During DDTree branching, sibling branches share parent hidden state
- Cache hit rate should be ~70%+ for game domains (limited action space)
- Reuses the papaya lock-free HashMap for thread safety

### Fusion 4: MTP → NextLat Drafter Upgrade (riir-ai)

**Core Insight:** Our existing MTP drafter (Plan 055, Gemma-style) uses extra transformer layers. NextLat shows a simple MLP achieves variable-length drafting — potentially replacing MTP layers entirely.

**Architecture:**
- During LoRA training, add NextLat auxiliary loss (`L_next-h` + `L_KL`)
- At inference, the MLP replaces MTP projection layers
- Variable draft length → higher acceptance rate → fewer verification passes
- Compatible with existing Cluster LM Head and Shared KV Cache

**Training impact:** +2-5% training overhead for NextLat losses. Negligible.

### Fusion 5: Self-Learning Adaptive CoT via Belief-State Planning (katgpt-rs + riir-ai)

**Core Insight:** NextLat proves latent dynamics compose into a planning engine. For our adaptive CoT (Plan 212), we can use the latent dynamics MLP to **plan in latent space** before committing to token generation.

**Architecture:**
- Before generating CoT tokens, rollout latent dynamics for K steps
- Score rollouts by KL divergence between predicted logits and current logits
- High divergence → model needs more CoT steps; low divergence → skip CoT
- This is the **collapse-aware** gate from Plan 212, but powered by latent planning instead of entropy alone
- No LLM training — the rollout is pure inference

**Connection to existing work:** This fuses with ThoughtFold (Plan 195) — the latent dynamics MLP is the "folding" engine that compresses multi-step reasoning into a single latent step.

---

## Related Work in Our Stack

| Our Component | NextLat Equivalent | Fusion Potential |
|---|---|---|
| DDTree speculative decoder | Variable-length self-speculative decoding | 🔴 High — replace draft model |
| MTP drafter (Plan 055) | Latent dynamics MLP | 🔴 High — simpler alternative |
| DomainLatent mid-layer injection | Belief-state compression | 🟡 Medium — complementary |
| ConstraintPruner | — | 🟢 Low — orthogonal |
| ThoughtFold (Plan 195) | Latent chain compression | 🔴 High — belief-state folding |
| Collapse-Aware Adaptive CoT (Plan 212) | Latent planning for think/skip | 🔴 High — latent divergence as gate |
| Raven RSM slots | Belief-state slots | 🟡 Medium — belief-state routing |
| AHLA streaming attention | Recurrent inductive bias | 🟡 Medium — complementary recurrence |
| GRAM SDE noise (Research 58) | Learned latent transition | 🔴 High — fusion of both approaches |
| MUX multiplexed tokens (riir-ai 048) | Compact latent representation | 🟡 Medium — both compress reasoning |
| RiM memory blocks (riir-ai 043) | Latent workspace | 🟡 Medium — both use latent compute |

---

## GOAT Verdict

| Criterion | Rating | Reasoning |
|---|---|---|
| Performance Impact | 🟢 Positive | MLP draft is ~50× cheaper than full forward; variable-length reduces verification overhead |
| Accuracy Impact | 🟢 Positive | Belief states improve planning, reasoning, and world model coherence |
| Implementation Cost | 🟢 Low | Single MLP + recursive application, no architecture change |
| Training Dependency | 🟡 Partial | Full benefit needs NextLat training loss (riir-ai); modelless path uses frozen MLP |
| Compatibility | 🟢 Full | No architectural change, feature-gated, composable with existing pruners |

**Decision: GOAT for inference-time belief-state drafting. Default-on feature gate.**

The latent dynamics MLP is the simplest speculative drafter possible — one matmul + residual per draft step. Combined with our existing DDTree pruning and ConstraintPruner validation, this creates a pipeline where:
1. MLP drafts belief-state tokens (fast, zero KV)
2. DDTree branches from drafts
3. ConstraintPruner validates branches
4. Target model verifies accepted branches

This is strictly better than the current draft model approach for single-layer configs and competitively better for multi-layer configs.

---

## Commercial Alignment

Per `.research/003`:
- **Engine (MIT):** Belief-state drafter MLP + DDTree integration → modelless inference improvement
- **Fuel (SaaS):** NextLat training loss for riir-ai LoRA training → better lora.bin quality
- **Flywheel:** Better belief states → better translations → better episodes → better validators
- The MLP drafter itself could be part of the "fuel" (trained NextLat weights) or the "engine" (random initialization with online adaptation via bandit)

**Recommendation:** Ship the MLP drafter as engine (MIT) with the ability to load pretrained NextLat weights as fuel (riir-ai trained). This maintains the engine/fuel split perfectly.

---

## References

- NextLat paper: arXiv:2511.05963
- Code: https://github.com/microsoft/NextLat
- Related: BST (Hu et al., 2024), JTP (Ahn et al., 2025), MTP (Gloeckle et al., 2024)
- Our related: Research 026 (Gemma MTP), Research 058 (GRAM), Research 048 (HRM), Plan 055 (MTP Drafter), Plan 195 (ThoughtFold), Plan 212 (Collapse-Aware CoT)

TL;DR: NextLat proves that a 3-layer MLP predicting next hidden states forces transformers to learn belief states, enabling 3.3× faster inference via variable-length self-speculative decoding. Our fusion: extract the MLP as a modelless speculative drafter for DDTree, use belief-state quality as a pruning signal, and integrate with existing ThoughtFold + Collapse-Aware CoT pipelines. GOAT — default-on.
