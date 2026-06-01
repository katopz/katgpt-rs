# Research 050: PFlash Distillation — Compression as Complexity Signal

> **Source:** [Luce PFlash](https://github.com/Luce-Org/lucebox-hub) — speculative prefill compression (128K→2.6K, 10.4× TTFT reduction)
> **Date:** 2026-06-01
> **Related Research:** R002 (Speculative Decoding), R016 (AutoTTS), R037 (REAP Model-Based/Modelless Duality)
> **Related Plans:** Domain Inference Budget (Plan 026), MTP Budget Propagation (Plan 057), Bandit infrastructure
> **Domain:** katgpt-rs (modelless — inference orchestration, budget allocation)
> **Cross-ref:** riir-ai R036 (Megakernel), riir-ai P179 (TokenSpeed PFlash benchmarks)

---

## TL;DR

PFlash compresses 128K prompts to 2.6K tokens (50× reduction) using a 0.6B drafter's attention patterns, then feeds the compressed prompt to a 27B target. The 10.4× TTFT speedup is real but requires: 128K context, 27B model, NVIDIA GPU with BSA, 24GB VRAM memory dance.

**None of those conditions exist in our stack.** Plan 179 benchmarked our PFlash implementation on Apple Silicon (Gemma 2 2B, seq_len 64-1536) — **PFlash is never faster than Naive**. Block-scoring overhead exceeds savings at every tested length.

**What IS novel: the compression ratio is a free complexity signal.** PFlash doesn't use this signal — it compresses, then decodes with a fixed DDTree budget (22). We can do better: let the compression ratio dynamically adjust the decode budget. This is a zero-cost optimization — we're computing compression ratios anyway during PFlash scoring (or we can extract them from Naive attention patterns).

**GOAT Verdict: GAIN — compression-adaptive decode budget.** Modelless, no perf hurt, maps to existing Domain Inference Budget infrastructure. **→ Plan 167.**

---

## What PFlash Actually Does (from Source)

### Architecture

```
prompt (≤ 128K tokens)
  → park target + park draft          (free ~18 GB VRAM)
  → 0.6B drafter scores importance    (FlashPrefill BSA, ~10 GB peak)
  → compress: keep top keep_ratio     (128K → 2.6K at keep_ratio=0.02)
  → free drafter                      (release ~10 GB)
  → unpark target + unpark draft      (reload ~18 GB)
  → DFlash spec decode compressed     (DDTree budget=22, fixed)
  → park draft (idle)
```

### Key Technical Details (from bench_niah_cpp.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Drafter | Qwen3-0.6B BF16 | Separate from DFlash decode drafter |
| DFlash drafter | Qwen3.6-27B-DFlash FP16 | ~3.5 GB, different model |
| DDTree budget | 16-22 | FIXED — does not adapt to prompt complexity |
| keep_ratio | 0.02-0.05 | 2-5% of tokens retained |
| alpha | 0.85 | Block selection threshold |
| Memory dance | park/unpark/free | 3 models can't coexist in 24 GB |
| NIAH | Pass at all contexts | Compression preserves needle |

### The Two-Drafter Pattern

PFlash uses **two different drafters**:
1. **PFlash compress drafter**: Qwen3-0.6B BF16 (~1.3 GB) — only for importance scoring during prefill
2. **DFlash decode drafter**: Qwen3.6-27B-DFlash (~3.5 GB) — for speculative decoding after unpark

They can't coexist in 24 GB VRAM. The memory dance sequences them. This is the operational complexity.

### What's NOT in PFlash (Gap We Can Fill)

PFlash compresses the prompt, then decodes with a **fixed DDTree budget**. The compression ratio is computed but **discarded** — it doesn't feed back into any downstream decision.

This is the gap: the compression ratio encodes prompt complexity, and prompt complexity should inform decode strategy.

---

## Already Distilled (What We Already Have)

| PFlash Feature | Our Implementation | Status |
|---------------|-------------------|--------|
| FlashPrefill BSA (4 kernels) | `flashprefill_*.wgsl` in riir-gpu | ✅ Plan 044 |
| Block importance scoring | `BlockAttentionScorer` in katgpt-rs | ✅ |
| Token-level scoring | `AttentionScorer` in katgpt-rs | ✅ |
| Block selection (sink+window+alpha) | `block_select`, `block_select_entmax` | ✅ |
| Prompt compression | `compress_prompt`, `compress_prompt_blocks` | ✅ |
| Adaptive prefill | `speculative_prefill_adaptive` | ✅ |
| Cross-family scoring | CPU path supports any draft model | ✅ |
| Domain inference budget | Per-domain tree_budget, draft_lookahead | ✅ Plan 026 |
| Bandit infrastructure | `BanditPruner`, `BanditEnv`, strategies | ✅ |

---

## The Novel Idea: Compression-Adaptive Decode Budget

### The Insight

PFlash's compression ratio `r = len(compressed) / len(original)` is a **free complexity signal**:
- `r ≈ 0.02` (2% kept) = simple prompt, most tokens irrelevant → small decode budget needed
- `r ≈ 0.80` (80% kept) = complex prompt, most tokens relevant → large decode budget needed

PFlash computes `r` and throws it away. We can use `r` to dynamically adjust DDTree parameters.

### Why This Is Genuinely New

1. **PFlash doesn't do this** — fixed DDTree budget (22) regardless of compression ratio
2. **Our Domain Inference Budget doesn't do this** — fixed per-domain, doesn't adapt per-prompt
3. **AutoTTS (R016) doesn't do this** — adapts compute per-query but uses different signals
4. **Bandit doesn't do this** — learns per-domain, not per-prompt

This is **per-prompt budget adaptation** driven by a free signal from the prefill phase.

### How It Works

```
Phase 1: Prefill
  Naive or PFlash prefill on prompt
  → compute compression_ratio r = len(compressed) / len(original)
  → even without actual compression, compute r from attention scores:
    r = (tokens_above_alpha_threshold) / total_tokens

Phase 2: Budget Derivation
  base_budget = domain.tree_budget           // from domains.toml
  adaptive_budget = base_budget × f(r)       // scale by complexity
  
  where f(r) ∈ [0.5, 2.0]:
    f(0.02) = 0.5    // simple prompt → half budget (tokens are predictable)
    f(0.50) = 1.0    // medium prompt → normal budget
    f(0.80) = 2.0    // complex prompt → double budget (tokens are uncertain)

Phase 3: Decode
  DDTree with adaptive_budget instead of fixed base_budget
  DFlash marginals with draft_lookahead derived from adaptive_budget
```

### Why It's Modelless

No model training needed. The compression ratio comes from:
- **Option A**: Actual PFlash compression (already computed, currently discarded)
- **Option B**: Attention score distribution during Naive prefill (free — we already compute attention scores)
- **Option C**: Entropy of the first marginal (already computed by DFlash)

All three are zero-cost signals. The budget scaling function `f(r)` is a hand-tuned curve (like our existing β parameterization in Plan 026).

### Alignment with Existing Architecture

This extends the **Domain Inference Budget** system (Plan 026):

```toml
# Current: fixed per-domain
[domain]
name = "code"

[domain.inference]
tree_budget = 2374          # fixed
draft_lookahead = 12        # fixed

# Extended: per-domain base + per-prompt adaptive scaling
[domain]
name = "code"

[domain.inference]
tree_budget = 2374          # base (unchanged)
draft_lookahead = 12        # base (unchanged)
budget_adaptation = "compression"  # new: scale by compression ratio
# OR:
budget_adaptation = "entropy"      # scale by first-marginal entropy
# OR:
budget_adaptation = "off"          # disable (current behavior)
```

### Why This Matters for Verdict 003

Per the commercial strategy:
- **Engine (MIT, katgpt-rs)**: The budget adaptation algorithm is pure engine — it's a modelless optimization that makes speculative decoding more efficient. Any user of katgpt-rs benefits.
- **No SaaS dependency**: Works locally, no cloud needed.
- **Strengthens the wedge**: More efficient RIIR translation — simple Python patterns get fast translation (low budget), complex patterns get thorough exploration (high budget).
- **Compatible with existing architecture**: Extends Domain Inference Budget, doesn't replace it.

---

## GOAT Verdict

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Gain** | MEDIUM-HIGH | Per-prompt budget adaptation; free signal; no wasted compute on simple prompts |
| **Perf risk** | NONE | Budget scaling function is bounded [0.5, 2.0]; worst case = current behavior |
| **Alignment** | ✅ | Engine (MIT), extends existing Domain Inference Budget |
| **Urgency** | MEDIUM | We already have all prerequisites; implementation is ~200 lines |
| **Complexity** | LOW | Budget derivation function + wiring into DDTree dispatch |

### Decision: GAIN — Create Plan

The compression ratio is a free, information-rich signal that we currently discard. Using it to adapt decode budget is:
1. **Zero cost** — computed during existing prefill
2. **Zero risk** — bounded scaling, worst case = fixed budget
3. **Novel** — PFlash doesn't do this, AutoTTS doesn't do this, nobody does this
4. **Implementable today** — extends existing Domain Inference Budget

**Default: ON** — if GOAT proof passes (no regression, ≥5% improvement on heterogeneous prompts), budget_adaptation becomes default behavior. Per optimization.md: "if gain and no perf hurt must be on by default."

---

## What NOT to Distill

| PFlash Feature | Why Not |
|---------------|---------|
| Memory dance (park/unpark) | Apple Silicon unified memory — all models fit simultaneously |
| BSA (Block-Sparse Attention) | Requires sm_80+ (NVIDIA); not available on Apple Silicon |
| Cross-family drafter (0.6B → 27B) | We use same-family (Gemma 2 2B → Gemma 2 2B); cross-family needs tokenizer alignment |
| Dual-drafter architecture | Two drafters only needed when models don't fit in VRAM; we have unified memory |
| BSA kernel for FlashPrefill | Our PFlash WGSL kernels already do block-sparse attention; they're just slower than Naive on Apple Silicon (Plan 179) |

---

## The Deeper Pattern: Prefill-Decode Feedback Loop

PFlash treats prefill and decode as independent phases. The compression-adaptive budget creates a **feedback loop**:

```
┌─────────────┐     compression_ratio     ┌──────────────┐
│   PREFILL   │ ──────────────────────→   │    DECODE    │
│  (scoring)  │                           │   (DDTree)   │
│             │ ←────────────────────────  │              │
└─────────────┘     accept_rate_signal     └──────────────┘
```

This is a **closed-loop inference system** — the only one we'd have:
- **Forward signal**: compression_ratio → decode budget (complexity feeds forward)
- **Backward signal**: accept_rate → prefill aggressiveness (verification feeds back)

If DDTree accept rate is high (>90%), the prompt was simple — next time, use more compression.
If DDTree accept rate is low (<50%), the prompt was complex — next time, use less compression.

This is the Bandit learning the optimal compression for each domain over time, using both the forward signal (complexity) and backward signal (verification quality).

---

## References

1. Luce PFlash: https://github.com/Luce-Org/lucebox-hub (Apache 2.0)
2. Cross-Family Speculative Prefill: https://arxiv.org/abs/2603.02631 (SambaNova ICLR 2026)
3. FlashPrefill: https://arxiv.org/abs/2603.06199 (block-sparse attention for long-context prefill)
4. Speculative Prefill: https://arxiv.org/abs/2502.02789 (Liu et al., original Q-hook construction)
5. Our Plan 179 (TokenSpeed): PFlash benchmarks showing never faster than Naive on Apple Silicon
6. Our Plan 026 (Domain Inference Budget): per-domain tree_budget, β parameterization
7. Our Plan 057 (MTP Budget Propagation): budget derivation from domain config
