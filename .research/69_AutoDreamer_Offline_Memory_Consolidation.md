# Auto-Dreamer: Offline Memory Consolidation for Language Agents

**Paper**: arXiv:2605.20616 (May 2026)
**Authors**: Chongrui Ye, Yuxiang Liu, Yu Wang, Haofei Yu, Yining Zhao, Ge Liu, Julian McAuley, Jiaxuan You
**Institution**: UIUC, UCSD
**Fetched**: 2026-05-23

---

## Core Idea

Auto-Dreamer is a **two-timescale memory system** that decouples fast per-session memory acquisition from slow cross-session consolidation. Inspired by complementary learning systems (CLS) theory (hippocampal fast encoding vs neocortical slow consolidation), it formulates offline consolidation as **region rewriting**:

1. **Fast Writer** (online): Append-only, per-session typed memory entries (semantic/procedural)
2. **Slow Consolidator** (offline): Selects a working region R ⊆ bank, treats it as read-only evidence, synthesizes a compact replacement set S that supersedes R
3. **Counterfactual Utility**: Random dropout ablation to identify load-bearing vs redundant vs harmful memories

Key equation: `B* = (B \ R) ∪ Cθ(R, T_R)` — region replacement, not per-entry CRUD.

## Key Results

| Metric | ScienceWorld | ALFWorld | WebArena |
|--------|-------------|----------|----------|
| Success Rate | 41.1% (+7.0pp vs UMEM) | 60.2% (+1.8pp) | 52.3% (best) |
| Memory Tokens | 6,947 (12× smaller) | 10,954 (6× smaller) | 927 (400× smaller) |
| Transfer | Trained on ScienceWorld only → applied to ALFWorld + WebArena without retraining |

- **Compactness is structural**: Region rewriting forces compression before learning
- **Counterfactual reward**: Suppresses redundancy without sacrificing task performance
- **Cross-domain transfer**: Consolidator trained on one domain transfers to others + different writer backbones

## Three Qualitative Patterns

1. **Slot Abstraction** (success): Replace concrete instances with generic rules → generalizable
2. **Filtering Wrong Entries** (success): Drop contradictions, emit higher-level rule
3. **Over-Abstraction** (failure): Lose task-specific details that are locally useful (e.g., exact locations in `look_at_obj_in_light`)

---

## Mapping to Our Architecture

### Existing Building Blocks (✅ Already Have)

| Auto-Dreamer Concept | Our Equivalent | Location |
|---------------------|----------------|----------|
| Fast Writer (per-session) | `TrialLog` + JSONL replays + `ReflectionQA` | `trial_log.rs`, `reflection.rs` |
| Typed Memory Bank | `DeltaMemoryState` (rank×rank matrix) + `BanditPruner` (q_values) | `delta_mem/state.rs`, `bandit.rs` |
| Region Selection | `DeltaGatedAbsorbCompress` (gated absorb by hint-δ quality) | `absorb_compress.rs` |
| Provenance Links | `AnchorTrace` (depth/reward/future_accuracy per trial) | `trial_log.rs` |
| Consolidation | `AbsorbCompress::compress()` (merge similar arms) | `absorb_compress.rs` |
| Persistence | Freeze/Thaw `repr(C)` binary I/O | `freeze.rs` |
| Counterfactual Utility | `MultiDomainMemory` with `AggregationStrategy::BanditWeighted` | `delta_mem/multi.rs` |
| Cross-domain Transfer | `MultiDomainMemory` per-domain instances | `delta_mem/multi.rs` |
| Training Reward | `HintDelta` (intrinsic log-prob shift) + ROPD Rubric | `g_zero/`, `ropd_rubric` |

### What We're Missing (❌ Gaps)

| Gap | Description | Priority |
|-----|-------------|----------|
| **Consolidation Scheduler** | No periodic trigger for "dreaming phase" — our G-Zero rounds are manual | HIGH |
| **Region Rewriting Primitive** | `AbsorbCompress` merges arms but doesn't synthesize new abstractions | HIGH |
| **Memory Decay/Forgetting** | `DeltaMemoryState` grows via `update_count` with no decay — only grows | MEDIUM |
| **Working ↔ Long-term Hierarchy** | Everything is one tier — no explicit fast/slow split | MEDIUM |
| **Replay + Imagination** | No synthetic trajectory generation from consolidated memories | LOW (modelless doesn't need this) |
| **Counterfactual Dropout** | No random ablation to identify load-bearing memories | MEDIUM |

---

## Distillation Strategy

### Modelless Path (microgpt-rs, CPU, no gradients)

This is where Auto-Dreamer's ideas have the **most immediate applicability**. We don't need LLM-based consolidation — we can apply the principles to our bandit/δ-mem system:

1. **Consolidation Scheduler** (`dreamer_cadence: k`):
   - Every `k` episodes, trigger consolidation on working region
   - Working region = recently written entries + recently retrieved entries (same as paper)
   - Feature gate: `dreamer` (depends on `bandit`)

2. **Region Rewriting for Bandit Arms**:
   - Select working region R from bandit arms (top-k recently updated)
   - Treat R as read-only evidence
   - Synthesize replacement set S by merging similar arms + computing weighted average Q-values
   - This is a **deterministic, modelless** version of the paper's LLM-based synthesizer
   - Existing: `AbsorbCompress::compress()` already does this partially

3. **Counterfactual Utility via Dropout**:
   - After consolidation, evaluate bank quality by randomly dropping entries and measuring performance delta
   - `rcf(S) = U(S) - E[U(S\{e})]` for random e
   - Identify load-bearing vs redundant entries
   - Use existing `TrialLog` metrics as utility signal

4. **Memory Decay**:
   - Add `last_access: u64` to bandit arms
   - Exponential decay: `q *= decay_factor` on consolidation events
   - Omission-based forgetting (don't rewrite = forget)

5. **Working ↔ Long-term Hierarchy**:
   - Fast tier: `DeltaMemoryState` (current, used at inference)
   - Slow tier: `FrozenBank` (consolidated, loaded on demand)
   - Consolidation moves entries from fast → slow tier

### Model-Based Path (riir-ai, GPU, gradients)

Less direct applicability since the paper operates on textual memory, not model weights. But the two-timescale principle reinforces our existing patterns:

1. **G-Zero Loop IS already a consolidator** — it filters experience (DeltaFilter) and trains from it (DPO/GRPO)
2. **SHINE Hypernet** generates LoRA from context — analogous to synthesizing new memories from evidence
3. **The paper's GRPO training** of the consolidator maps to our existing GRPO pipeline in `gzero_loop.rs`
4. **No new model-based work needed** — the principle is already embedded

---

## Verdict

### Adoptability: ⚠️ SELECTIVE ADOPTION

**What to adopt:**
1. ✅ **Consolidation scheduler** (region rewriting cadence) — directly applicable to bandit/δ-mem
2. ✅ **Counterfactual utility reward** — adds quality signal to compression
3. ✅ **Memory decay/forgetting** — addresses our growing-without-bound issue
4. ✅ **Working region concept** — select R = recent writes + recent retrievals

**What NOT to adopt:**
1. ❌ **LLM-based synthesizer** — overkill for our modelless path, our deterministic merge is sufficient
2. ❌ **Provenance-linked trajectory retrieval** — our trajectories are game states, not text traces
3. ❌ **Tool-use rollout** — our consolidation is O(1) deterministic, not iterative LLM calls
4. ❌ **Textual memory entries** — our memory is numeric (Q-values, state matrices), not natural language

### Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Over-abstraction (Pattern 3 failure) | Preserve concrete "slot" entries alongside abstract rules |
| Consolidation cadence sensitivity | Paper shows k=5-10 is robust; we can tune per-domain |
| Memory bank becomes too small | Counterfactual utility naturally preserves load-bearing entries |
| Feature gate explosion | Single `dreamer` gate that composes with existing `bandit`/`delta_mem` |

### Expected Impact

- **Memory compactness**: 5-10× reduction in bandit arm count (paper achieves 12×)
- **Cross-session stability**: Freeze/Thaw + Dreamer = persistent compact knowledge
- **No perf regression**: Counterfactual utility ensures load-bearing memories survive

---

## Proposed Feature Gate

```toml
[features]
dreamer = ["bandit"]  # Offline memory consolidation scheduler
```

New module: `src/pruners/dreamer/`
- `mod.rs` — index
- `types.rs` — `DreamerConfig`, `WorkingRegion`, `ReplacementSet`
- `scheduler.rs` — consolidation cadence, region selection
- `consolidator.rs` — region rewriting logic (deterministic, modelless)
- `counterfactual.rs` — dropout-based utility estimation
- `decay.rs` — memory forgetting policy

---

## Key Citations

- Complementary Learning Systems: McClelland et al. 1995, 2016
- GRPO: Shao et al. 2024 (DeepSeekMath)
- LightMem: Fang et al. 2025 (closest architectural counterpart — two-timescale prompted)
- UMEM: Ye et al. 2026 (strongest RL-trained baseline)
- Sleep-time Compute: Lin et al. 2025 (offline pre-computation)