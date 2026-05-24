# Research 081: ToaST — Tokenization with Split Trees

**Paper:** [Tokenization with Split Trees](https://arxiv.org/abs/2605.22705)
**Authors:** Craig W. Schmidt, Michael Krumdick, Adam Wiemerslage, Seth Ebner, Varshini Reddy, Yuval Pinter, Chris Tanner (Kensho Technologies, Ben-Gurion University, MIT)
**Date:** 2026-05-21
**Reviewed:** 2026-05-25

---

## Summary

ToaST introduces a subword tokenization method that decouples **tree construction** from **vocabulary selection**. Instead of BPE's greedy bottom-up merges or UnigramLM's ablative top-down approach, ToaST:

1. **Builds split trees** — greedy binary trees from byte n-gram counts, independent of vocabulary
2. **Selects vocabulary via LP** — Integer Program relaxed to Linear Program, minimizing total token count under recursive descent inference
3. **Near-optimal** — LP relaxation gap < 10⁻⁷ (worst case), 44/128 instances fully integral

### Key Results
- **11%+ compression gain** over BPE/WordPiece/UnigramLM at vocab ≥ 40,960
- **14–19× fewer single-byte fallback tokens** at vocab 65,536
- **Highest CORE score** in 1.5B parameter models (+2.6%–7.6% over baselines)
- **Best on 13/22 individual tasks**
- Training scales **quadratically** in number of split trees (700k trees ≈ 99% coverage)
- Inference: Python at ~600k tok/s vs Rust BPE at ~1000k tok/s (only 1.6× gap)

---

## Core Mechanism

### Split Tree Construction
- Pre-tokenize with regex (length-limited GPT-4o regex)
- For each pretoken, recursively split into binary tree using `min(c_p, c_p')` criterion
- Splits maximize minimum count of both children (avoids lopsided trees)
- Trees built once from n-gram statistics; **vocabulary-independent**

### Inference (Recursive Descent)
- Walk tree top-down; emit first in-vocabulary node
- If node not in vocab → recurse into children
- Removing token `t` from vocab simply descends deeper — **no cascading effects** (unlike UnigramLM/PathPiece)

### Vocabulary Selection (LP)
- Variables: `x_i ∈ {0,1}` (token in vocab), `z_jk ∈ {0,1}` (node used in tokenization)
- Objective: minimize `Σ_j c_j Σ_k z_jk` (total training tokens)
- Constraints: vocab size = m, leaf coverage (every byte covered), `z_jk ≤ x_ijk`
- LP relaxation near-integral → simple rounding heuristic suffices

---

## Distillation Ideas for katgpt-rs / riir-ai

### 1. Split Tree Inference as Alternative to BPE (`katgpt-rs/src/tokenizer/`)
**Verdict: ✅ Valuable — modelless compression gain**

Our `bpe.rs` uses standard greedy BPE merge. ToaST's recursive descent is:
- **Simpler to implement** than BPE merge loop (no rank-based pair search)
- **No merge rules needed** — just a vocabulary set + split trees
- **Vocabulary-agnostic** — same trees work for any vocab size
- **Better compression** → fewer tokens → extended effective context length

The split tree structure is a natural fit for Rust's recursive data structures. We already have:
- `BpeTokenizer` with `vocab_to_id`, `id_to_vocab`, `merges`
- `MergeRule` struct
- Feature gate pattern in `Cargo.toml`

**Integration point:** Add `toast_tokenizer` feature gate, implement alongside `bpe.rs`.

### 2. LP Vocabulary Optimization via `good_lp` (Model-Based)
**Verdict: ✅ Direct fit — we already have `good_lp` with HiGHS**

We already depend on `good_lp` (HiGHS solver) for Percepta MILP scheduling (Plan 064 TG-D). ToaST's IP formulation maps directly:

- Decision variables: token inclusion + node activation
- Objective: minimize weighted token count
- Constraints: vocab cardinality, leaf coverage, z-x linking
- LP relaxation → round (paper shows < 0.012% fractional variables)

This is **model-based training** — requires corpus statistics (pretoken counts + n-gram counts).

### 3. Hierarchical Split Preferences (Modelless Enhancement)
**Verdict: 🔬 Research direction — morpheme-aligned splits**

Paper's Appendix A describes a preference hierarchy:
- Superwords → morphemes → character boundaries → bytes
- Each level uses same `min(c_p, c_p')` criterion
- **No vocabulary change needed** — only affects tree construction

This could improve tokenization for:
- Code (preserve identifier boundaries)
- Games (preserve entity names as single tokens)
- Multilingual (preserve multi-byte character integrity)

### 4. Compression ↔ Effective Context Length
**Verdict: ✅ Immediate practical benefit**

ToaST's 11% compression gain means 11% more text fits in the same context window. For our projects:
- `riir-engine`: Gemma 2 inference gets longer effective context
- `katgpt-rs`: All downstream tasks benefit from more input per token
- Game state serialization: compressed token sequences for replay/training

### 5. Rényi Efficiency Improvement
**Verdict: 📊 Metric worth tracking**

ToaST achieves substantially higher Rényi efficiency (α=2.5) due to fewer high-frequency single-byte tokens. This metric could be added to our benchmark infrastructure.

---

## What We Already Have (Adaptation Map)

| ToaST Component | Our Equivalent | Gap |
|---|---|---|
| Pre-tokenizer regex | `bpe.rs` char-level split | Need length-limited GPT-4o regex |
| Byte n-gram counts | None | Need to build from corpus |
| Split tree structure | None | New `SplitTree` type needed |
| LP vocabulary optimizer | `good_lp` + HiGHS | Direct fit, need IP formulation |
| Inference (recursive descent) | `BpeTokenizerImpl::encode` | New `ToastTokenizerImpl::encode` |
| Token types | `BpeTokenizer`, `MergeRule` | New `ToastTokenizer` without merges |

---

## Verdict

### ⚡ HIGH VALUE — Implement as feature-gated alternative

**Why:**
1. **11%+ compression gain** is substantial — effectively free context extension
2. **We already have `good_lp`/HiGHS** for LP solving — zero new deps
3. **Simpler inference** than BPE (no merge rule search, just tree descent)
4. **Vocabulary-vocabulary independence** means we can swap vocab sizes without retraining trees
5. **Rust-native advantage** — paper's Python is only 1.6× slower than Rust BPE; our Rust implementation should match or beat BPE

**Scope:**
- **Modelless (katgpt-rs):** Split tree inference engine — tree construction, recursive descent tokenizer, vocabulary I/O
- **Model-based (riir-ai):** LP vocabulary optimizer — pretoken aggregation, n-gram counting, IP formulation, LP solving via `good_lp`

**Risk:** Low. ToaST is mathematically clean (LP relaxation near-integral), implementation is straightforward, and we have all dependencies.

**Priority:** Medium-high. Tokenizer improvement benefits all downstream tasks but is not on the critical path of current plans.

### Feature Gate Design
```toml
# katgpt-rs/Cargo.toml
toast_tokenizer = []  # ToaST split-tree tokenization — inference only (Plan 120)
toast_trainer = ["toast_tokenizer", "good_lp"]  # + LP vocabulary optimization (model-based)
```

---

## References
- Schmidt et al. (2026). Tokenization with Split Trees. arXiv:2605.22705
- Gage (1994). BPE original compression technique
- Kudo (2018). UnigramLM subword regularization
- Schmidt et al. (2024). PathPiece — tokenization is more than compression
- HiGHS LP solver: https://highs.dev/