# Research 87: ConvexTok — Tokenisation via Convex Relaxations

> **Paper:** [Tokenisation via Convex Relaxations](https://arxiv.org/pdf/2605.22821) — Tempus, Whittington, Schmidt, Komm, Pimentel (ETH Zurich, Kensho Technologies), May 2026
> **Date:** 2026-05, distilled 2026-05
> **Related Research:** 081 (ToaST Split Trees), 017 (Fast BLT — negative result)
> **Related Plans:** 122 (ToaST tokenizer — completed), 127 (ConvexTok LP vocabulary optimizer — proposed)
> **Reference Code:** `.raw/tokenisation_lp/` (paper authors' implementation)
>
> **PASS-Redirects (synthesis):** Arzt, Hanbury & Blevins [arXiv:2608.15129 "Left-Branching Transformers Excel at Right-Branching Languages: Data Shapes Word Order Preferences in Language Models"] — its BPEC metric (corpus NLL normalized by the ENGLISH character count of the parallel corpus, Cotterell et al. 2018) is the cross-tokenizer/cross-script comparability trick: the cross-lingual sibling of riir-data's shipped `bits_per_char(log_probs, original_chars)` / `bpb_from_loss` (`metrics/bpb_eval`). PASS: our tokenizer league is same-corpus ConvexTok-vs-BPE (`bench_comparison.rs`), so cross-lingual normalization has no consumer today; recorded here as the ready-made metric if a cross-tokenizer perplexity league (e.g. Bonsai vs Qwen tokenizer on identical text) ever lands.

---

## Summary

ConvexTok reformulates tokeniser vocabulary construction as a **Linear Program (LP)** and solves it via convex relaxation, yielding globally near-optimal tokenisers (within 1% of proven lower bound at vocab ≥ 128k). It consistently beats BPE on bits-per-byte (BpB) and intrinsic metrics, though downstream task (CORE) gains are less consistent.

**Key innovation:** Tokenisation graph → Integer Program (IP) → LP relaxation → rounding schemes. The LP dual provides a **certifiable lower bound** on compression, enabling optimality gap measurement for any tokeniser.

---

## Core Algorithm

### 1. Tokenisation Graph Construction

Given dataset D = {b₁, ..., bₙ} of byte-strings:

- **Vertices (V):** One per inter-byte position per string + start/end. Last vertex of string n merged with first vertex of string n+1.
- **Byte-edges (E_byte):** Connect adjacent vertices within each string.
- **Token-edges (E_tok):** Connect non-adjacent vertices (span ≥ 2 bytes) within each string.
- **Colour-partition (C):** Token-edges grouped by the byte-substring they represent. Each colour = one potential token.

```
Example: D = {abaa, aba}

  a → b → a → a ═══╗
                    ║
  a → b → a ════════╝

  → = byte-edges, ═ = token-edges (coloured by substring)
```

### 2. Integer Program (NP-hard)

Variables:
- `f ∈ {0,1}^F` — free edge usage (byte-edges)
- `p ∈ {0,1}^P` — priced edge usage (token-edges)
- `c ∈ {0,1}^C` — colour selection (vocabulary membership)

```
min  ⟨1, p⟩ + ⟨1, f⟩           (minimize total tokens = compression)
s.t. Pp + Ff = d               (flow conservation → valid segmentation)
     p - Cc ≤ 0                (can only use edges whose colour is selected)
     ⟨1, c⟩ ≤ K                (vocabulary budget)
     f, p, c ∈ {0,1}           (integrality)
```

### 3. LP Relaxation (Polynomial-time)

Relax integrality: `{0,1}` → `[0,1]`. The resulting LP is solvable in polynomial time.

**Key empirical finding:** Solutions are nearly integral at large vocab sizes (91% integral at 128k, 91% at 256k). This means rounding is less critical at production vocab sizes.

### 4. Rounding Schemes

| Scheme | Strategy | Best For |
|--------|----------|----------|
| **Det** (Deterministic) | Top-K colours by LP value `c` | BpB (bits-per-byte) — **consistently best** |
| **Bias** (Biased) | Top-K by `c / token_length` | Intrinsic metrics (compression, vocab util) — favours shorter tokens for OOD generalization |
| **Int** (Integral-only) | Keep only `c ≥ 0.999` | Certification — reveals which tokens the LP considers "forced" |

After rounding `c`, optimal `p` and `f` are recovered via shortest path.

---

## Key Results (Paper Tables)

### Compression Certification (Table 2)

| Vocab | BPE Gap | Det Gap | Bias Gap | Int Gap |
|-------|---------|---------|----------|---------|
| 8k | 3.35% | 0.86% | 4.86% | 22.63% |
| 32k | 1.29% | 0.07% | 1.11% | 3.95% |
| 128k | 0.39% | 0.23% | 0.01% | 0.41% |
| 256k | 0.21% | ~0.00% | 0.07% | 0.30% |

**Key insight:** At vocab ≥ 128k, all tokenisers (including BPE) are within 1% of LP-proven optimal. BPE is already close to optimal for compression. The remaining gains are marginal.

### Bits-per-Byte (Table 4, Depth 12)

| Vocab | BPE | Det | Bias | Δ Det vs BPE |
|-------|-----|-----|------|-------------|
| 8k | 0.8785 | 0.8782 | 0.8812 | -0.0003 |
| 32k | 0.8536 | 0.8525 | 0.8531 | -0.0011 |
| 128k | 0.8410 | 0.8403 | 0.8393 | -0.0007 |
| 256k | 0.8411 | 0.8398 | 0.8402 | -0.0013 |

Det wins BpB at all vocab sizes. Gains are **small but consistent** (~0.1-0.15%).

### Downstream CORE (Table 4, mixed)

Results are inconsistent. ConvexTok matches or slightly outperforms BPE at larger vocab sizes (128k, 256k) but the trend is noisy at smaller sizes. **Not a clear win for downstream tasks.**

### LP Solution Properties (Table 1)

At 32k vocab: 81.5% of colour variables are integral (already 0 or 1). Only ~18% need rounding. At 256k: 90.5% integral. **Rounding matters less at production scales.**

### Stability (Figure 3)

BPE is consistently more stable than ConvexTok across training data resampling (higher Jaccard similarity). This is expected — BPE's greedy frequency-driven merges are robust to sampling due to power-law token distributions.

---

## Distillation for katgpt-rs / riir-ai

### What We Already Have

1. **ToaST split-tree tokenizer** (Plan 122, `toast_tokenizer` feature gate) — modelless inference engine with recursive descent. Complete, GOAT-proven 17/17.

2. **LP solver infrastructure** — `good_lp` with HiGHS backend already in `Cargo.toml` (used for Percepta MILP scheduling, Plan 064 TG-D).

3. **Reference implementation** — `.raw/tokenisation_lp/` contains the paper authors' Python code including LP construction, rounding schemes, and tokenizer training.

### Architecture Fit

```
ConvexTok Pipeline:
  Corpus → Pre-tokenize → Build Tokenisation Graph → Solve LP → Round → Tokenizer

Our Pipeline (ToaST + ConvexTok):
  Corpus → Pre-tokenize → Count n-grams → Build Split Trees (ToaST)
                                ↓
                     Build Tokenisation Graph → Solve LP → Round
                                ↓
                     Select vocabulary from LP solution
                                ↓
                     ToaST inference with LP-optimized vocab
```

**Synergy:** ConvexTok solves the **vocabulary selection** problem (which tokens to include). ToaST solves the **segmentation** problem (how to tokenize with a given vocabulary). They are **orthogonal and complementary**.

### Modelless vs Model-Based

| Aspect | Modelless (katgpt-rs) | Model-Based (riir-ai) |
|--------|----------------------|----------------------|
| LP graph construction | Types only, no corpus needed | Full corpus pipeline |
| LP solving | `good_lp`/HiGHS (already available) | Same solver |
| Rounding schemes | All three (Det/Bias/Int) | Same |
| Training data | Pre-built vocab file | Live from ClimbMix/custom |
| Usage | Load pre-optimized vocab | Build vocab on-the-fly |

**Recommendation:** katgpt-rs gets the **modelless inference path** (load pre-built LP-optimized vocab, use with ToaST). riir-ai gets the **model-based training path** (LP graph construction + solving + corpus pipeline).

### LP Graph Construction in Rust

The paper's LP has dimensions:
- 105,997,943 variables (79M priced edges + 18M free edges + 8.8M colours)
- 99,168,445 constraints (20M equality + 79M inequality)

This is a **large LP** but solvable in ~4 hours on 1 GH200 (per paper). For our micro/production scales:
- Micro (vocab 256-4k): trivially small, seconds
- Production (vocab 32k-128k): moderate, minutes on CPU

The key data structures map cleanly to Rust:
- `F: SparseMatrix<V, FreeEdge>` — free incidence matrix
- `P: SparseMatrix<V, PricedEdge>` — priced incidence matrix
- `C: SparseMatrix<PricedEdge, Colour>` — edge-colour matrix
- CSR/CSC sparse format via `katgpt-core`

---

## Verdict

### ✅ ADOPT — LP Vocabulary Optimization for ToaST

**Why:**
1. **Orthogonal to ToaST.** ConvexTok selects vocabulary; ToaST segments text. Together they form a complete globally-optimal tokenization pipeline.
2. **Infrastructure exists.** `good_lp`/HiGHS already in `Cargo.toml`. ToaST types already defined. Tokenisation graph construction is pure combinatorics — no ML needed.
3. **Certifiable optimality.** The LP dual provides a proven lower bound. We can certify any tokenizer's distance from optimal. This is unique among all tokenizer research we've reviewed.
4. **Consistent BpB gains.** Det rounding consistently beats BPE on bits-per-byte. Not huge (~0.1%), but free and provably better.
5. **Synergy with ToaST Plan 122 T6.** The deferred Rényi efficiency benchmark can now use ConvexTok-optimized vocabularies.

**What NOT to adopt:**
- Full corpus training pipeline (belongs in riir-ai, not katgpt-rs)
- GPU-accelerated LP solving (NVIDIA cuOPT) — CPU `good_lp`/HiGHS sufficient for our scales
- Multilingual extensions — future work, not blocking

### Scope

| Component | Project | Feature Gate |
|-----------|---------|-------------|
| LP graph types (tokenisation graph, incidence matrices) | katgpt-rs | `convex_tok` |
| LP construction from pretokenized corpus | katgpt-rs | `convex_tok` |
| Rounding schemes (Det/Bias/Int) | katgpt-rs | `convex_tok` |
| Vocabulary import/export (ConvexTok → ToaST) | katgpt-rs | `convex_tok` + `toast_tokenizer` |
| Optimality certification (LP bound computation) | katgpt-rs | `convex_tok` |
| Full corpus n-gram counting pipeline | riir-ai | N/A |
| LM training with ConvexTok tokenizer | riir-ai | N/A |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LP too slow for large corpus | Low | Medium | Pre-built vocab files, incremental solving |
| Rounding artifacts (fractional tokens) | Low | Low | Det rounding is deterministic and stable |
| No downstream task improvement | Medium | Low | BpB improvement is free; CORE is noisy anyway |
| BPE already near-optimal at 128k+ | High | None | Paper confirms this; ConvexTok still certifies optimality |

### Honest Assessment

**The honest truth:** At production vocab sizes (128k+), BPE is already within 0.4% of LP-proven optimal compression. ConvexTok's Det rounding closes this to 0.2%. The BpB gain on LM training is ~0.1% — real but small.

**The real value is certification, not improvement.** ConvexTok lets us **prove** our tokenizer is near-optimal rather than guessing. For a research-heavy codebase that already has ToaST infrastructure and LP solver dependencies, this is essentially free knowledge.

**Expected GOAT outcome:** 10-12 tests pass (types, construction, rounding, certification, ToaST interop). This is a solid engineering contribution, not a breakthrough.

---

## Key Equations (for Implementation)

### LP Formulation

```
min  Σ p_e + Σ f_e                    (total path length = compression)
s.t. P·p + F·f = d                    (flow: source -1, sink +1, internal 0)
     p_e ≤ c_{colour(e)}  ∀e ∈ P     (edge usable only if colour selected)
     Σ c_c ≤ K                        (vocabulary budget)
     0 ≤ f, p, c ≤ 1                  (LP relaxation)
```

### Rounding: Det

```
det_round(K, c):
  c' = zeros(|C|)
  for colour in top_k(c, K):
    c'[colour] = 1
  return c'
```

### Rounding: Bias

```
bias_round(K, c):
  c' = zeros(|C|)
  scored = [(c_colour / len(token_colour), colour) for colour in C]
  for (_, colour) in top_k(scored, K):
    c'[colour] = 1
  return c'
```

### Rounding: Int

```
int_round(c):
  c' = zeros(|C|)
  for colour in C:
    if c[colour] ≥ 0.999:
      c'[colour] = 1
  return c'
```

### Optimality Gap

```
gap(T) = (f(T) - LP_value) / LP_value × 100%
```

Where `f(T) = Σ_{b∈D} |tok(b)|` is the compression of tokenizer T, and `LP_value` is the LP objective (proven lower bound).

---

## References

- Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821
- Schmidt et al. (2026). Tokenization with Split Trees. arXiv:2605.22705 (our ToaST implementation, Plan 122)
- Kudo (2018). Subword regularization. ACL 2018. (Unigram LM tokenization — similar shortest-path inference)
- Williamson & Shmoys (2011). The Design of Approximation Algorithms. (LP rounding theory)
- NVIDIA cuOPT (2025). GPU-accelerated LP solver — used in paper but not needed for our scales

---

> **PASS-Redirects (synthesis):** Saha, Li, Xue, Chaudhuri, Klivans, Kothari, Meka [arXiv:2608.11158 "New Lower and Upper Bounds for the Grothendieck Constant"] — same relax→round→certify-the-gap shape at SDP scale (Krivine rounding schemes + Arb-certified inverse-majorant gap vs our LP token rounding + dual-bound optimality gap); PASS — no SDP/sign-rounding surface in the stack, no actionable delta over ConvexTok's shipped certify pattern.