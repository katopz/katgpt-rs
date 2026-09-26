# Research 090: Epiplexity — Structural Information for Computationally Bounded Observers

**Paper**: [arXiv:2601.03220](https://arxiv.org/pdf/2601.03220) (Mar 2026)
**Authors**: Marc Finzi*, Shikai Qiu*, Yiding Jiang, Pavel Izmailov, J. Zico Kolter, Andrew Gordon Wilson (CMU / NYU)
**Verdict**: ✅ High Value — directly validates our modelless distillation, G-Zero self-play, and data selection pipeline; provides theoretical framework for `ScreeningPruner::relevance()` upgrade

> **PASS-Redirects (synthesis):** Cowsik, Dolev, Li, De Luca, Cohen, Goodman, Levine [arXiv:2609.30063 "Self-Play Pretraining with Zero Data"] — uses epiplexity (S_T growth across generator checkpoints) as the generator-health instrument for a self-play pretraining loop (their Fig. 3), two years after this note shipped it as a data-selection pruner; full distill + fusion at `riir-train/.research/458` and `katgpt-rs/.research/591`.

---

## Core Concepts

### Epiplexity (S_T)
Formalization of **structural information** extractable by a computationally bounded observer within time T. Defined as the program length |P*| that minimizes the time-bounded Minimum Description Length (MDL):

```
P* = argmin_{P ∈ P_T} { |P| + E[log 1/P(X)] }
S_T(X) := |P*|  (epiplexity — structural bits)
H_T(X) := E[log 1/P*(X)]  (time-bounded entropy — random bits)
```

### Time-Bounded Entropy (H_T)
The **random, unpredictable** component of information. CSPRNGs have near-maximal H_T but near-zero S_T — exactly matching intuition that pseudorandom data carries no learnable structure.

### Key Property: MDL_T(X) = S_T(X) + H_T(X)
Total information decomposes into structural (epiplexity) + random (entropy). For uniform random U_n: S_T ≈ constant, H_T ≈ n. For structured data: S_T grows with dataset size, H_T decreases per-token.

---

## Three Paradoxes Resolved

### Paradox 1: Information Cannot Be Created by Deterministic Transformations
- **Classical**: Data processing inequality says I(Y;W) ≤ I(X;W)
- **Reality**: AlphaZero, synthetic data, cellular automata (ECA Rule 54) create structural information
- **Resolution**: With bounded compute, deterministic f can increase MDL_T if f⁻¹ is hard to compute
- **Our connection**: G-Zero self-play (Plan 049) is exactly this — simple game rules → complex strategies via computation

### Paradox 2: Information Is Independent of Factorization Order
- **Classical**: H(Y|X) + H(X) = H(X,Y) = H(X|Y) + H(Y)
- **Reality**: LLMs learn better left-to-right than reverse; chess board→moves vs moves→board differ
- **Resolution**: One-way functions create asymmetry: H_Poly(X|Y) + H_Poly(Y) ≠ H_Poly(Y|X) + H_Poly(X) + O(1)
- **Our connection**: Data ordering in game traces (Plan 124), forward vs reverse factorization for distillation

### Paradox 3: Likelihood Modeling Is Merely Distribution Matching
- **Classical**: argmin_P E[-log P(X)] = Q (true distribution)
- **Reality**: Models learn induction circuits and emergent structures NOT in the generating process
- **Resolution**: Computationally bounded observers must learn richer programs than the generator
- **Our connection**: Modelless distillation extracts more structure than the teacher model encoded (Plans 052, 071, 072)

---

## Practical Measurement: Prequential Coding

The simplest epiplexity estimator — **area under the loss curve above final loss**:

```
|P_preq| ≈ Σ_{i=0}^{M-1} (log 1/P_i(z_i) - log 1/P_M(z_i))
```

This is the cumulative excess loss during training. Key insight: **we already compute loss curves during training** — epiplexity is essentially free to estimate.

### Requential Coding (more rigorous)
Cumulative KL divergence between teacher and student checkpoints:

```
|P_req| ≈ Σ_{i=0}^{M-1} KL(P_t^i || P_s^i)
```

Teacher-student gap integrated over training — directly applicable to our distillation pipeline.

---

## Key Results for Our Stack

### 1. Chess Data: Reverse Order = Higher Epiplexity + Better OOD
- Forward: moves → board (easy to compute, low S_T)
- Reverse: board → moves (requires inference, high S_T)
- **Result**: Reverse order yields higher epiplexity AND better downstream puzzle/centipawn accuracy
- **Distill**: Our game trace ordering should prefer inference-hard factorizations

### 2. ECA Rule 54: Emergence Creates Epiplexity
- Class IV cellular automata produce both random AND structural information
- Models trained on Rule 54 data transfer better to downstream tasks than Rules 15/30
- **Distill**: Complex game dynamics (Go, Bomber) are high-epiplexity training data

### 3. ADO Validation: Epiplexity Tracks Data Quality
- Adaptive Data Optimization (Jiang et al., 2025) selects data with faster-decreasing loss
- Paper confirms ADO achieves higher epiplexity than uniform sampling
- ADO also achieves better downstream performance + OOD perplexity
- **Distill**: Our `ScreeningPruner::relevance()` should weight by epiplexity signal

### 4. Language > Images for Transfer
- Text has highest S_T/D ratio among modalities
- Image pixels are >99% random information (H_T)
- VQ tokenization increases S_T by focusing on semantic structure
- **Distill**: Tokenization choices affect epiplexity; our ConvexTok work (Plan 127) aligns

### 5. Scaling Law Integration
- Epiplexity grows as S_T ∝ T^{α(1-β)/(β+1)} in compute-limited regime
- Saturates at S_∞ = (β/(1-β)) · D_0^β · D^{1-β} when compute is abundant
- Higher β and D_0 → more epiplexity per token
- **Distill**: Dataset-dependent; explains why some domains transfer better

---

## Distillation Ideas for katgpt-rs

### Idea 1: Prequential Epiplexity Scoring
Replace/augment `ScreeningPruner::relevance()` with epiplexity-aware scoring:
- Track per-position loss during training
- Compute area-above-final-loss as epiplexity proxy
- Use as bandit reward signal for data selection

### Idea 2: Epiplexity-Gated Modelless Distillation
In GFlowNet (Plan 052) / SDAR (Plan 072) / ROPD (Plan 071):
- Weight distillation loss by per-sample epiplexity estimate
- High-epiplexity samples → stronger distillation signal
- Low-epiplexity (random) samples → weaker signal or skip

### Idea 3: Factorization-Aware Game Traces
For Event Log / Game Trace (Plan 124):
- Evaluate forward vs reverse factorization epiplexity
- Prefer inference-hard orderings (board→moves, state→actions)
- Use as free data augmentation strategy

### Idea 4: SR²AM Epiplexity Context
Extend `ConfiguratorContext` (Plan 112):
- Add `epiplexity_bin` alongside `entropy_bin`
- High epiplexity + low entropy → extend plan (structure-rich, predictable)
- Low epiplexity + high entropy → skip plan (random, unpredictable)
- Better planning decisions than entropy alone

### Idea 5: Freeze/Thaw Epiplexity Checkpointing
For knowledge pipeline (Plan 092):
- Measure epiplexity at each freeze/thaw phase
- Phases that increase S_T are learning structure (keep)
- Phases that only decrease H_T are memorizing (prune)

---

## Feature Gate Design

```toml
[features]
epiplexity_scoring = []  # opt-in: prequential epiplexity estimator
epiplexity_bandit = ["epiplexity_scoring", "bandit"]  # SR²AM integration
```

### New Types (behind feature gate)
- `EpiplexityEstimator` — tracks loss curve, computes area-above-final
- `EpiplexityScreeningPruner<P>` — wraps any ScreeningPruner, weights by epiplexity
- `EpiplexityContext` — extends ConfiguratorContext with S_T bin

---

## What NOT to Distill

1. **Full requential coding** — requires teacher-student KL at every step, expensive; prequential is sufficient for ranking
2. **Scaling law estimation** — needs many training runs at different scales; not practical for inference-time
3. **Cryptographic proofs** — Theorems 9-13 are theoretical backing, not implementation targets
4. **CSPRNG analysis** — interesting but not relevant to our data pipeline
5. **MDL optimal program search** — intractable; neural network proxy is the practical approach

---

## Honest Assessment

### Strengths for Our Stack
- **Directly validates G-Zero** (AlphaZero is a primary example in the paper)
- **Prequential coding is nearly free** — we already have loss curves
- **Chess experiments** match our game arenas perfectly
- **ADO validation** confirms adaptive data selection works
- **Theoretical backing** for modelless distillation being meaningful

### Limitations
- Epiplexity is a **ranking metric**, not a guaranteed generalization predictor
- Requires training run to estimate — not applicable to cold-start
- Paper focuses on pre-training scale; our inference-time use is a novel extension
- No closed-form solution; depends on model class and compute budget
- Per-sample epiplexity estimation is noisy; batch-level is more reliable

### Risk Assessment
- **Low risk**: Prequential scoring as auxiliary signal (can always fall back to entropy)
- **Medium risk**: Replacing relevance() with epiplexity-weighted version (needs benchmarking)
- **High risk**: Using epiplexity as sole data selection criterion (paper warns it's not task-specific)

---

## References

- Finzi et al., "From Entropy to Epiplexity", arXiv:2601.03220, Mar 2026
- Jiang et al., "Adaptive Data Optimization", ICLR 2025 — validated by epiplexity
- Zhang et al., "Intelligence at the Edge of Chaos", arXiv:2410.02536 — ECA downstream transfer
- Koppel, "Structure", 1988 — sophistication (epiplexity precursor)
- Rissanen, "MDL Principle", 2004 — theoretical foundation