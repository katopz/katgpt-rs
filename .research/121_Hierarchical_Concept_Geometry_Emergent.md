# Research 121: Hierarchical Concept Geometry Emerges from Co-occurrence

**Date:** 2026-05-27
**Paper:** arXiv:2605.23821 — "Hierarchical Concept Geometry in Language Models Emerges from Word Co-occurrence" (Nava & Wyart, JHU/EPFL, 2026)
**Verdict:** ✅ CROSS-CUTTING THEORY — Validates KG × HLA spectral extraction, strengthens Pillar 1. Not a new pillar.
**Classification:** Open (katgpt-rs) — theory is public, game-specific co-occurrence matrices stay private (riir-ai)

---

## TL;DR

The paper proves that **hierarchical splitting geometry in embeddings is an inevitable consequence of co-occurrence statistics**, not a hierarchy-specific mechanism. Under mild positivity/decay assumptions on the co-occurrence kernel f(dist(i,j)), the eigenvectors of the Gram matrix organize coarse-to-fine, mirroring the tree topology. This is confirmed in both word2vec and Gemma 2B unembeddings.

**Why this matters for us:** Our HLA sufficient statistics ARE co-occurrence Gram matrices. Our Cold Tier → KG extraction pipeline computes spectral structure from game episode statistics. This paper proves that if game states co-occur proportionally to their distance in a concept hierarchy, the spectral decomposition WILL reveal that hierarchy automatically. No manual hierarchy encoding needed.

---

## Core Results

### Theorem 1: Hierarchy-Aligned Eigenvectors
Under the co-occurrence model M*_{ij} = f(dist(i,j)) on a binary tree, eigenvectors decompose into:
- **Scaling modes** ϕ_ℓ: constant on depth layers (depth-only variation)
- **Wavelet modes** ψ_{u,r}: supported on subtree T(u), antisymmetric across split at u

### Theorem 2: Coarse-to-Fine Spectral Ordering
1. Largest eigenvalue is a scaling mode (Perron-Frobenius — all positive entries)
2. Leading wavelet modes are ordered coarse-to-fine: λ^(L)_1 ≥ λ^(L-1)_1 ≥ ... ≥ λ^(1)_1
3. Wavelet blocks are nested (Cauchy interlacing): A^(h) is principal submatrix of A^(h+1)
4. Non-leading scaling modes interleave with split spectrum (exponential kernel)

### Key Quantitative Prediction
The Gram matrix eigenvectors separate tree branches from coarse to fine:
- PC 1: constant (depth variation)
- PC 2: root split (e.g., animal vs plant)
- PC 3-4: second-level splits (degenerate by symmetry)
- PCs 5+: progressively finer distinctions

### Top-k Eigenspace Alignment
```
g(k) = (1/k) ||U_k^T V_k||_F^2
```
Measures alignment between empirical and theoretical leading eigenspaces. Above shuffled baseline across many WordNet subtrees — confirmed in both word2vec and Gemma 2B.

### Concept-Vector Orthogonality
Park et al.'s parent-child orthogonal innovation diagnostic (cos(ℓ_w - ℓ_p, ℓ_p) ≈ 0) is reproduced by the co-occurrence model alone. The orthogonality is **not evidence for a hierarchy-specific mechanism** — it's a spectral consequence.

---

## Mapping to Our Architecture

### 1. HLA Sufficient Statistics = Co-occurrence Gram Matrix

Our HLA (Plan 057) maintains compact prefix sufficient statistics:
- C^{QV}_t = Σ_{τ≤t} q_τ v_τ^T  (second-order interaction)
- S^K_t = Σ_{τ≤t} k_τ k_τ^T  (key Gram matrix)

**S^K_t IS a co-occurrence Gram matrix.** When keys represent game states, S^K_t accumulates how often states co-occur in the causal prefix. The paper's spectral theory applies directly:

| Paper Concept | Our Component | What We Get |
|---------------|--------------|-------------|
| M* = co-occurrence Gram | S^K = HLA key statistics | Hierarchical state organization in key space |
| Top-k eigenspace alignment | Eigendecomposition of S^K | Game concept hierarchy recovery |
| Scaling modes ϕ_ℓ | Depth-layer-constant key patterns | Game state depth encoding |
| Wavelet modes ψ_{u,r} | Subtree-contrast key patterns | Game tactic hierarchy encoding |
| Cauchy interlacing λ^(h+1)_1 ≥ λ^(h)_1 | Nested split eigenvalues | Coarse-to-fine game concept ordering |

### 2. KG Extraction Pipeline Validated (Research 010)

Research 010 proposes: self-play → Cold Tier episodes → KG extraction → KG weights → role-conditioned HLA attention.

This paper proves the **middle step**: if game states co-occur proportionally to their distance in a game-concept hierarchy (which they do — similar game states co-occur in episodes), then spectral decomposition of the co-occurrence matrix recovers the hierarchy.

**Before this paper:** "We hope spectral decomposition recovers hierarchy."
**After this paper:** "Spectral decomposition provably recovers hierarchy under our co-occurrence model."

### 3. Dirichlet Energy × Spectral Alignment (Research 111, Plan 149)

The paper's top-k eigenspace alignment metric g(k) is the same mathematical structure as our Dirichlet Energy diagnostic:
- g(k) measures how well eigenspaces align between theoretical and empirical Gram matrices
- Dirichlet Energy measures geometric alignment across functor adjacency graphs
- Both quantify "how well does the representation capture the structure"

The paper validates that **spectral alignment is the right metric** for measuring hierarchical concept organization. Plan 149 (Dirichlet Energy diagnostic) should include g(k) as a complementary metric.

### 4. PEIRA Spectral Alignment (Research 011)

PEIRA's alignment metric α = (e^T N e) / (||e|| · ||Ne||) measures signal-noise eigenvector alignment.

The paper's g(k) = (1/k) ||U_k^T V_k||_F^2 measures empirical-theoretical eigenspace alignment.

**Connection:** PEIRA alignment at convergence IS top-k eigenspace alignment between learned representation and the "true" co-occurrence structure. The paper proves what the true structure looks like (hierarchical splitting geometry). PEIRA convergence to that structure is measurable with g(k).

### 5. LEO Goal Hierarchy (Research 012)

LEO's all-goals Q-values Q(s) → R^{G×A} have implicit hierarchical structure over goals:
- Coarse goals (survive, win) subsume fine goals (kill boss, reach floor X, craft item Y)
- The paper's coarse-to-fine spectral ordering predicts: Q-value eigenvectors should organize goals hierarchically
- If goal co-occurrence (achieving goal g1 makes g2 more likely) follows the paper's decay model, LEO's learned Q-representation will have hierarchical splitting geometry

### 6. Fourier Spatial AI (Pillar 1)

Fourier encoding IS spectral encoding. The paper proves that spectral decomposition of distance-based kernels produces hierarchical geometry. Fourier hashing of game positions computes exactly this kind of distance-based similarity:
- Positions close on the game map → similar Fourier hash → "co-occur" in the MCTS tree
- The spectral structure of the Fourier hash Gram matrix should reveal game spatial hierarchy
- This validates that Fourier MCTS discovers hierarchical spatial structure automatically

---

## KG Integration Answer

**Can we use this with KG? YES — this IS the theoretical foundation for KG extraction.**

| Step | Our Pipeline | Paper's Theory |
|------|-------------|----------------|
| 1. Generate game episodes | Self-play → Cold Tier | Co-occurrence data (Wikipedia for paper, game episodes for us) |
| 2. Compute co-occurrence | State-pair frequency in episodes | M*_{ij} = f(dist(i,j)) |
| 3. Spectral decomposition | Eigendecomposition of state Gram matrix | Hierarchical splitting geometry (Theorem 1+2) |
| 4. Extract hierarchy | Top-k eigenvectors → KG nodes | Coarse-to-fine spectral ordering |
| 5. Condition attention | Role-conditioned HLA (Research 010) | Hierarchy-aligned eigenvectors as attention structure |

**What was missing in Research 010:** Theoretical guarantee that spectral decomposition recovers hierarchy.
**What this paper adds:** Proof that under our co-occurrence model (Assumption 3.1+3.2), hierarchical recovery is guaranteed.

The game-specific part (which game states to count as co-occurring, what distance metric to use) stays private in riir-ai. The spectral extraction algorithm is generic and belongs in katgpt-rs.

---

## Open/Close Split

### katgpt-rs (Open — MIT)
- `spectral_hierarchy` utility: eigendecomposition + top-k eigenspace alignment g(k)
- `CooccurrenceGram<H>` trait: generic Gram matrix from distance kernel on any hierarchy
- `HaarWaveletBasis` utility: scaling modes + wavelet modes for binary/s-ary trees
- `cauchy_interlacing_check`: validation that split blocks satisfy interlacing
- Feature gate: `spectral_hierarchy = []` — default-off, used for KG extraction and diagnostics

### riir-ai (Private — Game Knowledge)
- Game state co-occurrence matrix construction from Cold Tier episodes
- Game-specific distance kernel f(d) fitting (which decay rate β for which game)
- KG node extraction from spectral hierarchy (which eigenvectors → which game concepts)
- Role transport operators derived from hierarchy-aligned eigenvectors
- Cross-game spectral alignment (do Bomber/Go/TFT share hierarchical structure?)

---

## What This Does NOT Do

1. **Not a new pillar** — theoretical validation of existing architecture
2. **Not game-specific** — the math is general; game specifics stay private
3. **Not a training method** — this is spectral analysis, not a training objective
4. **Does not replace PEIRA** — PEIRA is a training loss; this is a diagnostic/theory
5. **Does not change inference** — no perf impact on the inference path

---

## Impact on Decision Matrix

| Aspect | Before Paper | After Paper |
|--------|-------------|-------------|
| KG extraction (Research 010) | "Hope spectral works" | "Proven to work under our model" |
| Dirichlet Energy (Plan 149) | One alignment metric | g(k) as complementary metric |
| PEIRA convergence (Research 011) | Abstract alignment target | Concrete hierarchical splitting geometry target |
| LEO goal organization (Research 012) | Flat goal set | Hierarchical goal spectral structure |
| Fourier spatial encoding (Pillar 1) | Position hashing | Spectral position hierarchy discovery |

**Net effect:** Strengthens confidence in Research 010 pipeline, adds g(k) diagnostic to Plan 149, validates spectral alignment as the right metric across all cross-cutting improvements.

---

## Feature Gate Decision

**`spectral_hierarchy` — default-OFF, opt-in**

Rationale:
- This is a diagnostic/extraction tool, not an inference component
- Zero perf impact on inference (offline analysis only)
- Used during KG extraction from Cold Tier data, not during gameplay
- If proven useful for real-time attention conditioning → promote to default-on

No perf hurt → can be default-on IF we prove it helps at inference time. For now, it's offline analysis.

---

## Relationship to Existing Research

| Existing | Overlap | Delta |
|----------|---------|-------|
| Research 010 (KG × HLA) | KG extraction via spectral decomposition | This paper proves the spectral extraction works theoretically |
| Research 011 (PEIRA) | Spectral alignment metrics | This paper defines the target alignment structure |
| Research 111 (Analogy) | Hierarchical geometry | This paper explains WHY it emerges (co-occurrence, not mechanism) |
| Research 012 (LEO) | Goal hierarchy | This paper predicts goal spectral ordering |
| Plan 149 (Dirichlet Energy) | Alignment diagnostic | g(k) as complementary metric |
| Plan 151 (KG Role Transport) | Hierarchy → role conditioning | Hierarchy-aligned eigenvectors = role transport directions |

---

## References

- Nava & Wyart (2026). "Hierarchical Concept Geometry in Language Models Emerges from Word Co-occurrence." arXiv:2605.23821
- Park, Choe, Jiang, Veitch (2025). "The Geometry of Categorical and Hierarchical Concepts in LLMs." ICLR 2025.
- Korchinski et al. (2025). "On the Emergence of Linear Analogies in Word Embeddings." NeurIPS 2025.
- Karkada et al. (2025). "Closed-form Training Dynamics Reveal Learned Features and Linear Structure." NeurIPS 2025.
- Research 010: KG × HLA × Role Transport
- Research 011: PEIRA Game View Alignment
- Research 111: Emergent Analogical Reasoning
- Research 012: LEO All-Goals Learning
