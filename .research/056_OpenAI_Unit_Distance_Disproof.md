# Research 56: OpenAI Unit Distance Disproof — Number-Theoretic Constructions for GOAT Proofs

> **Paper 1:** [Planar Point Sets with Many Unit Distances](https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf) — OpenAI (2026, 18 pages)
> **Paper 2:** [Remarks on the Disproof of the Unit Distance Conjecture](https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-remarks.pdf) — Alon, Bloom, Gowers, Litt, Sawin, Shankar, Tsimerman, Wang, Wood (2026, 19 pages)
> **Date:** 2025-07
> **Related:** Plan 090 (unit_distance_goat), Research 21 (G-Zero), Research 37 (REAP), Research 51 (Deep Manifold)
> **Supersedes:** None — new domain

## Executive Summary

OpenAI's internal model **autonomously disproved Erdős's 1946 unit distance conjecture** — one of the most famous open problems in discrete geometry (Erdős Problem #90, $500 prize). The AI found a counterexample showing ν(n) ≥ n^(1+δ) for some δ > 0, refuting the widely believed bound ν(n) ≤ n^(1+o(1)).

**The key insight for our system:** The proof uses an **infinite tower of number fields with bounded root discriminant** (Golod–Shafarevich construction) where fixed rational primes split completely. This creates exponentially many "unit translations" — algebraic numbers of absolute value 1 in every embedding. The construction is a **modelless search** through a high-dimensional lattice, projecting to the plane.

**Distillation potential:**
1. **Lattice-based proof search** — The Minkowski lattice averaging argument is a generic tool for proving combinatorial bounds via number fields
2. **Class-field tower as search space** — Infinite towers with controlled discriminants give unbounded exploration without parameter explosion
3. **Chebotarev as pruning** — Prescribed splitting conditions = constraint satisfaction in algebraic number theory
4. **AI discovery methodology** — The model tried paths humans dismissed; it combined algebraic number theory + discrete geometry across domain boundaries

**Verdict:** High value for GOAT proof infrastructure. The mathematical technique (lattice averaging + pigeonhole on class groups) is a reusable pattern for combinatorial optimization proofs. The AI discovery methodology validates our model-based/modelless bandit approach.

---

## Paper Core (Proof Paper)

### Theorem (1.1)
There exists δ > 0 and infinitely many n such that ν(n) ≥ n^(1+δ), where ν(n) = max number of unit-distance pairs among n planar points.

### Three-Part Architecture

#### Part 1: Geometric Criterion (Section 2)
Given a CM field K = L(i) with L totally real of degree f:
1. **Pigeonhole on class group:** 2^(tf) ideal configurations / h(K) classes → ≥ e^(γf) norm-one elements
2. **Minkowski lattice embedding:** Λ = D^(-1)O_K in C^f
3. **Averaging over cosets:** Expected unit-distance pairs scale as |U|·ρ_R^f / covol(Λ)
4. **Injective projection:** π₁: X → C is injective (field embedding kills zero-conjugate elements)
5. **Packing bound:** |X| ≤ e^(Bf) via D-separated sup-norm packing

Key constants: δ = γ/(4B) > 0 where γ = t·log(2) - log(H) and B = 2·log(4RD).

#### Part 2: Field Construction (Section 3)
1. Start with cyclic cubic F from cyclotomic subfields Q(ζ_ri)
2. Build unramified pro-3 tower via Golod–Shafarevich (r ≤ d²/4 → infinite)
3. Kill Frobenius classes in Φ(G) → prescribed complete splitting
4. Root discriminant preserved: rd(F_j) = rd(F) = O(ℓ·log(ℓ))
5. After adjoining i: rd(K_j) ≤ 2·rd(F), h(K_j) ≤ H_ℓ^(f_j)

#### Part 3: Numerical Balance
- t = ⌊(ℓ-1)²/100⌋ split primes (quadratic in generators)
- H_ℓ = (2·rd(F))^(2C_class), log(H_ℓ) = O(ℓ·log(ℓ))
- t·log(2) > log(H_ℓ) for large ℓ → γ > 0

### Simplified Proof (Remarks Paper)

The remarks paper by Alon, Bloom, Gowers et al. simplifies using:
- **Pro-2 towers** instead of pro-3 (simpler notation)
- **Single split prime** q = 101 suffices (vs t primes in original)
- **T = {3,5,7,11,13,17}** with S = {101,∞}
- Explicit exponent: δ ≈ 6.24 × 10^(-38) (tiny but positive)

Key Lemma 2.1 (Geometry of Numbers): Given lattice Λ in C^f with many unit-coordinate elements, averaging + projection gives planar sets with ν(P) proportional to |U_Λ|·|Λ ∩ B_R|.

Key Lemma 2.2 (Class Group Pigeonhole): For CM field K with conjugate prime pairs {P_i, cP_i}:
- |U| ≥ Π(k_j + 1) / h(K) elements with |σ(u)| = 1 for all embeddings σ
- These are the "unit translations" that become unit distances after projection

---

## AI Discovery Analysis

### Chain-of-Thought Insights (from Remarks)

The AI's key reasoning steps:
1. **"Maybe that enormous degree is not just an annoyance but a source of possible counterexamples"** — reframed degree growth from obstacle to feature
2. **"Number fields deserve a closer look"** — identified the algebraic structure
3. Switched from **fixed field, varying primes** (Erdős) to **fixed primes, varying field degree** (novel)
4. Recognized that **increasing degree is scary** but committed to it (Tsimerman's remark)

### Why Humans Missed It

1. **Anchoring on truth:** Everyone believed the conjecture was true (Erdős himself, $500 prize)
2. **Domain boundary:** Number theorists don't work on discrete geometry; geometers don't know class field theory
3. **"Increasing degree" is intimidating:** The analytic regime is hard to reason about (Sawin, Tsimerman)
4. **Many parameters to tune:** Primes, ball size, splitting conditions, field choice — exponential search space

### What This Tells Us About AI Mathematics

1. **Kolmogorov complexity modulo experts** (Gowers): The proof has relatively short "hint sequences"
2. **AI advantage = patience + breadth:** Tried paths humans dismissed as not worth time investment
3. **Cross-domain connection:** Algebraic number theory → discrete geometry (no human had both in active working memory)
4. **Construction > proof:** Finding counterexamples is different from proving upper bounds; AI excels at exhaustive search

---

## Distillation to Our Architecture

### Model-Based ↔ Modelless Mapping

| Paper Concept | Our Equivalent | Type |
|--------------|---------------|------|
| Erdős grid construction (Z[i]) | `ConstraintPruner` static rules | Modelless |
| Class group pigeonhole | `BanditPruner` frequency counting | Modelless |
| Minkowski lattice embedding | `ScreeningPruner` embedding-based scoring | Light model-based |
| Golod–Shafarevich tower construction | MCTS tree expansion with controlled branching | Model-based |
| Chebotarev splitting conditions | `SpeculativeVerifier::speculate()` | Light model-based |
| Root discriminant preservation | Budget propagation (Plan 026/057) | Modelless |
| Coset averaging (probabilistic) | Self-play exploration (G-Zero) | Model-based |
| Single-coordinate projection | Dimensionality reduction (HLA/SpectralQuant) | Modelless |

### Reusable Patterns for GOAT Proofs

1. **Lattice Averaging Proof Pattern:**
   - Embed problem in high-dimensional lattice
   - Average over cosets → expected value bound
   - Project to low dimension → concrete instance
   - Use for: packing bounds, distance counting, chromatic number lower bounds

2. **Class Group Pigeonhole:**
   - Many ideal configurations / few classes → some class has many representatives
   - Ratio gives lower bound on "useful" algebraic objects
   - Use for: any counting problem over number-theoretic structures

3. **Tower Construction with Prescribed Splitting:**
   - Build infinite tower (Golod–Shafarevich)
   - Kill Frobenius in Frattini subgroup (adds relations without losing generators)
   - Result: infinite tower where fixed primes split completely
   - Use for: any problem requiring "many objects of bounded size"

### Feature Gate: `unit_distance`

**Purpose:** Combinatorial geometry GOAT proofs using number-theoretic lattice constructions.

**Scope:**
- Lattice averaging for combinatorial bounds
- CM field constructions for unit-distance/graph coloring proofs
- Class group pigeonhole counting
- Minkowski embedding projection utilities

**Default:** Off (opt-in, research feature)

---

## Key References from Papers

1. Erdős (1946) — Original conjecture: ν(n) ≤ n^(1+C/log log n)
2. Spencer–Szemerédi–Trotter (1984) — Best upper bound: O(n^(4/3))
3. Guth–Katz (2015) — Distinct distances resolved
4. Golod–Shafarevich (1964) — Infinite class field towers via r > d²/4
5. Hajir–Maire (2001) — Asymptotically good towers
6. Hajir–Maire–Ramakrishna (2021) — Cutting towers with prescribed splitting
7. Ellenberg–Venkatesh (2007) — Class group torsion bounds (related pigeonhole)
8. Alon–Bucić–Sauermann (2025) — Generic norms: O(n log n log log n)
9. Greilhuber–Schildkraut–Tidor (2025) — Matching lower bound for generic norms

---

## Verdict

**High research value, medium implementation priority.**

1. **The mathematical technique is reusable** — lattice averaging + class group pigeonhole is a general GOAT proof pattern
2. **The AI discovery methodology validates our architecture** — model-based exploration (Golod–Shafarevich tower search) + modelless exploitation (pigeonhole counting)
3. **The number-field dimensionality trick** — taking degree → ∞ while keeping root discriminant bounded — is analogous to our modelless distillation: no training but growing structural complexity
4. **Explicit construction code** could serve as a GOAT proof for combinatorial geometry claims in game AI (board coloring, distance constraints)

**Action:** Feature gate `unit_distance`, implement T1 (Minkowski lattice) and T2 (class group pigeonhole) as reusable GOAT proof primitives. The proof construction itself (field tower) is deferred to model-based phase.