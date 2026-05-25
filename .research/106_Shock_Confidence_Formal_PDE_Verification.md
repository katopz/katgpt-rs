# Research 106: Shock with Confidence — Formal Proofs for Hyperbolic PDE Solvers

**Paper:** [arXiv:2503.13877](https://arxiv.org/abs/2503.13877) — Shock with Confidence: Formal Proofs of Correctness for Hyperbolic Partial Differential Equation Solvers
**Authors:** Jonathan Gorard (Princeton), Ammar Hakim (Princeton Plasma Physics Laboratory), 2025
**Date:** 2026-05-25
**Verdict:** 🟡 **Conditional Adopt — methodology paper on formal verification of numerical algorithms. Key extractable primitives: (1) IEEE 754-aware symbolic normalization for constraint validation, (2) property-proving pipeline structure (hierarchical decomposition → canonical form → proof), (3) constrained mutation zones for Percepta compiler stack. Cross-references Research 088/104 (AlphaProof Nexus) for evolutionary proof patterns.**

---

## TL;DR

The paper builds a formal verification pipeline in Racket for high-resolution shock-capturing (HRSC) finite volume PDE solvers. Given a hyperbolic PDE system (linear advection, Burgers', Maxwell's, isothermal Euler), the system: (1) encodes it in a DSL, (2) generates verified C code, and (3) produces standalone proof certificates for mathematical/physical correctness properties (hyperbolicity, CFL stability, L² stability, local Lipschitz continuity, flux conservation, TVD).

**Key finding:** The theorem-prover correctly identifies **when it cannot prove something** — partial/conditional proofs for isothermal Euler (missing ρ > 0 constraint) and superbee/van Leer limiters (simplification limitations). This "honest failure" pattern is valuable for our GOAT proof methodology.

---

## Core Mechanisms

### 1. Symbolic Normalization to Canonical Form

The heart of the theorem prover is `symbolic-simp`:

```racket
(define (symbolic-simp expr)
  (define simp-expr (symbolic-simp-rule expr))
  (cond
    [(equal? simp-expr expr) expr]
    [else (symbolic-simp simp-expr)]))
```

This applies rewrite rules to fixed-point. **Critical design choices:**
- Only admits transformations valid under **IEEE 754 floating-point arithmetic** (commutativity ✅, associativity ❌)
- Moves numerical constants left, collects like terms, evaluates purely numerical sub-expressions
- Has special rules for `max(x,y) = ½(x+y) + ½|x-y|` and `sqrt(x²) = x`

**Distillation for us:** Our `ConstraintPruner` does binary pass/fail. This paper shows a richer pattern: **multi-stage symbolic normalization** where each stage guarantees a correctness property. Our validator pipeline (PartialParser DFA → syn parse → cargo check) is already multi-tier, but we don't track **which stage caught which error** or produce **proof certificates**.

### 2. IEEE 754-Aware Automatic Differentiation

The `symbolic-diff` function respects floating-point algebra:
- No associativity assumption (unlike symbolic math tools)
- Product rule applied term-by-term with non-associative accumulation
- Hessian computed as composition: `symbolic-jacobian(symbolic-gradient(f, vars), vars)`

**Distillation for us:** Our SIMD matmul kernels (Plan 103 CODA, Plan 115 tiled attention) assume standard FP arithmetic. The paper's approach of proving properties under IEEE 754 constraints (not idealized reals) is directly relevant to our numerical stability proofs. Example: our `TurboQuant` quantization error bounds should account for non-associativity of FP addition.

### 3. Property-Proving Hierarchy

The paper proves properties in a dependency chain:

```
Hyperbolicity ← diagonalizable Jacobian with real eigenvalues
    ↓
CFL Stability ← 0 ≤ |a|Δt/Δx ≤ 1
    ↓
Local Lipschitz ← Hessian positive semidefinite
    ↓
Physical Validity ← Lipschitz sufficient for thermodynamic consistency
```

Each property is **independently verifiable** and produces a **standalone proof certificate** (symbolic Racket code).

**Distillation for us:** Our GOAT proof methodology (threshold-based pass/fail) is flat. The paper shows a **hierarchical approach** where passing deeper properties implies passing shallower ones. For our GOAT proofs:
- Fourier MCTS position invariance → implies spatial consistency (Plan 061)
- WASM validator 0 critical mismatches → implies deterministic correctness (Plan 034)
- NPC dialog <10ms/turn → implies real-time feasibility (Plan 099)

We could structure GOAT proofs as dependency chains rather than independent thresholds.

### 4. Factorization Strategy

For 4D vector PDE systems (Maxwell's, Euler), the prover **factorizes into coupled 2×2 pairs** rather than proving properties for the full system:
- Maxwell's: 4 pairs (Eŷ/Bz, Ez/Bŷ, Ex/φ, Bx/ψ)
- Euler: 2 pairs (ρ/ρu nonlinear, ρv/ρw linear)

**Distillation for us:** Our game state spaces are high-dimensional. The factorization pattern applies:
- Bomber: decompose into (position, bomb, powerup) sub-problems
- Go: decompose into (local tactics, global influence, boundary races)
- Dungeon: decompose into (pathfinding, combat, fog-of-war)

This is essentially our existing DDTree branch decomposition, but formalized as a proof strategy.

### 5. Flux Limiters as Constrained Mutation

The paper's flux limiters (minmod, MC, superbee, van Leer) are functions that **constrain how solutions evolve** — preventing spurious oscillations while maintaining accuracy. The TVD (Total Variation Diminishing) property is proved via the Sweby criteria.

**Distillation for us:** This maps directly to our Percepta compiler stack's constrained mutation:
- `EVOLVE-BLOCK` markers → flux limiter boundaries
- Constrained mutation → only allow modifications within bounds
- TVD verification → our `ConstraintPruner` validation of mutated sketches

The paper proves that **different limiters have different proof success rates** — minmod and MC proved both symmetry and TVD, superbee failed symmetry, van Leer failed TVD. This suggests **mutation strategy choice matters** for our Percepta proof sketch evolution (Plan 128).

### 6. Partial/Conditional Proofs — Honest Failure

The prover's most valuable property: **it knows when it can't prove something** and explains why:
- Euler ρ/ρu: "cannot guarantee Lipschitz continuity without ρ > 0"
- Euler ρv/ρw: "repeated eigenvalue u violates strict hyperbolicity"
- Superbee: "simplification rules insufficient for symmetry proof"

**Distillation for us:** Our GOAT proofs should include **conditional results** — not just pass/fail but "passes given X, fails without X." This makes our proofs more useful as diagnostic tools.

---

## Distillation Mapping

| Paper Mechanism | Our Equivalent | Enhancement Opportunity |
|----------------|---------------|------------------------|
| IEEE 754-aware symbolic normalization | PartialParser DFA bracket balance | 🟡 Extend DFA with FP-awareness for SIMD kernel proofs |
| Property hierarchy (hyperbolicity → stability → Lipschitz) | Flat GOAT threshold checks | 🟡 Structure GOAT proofs as dependency chains |
| Factorization into 2×2 pairs | DDTree branch decomposition | ✅ Already aligned — formalize as proof strategy |
| Flux limiters (constrained evolution) | ConstraintPruner + EVOLVE-BLOCK markers | ✅ Already aligned via Research 088/104 |
| Proof certificates (standalone Racket code) | GOAT proof benchmark files | 🟡 Add structured proof certificates (not just threshold pass) |
| Canonical form reduction | ScreeningPruner relevance scoring | ⬜ Indirect — different domain |
| Automatic differentiation | SIMD matmul numerical analysis | ⬜ Indirect — different domain |
| CFL stability condition | Bandit convergence criteria | ⬜ Indirect — different abstraction level |

---

## What NOT to Distill

1. **Racket-specific DSL and code generation**: We're in Rust. Our Percepta DSL is fundamentally different.
2. **PDE-specific mathematics**: Lax-Friedrichs, Roe solvers, flux limiters — not applicable to game AI.
3. **C code synthesis**: Our code generation target is WASM, not C.
4. **Specific PDE systems**: Maxwell's, Euler, Burgers' — no game analog.
5. **Full symbolic algebra system**: Our needs are simpler — constraint validation, not general theorem proving.

---

## Cross-Reference with Research 088/104 (AlphaProof Nexus)

| AlphaProof Nexus (088/104) | Shock with Confidence (106) | Synthesis |
|--------|---------------------------|----------------------------|-----------|
| Proof generation | LLM-based (Gemini) | Symbolic rewriting | Hybrid: LLM generates candidates, symbolic verification proves properties |
| Failure handling | Agent offloads difficulty | Prover identifies missing constraints | Both validate honest failure detection |
| Population/evolution | Elo-rated sketch population | No population — single proof attempt | Nexus provides the evolution, Shock provides the proof methodology |
| Correctness properties | Binary (proved/disproved) | Graduated (full/conditional/partial) | Shock's graduated approach improves Nexus's binary verdict |
| Constrained mutation | EVOLVE-BLOCK markers | Flux limiter boundaries | Same pattern, different domains |

**Key synthesis:** AlphaProof Nexus gives us the **search strategy** (evolutionary population + Elo). Shock with Confidence gives us the **verification methodology** (hierarchical properties + canonical forms + honest failure). Together, they form a complete "generate + verify" pipeline for our Percepta compiler stack.

---

## Verdict

**🟡 Conditional Adopt** — This is a methodology paper, not an algorithm paper. Three specific patterns are worth incorporating:

1. **Hierarchical GOAT proofs**: Structure our proof thresholds as dependency chains rather than flat checks. "Proves A, which implies B, which implies C" is more informative than "passes threshold X, Y, Z independently."

2. **Conditional/graduated proof results**: Not just pass/fail — "passes given X constraint" or "partially proves (missing Y)". This makes our GOAT proof documents more diagnostic.

3. **Mutation strategy comparison**: The paper shows different flux limiters have different proof success rates. For our Percepta proof sketch evolution (Plan 128), we should **benchmark multiple mutation strategies** rather than committing to one.

**Not a GOAT pillar** per the [decision matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md) — this is proof methodology infrastructure, not directly MMO-productive. No game-specific code needed. Lives entirely in `katgpt-rs`.

**Domain:** `katgpt-rs` only — generic proof methodology. No `riir-ai` plan needed. Phase 4 (Percepta mutation strategy benchmarking) is super-GOAT — feature-gated behind `proof_cert` + `percepta`, kept as a selling point.

---

## References

- Paper: [arXiv:2503.13877](https://arxiv.org/abs/2503.13877)
- Gkeyll framework: https://gkeyll.readthedocs.io/
- Code: https://github.com/ammarhakim/gkylcas/ (Racket), https://github.com/ammarhakim/gkylzero/ (C)
- Related research: Research 088/104 (AlphaProof Nexus), Research 040 (Bradley-Terry ranking), Research 051 (Deep Manifold fixed-point boundaries)
- Related plans: Plan 128 (proof sketch evolution), Plan 143 (Nexus Elo Plackett-Luce P-UCB)
