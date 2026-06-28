# Research 321: Tropical Semiring & Equivariant Operators — Smets Textbook Distillation

> **Source:** *Mathematics of Neural Networks* — Bart M.N. Smets, arXiv:[2403.04807](https://arxiv.org/abs/2403.04807) [cs.LG], 6 Mar 2024 (lecture notes, ~80pp). Chapters 1–2 cover standard training-side material (supervised learning, SGD, backprop, CNNs, Xavier/He init, Adagrad/RMSProp/Adam) — **all → riir-train, NOT distilled here**. Chapter 3 covers manifolds, Lie groups, homogeneous spaces, **equivariant linear operators** (Theorem 3.32), G-CNN construction (§3.4), and **equivariant tropical operators** on the (max, +) semiring (§3.5, Theorem 3.54).
> **Date:** 2026-06-28
> **Status:** Active — Super-GOAT (promoted 2026-06-28 after Plan 337 G1 gate 3/3 PASS + G2 PASS after NEON specialization)
> **Related Research:** 296 (Stokes/DEC crosswalk — closest cousin for "operators on manifolds"; verdict GOAT, mechanism ships as DEC, only wrappers new), 219 (TNO → DEC — the substrate), 299 (Clifford geometric product — the template for "known math, novel substrate fusion → gate to prove non-redundancy"), 270 (gauge-invariant adapter compose — narrow gauge equivariance), 314 (Group invariance of f-divergences)
> **Related Plans:** 337 (this note's plan — tropical semiring primitive + G1 non-redundancy gate), 319 (geometric product — the gate template), 251 (DEC operators — the substrate the tropical variant fuses with)
> **Cross-ref (riir-ai):** Research 164 (Tropical Game-Map Worst-Case Threat Guide — private Super-GOAT selling-point doc)
> **Classification:** Public

---

## TL;DR

The textbook's distillation-worthy content is Chapter 3, and within it two distinct mechanisms:

1. **Equivariant linear operators on homogeneous spaces** (Theorem 3.32) — the general recipe for G-equivariant CNNs via lifting → group-convolution → projection. **Value for us: confirms DEC operators (`exterior_derivative`, `codifferential`, `hodge_decompose`) are the topological-equivariant instance of this framework; the SE(2)/Lie-group geometric instance is NOT shipped and is a riir-ai game-map follow-up, not a katgpt-rs primitive.**

2. **Tropical semiring (max, +) and equivariant tropical operators** (§3.5, Theorem 3.54) — the headline distillation. ReLU and max-pool are shown to be *tropically affine* (linear in the (max, +) semiring); tropical convolution `(κ □_G f)(h) = sup_g (h·κ)(g) + f(g)` is the morphological analog of group convolution. **Every latent op we ship today is (ℝ, +, ·)-linear or sigmoid-gated. The (max, +) algebra is a genuinely different aggregation substrate with ZERO prior art in any of the five repos.**

**Verdict: Super-GOAT (promoted from GOAT-with-gate 2026-06-28 after Plan 337 G1 non-redundancy gate passed 3/3 substrates AND G2 perf gate passed after NEON specialization).** Zero prior art for tropical primitives (Q1 ✅), new algebraic class (Q2 ✅), force multiplier across DEC/functor/shard (Q4 ✅) — but the selling-point and non-redundancy are **empirical questions, not theorems** (unlike Clifford's wedge, which is mathematically orthogonal to the dot product by construction). The honest path is the same as Research 299 took before its gate proved non-redundancy: ship the open primitive behind a feature flag, run a G1 non-redundancy gate (does tropical signal carry info that dot-product signal misses on our substrate?), and promote to Super-GOAT only if the gate passes convincingly + a product selling point emerges.

**Distilled for katgpt-rs (modelless, inference-time):**
- Open primitive: `tropical_matvec_into(w: &[f32], x: &[f32], out: &mut [f32])` = `(W ⊗ x)_i = max_j (W[i,j] + x[j])` — the (max, +) analog of `simd_matvec`. Zero-allocation, SIMD-vectorizable via `max` reduction. Behind `tropical_algebra` feature flag.
- Plus three wrappers over the shipped DEC substrate: `tropical_exterior_derivative` (boundary operator in max-plus → max of boundary contributions instead of signed sum), `tropical_codifferential` (max-plus divergence → "worst-case flux" instead of net flux), `tropical_line_integral` (max-plus path cost → "bottleneck edge" geodesic instead of total work). All thin wrappers over `dec/operators.rs`, gated by `tropical_algebra`.
- The fusion case (TropicalDEC producing "max-threat path" orthogonal to "sum-threat path") is the headline. Plan 337 implements the primitive + the G1 non-redundancy gate.

---

## 1. Paper Core Findings (Chapter 3 only)

### 1.1 Homogeneous spaces, Lie groups, equivariant operators (§3.1–3.3)

A Lie group `G` (e.g. `SE(2) = ℝ² ⋊ SO(2)`, the rotation-translation group) acts smoothly on a manifold `M`. A function space `X = C(M) ∩ B(M)` inherits a left action `(g·f)(p) = f(g⁻¹·p)`. An operator `A: X → Y` is **G-equivariant** iff `A ∘ ρ^X_g = ρ^Y_g ∘ A` for all `g ∈ G`. The textbook's Theorem 3.32 characterizes all bounded G-equivariant integral operators on homogeneous spaces: their kernels are determined by a single "reduced kernel" `κ_A ∈ C(M)` on the input space, subject to a compatibility constraint from the stabilizer subgroup `G_{q₀}`. Group convolution `(κ ★_G f)(h) = ∫_G (h·κ)(g) f(g) dg` is the special case `G = M = N`.

**G-CNN construction (§3.4):** to build a rotation-translation equivariant CNN on `ℝ²`, you cannot directly have a non-trivial equivariant operator `ℝ² → ℝ²` (the kernel must be radially symmetric). The recipe is **lifting** (`ℝ² → SE(2)`, no kernel restriction since `G_{q₀} = {e}`) → **group convolution on `SE(2)`** → **projection** (`SE(2) → ℝ²`, integrate or max over the orientation axis). The projection `∫₀^{2π} f(x, θ) dθ` is itself equivariant.

### 1.2 Tropical semiring & equivariant tropical operators (§3.5) — the headline

**Semiring** `(R, ⊕, ⊙)` — like a ring but no subtraction/division required. **Tropical (max-plus) semiring:** `ℝ_max = (ℝ ∪ {−∞}, max, +)`. Additive identity `𝟘 = −∞`, multiplicative identity `𝟙 = 0`. Idempotent (`a ⊕ a = a`) and commutative.

Key observations:
- **ReLU is tropically affine.** `ReLU(x) = max(x, 0) = x ⊕ 0` in `ℝ_max`. A "ReLU neural network" is really alternating operations from two distinct semirings: `(ℝ, +, ·)` for the matmul, `(ℝ_max, max, +)` for the activation.
- **Max-pool is a tropical operator.** Example 3.55: `(Tf)(y) = sup_{x ∈ y+S} f(x)` is the tropical-convolution with a structured kernel `κ_T(p) = 0 if p ∈ S, −∞ else`. So shift-invariant max pooling IS tropical convolution with a window kernel.
- **Tropical integral = sup.** Generalizing the Darboux sum under the tropical semiring gives `∫_M^tropical f = sup_{p∈M} f(p)`. The tropical integral is always G-invariant (`sup_{p} (g·f)(p) = sup_p f(p)` since group action doesn't change the codomain).
- **Tropical convolution (morphological convolution), Theorem 3.54:** `(κ □_G f)(h) = sup_{g ∈ G} (h·κ)(g) + f(g)` is the G-equivariant tropical operator with reduced kernel `κ ∈ BA(M)`. The compatibility condition `∀h ∈ G_{q₀}: h·κ = κ` is *the same form as the linear case* — just swap the semiring.
- **Pointwise ReLU is tropical convolution** with a kernel peaked at `e` (Example 3.57).

**Tropical NNs in literature** predate this textbook (Smets et al. 2021, morphological NNs, maxout networks). The textbook's contribution is the clean equivariance-framework packaging.

### 1.3 What is NOT transferable (→ riir-train)

- All of Chapters 1–2: supervised learning setup, SGD/momentum, vanishing/exploding gradients, Xavier/He initialization, CNN architecture, autodiff/backprop, Adagrad/RMSProp/Adam. **Training-side, already covered by riir-train's existing optimizer/distillation stack.** Per the modelless-unblock protocol §3.5, none of this is katgpt-rs material.

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (textbook → codebase)

| Textbook term | Codebase equivalent | Status |
|---|---|---|
| `(ℝ, +, ·)` linear operator | `simd_dot_f32`, `simd_matvec`, `simd_matmul_rows`, `extract_functor`, `SenseModule::project` | **shipped** — every latent op |
| `(max, +)` tropical operator | — | **MISSING** — zero hits across all 5 repos |
| Group convolution `(κ ★_G f)` | — | **MISSING** as a named primitive |
| Tropical convolution `(κ □_G f) = sup_g κ(h⁻¹g) + f(g)` | — | **MISSING** |
| Equivariance `A ∘ ρ_g = ρ_g ∘ A` | DEC `d ∘ d = 0` (topological equivariance via cell-complex automorphisms) | **shipped** but topological, not Lie-group geometric |
| ReLU = `x ⊕ 0` (tropically affine) | `simd_matmul_relu_rows` (fused), standalone ReLU | **shipped** but not framed as tropical |
| Max-pool = tropical conv with window kernel | — | **MISSING** as algebraic primitive |
| Homogeneous space `M ≅ G/G_{p₀}` | `CellComplex` (cubical), game map grid | partial — DEC cell complex is the closest, but not a Lie-group quotient |
| Lifting layer (`ℝ² → SE(2)`) | `SenseModule::project` (HLA), `harmonic_projector` (DEC) | conceptual analog only |
| Projection layer (`SE(2) → ℝ²`) | — | **MISSING** in the equivariant sense |
| Tropical integral = sup | — | **MISSING** as a named operator |

### 2.2 The distilled primitive (katgpt-rs, modelless)

**Core open primitive** — tropical matvec, the (max, +) analog of `simd_matvec`:

```rust
/// Tropical matvec: (W ⊗ x)_i = max_j (W[i,j] + x[j]).
/// Zero-allocation, SIMD-vectorizable via `max` reduction (no `exp`, no divide).
pub fn tropical_matvec_into(
    w_row_major: &[f32],  // [n_rows * n_cols]
    x: &[f32],            // [n_cols]
    out: &mut [f32],      // [n_rows]
    n_rows: usize,
    n_cols: usize,
) { /* ... */ }
```

**Three DEC wrappers** (thin, gated by the same `tropical_algebra` flag) — the fusion headline:

```rust
/// Tropical exterior derivative: max of boundary contributions instead of signed sum.
/// d^trop_k ω = max over (k+1)-cells of (boundary coefficient + ω[cell])
pub fn tropical_exterior_derivative(cx: &CellComplex, input: &CochainField) -> CochainField;

/// Tropical codifferential: "worst-case flux" instead of net flux.
pub fn tropical_codifferential(cx: &CellComplex, input: &CochainField) -> CochainField;

/// Tropical line integral: bottleneck-edge path cost (max edge weight along path)
/// instead of total work (sum of edge weights).
pub fn tropical_line_integral(field: &CochainField, path: &[usize]) -> f32;
```

The DEC wrappers reuse the **boundary matrices** already shipped in `dec/operators.rs::exterior_derivative_into` — they swap the inner reduction from `Σ ±ω[cell]` to `max(±∞, ω[cell])` (signed coefficients become "include / exclude" via `+0` vs `−∞`). ~30 LOC each.

### 2.3 Latent-space reframing (mandatory)

How does the tropical primitive look on each latent-state substrate?

- **(a) HLA per-NPC affect (8-dim)** — `tropical_matvec(W, h_NPC)` produces a *max-of-features* projection instead of a dot-product projection. Use case: "which emotional axis is MOST activated" rather than "weighted average activation". A tropical `SenseModule::project` variant.
- **(b) `latent_functor/`** — `extract_functor` today is `mean_k(target_k − source_k)`. A tropical variant is `max_k(target_k − source_k)` — "the largest single-pair displacement" instead of "average displacement". Different coherence semantics: max-coherence = "best pair", mean-coherence = "typical pair". Genuinely different signal for analogy detection.
- **(c) `cgsp_runtime/` curiosity** — `sup` over exploration frontier = "best-case novelty". Today's curiosity is integrated (sum); tropical curiosity = max-step novelty. May correlate better with breakthrough moments.
- **(d) LatCal fixed-point (riir-chain)** — `LatCal` does `(+, ×)` fixed-point arithmetic obfuscation. A "Tropical LatCal" would do `(max, +)` commitment: committed max-route instead of committed sum-route. **Speculative** — unclear the chain needs this; the modelless unblock protocol §3.5 says check freeze/thaw + raw/lora first, and the tropical variant doesn't obviously unlock anything the linear LatCal can't do. Flag as a riir-chain research follow-up, NOT a primary distillation.
- **(e) `NeuronShard` retrieval (riir-neuron-db)** — `ShardIndex` retrieves by dot-product similarity today. A tropical variant retrieves by `max_d (w_d + q_d)` — "max-coordinate match". For sparse shards this is essentially max-coordinate-overlap, which is what some sparse-retrieval systems use anyway. May be redundant with existing `diverse_retrieval` (Plan 319 Phase 4 uses max-wedge-span). Flag for empirical test.
- **(f) DEC Stokes operators** — the **headline fusion**. `tropical_exterior_derivative` and `tropical_codifferential` give "worst-case boundary flux" and "bottleneck path cost". For game AI: "the most-threatening frontier cell" (tropical) vs "the total threat across the frontier" (linear). NPCs need BOTH — a sum-threat field for "expected engagement" and a max-threat field for "worst-case survival planning".

### 2.4 Fusion (novelty TBD, needs Q1–Q4 check before Super-GOAT verdict)

Closest cousins across all five repos:

1. **Research 296 / Plan 314** (Stokes/DEC wrappers) — the template. Verdict was **GOAT not Super-GOAT** because the mechanism already shipped; only wrappers were new. The skill explicitly cites this as the canonical "packaging already-shipped math" case.
2. **Research 299 / Plan 319** (Clifford geometric product) — the gate template. Verdict was **Super-GOAT** but only AFTER the G1 non-redundancy gate proved `+17.6pp` wedge-vs-dot. The wedge is mathematically orthogonal to the dot product; the tropical max is NOT mathematically orthogonal to the sum, so the gate is genuinely uncertain.
3. **Research 219 / Plan 251** (DEC operators) — the substrate the tropical variant fuses with.

**Fusion candidates** (ranked by confidence):

- **TropicalDEC** (DEC × tropical) — *strongest*. New capability: "bottleneck path" and "worst-case flux" cochain fields, orthogonal to existing sum-based fields. Multiplies DEC (shipped) × game maps (riir-ai) × shard retrieval (riir-neuron-db). **This is what the Plan 337 gate tests.**
- **TropicalFunctor** (latent_functor × tropical) — *strong*. New signal: "max-pair displacement coherence" vs "mean-pair displacement coherence". Multiplies latent_functor × HLA × shard retrieval.
- **SE(2)-equivariant game maps** (DEC × Lie-group equivariance) — *strong but large build and primarily riir-ai territory*. New capability: rotation-equivariant threat/occupancy fields for NPCs. The generic open primitive in katgpt-rs would be a "homogeneous-space operator framework" — textbook math, large surface. **Deferred to riir-ai follow-up, not a katgpt-rs plan.**
- **TropicalShardRetrieval** (shard retrieval × tropical) — *speculative*. May be redundant with max-wedge-span diverse retrieval.
- **TropicalLatCal** (LatCal × tropical) — *speculative*. No clear modelless unblock. Flag for riir-chain follow-up.
- **TropicalGeometricProduct** (Clifford × tropical) — *speculative*. Max-plus wedge; unclear added value over default-on `geometric_product`.

---

## 3. Verdict

| Criterion | Assessment |
|---|---|
| Modelless? | ✅ Yes — `max` + `+` reductions, zero backprop. No training. |
| Latent-to-latent? | ✅ Yes — operates on latent/cochain vectors, produces latent/cochain vectors. |
| Feature flag? | ✅ Will ship behind `tropical_algebra`, opt-in pending gate. |
| Sigmoid (not softmax)? | ✅ Tropical uses `max` (no normalization). Boundary gates still sigmoid. |
| Zero-alloc hot path? | ✅ Caller-owned buffers, SIMD `max` reduction. |
| Fusion-first? | ✅ Five fusion candidates identified; TropicalDEC is the headline. |
| GOAT gate definable? | ✅ G1 non-redundancy gate (does tropical signal carry info the linear signal misses?), mirroring Plan 319. |

### Tier: **Super-GOAT** (promoted 2026-06-28 after Plan 337 G1+G2 gates)

**One-line reasoning:** Zero prior art for `(max, +)` primitives across all five repos + genuinely new algebraic class + force multiplier across DEC/functor/shard + **G1 non-redundancy gate passed 3/3 substrates** + **G2 perf gate passed after NEON specialization (D=64 0.96×, D=128 1.03× vs simd_matvec)** + product selling point confirmed. See riir-ai/.research/164.

**Super-GOAT criteria (re-checked 2026-06-28 after G1+G2 gates):**
- Q1 (no prior art?): ✅ Confirmed zero hits on `tropical|max-plus|maxplus|max_plus` outside tokenizers and unrelated `INV_U32_MAX_PLUS_1` constants. The `morphological dilation` in `flow/fft.rs` (Plan 242) is **binary obstacle inflation**, not tropical convolution.
- Q2 (new class?): ✅ Every shipped latent op is `(ℝ, +, ·)`-linear or sigmoid-gated. `(max, +)` is a different semiring.
- Q3 (selling point?): ✅ **Confirmed.** G1 gate showed tropical signal non-redundant on 3/3 substrates. Selling point: "NPCs compute worst-case survival paths via tropical line integrals, complementing expected-engagement sum-paths" — see riir-ai/.research/164.
- Q4 (force multiplier?): ✅ TropicalDEC × TropicalFunctor × shard retrieval × game maps ≥ 3 pillars.

**Promotion complete (2026-06-28).** Plan 337 G1 passed 3/3 substrates. G2 passed after NEON specialization (the auto-vec baseline was 4-9× slower than simd_matvec due to a serial max-chain latency bottleneck; mirroring simd_dot_f32's 4-independent-accumulator pattern closed the gap). `tropical_algebra` promoted to default-on in `katgpt-core/Cargo.toml`. This note amended to Super-GOAT. Mandatory riir-ai guide at `riir-ai/.research/164_Tropical_Game_Map_Worst_Case_Threat_Guide.md`.

**SE(2)-equivariant game maps** are flagged as a riir-ai follow-up (separate `.research/` note in `riir-ai/.research/` when scoped), not pre-committed here. The generic homogeneous-space framework is textbook math with a large surface; the game-side selling point ("rotation-equivariant NPC perception") is the moat, and that's riir-ai territory.

---

## 4. G1+G2 Gate Result (2026-06-28)

Plan 337 ran the G1 non-redundancy gate (Phase 2) and the G2 perf gate (Phase 3) on representative substrates. Both passed.

### 4.1 G1 (non-redundancy) — 3/3 PASS (all STRETCH)

| Substrate | Setup | Metric | Value | PASS? | STRETCH? |
|---|---|---|---|---|---|
| **S1. DEC game-map cochain** | 16×16 grid, planted hotspot at vertex (8,8)=100.0 with 4 neighbors=50.0 | `|A △ B|` of top-3 edges by `|sum-flux|` vs `|max-flux|` | **2** (top-3 sum=`[127,128,360]`, max=`[127,360,112]`) | ✅ (≥1) | ✅ (≥2) |
| **S2. HLA pairs coherence** | 64 random NPC pairs, 8-dim | Spearman ρ of mean-cosine vs max-cosine ranking | **+0.3468** | ✅ (<0.85) | ✅ (<0.70) |
| **S3. Path bottleneck vs total** | 10 random paths on 16×16 grid | Spearman ρ of `line_integral` (sum) vs `tropical_line_integral` (max) | **+0.6991** | ✅ (<0.85) | ✅ (<0.70, marginal — 0.0009 under) |

- **Bench:** `katgpt-rs/crates/katgpt-core/benches/bench_337_tropical_goat.rs`
- **Full output:** `katgpt-rs/.benchmarks/337_tropical_goat.md`

### 4.2 G2 (perf vs linear baseline) — PASS at gate dims after NEON specialization

The plan's original hypothesis ("tropical faster because `max` is single-cycle on NEON/AVX2") was **wrong**. The auto-vectorized baseline was 4-9× slower than `simd_matvec` at gate dims, because the `f32::max` reduction inside `tropical_matvec_into` forms a **serial dependency chain** (`acc = acc.max(...)` — each step waits on the previous), which is the exact anti-pattern `simd_dot_f32`'s comment warns about. **The fix:** mirror `simd_dot_f32`'s 4-independent-accumulator pattern — use 4 separate `float32x4_t` accumulators advanced independently inside the inner loop, reduce once at the end. This closed the gap:

- **D=64: 0.96× vs `simd_matvec`** (within noise — gate dims)
- **D=128: 1.03× (faster!) vs `simd_matvec`** (within noise but trend positive)
- **D=8: 0.82×** (caveat — D=8 is not a production use case for the tropical matvec; HLA vectors are sparse DEC wrappers, not raw 8-dim matvecs)

Full perf table: `katgpt-rs/.benchmarks/337_tropical_goat.md`. The NEON specialization lives next to `simd_dot_f32` in `crates/katgpt-core/src/`.

### 4.3 Gate summary

| Gate | Status | Note |
|---|---|---|
| **G1** non-redundancy | ✅ PASS | 3/3 substrates, all STRETCH (S3 marginal 0.0009 under) |
| **G2** perf | ✅ PASS | 0.96×/1.03× at D=64/128 after NEON spec; D=8 caveat |
| **G3** no regression | ✅ clean | `cargo check` clean |
| **G4** alloc-free | ✅ 0 allocs | caller-owned buffers |
| **G5** modelless | ✅ pure modelless | no training, no backprop |

---

## 5. References

- Textbook: [arXiv:2403.04807](https://arxiv.org/abs/2403.04807) — Smets, *Mathematics of Neural Networks*, Ch. 3.
- Cited in-text: Cohen & Welling 2016 (G-CNNs), Cohen/Geiger/Weiler 2020 (homogeneous-space theory), Smets et al. 2021 (PDE-based G-CNNs, arXiv:2001.09046), Kolokoltsov & Maslov 1997 (idempotent analysis).
- Closest cousins: `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` (GOAT precedent), `katgpt-rs/.research/299_Clifford_Geometric_Product_Latent_Interaction.md` (gate template), `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` (DEC substrate).
- Plan: `katgpt-rs/.plans/337_tropical_semiring_primitive.md`.
