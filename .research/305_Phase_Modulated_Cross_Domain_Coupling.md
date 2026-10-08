# Research 305: Phase-Modulated Cross-Domain Coupling — Norm-Preserving Subspace Rotation Gate

> **Source:** [UFO: A Domain-Unification-Free Operator Framework for Generalized Operator Learning](https://arxiv.org/abs/2605.12700) — Hanli Qiao, George Em Karniadakis, Muhammad Muniruzzaman, arXiv:2605.12700v1, May 2026
> **Date:** 2026-06-25
> **Status:** Active
> **Related Research:** 291 (Cross-Resolution Spectral Transport — linear cousin, no phase), 299 (Clifford Geometric Product — rotational *sensor*, this is the rotational *actuator*), 290 (Latent Field Steering — additive cousin, no rotation), 219 (DEC operators — Hodge mixer fusion target), 212 (Gemini Fourier × LatCal), 296 (Stokes/DEC vocabulary crosswalk)
> **Related Plans:** 322 (this primitive — open), 310 (Cross-Resolution Spectral Transport — shipped, linear-only), 319 (Clifford Geometric Product — shipped, default-on), 314 (Stokes wrappers — shipped), 251 (DEC operators — shipped)
> **Cross-ref (riir-ai):** Research 159 (Phase-Rotation Subspace Gate Game Runtime Guide) — private selling-point doc
> **Cross-ref (riir-chain):** Research 002 (K-Prior LatCal Commitment Bridge) — the committed-phase angle variant
> **Cross-ref (riir-neuron-db):** Research 008 (Shard Structural Retrieval Guide) — the spectral/spatial shard-half retrieval variant
> **Classification:** Public
>
> **PASS-Redirects (synthesis, added 2026-10-08):** Listopad [arXiv:2509.09691 "Wave-Based Semantic Memory with Resonance-Based Retrieval: A Phase-Aware Alternative to Vector Embedding Stores"] — the phase-family retrieval cousin of this note's rotation gate: complex waveforms ψ=A·e^iϕ with similarity S = ½β(1+β·cos_h), R = 2√(E1E2)/(E1+E2). PASS: cos_h is Plate's FHRR similarity and β·cos_h is soft Dice (V-Net arXiv:1606.04797) — published components; on the real-vector compat path (phase ∈ {0,π}) the score is order-equivalent to cosine for equal-norm candidates; evaluation circular (procedurally-constructed phases). The workspace already ships the algebraic superset — Research 299's `geometric_product` (default-on) carries both the inner (real) half the paper uses and the wedge (imaginary) half it discards.

---

## TL;DR

UFO's core contribution is a **cross-domain operator realized through adaptive phase-modulated coupling**: given a spectral-domain representation `Ψ_H(f) = a + i b` and a spatial-domain basis `Φ_S(x)`, the operator is `⟨Φ_S(x), cos α ⊙ a + sin α ⊙ b⟩` where `α = γ(η(Φ_S, Ψ_H))`. The key property is `sin²α + cos²α = 1` — the coupling is a **bounded, L2-norm-preserving rotation** in the (a, b) plane parameterized by a phase derived from a joint feature of both domains. In native form, `γ` is a trained MLP (→ riir-train); but the **modelless distillation** replaces `γ` with a deterministic sigmoid projection `α = sigmoid(⟨state, direction⟩) · π/2`, turning the coupling into a **frozen-artifact-gated unitary subspace rotation** — a genuinely new latent operation class.

**Distilled for katgpt-rs (modelless, inference-time):**
A zero-allocation, SIMD-vectorizable `phase_rotation_gate` primitive: given two latent slices `(a, b)` and a phase `α ∈ [0, π/2]`, produce `cos α ⊙ a + sin α ⊙ b`. The phase is computed modellessly via dot-product + sigmoid onto a frozen, BLAKE3-committed direction vector (exactly the Latent Field Steering / FUNCATTN artifact pattern). The result is a **norm-preserving** mix — unlike the existing sigmoid convex-combo gate `σ(w)·a + (1-σ(w))·b` (which preserves L1 mass but not L2 norm) or the additive steering `s + α·v` (which inflates norm), the phase rotation preserves `‖mix‖₂ = ‖a‖² + ‖b‖²` exactly when `a ⊥ b`, and is bounded by it otherwise.

---

## 1. Paper Core Findings

### 1.1 The mechanism — phase-modulated cross-domain coupling

UFO realizes an operator `G_α(f)(x) = C_α(Φ_S(x), Ψ_H(f))` where `Φ_S` is a **spatial basis network** (coordinate → feature), `Ψ_H(f) = a + i b` is a **spectral encoder** output (complex-valued, real part `a`, imaginary part `b`), and `C_α` is the **adaptive phase-modulated coupling operator**:

```
α   = γ_θ(η(Φ_S(x), Ψ_H(f)))     // phase from joint feature η = [Φ_S, a, b]
G_α = ⟨Φ_S(x), cos α ⊙ a + sin α ⊙ b⟩   // = Σ_c (Φ_S[c] · cos α[c] · a[c]  +  Φ_S[c] · sin α[c] · b[c])
```

**Three properties that make this distinct from every existing latent op in our codebase:**

1. **`sin²α + cos²α = 1` — L2-norm preservation.** The mix `cos α ⊙ a + sin α ⊙ b` has the same L2 norm as the concatenated `(a, b)` if `a ⊥ b` (Parseval-style identity), and is bounded by it otherwise. No inflation, no collapse. Compare:
   - Sigmoid convex combo `σ(w)·a + (1-σ(w))·b` — preserves L1 mass (Σ = 1), NOT L2 norm.
   - Additive steering `s + α·v` — inflates L2 norm by `‖α·v‖` (Research 290).
   - Dot-product projection `⟨state, direction⟩` — collapses to scalar.
2. **Joint feature `η = [Φ_S, a, b]` — the phase is a function of BOTH domains, not either alone.** This is the "non-separable" property the paper emphasizes: `α` depends on the *pair* `(spatial, spectral)`, not on either in isolation. Separable variants (α depends on `Φ_S` only, or `Ψ_H` only) are explicitly ablated in Table 2 and degrade sharply (L2 error 0.34→2.30 at s=0.39).
3. **Complex-valued / rotation interpretation.** `cos α + i sin α = e^{iα}` is a unit-modulus complex phase. The coupling is an elementwise rotation in `D` independent complex planes (one per channel). This is a **product of D planar rotations** — a special case of a unitary matrix.

### 1.2 Discretization decoupling (the consequence, not the primitive)

The paper's headline "discretization decoupling" (input observed at resolution A, output queried at resolution B) is a *consequence* of the dual-domain representation: `Ψ_H(f)` is resolution-agnostic (mean aggregation over observed points), `Φ_S(x)` is queried at any coordinate. **This is exactly Research 291's cross-resolution transport thesis**, restated in operator-learning vocabulary. The novel part for us is NOT the resolution decoupling (we shipped that in Plan 310) — it's the phase-modulated *coupling mechanism*.

### 1.3 Spectral encoder (the input-domain half)

```
ẑ_i = ω_θ(x'_i) ⊙ f̂_i       // coordinate-conditioned modulation of FFT(f)
z̄  = (1/N) Σ z_i              // mean aggregation → global spectral summary
Ψ_H(f) = ρ_r(Re z̄) + i ρ_i(Im z̄)   // learned nonlinearity on real/imag halves
```

The `ω_θ(x'_i) ⊙ f̂_i` is a coordinate-conditioned spectral modulation; the mean aggregation produces a global summary. **In latent-space terms:** lift to spectral domain → modulate per-coordinate → mean-pool → split into real/imag halves. This is a **global context vector with explicit real/imaginary structure** — distinct from the HLA kernel's purely real recurrent state.

### 1.4 Ablation: removing α collapses the operator (Table 2)

The separable variant `G(f)(x) = ⟨Φ_S(x), a + b⟩` (drop α, drop cos/sin) degrades dramatically on StepHeat: L2 error jumps from 0.11→0.34 (s=0.32) and 0.12→2.38 (s=0.41). **The phase modulation is not cosmetic — it is the mechanism.** This empirically validates that the rotation property (not just having two halves) is what carries the signal.

### 1.5 What's training-only (→ riir-train, do NOT distill here)

- The full UFO architecture (spectral encoder + spatial basis network + phase network γ_θ) trained end-to-end on PDE operator-learning benchmarks.
- The AdamW-style training of `γ_θ`, `ω_θ`, `ρ_r`, `ρ_i`, `L_θ`, `ϕ_θ`.
- The PDE benchmark results (StepHeat, δ-Helmholtz, Burgers, GRF-Helmholtz) — these measure *trained operator approximation quality*, not modelless properties.
- The "discretization decoupling" claim as an operator-learning contribution (we already ship cross-resolution transport, Plan 310).

**The training-only parts belong in riir-train.** The modelless transferable primitive is the **phase-modulated coupling operation itself** — `cos α ⊙ a + sin α ⊙ b` — with `α` constructed deterministically at inference time.

---

## 2. Distillation

### 2.1 Transferable primitive — phase rotation gate

```rust
/// Phase-Modulated Subspace Rotation Gate (modelless, zero-alloc).
///
/// Given two latent slices `a`, `b` (each of length `D`) and a phase angle
/// `alpha` (radians, one per channel OR a single scalar broadcast), produce
/// the norm-preserving rotation `cos(alpha) ⊙ a + sin(alpha) ⊙ b`.
///
/// Key property: sin²α + cos²α = 1 → the output L2 norm is bounded by
/// `sqrt(‖a‖² + ‖b‖²)` (exact equality if a ⊥ b). This is the rotation
/// invariant UFO exploits for stable cross-domain realization.
///
/// `phase` may be:
///   - a single scalar (broadcast to all D channels), OR
///   - a per-channel slice `[f32; D]` (UFO's full per-channel form).
///
/// All buffers are caller-provided scratch — zero allocation in steady state.
pub fn phase_rotation_gate_into(
    a: &[f32],         // [D] real-half latent slice
    b: &[f32],         // [D] imag-half latent slice
    cos_alpha: &[f32], // [D] precomputed cos(α) per channel (OR length-1 broadcast)
    sin_alpha: &[f32], // [D] precomputed sin(α) per channel (OR length-1 broadcast)
    out: &mut [f32],   // [D] output mix
) {
    debug_assert_eq!(a.len(), b.len());
    let d = a.len();
    if cos_alpha.len() == 1 {
        // Scalar-broadcast fast path — single cos/sin for all channels.
        let c = cos_alpha[0];
        let s = sin_alpha[0];
        for i in 0..d {
            out[i] = c * a[i] + s * b[i];
        }
    } else {
        // Per-channel path — UFO's full form.
        debug_assert_eq!(cos_alpha.len(), d);
        debug_assert_eq!(sin_alpha.len(), d);
        for i in 0..d {
            out[i] = cos_alpha[i] * a[i] + sin_alpha[i] * b[i];
        }
    }
}

/// Compute the phase α from a latent state via deterministic sigmoid projection.
///
/// `α = sigmoid(⟨state, direction⟩ · sharpness) · (π / 2)`
///
/// The direction vector is a frozen, BLAKE3-committed artifact (Plan 310 /
/// Plan 309 pattern). The phase is bounded in `[0, π/2]`, so cos ≥ 0 and
/// sin ≥ 0 — the mix is a *convex rotation*, never sign-flipping.
///
/// Returns (cos_alpha, sin_alpha) into caller-provided scratch.
pub fn compute_phase_from_projection(
    state: &[f32],         // [D] current latent state
    direction: &[f32],     // [D] frozen unit-norm direction vector
    sharpness: f32,        // λ — phase steepness (higher = sharper transition)
    cos_alpha: &mut f32,   // out: cos(α)
    sin_alpha: &mut f32,   // out: sin(α)
) {
    let dot: f32 = simd::simd_dot_f32(state, direction);
    let alpha = sigmoid(dot * sharpness) * (core::f32::consts::FRAC_PI_2);
    *cos_alpha = alpha.cos();
    *sin_alpha = alpha.sin();
}
```

**Complexity:** `O(D)` per call (one dot + one cos + one sin for scalar phase; `D` cos/sin for per-channel). Zero allocation after scratch init. SIMD-vectorizable for the inner mix loop.

**Numerical note:** for the per-channel form, `cos`/`sin` are expensive (libm). The scalar form uses a single `cos`/`sin` pair — fast. Per-channel callers should use the same polynomial-Padé approximation validated in Plan 319 Issue 003 (max error 4.9e-3 vs libm) when latency matters.

### 2.2 Where the pieces already live

| Piece | Existing location | Reuse |
|---|---|---|
| Sigmoid projection | `EmotionDirections::project` (Plan 162), FUNCATTN sigmoid basis, Latent Field Steering (Plan 309) | ✅ same math — dot + sigmoid |
| Direction vector storage | `EmotionDirections`, `NeuronShard::style_weights`, `LatentSteeringVector` | ✅ same artifact format |
| BLAKE3 commitment | `MerkleFrozenEnvelope` (`riir-neuron-db/src/freeze.rs`) | ✅ same envelope |
| SIMD dot | `simd::simd_dot_f32` (used by Plan 310, 319) | ✅ same primitive |
| Frozen-artifact hot-swap | `LoRAHotSwap`, `CrossResolutionBases` Arc-swap | ✅ same pattern |
| Pre-allocated scratch | `FuncAttnScratch`, `CrossResScratch` | ✅ extend with cos/sin slots |
| Polynomial Padé SiLU/cos/sin | Plan 319 Issue 003 (4.9e-3 accuracy) | ✅ same fast-approx |

**The math is 90% shipped.** What's new: the cos/sin rotation coupling (vs additive / convex-combo / projection), the explicit two-slice (a, b) split convention, and the norm-preservation invariant.

### 2.3 Closest cousins (4) — and why this is NOT redundant

| Cousin | What it does | Why phase-rotation is different |
|---|---|---|
| **CommittedFieldBlend (WIP/untracked, `katgpt-rs/crates/katgpt-core/src/committed_field_blend.rs`; riir-ai Guide 158, source FAME arxiv 2510.00621)** | `evolve(z) = Σ_k sigmoid(π_k/τ) · f_k(z)` — **sigmoid-gated convex blend of K=3 archetype fields**. BLAKE3-committed blend weights `π`. Includes a `RotationField` test fixture that applies a 2D Givens rotation `z'[i]=cos·zi−sin·zj; z'[j]=sin·zi+cos·zj` as the *content* of one archetype field. | This IS the convex-combo cousin. The blend weights are independent sigmoids (one per field); the cos/sin rotation appears only as *field content* (one possible `f_k`), NOT as the blend mechanism. Phase-rotation uses cos/sin AS the blend weights (`cos α·a + sin α·b`), giving `sin²α+cos²α=1` norm-preservation — a property CommittedFieldBlend's independent sigmoids do NOT have (its output norm depends on `Σ sigmoid(π_k)·‖f_k‖`, unbounded by any input-norm invariant). **This is the closest prior art and was nearly missed (untracked WIP code, no `.research/` note in katgpt-rs yet) — vocabulary translation to "committed blend" / "field blend" was required to find it.** |
| **Cross-Resolution Spectral Transport (R291, P310 SHIPPED)** | `Ψ_dst · C · Φ_src^T` — linear Tikhonov transport between asymmetric bases. Pure matmul, no phase. | Linear projection preserves L2 only approximately (Tikhonov regularization). Phase rotation preserves L2 *exactly* by construction (sin²+cos²=1). Different math, different invariant. |
| **Clifford Geometric Product (R299, P319 DEFAULT-ON)** | `u·v + u∧v` — anti-symmetric channel wedge via cyclic shifts. Detects rotational structure as a *signal*. | Clifford is the *sensor* (detects rotation); phase-rotation is the *actuator* (performs rotation). Compose: Clifford detects which phase to apply; phase-rotation applies it. |
| **Latent Field Steering (R290, P309)** | `s + α·v` — additive direction injection. Inflates L2 norm by `‖α·v‖`. | Additive steering shifts the state; phase-rotation *rotates within a fixed-magnitude subspace*. Steering moves; rotation re-orients. |
| **Spherical Steering (R382, P405 DEFAULT-ON)** | `sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T` — **single-target geodesic Slerp** rotation of one vector toward a target direction. Sibling, not cousin: same norm-preservation thesis (`‖c0·ĥ + c1·μ_T‖ = 1` Slerp identity for all θ ∈ (0,π)), different parameterization. | Plan 322 (this primitive) rotates *within* the `(a,b)` plane — input naturally splits into two halves, Pythagorean identity holds when `a ⊥ b`. Plan 405 rotates *toward* a target outside the input's direction — Slerp identity holds for **all** θ, not just orthogonal. They compose: Plan 322 balances two subspaces, Plan 405 pulls a drifted vector back to an archetype. Use case: Plan 322 = subspace balance; Plan 405 = archetype drift correction (F1 fusion hook). |

**Critical distinction from CommittedFieldBlend and the standard sigmoid convex-combo gate `σ(w)·a + (1-σ(w))·b`:**
- Convex combo / CommittedFieldBlend: output norm ∈ `[min(‖a‖,‖b‖), max(‖a‖,‖b‖)]` for two-field case (or scales with `Σ sigmoid(π_k)` for N-field). Preserves L1 mass if weights sum to 1. Cannot turn `a` into a rotation of `b`. Weights are independent sigmoids — no Pythagorean identity.
- Phase rotation: output norm ∈ `[0, sqrt(‖a‖²+‖b‖²)]`. Preserves L2 magnitude of the pair. Can represent any unitary 2×2 rotation `(a,b) → (cos α·a + sin α·b, -sin α·a + cos α·b)`. Weights obey `sin²α+cos²α=1` — a Pythagorean identity the convex combo lacks.

The rotation is a *new operation class relative to CommittedFieldBlend*: it can smoothly interpolate between two subspaces without magnitude drift (the L2 norm is bounded by `‖a‖²+‖b‖²` for ALL α), which matters for belief-state stability over many ticks (HLA), functor coherence across decision stages (latent_functor), and shard retrieval consistency (NeuronShard). CommittedFieldBlend's convex combo has no such per-α bound — its output norm varies with the (independent) sigmoid weights.

**Honesty note on the WIP prior art:** the `CommittedFieldBlend` + `RotationField` code is untracked WIP (likely a parallel session building the FAME primitive from riir-ai Guide 158 / katgpt-rs Research 302 + Plan 321). It does NOT have a katgpt-rs `.research/` note yet (only the riir-ai guide exists). Finding it required grepping for "committed" / "field blend" / "blend" — vocabulary translation from the paper term "phase-modulated coupling" to the codebase term "field blend". This is a near-miss of the "code ships without a note" canonical failure (R242 / `evolve_hla` lesson); the mitigation here is the explicit citation above and the recommendation that Plan 322's GOAT gate G1 *also* benchmark phase-rotation vs CommittedFieldBlend's two-field convex-combo on the same long-horizon stability task, to empirically prove the L2-preservation advantage isn't just theoretical.

### 2.4 Fusion (Fusion)

**F1 (PRIMARY — riir-ai, see Guide 159): Phase-Rotation × HLA Subspace Gating**
Split HLA's 8-dim state into two 4-dim halves: `a = [valence, arousal, desperation, calm]` (action affects), `b = [fear, reserved_1, reserved_2, reserved_3]` (social/strategic affects). The phase `α = sigmoid(⟨state, combat_direction⟩) · π/2` rotates between halves: combat tick → α ≈ 0 → state dominated by `a`; dialog/exploration tick → α ≈ π/2 → state dominated by `b`. **Novel capability:** NPCs whose affect smoothly rotates between combat and social subspaces over thousands of ticks without magnitude drift — the L2 norm of the affect vector stays bounded by construction, preventing the "emotional explosion" or "emotional collapse" failure modes that additive/convex gates exhibit over long horizons.

**F2 (SECONDARY — katgpt-rs/DEC): Phase-Rotation × Hodge Mixer**
DEC's `hodge_decompose` (Plan 251) splits a flow field into exact + coexact + harmonic channels. Today they're additive: `flow = exact + coexact + harmonic`. Phase-rotation mixer: `cos α ⊙ exact + sin α ⊙ coexact + harmonic` (harmonic as the rotation-invariant residual). The phase `α` gates which non-harmonic component dominates per cell. **Novel capability:** per-cell Helmholtz regime modulation — a cell in a "vortical" regime (high coexact) can be smoothly rotated toward "potential flow" (high exact) via the phase, preserving total energy (L2 norm). This is the *missing smooth-mixing primitive* on top of the shipped Hodge decomposition.

**F3 (TERTIARY — riir-neuron-db): Phase-Rotation × Shard Spectral/Spatial Halves**
Split `NeuronShard::style_weights[64]` into `a = style_weights[0..32]` (spectral/style half) and `b = style_weights[32..64]` (spatial/behavior half). At retrieval, phase-rotate the query against the shard: `cos α ⊙ a_query + sin α ⊙ b_query` matched against `cos α ⊙ a_shard + sin α ⊙ b_shard`. **Novel capability:** retrieval that smoothly interpolates between style-similarity and behavior-similarity — current cosine retrieval collapses to one axis; phase-rotation retrieval explores the (style, behavior) plane.

**F4 (QUATERNARY — riir-chain): Phase-Rotation × LatCal Committed Phase**
LatCal's 2×2 matrix arithmetic can represent rotations `[[cos α, -sin α], [sin α, cos α]]` exactly in fixed-point. **The phase α becomes a LatCal-committed raw scalar**, and the rotation is a deterministic raw→raw operation suitable for sync. This is the **sync-boundary bridge**: the phase angle is committed raw (deterministic replay, anti-cheat); the coupling happens locally in latent space. **Novel capability:** committed subspace-rotation events — a faction-wide "battle stance" is a committed phase rotation applied to all members' HLA halves, replayable and tamper-evident.

**Strongest fusion candidates:** F1 (HLA subspace gating, primary selling point) + F2 (Hodge mixer, complements shipped DEC) + F4 (LatCal committed phase, the sync-bridge moat). F1 and F4 together form the headline: "NPCs rotate affect between combat and social subspaces, with the rotation angle committed to the chain for deterministic replay."

---

## 3. Verdict

**Tier: Super-GOAT (candidate — pending G1–G4 validation).**

### Novelty gate (Q1–Q4)

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **YES (after honest WIP-prior-art check).** Paper-vocabulary grep (`phase.modulat`, `cross.domain`, `complex.valued`, `cos.*sin.*alpha`) → ZERO hits in any `.research/`, `.plans/`, or `.md`. Code-vocabulary grep (`cos_alpha`, `sin_alpha`, `phase_couple`, `spectral_spatial`, `dual_domain`, `complex_pair`) → ZERO hits for the phase-modulated *blend* mechanism. **However**, a second-pass vocabulary-translated grep ("committed", "field blend", "blend", "rotation") surfaced **`CommittedFieldBlend` (WIP/untracked, `committed_field_blend.rs`)** — the FAME-sourced (arxiv 2510.00621, riir-ai Guide 158, katgpt-rs Research 302/Plan 321) sigmoid-gated convex blend of K=3 archetype fields. Its `RotationField` fixture applies cos/sin as *field content*, and its `apply_blended` uses independent sigmoids as *blend weights*. **This is the convex-combo cousin, NOT the phase-rotation primitive** — the distinction (cos/sin-as-weights with Pythagorean identity vs sigmoid-as-weights without) is sharpened in §2.3 above. The phase-rotation *as a blend mechanism with `sin²α+cos²α=1` norm preservation* is genuinely not shipped. Other rotational code (RoPE rotation, TurboQuant QR rotation, SVD phase-transition detection, Box-Muller polar sampling, RotationField-as-content) is all unrelated to *blend-weight* rotation. | Vocabulary translation: "phase-modulated coupling" → "rotation", "unitary", "Givens", "complex pair", **"field blend"**, **"committed blend"**; "cross-domain" → "dual subspace", "two-half split"; "spectral encoder" → "FFT + mean-pool + real/imag split". The "field blend" translation was the one that surfaced CommittedFieldBlend — without it, the WIP prior art would have been missed (the R242/`evolve_hla` near-miss). |
| Q2 New class of behavior? | **YES.** Every existing latent op in the codebase is one of: additive (steering), convex-combo (sigmoid gate), dot-projection (HLA), wedge-detection (Clifford), linear-transport (FUNCATTN/cross-res), or spatial-sum (DEC). **None is a bounded unitary rotation with built-in L2-norm preservation.** The rotation is a new operation class — it can smoothly interpolate between two subspaces without magnitude drift, which is the key stability property for long-horizon belief-state evolution (HLA over thousands of ticks) and functor coherence across decision stages. | |
| Q3 Product selling point? | **YES.** "NPCs whose affect rotates smoothly between combat and social subspaces over thousands of ticks without magnitude drift — emotional stability by construction, not by regularization. The rotation angle is committed to the chain for deterministic replay and anti-cheat." Concrete, demoable (crowd-scale stability under mode-switching), hard to replicate without the rotation primitive + freeze/thaw + LatCal commitment stack. | |
| Q4 Force multiplier? | **YES.** Connects HLA (riir-engine `hla/`), latent_functor (riir-engine `latent_functor/`), DEC Hodge mixer (katgpt-rs `dec/`), FUNCATTN/cross-res (katgpt-rs `funcattn.rs`, `cross_resolution.rs`), NeuronShard retrieval (riir-neuron-db `shard.rs`), LatCal commitment (riir-chain `encoding/latcal*.rs`), freeze/thaw envelope (riir-neuron-db `freeze.rs`). ≥7 pillars. | |

**Selling point:** Norm-preserving subspace rotation for stable per-NPC belief state — affect rotates between combat and social modes without drift, with the rotation angle chain-committed for deterministic replay.

**Not Super-GOAT if:** G1 (L2-norm preservation holds to <1e-4) fails — if the cos/sin coupling doesn't preserve norm in practice (numerical drift), the whole stability thesis collapses and the primitive demotes to Gain (a curiosity rotation with no stability guarantee). OR if G2 (behavior rank preservation during rotation) < 0.95 — if rotating the HLA halves changes which action the NPC selects beyond the intended affect shift, the primitive is dangerous and demotes.

### One-line reasoning

The phase-modulated coupling `cos α ⊙ a + sin α ⊙ b` is a known math operation (complex rotation); its value here is as a **new latent operation class** (norm-preserving unitary rotation, parameterized by a sigmoid-bounded phase from a frozen direction vector) that is genuinely missing from our dot-product / additive / convex-combo / linear-transport substrate, with strong fusion hooks into HLA (subspace gating), DEC (Hodge mixer), NeuronShard (spectral/spatial retrieval), and LatCal (committed phase).

### Routing

- **katgpt-rs/.plans/322_phase_modulated_coupling_primitive.md** — open primitive. `phase_rotation_gate_into` + `compute_phase_from_projection` + scalar/per-channel variants. Feature flag `phase_rotation_coupling`. GOAT gate G1–G4.
- **riir-ai/.research/159_phase_rotation_subspace_gate_guide.md** — private guide (this Super-GOAT's selling-point doc, game-runtime domain).
- **riir-ai/.plans/** — deferred until katgpt-rs primitive passes G1–G2.
- Cross-reference guides (speculative, create only if a fusion is pursued): `riir-chain/.research/` (committed-phase variant), `riir-neuron-db/.research/` (shard-half retrieval variant).

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Phase is `sigmoid(dot) · π/2`; coupling is cos/sin Hadamard. No gradients. |
| Latent-to-latent preferred | ✅ Operates entirely in latent space on two halves `(a, b)`. |
| Use sigmoid not softmax | ✅ Phase is sigmoid-bounded; cos/sin is monotone rotation, not winner-take-all. |
| Freeze/thaw over fine-tuning | ✅ Direction vectors are BLAKE3-committed; per-faction/per-NPC directions are atomic Arc-swap. The rotation is an overlay, NOT a mutation of frozen state. |
| 5-repo discipline | ✅ Open primitive → katgpt-rs; game integration → riir-ai (primary guide); chain commitment → riir-chain (cross-ref); shard retrieval → riir-neuron-db (cross-ref). |
| Raw scalars at sync boundary | ✅ The *phase angle α* is a raw scalar that crosses sync (committed via LatCal in F4). The coupling happens locally in latent space. The two halves (a, b) are latent, never synced directly. |
| Zero-alloc hot path | ✅ All buffers caller-provided scratch; SIMD-vectorizable inner loop. |

---

## 5. §3.5 Modelless-First Check (mandatory before any riir-train deferral)

**Question:** Is UFO's phase-modulated coupling modelless-distillable, or does it require training?

**Native form:** Training-only. `γ_θ` (the phase network) is a trained MLP. The spectral encoder `Ψ_H` and spatial basis `Φ_S` are trained.

**Path 1 (freeze/thaw snapshot correction):** N/A. This is not a bias-correction problem; it's a new operation class. Path 1 does not apply.

**Path 2 (raw/lora reader-writer hot-swap with deterministic construction):** **PASSES.** Replace the trained `γ_θ` with a **deterministically constructed** phase function: `α = sigmoid(⟨state, direction⟩ · sharpness) · π/2`. The direction vector is a frozen artifact (BLAKE3-committed, thawed at runtime). The coupling `cos α ⊙ a + sin α ⊙ b` is then a closed-form operation — no gradient descent. This is exactly the freeze/thaw + frozen-artifact pattern already shipped in Plan 309 (Latent Field Steering) and Plan 310 (Cross-Resolution Bases). **The phase function is a deterministic construction, not a learned one.**

**Path 3 (latent-space correction):** Not needed — Path 2 unblocks.

**Verdict:** **MODELLESS-VALIDABLE.** No riir-train deferral. The phase-modulated coupling distills to a modelless primitive via Path 2. The PDE-benchmark quality claims (UFO beats DeepONet/FNO on StepHeat etc.) are training-only and belong in riir-train — but the *operation itself* is modelless.

**Documentation of why each path was checked:**
- Path 1: not a bias-correction problem; the primitive is a new op, not a fix.
- Path 2: the phase function `sigmoid(dot) · π/2` is closed-form; direction is a frozen artifact. ✅ unblocks.
- Path 3: subsumed by Path 2 (the latent-space projection IS the phase construction).

---

## 6. Open questions / risks

1. **Does L2-norm preservation hold numerically?** The headline risk. `cos²α + sin²α = 1` holds in exact arithmetic, but `f32` cos/sin can drift. **Mitigation:** G1 measures `|cos²α + sin²α - 1|` across α ∈ [0, π/2] sweep; gate requires < 1e-4. Use the Plan 319 polynomial-Padé approximation (4.9e-3 accuracy) for fast path; libm for cold path.
2. **Does rotating HLA halves preserve behavior rank?** If rotating `[valence, arousal, desperation, calm]` toward `[fear, ...]` changes which action the NPC selects beyond the intended affect shift, the primitive is dangerous. **Mitigation:** G2 measures cosine similarity of action rankings pre/post rotation; gate requires ≥ 0.95.
3. **What's the right direction vector for the phase?** In F1 (HLA), the direction is "combat-vs-social context". How is it derived? **Mitigation:** (a) designer-authored (a zone-level attribute vector), (b) derived from recent damage/deal counts (raw→latent bridge), (c) frozen per-faction (a faction's combat stance is a committed direction). All three are modelless — no training.
4. **Per-channel vs scalar phase.** UFO uses per-channel α (`[f32; D]`). Scalar α is faster (one cos/sin) but less expressive. **Mitigation:** ship both; G3 benchmarks both; default to scalar for hot path (HLA at 20Hz), opt into per-channel for cold path (shard retrieval).
5. **Curse of dimensionality for the per-channel form.** `D` cos/sin calls per tick is expensive at high D (shard D=64). **Mitigation:** polynomial Padé (Plan 319 Issue 003); or restrict per-channel to low-D use cases (HLA D=8).

---

## TL;DR

UFO (arxiv 2605.12700) realizes a cross-domain neural operator via phase-modulated coupling `cos α ⊙ a + sin α ⊙ b`, where the key property `sin²α + cos²α = 1` makes the coupling a **norm-preserving unitary rotation** in the (a, b) plane. In native form the phase network is trained (→ riir-train for the PDE-benchmark quality claims); but the **§3.5 modelless unblock (Path 2: deterministic phase construction)** replaces the trained `γ_θ` with `α = sigmoid(⟨state, direction⟩) · π/2`, turning the coupling into a frozen-artifact-gated rotation — a genuinely new latent operation class that no existing primitive in our codebase matches (additive steering inflates norm; convex-combo gate preserves L1 not L2; Clifford detects rotation but doesn't apply it; FUNCATTN/cross-res are linear not rotational). Super-GOAT candidate pending G1 (norm preservation <1e-4) and G2 (behavior rank preservation ≥0.95). Headline fusion: HLA subspace gating (NPCs rotate affect between combat and social modes without drift) × LatCal committed phase (the rotation angle is chain-committed for deterministic replay). Open primitive in katgpt-rs Plan 322; private guide at riir-ai/.research/159.
