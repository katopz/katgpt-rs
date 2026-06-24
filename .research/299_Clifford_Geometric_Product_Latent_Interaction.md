# Research 299: Clifford Geometric Product — Channel-Wise Latent Interaction Primitive

> **Source:** [CliffordNet: All You Need is Geometric Algebra](https://arxiv.org/abs/2601.06793) — Zhongping Ji, arXiv:2601.06793v2, Feb 2026
> **Date:** 2026-06-25
> **Status:** Active
> **Related Research:** 219 (DEC parent), 296 (Stokes/DEC vocabulary crosswalk), 065 (RotorQuant — Clifford rotors for quantization), 020 (OFT — skew-symmetric Cayley, riir-ai), 024 (Neuro-Symbolic Chain — skew-symmetric role embeddings, riir-ai)
> **Related Plans:** 251 (DEC operators), 314 (Stokes wrappers), 318 (latent functor rank-k — first-order cross-product), 214 (LinOSS symplectic, riir-ai), 319 (this primitive — new)
> **Classification:** Public

---

## TL;DR

CliffordNet's core modelless contribution is the **Geometric Product as a channel-wise latent-space interaction**: `uv = u·v + u∧v`, where the inner product `u·v` captures **coherence/alignment** and the exterior wedge product `u∧v` captures **structure/orthogonality** ("algebraic completeness"). The paper trains a vision backbone from scratch (AdamW, 200 epochs) and shows the geometric interaction is dense enough to **remove FFNs entirely** (2.6M params matches ResNet-18's 11.2M on CIFAR-100).

**Distilled for katgpt-rs (modelless, inference-time):** The channel-wise geometric product is a zero-allocation `O(D·|S|)` primitive — Hadamard + cyclic shift + subtract — that produces a **new signal dimension** (structural divergence) orthogonal to every existing dot-product-based latent op in the codebase. Every latent operation we ship (HLA projection, latent functor coherence, shard retrieval, DEC cochain ops, CGSP curiosity) uses the inner product only. The wedge product is genuinely missing.

**Verdict: GOAT** (not Super-GOAT). The primitive is mathematically known (Clifford 1878); the value is in fusing it with our specific substrate. The "new capability class" claim (structural complementarity detection across HLA/functor/shard) is plausible but **unproven** — it needs the GOAT gate to demonstrate the wedge signal carries information the dot product misses in our substrate. If the gate passes with a genuine information gain, elevate to Super-GOAT with the riir-ai guide. **No guide created yet** per skill rule (do not write "Super-GOAT candidate" without committing all 4 YES).

---

## 1. Paper Core Findings

### 1.1 The Geometric Product (the transferable primitive)

The Clifford geometric product of two vectors is:

```
uv = u·v + u∧v
```

- **`u·v` (generalized inner product)** — symmetric, captures alignment/similarity. Standard NN primitive (attention, gating).
- **`u∧v` (exterior wedge product)** — anti-symmetric, captures orthogonality/structural variation. Constructs a **bivector** (oriented plane / 2-blade). `u∧v = -v∧u`, `u∧u = 0`. **Discarded by every standard dot-product primitive.**

The paper's central claim: "algebraic completeness" requires BOTH. Standard NN uses only the scalar component, losing the bivector structure.

### 1.2 Sparse Rolling Realization (the efficient implementation)

The full wedge product `u∧v` over a D-dim channel space is `O(D²)` (full outer product matrix). CliffordNet approximates it via **cyclic channel shifts** `T_s` for a sparse set `S ⊆ {1, 2, 4, ..., D/2}`:

```
D_s(H, C) = SiLU(H ⊙ T_s(C))              # inner/coherence term (per shift s)
W_s(H, C) = H ⊙ T_s(C) − T_s(H) ⊙ C       # wedge/structure term (per shift s)
```

- `T_s(x)` = cyclic shift of vector `x` by offset `s` (channel `(c+s) mod D`).
- Each shift extracts one "spectral diagonal" of the full `D×D` interaction matrix.
- Total complexity: `O(N · D · |S|)` — linear in sequence length AND channel dim.

**This is the modelless core.** It's Hadamard product + roll + subtract + activation. Zero allocation, SIMD-friendly, O(D·|S|).

### 1.3 Context Instantiation (what `C` is)

The geometric product `F(H, C) = P(H · C + H ∧ C)` requires a context vector `C`:

- **gFFN-L (Local):** `C = ΔH` — discrete Laplacian (two stacked 3×3 depthwise convs minus identity). High-pass filter, captures local structural variation.
- **gFFN-G (Global):** `C = GlobalAvgPool(H)` — global mean field. Captures scene-level coherence.
- **gFFN-H (Hybrid):** `C = ΔH + β·GlobalAvg(H)` — superposition.

### 1.4 Gated Geometric Residual (GGR)

The layer update is a first-order Euler discretization with gating:

```
H_l = H_{l-1} + γ ⊙ ( SiLU(H_{l-1}) + Gate(H_{l-1}, H_geo) ⊙ H_geo )
```

where `H_geo` is the geometric product output and `Gate` is a sigmoid gate over the concatenation `[H_{l-1}, H_geo]`. This matches our existing sigmoid-gating discipline.

### 1.5 Empirical Results (the training claim — NOT modelless)

- CliffordNet-Lite (2.6M params, no FFN) = 79.05% CIFAR-100, beats ResNet-18 (11.2M, 76.75%) and ViT-Tiny (2.7M, 65.87%).
- Wedge-only variant (no energy/self-magnitude) = 77.76%, nearly matching Inner-only (78.17%). **The bivector alone is almost as discriminative as the scalar — structural topology carries most of the signal.**
- Differential mode (`C = ΔH`) beats Absolute mode (`C = C_loc`) by ~1.4%.

### 1.6 What's Training-Only (→ riir-train, do NOT distill here)

- The vision backbone architecture (Dual-Stream Geometric Block, isotropic columnar design).
- The AdamW training recipe, DropPath, AutoAugment, cosine annealing.
- The learned projection `P` that maps multivector → vector space.
- The "No-FFN" architectural claim (only holds after training the geometric block's projections).

**The training-only parts belong in riir-train.** The modelless transferable primitive is the channel-wise geometric product operation itself.

---

## 2. Distillation

### 2.1 The transferable primitive

```rust
/// Channel-wise Geometric Product (modelless, zero-alloc).
///
/// Computes both coherence (inner) and structure (wedge) terms for two latent
/// vectors via sparse cyclic shifts. Returns (scalar_energy, bivector_structure)
/// per shift — caller fuses them with a sigmoid gate.
///
/// `u`, `v`  : [D] latent vectors
/// `shifts`  : &[usize] sparse offset set S (e.g. &[1, 2, 4, 8])
/// `dot_out` : [D] coherence term (Hadamard + SiLU)
/// `wedge_out`: [D] structure term (anti-symmetric cross)
/// `scratch_u`, `scratch_v` : [D] pre-allocated roll buffers
pub fn geometric_product_into(
    u: &[f32], v: &[f32], dim: usize,
    shifts: &[usize],
    dot_out: &mut [f32],   // Σ_s SiLU(u ⊙ T_s(v))
    wedge_out: &mut [f32], // Σ_s (u ⊙ T_s(v) − T_s(u) ⊙ v)
    scratch_u: &mut [f32], scratch_v: &mut [f32],
) {
    dot_out[..dim].fill(0.0);
    wedge_out[..dim].fill(0.0);
    for &s in shifts {
        // T_s(v): cyclic shift v by s
        cyclic_shift_into(v, dim, s, scratch_v);
        cyclic_shift_into(u, dim, s, scratch_u);
        for c in 0..dim {
            let dot_term = u[c] * scratch_v[c];        // u_c · v_{c+s}
            let wedge_term = dot_term - scratch_u[c] * v[c]; // u_c v_{c+s} − u_{c+s} v_c
            // SiLU on dot term (coherence gate)
            dot_out[c] += dot_term / (1.0 + (-dot_term).exp());
            wedge_out[c] += wedge_term;
        }
    }
}
```

**Complexity:** `O(D · |S|)` per call, zero allocation after scratch init. SIMD-vectorizable (chunked 4-wide). Gateable by feature flag.

### 2.2 Why this is NOT redundant with DEC `exterior_derivative`

This is the critical distinction (vocabulary-translation defense, per R296 lesson):

| Aspect | DEC `exterior_derivative` (shipped, Plan 251) | CliffordNet geometric product (this primitive) |
|--------|-----------------------------------------------|------------------------------------------------|
| **Domain** | Cochains over a **spatial cell complex** | Two latent vectors at a **single point** |
| **Operation** | `d_k = B_{k+1}^T` — boundary matrix transpose | `uv = u·v + u∧v` — vector product |
| **Rank flow** | `C_k → C_{k+1}` (spatial: vertex→edge→face→volume) | `R^D × R^D → R^D` (channel cross-terms) |
| **Cross-channel?** | **No** — applies independently per feature channel | **Yes** — bivector is explicitly cross-channel `(u_c · v_{c+s} − u_{c+s} · v_c)` |
| **Captures** | Spatial boundary flux, curl, divergence | Channel-oriented orthogonality, structural rotation |
| **Anti-symmetric in** | Spatial boundary orientation (signed face/edge) | Channel index pair `(c, c+s)` |

**They are complementary.** DEC captures spatial structure (where things are on the map); Clifford wedge captures channel structure (how latent dimensions relate within a single vector). A fusion gives **spatial-channel algebraic completeness** — apply DEC `d` for spatial boundary, apply Clifford `∧` for channel cross-terms.

### 2.3 Why this is NOT redundant with RotorQuant (Research 65)

RotorQuant uses Clifford **rotors** to **construct orthogonal matrices** `R = Cayley(R')` for KV cache quantization rotation (decorrelation). It parameterizes a rotation and applies `vR`. CliffordNet uses the geometric product as an **interaction mechanism** between two vectors. Different application: RotorQuant = matrix construction for decorrelation; CliffordNet = interaction signal extraction.

### 2.4 Why this is NOT redundant with OFT (Research 020, riir-ai)

OFT uses skew-symmetric generators `R' = -R'^T` via Cayley transform `R = (I-R')(I+R')⁻¹` to parameterize orthogonal matrices for adapter training. Anti-symmetric structure, but for **orthogonal parameterization**, not interaction. Same distinction as RotorQuant.

### 2.5 Why this is NOT redundant with Latent Functor rank-k (Plan 318)

Plan 318's "first-order cross-product `Φ_t^T · Ψ_s`" is the closest cousin — it captures rotational structure that second moments miss. But it's a **batch Gram matrix** over `n` sample pairs, not a per-point anti-symmetric Hadamard. The Clifford wedge gives a **per-NPC rotational signal** at `O(D·|S|)` cost, vs Plan 318's `O(n·D·k)` batch estimate. They compose: Plan 318 estimates the operator `C` from a batch; the Clifford wedge gives a per-instance structural feature that could feed into `C` estimation.

### Fusion

The closest cousins across all five repos, and what fusing each with the channel-wise geometric product produces:

1. **× DEC `exterior_derivative` (Research 219, Plan 251)** → **spatial-channel algebraic completeness**. DEC captures spatial boundary structure on cochains; Clifford wedge captures channel cross-term structure at each cell. Apply `∧` per-cell as a channel-aware refinement of `d_k`. **Novel capability**: terrain cochains that distinguish "two zones with the same threat scalar but orthogonal threat structure" — currently DEC sees them as identical.

2. **× HLA per-NPC affect (riir-engine `hla/`)** → **emotional complementarity signal**. HLA's 8-dim state currently uses dot-product projections (valence/arousal/desperation/calm/fear). The wedge `h_NPC1 ∧ h_NPC2` produces a bivector = the "emotional plane" spanned by two NPCs' affect — captures **tactical complementarity** (one calm+brave, other afraid+desperate = orthogonal mood → formation complement). **Novel capability**: formation-quality scoring that current dot-product coherence cannot detect (two NPCs with high dot-product coherence are redundant; two with high wedge are complementary).

3. **× Latent Functor rank-k (Plan 318)** → **per-instance rotational feature for functor estimation**. Plan 318's primal-form operator `C = (Φ_t^T·Ψ_s)·(Ψ_s^T·Ψ_s + αI)⁻¹` uses first-order cross-products but over a batch. The Clifford wedge gives a per-instance `u∧v` that could serve as an additional feature column in `Ψ_s`, making the operator estimate rotation-aware per-sample, not just batch-rotation-aware.

4. **× NeuronShard style_weights (riir-neuron-db `shard.rs`, 64-dim)** → **structural complementarity retrieval**. Current shard retrieval uses dot-product similarity (`cosine(style_weights_query, style_weights_shard)`). Adding wedge retrieval (`∧`) finds shards with **orthogonal/complementary play styles** — useful for ensemble composition (retrieve a diverse set, not a redundant cluster). **Novel capability**: "retrieve K shards that maximally span the style manifold" instead of "retrieve K most similar shards".

5. **× CGSP curiosity (riir-engine `cgsp_runtime/`)** → **structural surprise dimension**. Current curiosity = entropy/coherence-driven. The wedge between current and predicted belief states = **structural divergence** (the predicted state is orthogonal to current, not just low-coherence). **Novel capability**: NPCs that explore toward structurally novel states, not just uncertain ones.

6. **× LatCal fixed-point (riir-chain `encoding/latcal*.rs`)** → **rotation/tamper detector on committed raw values**. The wedge `u∧v` produces a 2-blade (oriented area) — invariant under scaling but flips under reflection. Could detect tampering that preserves dot-product (norm) but flips orientation. **Speculative** — needs verification that LatCal's 2×2 matrix structure admits a meaningful wedge.

**Strongest fusion candidates**: #2 (HLA emotional complementarity) and #4 (shard structural retrieval). Both produce a new capability class (complementarity detection) that the dot-product-only substrate cannot match.

---

## 3. Verdict

**GOAT.** Provable-gain candidate, not a new-capability-class primitive in isolation.

### One-line reasoning

The channel-wise geometric product (Hadamard + roll + subtract) is a known math operation (Clifford 1878); its value here is as a **new signal dimension** (structural divergence) fused with our existing dot-product-only latent substrate (HLA/functor/shard/DEC). The GOAT gate must prove the wedge signal carries information the dot product misses in our specific substrate before this can be elevated.

### Why not Super-GOAT (honest down-grade)

The four novelty-gate questions:

1. **No prior art in codebase?** ✅ YES — after 3-layer check (notes + code + vocabulary translation). No existing primitive computes the per-point channel-wise anti-symmetric wedge.
2. **New class of behavior?** ⚠️ **PLAUSIBLE BUT UNPROVEN** — "structural complementarity detection" is a new signal, but whether it produces emergent behavior in our substrate needs the GOAT gate. The wedge product itself is known math.
3. **Product selling point?** ⚠️ **DEPENDS ON FUSION** — "NPCs detect emotional complementarity" requires the HLA fusion to actually work. The primitive alone is not a selling point.
4. **Force multiplier?** ✅ YES — connects ≥2 pillars (HLA, functor, shard, DEC, CGSP).

Q2 and Q3 are not confident YES. Per skill rule: "If you are NOT confident enough to commit all 4 YES right now, do not write 'Super-GOAT candidate'." Down-grade to GOAT. If the GOAT gate (Plan 319) shows the wedge carries genuinely orthogonal information to the dot product in HLA or shard retrieval, **then** create the riir-ai guide and elevate.

### Tier justification

| Criterion | Assessment |
|-----------|------------|
| Modelless? | ✅ Yes — Hadamard + roll + subtract, zero backprop. No training. |
| Latent-to-latent? | ✅ Yes — operates on latent vectors, produces latent vectors. |
| Feature flag? | ✅ Will ship behind `geometric_product` opt-in. |
| Sigmoid (not softmax)? | ✅ GGR gate uses sigmoid. Inner term uses SiLU (monotonic, no winner-take-all). |
| Zero-alloc hot path? | ✅ Pre-allocated scratch buffers, SIMD-vectorizable. |
| Fusion-first? | ✅ Six fusion candidates identified, strongest = HLA + shard. |
| GOAT gate definable? | ✅ See Plan 319 §"GOAT gate" — prove wedge carries orthogonal info vs dot product. |

### Routing

- **Open primitive** → `katgpt-rs/crates/katgpt-core/src/` (generic math, no game semantics). New module `geometric_product.rs` under a new `algebra/` subtree or directly in `math/`.
- **Plan** → `katgpt-rs/.plans/319_geometric_product_latent_interaction.md` (open primitive + benchmark).
- **riir-ai/riir-chain/riir-neuron-db application** → deferred until GOAT gate passes. If it passes, create:
  - `riir-ai/.research/155_clifford_wedge_npc_emotional_complementarity_guide.md` (HLA fusion selling point)
  - `riir-neuron-db/.research/007_shard_structural_retrieval_guide.md` (shard retrieval selling point)

### What stays public vs private (if elevated to Super-GOAT later)

- **Public (katgpt-rs)**: the `geometric_product_into` primitive, the cyclic-shift kernel, the GOAT benchmark harness. Generic math, no game/chain/shard semantics.
- **Private (riir-ai)**: HLA emotional-complementarity application, CGSP structural-surprise curiosity, formation-quality scoring.
- **Private (riir-neuron-db)**: shard structural-complementarity retrieval, manifold-spanning ensemble selection.
- **Private (riir-chain)**: LatCal orientation-tamper detection (speculative).

---

## 4. Modelless-First Check (§3.5 protocol)

The paper is training-focused (vision backbone, AdamW). Before deferring anything to riir-train, check the three modelless paths:

1. **Freeze/thaw snapshot correction?** N/A — no systematic bias to correct. The primitive is a deterministic math op, not a biased estimator.
2. **Raw/lora reader-writer hot-swap?** N/A — no adapter needed. The geometric product is applied directly to latent vectors.
3. **Latent-space correction?** ✅ **This IS the latent-space primitive.** The geometric product is a modelless latent-to-latent operation. No training required to use it.

**Conclusion: fully modelless.** No riir-train dependency. The training-only parts (backbone architecture, learned projection P, AdamW recipe) are noted as "→ riir-train" and not distilled here.

---

## 5. Open Questions (track in Plan 319)

1. **Does the wedge carry orthogonal information in our substrate?** The paper proves it on CIFAR-100 vision backbones. Does `h_NPC1 ∧ h_NPC2` on HLA's 8-dim affect carry formation-quality signal that `h_NPC1 · h_NPC2` misses? This is the G1 gate.
2. **Shift set S for low-dim latents?** CliffordNet uses `S = {1,2,4,8,16}` for D=64+. For HLA D=8, `S = {1,2,4}` covers all non-trivial shifts. For shard D=64, `S = {1,2,4,8,16,32}`. Need to verify the shift set is expressive enough at low D.
3. **Wedge magnitude scale?** The wedge `u∧v` has different magnitude scale than `u·v` (it's a difference, not a sum). The GGR gate `Gate(H, H_geo)` must calibrate the scale. Sigmoid gate handles this naturally.
4. **Anti-symmetric wrap-around sign?** Cyclic shift wraps channel indices; the wedge's anti-symmetry means wrapped terms flip sign. CliffordNet absorbs this into the learned projection P. For our modelless use (no learned P), we must either (a) use non-wrapping shifts (zero-pad), or (b) track sign explicitly. Plan 319 must resolve this.

---

## TL;DR

CliffordNet's channel-wise geometric product `uv = u·v + u∧v` is a modelless latent-interaction primitive (Hadamard + cyclic shift + subtract, `O(D·|S|)`, zero-alloc) that adds a **structural-divergence signal dimension** missing from our dot-product-only latent substrate. It is **complementary** (not redundant) to DEC's spatial `exterior_derivative`, RotorQuant's orthogonal rotors, OFT's skew-symmetric Cayley, and Plan 318's batch cross-product. **Verdict: GOAT** — the primitive is known math; the value is in fusing it with HLA (emotional complementarity), shard retrieval (structural diversity), and CGSP curiosity (structural surprise). Plan 319 implements the open primitive behind `geometric_product` feature flag with a GOAT gate that proves the wedge carries orthogonal information vs the dot product. **If the gate passes, elevate to Super-GOAT** with riir-ai + riir-neuron-db guides. No guide created now (honest down-grade per skill rule — Q2/Q3 not confident YES).
