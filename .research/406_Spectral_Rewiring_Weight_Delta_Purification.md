# Research 406: Spectral Rewiring (SAR) — Weight Delta Purification via Base SVD Projection

> **Source:** [Spectral Rewiring for Exploration, Purification, and Model Merging](https://arxiv.org/abs/2607.03065) — Z. Zhang, H. Yu, H. Gao, H. Wu, Y. Song, W.-Y. Ma, Y.-Q. Zhang, H. Zhou (Tsinghua AIR / ByteDance Seed), arXiv:2607.03065, Jul 2026
> **Date:** 2026-07-10
> **Status:** Active — GOAT fusion candidate
> **Related Research:** 231 (SOPTV — off-principal task vectors, complementary), 238 (LoRA-Muon gauge-invariant composition), 191 (Prism capability substrate), 039 (SpectralQuant KV), 100 (EGA spectral salience), 094 (MeMo TIES merging)
> **Related Plans:** 094 (TIES merging — upgrade target), 264 (SOPTV — complementary fusion), 025 (LoRA hot-swap — purification target), 216 (SubstrateGate — parallel capability extraction)
> **Classification:** Public — generic spectral matrix operation, no game/chain/shard semantics

---

## TL;DR

SAR projects a weight delta ΔW = W_RL − W_0 onto the base model's SVD spectral subspace, extracting a compact "rewiring matrix" M = UᵀΔWV. The projected delta ΔW* = UMVᵀ retains the reasoning-effective (or personality-effective) component while discarding off-manifold drift. The off-diagonal elements of M represent cross-skill "rewiring" — many-to-one logical synthesis that is the geometric mechanism for compositional reasoning.

**Distilled for katgpt-rs (modelless, inference-time):**

The SAR projection itself is a pure deterministic matrix operation: SVD of the base weights (computed once, offline) + two matrix multiplications to extract M + one reconstruction. No training. The headline result (extract reasoning core from RL deltas) requires trained W_RL → routes to riir-train. But the **modelless residue** — project ANY weight delta (freeze/thaw delta, LoRA overlay, consolidation delta) onto the base's spectral subspace to purify it — is a genuinely novel primitive not shipped anywhere in the 5-repo quintet.

**Verdict: GOAT.** Novel mechanism (Q1 ✓ no prior art), new capability class (Q2 ✓ spectral purification + positive-sum merging), force multiplier (Q4 ✓ connects freeze/thaw + consolidation + LoRA + TIES merging). Q3 (product selling point) needs GOAT-gate validation for NPC-scale deltas before Super-GOAT promotion.

---

## 1. Paper Core Findings

### 1.1 The Geometric Framework — SVD as Functional Basis

For a pretrained weight matrix W₀ ∈ ℝ^(d_out × d_in), the SVD W₀ = UΣVᵀ decomposes the forward pass into:

- **Information read-in:** right singular vectors vᵢ act as feature detectors (vᵢᵀx)
- **Information read-out:** left singular vectors uᵢ define output directions, weighted by σᵢ

The spectral manifold **S_r(W₀) = span(U ⊗ V)** defines the base model's "functional library" — pretrained atomic skills encoded as singular vector pairs.

### 1.2 Subspace-Aligned Rewiring (SAR)

Given a trained model W_RL and base W₀, the update ΔW = W_RL − W₀ decomposes:

```
ΔW = ΔW* + ΔW⊥

where:
  ΔW*  = P_U · ΔW · P_V = U(UᵀΔWV)Vᵀ = UMVᵀ    (on spectral manifold)
  ΔW⊥  = ΔW − ΔW*                                  (off-manifold residual)
  M    = UᵀΔWV ∈ ℝ^(r×r)                          (rewiring matrix)
```

**Algorithm 1 (SAR):**
1. SVD: W₀ = UΣVᵀ (top-k)
2. Isolate: ΔW ← W_RL − W₀
3. Low-rank: extract top-k component ΔW_k from ΔW
4. Project: M ← UᵀΔW_k V
5. Reconstruct: ΔW* ← UMVᵀ
6. Return: W_SAR ← W₀ + ΔW*

### 1.3 The Rewiring Matrix — Diagonal vs Off-Diagonal

The forward pass of the SAR-projected model is:

```
y = U(Σ + M)Vᵀx = Σᵢ [ (σᵢ + Mᵢᵢ)(vᵢᵀx) + Σ_{j≠i} Mᵢⱼ(vⱼᵀx) ] uᵢ
                     └── rescaling ──┘    └── cross-skill rewiring ──┘
```

- **Diagonal Mᵢᵢ:** rescale isolated pretrained capabilities (amplify/suppress)
- **Off-diagonal Mᵢⱼ (i≠j):** route evidence from skill vⱼ to output uᵢ — **many-to-one logical synthesis**, the geometric mechanism for compositional reasoning

### 1.4 Three Application Uses

| Use | Mechanism | Result |
|-----|-----------|--------|
| **Extraction** | Project RL delta onto base spectral subspace, keep top-1% rank | Preserves >99% Pass@1 using ~0.58% of params; improves Pass@k exploration |
| **Purification** | Project Mix-RL delta onto spectral subspace | Releases suppressed cross-domain capability (+1.48% LCB v5, +0.95% v6 on 32B) |
| **Merging** | Filter each expert delta through spectral geometry before merging | Merged model surpasses best single-domain experts (positive-sum) |

### 1.5 Boundary Cases (§5, Appendix D)

SAR works best in "elicitation regime" — RL that reorganizes existing capabilities:
- **Heavily trained RL (>4000 steps):** recovers ~43% (comparable to early DeepScaleR), not final ~52%. Some rewriting happens beyond the spectral manifold.
- **Code RL from base model:** less compatible — needs domain SFT first to establish conventions.
- **PPO with dense critic rewards:** reduced compatibility — dense rewards push updates off-manifold.

---

## 2. Distillation

### 2.1 §3.5 Modelless Unblock Protocol — MANDATORY

**The paper's headline (extract reasoning core from RL delta) requires a trained W_RL.** We have no RL training (modelless-first mandate). Before deferring, exhaust the three modelless paths:

| Path | Check | Result |
|------|-------|--------|
| **1. Freeze/thaw correction** | Can a frozen snapshot + spectral projection fix a delta? | **YES** — given two frozen states (W₀, W₁), the delta ΔW = W₁ − W₀ is available modellessly. Project onto W₀'s spectral subspace. The SVD is computed once at freeze time. ✓ MODELLESS |
| **2. Raw/lora reader-writer hot-swap** | Can a deterministically constructed LoRA fix it? | **YES** — ΔW* = UMVᵀ is rank ≤ r, so it IS a low-rank overlay (LoRA). The projection is deterministic (no gradient descent). The reader = U, writer = V, values = M. ✓ MODELLESS |
| **3. Latent-space correction** | Can a dot-product projection fix it? | **N/A** — SAR operates in weight space, not latent space. The mechanism is weight-space SVD projection. |

**§3.5 verdict: MODELLESS-VALIDABLE.** The SAR projection is a deterministic weight-space operation. The inputs (two frozen states, or a base weight + LoRA overlay) are available at inference time. No riir-train deferral.

**Important honest caveat:** The paper proves the mechanism for RL deltas at 1.5B–32B scale. Our modelless application (freeze/thaw deltas, LoRA overlays, consolidation deltas) operates at different scales and different semantics. The spectral concentration property may not hold identically. This needs the GOAT gate.

### 2.2 Where SAR Applies in the 5-Repo Quintet

SAR requires an actual weight **matrix** (d_out × d_in) to SVD-decompose. Three concrete application sites:

| Site | Repo | Matrix | Delta source | Purification target |
|------|------|--------|--------------|---------------------|
| **LoRA reader/writer pairs** | katgpt-rs (Plan 025) | Base weight W₀ (Wq, Wk, Wv, Wo, MLP) | Reader × writer product | Purify LoRA overlay: keep on-manifold, discard off-manifold drift |
| **Full transformer weights** | katgpt-rs | W₀ (attention + MLP linears) | Freeze/thaw delta between two snapshots | Purify personality/capability delta |
| **Consolidation deltas** | riir-neuron-db | Base shard weights | Multi-shard consolidation delta | Spectrally align consolidation to reduce interference |

**Does NOT directly apply to:** `NeuronShard::style_weights[64]` — this is a 64-dim **vector**, not a matrix. The SVD projection requires a matrix. The 64-dim style_weights would need to be reshaped (e.g., 8×8) or embedded in a larger matrix for SAR to apply. This is a known limitation.

### 2.3 Fusion — the Highest-Value Combinations

#### Fusion A: SAR × TIES Merging (Plan 094) — Spectral Upgrade

TIES merging currently uses magnitude-based filtering (trim top-ρ + sign election + disjoint merge). SAR replaces this with spectral filtering: project each task vector onto the base SVD subspace before merging.

```
TIES (current):   trim(τ) → elect_sign → disjoint_merge
SAR merge:        project(τ, base_SVD) → merge_projected → reconstruct
```

The paper proves SAR merging produces models that **surpass the best single-domain experts** (Table 4) — a capability TIES cannot achieve. This is a direct upgrade to Plan 094.

#### Fusion B: SAR × SOPTV (Plan 264 / Research 231) — Complementary Decomposition

**Fascinating tension:** SOPTV (Research 231) found that OPD distillation deltas are **off-principal** (p₁₀ ≤ 1% projection onto top-10% singular directions). SAR finds that RL deltas' reasoning-effective component is **on-principal** (projects onto base SVD subspace).

These are **complementary, not contradictory:**
- OPD (distillation) deltas move off-principal → student diverges from teacher's principal directions
- RL (reward optimization) deltas stay on-principal → reasoning is elicited by rewiring existing skills

**Fusion idea:** decompose any weight delta into TWO components:
```
ΔW = ΔW_on_principal (SAR)  +  ΔW_off_principal (SOPTV)
```
The on-principal component captures capability rewiring (SAR); the off-principal component captures task-specific adaptation (SOPTV). Store both compactly. This is a **two-component weight delta decomposition** that neither paper alone provides.

#### Fusion C: SAR × Freeze/Thaw (riir-neuron-db) — Purified Personality Snapshots

When freezing a personality snapshot, project the delta onto the base shard's spectral subspace. Store only the compact rewiring matrix M (r×r) in the `MerkleFrozenEnvelope` instead of the full delta. This:
- Reduces freeze storage (M is ~0.58% of params at top-1% rank)
- Purifies personality changes (removes drift/noise)
- Makes personality merging positive-sum (spectral-aligned deltas are compatible)

#### Fusion D: SAR × LoRA Hot-Swap (Plan 025) — Spectral LoRA

When applying a reader/writer LoRA pair, project the LoRA product onto the base weight's spectral subspace. The result is a "spectral LoRA" — a deterministically purified overlay that keeps only the on-manifold component. More compatible across domains than the raw LoRA.

---

## 3. Verdict

### Tier: GOAT

**One-line reasoning:** Novel modelless primitive (spectral projection of weight deltas — no prior art across 5 repos) with a new capability class (spectral purification + positive-sum merging) that connects to ≥4 existing pillars (freeze/thaw, consolidation, LoRA, TIES merging). Not Super-GOAT because Q3 (product selling point for NPC-scale deltas) needs GOAT-gate validation.

### Novelty Gate (§1.5)

| Q | Criterion | Answer | Evidence |
|---|-----------|--------|----------|
| Q1 | No prior art? | **YES** | Grep across all 5 repos for `spectral.*(project\|subspace\|align)`, `SVD.*(project\|delta)`, `rewiring matrix`, `subspace.aligned`. Closest cousins: TIES (Plan 094, magnitude-based), SOPTV (Plan 264, off-principal sparse storage), Prism (Plan 216, channel masks), LoRA-Muon (R238, gauge rebalancing). **None** project weight deltas onto base SVD spectral subspaces. |
| Q2 | New class of behavior? | **YES** | "Spectral purification of weight deltas" + "positive-sum merging via spectral alignment" are new capability classes. No incumbent can do positive-sum merging (TIES is zero-sum at best). |
| Q3 | Product selling point? | **PARTIAL** | Can state: "Our freeze/thaw runtime spectrally purifies personality deltas, so NPC personality drift is eliminated and personality merging is positive-sum." But this is **unvalidated** for NPC-scale deltas (64-dim style_weights, small LoRA overlays). Paper proves for 1.5B–32B LLM weights. |
| Q4 | Force multiplier? | **YES** | Connects to: freeze/thaw (P2 neuron-db), consolidation (Raven/δ-Mem), LoRA hot-swap (Plan 025), TIES merging (Plan 094), ArchetypeBlendShard (Plan 336). |

**Q3 is the gate.** If the GOAT gate validates spectral concentration for NPC-scale deltas → Super-GOAT promotion path opens. Until then, GOAT.

### MOAT Gate (§1.6)

| Domain | Assessment |
|--------|------------|
| **katgpt-rs** (public engine) | **In scope.** The primitive is a paper-derived fundamental spectral matrix operation. Ships behind feature flag `spectral_rewire`. GOAT gate decides promote-to-default. ✓ |
| **riir-neuron-db** (private shards) | **Application target.** Freeze/thaw purification + consolidation alignment are shard-internal. The open primitive (spectral_rewire function) lives in katgpt-rs; the shard-specific application (purified MerkleFrozenEnvelope) lives here. |
| **riir-ai** (private runtime) | **Application target.** LoRA hot-swap purification + personality merge upgrade are runtime concerns. |

### §3.5 Modelless Unblock — MODELLESS-VALIDABLE

The SAR projection is deterministic. No riir-train deferral. The three modelless paths:
1. Freeze/thaw correction: ✓ (project delta between two frozen states)
2. Raw/lora hot-swap: ✓ (ΔW* = UMVᵀ is a deterministically constructed low-rank overlay)
3. Latent correction: N/A (weight-space operation)

**Honest boundary:** The paper's "extraction" use case (extract reasoning core from RL delta) requires a trained W_RL — that specific application routes to riir-train. The modelless residue (spectral projection of ANY weight delta) is what we distill.

---

## 4. Vocabulary Crosswalk (Paper → Codebase)

| Paper term | Codebase equivalent | Grep target |
|------------|---------------------|-------------|
| "spectral subspace" / "spectral manifold" | SVD basis of weight matrix | `svd`, `singular`, `eigenbasis`, `spectral` |
| "rewiring matrix" M | Cross-skill interaction matrix | `rewiring`, `cross.*skill`, `interaction matrix` |
| "subspace-aligned" | Projected onto base SVD | `subspace.*project`, `spectral.*align` |
| "reasoning core" | Capability-effective delta component | `capability.*extract`, `substrate`, `prism` |
| "purification" | Remove off-manifold drift | `purif`, `drift.*remov`, `spectral.*filter` |
| "positive-sum merging" | Spectral-aligned merge | `ties.*merge`, `task.*arithmetic`, `spectral.*merge` |
| "off-diagonal rewiring" | Cross-direction interaction | `off.*diagonal`, `cross.*channel`, `rewiring` |

**Codebase SVD infrastructure that CAN be reused:**
- `JacobianSvdScratch` (katgpt-core) — SVD via Jacobian trick
- Tucker/HOSVD (katgpt-core, Plan 326) — tensor SVD
- `subspace_phase_gate` (Plan 301) — PCA via Jacobian-SVD
- `spectral_flatness` (riir-neuron-db) — spectral energy concentration metric

---

## 5. Open Primitive Sketch (katgpt-rs)

```rust
/// Spectral Rewiring — project a weight delta onto the base model's SVD spectral subspace.
///
/// Given base weights W₀ and delta ΔW, compute the spectral projection:
///   W₀ = UΣVᵀ  (SVD of base, top-rank)
///   M = UᵀΔWV  (rewiring matrix — cross-skill interactions)
///   ΔW* = UMVᵀ  (spectral projection — on-manifold component)
///
/// The rewiring matrix M is compact (r×r) and captures the capability-effective
/// component of the delta. Off-diagonal Mᵢⱼ represent cross-skill rewiring.
///
/// Modelless: SVD + matrix multiply. No training, no gradient descent.
pub fn spectral_rewire(
    w0: &[f32],       // base weights, row-major (d_out × d_in)
    delta: &[f32],    // weight delta, same shape
    d_out: usize,
    d_in: usize,
    rank: usize,      // top-k spectral rank
    scratch: &mut SpectralRewireScratch,
) -> SpectralRewireResult;

pub struct SpectralRewireResult {
    /// The purified delta ΔW* = UMVᵀ (on-manifold component)
    pub delta_star: Vec<f32>,
    /// The compact rewiring matrix M = UᵀΔWV (r×r)
    pub rewiring_matrix: Vec<f32>,
    /// The off-manifold residual ΔW⊥ = ΔW − ΔW*
    pub residual: Vec<f32>,
    /// Frobenius norm ratio ‖ΔW*‖/‖ΔW‖ — on-manifold energy fraction
    pub on_manifold_fraction: f32,
}
```

**Feature flag:** `spectral_rewire` (opt-in, GOAT gate required before default promotion)

**GOAT gate plan:**
- G1: On-manifold fraction > 0.5 for synthetic rank-k deltas (spectral concentration holds)
- G2: Purified delta preserves the base model's top-k singular directions
- G3: No regression on existing TIES merging tests
- G4: Alloc-free inner loop (reuse scratch buffers)
- G5: Latency < 1ms for 512×512 matrix at rank-32 on SIMD
- G6: Feature-isolation clean (no interaction with other features)

---

## 6. Related Work — Closest Cousins

| Note/Plan | Mechanism | Relationship to SAR |
|-----------|-----------|---------------------|
| **R231 / Plan 264 (SOPTV)** | Sparse off-principal task vector storage | **Complementary** — SOPTV captures off-principal, SAR captures on-principal. Fusion B decomposes delta into both. |
| **R238 (LoRA-Muon)** | Gauge-invariant factor composition | **Different axis** — gauge rebalancing of factor pairs, not delta projection onto base SVD |
| **R191 / Plan 216 (Prism/SubstrateGate)** | MLP channel mask capability extraction | **Parallel** — channel-space masking vs SVD-subspace projection. Both extract capability, different geometric domain |
| **Plan 094 (TIES Merging)** | Magnitude-based task vector merge | **Upgrade target** — SAR replaces magnitude filtering with spectral filtering for positive-sum merging |
| **R039 (SpectralQuant)** | Eigenbasis KV cache compression | **Different domain** — KV cache, not weight deltas |
| **R100 (EGA)** | Spectral salience attention gating | **Different domain** — attention gating, not weight delta purification |
| **Plan 156 (spectral_hierarchy)** | Eigenspace alignment KG diagnostics | **Different domain** — KG extraction validation, not weight deltas |
| **spectral_flatness (riir-neuron-db)** | Wiener entropy for lottery-ticket init | **Different use** — initialization diagnostic, not delta purification |

---

## 7. Honest Caveats and Limitations

1. **Scale mismatch.** The paper proves SAR for 1.5B–32B parameter transformers. Our NPC-scale weight matrices are much smaller (LoRA overlays, 64-dim style_weights). The spectral concentration property (reasoning-effective component in top-1% rank) may not hold identically. The GOAT gate must validate this.

2. **Vector vs matrix.** `NeuronShard::style_weights[64]` is a vector, not a matrix. SAR requires a matrix to SVD-decompose. Direct application requires reshaping (8×8) or a different formulation. The most direct application is to LoRA overlays and full transformer weights, which ARE matrices.

3. **No RL deltas.** The paper's headline "extraction" use case (extract reasoning core from RL delta) requires trained W_RL. We have no RL training. The modelless residue is spectral projection of freeze/thaw and LoRA deltas — a different, unvalidated application.

4. **The SOPTV tension.** Research 231 found OPD deltas are off-principal. SAR finds RL deltas' reasoning component is on-principal. These study different training regimes (distillation vs RL). For our modelless deltas (freeze/thaw, LoRA), the geometric signature is unknown — could be either. Fusion B (two-component decomposition) hedges this.

5. **Precision sensitivity.** The paper notes SAR is sensitive to numerical precision (FP32 for SVD, FP16 for storage). Our runtime uses mixed precision. The GOAT gate must check precision robustness.

---

## 8. Addendum (2026-07-15) — GOAT Gate Result + Post-406 Fusions

This addendum was added when the paper was re-requested. The original note
(2026-07-10) predated (a) the GOAT gate run and (b) several post-406 research
notes that are direct fusion cousins. Both are captured here.

### 8.1 GOAT Gate Run — G1b CONCENTRATION FAILED at NPC scale

The GOAT gate has run ([`.benchmarks/423_spectral_rewire_goat.md`](../.benchmarks/423_spectral_rewire_goat.md), 2026-07-10). Caveat #1 (scale mismatch) is now **empirically resolved — in the negative direction**:

| Gate | Result |
|---|---|
| G1a numerical stability | **PASS** — on-manifold deltas recovered, fraction >0.999, rel err ~8e-6 at 64×64 / 512×64 |
| G2 singular-direction preservation | **PASS** — cosine = 1.000000 |
| G3 determinism | **PASS** — bit-identical across 100 runs |
| G4 alloc-free | **PASS** — 0 bytes / 1000 calls |
| G5 latency (cached-index) | **PASS** — 8×8 r=4 = 0.41µs (NPC `style_weights[64]`→8×8); 512×64 r=32 = 947µs; 64×64 r=8 = 29µs |
| G6 feature isolation | **PASS** |
| **G1b concentration (the make-or-break)** | **REPORT — concentration does NOT hold at NPC scale** |

**Issue 123 CLOSED (2026-07-10):** riir-train Issue 374 produced trained LoRA
adders via gradient descent across 12 scenarios (d=16/32/64, r=4/8/16, linear +
nonlinear + pretrained). All scenarios produce `on_manifold_fraction` in
**[0.27, 0.58]** — far below the SAR threshold (> 0.8). The linear baseline
shows the SAME fraction as nonlinear scenarios, confirming the elevated
fraction is a LoRA-training artifact (rank-r overlap with r-dim subspace), NOT
the SAR phenomenon. **The SAR concentration requires LLM-scale (4096×4096)
weight matrices; it does NOT transfer to ≤64×64 NPC-scale matrices.**

This is a **§3.6 defend-wrong PoC result**: architectural coverage + latency
parity confirmed, quality claim (concentration at NPC scale) **refuted**.
`spectral_rewire` stays OPT-IN as a cold-tier tool; it is NOT promoted to
default. The primitive is correct machinery in search of an NPC-scale
application it does not have.

**Revised verdict:** Still **GOAT** (the mechanism is novel, correct, fast,
zero-alloc, and ships as a cold-tier tool). NOT Super-GOAT — Q3 (product
selling point for NPC-scale deltas) is now **empirically NEGATIVE**, not merely
"unvalidated." The Super-GOAT promotion path via NPC-scale concentration is
**closed**. Any future Super-GOAT claim must come from a fusion that does NOT
depend on NPC-scale concentration (see §8.3).

### 8.2 Post-406 fusion cousins (the corpus 406 couldn't see)

Four research notes landed after 2026-07-10 that are direct geometric cousins:

| Note | Primitive | Relationship to SAR |
|---|---|---|
| **R408 / Plan 425 (TILR)** | Alignment-gated subspace correction (γ = ‖Πd‖/‖d‖ modulates step size; bit-identical no-harm at γ=0) | **Family sibling** — `katgpt-rs/.docs/05_adaptation/tilr_subspace_family.md` tabulates the 4-member subspace-projection family: `subspace_steering` (ungated, per-axis α) / `spectral_rewire` (ungated, projection) / `tilr` (γ-gated) / `river_valley` (diagnostic). SAR is the decomposition member; TILR is the alignment-gated injection member. |
| **R409 / Plan 426 (MANCE)** | Local k-NN tangent + spectral weighting (σ^α) + per-sample trust region (ε·r_i) | **Local vs global axis.** MANCE estimates a LOCAL per-sample tangent basis from natural neighbors; SAR uses a GLOBAL pre-computed W₀ SVD. `manifold_erasure.rs` tabulates this in its family table. |
| **R367 / Plan 367 (QuasiMoTTo)** | QMC lattice sampler — Pass@k champion (50% sample reduction, K_qmc=8 vs K_iid=16) | **Orthogonal Pass@k mechanism.** SAR widens the reachable set by removing off-manifold drift (deterministic, weight-space); QMC covers a fixed reachable set more efficiently (stochastic, sample-space). Compound effect possible. |
| **R422 (Backdoors)** | Statistically undetectable backdoor in frozen Gaussian first layer; 2-query black-box provenance verification | **Spectral backdoor signature.** A planted backdoor's `z` vector is constructed so `‖Az‖∞` is cryptographically small — a spectral anomaly. SAR's `on_manifold_fraction` + `rewiring_matrix` diagnostics could expose this. |

### 8.3 Fusion E/F/G (NEW — survived the concentration-failure filter)

The original fusions A–D (§2.3) all implicitly assumed NPC-scale
concentration; with G1b refuted, they are weakened. The fusions below are the
post-406 additions that either (i) do NOT depend on NPC-scale concentration,
or (ii) reframe SAR as a diagnostic rather than a purification step.

#### Fusion E: SAR × MANCE — Global-to-Local Manifold Decomposition

Decompose any latent edit into THREE components:
```
Δx = Δx_global_on_principal (SAR, global W₀ SVD)
   + Δx_local_tangent      (MANCE, local k-NN tangent)
   + Δx_residual            (off both manifolds)
```
MANCE's note (R409 §1.2) already identifies the global-vs-local axis as the
family differentiator. The fusion — using BOTH a global pre-computed basis
AND a local per-sample tangent — is documented as a family relationship but
NOT shipped as a combined primitive. **However:** both SAR and MANCE fail the
NPC-scale concentration test in different ways (SAR's concentration is
LLM-scale-only; MANCE's local tangent is validated for concept erasure, not
weight deltas). This fusion is **blocked at NPC scale** — it only makes sense
at LLM scale, which routes to riir-train. **Verdict: documented as family
relationship, not a katgpt-rs plan.**

#### Fusion F: SAR × QuasiMoTTo — Purified-Weights × Low-Discrepancy Sampling

**This is the strongest NEW fusion.** The paper's headline Pass@k improvement
(SAR removes off-manifold drift → wider reachable set, Fig 2) is **orthogonal**
to QuasiMoTTo's mechanism (QMC lattice → more efficient coverage of a fixed
reachable set):

```
Reachable-set widening (SAR, weight-space, deterministic)
  ×
Coverage efficiency (QMC, sample-space, stochastic)
  = Compound Pass@k gain neither alone provides
```

- SAR alone: widens the set of problems reachable by repeated sampling, but
  each sample is i.i.d. (potentially redundant coverage).
- QMC alone: covers a fixed reachable set with 50% fewer samples, but cannot
  reach problems outside the set.
- **SAR + QMC: widen the set AND cover it efficiently.**

Critically, this fusion **does NOT depend on NPC-scale concentration** — it
depends on the LLM-scale path (4096×4096 weight matrices), which is exactly
where Pass@k matters (reasoning models, the riir-ai Reasoning Pack pillar).
The modelless residue (QMC sampling on a SAR-purified weight set) is
inference-time. The SAR purification itself is a one-time cold-tier op on a
trained delta (riir-train produces the delta; katgpt-rs purifies;
riir-ai/riir-train consumes the purified weights with QMC sampling).

**Novelty gate (§1.5):**
- Q1 prior art: NO primitive combines weight-delta purification with
  low-discrepancy sampling. QuasiMoTTo (R367) is sample-space only; SAR is
  weight-space only.
- Q2 new class: YES — "purified-weights × low-discrepancy sampling" is a new
  compound capability (compound Pass@k gain).
- Q3 product selling point: PARTIAL — "our reasoning pack covers more problems
  with fewer samples by combining spectral weight purification with QMC
  sampling." Needs a PoC to validate the compound gain is super-additive.
- Q4 force multiplier: YES — connects Reasoning Pack (P8) + test-time scaling
  (SimpleTES/BoM/CLR family).

**Q3 is the gate.** This is a **fusion idea — novelty TBD, needs a PoC before
Super-GOAT commitment** (per skill §1.5 "no candidate escape hatch"). The PoC
would run on a real reasoning model (riir-train produces a trained delta;
katgpt-rs purifies; measure Pass@k with i.i.d. vs QMC on purified vs unpurified).
File as an issue, not a plan, until the PoC validates super-additive gain.

#### Fusion G: SAR × Statistically-Undetectable Backdoors — Spectral Backdoor Detection

R422 plants a backdoor in a frozen Gaussian first-layer matrix `A` via a secret
`z ∈ {±1}^n` so that `‖Az‖∞` is cryptographically small. SAR decomposes a
weight delta into on-manifold (ΔW*) + off-manifold (ΔW⊥) components relative
to the base SVD.

**Fusion hypothesis:** a backdoored model's delta (backdoored vs honest)
should have **anomalously structured off-manifold energy** — the backdoor's
`z` is constructed to be invisible in `‖Az‖∞` (a uniform-norm test), but SAR's
`rewiring_matrix` M = UᵀΔWV exposes the *directional* structure of the delta
in the base's singular coordinates. A planted `z` aligned with a specific
low-σ singular direction would show up as an anomalous spike in a specific
row/column of M, even when uniform-norm tests miss it.

This is genuinely novel (R422's prior-art audit found zero modelless backdoor
detectors; SAR adds a spectral lens). It is modelless (SAR is deterministic
SVD + matmul). It does NOT depend on NPC-scale concentration — it depends on
the backdoor's spectral signature, which is a different property.

**But:** this is speculative. R422 proves the backdoor is *statistically
undetectable* in the TV-distance sense; whether SAR's rewiring matrix breaks
that undetectability is an open question. The honest verdict is **fusion idea
— novelty TBD, needs a PoC** (can SAR's M distinguish backdoored vs honest
deltas on R422's Fashion-MNIST proof-of-concept setup?). File as an issue
referencing both R406 and R422.

### 8.4 Revised routing (post-addendum)

| Item | Status | Owner |
|---|---|---|
| `spectral_rewire` primitive | SHIPPED, OPT-IN, GOAT gate PASS (mechanism), G1b FAIL (concentration at NPC scale) | katgpt-rs |
| Issue 123 (NPC-scale concentration) | CLOSED (negative result) | riir-train Issue 374 |
| Issue 124 (SVD 64-col cap) | RESOLVED | katgpt-rs |
| Fusion A–D (original, §2.3) | WEAKENED — all assume NPC-scale concentration, now refuted | — |
| Fusion E (SAR × MANCE global-local) | Documented as family relationship; blocked at NPC scale | — |
| **Fusion F (SAR × QuasiMoTTo Pass@k)** | **CLOSED — Issue 151 (negative result).** Phase 1 PoC on DeepSeek-R1-Distill-Qwen-1.5B → DeepScaleR-1.5B-Preview showed on_manifold_fraction ∈ [0.001, 0.191] across 196 layers × 3 ranks — SAR would discard >97% of the RL delta. Concentration precondition refuted at 1.5B scale. | katgpt-rs (commit b7ca596c) |
| **Fusion G (SAR × Backdoors detection)** | **CLOSED — Issue 152 (impractical).** Fatal scope problem: SAR operates on ΔW = W_backdoored − W_honest, presupposing a trusted honest reference. With it, direct weight comparison suffices; without it (the realistic threat model), ΔW is undefined. R422 verdict stands unchallenged. | katgpt-rs (commit 2524918b) |

### 8.5 SAR concentration verdict — evidence ledger (3 scales)

| Scale | Concentration (on_manifold_fraction) | Verdict | Source |
|---|---|---|---|
| NPC (≤64×64, trained LoRA) | [0.27, 0.58] | REFUTED | bench 423 G1b, Issue 123 |
| LLM 1.5B (GRPO RL delta) | [0.001, 0.191], 0/196 layers exceed 0.8 | REFUTED | Issue 151 Phase 1 PoC |
| LLM 7B+ | untested | OPEN | — (needs larger models / GPU resources) |

The `spectral_rewire` primitive stays OPT-IN — its mechanism is correct
(G1a-G6 pass), but the concentration phenomenon doesn't hold at any
accessible scale. Fusions F and G both closed; no further SAR-fusion work
planned unless a 7B+ opportunity arises.

> **Post-406 redirect (2026-10-23, corrected per verdict round 1):** arXiv:2610.22293 "Towards Continual Learning: Mitigating Forgetting With Subspace Tuning" (Research 631) — applied-update complement projection + certification composition law + riir-train Plan 470. ⚠ CUT DISCIPLINE: this paper protects the TOP 99% of the spectrum and adapts in the ~1% tail — the OPPOSITE cut from SAR's top-~1% on-manifold fraction; its unconstrained-delta measurement (98% of update energy in the top-99% span) says nothing about, and does not revive, Fusion A (which keeps the top block and is already marked WEAKENED in §8.4). The two are complementary diagnostics (where capability concentrates vs where updates may write), nothing more. The 7B+ SAR-fusion trigger is NOT fired by it.
