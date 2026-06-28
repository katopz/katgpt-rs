# Research 317: Reasoning as Attractor Dynamics — Gibbs-Weighted Latent Memory Retrieval

> **Source:** [Reasoning as Attractor Dynamics: Latent Memory Retrieval via Gibbs-Weighted Energy Minimization](https://arxiv.org/abs/2606.24543) — Kanishk Awadhiya, *New Frontiers in Associative Memory workshop at ICLR 2026*, arXiv:2606.24543v1, 23 Jun 2026
> **Date:** 2026-06-27
> **Status:** Done — GOAT verdict
> **Classification:** Public
> **Related Research:** 248 (DeltaTok/DeltaWorld BoM — particle cloud), 260 (MaxProof — population test-time scaling), 263 (Latent Thought Flow — reward-proportional trajectory scorer), 269 (Chiaroscuro — spectral entropy KV routing), 276 (MicroRecurrentBeliefState — attractor kernel, honest null), 284 (Simplicity Bias Sampler), 296 (Stokes/DEC vocabulary crosswalk — Fusion C), 305 (Algorithmic-Probability Sampler)
> **Related Plans:** 129 (OPUS Boltzmann), 180 (SDPG — negative result), 269 (Chiaroscuro spectral entropy), 276 (MicroRecurrentBeliefState — AttractorKernel), 281 (BoMSampler), 301 (subspace_phase_gate), 306 (depth_invariance), 333 (CUCG — `can_freeze` spectral_flatness gate)
> **Cross-ref (riir-neuron-db):** Research 317 — §3.5 modelless unblock for Plan 276's "needs trained weights" blocker via NeuronShard `style_weights[64]` → `AttractorKernel::W_s` freeze/thaw (Fusion B, future plan).

---

## TL;DR

Awadhiya (ICLR 2026 AM workshop, 6 pages) reframes LLM inference as thermodynamic relaxation in an energy landscape: correct reasoning chains are **flat minima** (wide attractor basins, low spectral entropy), hallucinations are **sharp minima** (narrow valleys, high local curvature). Sample K trajectories, score each by `E(y) = (1/L)·Σ -log p(t_i | t_<i, x)` (length-normalized NLL = "spectral entropy"), and aggregate per-answer mass using a Gibbs retrieval operator `w_k ∝ 1/E(y_k)² + ε` (inverse-square sharpening). Empirically +5.38% on GSM8K with Phi-3.5-mini (84.7% → 90.1%), +11.6% vs greedy.

**Distilled for katgpt-rs (modelless, inference-time):**
The paper is a clean modelless test-time scaling recipe, but **every load-bearing mechanism already ships in this codebase under different vocabulary** — spectral entropy (`chiaroscuro::spectral_entropy_dct`), K-trajectory sampling (`BoMSampler`), Gibbs/Boltzmann weighting (`opus::boltzmann`, PlackettLuce Gibbs), per-answer mass aggregation (`majority_vote`, `rank_by_consistency`), flat-vs-sharp detection (`spectral_flatness` + CUCG `can_freeze`), phase-transition machinery (`subspace_phase_gate::phase_transition_gate`). The paper's *specific* contribution reduces to: "use `1/E²` as the self-consistency weight instead of uniform." This is a tuning refinement on existing primitives, not a new capability class — **GOAT, not Super-GOAT**.

The genuine value is **fusion**: the paper supplies the physics framing that lets us (a) extend `BoMSampler::select_best` from argmax-on-dot-product to Gibbs-weighted basin aggregation using `spectral_entropy_dct` as the energy, and (b) §3.5-modellessly unblock Plan 276's documented G2.1 failure ("needs trained weights") by loading `NeuronShard::style_weights[64]` as the `AttractorKernel::W_s` — exactly the modelless freeze/thaw path the §3.5 protocol describes.

---

## 1. Paper Core Findings

### 1.1 The energy-landscape framing

LLM as Dense Associative Memory / Energy-Based Model: `P_θ(y|x) = exp(-E_θ(y,x)) / Z(x)`. Standard decoding finds the mode `argmin_y E(y)` — but in a rugged landscape this traps the decoder in sharp local minima = "confident hallucinations."

### 1.2 Flat vs sharp minima

| Minima type | Geometry | Hallucination analog |
|---|---|---|
| **Flat** (wide basin, low curvature) | High entropic volume, convex neighborhood | Robust reasoning chain |
| **Sharp** (narrow valley, high curvature) | Low volume, brittle | Confident hallucination |

Even if a sharp minimum has lower point-energy, the **flat basin has higher integrated probability mass** — a Gibbs measure over the particle cloud picks the flat basin, not the sharp peak.

### 1.3 Trajectory energy = spectral entropy

For trajectory `y = (t_1, ..., t_L)`:

```
E(y) = (1/L) · Σ_i -log P_θ(t_i | t_<i, x)        ... (length-normalized NLL)
```

The paper calls this "spectral entropy" of the trajectory.

### 1.4 The Gibbs retrieval operator

Sample K trajectories `Y = {y^(1), ..., y^(K)}`. Re-weight:

```
w_k ∝ exp(-β · H(y^(k)))      with the paper's empirical choice  β ∝ E^{-2}
   = 1 / (E(y^(k))² + ε)
```

Inverse-square sharpening: "hot" (high-entropy) particles are suppressed, "cold" (low-entropy) particles in attractor basins are amplified. Mimics the contrastive sharpening step in Modern Hopfield Networks.

### 1.5 Basin aggregation (per-answer mass)

```
P(a|x) = Σ_k  1[ϕ(y^(k)) = a] · P_retrieval(y^(k))
â      = argmax_a  P(a|x)         ... the "Dominant Attractor"
```

Standard self-consistency (Wang et al. 2022) is the β=0 limit (uniform vote).

### 1.6 Results

| Strategy | Physics interpretation | GSM8K Acc | Δ |
|---|---|---|---|
| Greedy | Point estimate (β→∞) | 78.4% | — |
| Standard sampling (K=12, majority) | High-temp ensemble (β=0) | 84.69% | +6.3% |
| **Gibbs-weighted (K=12, β∝E⁻²)** | Attractor relaxation | **90.07%** | **+11.6%** |

Phase transition: as K grows, probability mass of "sharp minima" (hallucinations) evaporates while "flat minima" retain mass. The system relaxes to equilibrium.

### 1.7 Paper's stated future work

> "Latent Space Navigation: Instead of sampling discrete tokens, future work could perform gradient descent on the energy function ∇_h E(h) with respect to latent states h, strictly enforcing the 'Flat Minima' constraint."

> "Iterative Attractor Refinement: We could use the 'Weighted Consensus' of the current particle cloud to prompt the model for a new set of particles, creating a Recurrent Cognitive Cycle."

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper → codebase — both layers hit)

| Paper term | Codebase equivalent (shipped) | Location |
|---|---|---|
| Trajectory energy / spectral entropy | `spectral_entropy_dct`, `spectral_entropy_dct_into` (Type-II DCT, per-token, [0,1], zero-alloc) | `katgpt-rs/src/chiaroscuro/entropy.rs` (Plan 269) |
| K-trajectory particle cloud | `BoMSampler::sample_k_states` (K Gaussian noise queries, single batched matvec) | `katgpt-rs/crates/katgpt-core/src/micro_belief/bom.rs` (Plan 281) |
| Gibbs weight `exp(-β·H)` | `boltzmann_probabilities`, `boltzmann_sample_batch` (τ-controlled softmax); PlackettLuce Gibbs sampling (I=1000, B=200) | `src/pruners/opus/boltzmann.rs` (Plan 129, B039) |
| Inverse-energy trajectory scorer | `exp(-λ_c · T)` cost-penalty scorer | `benches/latent_thought_flow_scorer_bench.rs` (R263) |
| Per-answer basin aggregation | `majority_vote`, `rank_by_consistency`, `select_best_variant`, `ConvergenceSelector::MajorityVote` | `src/speculative/parallel_probe.rs`, `src/speculative/ppot/rank.rs`, `types.rs` |
| Flat vs sharp minima | `spectral_flatness` (used by CUCG `can_freeze` gate: `< 0.3` ⇒ freeze); `nds_proxy = 1 - flatness`; `classify_chain` flatness pass | `riir-neuron-db/src/spectral_flatness.rs`, `riir-neuron-db/src/phase_gate.rs` (Plan 333 CUCG), `src/pruners/nds_proxy.rs` (Plan 186), Plan 306 |
| Phase transition (mass evaporates at critical β) | `phase_transition_gate(N, d)`, participation_ratio, numerical_rank, Jacobian SVD | `crates/katgpt-core/src/subspace_phase_gate/` (Plan 301) |
| Hopfield MHN update / attractor | `AttractorKernel` (Hopfield-style recurrent sigmoid update) + **honest G2.1 null result** | `crates/katgpt-core/src/micro_belief/` (Plan 276, B276) |
| Relaxation to equilibrium | `MicroRecurrentBeliefState::step` iterated to fixed point | same |
| Robustness encoded in basin geometry, not depth | CUCG isomorphism: `can_freeze = (n_wake_events ≥ intrinsic_dim) ∧ (spectral_flatness < 0.3)` | `riir-neuron-db/src/phase_gate.rs` (Plan 333 G7) |

### 2.2 What the paper adds beyond our shipped stack

| Paper claim | Shipped? | Gap |
|---|---|---|
| Trajectory energy = length-normalized NLL | ✅ Mechanism ships as `spectral_entropy_dct` (different unit: DCT-of-embeddings vs token-NLL) | Token-NLL variant not shipped as a named fn — but `ac_prefix::conditional_logprob` (Plan 313) computes exactly this |
| K trajectories + per-answer vote | ✅ `BoMSampler` + `majority_vote` | — |
| General Gibbs `exp(-β·H)` weighting | ✅ `opus::boltzmann` (any τ schedule) | — |
| **Specific `β ∝ E⁻²` law** | ⚠️ Not shipped as a named function | The novel (narrow) contribution — a 5-line function |
| Hopfield-as-retrieval-operator framing | ✅ `AttractorKernel` + Plan 276 documents the **exact caveat the paper glosses over** | See §2.4 |
| Latent gradient descent on energy | ❌ Paper's own Future Work | Genuinely open |

### 2.3 The actual novel primitive (5 lines)

```rust
/// Paper eq. (4): inverse-square Gibbs sharpening.
/// `energies[k]` = trajectory spectral entropy / length-normalized NLL.
/// Writes unnormalized Gibbs weights `w_k ∝ 1 / (E_k² + ε)` into `out`.
pub fn gibbs_inverse_square_weights(energies: &[f32], out: &mut [f32]) {
    debug_assert_eq!(energies.len(), out.len());
    const EPS: f32 = 1e-8;
    for (e, w) in energies.iter().zip(out.iter_mut()) {
        *w = 1.0 / (*e * *e + EPS);
    }
}
```

That is the entire paper-specific code. Everything else is composition of existing primitives.

### 2.4 The honest caveat the paper glosses over — Plan 276 already proved it

Plan 276's `AttractorKernel` is a Hopfield-style attractor network. Its G2.1 GOAT gate failed: at random Xavier init, the attractor **flip-flops 569× more than a leaky integrator** because "the attractor's hysteresis property is real but it is a property of TRAINED attractor networks (Hopfield-style content-addressable memory), not of randomly-initialised ones" (`katgpt-rs/.benchmarks/276_micro_belief_goat.md` L93-94).

The paper's "flat minima ⇒ robust attractor" claim **assumes trained weights** that create actual basins. In the modelless / freeze-thaw-only regime that the katgpt-rs AGENTS.md mandate enforces, **random-init attractors do not have flat basins** — they are noisy dynamical systems. So the paper's headline mechanism, applied modellessly to a randomly-initialized attractor, reproduces Plan 276's null result.

This is not a defect of the paper — it is a clarification of where the mechanism is valid. Two modelless paths rescue it:

1. **Frozen-shard path (Fusion B, §3.5):** load `NeuronShard::style_weights[64]` (committed, BLAKE3-checked, trained-offline) as the `AttractorKernel::W_s`. Now the attractor has trained basins. This is exactly the §3.5 "freeze/thaw snapshot correction" path — the bias (random init ⇒ no basins) is systematic and characterizable; the correction (load trained W_s) is a modelless freeze/thaw, not gradient descent.
2. **Vocabulary-translation path (Fusion C):** "flat minimum" = "harmonic component of belief cochain" under DEC. `harmonic_projector(belief)` extracts the wide-basin component deterministically — no K-particle cloud, no trained attractor needed.

### 2.5 Fusion (the Super-GOAT-tier move — but not novel to this paper alone)

**Fusion A — `BoMSampler::select_best` extension (GOAT, the direct ship).**
Currently `BoMSampler::select_best(hypotheses, scorer, k)` takes a caller scorer (default `dot_product_scorer`) and returns the argmax index. **Replace the scorer with a Gibbs-weighted basin aggregator** that uses `spectral_entropy_dct` as the energy:

```rust
/// Fusion A: Gibbs-weighted basin aggregator over K BoM hypotheses.
/// Returns the dominant attractor index (paper §3.3 basin aggregation).
pub fn gibbs_basin_aggregator(
    hypotheses: &[f32],   // [K * D]
    k: usize,
    energy: impl Fn(&[f32]) -> f32,  // = spectral_entropy_dct_into
    scratch_weights: &mut [f32],     // [K], caller-allocated
) -> usize {
    // w_i = 1 / E(h_i)^2; argmax_i w_i = argmin_i E(h_i)
    // (degenerates to plain argmin of energy — the paper's K=12 result
    // comes from per-answer mass aggregation, which needs ϕ: h -> answer
    // and is the caller's responsibility, not the kernel's.)
    gibbs_inverse_square_weights_per_hypothesis(hypotheses, k, &energy, scratch_weights);
    argmax(scratch_weights)
}
```

This is a 30-line extension to `micro_belief/bom.rs` behind a `gibbs_basin_aggregator` feature, NOT a new module. Goes in **katgpt-rs** (public, generic).

**Fusion B — `NeuronShard::style_weights → AttractorKernel::W_s` (modelless §3.5 unblock of Plan 276).**

Plan 276 benchmarked `AttractorKernel::from_seed` (random Xavier W_s) and got the G2.1 null result. The benchmark explicitly flagged: "the recurrent weights would need to be trained (or hand-set) so that the target beliefs correspond to actual stable fixed points of the dynamics. That training is out of scope for Plan 276 (which is training-free / freeze-thaw only)."

**The §3.5 fix:** "training-free / freeze-thaw only" does NOT mean "random init only." It means: the weights must arrive via freeze/thaw, not via runtime gradient descent. `NeuronShard::style_weights[64]` is exactly a frozen, BLAKE3-committed, trained-offline latent state that arrives via the freeze/thaw pipeline. Loading `style_weights` as the `AttractorKernel::W_s` is a modelless path-1 unblock (§3.5):

- **Is the failure systematic and characterizable?** YES — "random Xavier W_s creates no useful basins" (Plan 276 L87-91).
- **Can freeze/thaw (path 1) fix it?** YES — load a committed, trained `style_weights` as `W_s`.
- **Does it require gradient descent?** NO — the gradient descent happened offline when the shard was consolidated (Raven/δ-Mem); at runtime we just swap the frozen matrix in.

This converts Plan 276 from "opt-in experiment, G2.1 FAIL" → "candidate for re-benchmarking under shard-loaded W_s." That is a **riir-neuron-db × katgpt-rs** fusion — the shard source lives in `riir-neuron-db/src/shard.rs`, the attractor lives in `katgpt-rs/crates/katgpt-core/src/micro_belief/`. Private IP is the "shard-trained NPC personality attractor" selling point; open primitive is the `W_s` loading trait.

**Fusion C — DEC `harmonic_projector` as deterministic basin-volume estimator (Super-GOAT-tier cross-reference, not this paper's contribution).**

Under the Generalized Stokes' theorem substrate (Plan 251, Research 296), a "flat minimum" of an energy field = a locally-constant cochain = a cochain with zero exterior derivative `dω = 0` = the **harmonic component** of the Hodge decomposition. So `harmonic_projector(belief_cochain)` extracts the wide-basin component deterministically — no K-particle cloud, no trained attractor, no `1/E²` weight. The harmonic mass **is** the basin volume.

```
basin_volume(belief) ≈ ‖harmonic_projector(belief)‖²
gibbs_weight(belief) ≈ 1 / E(belief)²
                 ≈ 1 / (1 - normalized_harmonic_mass(belief))²    [under Fusion C ident]
```

This fuses **this paper × Research 296 (Stokes/DEC) × Plan 333 (CUCG spectral_flatness freeze gate)** into a deterministic, geometry-grounded "is this belief in a robust basin?" detector that subsumes the K-particle Gibbs cloud as the stochastic special case. **This is the Super-GOAT-tier reframing**, but it is a fusion across three prior research notes — not a property of this paper alone. Flagging it as the cross-reference for a future Super-GOAT pass; this note's verdict stays GOAT.

### 2.6 Latent-space reframing (mandatory per workflow step 3)

Re-cast the Gibbs retrieval operator as a latent-to-latent op on each Super-GOAT factory module:

| Substrate | Reframing | Status |
|---|---|---|
| **HLA** (`sense/`) | Per-NPC 8-dim HLA state = particle in attractor landscape. K-trajectory BoM cloud over HLA = particle cloud. Gibbs weight `1/E²` where `E = spectral_entropy_dct(hla_state)`. Dominant attractor = argmin-energy hypothesis. | Direct ship via Fusion A — `BoMSampler` already operates on HLA. |
| **`latent_functor/`** (riir-ai) | Single Hopfield retrieval step = one functor application. Coherence-driven re-estimation (`reestimation.rs`) = "when the dominant attractor's energy drifts, re-estimate functors." This is the paper's "Recurrent Cognitive Cycle" future work, already shipped under different vocab. | Already ships — Research 123 / Plan 303 framing. |
| **`cgsp_runtime/`** | Curiosity = high-entropy particle (exploration); relaxation to attractor = exploitation. Gibbs weighting becomes "how to collapse K curiosity threads into one decision per tick." | Novel application; riir-ai game runtime. |
| **LatCal** (`riir-chain/src/encoding/`) | The Gibbs weight `1/E²` is a deterministic scalar that crosses `SyncBlock → ChainConsensus` as a raw value. The K hypotheses stay local-latent; only the chosen index + its weight sync. | Sync boundary respected. |
| **`NeuronShard`** (riir-neuron-db) | `style_weights[64]` = stored attractor pattern. `MerkleFrozenEnvelope` = committed flat minimum. `can_freeze = (spectral_flatness < 0.3)` (Plan 333) IS the paper's "flat minimum ⇒ robust" claim, restated as a freeze gate. CUCG G7 isomorphism: trajectory compaction ≡ shard freeze. | Already shipped — paper is confirmatory. |
| **DEC** (`katgpt-rs/crates/katgpt-core/src/dec/`) | Flat minimum = harmonic component. `harmonic_projector(belief)` = deterministic basin detector. Stokes `∫_M dω = ∫_∂M ω` ⇒ basin mass via boundary flux (Plan 314). | Fusion C; future Super-GOAT cross-ref. |

---

## 3. Verdict

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars) | — |
| **GOAT** | Provable gain over existing approach, but not a new class | **← this paper** |
| Gain | Incremental improvement | — |
| Pass | Not relevant, OR training-only | — |

**One-line reasoning:** The paper is a clean modelless test-time scaling recipe (+5.38% GSM8K), but vocabulary-translation grep across notes + code in all 5 repos shows every load-bearing mechanism already ships — spectral entropy (`chiaroscuro`), K-trajectory sampling (`BoMSampler`), Gibbs/Boltzmann weighting (`opus::boltzmann`), per-answer aggregation (`majority_vote`), flat-vs-sharp detection (`spectral_flatness` + CUCG `can_freeze`), phase-transition (`subspace_phase_gate`). The paper-specific contribution is the `β ∝ E⁻²` law — a 5-line function on top of `opus::boltzmann`. **Not Super-GOAT** because Q1 (no prior art) FAILS, Q2 (new capability class) FAILS, Q3 (selling point) PARTIAL — Plan 276's G2.1 honest null result directly undercuts the "untrained attractor ⇒ robust basin" claim in the modelless regime, narrowing the selling point to "shard-trained attractors relax to robust basins" (Fusion B), which is a riir-neuron-db × katgpt-rs fusion selling point, not this paper's alone. Q4 (force multiplier) PASSES — connects to BoMSampler, ParallelProbe, OPUS Boltzmann, Chiaroscuro, CUCG, subspace_phase_gate, Latent Thought Flow scorer, MaxProof.

**Routing:**
- **katgpt-rs (public):** Plan (TBD) — `gibbs_inverse_square_weights` + `gibbs_basin_aggregator` extension to `BoMSampler::select_best`. Generic primitive, no game/chain/shard semantics. Behind `gibbs_basin_aggregator` feature flag. GOAT gate: G1 the `1/E²` weight preserves ranking vs argmax-on-energy (trivially true: argmax(1/E²) = argmin(E)), G2 vs uniform-vote self-consistency on a synthetic reasoning trace benchmark (target: matches paper's +5pp on a 2-class answer-distribution), G3 zero-alloc, G4 feature-isolated.
- **riir-neuron-db (private):** Fusion B — the "shard `style_weights` as attractor `W_s`" modelless unblock of Plan 276. Cross-ref to a future plan; this is a §3.5 win, not new code in this note.
- **riir-ai (private game runtime):** cgsp_runtime / NPC curiosity collapse via Gibbs weighting (Fusion A applied to per-tick K-belief BoM cloud). Game IP; deferred to riir-ai plan if katgpt-rs GOAT gate passes.
- **riir-chain / riir-train:** nothing — fully modelless.

**No private Super-GOAT guide created** — GOAT verdict, mandatory-guide rule not triggered.

---

## 4. Why not Super-GOAT (honest accounting)

The paper's headline — "inference is relaxation into an attractor basin, not greedy token prediction" — sounds like a new capability class. It is not, for three concrete reasons grounded in this codebase:

1. **The capability (K-trajectory Gibbs-weighted answer aggregation) already ships** as the composition `BoMSampler + opus::boltzmann + majority_vote`. The paper adds the `E⁻²` law, not the capability.
2. **The "flat minima" detector already ships** as CUCG `can_freeze = (spectral_flatness < 0.3)` (Plan 333), proven equivalent across trajectory-compaction and shard-freeze domains (G7 isomorphism). The paper restates this in physics vocabulary.
3. **The Hopfield/attractor substrate already shipped AND failed its GOAT gate** (Plan 276 G2.1) for exactly the reason the paper glosses over — random-init attractors don't have useful basins. Plan 276's benchmark is the more honest treatment of the same idea in the modelless regime.

The genuine Super-GOAT-tier move — Fusion C, deterministic basin-volume via DEC `harmonic_projector` — is a **cross-paper fusion** (this paper × Research 296 × Plan 333), not a property of this paper alone. It is flagged in §2.5 as the cross-reference for a future Super-GOAT pass.

---

## 5. Connection map (force multiplier, Q4)

```
                  ┌─ BoMSampler (Plan 281) ──── Fusion A: select_best extension
                  │
  Paper 2606.24543┼─ opus::boltzmann (Plan 129) ─── Gibbs core, τ=E⁻²
  (this note)     │
                  ├─ chiaroscuro::spectral_entropy (Plan 269) ─── E(y) definition
                  │
                  ├─ CUCG spectral_flatness (Plan 333) ─── flat-vs-sharp ≡ can_freeze
                  │
                  ├─ subspace_phase_gate (Plan 301) ─── phase-transition machinery
                  │
                  ├─ Plan 276 AttractorKernel ──── Fusion B: §3.5 unblock via
                  │                                NeuronShard style_weights → W_s
                  │
                  ├─ MaxProof (R260) ──── population-search cousin (tournament
                  │                     vs Gibbs aggregation — two ways to pick
                  │                     the dominant attractor)
                  │
                  ├─ Latent Thought Flow (R263) ──── exp(-λ·T) trajectory scorer cousin
                  │
                  └─ DEC harmonic_projector (R296, Plan 251) ──── Fusion C:
                                                      deterministic basin-volume
                                                      (Super-GOAT-tier cross-ref)
```

---

## 6. Latent vs raw boundary (per AGENTS.md)

- **Local-latent (never synced):** the K BoM hypotheses, their per-hypothesis spectral entropies, the per-hypothesis Gibbs weights, the per-answer mass distribution. All consumed inside the planner's scratch.
- **Synced (raw scalars):** only the chosen dominant-attractor index + its scalar projection (the 5 HLA affect scalars: valence/arousal/desperation/calm/fear). The K-vector distribution NEVER crosses `SyncBlock → ChainConsensus`.
- **Bridge:** `project_to_scalars` (Plan 276) is the existing bridge; Fusion A reuses it unchanged.
- **LatCal crossing:** the Gibbs weight itself is a deterministic scalar — it can be committed via LatCal fixed-point if chain-side attestation of "this NPC's attractor relaxation was honest" is needed. Not in scope for the GOAT plan.

---

## 7. Open questions / risks

1. **`E⁻²` vs other β schedules.** The paper picks `E⁻²` empirically with no theoretical justification beyond "data-dependent temperature schedule." Our `opus::boltzmann` already supports arbitrary τ schedules — the GOAT plan should benchmark `E⁻¹`, `E⁻²`, `exp(-E)`, `softmax(-E)` against uniform vote and pick the winner by G2. May demote `E⁻²` to one option among several.
2. **Fusion B's "trained W_s" is offline-trained.** This is consistent with the modelless mandate (no runtime gradient descent), but it does mean the `style_weights` had to come from somewhere — that somewhere is `riir-train` or the Raven/δ-Mem consolidation pipeline. The §3.5 protocol is satisfied because the **runtime** path is pure freeze/thaw; the offline training is explicitly out of this workflow's scope.
3. **Plan 276 G2.1 re-benchmark.** Fusion B is a claim, not a proof. The re-benchmark (AttractorKernel with shard-loaded W_s vs leaky integrator on the same coherence suite) is a separate plan; if it still fails, Fusion B is wrong and the attractor family stays demoted. Honest null result is the floor.
4. **Per-answer aggregation needs an answer-extractor ϕ.** The paper's `ϕ: y → a` maps a trajectory to a final answer. For HLA belief states there is no natural "answer" — the analog is the projected scalar action. This is a riir-ai game-runtime concern, not a katgpt-rs primitive concern; Fusion A ships only the energy-based scorer, not the ϕ.

---

## TL;DR

Awadhiya 2606.24543 (ICLR 2026 AM workshop) reframes LLM inference as thermodynamic relaxation: correct chains are flat minima (low spectral entropy, wide attractor basins), hallucinations are sharp minima. Sample K trajectories, weight each by `1/E²` where `E = length-normalized NLL`, aggregate per-answer mass. +5.38% GSM8K on Phi-3.5. **Verdict: GOAT, not Super-GOAT.** Vocabulary-translation grep across all 5 repos (notes + code) shows every load-bearing mechanism already ships under different names: spectral entropy = `chiaroscuro::spectral_entropy_dct`; K-particle cloud = `BoMSampler`; Gibbs weighting = `opus::boltzmann`; per-answer aggregation = `majority_vote` / `rank_by_consistency`; flat-vs-sharp = `spectral_flatness` + CUCG `can_freeze`; phase transition = `subspace_phase_gate::phase_transition_gate`. The paper-specific contribution (`β ∝ E⁻²`) is a 5-line function on top of `opus::boltzmann`. Plan 276's honest G2.1 null result undercuts the "untrained attractor ⇒ robust basin" selling point — random-init attractors flip-flop 569× more than leaky integrators, exactly because basins need trained weights. **Three fusions, not direct mapping:** (A) `BoMSampler::select_best` extension to Gibbs-weighted basin aggregation — direct ship in katgpt-rs; (B) §3.5 modelless unblock of Plan 276 via `NeuronShard::style_weights[64]` → `AttractorKernel::W_s` (freeze/thaw path, no riir-train); (C) DEC `harmonic_projector` as deterministic basin-volume estimator (Super-GOAT-tier cross-ref to a future Research 296 × 317 × 333 fusion pass). The paper's "Future Work" — latent gradient descent on energy, recurrent cognitive cycle — is already shipped under codebase vocabulary (`latent_functor/reestimation.rs` coherence-driven re-estimation = the recurrent cognitive cycle; Fusion C harmonic projection = the latent-space flat-minima navigation). **No Super-GOAT guide created** — GOAT verdict, mandatory-guide rule not triggered. **5-repo routing:** katgpt-rs (gibbs_basin_aggregator open primitive, GOAT plan TBD); riir-neuron-db (Fusion B cross-ref for Plan 276 re-benchmark); riir-ai (game-runtime application, deferred); riir-chain / riir-train (nothing).
