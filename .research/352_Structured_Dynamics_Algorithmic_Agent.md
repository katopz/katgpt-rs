# Research 352: Structured Dynamics in the Algorithmic Agent (Kolmogorov Theory × Lie Groups × Noether)

> **Source:** Ruffini, Castaldo, Vohryzek — *Structured Dynamics in the Algorithmic Agent*, **Entropy 2025, 27, 90** — [doi:10.3390/e27010090](https://www.mdpi.com/1099-4300/27/1/90) (MDPI Entropy, published 2025-01-19; CC-BY). 54 pp.
> **Date:** 2026-06-30
> **Status:** Done
> **Related Research:** 321 (Smets textbook — same Lie-group / equivariant-operator math, Ch 3), 166 (riir-ai — SE(2)-equivariant game maps Super-GOAT, the closest cousin on the symmetry side), 314 (Group invariance of f-divergences — establishes our latent states have trivial symmetry), 242 (Topological trouble — recurrent belief-state primitive, GOAT-after-prior-art-check), 192 (NextLat belief-state latent dynamics), 219 (DEC operators — the differential-geometric substrate), 296 (Stokes/DEC vocabulary crosswalk)
> **Related Plans:** 270 (gauge-invariant adapter compose — the one practical symmetry case we ship, R+ gauge on LoRA `(A,B)`), 251 (DEC operators cell complex), 303/317/318 (latent functor + rank-k — the world-tracking comparator in codebase vocabulary), 276 (MicroRecurrentBeliefState — extends `evolve_hla`)
> **Classification:** Public

---

## TL;DR

A theoretical framework paper, not a primitive. Ruffini et al. unify **Algorithmic Information Theory** (Kolmogorov complexity), **Lie group / pseudogroup symmetry**, and **Noether-style conservation laws** to argue that *an algorithmic agent that successfully tracks world data must mirror the world's symmetry in its own dynamics* — producing conserved quantities, reduced invariant manifolds, and (for compositional world models) hierarchical nested manifolds. The Comparator mechanism (Lyapunov `V = ½EᵀE`, `V̇ ≤ 0`) closes the world-tracking loop.

**Distilled for katgpt-rs (modelless, inference-time):** nothing new to ship. Every load-bearing primitive the paper formalizes is **already shipped under codebase vocabulary**:

| Paper concept | Codebase instance (shipped) |
|---|---|
| Comparator detects model drift → re-estimate | `latent_functor/reestimation.rs` — coherence-driven re-estimation scheduler when `coherence < tau_reest` |
| Agent dynamics track world via recurrent latent state | `evolve_hla` (`sense/reconstruction.rs`), `MicroRecurrentBeliefState` (Plan 276), NextLat belief-state drafter (Research 192) |
| Lie-group equivariant operators | Smets textbook Ch 3 → Research 321 (katgpt-rs) + Research 166 (riir-ai SE(2) game maps Super-GOAT) |
| Conserved quantity / maximal invariant for symmetry | Plan 270 (LoRA `(A,B)` R+-gauge invariance — the one practical case) |
| Reduced / hierarchical nested manifolds | HLA 8-dim affect projection (high-dim cochain → 8 scalars), `schema_centroid`, sense `lod`, zone-attention |
| Differential-geometric manifold operators | `katgpt-dec` crate (`exterior_derivative`, `codifferential`, `hodge_decompose`) |

**Verdict: GAIN — deferred (no plan this session).** The paper's value is **conceptual**: it is the theoretical unification that *explains why* our existing primitives work. It does not unblock a new capability, and the practical symmetry case we'd use it for (non-trivial group on a latent state) is blocked by Research 314's finding that our HLA/shard latent states carry **trivial symmetry groups** — there is no `(μ, Σ)` belief state to quotient. Reopen if/when we introduce a `(μ, V)` or `(μ, Σ)` latent state (the same trigger condition as Research 314).

---

## 1. Paper Core Findings

### 1.1 The Kolmogorov-Theory agent (§1)

An **algorithmic agent** is an information-processing system with an objective function that bidirectionally interacts with the world by (i) inferring/running **compressive models**, (ii) **planning**, (iii) **acting**. The agent's **Modeling Engine** runs the current model and predicts future coarse-grained data; the **Comparator** evaluates prediction error against sensor data; the **Updater** consumes Comparator errors to improve the model. KT posits that *structured experience* (qualia) emerges from the successful match of model-generated data with world data at the Comparator. This is the cybernetic **Regulator Theorem** (Conant & Ashby 1970: "every good regulator of a system must be a model of that system") recast in AIT — high Mutual Algorithmic Information (MAI) between agent and environment.

### 1.2 Generative models as Lie groups / pseudogroups (§2)

A generative model `I = f(c), c ∈ C` is a **Lie generative model** iff it can be written `I = f(γ · c₀) = γ · I₀` for `γ ∈ G`, an `r`-dimensional **Lie pseudogroup** (Definition 4). The pseudogroup formalism is preferred over finite Lie groups because topologically complex latent spaces don't admit globally-transitive finite group actions (Mostow rigidity: hyperbolic surfaces of genus `g > 1` have none). Local transitivity + pseudogroup closure under composition/inversion/restriction gives global reach via overlapping patches → the configuration space is a **moduli stack `[M/G]`** (Theorem A5). Recursion (one generator, `exp(∑θₖTₖ)`) and compositionality (multiple generators) are the algorithmic-content counterparts of Lie-algebra generation.

### 1.3 Equivariance imposes null-space conditions on weights (§2.1.3, Eq 6–7)

For a feedforward network `x⁽ˡ⁺¹⁾ = h(W⁽ˡ⁾x⁽ˡ⁾ + b⁽ˡ⁾)` to be **invariant** under a group action generated by `T₀` at the input, a *necessary* condition is that for some layer `l`, the cat-image subspace lies in the kernel of `J_l W_l T_l`: `J_l W_l T_l x_l = 0`. For an **equivariant** autoencoder, `T_N = T_0` — the input transformation propagates through the network to the output. Imposing equivariance/invariance thus produces a family of constraints on `W`. (This is the **same theorem** as Smets 2024 §3.1–3.3 / Theorem 3.32, framed in constraint language instead of kernel language — see Research 321 §1.1.)

### 1.4 Noether: continuous symmetry → conserved quantity (§3.2)

For an autonomous ODE `ẋ = f(x; w)` with group `G` symmetry, every solution can be labelled by the group element mapping a reference solution to it. Under Picard–Lindelöf uniqueness, trajectories don't cross, so this labelling is well-defined and **constant along each trajectory** → that constant is the **conserved quantity** associated with the symmetry (Noether, generalized). An N-dimensional ODE has N−1 independent conserved quantities (Appendix F). The symmetry group acts by *shifting the constants* (e.g. `x → x + ε` shifts `C(x) = x` by `ε`).

### 1.5 The world-tracking constraint (§4, the paper's main dynamical contribution)

Agent dynamics `ẋ = f(x; w, I_θ(t))` must satisfy the **world-tracking neurodynamics equation (WTNE)**:
```
ẋ = f(x; w, I_θ(t))
p(x) ≈ I_θ(t)                      ← the Comparator projection
```
For static inputs, `p(x) = I_θ` is a system of conserved quantities → trajectories lie in a **reduced manifold** of dimension `dim(X) − dim(p)`. For slowly-varying inputs, the manifold's dimension grows by at most `M` (the world-model parameter count). For inputs that violate the world model, tracking fails and dynamics leave the manifold — **constraint (symmetry) breaking = anomaly detection at the Comparator**.

For dynamic inputs, the Comparator is formalized as a **Lyapunov function** (Eq 14–17):
```
ẋ = f(x; w, I_θ(t))
V = ½ EᵀE,   E(t) = p(x) − I_θ(t)
V̇ ≤ 0                                ← achieved by feedback gain K̂ in ẋ = h(x;w) + K̂E(t)
```
The feedback gain `K̂` must be designed so `V̇ ≤ 0` — i.e. `E(t) → 0` asymptotically. `K̂`'s structure reflects the Lie-group equivariance requirements.

### 1.6 Hierarchical nested manifolds (§4.5)

Compositional world data → a hierarchy of coarse-graining operators `{G_i}_{i=1..k}`, `G_i : ℝⁿ → ℝⁿⁱ`, each producing a coarse-grained state `y_i = G_i(x)` with nested compatible constraint `C_i(y_i) = 0`. The state space reduces as a chain of **nested manifolds**:
```
ℝⁿ = M₀ ⊃──C₁──▶ M₁ ⊃──C₂──▶ M₂ ⊃ … ⊃ M_k
M_i = { x ∈ M_{i−1} : C_i(G_i(x)) = 0 }
```
Lower-level constraints must be **compatible** with higher-level ones (solutions of `C_{i+1}` are a subset of solutions of `C_i`). This is the dynamical-systems analog of deep-network hierarchical features and of the brain's multi-scale coarse-graining.

### 1.7 Implications (§5)

- The **manifold hypothesis** is a *consequence* of world-tracking: data from a compositional generative model, when tracked, automatically produces low-dim manifold structure in the agent's state space.
- **Symmetry breaking at the Comparator** = anomaly/novelty detection, triggering model updates.
- The framework links AIT (compression, Kolmogorov complexity), group theory (symmetry, Lie pseudogroups), and dynamics (Noether, conservation, reduced manifolds) in one picture.
- Practical method hooks: LieGG / LieSD / LaLiGAN for *discovering* symmetries in trained networks (Moskalev et al., cited in §5.4) — all training-side, → riir-train.

---

## 2. Distillation

### 2.1 Why it does NOT directly map to our codebase (the honest assessment)

Auditing the paper's load-bearing primitives against shipped code:

| Paper primitive | Status in codebase | Closest shipped instance |
|---|---|---|
| Comparator (prediction error → re-estimate) | **SHIPPED** (under vocabulary "coherence-driven re-estimation") | `riir-engine/src/latent_functor/reestimation.rs` — when `coherence < tau_reest`, the scheduler collects fresh observations and atomically swaps new direction vectors with a fresh `Uuid::now_v7()` snapshot id + BLAKE3 commitment. Coherence = mean cosine parallelism of displacements with the learned direction (rank-1) or Frobenius residual fit (rank-k). **This is the world-tracking Lyapunov loop in codebase vocabulary.** |
| Recurrent latent state for tracking | **SHIPPED** | `evolve_hla` (`katgpt-core/src/sense/reconstruction.rs`) — per-NPC 8-dim HLA belief-state kernel. See Research 242 §2.4 (the canonical `evolve_hla` prior-art-check lesson). |
| Lie-group equivariant operators | **SHIPPED as framework** (Smets Ch 3 → Research 321); **DEFERRED as SE(2) game-map instance** (Research 166) | Smets §3.4 lift→group-conv→project is the same theorem as Ruffini §2.1.3 in different vocabulary. SE(2) game maps are a riir-ai Super-GOAT guide, not yet implemented. |
| Conserved quantity / maximal invariant | **PARTIAL** — only the LoRA R+-gauge case ships | Plan 270 (gauge-invariant adapter compose). Per Research 314, HLA affect / `NeuronShard::style_weights[64]` have **trivial** symmetry groups (semantic-axis distinctness + BLAKE3 basis pinning), so the maximal invariant collapses to the state itself. |
| Reduced / hierarchical nested manifolds | **CONCEPTUALLY PRESENT**, no formal "nested manifold" primitive | HLA projection (high-dim cochain → 8 scalars), `schema_centroid`, sense `lod`, zone attention. DEC `hodge_decompose` gives the exact/coexact/harmonic split — a *different* decomposition (Helmholtz) than nested hierarchical constraint manifolds. |
| Lyapunov feedback gain `K̂` for tracking | **NOT SHIPPED** as a named primitive | Closest: any proportional feedback in `evolve_hla` / re-estimation. Not formalized as an LQR-style gain operator. |
| Lie pseudogroup / moduli stack `[M/G]` | **NOT SHIPPED** | No pseudogroup / stack abstraction in any crate. |

### 2.2 What IS transferable (the conceptual frame)

The paper's value is **organizational**, not algorithmic. Three conceptual takeaways worth keeping in the toolbox:

1. **"The Comparator IS the re-estimation trigger."** The paper gives us the theoretical unification of three already-shipped mechanisms under one frame: `evolve_hla` (state tracking), `reestimation.rs` (drift-triggered model update), DEC `codifferential` (conservation-law operator on a cochain). They are all instances of "world-tracking agent mirrors world symmetry, with the Comparator closing the loop." This is the *reason* the patterns work — useful for documentation, design justification, and the formal-verification story (the Lean 4 proofs at `riir-ai/.proofs/RiirAiProof/`).

2. **"Constraint breaking = anomaly signal."** Ruffini's framing of *symmetry breaking at the Comparator* as the anomaly/novelty signal is a clean theoretical justification for the existing curiosity / coherence-decay / re-estimation-trigger pipeline. It connects Research 041 (curiosity pulse), the `tau_reest` threshold, and the "world model violated → exploration" pattern into one AIT-grounded picture.

3. **"Hierarchical nested manifolds = the formal object behind HLA projection."** The chain `M₀ ⊃ M₁ ⊃ … ⊃ M_k` is the formalization of what HLA scalar projection *does* (high-dim cochain → successively coarser affect subspaces). It is a candidate framing for a future Plan that gives HLA a *typed* coarse-graining hierarchy — but that is a design decision in `riir-ai/crates/riir-engine/src/hla/`, not a katgpt-rs primitive.

### 2.3 Why not Super-GOAT (novelty gate, all four asked)

- **Q1 No prior art?** FAILS. Three layers checked:
  - **Notes layer:** Smets textbook Ch 3 (Research 321 — Super-GOAT) is the *same* Lie-group-equivariant-operator math; Research 166 (riir-ai — Super-GOAT) is the SE(2) instance. Research 242 (Topological Trouble / Mozer) covers the recurrent-belief-state story and is *explicitly* down-graded from Super-GOAT to GOAT because `evolve_hla` already ships the primitive. Research 192 (NextLat) covers belief-state latent dynamics. Research 314 establishes that our latent states have trivial symmetry, so the maximal-invariant machinery has no target.
  - **Code layer:** `latent_functor/reestimation.rs` ships the exact Comparator-→-re-estimate pattern under the name "coherence-driven re-estimation scheduler when `coherence < tau_reest`". This is the DiPOD-style vocabulary-mismatch failure mode — paper-vocabulary grep ("world-tracking", "Comparator Lyapunov") returns nothing, but the mechanism ships.
  - **Vocabulary translation** (paper → codebase): "world-tracking" → "coherence > tau_reest"; "Comparator" → "re-estimation trigger", "CLR vote"; "conserved quantity" → "BLAKE3 commitment", "FAME Proposition 3 sampling invariant"; "reduced manifold" → "HLA scalar projection", "subspace phase gate". Both sets grepped; all hits are documented above.
- **Q2 New class of behavior?** FAILS. The paper is a *theoretical reframing* of capabilities we already have. It does not introduce a new operator, a new gate, or a new capability — it explains why the existing ones work.
- **Q3 Product selling point?** Cannot finish the sentence. "Our NPCs track world data with Lie-group symmetry" fails because our latent states have trivial symmetry (Research 314). "Our NPCs do hierarchical coarse-graining" already ships as HLA projection without the Lie-group formalism.
- **Q4 Force multiplier?** Theoretical only. Connections to HLA, functor, DEC, LatCal, and the Lean proof infra all exist on paper but require introducing a `(μ, Σ)` latent state we deliberately don't have.

→ Fails Q1/Q2/Q3/Q4. **Not Super-GOAT. Not GOAT either** (no provable gain — there's nothing to benchmark because nothing new ships).

### 2.4 Fusion (speculative — documented for the record, no plan this session)

| Fusion | What it would produce | Blocker |
|---|---|---|
| World-tracking Lyapunov `K̂` × **`reestimation.rs`** | Formalize the implicit proportional feedback in the re-estimation scheduler as an explicit gain operator; prove `V̇ ≤ 0` for the closed loop | Requires a `(μ, Σ)` belief state to make `K̂` non-trivial — same blocker as Research 314 |
| Hierarchical nested manifolds × **HLA projection** | A typed coarse-graining chain `M₀ ⊃ … ⊃ M_k` over HLA's 8 affect channels, with per-level conserved quantities | Design decision in `riir-ai/crates/riir-engine/src/hla/`, not a katgpt-rs primitive |
| Moduli stack `[M/G]` × **DEC quotient cochain** | Symmetry-reduced Hodge decomposition on a quotient cell complex | Tenuous bridge — DEC operates on spatial meshes, not parameter manifolds (same verdict as Research 314's row 4) |
| Comparator-as-anomaly × **curiosity pulse (Research 041)** | Theoretical justification: curiosity = integral of `V̇` over the recent window; high curiosity = sustained Comparator disagreement | Reframing only; curiosity already ships |

None actionable today. The first (Lyapunov `K̂`) is the most likely to become real, but it requires the same `(μ, Σ)` belief-state redesign that Research 314 flags.

---

## 3. Verdict

**GAIN — deferred (no plan this session).**

| Criterion | Result |
|---|---|
| Modelless? | ✅ the math is modelless (no training required for any of the constructions) |
| Latent-to-latent? | △ would be, *if* we had `(μ, Σ)` latent states with non-trivial symmetry |
| Novel vs corpus? | ✗ Smets 321/166 covers the Lie-group-equivariance side; `reestimation.rs` ships the Comparator; Research 242 + `evolve_hla` ship the recurrent belief state; Research 314 establishes trivial-symmetry blocker |
| Maps to current data shapes? | ✗ HLA/shards have trivial symmetry; no `(μ, Σ)` latent state |
| New capability class? | ✗ theoretical reframing of capabilities that ship under different vocabulary |
| Product selling point? | ✗ cannot articulate one on current shapes |
| Force multiplier (≥2 pillars)? | ✗ connections are theoretical, require redesign |

**One-line reasoning:** A beautiful AIT-meets-Noether unification paper that *explains why* our existing primitives (`evolve_hla`, `reestimation.rs`, DEC `codifferential`, Plan 270 gauge invariance) work — but every load-bearing mechanism it formalizes is already shipped under codebase vocabulary, and the non-trivial-symmetry case it would unlock is blocked by Research 314's finding that our latent states have trivial symmetry groups. The practical value is conceptual (a theoretical frame for the docs / Lean proof story) and a trigger condition companion to Research 314.

**Trigger condition for re-evaluation (→ upgrade to GOAT + plan):**
- A future plan introduces a `(μ, V)` or `(μ, Σ)` latent state (Gaussian belief, distributional embedding, second-order personality representation) — *the same trigger as Research 314*.
- At that point, three Ruffini-grounded primitives become concrete GOAT candidates:
  1. **Lyapunov feedback gain `K̂`** as a named primitive over the `(μ, Σ)` belief — the formalization of the implicit proportional feedback in `reestimation.rs`.
  2. **Hierarchical nested-manifold projection** `M₀ ⊃ … ⊃ M_k` as a typed coarse-graining chain — the formal object behind HLA scalar projection.
  3. **Symmetry-discovery bridge** to LieGG / LieSD / LaLiGAN — but only the *inference-time read-out* side; the discovery itself is training-side (→ riir-train).
- Until then, the value of this note is **conceptual**: it documents (a) the theoretical unification of `evolve_hla` + `reestimation.rs` + DEC `codifferential` under the "world-tracking agent mirrors world symmetry" theorem, and (b) that the practical non-trivial-symmetry case is the same Research 314 blocker.

**Cross-references for the moat book:** No update to `riir-ai/.docs/pillars/` or `supergoat_candidates/` — this paper does not create or amplify a pillar. The Lie-group-equivariance moat angle is already captured by Research 166 (SE(2) game maps) and Research 321 (Smets textbook). The Comparator / world-tracking angle is already captured by Research 242 (Topological Trouble) + the shipped `reestimation.rs`.
