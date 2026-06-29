# Research 346: Learning Iterative Reasoning through Energy Diffusion (IRED)

> **Source:** [Learning Iterative Reasoning through Energy Diffusion](https://arxiv.org/abs/2406.11179) — Yilun Du, Jiayuan Mao, Joshua B. Tenenbaum (MIT), ICML 2024, PMLR 235
> **Survey context:** Gap G4 from `katgpt-rs/.research/325_Survey_Latent_Reasoning_Taxonomy_Unifying_Map.md` §7.2 — "Reasoning as iterative energy minimization via diffusion; multi-constraint problems refined from vague paths to precise solutions."
> **Date:** 2026-06-29
> **Status:** Done — **Gain** verdict (substance is → riir-train; modelless residual is small and largely covered)
> **Classification:** Public
> **Related Research:** 317 (Reasoning as Attractor — closest modelless cousin; cites the same energy-minimization framing), 111 (Dirichlet Energy structural alignment — the deterministic energy already shipped), 296 (Stokes/DEC vocabulary crosswalk — Hodge Laplacian IS an energy operator), 219 (TNO → DEC substrate), 271 (MIT 6.S184 diffusion/flow crosswalk), 308 (NSM "Price Is Not Right" — concurrent diffusion-policy training dependency)
> **Related Plans:** 251 (DEC operators — COMPLETE), 149 (Dirichlet Energy diagnostic), 139 (EGA Energy-Gated Attention), 276 (MicroRecurrentBeliefState — attractor kernel, honest null on "needs trained weights")
> **Cross-ref (riir-train):** `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs` — Hodge-decomposed reward shaping already exists for the *training-side* energy decomposition that IRED's energy head would require.

> ⚠️ **Pre-flight correction.** The user-supplied fetch URL (`arxiv.org/pdf/2410.22344`) resolved to a **different paper** — Scoville & Wen, *On Merge Trees with a Given Homological Sequence* (math.GM). The correct arxiv ID for IRED is **2406.11179** (verified via web search against the authors + ICML 2024 poster 34671 + the project page `energy-based-model.github.io/ired`). This note distills 2406.11179, NOT 2410.22344. (Interestingly, the mis-resolved paper is itself a discrete-Morse-theory paper — adjacent to the DEC/Hodge substrate we ship — but it is not the assigned IRED paper and is not distilled here.)

---

## TL;DR

IRED formulates reasoning as **gradient descent on a learned energy function** `Eθ(x, y)` over a sequence of K=10 annealed energy landscapes (cosine β-schedule, smoother→sharper), trained via denoising score supervision (`||∇Eθ(x, √(1−σ²)·y + σε) − ε||²`) plus contrastive energy shaping (`−log(e^{−E+} / (e^{−E+} + e^{−E−}))`). At inference, run T steps of `y_t = y_{t-1} − λ∇_y Eθ(x, y_{t-1})` per landscape with energy-decrease acceptance; the energy head naturally acts as a termination criterion. Empirically: 99.4% Sudoku (test), 62.1% (harder, vs SAT-Net 3.2%); 92.6% shortest-path (test), 91.9% (harder, vs diffusion baseline 46.9%). Adaptive compute (more T) generalizes to harder instances.

**Distilled for katgpt-rs (modelless, inference-time):** the load-bearing mechanism — the **trained energy head `Eθ(x,y)` that learns task constraints from (x,y) pairs** — is **fundamentally training-dependent** (Algorithm 1: Adam backprop). The §3.5 modelless-unblock protocol is exhausted: freeze/thaw cannot synthesize the conditional energy; a deterministically-constructed LoRA cannot represent learned task constraints; a latent-space projection onto a deterministic energy (Dirichlet, spectral, Hodge-Laplacian) degenerates to Laplacian smoothing and **does not learn Sudoku rules or matrix-inverse correctness**. The modelless residual — (a) the annealed-schedule inference loop, (b) gradient-descent-on-a-fixed-energy as an attractor-relaxation kernel — is **already substantially covered** by Research 317 (Gibbs attractor relaxation), Research 111 (shipped `dirichlet_energy`), Research 296 (Hodge Laplacian = energy operator), and shipped `ega_attn` / `module_energy_route`. **Verdict: Gain** — the paper validates the energy-as-reasoning framing our substrate already supports, but its specific contribution (a *trained* conditional energy head for multi-constraint satisfaction) is a genuine → riir-train dependency that no modelless path recovers.

---

## 1. Paper Core Findings

### 1.1 Reasoning as energy minimization

Cast reasoning as finding `y* = argmin_y Eθ(x, y)` where `Eθ` is a neural-network-parameterized energy function over (input `x`, candidate output `y`). Logical deduction = finding variable assignments satisfying constraints; theorem proving = finding valid deduction sequences; planning = finding action sequences respecting the transition model. Solve via gradient descent: `y_t = y_{t-1} − λ∇_y Eθ(x, y_{t-1})`, init `y_0 ~ N(0, I)`. The energy's value acts as a natural termination criterion and a difficulty signal — harder problems get more optimization steps T at inference.

### 1.2 The two techniques that make it work

**(a) Annealed sequence of energy landscapes.** Instead of one rugged `Eθ`, learn K=10 landscapes `Eθ^k` over a cosine β-schedule of noise scales `σ_k`. Each landscape represents `e^{−Eθ^k(x,y)} ∝ ∫ p(y*|x) · N(y; √(1−σ²)·y*, σ²·I) dy*`. Earlier landscapes (large σ) are smoother → easier to optimize; later landscapes (small σ) are sharper → precise. Inference optimizes landscapes sequentially, scaling the solution between them by `√(1−σ_k²) / √(1−σ_{k-1}²)`. **This is the diffusion-model insight ported to EBMs:** smoother→sharper annealing avoids local minima.

**(b) Score supervision + contrastive shaping (the training trick).** Prior EBMs (Du 2022, IREM) backprop through T optimization steps — slow and unstable. IRED instead supervises only single-step gradients:

```
L_MSE(θ) = || ∇_y Eθ(x, √(1−σ²)·y* + σ·ε; k) − ε ||²     (denoising score matching)
```

Plus a contrastive loss to enforce that the *global* energy minimum is the ground truth (not just the local gradient):

```
L_Contrast(θ) = −log( e^{−E+} / (e^{−E+} + e^{−E−}) )      (positive y* vs corrupted y⁻)
```

Both are **single-step losses** — no backprop through the optimization chain. This is the contribution over IREM.

### 1.3 Inference (Algorithm 2) — the part that's modelless-friendly

```
ỹ ~ N(0, I)
for k = 1..K:
    for t = 1..T:
        ỹ' ← ỹ − λ_k · ∇_y Eθ(x, ỹ; k)
        if Eθ(x, ỹ'; k) < Eθ(x, ỹ; k):  ỹ ← ỹ'    (energy-decrease acceptance)
    ỹ ← ỹ · √(1−σ_k²) / √(1−σ_{k-1}²)              (cross-landscape scaling)
return ỹ
```

This loop is **pure inference** — given a fixed `Eθ`. The energy-decrease acceptance is a discrete analog of Metropolis-without-the-reject-step. The annealed K-schedule is identical in spirit to diffusion-model reverse-process scheduling.

### 1.4 Results that matter

| Task | IRED (test) | IRED (harder) | Best baseline (harder) |
|---|---|---|---|
| Sudoku | 99.4% | **62.1%** | SAT-Net 3.2%, RRN 28.6%, IREM 24.6% |
| Visual Sudoku | 98.3% | 46.6% | SAT-Net 0.0%, RRN 28.6% |
| Connectivity | 99.1% | 93.8% | IREM 89.8%, Diffusion 61.3% |
| Shortest Path | 92.6% | 91.9% | Diffusion 46.9% |
| Matrix Inverse | 0.0095 MSE | 0.2063 | IREM 0.2083 |

**Adaptive compute matters most on harder instances** (Sudoku Table 5, matrix-inverse Table 2): more T → substantially better generalization. The energy landscape's geometry (Fig 4) shows clear smooth→sharp sharpening across K=1→10.

### 1.5 Stated limitations (paper §5)

- Many gradient steps needed at inference (slower than domain-specific solvers like Dijkstra).
- Annealed sequence is fixed Gaussian noise increments — could be *learned* (paper's own future-work note).
- **No additional memory** — out-of-the-box IRED cannot store intermediate results, so chain-of-thought-style multi-step reasoning is not directly supported.
- Could serve as a policy model if supervised with reward signals instead of IID positives.

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper → codebase — both layers)

| Paper term | Codebase equivalent (shipped) | Location |
|---|---|---|
| Energy function `Eθ(x,y)` (generic) | `dirichlet_energy` (deterministic: `Σ A_ij ‖h_i − h_j‖²`); `ega_attn` "Energy-Gated Attention spectral salience"; `module_energy_route` (DEFAULT-ON) | `katgpt-rs/crates/katgpt-core/src/dirichlet.rs`; features `ega_attn`, `module_energy_route` |
| Energy landscape → Laplacian / Hodge | `hodge_laplacian` (Δ = δd + dδ — IS the Dirichlet-energy Euler-Lagrange operator); `hodge_decompose` (exact/coexact/harmonic) | `katgpt_dec` crate (Plan 251, Research 219/296) |
| Gradient descent on energy → attractor relaxation | `AttractorKernel` (Hopfield-style recurrent sigmoid update); `MicroRecurrentBeliefState::step` (iterated to fixed point) | `katgpt-rs/crates/katgpt-core/src/micro_belief/` (Plan 276) |
| Annealed schedule (smooth→sharp) | cosine β-schedule equivalent: `subspace_phase_gate::phase_transition_gate` (N≥d transition); `spectral_hierarchy` (multi-resolution spectral); diffusion β-schedules in `katgpt_dllm` | `crates/katgpt-core/src/subspace_phase_gate/`; feature `spectral_hierarchy` |
| Gibbs / energy-weighted retrieval | `opus::boltzmann_probabilities` (τ-controlled); PlackettLuce Gibbs sampling; **Research 317's `1/E²` sharpening** | `src/pruners/opus/boltzmann.rs`; R317 |
| Energy-decrease acceptance (Alg 2) | `viable_manifold_graph` (discrete safe-manifold navigation); FPRM damped fixed-point halting (R266) | `crates/katgpt-core/src/viable_manifold_graph.rs`; `gain_cost_halt` |
| Trajectory energy = length-normalized NLL | `spectral_entropy_dct`; `ac_prefix::conditional_logprob` (computes exactly this) | `src/chiaroscuro/entropy.rs`; `crates/katgpt-core/src/ac_prefix/` |
| Score function supervision `∇E` | `temporal_deriv` (dual fast/slow surprise); `latent_trajectory_geometry` | `temporal_deriv.rs`, `latent_trajectory_geometry.rs` |
| Multi-constraint satisfaction | `ConstraintPruner` trait; QuestBench CSP (R008); `closure` instrument | `traits.rs`, `questbench.rs`, `closure/` |
| Hodge-decomposed reward (training-side analog) | `hodge_reward::HodgeRewardConfig` (α/β/γ exact/coexact/harmonic weights) | **`riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs`** |

### 2.2 What's the load-bearing contribution?

Decompose IRED into three layers and audit each for modelless-ness:

| Layer | What it does | Modelless? |
|---|---|---|
| **(A) Annealed-schedule inference loop** (Alg 2) | K landscapes × T gradient steps with energy-decrease acceptance + cross-landscape scaling | **YES** — pure inference given a fixed energy. The K-schedule is a hyperparameter (cosine β). |
| **(B) Energy-decrease acceptance + gradient descent** | `y_t = y_{t-1} − λ∇_y E` | **YES** — closed-form given `E` and its gradient. This is generic optimization, no training. |
| **(C) The conditional energy head `Eθ(x, y)`** | A trained NN that, given the input `x` (Sudoku board, matrix, graph) and a candidate `y`, returns a scalar measuring how well `y` satisfies the task's *learned* constraints. | **NO — fundamentally training-dependent.** Algorithm 1 trains `θ` via Adam on denoising + contrastive losses over (x, y) pairs. |

Layer (C) is the substance. Layers (A) and (B) are scaffolding. **The 99.4% Sudoku and 92.6% shortest-path numbers come from `Eθ` having learned the task constraints from data** — not from the gradient-descent loop (which is generic).

### 2.3 The §3.5 modelless-unblock protocol — honest audit

The protocol demands checking all three paths before deferring to riir-train. Apply each to layer (C):

**Path 1 — Freeze/thaw snapshot correction.** Could a frozen snapshot, thawed at inference, substitute for the trained `Eθ`? **No.** Freeze/thaw swaps a frozen weight state. IRED's `Eθ` is not "a biased weight state needing correction" — it is a *learned-from-scratch function* mapping `(x, y)` pairs to constraint-satisfaction scores. There is no pre-existing deterministic construction to correct. **Path 1 fails.**

**Path 2 — Deterministically-constructed reader/writer LoRA hot-swap.** Could a closed-form LoRA overlay synthesize the energy landscape? **No.** A LoRA overlay is `W + BA` — a rank-r perturbation of an existing weight matrix. IRED's `Eθ` is a 512×512×512 MLP trained from random init on (x,y) pairs. There is no base `W` to perturb; the entire function is learned. A deterministically-constructed LoRA can do scale-by-½, zero-out-positions, identity-minus-projection (the AC-Prefix G1 unblock pattern, R295/Issue 003) — but it cannot synthesize "does this Sudoku board satisfy the rules" from nothing. **Path 2 fails.**

**Path 3 — Latent-space correction (deterministic energy).** Could a deterministic energy (Dirichlet, spectral, Hodge-Laplacian) substitute for the learned `Eθ`? **Partially — and this is where the verdict crystallizes.** Three candidate deterministic energies ship in our codebase:

   - **Dirichlet energy** `E(h) = Σ A_ij ‖h_i − h_j‖²` (shipped, `dirichlet.rs`, Plan 149). Its gradient is `∇_h E = L·h` (graph Laplacian). Running IRED's Algorithm 2 with `Eθ → dirichlet_energy` produces **graph Laplacian smoothing** — a known weak primitive that pushes embeddings toward their neighbors. It captures "smoothness w.r.t. adjacency," not "satisfies Sudoku rules" or "is a valid matrix inverse." Laplacian smoothing is already implicit in `viable_manifold_graph` and several spectral primitives.
   - **Hodge Laplacian** `Δ = δd + dδ` (shipped, `katgpt_dec`, Plan 251). The Dirichlet energy's Euler-Lagrange equation is `Δh = 0` — so optimizing the Dirichlet energy *is* solving the Laplace equation on the cell complex. Same degeneration to smoothing.
   - **Spectral energy** (shipped via `spectral_hierarchy`, `spectral_pruner`, `ega_attn`). Captures frequency-domain concentration. Same smoothing character.

   **None of these learn task constraints.** A deterministic energy is a fixed functional; IRED's value is that `Eθ` is *learned from data* to represent whatever constraints the task has. Replacing `Eθ` with Dirichlet energy gives "iterative Laplacian smoothing" — a primitive we already ship in several forms. **Path 3 fails to recover IRED's actual capability.**

**§3.5 verdict: genuine riir-train dependency.** All three paths fail. The documentation requirement: Path 1 fails because `Eθ` is a learned function, not a correctable state; Path 2 fails because there is no base weight to perturb; Path 3 fails because deterministic energies degenerate to smoothing and do not learn task constraints. **What specifically requires gradient descent: learning a conditional function from (x,y) pairs that scores constraint satisfaction — this is supervised regression and cannot be done in closed form for arbitrary constraint structures.**

### 2.4 The DEC fusion angle — real but doesn't rescue modellessness

The strongest Super-GOAT framing the assignment asked us to attempt: **energy diffusion as gradient descent on the Hodge Laplacian over a belief cochain.** Audit:

- The math is sound: `⟨h, Δh⟩` is the Dirichlet energy; gradient flow on it is the heat equation `∂_t h = −Δh`, which is exactly Laplacian smoothing on the cell complex.
- The substrate ships: `katgpt_dec::hodge_laplacian`, `hodge_decompose` (exact ⊕ harmonic ⊕ coexact = Helmholtz), `DecFlowField` with three channels.
- The fusion *would* be: run IRED's Algorithm 2 with `Eθ(x, y) → ⟨y, Δ_x y⟩` where `Δ_x` is the Hodge Laplacian of the input graph `x`. This gives "annealed heat-equation relaxation on the input's cell complex."

**Why this is not a Super-GOAT:**

1. **Capability collapse.** The fused primitive is graph-Laplacian smoothing on the input's topology. It captures *structural* constraints (connectivity, boundary coherence) but not *semantic* constraints (Sudoku digit rules, matrix-inverse correctness). On Sudoku it would smooth the digit field toward neighbors — producing invalid boards, not solutions. On matrix inverse it would smooth the matrix toward itself — not invert it.
2. **Already shipped under different vocabulary.** `viable_manifold_graph` (Plan 312, DEFAULT-ON) ships discrete safe-manifold navigation; `dirichlet_energy` ships the energy; `ega_attn` ships energy-gated attention; `hodge_decompose` ships the decomposition. The fused primitive is a re-skinning of existing machinery in IRED vocabulary, not a new capability.
3. **Research 317 already covers the modelless half.** The closest cousin (R317, *Reasoning as Attractor Dynamics*) explicitly maps "energy-weighted retrieval on K trajectories" onto shipped `BoMSampler + boltzmann + spectral_entropy_dct`, and explicitly lists "latent gradient descent on energy" as future work. IRED *is* that future work — and the answer is "yes, but the energy head must be trained." R317's §1.7 quotes the paper's own future-work note: *"Instead of sampling discrete tokens, future work could perform gradient descent on the energy function ∇_h E(h) with respect to latent states h."* IRED delivers exactly this — at the cost of a trained `Eθ`.

**The DEC fusion is a clean GOAT-tier primitive if scoped honestly:** "annealed Laplacian smoothing on a cell complex as an attractor-relaxation kernel" — but it is **subsumed by R317 + R111 + R296 + shipped `dirichlet_energy`/`viable_manifold_graph`/`ega_attn`**. It does not clear novelty gate Q1 (prior art) or Q2 (new capability class).

### 2.5 Closest-cousin fusion (the Super-GOAT attempt that fails)

Per the fusion protocol, list the 2–3 closest notes and ask: "what novel combination does paper × A × B produce that none alone can?"

- **A = R317 (Reasoning as Attractor):** Gibbs-weighted energy scoring on K trajectories.
- **B = R296 (Stokes crosswalk) + shipped DEC:** Hodge Laplacian as energy operator.
- **Paper = IRED:** annealed gradient descent on a *learned* energy.

**Fusion candidate:** "Annealed Gibbs-weighted attractor relaxation on the Hodge Laplacian over a per-NPC belief cochain, where the energy is the Dirichlet energy of the belief field w.r.t. the zone's cell complex." This is *theoretically* a Super-GOAT — it would give every NPC an energy-minimization reasoning loop over their spatial belief manifold, annealed across K noise scales, converging to a belief minimum.

**Why it fails the novelty gate:**

- **Q1 (no prior art): FAILS.** `dirichlet_energy`, `hodge_laplacian`, `viable_manifold_graph`, `AttractorKernel`, `MicroRecurrentBeliefState::step`, R317's Gibbs weighting — every load-bearing piece ships. The combination is a re-wiring of existing primitives.
- **Q2 (new capability class): FAILS.** "NPC belief converges to a smooth minimum on the zone manifold" is what `evolve_hla` + `viable_manifold_graph` already do — HLA's leaky integrator IS a (first-order) energy-minimization step on the per-NPC latent state. IRED's annealed K-schedule adds a multi-resolution refinement, but that is incremental over `spectral_hierarchy` (multi-resolution spectral) and `subspace_phase_gate` (phase-transition gating).
- **Q3 (product selling point):** "Our NPCs anneal their beliefs across K energy landscapes" — true but indistinguishable in product terms from "our NPCs converge to stable beliefs" (already a selling point via HLA + attractor kernel).
- **Q4 (force multiplier, ≥2 pillars):** Passes (Foundation Layer DEC + AI Layer HLA + Reasoning Pack), but Q4 alone ≠ Super-GOAT.

**0/4 YES. Not Super-GOAT.** The DEC angle is real and worth recording, but it does not produce a new capability class — it produces a re-skinning of shipped primitives in IRED vocabulary.

---

## 3. Verdict

### Tier: **Gain** — paper is mostly → riir-train; the modelless residual is small and largely covered by existing research + shipped primitives.

**One-line reasoning:** IRED's load-bearing contribution is a *trained conditional energy head* `Eθ(x,y)` that learns task constraints from data; the §3.5 protocol's three modelless paths all fail to recover this capability (freeze/thaw has nothing to correct, raw/lora has no base to perturb, deterministic energies degenerate to Laplacian smoothing which already ships), and the DEC/Hodge fusion angle collapses to "annealed Laplacian smoothing on a cell complex" — a re-skinning of `dirichlet_energy` + `viable_manifold_graph` + `hodge_laplacian` in IRED vocabulary, not a new capability class.

### Novelty gate (Q1–Q4) — all NO

| Gate | Question | Honest answer |
|---|---|---|
| **Q1 Novelty** | Any existing code cover this? | **FAILS.** R317 covers the energy-as-attractor framing; R111 ships `dirichlet_energy`; R296 maps Hodge Laplacian = energy operator; `ega_attn`/`module_energy_route` ship energy-gated routing; `AttractorKernel` + `MicroRecurrentBeliefState` ship gradient-descent-on-energy kernels. The modelless half is covered. |
| **Q2 New capability class** | New behavior, not just better numbers? | **FAILS.** "Annealed gradient descent on a fixed energy" is Laplacian smoothing + multi-resolution refinement — both shipped. IRED's actual capability (solving unseen Sudoku / matrix inverse) requires the trained `Eθ`, which is → riir-train. |
| **Q3 Product selling point** | "Our NPCs/systems do X that no competitor can"? | **FAILS.** The modelless subset ("NPCs anneal beliefs across K energy landscapes") is indistinguishable in product terms from existing HLA + attractor convergence. |
| **Q4 Force multiplier (≥2)** | Connects to ≥2 existing pillars? | Passes (DEC + HLA + Reasoning Pack), but Q4 alone ≠ Super-GOAT. |

**0/4 YES → Gain.** No private guide required (verdict ≠ Super-GOAT). No katgpt-rs implementation this session — the modelless residual is already covered; the substantive contribution is a → riir-train dependency.

### Tiers audit (why not higher)

- **Not Super-GOAT:** fails Q1, Q2, Q3. The DEC fusion is the strongest framing and it still collapses to re-skinned existing primitives.
- **Not GOAT:** GOAT requires a *provable gain over existing approach*. We have no provable gain because the substantive half (the trained energy head) is out of scope for katgpt-rs. The modelless residual (annealed Laplacian smoothing) is not provably better than `viable_manifold_graph` + `spectral_hierarchy` for any workload we have.
- **Not Pass:** the paper is genuinely relevant to our substrate (the DEC/Hodge energy framing is real and worth recording), and the §3.5 audit is non-trivial — it's worth a note so future distillations don't re-attempt the same modelless rescue. The survey §7.2 G4 explicitly flagged this as a candidate; this note closes that gap with an honest verdict.

### Routing

- **No katgpt-rs implementation this session.** The modelless half is covered by shipped primitives.
- **No private guide (riir-ai / riir-chain / riir-neuron-db).** Verdict ≠ Super-GOAT.
- **→ riir-train note (if/when a training-side distillation session happens):** the IRED training recipe (denoising score supervision + contrastive energy shaping, Algorithm 1) is a clean, stable EBM training method that improves on IREM (Du 2022) by avoiding backprop-through-optimization. It would slot naturally next to the existing `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs` (which already does Hodge-decomposed reward shaping on the DEC substrate). The combination — IRED's annealed energy landscapes + Hodge-decomposed reward — is a genuine riir-train research direction. **Not in scope for this session** (this is the katgpt-rs research workflow); flagged for future triage.
- **Update survey §7.2 G4 status:** the gap is now closed with a **Gain** verdict. The DEC/Hodge fusion angle is real but does not produce a new capability class modellessly.

---

## 4. Honest caveats on the trained-vs-modelless question

(Per the assignment's explicit ask.)

1. **The energy head `Eθ` is unambiguously trained.** Algorithm 1 is Adam backprop on a 512³ MLP (continuous tasks) or a 7-layer ResNet (Sudoku). There is no modelless interpretation of "learn `Eθ` from (x,y) pairs." This is not a borderline case like LATENTSEEK (G5, policy-gradient-on-latents-at-test-time) — IRED's training is offline, batch, backprop-through-weights.

2. **The §3.5 protocol is genuinely exhausted, not shortcut.** All three paths were checked concretely:
   - Path 1 (freeze/thaw): no correctable state exists — `Eθ` is learned from scratch.
   - Path 2 (raw/lora hot-swap): no base weight to perturb — the entire function is the learned component.
   - Path 3 (deterministic energy): Dirichlet/Hodge/spectral energies all degenerate to smoothing and do not learn task constraints. This is the *substantive* path and it fails for a *substantive* reason: deterministic energies are fixed functionals; IRED's value is that the energy is *learned*.

3. **The DEC fusion angle is the strongest modelless framing and it still fails Q2.** "Annealed gradient descent on the Hodge Laplacian over a belief cochain" is mathematically clean (it's the heat equation with an annealed diffusion coefficient) and the substrate fully ships. But it produces Laplacian smoothing, not reasoning. A cell complex's Hodge Laplacian captures *topological* constraints (which fields are curl-free, which are divergence-free); it does not capture *task* constraints (Sudoku digit rules, matrix-inverse correctness, path validity). For game AI, this means "NPC belief converges to a smooth minimum on the zone topology" — which `evolve_hla` + `viable_manifold_graph` already approximate.

4. **The closest the modelless half gets to a real primitive** is what R317 already identified as future work: "latent gradient descent on a fixed energy" as an attractor-relaxation kernel. IRED supplies the inference loop (Algorithm 2) for this — and that loop is generic optimization given a fixed energy. R317 + shipped `AttractorKernel` + shipped `dirichlet_energy` already cover this combination; the only missing piece is a named `annealed_energy_relaxation` kernel that wires Algorithm 2's K-schedule × T-steps × energy-decrease-acceptance over a caller-supplied energy function. That is a ~50-line GOAT-tier wrapper (not Super-GOAT — it's a re-arrangement of existing primitives), and it is **not in scope for this session** because the verdict is Gain, not GOAT. If a future plan needs it, the recipe is: take `MicroRecurrentBeliefState::step` (the inner T-loop), wrap it in an outer K-loop over a cosine β-schedule of noise scales, accept on `E(y') < E(y)` where `E` is caller-supplied (default `dirichlet_energy`). File as a future plan if a concrete consumer materializes (e.g., a per-NPC belief-refinement runtime that wants multi-resolution convergence).

5. **The riir-train angle is the honest follow-up.** IRED + the existing `hodge_reward.rs` (Hodge-decomposed reward shaping) is a genuine research direction: train an energy head whose landscape decomposes into exact (goal-seeking) / coexact (exploration) / harmonic (strategic) components via the Hodge decomposition, with IRED's denoising + contrastive losses. This is **strictly a riir-train concern** — out of scope for katgpt-rs but worth flagging for triage.

---

## 5. Cross-references

- **Survey gap closed:** `katgpt-rs/.research/325_Survey_Latent_Reasoning_Taxonomy_Unifying_Map.md` §7.2 G4 (IRED) — verdict recorded as Gain.
- **Closest modelless cousin:** `katgpt-rs/.research/317_Reasoning_As_Attractor_Dynamics_Gibbs_Retrieval.md` — R317 §1.7 quotes the paper's future-work note that IRED delivers; R317's §2 documents that the modelless half ships; this note documents that the substantive half (trained `Eθ`) is → riir-train.
- **DEC/Hodge substrate:** `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` + `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` + Plan 251.
- **Deterministic energy already shipped:** `katgpt-rs/crates/katgpt-core/src/dirichlet.rs` (feature `dirichlet_energy`, Plan 149, Research 111).
- **Training-side Hodge reward (riir-train):** `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs` — the natural home for an IRED-style trained energy head if/when that work is triaged.

---

## TL;DR

IRED's headline numbers (99.4% Sudoku, 92.6% shortest-path) come from a **trained conditional energy head `Eθ(x,y)` that learns task constraints from data** — this is fundamentally training-dependent (Algorithm 1: Adam backprop on denoising + contrastive losses). The §3.5 modelless-unblock protocol is exhausted: freeze/thaw has nothing to correct, raw/lora has no base to perturb, deterministic energies (Dirichlet/Hodge/spectral — all shipped) degenerate to Laplacian smoothing and do not learn task constraints. The DEC fusion angle ("annealed gradient descent on the Hodge Laplacian over a belief cochain") is mathematically clean and the substrate fully ships, but it produces smoothing not reasoning, and is subsumed by R317 + R111 + R296 + shipped `dirichlet_energy`/`viable_manifold_graph`/`ega_attn`/`AttractorKernel`. **Verdict: Gain** — the modelless residual is small and largely covered; the substantive contribution is a → riir-train dependency (IRED + existing `riir-train/.../hodge_reward.rs` is a genuine training-side research direction). Survey §7.2 G4 closed with an honest verdict. **Correction note:** the user-supplied fetch URL resolved to the wrong paper (Scoville & Wen discrete Morse theory); correct arxiv ID is 2406.11179.
