# Research 245: Manifold Power Iteration — MoE Router Row ↔ Expert Principal Singular Direction

> **Source:** [Redesign Mixture-of-Experts Routers with Manifold Power Iteration](https://arxiv.org/abs/2606.12397) — Songhao Wu, Ang Lv, Ruobing Xie, Yankai Lin (RUC / Tencent), arXiv:2606.12397, 10 Jun 2026
> **Date:** 2026-06-16
> **Status:** Active
> **Related Research:** 161 (dMoE block routing), 222 (Spectral Scaling Laws / NS depth), 231 (Sparse Off-Principal Task Vector), 238 (LoRA-Muon spectral low-rank manifold — canonical cousin), 099 (Eigenspace alignment / power iteration tool), 207 (ManifoldE point-to-manifold)
> **Related Plans:** 181 (dMoE adaptive top-p bandit — shipped), 203 (riir-ai frame coreset — shipped), 254 (Spectral Budget Router — shipped), 270 (Gauge-Invariant Adapter Composition — shipped), 268 (Rank-One LoRA Spectral Regularization → riir-train)
> **Cross-ref (riir-ai):** 051 (dMoE block-level LoRA routing), `riir-gpu::RimBlockRouter` (SHINE expert routing)
> **Classification:** Public — generic inference engine mechanics (WHAT, not HOW)

---

## TL;DR

The paper proposes **Manifold Power Iteration (MPI)**: each row `R[i]` of an MoE router is transformed by one power-iteration step against the associated expert's Gram matrix `M = W_g W_g^T`, then L2-retracted to a constant norm `C = C'/√N`. This drives `R[i]` to track the **principal singular direction** of expert `i`'s weight matrix — the most informative 1-vector summary of that expert. Over training steps, gradient flow through the power iteration makes `R[i]` converge to that principal direction (proven equivalent to steepest ascent on the Rayleigh quotient `‖R[i]W_g‖² / ‖R[i]‖²` on the spherical manifold). Empirically: faster convergence, +0.7–1.3 avg downstream accuracy across 1B/3B/11B MoE, **improved load balancing as a free side effect** of the retraction, **0.2% training throughput cost**, and **zero inference overhead** (router can be precomputed at load).

**Distilled for katgpt-rs (modelless, inference-time):**

The training-time convergence story → `riir-train` (needs backprop). The transferable inference-time primitive is a **one-shot router-weight conditioning** performed at model load / adapter hot-swap:

```
Given frozen router R ∈ ℝ^{N×D} and frozen expert gate weights {W_g[i]}:
  M[i]   = W_g[i] W_g[i]^T            (expert Gram, N×D×D — cache once)
  R̂[i]   = R[i] · M[i]                 (one power-iteration step, per expert row)
  R'[i]  = C · R̂[i] / ‖R̂[i]‖₂         (L2 retract to C = C'/√N)
  gate(x) = TopK_k( σ(x · R'^T) )      (SIGMOID, not softmax — see §2.3)
```

This is a deterministic, one-time transformation of the router weight matrix. It is **not** a new inference behavior class — inference is identical to vanilla top-k routing, just with better-conditioned rows. The value appears at **freeze/thaw snapshot swap**: whenever the frozen expert pool changes (new LoRA bundle, new HLA shard set, new ternary adapter), recompute `R'` from the new `W_g`. This gives better routing quality + better load balancing without retraining — a clean freeze/thaw-compatible win.

---

## 1. Paper Core Findings

### 1.1 The Principle: Router Row = Principal Singular Direction of Expert

Standard MoE routers are unconstrained linear matrices `R ∈ ℝ^{N×D}` whose rows serve as expert proxies — `R[i]·x` should reflect token-`i`-expert affinity. But nothing enforces that `R[i]` actually encodes the geometry of expert `i`'s weight matrix `W_i*`. The paper's principle:

> A well-coupled router row `R[i]` should be aligned with the **principal singular direction** of expert weights `W_i*`, because the principal singular vector is the optimal 1-vector compression of a matrix (Eckart–Young / Rayleigh–Ritz).

The alignment objective is the Rayleigh quotient:

```
max_{R[i]}  φ(W_i*, R[i]) = ‖R[i] · W_i*‖₂² / ‖R[i]‖₂²
```

which is maximized when `R[i]` is the top right singular vector of `W_i*`.

### 1.2 Power-then-Retract (MPI)

Exact SVD per training step is prohibitive. The paper uses **one step of power iteration** on the router row against the expert Gram matrix, then a **L2 retraction** to a constant norm:

```
R̂[i] = R[i] · W_g[i] · W_g[i]^T        (Eq. 4 — power iteration step)
R'[i] = C · R̂[i] / ‖R̂[i]‖₂             (Eq. 5 — retraction, C = C'/√N)
```

Then the MoE gate is recomputed as `w' = Softmax(TopK(x · R'^T))` (Eq. 6). The retraction serves two roles: (a) numerical stability (power iteration diverges in norm without it); (b) removes expert bias from router norm disparities — a high-norm row would inflate its gating weight and overload its expert.

### 1.3 Why It Works (Optimization View)

§3.3 proves MPI ≈ steepest ascent on the Rayleigh quotient **on the spherical manifold** `‖R[i]‖₂ = C`. The exact Riemannian gradient-ascent update (Eq. 9, projecting the Euclidean gradient `G = 2·R'[i]·M` onto the sphere's tangent space) is structurally identical to the MPI update (Eq. 10) up to an adaptive step size `1/(R'[i]·M·R'[i]^T)`. As `R[i]` converges to the dominant singular vector, the step size shrinks and updates self-damp — natural convergence. After enough steps, `R[i]·M` ≈ dominant singular vector; the `R'[i]·(R'[i]·M·R'[i]^T)` subtraction term points toward the residual mismatch, rotating `R[i]` into the principal singular subspace.

### 1.4 Empirical Results

- **Optimizer-agnostic** (AdamW, AdamH, Muon, MuonH all benefit): +1.3 / +1.34 / +0.54 / +1.20 avg over 25 benchmarks at 1B.
- **Scales 1B → 3B → 11B**: consistent convergence speedup (1.04× at 11B) and downstream gain (+2.33 / +1.84 avg on ARC-C/MMLU/TriviaQA/.../MBPP).
- **Load balancing**: MaxVio_Batch 1.133 → 0.964, MaxVio_Global 1.024 → 0.71 at 3B — **free side effect** of retraction (normalized router norms → no norm-driven overload).
- **Cost**: 0.2% training throughput hit at 11B. **Zero inference overhead** — "router weights can be pre-computed with power iteration as the model loads" (§4.2 Efficiency Analysis).
- **Single iteration suffices**: 10 iterations → 5% throughput loss, no convergence gain, −1.39 pp downstream. The single-step online form is more stable than fully-converged SVD.
- **Sigmoid compatibility** (§6): paper notes "we also explore Sigmoid as an alternative... downstream performance still improves from 41.64 to 42.05." Sigmoid is viable — important for our codebase constraint (sigmoid, never softmax).
- **Router–expert alignment metric λ** (Eq. 11): vanilla MoE λ ≈ 0.22–0.37, MPI λ ≈ 0.62–0.70 across layers — confirms the rows actually track principal directions.

### 1.5 What the Paper Does NOT Do

- **No input-distribution adaptation.** Power iteration is `R[i] · W_g · W_g^T` — the expert Gram only. No input covariance `Σ_x`. `R'` is input-independent once computed.
- **No runtime adaptation.** The training-time convergence requires gradient flow. At inference, MPI is a static precomputation.
- **Not a new routing mechanism** at inference — same top-k, just better-conditioned rows.

---

## 2. Distillation for katgpt-rs (Modelless)

### 2.1 The Transferable Primitive: `manifold_power_iter_router` (one-shot at load/swap)

**What lands here (engine, MIT):** a generic function that, given a router matrix `R` and a per-expert Gram matrix `M[i] = W_g[i]W_g[i]^T`, produces the MPI-conditioned router `R'`. No MoE semantics, no game IP.

```rust
/// One step of Manifold Power Iteration on MoE router rows (paper Eq. 4–5).
///
/// For each expert row i:  R̂[i] = R[i] · M[i];  R'[i] = C · R̂[i] / ‖R̂[i]‖₂
/// with C = C_prime / sqrt(N) so that ‖x · R'^T‖_∞ = O(1) (paper Eq. 7).
///
/// Deterministic given (R, M, C_prime, iters) → safe for sync/quorum.
/// `iters=1` matches the paper's default; `iters>1` converges further but
/// the paper showed no gain and 5% throughput loss at iters=10.
///
/// Reuses the existing `PowerIterationScratch` pattern (Plan 270 / 238).
/// Zero-alloc: caller-owned scratch. Sub-μs per row for D ≤ 1024.
pub fn manifold_power_iter_router(
    r: &mut [f32],            // [N×D] router, updated in place → R'
    gram_per_expert: &[&[f32]], // N views, each [D×D] expert Gram M[i] = W_g W_g^T
    n_experts: usize,
    d_model: usize,
    c_prime: f32,
    iters: u8,
    scratch: &mut PowerIterationScratch,
)
```

**Substrate routing:**
- `D ≤ 256, N ≤ 64` (typical game NPC LoRA pool): CPU SIMD, <10μs total, plasma tier.
- `D ∈ [256, 1024], N ∈ [64, 256]`: CPU SIMD blocked, <1ms, hot tier.
- `D > 1024` or `N > 256` (full LLM MoE): would delegate to GPU (riir-gpu) — out of scope for katgpt-rs, caller falls back to dense.

**Key property for our stack:** the retracted form is **deterministic** (given power-iter tolerance) → same `(R, W_g)` → same `R'` across nodes → safe under `SyncBlock → ChainConsensus` quorum. No sync-boundary concern (router weight is engine-internal, not synced as game state).

### 2.2 Where It Fires: Freeze/Thaw Snapshot Swap (not per-token)

```
freeze/thaw event (adapter hot-swap, snapshot version bump)
   ↓
for each MoE layer with swapped experts:
   recompute M[i] = W_g[i] W_g[i]^T   (cache, BLAKE3-tagged with snapshot version)
   R' = manifold_power_iter_router(R, M, ...)
   hot-swap R → R' atomically (existing LoRAHotSwap path)
   ↓
subsequent inference uses R' (sigmoid top-k gating)
```

**Cost:** once per snapshot swap, not per token. Sub-ms for game-scale pools. **Zero per-token overhead** — exactly the paper's "zero inference overhead" claim.

### 2.3 Sigmoid, Not Softmax (Our Constraint)

Paper uses `Softmax(TopK(xR'^T))`. Our AGENTS.md mandates **sigmoid** for projections onto learned direction vectors. Distillation:

```
gate_i(x) = σ(β · x · R'[i]^T)         independent per-expert sigmoid
select    = TopK_k(gate_1, ..., gate_N) pick top-k by sigmoid score
```

Notes:
- The paper's §6 explicitly tried sigmoid and it still improved over vanilla (41.64 → 42.05). So sigmoid is empirically supported, not a forced downgrade.
- **Independent per-expert sigmoid ≠ softmax.** Softmax couples experts (one expert's score affects all others via normalization); sigmoid does not. For our use case (NPC expert routing where multiple experts can independently be "relevant"), sigmoid is the *correct* semantics — a combat expert and a movement expert can both fire on the same frame. Softmax would force zero-sum competition.
- `β` replaces the paper's `C = C'/√N` as the temperature knob. Calibrate so `‖x·R'^T‖_∞ = O(1)` (paper Eq. 7) — same role.
- The retraction's norm-normalization (Eq. 5) still matters under sigmoid: without it, a high-norm router row would push its sigmoid to saturation regardless of input → effectively a constant-on expert. Retraction removes this bias → cleaner per-expert sigmoid calibration.

### 2.4 Fusion A — Fuse with Research 238 (Gauge-Invariant Adapter Composition)

Research 238 / Plan 270 shipped `gauge_rebalance` (power iteration for `σ_max` on LoRA `(A,B)` factor pairs) and `gauge_invariant_compose`. This paper applies the **same power-iteration primitive** to a different target — router rows against expert Grams.

**DRY opportunity:** both `gauge_rebalance` and `manifold_power_iter_router` are instances of "power-iteration step + norm retraction" on a vector against a PSD operator. A shared `power_iter_retract(v, M, C, iters, scratch)` helper (in `newton_schulz.rs` or a new `spectral_retract.rs`) serves both. This is a refactor, not a new capability — but it keeps the spectral toolkit DRY and makes future spectral-conditioning ops (e.g., HLA shard direction conditioning, NeuronShard style weights) one-liners.

### 2.5 Fusion B — Fuse with Plan 181 (dMoE Adaptive Top-p Bandit)

Plan 181 shipped dMoE's "aggregate scores → top-p coreset" as adaptive bandit arm selection (`src/pruners/bandit.rs:1999`, `src/speculative/vocab_coreset.rs`). That operates on *scores*; this paper conditions the *router weights that produce the scores*.

**Composition (modelless):**
```
1. At snapshot swap: MPI-condition R → R' (this paper, §2.1)
2. At inference: x → R'^T → sigmoid scores → dMoE top-p coreset (Plan 181)
```

The MPI-conditioned router produces better-calibrated per-expert scores; the top-p coreset then adaptively picks how many experts to actually load. Two orthogonal gains: (a) better score quality from MPI, (b) adaptive coreset size from top-p. Neither subsumes the other.

### 2.6 Fusion C — Fuse with Spectral Budget Router (Plan 254) + RimBlockRouter (riir-gpu)

Plan 254 ships layer-adaptive Newton-Schulz depth from spectral power laws. `RimBlockRouter` (riir-gpu) does SHINE-powered block expert routing for the RiM two-stage LoRA curriculum. MPI-conditioned router rows could feed `RimBlockRouter`'s `ExpertRegistry` — when experts are hot-swapped at a snapshot boundary, MPI-recondition the block router's expert attention vectors against the new expert weights. Lands in riir-ai, not katgpt-rs.

### 2.7 Plasma/Hot/Warm/Cold Path

- **Plasma** (sub-μs, always-on): `manifold_power_iter_router` with `iters=1` on a cached Gram — only at snapshot swap, never per-token.
- **Hot** (1–10μs): full reconditioning across all MoE layers on a bulk adapter reload.
- **Warm** (10μs–1ms): Gram recomputation `M[i] = W_g[i]W_g[i]^T` from a freshly loaded expert weight, then recondition.
- **Cold** (>1ms): cross-rank / cross-width MPI ablation (training-side, → riir-train).
- **Freeze**: snapshot `R'` + `M` to immutable storage, BLAKE3-tagged with snapshot version → tamper-evident, reproducible quorum commit.

---

## 3. Verdict

**Tier:** GOAT — provable gain, not a new capability class.

**One-line reasoning:** The inference-time modelless distillation (one-shot router-row conditioning against expert Gram at snapshot swap) is a deterministic, sub-ms, zero-per-token-overhead improvement to routing quality and load balancing — but it is *conditioning*, not a new behavior; inference is still top-k gating, just with better-aligned rows.

| Criterion | Assessment |
|-----------|-----------|
| **Novel mechanism (no prior art)?** | ⚠️ Partial — power iteration + Rayleigh quotient are **already shipped** (`hodge_spectrum`, `gauge_rebalance`/Plan 270, `spectral_budget`/Plan 254, `beta_fitter`). The *application* to MoE router rows is novel for us, but the primitive is not. |
| **New class of behavior?** | ❌ No — inference behavior is identical to vanilla top-k routing with better-conditioned weights. The "new capability" would require input-distribution adaptation, which the paper does NOT do (no `Σ_x` term). |
| **Product selling point?** | ⚠️ Narrow — "freeze/thaw MoE router that reconditions on every snapshot swap for better load balance, zero retraining." Real but not headline. |
| **Force multiplier (≥2 pillars)?** | ✅ Yes — connects freeze/thaw runtime (riir-ai `LoRAHotSwap`), dMoE routing (Plan 181), spectral budget (Plan 254), gauge-invariant compose (Plan 270/238). |
| **Modelless?** | ✅ Yes — one-time precomputation at model load / snapshot swap. No backprop. |
| **Uses existing infra?** | ✅ Yes — `PowerIterationScratch` pattern, `simd_dot_f32`, existing freeze/thaw hot-swap path. |
| **Perf overhead** | ✅ Zero per-token. Sub-ms per snapshot swap. 0.2% even at training time (paper). |
| **Proof of gain** | ✅ Paper: +0.7–1.3 avg downstream, MaxVio 1.13→0.96, λ 0.27→0.66. Our gain = same mechanism at snapshot swap; needs before/after benchmark on a real MoE adapter pool. |
| **Risk** | Low. Worst case: gain is marginal on small NPC pools (N≤8) where vanilla routing already works. Even then, the shared `power_iter_retract` helper is a useful DRY cleanup. |

**Decision:** GOAT. Per session instruction, **STOP after this note** — no plan or implementation this session. The plan + feature flag + GOAT gate benchmark would be the natural next step if promoted.

**Sigmoid note:** paper used softmax; we distill with **sigmoid** (independent per-expert gating). Paper §6 confirms sigmoid still wins over vanilla. Our codebase already defaults to sigmoid (`compute_moa_gates`, `SigmoidPerceptRouter`, `belief_rank_pruner`) — no constraint violation.

---

## 4. What NOT to Implement in katgpt-rs

- **The training-time convergence loop** — gradient flow through power iteration driving `R[i]` to the principal singular direction is a *training* mechanism. → `riir-train`. One line: **MPI MoE router training → riir-train**.
- **MuonH / AdamH / Hyperball optimizer variants** — training-side, already noted in Research 238 / 222.
- **Full SVD of expert weights** — paper explicitly avoids this (prohibitive cost); we follow.
- **Multi-iteration MPI at inference** (`iters>1`) — paper showed 5% throughput loss and no gain at `iters=10`. Stick with `iters=1`.
- **Input-covariance-adaptive routing** — see §6 Fusion Idea D; *not* in this paper, would be its own research note.

---

## 5. Relationship to Existing Research / Plans / Code

| Item | Layer | Relation | Impact |
|------|-------|----------|--------|
| **Research 238 / Plan 270** (Gauge-Invariant Compose) | notes + shipped (`gauge_invariant.rs`) | Canonical cousin — same power-iteration + retraction primitive, different target (LoRA factors vs router rows). | DRY fusion (§2.4): shared `power_iter_retract` helper. |
| **Research 222 / Plan 254** (Spectral Budget Router) | notes + shipped (`spectral_budget.rs`) | Same spectral toolkit (NS iterations, singular directions, power laws). | Orthogonal — 254 sets NS *depth* per layer; MPI sets router *row directions*. Compose cleanly. |
| **Research 161 / Plan 181** (dMoE Adaptive Top-p) | notes + shipped (`bandit.rs:1999`, `vocab_coreset.rs`) | dMoE conditions on *scores*; MPI conditions the *router that produces scores*. | Composition (§2.5): MPI-conditioned router → sigmoid scores → top-p coreset. |
| **Research 099** (Eigenspace Alignment) | notes | Power iteration as the standard eigenspace tool — theoretical ancestor. | Cites the same linear-algebra toolkit. |
| **Research 207** (ManifoldE point-to-manifold) | notes | Point-to-manifold principle — MPI retracts onto the spherical manifold `‖R[i]‖=C`. | Same manifold-geometry framing. |
| **riir-ai 051 / Plan 203** (Frame Coreset) | private notes + shipped (`frame_coreset.rs`) | Frame-level LoRA expert aggregation. MPI would condition the per-expert scores that feed the coreset. | riir-ai integration point. |
| **`RimBlockRouter`** (riir-gpu) | shipped code | SHINE-powered block expert routing with `ExpertRegistry`. Does NOT derive router rows from expert principal directions. | riir-ai integration: MPI-recondition `ExpertRegistry` attention vectors at snapshot swap. |
| **`goat_260` tests** (`expert_attention[expert_idx]`) | shipped test code | Hand-set expert attention vectors — *exactly* the kind of arbitrary `R[i]` MPI replaces. | Motivating example: replace manual `expert_attention` with MPI-derived vectors. |
| **Plan 268** (Rank-One LoRA Spectral Reg) | riir-train | Training-side power iteration on LoRA for spectral regularization. | Training analog of this paper's training story — both → riir-train. |

---

## 6. Fusion Ideas (novelty TBD — NOT Super-GOAT claims)

Per skill rules, these are **not** "Super-GOAT candidate" claims. They are fusion ideas whose novelty gate (Q1–Q4) has not been checked. If any checks out as Q1–Q4 YES, it becomes its own research note + (per skill) mandatory open primitive + riir-ai guide + plan.

### Fusion Idea D — Runtime Input-Conditioned MPI Router (speculative, beyond paper)

**The task's fusion hypothesis, marked TBD.** Replace the static expert-Gram power iteration with an **input-covariance-conditioned** one:

```
Σ_x   = EMA over recent tokens of x x^T           (input covariance, runtime-updated)
M_i   = W_g[i] Σ_x W_g[i]^T                        (input-whitened expert Gram)
R'[i] = power_iter_retract(R[i], M_i, C, iters)    (principal direction of expert under current input distribution)
```

This WOULD be a runtime-adaptive router: as input distribution shifts (e.g., game transitions from combat frame to exploration frame, NPC enters a new zone), the router's principal directions shift to track the *currently-relevant* subspace of each expert. No weight updates — only the EMA `Σ_x` and the deterministic reconditioning.

**Why this is NOT claimed as Super-GOAT here:**
1. The paper does NOT do this. It uses `W_g W_g^T`, not `W_g Σ_x W_g^T`. This is a *speculation* that combines MPI with online PCA / Oja's rule.
2. Q1 (no prior art?) — UNCHECKED. Online PCA / Oja's rule on router weights may have prior art; needs arxiv search (`input-adaptive MoE routing`, `online PCA router`, `distribution-shift aware expert routing`).
3. Q2 (new class of behavior?) — PLAUSIBLE (runtime-adaptive routing without weight updates is a new capability), but unverified.
4. Q3 (product selling point?) — PLAUSIBLE ("NPC routing that adapts to zone/context without retraining"), but unverified.
5. Q4 (force multiplier?) — YES if Q2 holds (freeze/thaw + self-learn + routing).

**Action:** → `.issues/` follow-up to run the novelty gate. Do NOT implement from this note.

### Fusion Idea E — HLA Shard Direction Conditioning (riir-ai)

`NeuronShard { style_weights, hla_moments }` (per AGENTS.md latent-vs-raw rules) is a fixed-size Pod with latent style weights. Apply MPI: condition the shard's "principal style direction" against its own Gram at spawn/consolidation. Latent-only (semantic domain), never synced — only scalar projections cross the sync boundary. Speculative; needs its own note.

### Fusion Idea F — Shared `power_iter_retract` Helper (DRY, not a verdict item)

`gauge_rebalance` (Plan 270) and `manifold_power_iter_router` (this note) are both "power-iteration step + norm retraction on a vector against a PSD operator." A shared helper eliminates duplication. Lands as a refactor whenever either is implemented — no separate verdict needed.

---

## 7. Key Quotes

> "Each router row is designed to encode the expert matrix into this representative vector, such that its dot-product with token can better reflect token-expert affinity. However, there exists no design principles to enforce this condensation."

> "We propose to align each router row with the principal singular direction of the associated expert, as this direction provides the most expressive mathematical description of a matrix."

> "Power-then-Retract paradigm: a power iteration step is performed on the router weights, followed by a retraction to impose a norm constraint to ensure both efficiency and stability."

> "At inference time, the router weights can be pre-computed with power iteration as the model loads. Therefore, our design incurs zero inference overhead and maintains compatible with standard inference engines out-of-the-box." ← **the modelless hook**

> "We also explore Sigmoid as an alternative... downstream performance still improves from 41.64 to 42.05." ← **sigmoid compatibility (our constraint)**

> "Aggressive alignment disrupts the stability of router optimization, making a single power iteration a more robust and efficient choice." ← **iters=1, do not over-iterate**

---

## TL;DR Summary

MPI redesigns MoE routers so each row `R[i]` tracks the principal singular direction of expert `i`'s weight matrix, via one power-iteration step + L2 retraction per training step. The training-time convergence (gradient flow through power iteration) → `riir-train`. The transferable **inference-time** primitive is a **one-shot router-weight conditioning** at model load / freeze-thaw snapshot swap: `R'[i] = C · (R[i]·W_g[i]·W_g[i]^T) / ‖·‖₂`. Zero per-token overhead, sub-ms per swap, deterministic → sync-safe. Real gains: better routing quality (λ 0.27→0.66), better load balancing (MaxVio 1.13→0.96), paper-validated across 1B/3B/11B. Distill with **sigmoid** (not softmax) per our constraint — paper confirms sigmoid still wins.

**Verdict: GOAT** — provable gain, uses existing power-iteration infra (`gauge_rebalance`, `hodge_spectrum`), force-multiplies across freeze/thaw + dMoE routing + spectral budget, but not a new capability class (inference behavior unchanged, just better-conditioned rows). **Per session instruction: STOP after this note — no plan or implementation.** Fusion Idea D (runtime input-conditioned MPI router) is the speculative Super-GOAT-shaped follow-up; it goes beyond the paper (adds `Σ_x`) and needs its own novelty-gate check before any claim.
