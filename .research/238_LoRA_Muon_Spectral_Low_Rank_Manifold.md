# Research 238: LoRA-Muon — Spectral Steepest Descent on the Low-Rank Manifold

> **Source:** [LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold](https://arxiv.org/pdf/2606.12921) — Cesista, Crowson, Simal, Biderman (EleutherAI / NaXys UNamur / Ateneo de Manila), arXiv:2606.12921, Jun 2026
> **Date:** 2026-06-14
> **Status:** Active — Fusion Research
> **Related Research:** 114 (AMUSE → Plan 152), 152 (Newton-Schulz infra), 166 (Muon Curvature/NDS), 222 (Spectral Scaling Laws), 231 (Sparse Off-Principal Task Vector), 135 (Parallax Attention)
> **Related Plans:** 152 (Newton-Schulz — DONE default-on GOAT 25/25), 254 (Spectral Budget Router), 270 (this doc — Gauge-Invariant Adapter Composition), 094 (Memo Reflections TIES Merge — UNBLOCKED), 201 (Rosetta Pruner — UNBLOCKED), 233 (Rosetta Cross-Game Alignment — UNBLOCKED)
> **Cross-ref (riir-ai):** Research 124 (LoRA-Muon training integration), Plan 299 (LoRA-Muon optimizer)
> **Classification:** Public — generic inference engine mechanics (WHAT, not HOW)
> **PASS-Redirects (synthesis):** CMuon [arXiv:2608.02502 "CMuon: Accelerating and Stabilizing Diffusion Transformer Training via Chunked Momentum Orthogonalization"] — training optimizer that chunks fused QKV/AdaLN/FFN matrices before Newton-Schulz to avoid subspace interference (shared `(GᵀG)^{-1/2}` preconditioner mixes disjoint gradient covariance); no modelless analog at inference (we don't orthogonalize gradient matrices at runtime). Minor design-principle echo: if Plan 152's NS infra is ever applied to a *fused* projection at inference, chunk by functional sub-block first — but this is already implied by our keep-distinct-signals-separate discipline. → riir-train for the optimizer itself.
> **PASS-Redirects (synthesis):** Dang, Wen & Malladi [arXiv:2610.08975 "The Best Optimizer Depends on Batch Size"] — no universal Muon LR scaling rule across batch sizes/tasks + optimizer rankings reverse with batch size even after extensive retuning; any LoRA-Muon adoption (this note's training half) must ship a per-batch measured LR rule, never a transplanted constant — riir-train Research 473 + Plan 455 T4 + Issue 625 own the two-batch gate.

---

## TL;DR

LoRA-Muon derives the Muon optimizer's spectral steepest-descent rule directly on the **low-rank manifold** `M_r = {W : rank(W) = r}` instead of on individual LoRA factors. The result is **gauge-invariant**: the induced update on `W = AB^T` is unchanged under arbitrary factor reparametrizations `(A, B) → (AR, BR^{-T})`. Optimal learning rates transfer across rank, width, depth, and factor-rescaling — a rank-2 proxy recovers the dense best LR.

**Distilled for katgpt-rs (modelless, inference-time):**

The gauge-invariance theorem is not just a training property. At inference time, when we **compose** trained LoRA adapters (`SparseTaskVector` from Plan 231, `task_arithmetic` from Plan 094, Rosetta cross-game alignment from Plan 233), the result of arithmetic on `(A, B)` pairs is **gauge-dependent** — different factorizations of the same `W` give different merged outputs. The paper's **scalar gauge rebalancing** (Algorithm 2) is a pure matrix operation that removes this artifact. Combined with the missing **Newton-Schulz inverse square root** for Gram matrices, this gives us a generic, gauge-invariant adapter-merge primitive that unblocks composition-heavy plans.

**Three modelless fusions:**

1. **`ns_inv_sqrt_psd`** — Newton-Schulz inverse square root for PSD Gram matrices. Currently missing from `newton_schulz.rs`. Generic building block. Used by gauge-invariant merge and any future Muon-family LoRA training.
2. **`gauge_rebalance`** — scalar gauge rebalancing for `(A, B)` factor pairs. Pure inference-time, ~O(r) power iteration. Makes task-vector arithmetic well-defined.
3. **`gauge_invariant_compose`** — composition primitive that rebalances inputs before any linear combination (lerp, TIES, weighted average). Drop-in for `SparseTaskVector::apply_to` callers.

**All three are engine plumbing — no know-how leak.** The specific LoRA training recipe that exploits them is `riir-ai/.research/124`.

---

## 1. Paper Core Findings

### 1.1 The Problem: Factor-Coordinate Optimizers Are Gauge-Sensitive

A LoRA weight `W = AB^T` has many equivalent factorizations `(A, B) ~ (AR, BR^{-T})` for any invertible `R`. Standard optimizers (AdamW, Spectron) treat `A` and `B` as independent coordinates, so the realized update on `W` depends on the arbitrary choice of factorization. In particular:
- AdamW's per-factor weight decay induces `(1 - λη)²` product mismatch vs dense Muon's `(1 - λη)`.
- Spectron's trust radius depends on `‖A‖ + ‖B‖`, which is not gauge-invariant.
- Empirically (Fig 4): scalar gauge rescaling by `c=9` changes Spectron's validation loss by `2.0e-2` (95% CI `[1.6e-2, 2.5e-2]`), while LoRA-Muon changes by `-5.1e-5` (consistent with zero).

### 1.2 The Fix: Projector-Form Update on the Tangent Space

Steepest descent on the low-rank manifold `M_r` under unitary-invariant norms collapses onto projector-form updates:

```
ΔW = (η/2) · LMO(G · P_B) + (η/2) · LMO(P_A · G)
```

where `P_A = A(A^T A)^{-1} A^T`, `P_B = B(B^T B)^{-1} B^T`. Specializing the LMO to the spectral norm (`LMO = -msign`) yields **LoRA-Muon**:

```
ΔA = -(η/2) · msign(∇_A f · S_B^{-1/2}) · S_B^{-1/2}
ΔB = -(η/2) · msign(∇_B f · S_A^{-1/2}) · S_A^{-1/2}
```

where `S_A = A^T A`, `S_B = B^T B`. **This requires Newton-Schulz inverse square root of PSD matrices** — currently absent from `newton_schulz.rs`.

### 1.3 Split Weight Decay

To match dense Muon's `(1 - λη)` weight decay on the product `W = AB^T`:

```
s = √(1 - λη)
A_{t+1} = s · A_t + ΔA / s
B_{t+1} = s · B_t + ΔB / s
```

This is **gauge-invariant** (Appendix A.3, Prop 4) and ensures LR/WD transfer across rank.

### 1.4 Scalar Gauge Rebalancing (Algorithm 2) — The Modelless Gem

For numerical conditioning, the paper periodically rescales factors to balance their spectral norms:

```
c = (σ_max(B) / σ_max(A))^{α/2}
A ← c · A
B ← B / c
```

with `α ∈ (0, 1]` damping exponent. Crucially: **this does not change the induced update on `W`** (Corollary 1) — it's a pure numerical conditioning step. First moments transform oppositely (`m_A ← m_A / c`, `m_B ← c · m_B`) to preserve EMA equivariance (Prop 7).

**This is a pure inference-time primitive**: given any `(A, B)` pair from any source, we can rebalance them without changing `AB^T`. Useful for **adapter composition, task-vector arithmetic, and cross-game alignment**.

### 1.5 Empirical Results

- **Rank-2 LoRA-Muon** (4.3% of dense params) recovers dense best LR (val loss 2.156 vs dense 1.789, but same LR optimum 0.1).
- **Rank-32 LoRA-Muon** beats dense baseline in seed-averaged sweep (val loss 1.776 vs 1.789).
- LR transfers across rank (1, 2, 4, 8, 16, 32), width (64, 128), depth (1, 2, 4 layers), and gauge scale (1, 3, 9, 27).
- Spectron is gauge-sensitive; LoRA-RITE's QR-core is algebraically equivalent but QR-free + first-moment-only is more accelerator-friendly.

### 1.6 Newton-Schulz Inverse Square Root (Algorithm 4)

For PSD matrix `P`, compute `P^{-1/2}` via polynomial Newton-Schulz with damping `γ = 1.001`:

```
t = ‖P‖_F
P_0 = P/t + ε·I
X_0 = I
for k = 0..K-1:
    W_k = (a_k/γ)·I + (b_k/γ³)·P_k + (c_k/γ⁵)·P_k²
    X_{k+1} = X_k · W_k
    P_{k+1} = sym(P_k · W_k²)
return t^{-1/2} · X_K
```

Coefficients (Table 2): `{(7.425, -18.396, 12.897), (3.488, -2.330, 0.440), (2.777, -2.071, 0.463), (1.991, -1.374, 0.388), (1.875, -1.251, 0.375), (1.875, -1.250, 0.375), (1.875, -1.250, 0.375)}` — 7 iterations converge stably.

For our LoRA ranks (typically r ∈ [4, 64]), the Gram matrices are `r × r` — tiny. SIMD-only, no GPU needed.

---

## 2. Distillation for katgpt-rs (Modelless)

### 2.1 Newton-Schulz Inverse Square Root — `crates/katgpt-core/src/newton_schulz.rs`

**Status:** MISSING. Currently `newton_schulz.rs` only has `msign` (5-iter NS5) and `muon_update`. The inverse square root for PSD Gram matrices is needed for gauge-invariant updates and gauge rebalancing.

**Design:**
- Add `pub fn ns_inv_sqrt_psd(p: &[f32], r: usize, out: &mut [f32], n_iters: u8)` — 7-iter NS for `r × r` PSD matrix.
- Add zero-alloc variant `ns_inv_sqrt_psd_into(p, r, out, scratch, n_iters)` reusing `NewtonSchulzScratch`.
- Coefficients from paper Table 2 (7-iter default; configurable for r ≤ 16 where 5 iters suffice).
- Damping `γ = 1.001`, regularization `ε = 1e-5`.
- SIMD via existing `simd_dot_f32`, `simd_sum_sq`.

**Substrate routing:**
- `r ≤ 64`: CPU SIMD (current path, ~1KB scratch for r=16).
- `r > 64`: would route to GPU Muon kernels (`riir-gpu`), but this is out of scope for katgpt-rs — call site should fall back to dense if needed.

### 2.2 Gauge Rebalance — `crates/katgpt-spectral/src/gauge_invariant.rs` (new module)

**Pure inference-time primitive** — no training. Given `(A, B)` factor pair:

```rust
/// Rebalance (A, B) so σ_max(A) ≈ σ_max(B) without changing AB^T.
///
/// Paper Algorithm 2: c = (σ_max(B) / σ_max(A))^{α/2}, then A ← c·A, B ← B/c.
/// Gauge-invariant by Prop 1: P_A, P_B unchanged, so any downstream spectral
/// operation (msign, projector products) is unchanged.
///
/// `alpha ∈ (0, 1]` is a damping exponent; α=1 fully rebalances, α=0.5 damps.
pub fn gauge_rebalance(
    a: &mut [f32], b: &mut [f32],
    a_rows: usize, a_cols: usize,
    b_rows: usize, b_cols: usize,
    alpha: f32,
    power_iter_steps: u8,
)
```

- Uses `power_iterate` for `σ_max` estimate (existing pattern from `crates/katgpt-spectral/src/peira.rs::PowerIterationScratch`).
- ~O(r · (m+n)) per call — sub-microsecond for typical LoRA sizes (r=16, m=n=256).
- No allocations: caller-owned scratch buffers.

### 2.3 Gauge-Invariant Compose — `crates/katgpt-spectral/src/gauge_invariant.rs`

**Drop-in replacement for naive task-vector arithmetic** when composing trained LoRAs:

```rust
/// Compose multiple LoRA task vectors with gauge-invariant rebalancing.
///
/// For each pair (A_i, B_i):
///   1. Rebalance so σ_max(A_i) ≈ σ_max(B_i) — removes factorization artifact.
///   2. Apply composition weights η_i.
///   3. Sum: W_merged = Σ_i η_i · A_i · B_i^T
///
/// Without rebalancing, the merged result depends on the arbitrary
/// factorization of each input — e.g., if game_1's adapter was trained
/// with A scaled by 100 and B scaled by 0.01 (gauge c=100), naive sum
/// would weight game_1's contribution by 100² vs game_2.
pub fn gauge_invariant_compose(
    pairs: &[(f32, &[f32], &[f32])], // (eta, A, B) tuples
    shapes: &[(usize, usize, usize)], // (a_rows, a_cols, b_cols)
    out_a: &mut [f32],
    out_b: &mut [f32],
)
```

**Unblocks:**
- Plan 094 (Memo Reflections TIES Merge) — currently naive sum-of-deltas, gauge-dependent.
- Plan 201 (Rosetta Pruner) — cross-game adapter alignment assumes common gauge.
- Plan 233 (Rosetta Cross-Game Alignment) — same issue.
- SparseTaskVector composition — current `apply_to` is gauge-blind.

### 2.4 Substrate Routing (CPU/SIMD/GPU/ANE)

| Op | Size | Routing | Threshold |
|----|------|---------|-----------|
| `ns_inv_sqrt_psd` r ≤ 16 | 256 floats | CPU SIMD (f32 wide) | always |
| `ns_inv_sqrt_psd` 16 < r ≤ 64 | ≤ 16KB | CPU SIMD (existing `simd_dot_f32`) | always |
| `ns_inv_sqrt_psd` r > 64 | > 16KB | Delegate to GPU Muon (riir-gpu) via trait | rarely hit for game LoRA |
| `gauge_rebalance` | O(r·(m+n)) | CPU SIMD | always — sub-μs |
| `gauge_invariant_compose` | O(pairs · r · m · n) | CPU SIMD, Rayon for >4 pairs | `pairs.len() > 4` |

**ANE exclusion**: NS polynomial iterations are sequential matmul chains — no good ANE mapping. ANE is reserved for the *applied* adapter forward pass (existing `npc_ane_backend`), not the composition step.

### 2.5 Plasma/Hot/Warm/Cold Path

- **Plasma** (sub-μs, always-on): `gauge_rebalance` on cached adapter pair during hot-swap.
- **Hot** (1–10μs): `gauge_invariant_compose` per inference call when multiple adapters active.
- **Warm** (10μs–1ms): Adapter reload from disk → rebalance → cache.
- **Cold** (>1ms): Cross-rank LR sweep (training side, riir-ai).
- **Freeze**: Snapshot of rebalanced adapters to immutable storage (BLAKE3-hashed for chain provenance).

The rebalanced form is **deterministic** (given power_iter tolerance), so it's safe for sync/quorum — same input → same output across nodes.

---

## 3. Why This Is GOAT (Modelless Verdict)

| Criterion | Assessment |
|-----------|-----------|
| **Strengthens moat (engine)** | ✅ Yes — gauge-invariant adapter composition is a generic engine mechanic. Anyone deploying multi-adapter inference needs this. |
| **Zero training?** | ✅ Yes — all three primitives are pure inference-time matrix ops. |
| **Uses existing infra?** | ✅ Yes — `NewtonSchulzScratch`, `simd_dot_f32`, `PowerIterationScratch` pattern. |
| **Perf overhead** | ✅ Negligible — sub-μs for typical sizes, only on adapter swap or compose. |
| **Proof of gain** | ⚠️ Paper proves gauge-invariance theoretically + empirically (Fig 4: Spectron shifts by 2e-2, LoRA-Muon by 5e-5). Our gain is **adapter composition stability** — needs before/after benchmark on TIES merge. |
| **Risk** | Low. Even if composition gain is marginal, NS inv-sqrt is a needed primitive for any future Muon-family training. |
| **Unblocks** | Plans 094, 201, 233, and any future "compose N adapters" feature. |

**Verdict: GOAT — IMPLEMENT.** Three primitives, all modelless, all unblock downstream work. NS inv-sqrt is needed regardless (any future Muon training needs it). Gauge-invariant compose is the novel fusion.

---

## 4. What NOT to Implement (katgpt-rs)

- **LoRA-Muon optimizer itself** — that's training, goes to `riir-ai/.research/124` and Plan 299.
- **Cross-rank LR sweep harness** — training-side, needs GPU.
- **Schedule-Free interpolation** — already in AMUSE (Plan 149).
- **Heavy-tailed Σ^p correction** — already in HTMuon (Plan 177).
- **Layer-adaptive NS depth** — already in Spectral Budget Router (Plan 254).

---

## 5. Relationship to Existing Plans

| Plan | Relation | Impact |
|------|----------|--------|
| **152 (Newton-Schulz)** | Parent — extends `newton_schulz.rs` with `ns_inv_sqrt_psd` | Additive — no regression risk |
| **231 (SparseTaskVector)** | Consumer — composition becomes gauge-invariant | Optional: `SparseTaskVector::compose_gauge_invariant()` |
| **094 (Memo TIES Merge)** | Unblocks — TIES sign-election on gauge-rebalanced pairs | Quality improvement |
| **201 (Rosetta Pruner)** | Unblocks — cross-game alignment assumes common gauge | Correctness improvement |
| **233 (Rosetta Cross-Game)** | Unblocks — same | Correctness improvement |
| **254 (Spectral Budget Router)** | Composes — layer-adaptive NS depth can use inv-sqrt for Schatten-q norms | Future extension |
| **135 (Parallax Attn)** | Independent — uses NS5 for `W_R` conditioning, not LoRA factors | None |

---

## 6. Key Quotes

> "LoRA-Muon is obtained by applying Muon's spectral steepest-descent principle directly on the low-rank manifold. This viewpoint gives a simple projector-form update, explains why the update is invariant to arbitrary LoRA factorization choices."

> "The low-rank parametrization should not determine the optimizer geometry: for LoRA, the useful update is the one defined by the low-rank matrix manifold itself."

> "The same factor-coordinate operation is nearly invisible to LoRA-Muon but visible to Spectron, matching the gauge-sensitivity analysis."

> "LoRA-Muon stays QR-free and GEMM-heavy, whereas a QR-coordinate realization requires repeated QR factorizations that typically want float32-stable arithmetic."

---

## TL;DR Summary

LoRA-Muon paper derives a gauge-invariant Muon variant for LoRA training. For katgpt-rs (modelless), the novel distillation is **gauge-invariant adapter composition** — apply the paper's scalar gauge rebalancing (Algorithm 2) before any task-vector arithmetic. Combined with the missing Newton-Schulz inverse square root primitive, this gives us:

1. **`ns_inv_sqrt_psd`** — generic PSD inverse square root (needed primitive).
2. **`gauge_rebalance`** — pure inference-time factor pair conditioning.
3. **`gauge_invariant_compose`** — drop-in for naive adapter arithmetic.

All three are engine plumbing (WHAT, not HOW). The specific LoRA training recipe is `riir-ai/.research/124`.

**Verdict: GOAT — IMPLEMENT (Plan 270).** Unblocks Plans 094, 201, 233.
