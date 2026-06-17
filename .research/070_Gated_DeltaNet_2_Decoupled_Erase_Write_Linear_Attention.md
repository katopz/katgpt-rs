# Research 70: Gated DeltaNet-2 — Decoupled Erase and Write in Linear Attention

> **Paper:** [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/pdf/GDN2_paper.pdf) — Hatamizadeh, Choi, Kautz (NVIDIA), May 2026
> **Code:** [github.com/NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) (PyTorch + Triton)
> **Date:** 2026-05, distilled 2025-07
> **Related Research:** 28 (HLA), 061 (Delta Attention Residuals), 22 (Lighthouse Attention), 42 (SP-KV), 24 (δ-Mem), 48 (HRM-Text)
> **Related Plans:** 104 (Gated DeltaNet-2 Recurrent Attention)
> **Verdict: COMPLEMENTARY TO HLA — GDN2 solves a different problem than our HLA/AHLA: memory-editing vs. memory compression. The channel-wise erase gate `b_t` (key-axis selective erasure) is the key innovation, accounting for ~90% of GDN2's gains over KDA. Best fit: riir-ai LoRA training pipeline where recurrent layers replace full attention for long-context. CPU SIMD recurrent decode is feasible for katgpt-rs. Feature-gate as `gdn2_attention` alongside `hla_attention`. Not a replacement for HLA — both address different linear attention axes.**
>
> **Cross-reference (2026-06-17, Plan 287):** GDN2's decoupled erase/write duality (`b_t` for keys = erase = suppress; `w_t` for values = write = broadcast) is the **linear-attention analog** of Research 258's NOP/Broadcast duality for softmax attention. NOP sinks (suppress residual) ↔ erase gate; Broadcast sinks (rank-1 write of load-bearing global info) ↔ write gate. The sink-aware classifier (`sink_aware_attn`, Plan 287) ships the softmax-side equivalent: classify whether a softmax sink is erasing (NOP) or writing (Broadcast), then gate accordingly. The two mechanisms are duals across the softmax/linear attention boundary.

---

## TL;DR

Gated DeltaNet-2 (GDN2) decouples the scalar delta-rule gate `β_t` into two channel-wise gates:
- **Erase gate `b_t ∈ [0,1]^{d_k}`** — selects which *key-side* coordinates to remove from the decayed state
- **Write gate `w_t ∈ [0,1]^{d_v}`** — selects which *value-side* coordinates to commit

This replaces KDA's tied scalar `β_t` (one number controls both erase and write) with independent per-channel control. The erase gate alone recovers ~90% of GDN2's gains — the key insight is that **selective key-axis erasure** is what matters most for long-context retrieval under fixed-state memory.

**Core recurrence (Gated Delta Rule-2):**
```
S_t = (I − k_t (b_t ⊙ k_t)ᵀ) Diag(α_t) S_{t−1} + k_t (w_t ⊙ v_t)ᵀ
```

Where:
- `α_t ∈ (0,1]^{d_k}` — channel-wise decay (from KDA)
- `b_t ∈ [0,1]^{d_k}` — erase gate (new, key-axis)
- `w_t ∈ [0,1]^{d_v}` — write gate (new, value-axis)
- `S_t ∈ R^{d_k × d_v}` — fixed-size recurrent state (constant memory)

**Recoveries:** b_t = β_t·1, w_t = β_t·1 → KDA. Further α_t = α_t·1 → Gated DeltaNet.

**Results at 1.3B / 100B tokens FineWeb-Edu:**
- Language modeling: Wiki PPL 15.62 (hybrid), best-in-class
- Long-context retrieval (RULER): S-NIAH-3 @2K = 99.0%, MK-NIAH-1 @4K = 48.0% (hybrid)
- Real-world retrieval: 42.28% avg (hybrid), best-in-class
- Throughput: 36.1 Kt/s at 8K seq (H100), near-flat scaling

---

## Core Mechanism

### The Problem: Scalar Tie Between Erase and Write

In KDA/Gated DeltaNet, one scalar `β_t` controls two distinct operations:

| Operation | Axis | What it does |
|-----------|------|-------------|
| **Erase** | Key-side | How much of the old read `(S_{t-1})ᵀ k_t` to remove |
| **Write** | Value-side | How much of the new value `v_t` to commit |

These live on **different axes** of the state matrix `S_t ∈ R^{d_k × d_v}`:
- Erasing is a *key* operation: "which coordinates of the old association at key `k_t` should I clear?"
- Writing is a *value* operation: "which coordinates of the new value should I store?"

A single scalar cannot independently control both.

### Gated Delta Rule-2: The Fix

```
e_t = b_t ⊙ k_t     (gated erase direction)
z_t = w_t ⊙ v_t     (gated write content)

S̄_t = Diag(α_t) S_{t-1}                          (decay)
r_t = S̄_tᵀ e_t                                    (read old content along gated key)
S_t = S̄_t + k_t (z_t − r_t)ᵀ                     (delta update)
```

Equivalent compact form:
```
S_t = (I − k_t (b_t ⊙ k_t)ᵀ) Diag(α_t) S_{t-1} + k_t (w_t ⊙ v_t)ᵀ
```

**Key structural properties:**
1. Left factor of erase remains `k_t` (write direction preserved from delta rule)
2. Right factor becomes `b_t ⊙ k_t` (channel-selective read direction)
3. Write term becomes `k_t (w_t ⊙ v_t)ᵀ` (channel-selective value insertion)
4. Channel-wise decay `Diag(α_t)` from KDA retained

### Fast-Weight Update View

GDN2 is one online gradient step on the local loss:
```
L_t(S) = ||S − S̄_t||²_F − 2⟨Sᵀk_t, z_t − S̄_tᵀ e_t⟩
```

Minimizer: `S_t = S̄_t + k_t(z_t − S̄_tᵀ e_t)ᵀ`

This is exactly Eq. 9 — the delta between the gated write target and the gated read.

### Chunkwise Parallel Training (WY Form)

GDN2 preserves KDA's efficient chunkwise structure via decay normalization:

```
Define: γ_r = exp(Σ_{i≤r} g_i)  (cumulative decay)
Normalize: k̄_r = γ_r⁻¹ ⊙ k_r, ē_r = γ_r ⊙ (b_r ⊙ k_r)
WY solve: A = (I + T)⁻¹, T = tril(ĒK̄ᵀ, -1)
Auxiliaries: Y = AĒ (erase), U = AZ (write, Z = W ⊙ V)
Output: O[n] = Q_γ S[n] + A_qk (U − YS[n])
```

The channel-wise decay is absorbed into the rank-one erase factors. Only `Y` and `U` differ from KDA — the matrix shapes are identical.

**Gate-aware backward** is the key training change: gates must be inside the dot products that accumulate `dA`, not post-scaled outside. Scalar post-scaling is invalid when erase/write are different diagonal operators per row.

---

## Key Results

### Language Modeling & Commonsense Reasoning (1.3B)

| Model | Wiki PPL ↓ | LMB PPL ↓ | LMB Acc ↑ | Avg Acc ↑ |
|---|---|---|---|---|
| **Recurrent** | | | | |
| Mamba-2 | 16.79 | 12.38 | 45.24 | 51.82 |
| Gated DeltaNet | 16.40 | 11.89 | 49.62 | 52.07 |
| KDA | 16.81 | 11.68 | 48.13 | 52.28 |
| Mamba-3 (MIMO) | 16.45 | 11.66 | 47.82 | 52.39 |
| **GDN2** | **15.90** | **11.41** | 48.09 | **53.11** |
| **Hybrid (+ SWA)** | | | | |
| Gated DeltaNet | 16.00 | 10.82 | 48.71 | 52.25 |
| KDA | 16.01 | 10.66 | 49.21 | 52.68 |
| Mamba-3 (MIMO) | 15.81 | 10.92 | 49.82 | 52.72 |
| **GDN2** | **15.62** | **10.43** | **50.90** | **53.97** |

### Long-Context Retrieval (RULER)

| Model | S-NIAH-2 @4K | S-NIAH-3 @2K | MK-NIAH-1 @4K |
|---|---|---|---|
| **Recurrent** | | | |
| Gated DeltaNet | 87.2 | 54.2 | 27.8 |
| KDA | 89.0 | 63.2 | 28.0 |
| Mamba-3 (MIMO) | 64.2 | 72.4 | 18.0 |
| **GDN2** | **93.0** | **89.8** | **37.8** |
| **Hybrid** | | | |
| Gated DeltaNet | 57.3 | 91.2 | 44.8 |
| KDA | 56.0 | 93.4 | 40.4 |
| Mamba-3 (MIMO) | 53.0 | 98.4 | 46.6 |
| **GDN2** | **57.9** | **99.0** | **48.0** |

### Gate Structure Ablation (Critical Finding)

| Variant | Wiki PPL | Avg Acc | S-NIAH-3 @2K | MK-NIAH-1 @4K |
|---------|----------|---------|---------------|----------------|
| w-only (scalar b, channel w) | 16.55 | 52.45 | 71.4 | 30.6 |
| **b-only (channel b, scalar w)** | **16.12** | **52.79** | **84.6** | **35.2** |
| Full GDN2 (channel b + w) | **15.90** | **53.11** | **89.8** | **37.8** |

**Key insight:** The erase gate `b_t` (key-axis) accounts for ~90% of the gain. The write gate `w_t` adds the remaining ~10%. This makes sense — selective key-side erasure directly addresses interference in compressed memory.

### Expanded Erase Range Ablation

| Variant | Wiki PPL | Avg Acc |
|---------|----------|---------|
| b_t ∈ [0,1]^{d_k} (standard) | **15.90** | **53.11** |
| b_t ∈ [0,2]^{d_k} (neg eigval) | 15.95 | 53.04 |

No consistent gain from expanded range at 1.3B scale. Negative eigenvalue trick may help at larger scales.

### Throughput (H100, Hybrid 1.3B)

| Seq Length × Batch | Transformer | KDA | GDN2 |
|-------------------|-------------|-----|------|
| 2K × 8 | ~12 Kt/s | 38.0 | 38.0 |
| 8K × 2 | ~3 Kt/s | 37.0 | 36.1 |

GDN2 retains near-flat scaling with only ~5% constant overhead vs KDA for the added channel-wise gates.

---

## Comparison with Our Stack

### GDN2 vs HLA/AHLA: Different Problems

| Aspect | HLA/AHLA (Research 28) | GDN2 |
|--------|----------------------|------|
| **State type** | Prefix sufficient statistics (KᵀK, QKV moments) | Delta-rule recurrent state (KV memory) |
| **Mechanism** | Higher-order moment compression | Targeted memory editing |
| **Decay** | Single scalar γ (global) | Channel-wise α_t per key channel |
| **Erase control** | None (additive accumulation) | Channel-wise b_t (key-axis selective) |
| **Write control** | None (raw q/k/v entered) | Channel-wise w_t (value-axis selective) |
| **State size per head** | O(d²) (symmetric) / O(d·dv) (AHLA) | O(d_k × d_v) (standard linear attn state) |
| **Best for** | Small d, quality-critical | Large d, retrieval-critical |
| **Training** | Trained from scratch | Trained from scratch |
| **Composable** | Yes, orthogonal | Yes, orthogonal |

**They are NOT competing approaches.** HLA compresses the attention *mechanism* (higher-order statistics). GDN2 improves the attention *update rule* (selective memory editing). Both replace softmax attention with linear-time recurrence.

### Where GDN2 Fits in Our Architecture

| Path | Current | GDN2 Role |
|------|---------|-----------|
| **katgpt-rs CPU inference** | SDPA / HLA / AHLA / SP-KV | Recurrent decode for long-context (O(1) per step) |
| **riir-ai GPU LoRA training** | SDPA / wgpu attention | Recurrent mixer replacing full attention for long-context layers |
| **Hybrid models** | D2F + SDPA | GDN2 + SWA (same hybrid pattern as paper) |

### Structural Alignment

| GDN2 Concept | Our Implementation | Status |
|-------------|-------------------|--------|
| Recurrent state `S_t ∈ R^{d_k × d_v}` | AHLA `PKV ∈ R^{hd × hd}` | ✅ Similar structure |
| Channel-wise decay `Diag(α_t)` | HLA `gamma: f32` scalar | ⚠️ Need per-channel |
| Erase gate `b_t ∈ [0,1]^{d_k}` | N/A | ❌ New |
| Write gate `w_t ∈ [0,1]^{d_v}` | N/A | ❌ New |
| L2 normalization on q, k | N/A | ❌ New (we use RMSNorm) |
| SiLU output gate | N/A | ⚠️ We have SiLU in MLP |
| Short causal convolution | N/A | ❌ New |
| WY chunkwise solve | N/A | ❌ Training-only, GPU |

---

## Extractable Techniques

### E1: Channel-Wise Erase Gate (High Value)

**What:** `b_t = σ(W_b x_t) ∈ [0,1]^{d_k}` — independent projection + sigmoid, applied elementwise to key before reading old content.

**Why valuable:** This single mechanism accounts for ~90% of GDN2's gains over KDA. It directly addresses the core problem of fixed-state recurrent attention: interference among many compressed associations.

**Implementation for CPU SIMD (katgpt-rs):**
```rust
// In recurrent decode step:
// 1. Project erase gate: b = sigmoid(w_erase @ x)  → d_k values
// 2. Compute gated key: e = b * k  (elementwise)
// 3. Read old: r = S^T @ e  (matvec, d_v values)
// 4. Delta: S += k ⊗ (w*v - r)  (outer product update)
```

**Cost per token:** One extra matvec (n_embd → d_k) + elementwise multiply. ~O(n_embd × d_k) FLOPs. For micro config (n_embd=48, d_k=4): 192 FLOPs. Negligible.

### E2: Channel-Wise Decay (Medium Value)

**What:** `α_t = exp(g_t)` where `g_t = -exp(a) ⊙ softplus(W_f x_t + δ)` — per-key-channel decay rate.

**Why valuable:** Finer-grained forgetting than our HLA's single scalar γ. Each key channel can have its own retention horizon.

**Caution:** Our HLA already uses scalar decay γ. Channel-wise decay adds `d_k` multiplications per step. For small `d_k` (4-16), negligible overhead.

### E3: Gated Output (Low Value for Us)

**What:** Recurrent output → RMSNorm → SiLU gate → output projection.

**Why low value:** We already have RMSNorm per sublayer. Adding a SiLU gate is a minor architectural change. Not the key innovation.

### E4: Recurrent Decode Kernel (Medium Value)

**What:** Token-by-token state update for autoregressive generation. No chunk parallelism needed.

**Why valuable:** This is the path we'd use for CPU inference. No Triton/WY needed — just a simple loop applying Eq. 10 token by token. Perfectly suited for our SIMD-optimized matvec.

**Implementation sketch:**
```rust
fn gdn2_recurrent_step(
    state: &mut [f32],           // S_t: d_k × d_v, stored row-major
    k: &[f32],                   // key: d_k
    v: &[f32],                   // value: d_v
    b: &[f32],                   // erase gate: d_k
    w: &[f32],                   // write gate: d_v
    alpha: &[f32],               // decay: d_k
    q: &[f32],                   // query: d_k (for readout)
    dk: usize, dv: usize,
) -> Vec<f32> {
    // 1. Decay: S *= Diag(alpha)
    for i in 0..dk {
        for j in 0..dv {
            state[i * dv + j] *= alpha[i];
        }
    }
    // 2. Read old content along gated key: r = S^T (b ⊙ k)
    let mut r = vec![0.0f32; dv];
    for j in 0..dv {
        for i in 0..dk {
            r[j] += state[i * dv + j] * b[i] * k[i];
        }
    }
    // 3. Delta update: S += k ⊗ (w ⊙ v − r)
    for i in 0..dk {
        for j in 0..dv {
            state[i * dv + j] += k[i] * (w[j] * v[j] - r[j]);
        }
    }
    // 4. Readout: o = S^T q
    let mut o = vec![0.0f32; dv];
    for j in 0..dv {
        for i in 0..dk {
            o[j] += state[i * dv + j] * q[i];
        }
    }
    o
}
```

**Cost per token:** O(d_k × d_v) for state update + readout. Same as standard linear attention. The erase/write gates add negligible overhead.

### E5: Hybrid Recurrent + SWA Architecture (Medium Value)

**What:** Alternate GDN2 mixer + MLP + Sliding Window Attention + MLP.

**Why valuable:** This is the configuration that achieves the best results (PPL 15.62, avg acc 53.97). SWA handles exact local interactions; GDN2 handles long-range compressed memory.

**Fit with our stack:** Our D2F (Plan 066) and TileRT (Plan 102) already provide block-structured attention. A hybrid GDN2 + local attention pattern fits naturally.

---

## What NOT To Do

1. **Don't replace HLA with GDN2.** They solve different problems. HLA compresses attention statistics; GDN2 improves memory editing. Both can coexist.
2. **Don't implement chunkwise WY training kernels in CPU Rust.** The WY forward/backward is GPU-specific (Triton). For CPU inference, token-by-token recurrent decode is sufficient.
3. **Don't implement the full GDN2 block design (short conv, SiLU gate, L2 norm) all at once.** Start with the core recurrence (erase gate + write gate + channel decay), validate on micro config, then add architectural components.
4. **Don't implement channel-wise decay before validating the erase gate alone.** The ablation shows b-only (with scalar decay) recovers 90% of the gain. Channel-wise decay is a second-order improvement.
5. **Don't train GDN2 models in katgpt-rs.** Training belongs in riir-ai's GPU LoRA pipeline. katgpt-rs should only implement the recurrent decode path for inference.

---

## Verdict

### Why COMPLEMENTARY, Not Replacement

| Factor | Assessment |
|--------|-----------|
| GDN2 vs HLA state size | GDN2: d_k × d_v per head. HLA: d² + d·dv per head. Same order for d_k=d_v=d. |
| GDN2 vs HLA mechanism | GDN2: delta-rule editing. HLA: moment compression. Different inductive biases. |
| GDN2 vs HLA best case | GDN2: long-context retrieval (interference control). HLA: infinite-context generation (compression). |
| Our HLA benchmarks | AHLA: 95% of flat KV speed, 88% less memory. Already proven. |
| GDN2's erase gate | Addresses interference in fixed-state memory — orthogonal to HLA's moment approach. |
| Training complexity | GDN2 requires Triton kernels (GPU only). HLA works with simple matmul. |
| Inference complexity | Both O(1) per token. GDN2 adds erase/write gate projections. |

### Adoption Strategy

| Phase | What | Where | Feature Gate |
|-------|------|-------|-------------|
| **Phase 1** | Recurrent decode only (E1: erase gate) | katgpt-rs | `gdn2_attention` |
| **Phase 2** | Full recurrent step (E1+E2+E4) | katgpt-rs | `gdn2_attention` |
| **Phase 3** | Training integration (chunkwise) | riir-ai | GPU kernel |
| **Phase 4** | Hybrid GDN2 + SWA (E5) | riir-ai | GPU kernel |

### Phase 1 Value Proposition

For micro config (hd=4, n_embd=48):
- **Extra state per head:** 4 floats (erase gate b_t) + 4 floats (decay α_t) = 32 bytes
- **Extra compute:** 1 matvec (48→4 = 192 FLOPs) + elementwise (4 muls) ≈ 200 FLOPs
- **Total overhead:** <1% of forward pass
- **Potential gain:** Better long-context retrieval in GDN2-trained models

### When to Implement

1. **Now (Phase 1-2):** If we want to run GDN2-trained models on CPU. The recurrent decode is simple SIMD matvec + elementwise.
2. **When scaling (Phase 3-4):** If riir-ai's LoRA training pipeline encounters long-context limits (seq_len > 4K). GDN2 hybrid is the SOTA recurrent architecture for this regime.
3. **For GOAT proof:** Feature-gate `gdn2_attention` allows benchmarking GDN2 recurrent decode vs HLA vs flat KV on our existing micro/small configs.

### When NOT to Implement

1. If we never train recurrent models from scratch (HLA/AHLA/GDN2 all need training).
2. If our context lengths stay under 2K tokens (standard attention is fine).
3. If we don't need the memory editing advantage (HLA's moment compression may suffice).

---

## Relationship to Existing Research

| Research | Overlap | Relationship |
|----------|---------|-------------|
| 28 (HLA) | Linear attention, O(1) cache | **Complementary**: HLA compresses moments, GDN2 edits memory. Different inductive biases. |
| 061 (Delta Attention Residuals) | Delta-based routing | **Orthogonal**: DeltAtt routes across layers; GDN2 edits within a layer's recurrent state. |
| 22 (Lighthouse Attention) | Efficient attention patterns | **Orthogonal**: Lighthouse restructures attention; GDN2 replaces attention with recurrence. |
| 42 (SP-KV) | KV cache optimization | **Orthogonal**: SP-KV prunes KV entries; GDN2 replaces KV cache with fixed state. |
| 24 (δ-Mem) | Delta-based memory updates | **Conceptual alignment**: Both use "read old, compute delta, write" pattern. GDN2 adds channel-wise gates. |
| 48 (HRM-Text) | Recurrent pretraining | **Complementary**: HRM is hierarchical recurrence; GDN2 is per-layer recurrence. Could combine. |
| 39 (SpectralQuant) | KV compression | **Orthogonal**: SQ quantizes KV cache; GDN2 replaces KV cache. |
| 55 (Tri-Mode Diffusion) | Hybrid inference modes | **Orthogonal**: Tri-Mode is about diffusion/AR switching; GDN2 is about attention replacement. |

---

## References

- Paper: Gated DeltaNet-2 (Hatamizadeh et al., 2026)
- Gated DeltaNet: [arXiv:2505](https://arxiv.org/abs/2505) (predecessor)
- Kimi Delta Attention (KDA): [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
- Mamba-2: [arXiv:2405](https://arxiv.org/abs/2405) (SSD framework)
- Mamba-3: Improved sequence modeling using SSM principles (2026)
- Delta Rule: Yang et al., "Parallelizing linear transformers with the delta rule" (2024)
- WY Representation: Bischof & Van Loan (1985)
- Negative Eigenvalues: [arXiv:2411.12537](https://arxiv.org/abs/2411.12537)