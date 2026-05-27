# Research 122: EDGE-OPD — Evidence Guided On-Policy Distillation

> **Paper:** [EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation](https://arxiv.org/abs/2605.23493) — Lazaridis et al., 2026
> **Date:** 2026-05-27
> **Related Plans:** None (verdict: no standalone gain)
> **Related Research:** R036 (ROPD), R038 (SDAR), R117 (GKD)

## Executive Summary

EDGE-OPD addresses a specific failure mode of On-Policy Self-Distillation (OPSD): when privileged context contains rare tokens the student never samples, standard OPSD cannot transfer them. The paper proposes two modifications: (a) guided rollouts that inject privileged context at sampling time, and (b) a hard positive-evidence mask that trains only on tokens where the privileged context raises the sampled token's log-probability.

**Verdict: NO STANDALONE GAIN for our system.**

Our existing SDAR sigmoid gate (Plan 072, Research 038) already implements a softer, continuous version of the evidence masking idea. Our ROPD rubric vectors (Plan 071, Research 036) already provide per-criterion credit assignment. The guided-rollout technique is specific to rare-token identity internalization, which is not a game AI use case. The evidence mask is a hard binary version of SDAR's soft sigmoid gate — and our ablations show the soft gate is sufficient.

However, the paper provides **valuable diagnostic tools** (kept-token fraction, leverage fraction, agreement rate) that could enhance our SDAR modelless infrastructure.

---

## Paper Core

### Problem

OPSD fails when the student never samples the target behavior (rare-token problem). Even with privileged context in the teacher, if the student's on-policy trajectories never visit tokens the privileged context would produce, there's no gradient signal to transfer.

### Solution: EDGE-OPD

1. **Guided rollouts** — Sample ρg=0.5 fraction of rollouts with privileged context attached:
   ```
   πb(·|x,r) = ρg · πT(·|x,r) + (1-ρg) · πS(·|x)
   ```
   This ensures rare target behavior appears in on-policy data.

2. **Positive-evidence mask** — For each sampled token, compute evidence ratio:
   ```
   e_t = log πT(yt|x,r,y<t) - log πT(yt|x,y<t)
   ```
   Only include token in loss if `e_t > τ` (τ=0):
   ```
   L_EDGE-OPD(θ) = E[Σ 1{e_t > 0} · log πθ(yt|x,y<t)]
   ```

3. **KL anchor** — βKL=0.05 to frozen base policy for capability preservation.

### Key Formulas

**Evidence ratio (per-token privileged information gain):**
```
e_t = log πT(yt|x,r,y<t) - log πT(yt|x,y<t)
```

**Eligibility mask:**
```
m_t = 1{e_t > τ}    (stop-gradient, τ=0)
```

**Guided behavior policy:**
```
πb(·|x,r) = ρg · πT(·|x,r) + (1-ρg) · πS(·|x)
```

**Token-level diagnostics:**
- ρ+ = kept-token fraction (Pr[e_t > τ])
- ρlev = leverage-token fraction (Pr[|exp(e_t)-1| > 0.05])
- ρagree = agreement rate (Pr[sign(e_t) = -sign(δ_t)])

---

## Main Results

### Identity/Persona Axis (Nemotron-3-Nano-4B)

| Method | ID Self-Name ↑ | Persona Self-Name ↑ | AIME25 ↑ |
|--------|---------------|---------------------|----------|
| Base | 0.000 | 0.000 | 0.531 |
| OPSD (unguided) | 0.000 | 0.000 | 0.517 |
| Guided OPSD (user) | **0.667** | **0.688** | 0.544 |
| RLSD-no-verifier (guided) | 0.625 | 0.646 | **0.569** |
| EDGE-OPD (user) | 0.562 | 0.583 | 0.556 |

Key finding: **Guided rollouts are the bottleneck**, not masking. Every guided variant learns the identity.

### Mask-Region Ablation (Critical Insight)

| Mask Region | ID Self-Name ↑ | ID Counter-Name ↓ | AIME25 |
|-------------|---------------|-------------------|--------|
| Positive (e_t > 0) | **0.500** | **0.104** | 0.556 |
| Negative (e_t < 0) | 0.000 | 0.583 | 0.517 |
| Near-zero (|e_t| ≤ 0.1) | 0.000 | 0.708 | 0.508 |

Only positive-evidence positions transfer identity. This localizes the persona signal.

### Math Axis (Negative Result for EDGE-OPD)

Positive-evidence masking **hurts** math reasoning (0.392 vs 0.531 base). The mask selects answer-revealing tokens rather than transferable strategies. Near-zero masking preserves base score (0.583).

---

## Mapping to Our System

### What We Already Have (SDAR covers this)

| EDGE-OPD Concept | Our Equivalent | Relationship |
|------------------|----------------|--------------|
| Evidence ratio e_t | SDAR gap signal Δ_t | Same formula: log πT(with ctx) - log πT(without ctx) |
| Hard positive mask | SDAR sigmoid gate σ(β·Δ_t) | EDGE-OPD uses hard binary mask; SDAR uses soft sigmoid. SDAR is strictly more general. |
| Guided rollouts | — | Not applicable: game AI doesn't have rare-token identity problem |
| KL anchor (β=0.05) | Our KL regularization in GRPO | Same concept, already implemented |
| Kept-token fraction ρ+ | — | **Gap**: diagnostic not implemented. Could enhance SDAR modelless. |
| Leverage fraction ρlev | — | **Gap**: diagnostic not implemented. Could enhance SDAR modelless. |
| Agreement rate ρagree | — | **Gap**: diagnostic not implemented. Could enhance SDAR modelless. |

### Why No Standalone Gain

1. **SDAR gate is strictly more general than EDGE-OPD mask.** σ(β·Δ) smoothly interpolates between full inclusion (β→0) and hard mask (β→∞). EDGE-OPD's hard mask is a special case of SDAR's soft gate at β→∞. Our β=5.0 is already near the hard-mask regime for negative-evidence tokens.

2. **Guided rollouts solve a problem we don't have.** Game AI doesn't need to internalize rare identities. Our self-play explores the full action space; there's no "rare token" equivalent. Bomber/Go/TFT action spaces are bounded and fully reachable by the student policy.

3. **Math-axis failure validates our SDAR approach.** EDGE-OPD's hard mask fails on reasoning tasks because it selects answer-revealing shortcuts. SDAR's soft gate avoids this by not completely suppressing negative-evidence tokens — it attenuates them, preserving the reasoning trace structure.

4. **Our ROPD rubric vectors provide better credit assignment.** EDGE-OPD's binary mask is 1D (privileged vs non-privileged). Our RubricVector provides multi-dimensional per-criterion credit assignment, which ROPD proves is superior (3.6× wider discrimination margin).

### What We Should Take (Diagnostic Value)

The three token-level diagnostics (ρ+, ρlev, ρagree) are lightweight, interpretable metrics that could enhance our SDAR modelless distillation:

1. **Kept-token fraction (ρ+)** — Measures what fraction of tokens the SDAR gate actively promotes. If ρ+ is very high (>0.9), the gate isn't doing anything useful. If very low (<0.1), the teacher signal is misaligned.

2. **Leverage fraction (ρlev)** — Measures what fraction of tokens have meaningful gate magnitude. Complementary to ρ+.

3. **Agreement rate (ρagree)** — Measures how often the SDAR gate direction agrees with the K1 (OPD) gradient direction. High agreement = gate is redundant. Low agreement = gate is correcting.

These could be added to `SdarGatedAbsorbCompress` as optional diagnostics behind the existing `sdar_gate` feature flag — **no new feature gate needed**.

---

## Verdict

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| New feature gate | ❌ No | SDAR already covers the core idea |
| New plan | ❌ No | No standalone gain beyond SDAR diagnostics |
| Diagnostic enhancement | ⬜ Maybe | ρ+/ρlev/ρagree could be added to SDAR infra if benchmarking shows value |
| Guided rollouts | ❌ No | Not applicable to game AI action spaces |
| Hard evidence mask | ❌ No | SDAR soft gate is strictly more general |
| Super-GOAT potential | ❌ No | Standard technique, no competitive advantage |

**Bottom line:** EDGE-OPD validates that SDAR's approach (evidence-gated distillation) is sound. The paper proves evidence direction matters and positive evidence localizes transferable signal — which is exactly what our SDAR sigmoid gate captures. No new implementation needed.

---

## References

- Lazaridis, A. et al. (2026). EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation. arXiv:2605.23493.
- Our SDAR implementation: Plan 072 (modelless), Research 038 (analysis)
- Our ROPD implementation: Plan 071 (modelless), Research 036 (analysis)
- Agarwal, R. et al. (2024). GKD: Generalized Knowledge Distillation. NeurIPS 2024. [Reference for OPD/K1 estimator]
