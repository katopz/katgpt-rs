# Research 114: AMUSE — Anytime Muon with Stable Gradient Evaluation

> **Source:** [AMUSE: Anytime Muon with Stable Gradient Evaluation](https://arxiv.org/pdf/2605.22432) — Jueun Kim, Jihun Yun, Minhak Song, Baekrok Shin, Beomhan Baek, Chulhee Yun (KAIST / KRAFTON), 2026
> **Code:** https://github.com/kjeiun/amuse
> **Raw reference:** `.raw/amuse/` (full Python reference implementation + Llama 124M/720M training scripts)
> **Date:** 2026-05-26
> **Related Research:** 004 (LoRA Architecture), 037 (REAP Model-Based/Modelless), 054 (ASFT), 062 (SHINE), 059 (MoE Speculative), 097 (Training-Free Loop)
> **Related Plans:** 152 (katgpt-rs — Newton-Schulz + river-valley diagnostics), 149 (riir-ai — AMUSE game LoRA optimizer)
> **Related MMO Pillars:** Pillar 2 (WASM Validators — LoRA quality improves validator confidence scores), Pillar 3 (NPC Dialog — per-NPC LoRA adapters trained faster)

---

## TL;DR

AMUSE combines **Muon** (matrix orthogonalization via Newton-Schulz) with **Schedule-Free** (interpolated gradient evaluation) and a **time-varying β schedule** to create an optimizer that requires no learning rate decay, supports anytime training, and consistently beats AdamW, Muon, and SF-AdamW across vision tasks and LLM pretraining (124M–1B). The key insight: Muon accelerates progress along the "river" (bulk subspace) but amplifies oscillations along "valley walls" (dominant subspace). Schedule-Free averaging stabilizes these oscillations by evaluating gradients closer to the averaged trajectory.

**Why it matters to us:** Our riir-gpu LoRA training pipeline currently uses standard AdamW. AMUSE could:
1. **Accelerate game LoRA convergence** — 1.5–3× fewer steps to reach Muon's final performance
2. **Remove LR schedule tuning** — one less hyperparameter to tune per game domain
3. **Improve modelless→model-based transition** — better LoRA = better Pillar 2 validator confidence scores, better Pillar 3 NPC personality adapters
4. **Complement existing losses** — orthogonal to ASFT (anchor loss), NITP (representation geometry), SDAR (gated distillation) — they compose

---

## 1. Key Ideas

### 1.1 River-Valley Loss Landscape

Neural network Hessians have a **low-rank spectral structure**: a few large outlier eigenvalues (dominant subspace = valley walls) and a vast bulk of small eigenvalues (bulk subspace = river). Training progress happens along the river; valley-wall components cause oscillations.

**Formal decomposition (Definition 3.1):**
- Top-k eigenvectors → dominant subspace Sk(θ)
- Remaining → bulk subspace S⊥k(θ)
- r_dom(v; θ) = ∥P_k v∥ / ∥v∥ (alignment ratio with dominant)
- r_bulk(v; θ) = ∥P⊥_k v∥ / ∥v∥ (alignment ratio with bulk)

### 1.2 Why Muon Works (and Why It Oscillates)

**Muon's update:** Mt = μMt-1 + Gt, then Wt+1 = Wt − η·O(Mt) where O is Newton-Schulz orthogonalization.

- ✅ Orthogonalization **increases bulk component** → faster river progress
- ❌ Orthogonalization is **non-selective** → also amplifies noisy dominant components
- ❌ Valley-wall oscillations persist throughout training

**Evidence:** Muon's r_dom is much lower than SGD/AdamW, but post-orthogonalization still retains nonzero dominant components that cause bouncing.

### 1.3 Schedule-Free Stabilization

SF evaluates gradients at an **interpolation** between fast sequence zt and averaged sequence xt:

- yt = (1 − β)zt + βxt (gradient evaluation point)
- zt+1 = zt − η·O(Mt) (base update)
- xt+1 = (1 − ct+1)xt + ct+1·zt+1 (averaged inference point)

**Key finding:** Gradients evaluated closer to xt have **lower dominant components** → less noise before orthogonalization → more stable updates.

### 1.4 AMUSE: Time-Varying β

The problem with fixed β:
- Small β → good early (fast adaptation) but bad late (oscillations)
- Large β → stable late but too slow early (xt hasn't reached the river)

**AMUSE's solution** (Eq. 5): βt = 1 − ((T₀−1)/(t−1))^ρ · (1−β₁)

- Start with small β₁ for fast early adaptation
- Gradually increase toward 1 for late-stage stability
- ρ controls the speed of transition (0 = fixed β, 1 = constant averaging window)
- **Independent of total training horizon** → true anytime training

### 1.5 Memory Overhead

AMUSE requires exactly **one extra state copy (zt)** compared to vanilla Muon — same memory as AdamW/SF-AdamW. The averaged sequence xt is computed on-demand from zt and the running average.

---

## 2. Experimental Results

### 2.1 LLM Pretraining (FineWeb-100B, Llama-style)

| Model | AMUSE vs Muon | AMUSE vs SF-AdamW | AMUSE vs AdamW |
|-------|---------------|-------------------|----------------|
| 124M | −0.6 ppl | −1.5 ppl | −4.5 ppl |
| 720M | 1.51× fewer steps to Muon's final perf | − | − |
| 1B | Lowest throughout | Lowest throughout | Lowest throughout |

### 2.2 Vision Tasks

| Task | AMUSE vs Muon (steps saved) |
|------|------------------------------|
| ImageNet ResNet-50 | 1.12× fewer steps |
| ImageNet ViT MAE fine-tune | 3.08× fewer steps |
| CIFAR-10/100, SVHN, ISIC | Best throughout |

### 2.3 Hyperparameter Sensitivity

- ρ = 0.8 consistently good across all scales
- β₁ ∈ {0.4, 0.6} sweep shows mild variation
- **Every AMUSE config beats tuned Muon with cosine decay**
- Only 2 new HPs: β₁ and ρ

### 2.4 Wall-Clock

AMUSE is ~1.067× AdamW per iteration (Newton-Schulz overhead). Same overhead as Muon. The step savings (1.5–3×) far outweigh the ~7% per-iteration cost.

---

## 3. Distillation to Our Stack

### 3.1 katgpt-rs (Open, MIT)

**What goes in katgpt-rs:**

| Component | Type | Feature Gate | GOAT Path |
|-----------|------|-------------|-----------|
| Newton-Schulz orthogonalization | Infrastructure | `newton_schulz` | GOAT proved 25/25 (Bench 050) |
| River-valley diagnostic metrics | Modelless | `river_valley` (opt-in) | GOAT proved 25/25 (Bench 050) |
| Dominant/bulk subspace projection | Diagnostic | `river_valley` | GOAT proved 25/25 (Bench 050) |
| Muon momentum buffer | Infrastructure | `newton_schulz` | GOAT proved 25/25 (Bench 050) |
| Schedule-Free interpolate/average | Infrastructure | `schedule_free` | Convergence trajectory tracking |

**Why katgpt-rs:** These are generic optimizer building blocks. Newton-Schulz is a standalone matrix operation. River-valley diagnostics apply to any training. The open engine ships trait definitions + generic defaults.

### 3.2 riir-ai (Private, Game-Specific)

**What stays in riir-ai (super GOAT, selling point):**

| Component | Type | Feature Gate | GOAT Path |
|-----------|------|-------------|-----------|
| AMUSE optimizer (full βt schedule) | Model-based | `amuse_optimizer` | LoRA training convergence benchmark |
| Game-specific β₁/ρ tuning per domain | Domain knowledge | — (compiled in) | Per-game convergence curves |
| Muon+AMUSE hybrid for LoRA matrices | Training pipeline | `amuse_optimizer` | Bomber/Go/FFT LoRA quality |
| Per-game warmup schedule optimization | Domain knowledge | — (compiled in) | Warmup step sensitivity grid |
| River-valley landscape for game LoRA diagnostics | Model-based | `amuse_optimizer` | Training health monitoring |

**Why riir-ai:** The AMUSE time-varying β schedule with game-specific tuning (which β₁ works for Bomber LoRA vs Go LoRA) is accumulated domain knowledge. The per-game warmup schedules are private game data. Nobody can replicate Bomber validator LoRA convergence without knowing the right AMUSE hyperparameters.

### 3.3 Mapping to MMO GOAT Pillars

| Pillar | How AMUSE Helps |
|--------|-----------------|
| **Pillar 2: WASM Validators** | Better LoRA → higher Plan 045 A/B scores (LoRA+WASM vs WASM-only gap widens) |
| **Pillar 3: NPC Dialog** | Per-NPC LoRA adapters converge faster → more NPC personality variety |
| **Pillar 1: Fourier Spatial** | Indirectly — Go LoRA quality affects AutoGo distillation accuracy |
| **Pillar 4: Frame Sampling** | No direct impact (pure algorithmic) |

**Pillar impact assessment:** AMUSE primarily strengthens the **model-based secondary bets** (SHINE, D2F, Fourier-AHLA) rather than the 4 modelless pillars. But Pillar 2 and 3 each have an optional LoRA layer that AMUSE accelerates.

---

## 4. Algorithm Pseudocode (Rust Translation Target)

```
AMUSE State:
  - z: anchor parameters (same shape as weights)
  - m: Muon momentum buffer (for matrix params)
  - v: AdamW second moment (for non-matrix params)
  - weight_sum: f64 (running sum for averaging)
  - k: step counter

Per step:
  1. t = k + 1
  2. lr = base_lr * min(1, t / warmup_steps)     // warmup only, no decay
  3. weight = (t^r) * (lr^2)
  4. ckp1 = weight / (weight_sum + weight)         // averaging coefficient
  5. βt = compute_beta1(t, ckp1, warmup_steps)     // time-varying
  6. y = (1 - βt) * z + βt * x                     // gradient eval point
  7. grad = ∇L(y)
  8. For matrix params (use_muon):
     a. m = μ*m + grad                              // momentum
     b. O = newton_schulz5(m)                       // orthogonalize
     c. update = 0.2 * sqrt(max(dim0, dim1)) * O   // scaling
     d. z = (1 - lr*wd) * z - lr * update           // base update
  9. For non-matrix params (AdamW fallback):
     a. v = β2*v + (1-β2)*grad²                    // second moment
     b. denom = v / (1-β2^t) + eps                  // bias correction
     c. update = grad / denom
     d. z = z - lr * update                          // base update
 10. x = (1 - ckp1) * x + ckp1 * z                  // average update
 11. y_next = (1 - βt) * z + βt * x                 // rebuild for next step
```

---

## 5. Verdict

**ADOPT — HIGH VALUE for riir-ai, INFRASTRUCTURE for katgpt-rs.**

| Aspect | Rating | Reasoning |
|--------|--------|-----------|
| Training quality improvement | ⭐⭐⭐⭐⭐ | 1.5–3× fewer steps, consistently best across all benchmarks |
| Implementation complexity | ⭐⭐⭐ | Medium — Newton-Schulz is ~30 LOC, but Schedule-Free state management + train/eval mode switching needs care |
| Memory overhead | ⭐⭐⭐⭐⭐ | Same as AdamW (1 extra copy) — no issue for our LoRA pipeline |
| Risk | Low | AMUSE degrades gracefully to Muon (ρ=0) or SF-AdamW (no Muon params) |
| Game LORA impact | ⭐⭐⭐⭐ | Directly improves the thing Pillar 2/3 optionally depend on |
| Super GOAT potential | ⭐⭐⭐ | Not a pillar itself, but amplifies existing model-based bets |

**Priority:** Implement Newton-Schulz + river-valley diagnostics in katgpt-rs (infrastructure). Then wire AMUSE optimizer into riir-gpu as feature-gated alternative to AdamW (Plan 149, riir-ai). The game-specific β₁/ρ tuning data stays private.

**Risk mitigation:** If AMUSE doesn't improve game LoRA convergence, we still gain:
1. Newton-Schulz as a standalone matrix operation (useful for other things)
2. River-valley diagnostics for understanding why LoRA isn't converging
3. Schedule-Free as a drop-in AdamW replacement (no LR schedule needed)

---

## 6. Open Questions

1. **LoRA-specific behavior:** The paper trains full models. Does AMUSE's bulk-oriented update help LoRA (low-rank) parameters where the effective dimension is small?
2. **Game domain warmup:** Our game LoRA training is typically 50-200 steps. Is AMUSE's warmup schedule well-suited for such short horizons?
3. **Newton-Schulz on GPU:** The 5-iteration Newton-Schulz is a CPU-side operation in the reference. We need a wgpu compute kernel for it.
4. **Interaction with ASFT:** ASFT anchors prevent catastrophic forgetting. AMUSE's schedule-free averaging also acts as implicit regularization. Do they conflict or compose?
5. **Interaction with NITP:** NITP adds an auxiliary cosine loss. Does AMUSE's bulk-oriented update amplify or suppress the NITP gradient direction?
