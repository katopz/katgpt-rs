# Research 488: Effective Degree — Function-Space Simplicity via Polynomial Representations

**Paper:** Zhang, Li, Xiao, Chen, Chen (Tsinghua). *Quantifying and Optimizing Simplicity via Polynomial Representations*. [arXiv:2605.29823](https://arxiv.org/abs/2605.29823), ICML 2026.
**Code:** https://github.com/xinzaixinzai/Effective-Degree
**Date:** 2026-08-17
**Status:** DISTILLED — **CLOSED.** Open primitive **SHIPPED opt-in** 2026-08-17 (`Issue 668` T1-T6 complete; GOAT record [Bench 665](../.benchmarks/665_effective_degree_goat.md), G1+G2+G3+G4 ALL PASS, feature `effective_degree`, default-OFF). Freeze-gate PoC **ANSWERED 2026-08-17 — SCOPE-LIMITED (Simpson reversal), no gate change** ([Bench 484](../../riir-neuron-db/.benchmarks/484_ed_vs_flatness_freeze_gate_poc.md); riir-neuron-db Issue 602 closed and removed). ED out-correlates the incumbent `output_flatness` 12.6× pooled (0.598 vs 0.047) but its sign inverts between the across-regime and within-regime grains, so it is NOT wirable as the proposed one-sided gate. Verdict stays **Gain**; the Super-GOAT path via the freeze gate is **closed**. Primitive stays opt-in as a regime diagnostic (§10, §4 KARC regime-mismatch probe). See **§10 PoC Addendum** for what transferred, what did not, and the new grain-dependent-sign risk.
**Classification:** Open (the metric is generic math, no game IP). The freeze-gate consumer story is private (riir-neuron-db).

---

## TL;DR

The paper approximates a network's predictive behavior along **data-dependent interpolation paths** (random input pairs, α ∈ [0,1]) using **orthogonal polynomial bases** (Chebyshev), and defines the **Effective Degree (ED)** of the fitted expansion — `ED(P) = Σ|c_k|·k` — as a function-space simplicity metric. Empirically ED correlates with the generalization gap better than sharpness/ASAM/L2-norm across ResNet/ViT/CLIP, and uniquely **tracks the grokking transition** (rises during memorization, peaks at the transition, decreases as the generalizing — simpler — solution emerges; sharpness measures do not). ED admits analytic gradients through the damped least-squares fit, yielding a differentiable regularizer (+3.0 CIFAR-10 ViT-Tiny, consistent CLIP/BERT/Procgen gains) at ~2× training cost.

**Distilled for the stack:** the *measurement* side is fully modelless — a frozen function (shard decode, KARC readout, adapter, NPC policy) can be probed by ED with zero training. Two of the three heavy components already ship: `karc::ChebyshevBasis` + the ridge solve (`katgpt-core/src/karc/`), and interpolation-path construction (`riir-neuron-db/src/interpolation_geometry/` — `LatentSpace::midpoint`, with a doc-comment that explicitly anticipates the richer v2 decode this note measures). The missing piece is ~150 LOC: the ED metric + randomized-cosine node sampling. The *regularizer* side is training-only → Path 0.5 recipe record (§7), no live consumer today.

**The falsifiable stack question** (PoC, riir-neuron-db Issue 602): `can_freeze`'s output side uses spectral `output_flatness < 0.3` — a parameter-space proxy of exactly the family the paper shows ED beats. Does ED (function-space, data-anchored) predict held-out shard-decode error better than flatness on our consolidation substrate? Neuron-db Bench 010 (Morris grokking) concluded the grokking framing "reduces to a convergence check that can_freeze's output_flatness already captures" — **in flatness terms**. This paper's headline is that the function-space family strictly dominates that proxy class. Until the PoC runs, any "ED upgrades the freeze gate" claim is architectural-only.

---

## 1. Paper core findings

### 1.1 Polynomial representations + ED

- Restrict `f: ℝ^d → ℝ^m` to 1-D interpolation paths `x(α) = α·x₁ + (1−α)·x₂` between data pairs (avoids the combinatorial blow-up of multivariate polynomial bases; anchors the estimate to the data distribution).
- **Theorem 3.1 (order preservation):** for multivariate polynomials, degree drops under path restriction occur only on a measure-zero set of interpolation directions — averaging over random paths preserves degree ordering almost surely. (Justifies the 1-D surrogate.)
- Fit a degree-K Chebyshev expansion `P(α) = Σ c_k·T_k(2α−1)` at r nodes; **ED(P) = Σ|c_k|·k** (coefficient-weighted degree; Lipschitz in coefficients, robust to fitting noise — unlike algebraic degree). Normalized variant `ED_norm = Σ|c_k|k / Σ|c_k|` is scale-invariant.
- Numerical stability: Chebyshev nodes (or **randomized cosine sampling** — stratified θ_i ~ U[(i−1)π/r, iπ/r], α = (1−cosθ)/2); orthogonal basis (Legendre works equally well); damped normal equations `(TᵀT + εI)c = Tᵀy`; optional per-path PCA output compression (not the source of the gains).

### 1.2 Measurement results (modelless half)

- ED vs generalization gap: strongest linear correlation across 27-config model pools (ResNet18 + ViT-Tiny on CIFAR-10; CLIP ViT-B/32 fine-tunes on ImageNet), beating sharpness/ASAM (which correlate *negatively* under mixup recipes) and L2 norm (negative).
- **Grokking** (modular division, ℤ97, 30%): ED peaks at the validation-loss drop and decreases after; parameter norm rises monotonically; sharpness decays early or fluctuates. Only ED gives a clean transition signal.
- Controlled PNN study (Appendix I): ED preserves ground-truth algebraic-degree ordering across basis change (Chebyshev/Legendre) and PCA reduction — the shipped G1 gate for our primitive mirrors this.

### 1.3 Regularizer results (training half — §7 here)

`L = L_task + λ·ED̂`, label-anchored boundary nodes for classification; efficiency config r=4, K=3, n_p=B/2 → exactly one extra batch of forward passes (~2× cost). Gains: ViT-Tiny CIFAR-10 87.80→90.82; ImageNet ViT-S/16 +1.39/+0.59 (both recipes); CLIP B/16+B/32 ID and all 5 OOD sets; BERT GLUE (small, consistent); Procgen PPO actor regularization (all 4 envs). **Failure mode (MNIST-CIFAR):** when the simpler feature is easier to exploit but less robust, ED does not help — both baseline and ED latch onto the simple feature. ED is a simplicity *enforcer/measurer*, not a feature-quality judge.

## 2. Path 0 decomposition (component inventory)

| # | Paper component | Stack analog | Status |
|---|---|---|---|
| 1 | Interpolation path construction (data pairs, α nodes) | `interpolation_geometry::LatentSpace::midpoint` (style weights); plain `lerp` for adapters/policies | **exists** (α=0.5 only — general-α + node sampling is new, trivial) |
| 2 | Chebyshev basis evaluation | `katgpt-core/src/karc::ChebyshevBasis<M>` | **exists** |
| 3 | Damped least-squares coefficient fit | KARC ridge solve (`KarcForecaster` — ridge λ ≡ damping ε) | **exists** |
| 4 | **ED = Σ\|c_k\|·k metric** | — | **missing** (zero-alloc, ~30 LOC) |
| 5 | Randomized cosine node sampling | — | **missing** (~20 LOC, stratified sampler) |
| 6 | Gradient ∂ED/∂y through the fit (regularizer) | — | **missing, training-only** → riir-train §7 |

Rows 1–3 exist; rows 4–5 are the cheap modelless delta (Issue 668); row 6 is the only GD-dependent piece.

## 3. Signal-diff vs closest shipped cousins (§3.6 defense)

| Cousin | Consumes | ED consumes | Diff |
|---|---|---|---|
| `karc::ChebyshevBasis` + ridge (Bench 308) | time-series delay window → **forecast** | cross-sectional data pairs → **measure complexity** | same basis machinery, orthogonal purpose; reuse only basis+solve |
| `can_freeze::output_flatness` (neuron-db, spectral_flatness.rs) | eigenvalue **spectrum of stored weights** (parameter-space, static, data-blind) | **predictive outputs along data-anchored paths** (function-space, distribution-aware) | the paper's core claim: this proxy class is what ED beats; flatness is reparameterization-fragile + architecture-dependent, ED is not |
| `interpolation_geometry` (neuron-db) | RavenSlot subsample-vs-average **divergence** (truthfulness audit); `midpoint` at α=0.5 | polynomial **degree of the decode map** along general-α paths | audit-vs-metric; complementary — its `decode` doc-comment (L93-96) explicitly plans the "v2 richer decode (KARC ridge readout…)" whose complexity ED measures |
| R284 `ComplexityProxy` (Dingle-Hutter) | K̃(x) of **objects** (RLE/L1/entropy — description length) | degree of **functions** (input→output behavior) | object-complexity vs function-complexity; composable — ED is a new, principled `ComplexityProxy` for callable things |
| R125 weight-norm = K | ‖θ‖ (theoretical sandwich bound) | function behavior | parameter- vs function-space; R125 was verdict "theoretical validation only" |
| HOPE 302 (HS capacity kernel) | kernel capacity of shard weights | function behavior on data paths | capacity-of-weights vs complexity-of-decode |

**Verdict: no shipped function-space simplicity metric exists.** Greps: `chebyshev|legendre` → KARC/DEC-quadrature/NCA-distance only; `grokking` → notes only; `mixup|interpolation_path` → none in training code; `sharpness|output_flatness|generalization_proxy` → sigmoid-gate sharpness + the flatness gate; `effective_degree|simplicity` → zero metric hits. Published prior art (web): only the paper itself and its own citations (SAM/ASAM, region counts, mixup, decision-boundary path sampling).

## 4. Latent-space reframing

- **Per-NPC KARC shard** (`karc_shard.rs`): ED of the stored readout = effective complexity of the NPC's forward model. Direct diagnostic for the documented KARC scope-limit (Bench 010: "Chebyshev basis + ridge-fit doesn't fit periodic data regardless of K") — high ED on wake-event data = the basis is straining = regime mismatch, *measured* rather than inferred from a CRPS loss. Latent-side scalar, never synced raw.
- **Freeze/thaw certification** (the PoC question): pre-freeze ED of the shard decode along `interpolation_geometry` paths as the output-side gate signal. Sync boundary respected — ED is computed locally from latent style weights + wake events; only a scalar verdict would ever be logged/committed.
- **Adapter/snapshot selection** (dMoE, Dynamic Pair, Bonsai comparison): ED as a validation-free, data-anchored complexity score per frozen candidate at swap time. Modelless selection prior — composes with R284's `CompressionPriorSampler` (sigmoid(−α·ED − β) as the sampling prior for *functions*).
- **Per-NPC policy monitor** (riir-ai): ED of `sense::evolve_belief` / swarm brain along state-interpolation paths = belief-map wiggliness. High ED = brittle/oscillatory cognition — a crowd-volatility scalar for consolidation-window gating (sleep when ED stabilizes low — the grokking-decrease analog).

## 5. Game-context reframing

- Per-entity bounded scalar: ED bounds "policy wiggliness along state paths" — an NPC whose action map needs degree-7 Chebyshev between nearby states is erratic; a crowd-average ED is a volatility index.
- Consolidation trigger: the paper's grokking signature (ED peaks then falls) maps to "the shard has stopped memorizing and started generalizing — safe to freeze." This is exactly the Morris-plateau question Bench 010 closed *in flatness terms*; ED is the stronger instrument the paper claims.
- Selling point candidate (pending PoC): "Shards that prove their own simplicity in **function space** before freezing — a data-anchored behavioral measurement, not a weight-norm heuristic." Upgrades the neuron-db R001 "self-certifying shard" story from spectral to functional.

## 6. Verdict — Gain (GOAT-tier open primitive; Super-GOAT pending PoC)

Novelty gate (§1.5), scored honestly:

1. **No prior art?** YES — in-stack greps clean (§3), published search returns only the paper's own lineage.
2. **New behavior class?** Conditional — a function-space measurement class is new to the stack, but the *behavior change* (better freeze timing) is a quality claim. §3.6: PoC mandatory before claiming.
3. **Product selling point?** Conditional on 2 — the sentence completes only if ED actually beats flatness on our substrate.
4. **Force multiplier?** YES — KARC basis, interpolation_geometry paths, freeze gate, adapter selection, R284 sampler.

Not all 4 unconditionally → **Gain**, not Super-GOAT (no "candidate" wording beyond this). If Issue 602's PoC confirms ED > flatness, the freeze-gate upgrade + runtime monitors re-opens the gate with the quality axis proven.

## 7. Path 0.5 — the regularizer (riir-train record)

Training-cost-weighted assessment: the ED regularizer is a real recipe (label-anchored, r=4/K=3/n_p=B/2, λ via sweep, sinusoidal ramp-up, ~2× cost) with consistent multi-modal gains. Our live training surfaces: quest_grammar LoRA (riir-ai), edge_lora trainers, gemma/kimi GPU paths, civ LEO (closed negative). None currently shows a generalization-gap complaint that ED would unblock; the measurement side is where this paper's value lands for us **first** (it gates whether the regularizer story even matters for shards). **Deferred with cause** — not a lazy redirect: Path 0 decomposition (§2) shows rows 1–5 are modelless; only row 6 (gradient) is training-side, and its consumer (training runs needing generalization help) does not exist today. Revisit trigger: any riir-train plan whose G5/generalization gate FAILs, or the Issue 602 PoC confirming ED is the freeze signal (making ED-regularized consolidation training the natural follow-up).

## 8. Connection map

- **Issue 668** (katgpt-rs) — **SHIPPED** (Bench 665): `effective_degree` open primitive — `EdConfig{r,K,ε,n_pairs,seed}` + `randomized_cosine_nodes` + `effective_degree_along_path` over caller-supplied outputs; reuses `karc::ChebyshevBasis`; feature `effective_degree`, opt-in; G1 = PNN degree-ordering preservation (paper Appendix I protocol), G2 = per-path latency, G4 = scratch alloc-free.
- **riir-neuron-db Issue 602**: defend-wrong PoC — ED vs `output_flatness` vs input-gate control on the Bench-010-style consolidation benchmark; ground truth = held-out wake-event decode error; feature `ed_freeze_poc`, revert after; three arms per §3.6.
- **riir-ai (post-PoC, if confirmed)**: per-NPC KARC ED diagnostic; crowd-volatility monitor; consolidation-window gate.
- **R284**: `EdComplexity` as a new `ComplexityProxy` impl for callable/function objects — the sampler gains a principled function-space arm.

## 9. Risks / honest caveats

- **MNIST-CIFAR failure class**: ED enforces simplicity, not robustness — a shard whose simple decode is wrong-by-simple will pass an ED gate. The two-sided conjunction (input sufficiency `n ≥ d` AND output simplicity) must stay; ED only upgrades the output arm, never replaces both.
- **Scale dependence**: raw ED scales with output magnitude (paper Table 12: ×2 outputs ≈ ×2 ED); use `ED_norm` or fit post-softmax/normalized outputs — the paper's regularizer fits post-softmax probabilities for exactly this reason.
- **Data-manifold dependence**: random-pixel endpoints destroy the signal (paper C.1: ED with random pixels = baseline accuracy). Paths must come from real wake events / real states — `interpolation_geometry`'s RavenSlot events are the right anchors.
- Cost: correlation-grade estimates used r=200/K=40/400 averages (expensive); regularizer-grade r=4/K=3 is cheap. The freeze gate needs the cheap end with enough pairs for stability — PoC must sweep this.

---

## 10. PoC Addendum — the freeze-gate question is ANSWERED (2026-08-17)

**Verdict: SCOPE-LIMITED (Simpson reversal). No gate change. ED ships as a regime
diagnostic, not a `can_freeze` arm.** Full measurement:
[riir-neuron-db Bench 484](../../riir-neuron-db/.benchmarks/484_ed_vs_flatness_freeze_gate_poc.md).
Issue 602 is closed and removed.

§"The falsifiable stack question" (TL;DR) and §6's conditional novelty items 2/3
resolve as follows — **both halves matter, and they point opposite ways**:

**Half 1 — the paper's headline transfers, and Bench 010's closure WAS
instrument-limited.** On 360 shard states (30 cycles × 4 scenarios × 3 seeds,
held-out wake-event recall error as ground truth), pooled |Pearson| with the
generalization gap: **`ed_norm` 0.598 vs `output_flatness` 0.047** — a 12.6×
advantage, against a distribution-matched control at 0.032 and a permutation
floor at 0.042. `ed_norm` beat flatness in **4/4** scenarios, raw and
cycle-controlled, and the advantage reproduced on 3 disjoint seed sets. Flatness
is statistically indistinguishable from noise as a gap predictor on this
substrate. Bench 010's "reduces to a convergence check `output_flatness` already
captures" was measured with a flatness-family instrument, and §"falsifiable stack
question" was right to flag that as suspect.

**Half 2 — but ED is not wirable as the proposed gate, for a reason §9 did not
anticipate.** ED's sign **inverts between grains** (Simpson's paradox, reproduced
on all 3 seed sets):

| grain | r(ed_norm, gap) | reading |
|---|---|---|
| pooled across regimes | **+0.598** | higher ED ⇒ higher gap (the paper's direction) |
| within a regime, cycle-controlled | **−0.18, −0.68, −0.35, −0.25** (all negative) | higher ED ⇒ *lower* gap |

A gate is one threshold on one shard state, so it must commit to a direction;
`ed_norm < τ_e` correctly rejects the memorizing regime wholesale while, inside
every regime, preferring exactly the shards with the **largest** held-out gap.
No τ_e fixes this — it is a property of the signal, not of calibration. So the
§"selling point candidate" sentence ("shards that prove their own simplicity in
function space before freezing") **does not complete**, and must not be used.

**Half 3 — the mechanism does NOT transfer, which qualifies Half 1 sharply.**
`ed_norm` is a magnitude-weighted mean of the degree index over *all*
coefficients, `k=0` included. Removing the DC term (`ed_ac`) collapses the
correlation to **+0.122 pooled** — below flatness-plus-noise and 5× below
`ed_norm`. Nearly all of ED's power here lives in `coeff_norms[0]`, which for the
cosine decode is the along-path *mean level* — i.e. **how well `style_weights` is
aligned to the event cluster**. That is alignment information wearing a
complexity costume. When the complexity-only component is isolated, it does not
beat the incumbent. **The paper's actual thesis (function-space complexity
predicts generalization) is NOT confirmed at shard scale**; what was confirmed is
that a data-anchored function-space *probe* out-predicts a data-blind
parameter-space one, for reasons partly incidental to ED's own theory.
(DC-term mechanism reported by the Issue 668 owner from that primitive's
Bench 665; it is why `ed_ac` was added as a 4th arm.)

### What this changes in this note

- §5 "Selling point candidate (pending PoC)" — **retired, do not use.** The
  freeze-gate framing is refuted as a *gate*.
- §4 "Freeze/thaw certification (the PoC question)" — **closed negative.**
- §4 "Per-NPC KARC shard" (regime-mismatch diagnostic) — **strengthened, and it
  is now the primary consumer story.** Cross-regime triage is precisely the grain
  where ED's sign is correct, and it is 12.6× better than the incumbent there at
  196 ns/path. A shard with high decode ED and low flatness is
  polynomial-basis-strained: Bench 010's documented KARC scope-limit made
  *measurable* rather than inferred from a CRPS loss.
- §6 novelty gate — item 2 (new behavior class) and item 3 (product selling
  point) **resolve NO** for the gate framing. Verdict stays **Gain**, not
  Super-GOAT. The Super-GOAT path via the freeze gate is closed; a future
  re-open would need a different consumer (adapter selection §4, crowd-volatility
  monitor §4) with its own PoC.
- §9 risks — **add a fourth**, now measured rather than hypothesized:
  *grain-dependent sign*. ED's association with generalization can reverse
  between the across-model and within-trajectory grains. Any future ED consumer
  MUST state which grain it operates at and verify the sign there. This is the
  generalizable lesson; it is not specific to the freeze gate, and it applies to
  the paper's own pooled 27-config correlation study, which measures only the
  across-model grain.
- §7 revisit trigger ("the Issue 602 PoC confirming ED is the freeze signal") —
  **did not fire.** The regularizer stays deferred with cause.

### Cheap config is validated (T4), so cost is not the limiter

`EdConfig::cheap()` (r=4/K=3, 8 paths) reaches 0.598 vs `precise()`
(r=15/K=7, 32 paths) at 0.623 — ~4% of the correlation for ~5× less work. Zero
ranking flips across `n_pairs ∈ {1..64}`, and per-path spread falls monotonically
8.8× (seed_std 0.0335 → 0.0038) as paths accumulate, confirming Theorem 3.1's
path-averaging prediction empirically. §9's cost caveat is discharged: the cheap
end is sufficient. What limits ED here is the *grain-dependent sign*, not r/K/pairs.
