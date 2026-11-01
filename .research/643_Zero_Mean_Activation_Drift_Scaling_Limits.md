# Research 643: Zero-Mean Post-Activations & the Scaling-Limit Verdict (Shaped Activations / Shaped Attention / CompleteP)

> **Source:** Mufan Li, "Lecture Notes on Neural Network Scaling Limits" (Waterloo STAT 946, draft 2026-09-05, 98pp, https://mufan-li.github.io/files/Lecture%20Notes/nn_scaling.pdf) · Li/Nica/Roy, "The Neural Covariance SDE" [arXiv:2206.02768](https://arxiv.org/abs/2206.02768) (NeurIPS 2022) · Noci/Li/He/Hofmann/Maddison/Roy, "The Shaped Transformer" [arXiv:2306.17759](https://arxiv.org/abs/2306.17759) (NeurIPS 2023) · Dey et al., "Don't be lazy: CompleteP" [arXiv:2505.01618](https://arxiv.org/abs/2505.01618) (NeurIPS 2025)
> **Date:** 2026-11-01
> **Status:** Active
> **Related Research:** 120 (PowLU — activation-growth cousin), 632 (Softmax Fix — the FP-evaluation sibling), 266 (FPRM — rank-collapse in looped transformers), 487 (Massive Activations), 131/315/582/371 (scaling-laws / mean-field / DMFT cousins)
> **Related Plans:** — (filed as issue: riir-train Issue 661 — shaped/centered init probe)
> **Cross-ref (riir-train):** Issue 661
> **Classification:** Public

---

## TL;DR

The owner shared the "zero-mean activation bug" material: ReLU/SiLU/GeLU have non-zero mean post-activations (E[ReLU(g)] = 1/√(2π) ≈ 0.399), and the claim is this causes loss spikes + representation overshooting, understood via scaling limits. Verdict after distillation + adversarial panel + verdict round 1: **mechanism CONFIRMED at the kernel level, framing REFINED (round-1 correction), scope BOUND.** The nonzero mean is real and injects a common-mode (DC) component — uncorrelated inputs become correlated after ONE ReLU layer (T(0) = 1/π ≈ 0.318) and attention weights that sum to 1 per query average tokens into doubly-exponential rank collapse — but the precise statement of the fix is subtler than "make the mean zero": critical exponent-½ shaping makes the activation NEAR-LINEAR at rate O(1/√n), and the mean → 0 as a CONSEQUENCE (E[ϕs(g)] ≈ ϕ''(0)/(2a√n) → 0), together with every other nonlinear kernel term. Zero-mean is emergent from the fix, not an independent knob; centering alone removes one term of many. No serving-path adoption exists (frozen checkpoints ⇒ G5 byte-parity forbids); the binding surfaces are init/training-time (riir-train Issue 661) and the substrate connection: the DC component the papers subtract **is exactly katgpt-dec's `harmonic_projector`** (rank-0 harmonic = mean·𝟙, test `rank0_harmonic_is_mean`).

**Distilled for katgpt-rs (modelless, inference-time):**
Closed-form, deterministic, no-training items: (1) the **stability constant** `b = (¾ϕ''(0)² + ϕ'''(0))/a²` with the if-and-only-if explosion law P(blowup) = 0 ⟺ b ≤ 0 (notes Prop 5.11) — a per-activation screening table computable from derivatives at 0; (2) the **mean-constant table** for any activation under N(0,1) (exact: ReLU 1/√(2π), GeLU 1/(2√π); numeric: SiLU ≈ 0.2066); (3) **shaped attention** `A_s = I_m + Softmax(s⁻¹S) − (1/m)11ᵀ`, s = s0·√(n·n_k) — a one-line zero-mean operator; (4) the **covariance-drift monitor**: estimate the SDE drift b̂ from per-layer Gram off-diagonals across a few forwards, extrapolate collapse depth. All are init/architecture/diagnostic-time primitives — never retrofittable onto frozen checkpoints as behavior changes.

---

## 1. Core Findings (the three papers + notes)

| Object | Result | Mechanism |
|---|---|---|
| Notes ch.2 (NNGP/NTK) | Infinite-width at init = GP with deterministic kernel; NTK regime freezes features | Kernel recursion Φ'₊₁ = C_ϕ(Φ') |
| Notes ch.4 (μP/mean-field) | Feature learning requires the γₙ=√n, η∝n balance; maximal update = every layer moves Θ(1) | Scaling-triad balance |
| Notes ch.5 (proportional d/n→τ̄) | Per-layer O(1/√n) fluctuations ACCUMULATE over depth into the **Neural Covariance SDE** dΦ = b(Φ)dτ + Σ(Φ)^½dB | Finite-width effects are invisible at fixed depth and O(1) at proportional depth |
| ReLU unshaped (5.4) | Correlation update **T(ρ) > ρ strictly**, unique fixed point 1 ⇒ correlation collapse; T(0)=1/π | E[ϕ(z₁)ϕ(z₂)] carries the common-mode mean c² |
| Fix: shaping (5.4/5.5) | ϕs(x)=s₊max(x,0)+s₋min(x,0), s±=1+c±/√n (critical exponent ½), or smooth ϕs(x)=s·ϕ(x/s), s=a√n ⇒ kernel map = identity + O(1/n) drift | The exponent kills the O(1) per-layer bias before it accumulates |
| Explosion law (Prop 5.11) | Scalar limit dΦ = b·Φ(Φ−1)dτ + √2·Φ dB, b = (¾ϕ''(0)²+ϕ'''(0))/a²; finite-time explosion possible **iff b > 0** | Closed-form if-and-only-if |
| Shaped Transformer | Unshaped softmax attention rank-collapses doubly exponentially with depth (attention weights sum to 1 per query ⇒ each output token is a convex average of value vectors); shaped A_s = I + Softmax(s⁻¹S) − 11ᵀ/m ⇒ stable SDE at depth-time | Uniform-centering = zero-mean projection of the attention operator |
| CompleteP | Depth-wise LR HP transfer AND non-lazy feature learning in all layers; 12–34% compute-efficiency gain over μP baselines | Per-layer parameterization exponents; kills per-layer lazy regimes |

## 2. Owner-claim adjudication

**CONFIRMED (the mechanism is real):** post-activation means are exact nonzero constants (table §4.1); at the kernel level the mean injects a rank-one common-mode component — ReLU maps uncorrelated inputs to correlated outputs in one layer (T(0) = c²/E[ϕ²] = (1/2π)/(1/2) = 1/π); unshaped attention's per-query normalization is a pure averaging operator. In the proportional depth–width regime these produce correlation/rank collapse and unstable init-time covariance; trained-stack loss spikes are PLAUSIBLY related (the papers' own results are init-time SDE statements — see BOUND).

**REFINED (round-1 verdict correction — the "zero-mean" framing and my first nuance were both off):** the exact statement is that critical exponent-½ shaping makes the activation NEAR-LINEAR at rate O(1/√n), and the mean → 0 as a consequence, not by a separate centering step. Taylor expansion with ϕ(0)=0, ϕ'(0)=1: E[ϕs(g)] = s·E[ϕ(g/s)] = ϕ''(0)/(2s) + O(s⁻³) = ϕ''(0)/(2a√n) → 0 (SiLU numeric check: 1/(4a√n) at ϕ''(0)=½ matches to 4 decimals at n=64/1024/16384). The piecewise ReLU-like form behaves identically: mean = (c₊−c₋)/(√(2π)·√n). My draft claim that smooth shaping "leaves the mean invariant" was wrong — it held only for positively-homogeneous ϕ, where s·ϕ(x/s) ≡ ϕ(x) identically (which is exactly why ReLU is shaped via the piecewise s± form instead: the smooth form would be a no-op). Centering alone is still insufficient — T̃(ρ) ≠ ρ for centered ReLU without slope/exponent tuning — so the faithful summary is: **the pathology is an O(1) depth-accumulating kernel bias carried by the activation's nonlinear structure (the mean is its most visible component); the fix is critical-rate near-linearization, which zeroes the mean along with every other nonlinear term.** Uniform-centering in attention is the case where the fix genuinely IS mean subtraction — because there the offending operator component is exactly the constant matrix. Two pre-registered falsification arms (restated round 1): (a) SHAPED + O(1) CONSTANT ADDED BACK (ϕs + c) — isolates whether the mean alone reintroduces the drift ("shaped-but-uncentered" as such does not exist — a shaped activation is already ≈ centered); (b) unshaped-but-centered still collapses ⇒ "centering suffices" refuted.

**BOUND (where it binds for us):** the SDE regime is initialization-time, proportional depth–width. Our serving surface consumes frozen checkpoints (byte-parity/G5 law) — a shaped activation or centered attention CHANGES the forward function, so it is not adoptable there at any price (lossy-surface rule). It binds where WE construct models: distillation-student init, QAT dynamic range (RMSNorm re-centers activations per block but does NOT tame pre-softmax logit growth or quant dynamic range), and any future from-scratch run. The model-based advocate's honest line, adopted: at ≤~50 layers + pre-norm/RMSNorm, most mean-drift is pre-neutralized at fine-tune scale; "fixes loss spikes" at our scale is overstated — the binding claims are init stability of deep students and quant-outlier control.

## 3. Distillation (Path 0 inventory — panel-merged)

| # | Component | Modelless-extractable? | Home / disposition |
|---|---|---|---|
| M1 | Stability constant b(act) + iff explosion law | YES — closed form from derivatives at 0; deterministic screen | Candidate katgpt-core table (see §4.1); doc-grade now |
| M2 | Critical-exponent activation shaping | YES at construction (freeze/thaw-safe NEW snapshots); NOT retrofittable to frozen checkpoints | riir-train init recipe → **Issue 661** |
| M3 | Shaped attention A_s = I + Softmax − 11ᵀ/m | YES at construction; runtime correction on frozen checkpoints = LOSSY surface ⇒ only via the lossy-surface promotion rule (per-family retention), entropy-gated, demote-only | Recorded; POC-gated if ever |
| M4 | Mean-constant table per activation | YES — offline constants | §4.1 (carried in this note) |
| M5 | Covariance-drift monitor (b̂ from Gram off-diagonals → collapse-depth extrapolation) | YES — forward-pass measurement only; sigmoid-gated readout | Candidate diagnostic; no consumer today |
| T1 | Shaped/centered init for students + QAT | — (training track) | riir-train **Issue 661** (Path 0.5: applicable, ~5–15 GPU-h probe) |
| T2 | CompleteP depth-wise LR transfer | — | Recorded, NOT filed: binds only where depth is a swept training axis; our training is fine-tune/distill on fixed archs; full pretraining is out of scope per the fusion ladder. The 12–34% headline is pretraining-scale. |
| T4 | GRPO logit-scale anchoring | — | **Discarded** — theory is init-time, RL spikes are optimization dynamics; no evidence at our scale (auditable reason: mechanism-level mismatch, not cost) |
| M6 | CompleteP-as-laziness-routing for overlay placement | Weak | **Discarded** — routing table with no consumer; re-opens only with an overlay-placement consumer |

### 4.1 The constants (coordinator-verified; §1.5 discipline)

| Activation | E[ϕ(g)], g~N(0,1) | b = (¾ϕ''(0)²+ϕ'''(0))/a² | Status |
|---|---|---|---|
| ReLU | 1/√(2π) ≈ 0.399 | n/a (kink — rides the piecewise s± branch; drift ν(ρ) = ((c₊−c₋)²/2π)(√(1−ρ²)−ρ·arccos ρ) ≥ 0) | exact, hand-verified |
| GeLU (tanh/erf form) | 1/(2√π) ≈ 0.282 | **3/(2πa²) ≈ 0.477/a² — EXACT** (xΦ(x) = x/2 + x²/√(2π) + O(x⁴) ⇒ ϕ''(0)=2/√(2π), ϕ'''(0)=0; verified at verdict round 1) | mean exact (erf integration by parts); b exact |
| SiLU | **0.2066** (numeric: ±12σ Simpson + Stein check E[xσ(x)] = E[σ'(x)] = E[σ(1−σ)]; no clean closed form) | 3/(16a²) (hand-verified: ϕ''(0)=½, ϕ'''(0)=0) | b exact; mean numeric (two independent methods agree) |
| tanh | 0 (odd) | −2/a² < 0 — the only drift-stable one in the smooth family | exact, hand-verified |
| identity | 0 | 0 — neutral | trivial |

⚠ Panel corrections, provenance chain: the No-GD advocate first claimed E[SiLU]=E[GeLU]=1/(2√π); the coordinator's re-derivation corrected the DISAGREEMENT (SiLU ≠ GeLU) but its own numeric 0.23 was coarse-grid; verdict round 1 pinned 0.2066 (±12σ Simpson + Stein identity, coordinator-reconfirmed). The b>0 reading for SiLU/GeLU does NOT mean "SiLU explodes when you serve it" — it is the shaped-family residual drift sign in the covariance SDE at init; tanh-class (b≤0) is the never-exploding class.

## 4. Fusion

1. **DEC/harmonic (`katgpt-dec/src/hodge.rs::harmonic_projector`)** — for rank-0 on a connected complex, harmonic = mean·𝟙 (test `rank0_harmonic_is_mean`). The papers' uniform-centering `(1/m)11ᵀ` IS the harmonic projector on the token complex; zero-meaning any activation field = applying I − harmonic_projector. If an entropy-gated centered-attention correction (M3) is ever built, express it as harmonic projection consuming katgpt-dec — a latent-space op consistent with the manifold-geometry rules, not an ad-hoc mean subtraction. (DEC earns its keep on structured complexes; for plain attention it is the principled NAME for the operator.)
2. **Softmax doctrine layering (Research 632 / Issue 959)** — two orthogonal layers of the same DC story: 632 fixes the floating-point EVALUATION of softmax (collapse/cancellation); shaped attention fixes the OPERATOR STRUCTURE (the constant component of the output distribution). Orthogonal, composable. ⛔ Round-1 correction: this material is NOT doctrine ammunition for sigmoid-vs-softmax — the Shaped Transformer keeps softmax and merely centers it, and a sigmoid-based attention (also non-negative) would carry the same DC component and need the same centering. The fixed-CE-gradient-is-a-sigmoid connection stands on 632's own math, not here.
3. **PowLU (Research 120) relative placement** — PowLU fixes activation GROWTH (quadratic → linear asymptotics ⇒ outlier channels ⇒ FP8 loss spikes); this material fixes activation DRIFT STRUCTURE (mean/common-mode ⇒ depth-accumulating kernel bias). Same consumer class (training stability), disjoint mechanism. 120's "we don't train models" rationale is now stale (riir-train trains: ternary distillation, LoRA, GRPO, encoders) — the init-recipe surface (Issue 661) is where both would land if ever adopted.

## 5. Verdict

**Gain** (not GOAT: no serving-path change is adoptable — frozen checkpoints; not Super-GOAT: no new product capability class; the falsification-arm table + constants + Issue 661 are the durable outputs).

| Track | Verdict | Disposition |
|---|---|---|
| (a) modelless inference | Gain-as-record | This note IS the artifact: constants, b-criterion, fusion map. M1/M5 primitive candidates parked here pending a consumer (Ungoat discipline: no runnable proof path ⇒ no plan). |
| (b) self-adaptive runtime | POC-gated, demote-only | M3 runtime correction = lossy surface ⇒ lossy-surface promotion rule (per-family conditional retention) before any wiring; entropy-gated, `off` byte-identical. |
| (c) trained weights | Gain → Issue | **riir-train Issue 661**: shaped/centered init probe (student init + QAT outliers), ~5–15 GPU-h, G1 = teacher-KL/eval at equal FLOPs vs baseline init, G3 = flag-off byte-identical. |

**PoC:** none run — this note makes no quality-parity claim; the pre-registered falsification arms (§2) are the proof path for the adjudication, and Issue 661 carries the training-track gate.

**PASS-Redirects:** n/a (Gain verdict; cousin map lives in §4).

---

## Summary

**(1) Original task:** distill the owner-shared scaling-limits material (lecture notes + arXiv:2206.02768 / 2306.17759 / 2505.01618) claiming non-zero-mean post-activations are a "bug" causing loss spikes and representation overshooting, and route it per the three-track system.
**(2) Accomplished:** full-text distillation of all four sources; adversarial panel (No-GD + model-based advocates) merged with coordinator spot-checks; **verdict round 1 (REVISE) applied**: the mean-invariance nuance was WRONG and is corrected (shaping drives the mean to zero at the critical rate — zero-mean is a consequence of the fix, not an independent knob); falsification arm (a) restated as shaped + O(1) constant added back; E[SiLU] pinned 0.2066 (Stein + fine Simpson, coordinator-reconfirmed); GeLU b marked EXACT at 3/(2πa²); softmax direction fixed (per-query rows); "loss spikes" downgraded to plausibly-related; sigmoid-doctrine overreach deleted. Verdict **Gain**: this note (643) + riir-train Issue 661 (shaped/centered init probe); substrate fusion pinned (harmonic_projector = the DC operator; Research 632 layering; Research 120 placement); allocation gates clean (643 / 661).
**(3) What remains:** Issue 661 probe unexecuted (owner call — GPU budget); M1/M5 primitive candidates need a consumer before any plan; the two falsification arms (§2, restated) unrun — running them on a small depth-sweep would upgrade the adjudication from analysis to measurement; highwater 641→643 stale-counter repair carried in this commit.
**(4) Active plan state:** `katgpt-rs/.research/643_Zero_Mean_Activation_Drift_Scaling_Limits.md` (Active); `riir-train/.issues/661_shaped_centered_init_probe.md` (Open, POC candidate, unclaimed).
