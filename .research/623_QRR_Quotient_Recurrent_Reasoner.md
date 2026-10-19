# Research 623: QRR — Quotient Recurrent Reasoner (Rate-Inheritance Theorem + Symmetry-Modulo Convergence)

**Status:** GAIN — note + katgpt-rs `.issues/945` (modelless extraction POC) + riir-train `.plans/462` (training track). No code this session.

> **Source:** "Looped Reasoning Finishes Earlier Than You Think" — Sumin Park, T. Konstantin Rusch (ELLIS Tübingen / MPI / Liquid AI), [arXiv:2610.18174](https://arxiv.org/abs/2610.18174), 2026-10-13. Code: github.com/camail-official/Quotient_Recurrent_Reasoner_QRR — MIT, pinned `3e3c9ebecad534296482316e1c39f1063f0a6a32` (`.raw/qrr` clone, full-tree grep done, removed post-distill).
> **Date:** 2026, distilled 2026-10-16.
> **Related Research:** **626 (InfiLoop — loop-native attention residuals; the NEXT paper in this exact lineage: carry-last overwriting → content-gated N/D summary; supplies the content gate this note's magnitude-only commit law lacked)**, 620 (Mamba-3 oscillatory dynamics — the dominant cousin on the oscillation axis), 529 (HRM dissection — convergence cadence; "refinement is rotational, not radial"), 035 (Attractor Models / the FPRM line + Issue 717 damping knob), 079 (EqR), 049 (PTRM), 169/371 (oscillation architecture cousins), 051 (Deep Manifold fixed points).
> **Related Issues/Plans:** katgpt-rs Issue 939 (`phase_belief` — landed 2026-10-16, the fusion target), Issue 717 (damping/tangential knobs for `forward_looped` deep runs), riir-train Plan 459 (oscillatory GDN damping probe), Plan 373 (late-state looped solver / the Anderson-divergence negative).
> **Classification:** Public.

---

## TL;DR

Looped-reasoning latent trajectories oscillate instead of converging. The paper **proves** a slowly decaying oscillatory Jacobian mode rate-limits the whole coupled system (Theorem 3.2, via an exact Sylvester decomposition: the complement's settling time inherits the slower oscillatory rate `τ_δ ≈ log(1/δ)/(−log r)`; at `r = 1` the complement never converges), **verifies it causally** in FPRM/HRM by one-step amplitude interventions predicted from the root Jacobian alone, and then resolves the damp-vs-disrupt tension **architecturally**: a token-local `U(1)^S` gauge symmetry under which phase motion leaves the quotient state unchanged — so the raw trajectory keeps rotating while the quotient converges, and halting on the **quotient residual** fires at median step 27 vs FPRM's raw-residual 122 (4.5× earlier; end-to-end p95 2.8× faster).

**Modelless extraction (the paper's convergence laws need no training to USE):**
1. **Rate-aware oscillation verdict** — direction (cos reversal, what `GainCostLoopHalter` ships) × modulus `r = ‖δ_{t+1}‖/‖δ_t‖` recovered from the delta sequence by a 3-coefficient closed-form 2nd-order fit; flip-with-`r<1` = converging *through* the oscillation (never kill), flip-with-`r≈1` = limit cycle (stop wasting).
2. **Settling-budget predictor** — `(r, δ) -> steps` and its inverse, from Theorem 3.2; turns constant patience into a budgeted verdict.
3. **`r=1` non-convergence certificate** — bounded-but-never-settles is structural, not slow; two-signal deployable form (mode fit + complement stagnation).
4. **Quotient-style residual — TWO applications, do not confuse them** (verdict rounds 1–2): (a) gauge redundancy — the paper's case, where the phase carries NO information; closed form below. (b) known free dynamics — the `phase_belief` case, where the rotation phase IS the signal (θ FIXED per channel at `new`): the commit law is the write residual `r_free = ‖u_t‖/(‖z_t‖+ε)` computed directly from the write — committed when writes stop (persistence window W), never a phase-quotient. Also the guard: no quotient residual may be wired into `forward_looped`/`GainCostLoopHalter` — that state is not equivariant, and the paper's FPRM+momentum ablation (77.0 acc, 23% false halts) is exactly that failure class.
5. **fp32-rotation numerics law** — bf16 `cos²+sin² = 1±2⁻⁸` random-walks `‖y‖` ~13% over 1000 steps; every rotation-carrying primitive rotates in f32 internally.
6. **Tension law as a damping predicate** — the paper measured that damping a settling oscillation *degrades reasoning*; `can_damp` requires BOTH `r≈1` AND stagnant complement, sigmoid-composed.

**Training track (riir-train Plan 462):** learned-ζ momentum (`β = σ(b_H)`, **no weight decay on b_H** — WD pins it at the 0.5 init), Anderson-certification + Sylvester attribution as a zero-GPU eval upgrade for Plans 373/459, and the gauge×momentum 2×2 probe gated against sotaku's 99.12%@1024 (accuracy is NOT the claim — certified convergence at fewer inference iterations is).

**Verdict: GAIN, not Super-GOAT.** Every mechanism here is published mathematics (this paper, 3 days old); the claims are "first workspace consumer" + "modelless extraction of published convergence laws" — GOAT-shaped engineering claims, never novelty claims. Same posture as Issue 939 ("the mechanism is published mathematics … never a Super-GOAT").

---

## 1. Paper core findings

1. **Rate inheritance (Theorem 3.2 + Lemma 3.1).** Around a solution root `z*`, linearize `δ_{t+1} = Jδ_t`. Split the displacement into an oscillatory spectral coordinate `u_t` (left-invariant subspace `U`, reduced block `S = UᵀJU`, complex pair `re^{±iω}` or real flip `−r`) and the complement `e_t`. Exact Sylvester decomposition (`PS − AP = B`, unique when `σ(A)∩σ(S)=∅`): `e_t = Aᵗ(e₀ − Pu₀) + PSᵗu₀` — intrinsic transient vs coupled response. If `ρ(A) < r ≤ 1` and `B ≠ 0`, the complement's `lim ‖e_t‖^{1/t} = r > ρ(A)`: **the slow oscillatory mode imposes its rate on everything coupled to it**. At `r = 1`: bounded but never converges, `E(e) ≥ ‖u₀‖‖B‖_F/(2κ(M)(1+‖A‖²)) > 0`.
2. **Causal verification without training.** Intervene ONCE on the oscillatory amplitude (`α ∈ {1, ½, 0}`, project at one step), resume the ORIGINAL nonlinear recurrence, compare measured response to the closed form `α·QJᵏUu₀` and settling gains `Δτ` to the local prediction. Identity-line agreement on FPRM (alternating flip) AND HRM (2-D spiral) — the mode attribution is causal, not a spectral-fitting artifact.
3. **The tension law.** Suppressing the alternating mode in FPRM *before semantic stabilization degrades task performance* (App. A.4.4) — oscillatory motion may BE the computation. Damping decisions need evidence, not appearance.
4. **QRR architecture.** Token-local `U(1)^S` gauge: channels paired into complex `h_{s,c} = y_{2c−1} + i·y_{2c}`, one phase per token. Equivariant trunk (gauge attention with relative transport `U_{s←t} = γ_s γ_t*`, scores `Re(U_{s←t} q†k)/√d` invariant; G-SwiGLU `W_down[SiLU(|W_g h|) ⊙ W_v h]`; covariant input injection via phase carrier `D_s = ⟨W_d x_s, h_s⟩/|⟨W_d x_s, h_s⟩|`), INVARIANT readout, order-2 momentum `p = βm + (1−β)(z−y)`, `β = σ(b_H)`, and a parameter-free gauge rephasing `a = Im⟨y,F⟩/‖y‖²`, `R = e^{aJ}` (detached). Gauge directions are dynamical dead ends (`B_G = 0`): symmetry motion cannot force the quotient.
5. **Quotient convergence + halting.** `r_G(z) = inf_g ‖T(z) − g·z‖/(‖z‖+ε)` — halt when below threshold for a persistence window. Raw-state halting mistakes orbit motion for unfinished computation; quotient halting fires at 27 vs 122 median steps. Roots are *relative equilibria* (`T(z*) = g*·z*`), not fixed points.
6. **Results.** 7M params: Sudoku-Extreme **97.9** (HRM 55.4, TRM 75.7, EqR 89.9, FPRM 91.5), Maze-Unique 97.0, Mini-ARC pass@1 62.4, Maze-Hard 85.8. Root-finding τ median **122 vs 234** (Anderson-certified). Rollout p95 **2.8× faster** despite 1.5× per-LFWD cost (complex ops + phase transport).
7. **Ablations (the load-bearing parts).** Global `U(1)` instead of per-token: 97.3 acc but **halting NEVER fires** (token-locality is what quotient halting needs). Momentum removed: accuracy + coverage drop. **FPRM trunk + momentum (no equivariance): 77.0 acc, 23% false halts, halt median 12,650** — the quotient residual is only sound on an equivariant trunk; bolting it onto a non-equivariant loop produces confident wrong halts.
8. **Numerics law.** The rotation runs fp32 ALWAYS: at bf16 `cos²+sin² = 1±2⁻⁸`, `R` stops being orthogonal and `‖y‖` random-walks ~13% over 1000 steps. (Their `a` is also detached — `dL/da = 0` in exact arithmetic, so any gradient through it is roundoff noise.)

## 2. Path 0 inventory (per-track decomposition)

| Component | Value type | Track | Disposition |
|---|---|---|---|
| Theorem 3.2 settling law `τ_δ ≈ log(1/δ)/(−log r)` | closed form | (a) modelless | Extract → issue 945 P3 (budget predictor; calibration gate) |
| Sylvester decomposition `PS−AP=B` | closed form (offline linear solve, ~O(8k³) flops at complement dim k) | (a) | Extract → issue 945 P6 (report-only attribution) |
| δ-sequence 2nd-order mode fit → `(r, ω)` without any Jacobian oracle | derived predictor | (a) | Extract → issue 945 P1 (the unlock for 2/3/4/6) |
| Two-channel oscillation verdict (angle × modulus) | runtime signal | (a) | Extract → issue 945 P2 (GainCostLoopHalter upgrade) |
| `r=1` bounded-nonconvergence certificate | closed form + two-signal deployable form | (a) | Extract → issue 945 P3 (advisory certificate) |
| Quotient residual `r_G` (closed form, per-token phases in quadrature) | runtime signal | (a) | Extract → issue 945 P4 (phase_belief commit law; defer-until-pull aligned with Issue 939) |
| fp32-rotation numerics law | standing law + canary | (a) | Extract → issue 945 P5 (must-fail-when-broken arm) |
| Tension law → two-signal damping predicate | design law | (a) | Recorded in note; folds into P2's verdict semantics (never auto-damp) |
| α-intervention causal protocol | diagnostic methodology | (a) | Extract → issue 945 P6 (replay-only A/B; rides the determinism guarantee) |
| Gauge-equivariant trunk (GaugeConv / G-attn / G-SwiGLU) | architecture (trained) | (c) | riir-train Plan 462 Phase 2 treatment arm — NOT a drop-in anywhere |
| Momentum recurrence `β=σ(b_H)` no-WD | training recipe | (c) | Plan 462 Phase 1 → Plan 459 learned-ζ arm |
| Deep supervision + carry-`(y,m)`-across-segments BPTT | training recipe | (c) | Plan 462 Phase 1 recipe rows (Plan 373 delta) |
| Quotient halting as eval primitive (halt-or-cap) | eval protocol | (c) | Plan 462 Phase 2 (contingent on certified roots) |
| AdamAtan2 | optimizer | (c) | Plan 462 hygiene row (Paper 48 already flagged it; adopt-on-tie) |
| Sudoku/Maze-Unique/Mini-ARC fixtures | eval data | (c) | Plan 462 Phase 2 fixtures |

**Panel record:** two advocates ran (No-GD: 16 candidates; Model-based: 8 items), merged above. Discards, with reasons: (i) No-GD candidate 12 (fixed-β momentum channel for leaky beliefs) — the consumer belief kernels are convex EMAs bounded by construction (Research 620 §2 verified the same fact); direction-persistence is a real want but no loop in the workspace currently stalls for lack of it; recorded as available math in the note, not filed. (ii) No-GD candidate 10's strongest form (gauge rephasing at *similarity* sites) — conditional on explicit complex-pair structure that only `phase_belief` has today; the underlying need (phase-immune comparison) is folded into P4's residual, which compares moduli + alignment instead of raw dots. (iii) Model-based "carry-(y,m) BPTT" as a standalone arm — zero marginal cost, bundled into Plan 462 Phase 1 rather than a separate probe.

## 3. Signal-diff table (§3.6 — closest shipped cousins; each read at the formula level)

| QRR mechanism | Closest shipped analog | What the analog consumes | Delta (the gap) |
|---|---|---|---|
| Rate-aware oscillation verdict | `katgpt_core::gain_cost_halt::GainCostLoopHalter` (`HaltReason::Oscillation`) | cos_theta between successive steps (direction reversal) + patience + gain/cost scissors | **No rate channel.** Cannot distinguish flip-with-`r<1` (converging THROUGH the oscillation — the paper's measured failure mode is killing exactly this) from flip-with-`r≈1` (limit cycle). Two scalars replace one patience counter's verdict |
| Trajectory-geometry classification | `benches/bench_011`/`bench_013` (oscillation / committed_wrong / drift / stuck / converged_correct; curvature + length features, SWE patch trajectories) | turn-angle heuristics | No spectral attribution: "stuck" vs "orbiting" is `B≈0, r≈1` vs `r≈1, B≠0` — decidable only with the mode fit. Additive features, never replacing |
| Phase-carrying belief block | `katgpt_sense::phase_belief` (Issue 939, landed 2026-10-16) | per-channel `z = ρR(θ)z + write`; fade-together smoke already shows phase meaning outliving magnitude | **No commit/settle law.** Raw-norm halting on a rotating channel fires while content is live or never; the write residual `r_free = ‖u‖/(‖z_t‖+ε)` against the known free dynamics (§4.3) is the missing commit law — the gauge-quotient does NOT apply on this substrate (single-scalar orbits collapse it, §4.1) |
| Oscillation/damping axis | Research 620 + riir-infer issue 046 + riir-train Plan 459 + Issue 939 | ζ-regime of the PROPAGATION operator `ρR(θ)` (capacity: state tracking 30–300×) | **Different question, same machinery.** 620 classifies what the recurrence DOES to state (capacity/regime); QRR classifies how an iterative refinement SETTLES near its root (rate inheritance, attribution, halting). 620 ships no convergence criterion, no attribution protocol, no quotient |
| Damping knob | Issue 717 (`forward_looped` damping/tangential arms) | arms damping for deep loops; own rule "don't damp unless inference degrades" | QRR supplies the missing TRIGGER + the causal test (α-intervention) + the tension law (damping a settling oscillation degrades). The knob gains a principled gate |
| Meaningful ⊕ ignorable decomposition | `katgpt-dec::hodge_decompose` (exact ⊕ harmonic ⊕ coexact) | static field decomposition | Philosophical relative only (split the state into what matters vs symmetry motion); hodge has no loop/convergence semantics. No coverage claim |
| Least-squares fitting | `katgpt-attn-match::fit_cv_least_squares` | attention value reconstruction from score matrices | Different object entirely (KV compaction); no delta-sequence recurrence fit ships anywhere |

**Published prior art (§4 searches, 2026-10-16):** headline + component searches return the paper itself and its own cited lineage (Gauge Equivariant Transformer, He et al. 2021; FPRM/EqR/HRM/TRM/Attractors — all already distilled here as Research 035/049/051/079/529). No competing "symmetry-modulo stopping criterion as a reusable technique" surface exists — the field is 3 days old. Internal-first grep: zero prior mention of `2610.18174`/QRR in any workspace `.research/`.

## 4. Modelless distillation

### 4.1 The closed-form quotient residual — and its collapse on single-scalar orbits

For state `z` and proposal `w = T(z)` under a **shared-phase vector orbit** (one phase for C complex pairs — the paper's per-token gauge), `inf_φ ‖w − e^{iφ}z‖² = ‖w‖² + ‖z‖² − 2|⟨w,z⟩|` with `⟨·,·⟩` SUMMED over the phase-sharing pairs, minimized ONCE per orbit; residuals combine in quadrature only across INDEPENDENT orbits (tokens). Minimizing per-pair gives a lower bound that under-reads `r_G` → false early halts (the paper's 23%-false-halt class). ⚠ Verdict round 1 correction: on a **single complex scalar** per orbit (`phase_belief`'s `[f32; 2]` channels, one θ each), `|⟨w,z⟩| = |w|·|z|` always, so the form collapses EXACTLY to `|‖w‖ − ‖z‖|` — the No-GD advocate's original formula was correct there, and the residual is magnitude-only. **That is why the gauge-quotient is the WRONG commit law for `phase_belief`**: quotienting the channel phase destroys the very content issue 939 exists to carry. The sound transplant there is §4.3's innovation residual. Discrete sibling: `ModularPhaseCounter` cycling with static carry = pure orbit motion (`r_G ≈ 0`), not change.

### 4.2 The convergence-law family (all from delta sequences, no oracle)

Fit `δ_{t+2} = a·δ_{t+1} + b·δ_t` per channel over a sliding window (3-coefficient normal equations, O(1)/step against a ring buffer). Companion roots give the mode: complex pair → `r = √(−b)`, `ω = acos(−a/(2r))`; real flip → `S = −r`. Then: two-channel verdict (angle × modulus), settling budget `τ_δ = log(1/δ)/(−log r)` (invertible: budget buys tolerance `δ_B = r^B`), `r≈1` + complement-stagnation = non-convergence certificate, energy split via the Sylvester `P` (offline, ~O(8k³) at complement dim k) attributes *why* a loop stalled. The α-intervention protocol turns any attribution claim into a causal one on the stack's deterministic replays (SWE trajectories, cognition traces) — a failing prediction-match is a first-class recorded negative.

### 4.3 Fusion: the phase_belief commit law — write residual against the free dynamics, not a gauge quotient (verdict rounds 1–2 repair)

Issue 939 landed `phase_belief` on 2026-10-16 and immediately hit the ambiguity its own P2 documents: *"still counting time while its confidence fades"* — a damped rotating channel NEVER settles in raw norm, and settles trivially under a magnitude-only test. The paper's gauge-quotient is NOT the fix here: `phase_belief`'s phase is the SIGNAL, not a redundancy (θ is FIXED per channel at `new(rho, theta)`), and one complex scalar per orbit collapses `r_G` to the magnitude residual (§4.1). The correct transplant of the relative-equilibrium idea is: **measure change against the KNOWN free dynamics**. Each channel's free orbit is `z_{t+1} = ρR(θ)z_t` (fade + rotation, parameters known at construction — no training). Since `DampedPhaseBlock::step` computes `z_{t+1} = ρR(θ)z_t + u_t` exactly, the free-dynamics innovation IS the write:

`r_free = ‖u_t‖ / (‖z_t‖ + ε)` — a normalized write magnitude, computed DIRECTLY from the caller's write `u`, never reconstructed from `z` (mul_add rounding leaves ULP residue that would break the exact-zero arm).

The belief has **committed when `r_free` stays below a calibrated threshold for a persistence window W** — writes have stopped moving the channel off its own orbit, while the phase keeps rotating meaningfully. Two honest boundaries (verdict round 2): (i) this is a DESIGN DECISION, not an emergent property — "sustained identical writes ⇒ committed" is the consumer's call at pull time; under constant write `u` with ρ<1 the state converges to `z* = (I−ρR)⁻¹u`, where raw Δz → 0 reads "settled" while `r_free` stays `‖u‖/‖z*‖ > 0` reads "still writing" — the two signals genuinely disagree and P4 documents rather than resolves that. (ii) On a FREE run the raw residual only "never settles" in RELATIVE form: `‖Δz‖/‖z_t‖ = |ρe^{iθ}−1|` — exactly `budget_scale`, already computed at construction — while the absolute `‖Δz‖ = ρⁿ‖(ρR−I)z₀‖` decays under any fixed threshold; the gate compares the relative raw residual (normalized by `‖z_t‖`, not `‖z_{t+1}‖`), never an absolute one. P4 of issue 945 carries the gate; consumer posture follows Issue 939's defer-until-pull.

### 4.4 Game-context reframe (priority #1)

Per-NPC belief channels (leaky today, phase-carrying behind `phase_belief`) drive the FSM; the known pathology is oscillation/flip-flop, currently detected by sigmoid frustration drives (a counter). The quotient transplant: label churn (A↔B) with STATIC drives/beliefs is orbit motion — not a pathology; label churn with diverging content is real. The frustration drive reads the quotient (content motion under the label orbit), and the tension law becomes a damping predicate: never re-plan against a settling oscillation. Same real-boot PROGRESS-test convention as Issue 939 P3; same defer-until-pull posture.

### 4.5 Healer-context reframe (priority #2, honest)

The healer's recurrent surface is its fix-trajectory memory and the SWE trajectory-geometry classifier (bench_011/013) — and that classifier is exactly candidate-shaped for the attribution lane: spectral features (`r`, `ω`, energy split) turn "oscillation/drift/stuck" from curvature heuristics into mode semantics (`stuck` = `B≈0, r≈1`; `drift` = spectral mass unconcentrated), additively, with the labeled synthetic corpus as the pre-registered oracle. The corpus/retrieval/selection machinery itself has no iterative latent loop — recorded as validation, not action (same verdict shape as Research 529's healer row).

## 5. Verdict

**GAIN** (not Super-GOAT, not PASS).

- Q1 prior art: published (this paper) — extraction, not novelty; workspace-clean (signal-diff table).
- Q2 new behavior class (workspace): yes — phase-carrying beliefs gain a commit law; loop halters gain a rate channel; trajectory classes gain mode semantics.
- Q3 selling point: "beliefs that commit exactly when their content stops changing; loops that finish earlier without killing the oscillation they think with" — real but engineering-shaped.
- Q4 force multiplier: ≥2 surfaces (gain_cost_halt, phase_belief, bench_011/013, FSM drives, Issue 717 knob, riir-train Plans 373/459).

Per-tier outputs filed this session: research note (this file) + **katgpt-rs `.issues/945`** (modelless POC; P1/P2/P3/P5 landable against the live `forward_looped`/halter consumer, P4/P6 defer-until-pull / report-only) + **riir-train `.plans/462`** (training track; Phase 1 zero-GPU, Phase 2 owner-gated GPU). The architectures's own training recipe is NOT scheduled beyond the probe — no product lane needs a puzzle reasoner; the probe's value is the mechanism (certified convergence + learned-ζ), priced in Plan 462.

**GOAT posture for every extraction:** feature-gated, benchmark-before-promote, sigmoid not softmax, G3 byte-identity when disarmed, fp32 rotation internally, u32 counters where undamped (Issue 939's exact-integer law).

## 6. Landing record

- 2026-10-16: note + `.issues/945_qrr_quotient_convergence_extraction.md` + riir-train `.plans/462_qrr_learned_zeta_and_certified_convergence.md` + one-line cross-ref added to Research 620's Related header. No code.
- 2026-10-16 (verdict round 1 REVISE, addressed): the quotient-residual claim on `phase_belief` was WRONG — single-complex-scalar orbits collapse `r_G` to the magnitude residual (§4.1); P4's commit law replaced with the free-dynamics write residual; per-orbit (not per-pair) minimization pinned; non-equivariant-wiring guard added to issue 945; gate arms made non-trivial. Reviewer verified against `crates/katgpt-sense/src/phase_belief.rs` source.
- 2026-10-16 (verdict round 2 REVISE, addressed): `r_free` is exactly `‖u‖/(‖z‖+ε)` — computed from the write `u` directly, never reconstructed from `z`; the input-conditional-θ sentence deleted (θ is fixed in `new`, and a time-varying rotation term would absorb the angle the sentence claimed to detect); the constant-write gate arm DROPPED (its predicted directions were inverted — raw settles at `(I−ρR)⁻¹u`, `r_free` stays positive; the divergence is documented as a design boundary, not a gate); the raw-never-settles leg qualified to the RELATIVE raw residual `‖Δz‖/‖z‖ = |ρe^{iθ}−1| = budget_scale` (already computed in `new`); exact `r_free = 0` identity arm kept (u = 0 is bit-exact zero).
- 2026-10-16: P1/P2/P3/P5 LANDED — `qrr_convergence` feature (P1 `ModeFitter` + P3 budget/certificate in `crates/katgpt-core/src/qrr_convergence.rs`; P2 verdict in `gain_cost_halt.rs::with_mode_verdict`, kill-switch `KATGPT_QRR_OSC_VERDICT=0`; P5 gate in katgpt-sense `tests/issue945_fp32_rotation_law.rs`; 14 + 3 + 3 gates green, default-features lib 2157 green, combo clippy 0). The gates found two numerics laws the extraction must carry: f32 sliding-window sums cancel catastrophically (f64 accumulation landed) and single-mode AR(2) windows are rank-deficient (relative Tikhonov ridge, dominant root exact). Catalog §150; README feature table + counts 694.
