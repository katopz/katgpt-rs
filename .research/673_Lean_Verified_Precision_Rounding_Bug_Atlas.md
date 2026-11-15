# Research 673: Lean-Verified Precision + the Rounding-Bug Atlas (TorchLean / FloatLib / LeanSoftmax)

> **Source:** two companion blog posts extending the already-digested arXiv:2610.31642 (Barron et al., "A Fix for Softmax and Cross-Entropy" — Research 632, Issue 959):
> 1. Sasha Rush & Robert Joseph George, ["Lean Verified Precision"](https://srush.github.io/LeanSoftmax/) (2026-11) — Lean formalization of relative precision; machine-checked certificates for the softmax fix. Code: `srush/LeanSoftmax`.
> 2. Robert Joseph George, ["When Rounding Breaks Machine Learning"](https://www.robertj1.com/ai4science/rounding-bugs-in-ml-formulas/) (2026-11) — the TorchLean/FloatLib operator-atlas sweep: 162 operators × 7 libraries × ~780M outputs, 526-format rewrite sweep, ~15 rounding-bug rewrites beyond softmax/CE. Code: `lean-dojo/TorchLean`, `lean-dojo/FloatLib` (arXiv:2609.19352).
> **Date:** 2026-11-15
> **Status:** Active
> **Related Research:** 632 (the controlling note — the paper itself; this note owns the companion verification arc + extended atlas)
> **Related Issues:** katgpt-rs 959 (closed — softmax_fix landed), katgpt-rs 1220 (gegelu_tanh tail-lane divergence — filed this session), katgpt-rs 1221 (Lean RP certificate poc — filed this session)
> **Inspection method:** both pages fetched live over the wire; the three Lean repos NOT cloned (nothing Rust-shaped to grep; quotes below are blog-stated and re-derivable at mine time)
> **Classification:** Public

## 1. What these add over the digested paper

Research 632 / Issue 959 landed the paper's *forms* (`logspace_diff`, `fixed_log_softmax`, `precision_cliff`, `margin_equilibrium`). The companion arc adds two things the paper did not carry:

1. **Machine-checked precision certificates.** Not tests — theorems. The centerpiece is `softmaxGradient_no_cert`: under absorption, **no finite relative-precision bound exists** for the naive form (`1 − 1/(1+s)`), while the rearranged form (`s/(1+s)`, algebraically identical) certifies at `RP ≤ 2ε + 2ρ`. A proof that the naive form is *unprovable* is a stronger artifact than a test that it fails.
2. **The bug atlas beyond softmax/CE.** ~15 rewrites with regime conditions (§3), several of which touch shapes our stack ships or will ship (§4).

## 2. The verification methodology (the transferable primitive)

- **Olver relative precision (RP)** as the spec predicate: `y = x·e^δ, |δ| ≤ ε`. Metric — triangle inequality composes bounds across successive steps; `add_pos` preserves `max(ε,η)`; division adds; **subtraction has no cheap bound** (the entire bug class lives in the subtraction theorems can't cover).
- **Mixed bound** `|ŷ − y| ≤ r·|y| + a` — the relative term scales with the answer, the absolute term covers underflow. This is the right bound *shape* for our numerics gates, which currently assert ulp counts against an f64 oracle (Issue 959 G1) — a test, not a theorem.
- **FloatLib `roundoff` tactic** — reads a floating-point program's operations, applies per-op rounding theorems, composes the bound; Lean's kernel checks the result. A `decide +kernel` proof closes concrete instances exactly.
- **Sign theorems** for rewrites — e.g. the rewritten CE target gradient stays negative for any class count and any IEEE format (given a maximal target logit and one positive tail term). Sign theorems are cheap and are exactly what a *gradient-survival* gate needs.
- **The underflow fallback**: when the rewritten form's small terms underflow (binary16), combine the tail through its logarithms — finiteness-preserving, proved. Our `fixed_log_softmax` resolves the target slot "to exponent underflow" but does not carry a log-domain tail fallback; binary16-class lanes would need one.
- **Format sweep discipline**: precision (fraction bits) and range (exponent bits) are *separate* failure axes — bfloat16 (wide range, coarse spacing) and float16 (fine spacing, narrow range) fail differently on the same formula. Any claim of the form "the fix works" must name the format; 526-format sweeps are how TorchLean prices a rewrite.

## 3. The extended bug atlas (regimes included)

| Shape | Before (collapses) | After (rewrite) | Regime |
|---|---|---|---|
| tanh-GELU | `0.5x(1+tanh z)` → −0.0 at x ≈ −6 (f32) | `x·σ(2z)` (algebraic identity) | g ≲ −5.2 (where `z` ≲ −9.16: `1+tanh` rounds to 0 when 2e^{−2z} < half-ulp-below-1, 2.98e−8); lane-gap peak ~7e−8 near onset |
| GRPO k3 KL penalty | `e^d − d − 1` → 0.0 at d = 2^−12 (f32) | `expm1(d) − d` | small \|d\|; TRL runs it only with β > 0 |
| KL of nearby normals | `kl_divergence(N(0,1), N(1+2^−13,1))` → 0.0 or **negative** | closed-form in log-space terms | σ-ratio near 1 |
| Variance | `E[x²] − E[x]²` → −2 vs true 8/9 (PG `VarAggregation`) | two-pass centered `mean((x−μ)²)` | large common mean, small variance |
| Diffusion noise level | `sqrt(1−ᾱ)`, ᾱ=∏(1−βᵢ) → 0.0 (binary16) | accumulate `log1p(−βᵢ)`, `sqrt(−expm1(s))` | long schedules in low precision |
| Moving average / decay gates | EMA stuck at 1/64 for 5000 steps (bf16); xLSTM forget-gate decay → 1.0 | compute the increment in log space / wider accumulator | low-precision lanes; update Δ ≲ half-ulp of the state |
| Sigmoid | library kernel → 0.0 at x = −16.1 (ONNX f32) | two-branch `eˣ/(1+eˣ)` for x<0 | deep-negative tail |
| logit clamp | `torch.logit(1, eps=1e−8)` → +∞ (1−eps absorbs to 1.0 in f32) | clamp in the output domain, not via `1−eps` | eps ≤ half-ulp near the clamp point |
| Loss sums (binary16) | incremental batch sums → +∞ (label-smoothing rows) | exact sum rounded once | narrow-range formats, long reductions |
| Soft-margin / ℓ₃-norm / focal loss | ±∞ loss, NaN gradients in binary16 / f32 | log1p / max-shift forms | documented per-case (torchvision #9743 open) |

**Not new to the corpus:** the softmax/CE row of this table is Issue-185's family (`sigmoid-minus-one`, `softmax-onehot-collapse`, `neg-ln-softmax-prob`) — already mined.

## 4. In-stack audit (live greps this session)

1. **`gegelu_tanh` tail-lane divergence — CONFIRMED, live** (`katgpt-types/src/math.rs` L118–169; measured by execution during the verdict round). The SIMD body computes `g·u·(exp(2z)/(exp(2z)+1))` — already the sigmoid form `x·σ(2z)`, tail-correct. The scalar remainder computes `0.5·g·(1+tanh(z))` — the cancellation form. Measured lane gap (length 65, up = 1): onset at g ≈ **−5.2** (scalar returns −0.0 from there), peak ≈ **7e−8** near the onset (g = −5.0: SIMD −2.29e−7 vs scalar −2.98e−7), −8.44e−11 at g = −6 (matching the blog's −8.4e−11), and a **positive-side** one-ulp effect too (+4.8e−7 at g = +5, a `1+tanh` rounding difference rather than cancellation). **The same public function returns different values depending on input length mod 64** — my earlier "g ≲ −9" band confused z with g and the "≤1e−9" cap under-priced the peak by ~70×; both corrected per the measurement. The in-code comment ("the chunked path … also matches") is true over the normal range and false in the tail. Site census of the same shape: `coda.rs::GateActivation::GegeluTanh` (fast_tanh — Padé also saturates to ±1), `belief_drafter.rs::gelu`, `riir-engine policy_head.rs::gelu`, bench copies. **Severity: LOW** (≈1e−7-scale absolute on a hidden activation; the Gemma-2 hot path is the SIMD lane) — parity/audit finding, filed as **Issue 1220**, not a hotfix. `gemma2_train::gegelu_backward_into` parity (the FD-check contract that pins the scalar lane to standard `tanh`) must be re-derived if the scalar lane moves.
2. **GRPO k3 — LIVE (two sites, found by the verdict round).** The exact `e^d − d − 1` form runs in f32 at `riir-train-engine/src/nf_grpo.rs:252-256` (`kl_penalty`, consumed by `grpo_loss` for both factorizations) and `svr_driver.rs:1151-1153` (`svr_policy_loss`). Collapse regime: for `|d| ≲ 3.4e−4` (`d²/2 < half-ulp-of-1`), correctly-rounded `exp(d)` lands within a half-ulp of `1 + d` — at `d = 2^−12` the excess is EXACTLY a quarter-ulp (2^−25), so the form returns exactly 0.0 (true value 2.98e−8, representable) and the gradient `e^d − 1` dies at still-smaller d. Impact: the β_KL regularizer silently switches off in the near-converged band where it exists to keep the policy pinned. → **riir-train Issue 691** (fixture-first fix: `exp_m1(d) − d`, zero-preserving). The GPU-crate lanes are clean: `loss_grpo.rs` carries only the clipped importance ratio; `entropic_tilt::kl_vs_uniform` is log-domain; `loss_uopsd`'s KL is a clipped top-k estimator in f64.
3. **Issue-940 tension, resolved as a carve-out.** TorchLean's "exact sum rounded once" fix for low-precision sums reads against our determinism law (sequential ascending fold pinned at sync crossings). Not contradictory: the law pins **order for determinism in f32 sync lanes**; the rewrite serves **accuracy in low-precision local reductions**. The corpus rule carries the carve-out explicitly (sync-crossing reductions stay order-pinned).
4. **dllm noise schedule — not-transferable.** `dllm::NoiseSchedule` is a discrete-diffusion masking schedule (monotonic ratios), not a DDPM `ᾱ = ∏(1−βᵢ)` cumprod — the noise-level rewrite has no stack surface. Dropped from the distill's lane-intel; corpus-eligible only as a generic consumer rule.

## 5. Verdict + fusion

**Tier: Gain.** The paper is landed (Research 632/Issue 959); the companion arc's actionable residue is filed, not parked — including one LIVE in-stack defect the audit surfaced:

- **Issue 1220** — gegelu_tanh tail-lane divergence audit (property test + site census + scalar-lane alignment decision gated on the backward-parity contract). Band/cap re-pinned per the verdict-round measurement (onset g ≈ −5.2, peak ≈ 7e−8, positive-side one-ulp effect included).
- **Issue 1221** — Lean precision certificate poc: formalize the RP predicate + one sign/mixed-bound theorem for `fixed_log_softmax_target` in `.proofs/KatgptProof`, paired spec-match test per the Plan-441 convention. This is the fusion the arc earns: **certificate-grade numerics family** — the numerics primitives (softmax_fix, Issue 959) × the FV pillar discipline (four `.proofs/` instances, none carrying a float-precision certificate today). Our G1 gates currently assert ulp counts against f64 oracles; the offered upgrade is machine-checked bounds (`RP ≤ 2ε+2ρ`-class, sign theorems) over the same surface. No "candidate" hedge: the proof scope is the issue.
- **riir-train Issue 691** — the k3 collapse is LIVE (nf_grpo + svr_driver, both f32): fixture-first fix to `exp_m1(d) − d`, zero-preserving.

**Distill half:** the atlas routes to riir-refine `sec`/Numerics (Research 412, nominal Batch 366) — the softmax/CE rows are Issue-185-covered; five new transcription candidates (gelu-tanh sigmoid form, k3, two-pass variance, low-precision decay gates, logit-clamp absorption) + the Issue-940-carve-out sum rule.

## 6. Prior-art position

- **Internal:** zero workspace hits for `TorchLean|FloatLib|LeanSoftmax|lean-dojo` (grep, this session) — the FV arc is un-distilled; the softmax/CE half is Research 632's.
- **Published (named by the sources themselves):** Blanchard/Highams softmax & LSE analysis; Prieto et al. arXiv:2501.04697 (Softmax Collapse — already in the softmax_fix lineage); Flocq/Gappa (Boldo/Melquiond/Daumas), VCFloat2 (Appel/Kellison), Harrison's HOL Light FP theory, FPTaylor, Herbie; FloatLib arXiv:2609.19352. Upstream bug reports: PyTorch draft fix #202049 (softmax collapse), torchvision #9743/#9751 (focal loss NaN), TFP #2040 (1−CDF survival), diffusers #9004 (noise-level cast, partial).

> **PASS-Redirects (synthesis):** Rush & George [LeanSoftmax, srush.github.io/LeanSoftmax] + George ["When Rounding Breaks Machine Learning", robertj1.com/ai4science/rounding-bugs-in-ml-formulas/] — companion verification arc to Research 632: Lean RP certificates + the operator-atlas rounding-bug atlas; softmax/CE forms already landed (Issue 959), residue filed as Issues 1220/1221 + riir-refine Research 412.
