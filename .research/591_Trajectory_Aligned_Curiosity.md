# 591 — Trajectory-Aligned Curiosity (per-arm drift-alignment gate; arXiv:2609.30063 extraction)

**Status:** NOVELTY 4/4, **GOAT FAIL as shipped** ([Bench 900](../.benchmarks/900_arm_drift_alignment_goat.md), 2026-09-26): G1 planted-drift AUC 0.783 / held-out 0.602 < 0.8. The G3 loop win reverses when the better family has a zero pull centroid. Two structural defects: the first-moment mean pull is blind to spread families, and an axis-aligned preconditioner cannot separate drift from pool geometry. Also, the drift as written here (arm-indexed) cannot be dotted against latent directions (Plan 610 design correction). Redesign (second-moment, null-normalized) → Issue 899. Original status: SUPER-GOAT (novelty gate 4/4 below) — open primitive → `katgpt-core::cgsp` (Plan 610, feature `arm_drift_alignment`); architectural guide → `riir-ai/.research/389_Trajectory_Aligned_Curiosity_Guide.md`. Source: arXiv:2609.30063 "Self-Play Pretraining with Zero Data" (2026-09-24); full training-track distill lives in `riir-train/.research/458`. Panel: No-GD + model-based + web prior-art, one batch 2026-09-26.

## Pinned claim (§4 precondition — written before the searches ran)

> **Per-arm trajectory-aligned curiosity** — `r̃_k = sigmoid(β·|⟨ĝ_k, P⊙û⟩|)`: each CGSP arm's candidate direction `ĝ_k` projected (dot product, absolute value, sigmoid) onto the **RMS-preconditioned preference-drift direction** `û = d/‖d‖` where `d` is the per-dimension fast−slow temporal derivative the kernel already computes — for CGSP/self-adaptive game cognition, distinguished from `DerivativeCuriosity` (global norm `sigmoid(β·‖d‖₂)`: direction-blind, per-cycle-global) and from published direction-aware learning progress (AdaS: optimizer-side step sizing; LESS: validation-anchored) by per-arm attribution + latent-space (no gradients) + behavioral-drift proxy.

## The extraction

The paper's generator reward `r_i = |⟨∇_θ L(y_i; θ_e), P_e ⊙ δθ_e⟩|` answers: *"is this item involved in the direction the learner is currently moving?"* — per item, absolute-valued, preconditioned, windowed. Strip the calculus (Path 0, row 1 of Research 458):

| Paper | Modelless form | Why faithful |
|---|---|---|
| `g_i` (item gradient) | the arm's `Candidate.direction` — the d-dim latent vector **already on every cgsp candidate** (`cgsp/types.rs`) | the functional direction "what this candidate pulls toward" |
| `δθ_e` (parameter movement, growing lookback) | the drift direction `û = d/‖d‖` from `TemporalDerivativeKernel::observe()` — per-dimension fast−slow EMA difference (α 0.3/0.03, 10:1 two-timescale) | the learner's *behavioral* movement, EMA-windowed |
| `P = lr·√v̂ + ε` (AdamW preconditioner) | per-coordinate RMS normalization of the drift before projecting: `d_j → d_j/(rms_j+ε)`, `rms_j` = EMA of \|d_j\| | covariance→correlation; scale-free alignment, O(d), fixed-size state |

Then per arm k: `r̃_k = sigmoid(β·|⟨ĝ_k, P⊙û⟩|)` — dot + abs + sigmoid (house law), zero-alloc (fixed arrays + `simd_dot_f32`, already used in the module). The `|·|` credits anti-drift arms too — "involved in the current learning direction", strictly stronger than the incumbent's "any change".

**The load-bearing substrate fact:** `TemporalDerivativeKernel::observe(&pref_buf)` already returns the per-dimension derivative vector; `sigmoid_surprise_gate(&d, β)` then **pools it away by norm**. The incumbent's own module docs record the loss: *"Reward is global per cycle, not per-arm … derivative-curiosity cannot [differentiate arms]. This is the key semantic loss."* This plan consumes state that exists and is discarded.

## Novelty gate (§1.5, all four)

1. **No prior art?** In-stack: grep-verified — `DerivativeCuriosity` (global norm), `hint_regret` (paired-CRN VoI of a hint: counterfactual utility signal, not drift alignment), CGSP `(1−solve_rate)·guide_score` (per-arm, target-relevance, not trajectory), `npc_clr thaw_with_update` (reliability-weighted latent update accumulation, not a curiosity gate). No mechanism computes per-arm alignment with the system's drift direction. Published (11 queries, 2026-09-26): LESS 2402.04333 (AdamW-preconditioned gradient dots **against a validation gradient**), RHO-LOSS 2202.03258 (magnitude-only LP), **AdaS 2006.06587 (VERIFIED 2026-09-26: Hosseini & Abdoli, "AdaS: Adaptive Scheduling of Stochastic Gradients" — learning rate exponentially regressed onto the cosine similarity between successive gradients; direction-aware progress but OPTIMIZER-SIDE step sizing, never a data/candidate-selection gate)**. No per-arm latent-space trajectory-alignment selection gate found. **YES.** ⚠ Provenance note: the prior-art agent's first AdaS ID (~1912.09965) was WRONG — resolved to a combinatorics paper on direct fetch; the correct ID is 2006.06587. Plan 610 T1 pins the verified AdaS citation + this signal-diff into the `arm_drift_alignment` module docstring (the novelty-asserting doc ships only with verified neighbors — verdict-review round-3 condition).
2. **New behavior class?** Per-arm frontier-seeking that prioritizes candidates *consistent with the system's current learning direction* — a capability the incumbent demonstrably lacks (its own docs). **YES.**
3. **Product selling point?** "Our NPC swarms seek experiences that reinforce their current learning direction — per-NPC, direction-aware curiosity" — finishable sentence. **YES.**
4. **Force multiplier?** Connects cgsp curiosity (pillar) + self-adaptive EMA trajectories + hint_regret frontier triage + the training-side LP reward (riir-train Plan 420 T3) — ≥2 pillars. **YES.**

## Signal-diffs (§3.6) on the coverage dismissals

- **vs `DerivativeCuriosity`:** consumes magnitude of preference change (`‖d‖₂`, global) vs direction-consistency per arm (`|⟨ĝ_k, P⊙û⟩|`). The diff is recorded *in the incumbent's own docstring* — the strongest form.
- **vs `hint_regret` VoI:** consumes counterfactual utility of revealing a hint (paired-rollout return delta) vs involvement in the observed drift. Complementary — the gate ranks *what to seek*; VoI ranks *what to reveal*. Guide 389 maps the composition.
- **vs CGSP solve_rate·guide_score:** consumes target relevance × failure rate vs trajectory alignment. An arm can be relevant, unsolved, and orthogonal to the current direction — CGSP rewards it, this gate does not, by design.

## GOAT gate (Plan 610)

- **G1 mechanism (planted-truth):** synthetic bandit with a planted drift axis — alignment score must rank drifting-family arms above stationary arms (rank-AUC ≥ 0.8) + **negative control**: under pure-noise drift the score distribution is flat across groups + **scale-invariance arm**: benign per-coordinate rescaling of the drift must not move the rank order (the preconditioner's contract).
- **G2 discrimination vs incumbent:** the global-norm gate MUST FAIL G1's per-arm ranking — that is the delta this exists for. If the incumbent passes, the fixture is mis-aimed; fix the fixture, not the bar.
- **G3 end-to-end (Bench-950 pattern):** CGSP loop A/B — per-arm alignment vs global-norm vs uniform at matched budget, paired, bit-identical-when-off. Honest null is an acceptable outcome.
- **G4:** zero steady-state allocations; ns budget ≤ ~2× current `observe_interestingness` cost.
- **Bridge validation (riir-train-side, once):** correlate probe-fingerprint drift against actual parameter movement across two frozen checkpoints — forward-only arithmetic; until measured, the reward's claims stay mechanism-level (behavioral drift ↔ δθ is an assumption, stated not hidden).

## Fusion

Paper × `DerivativeCuriosity` (the pooled-away per-dim derivative) × `hint_regret` (frontier triage vocabulary: learnable-hard vs mastered vs intractable) × riir-train Plan 420 T3 (the same reward, gradient form, for trained lanes) × Bench-950 A/B pattern (the gate harness). What none alone has: a **modelless curriculum-control loop** — per-arm, direction-aware, zero-gradient frontier seeking that shares one reward algebra with the training track.

## PASS-Redirects (synthesis)

> **PASS-Redirects (synthesis):** Cowsik, Dolev, Li, De Luca, Cohen, Goodman, Levine [arXiv:2609.30063 "Self-Play Pretraining with Zero Data"] — training-track GOAT filed in riir-train (Research 458 / Plan 420); this note extracts the modelless flagship (per-arm direction-aware reward) and records the sibling extraction.
> Xia et al. [arXiv:2402.04333 "LESS: Selecting Influential Data for Targeted Instruction Tuning"] — nearest published reward-form neighbor (validation-anchored preconditioned gradient dots); delta pinned above.
> Mindermann et al. [arXiv:2202.03258 "RHO-LOSS: Prioritized training on points that are learnable, worth learning, and not yet learnt"] — magnitude-based learning progress; no direction term.
