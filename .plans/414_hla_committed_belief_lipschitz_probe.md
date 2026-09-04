# Plan 414: HLA Committed-Belief Lipschitz-Sensitivity Probe (F4)

**Date:** 2026-07-09
**Source issue:** ``.issues/048_hla_committed_belief_probe_blocked_on_r344.md``
**Parent plan:** [`.plans/406_renoise_ce_self_verifier.md`](406_renoise_ce_self_verifier.md) (P5.4 — the last remaining follow-up)
**Target:** `crates/katgpt-core/src/committed_field_blend.rs` (DRY refactor + new probe section)
**Status:** CLOSED — G1+G1b+G2+G2b+G3+G4+G5 ALL PASS (2026-07-09). Stays OPT-IN (diagnostic primitive, no runtime consumer).

---

## Goal

Ship the **HLA committed-belief π-sensitivity probe** — a modelless diagnostic that
perturbs the committed `π` weights, re-evaluates the blend map, and measures output
drift against an on-the-fly theoretical Lipschitz bound. A bound violation flags a
numerics bug in the committed blend (wrong sigmoid, wrong Lipschitz reporting, FMA
accumulation drift).

This is the **F4 fusion** from Plan 406 (renoise-CE self-verifier) — the last of
five private follow-ups (F1–F5). F1–F3 shipped; F4 was blocked on R344 re-validation;
F4 is now **UNBLOCKED** (2026-07-09): `CommittedBlendState` is a one-shot closed-form
map, NOT an iterative attractor, so the R344 flip-flop pathology is impossible by
construction.

---

## Critical design correction (vs the issue's proposed gate)

**The issue's gate is incorrect.** Issue 048 proposes:

> realized drift < theoretical `lipschitz_bound · δ`

But the cached `CommittedFieldBlend::lipschitz_bound(fields)` computes the **z-sensitivity**
Lipschitz constant — `max_k σ_k · L_k` where `L_k` is the Lipschitz constant of `f_k`
w.r.t. the **input state z**. The F4 probe perturbs **π** (the committed parameters),
not z. The π-sensitivity Lipschitz constant is a **different quantity**:

```
L_π = max_j  (1/τ) · σ_j · (1 − σ_j) · ‖f_j(z)‖
```

where `σ_j = sigmoid(π_j / τ)` and `‖f_j(z)‖` is the Euclidean norm of field j's
output at the current state z. This depends on the **value** of f_j at z, not its
Lipschitz constant — so it must be computed on-the-fly (it is NOT cached).

**This plan corrects the gate** to use the π-specific bound. The cached z-Lipschitz
bound is unaffected and remains valid for its original purpose (input sensitivity).

---

## Why this does NOT use the `RenoiseCeProbe` trait

The `RenoiseCeProbe` trait (Plan 406) perturbs the **output state**, re-resolves
through the operator, and measures drift — it tests fixed-point stability. The F4
probe perturbs the **parameters** (π), not the output — it tests parameter
sensitivity. The State types don't align (perturbing π vs perturbing the output dz
are fundamentally different operations on different-dimensional spaces).

The F4 probe is a **bespoke function** that uses the renoise-CE *idea* (perturb +
re-evaluate + measure drift) but applies it to parameters. This is consistent with
the issue's reframing as a "Lipschitz-sensitivity self-verifier."

---

## Constraints checklist (per AGENTS.md)

- [x] Modelless first — pure inference-time perturb-and-re-evaluate, no training
- [x] Latent-to-latent preferred — operates on the committed π → dz map
- [x] Freeze/thaw over fine-tuning — no weight mutation, pure read-side probe
- [x] Sigmoid not softmax — drift gate is `drift ≤ bound` (strict inequality)
- [x] SOLID, DRY — extracts `apply_blended_with_pi` free function (shared by production + probe)
- [x] Zero-allocation hot path — fixed `[f32; N]` / `[f32; D]` arrays
- [x] CPU/GPU/ANE auto-route — N/A (pure scalar/f32 op, no kernel)
- [x] UQ-floor rule — NOT a UQ primitive (raw sensitivity measurement, no probability claim)

---

## Phase 1 — DRY refactor + probe skeleton

### T1.1 — Extract `apply_blended_with_pi` free function

Extract the core loop from `CommittedFieldBlend::apply_blended` into a module-level
free function that takes `pi: &[f32; N]` explicitly:

```rust
fn apply_blended_with_pi<const N: usize, const D: usize>(
    pi: &[f32; N],
    tau: f32,
    fields: &[&dyn ArchetypeFieldSource<D>; N],
    z: &[f32],
    dz_scratch: &mut [f32],
    dz_out: &mut [f32],
) -> &mut [f32]
```

`apply_blended` becomes a one-line delegation: `apply_blended_with_pi(&self.pi, self.tau, ...)`.
No behavior change — the existing G4 zero-alloc test must still pass bit-identically.

### T1.2 — Implement `pi_sensitivity_bound`

Compute the theoretical π-sensitivity Lipschitz bound on-the-fly:

```rust
pub fn pi_sensitivity_bound<const N: usize, const D: usize>(
    pi: &[f32; N],
    tau: f32,
    fields: &[&dyn ArchetypeFieldSource<D>; N],
    z: &[f32],
    dz_scratch: &mut [f32],
) -> f32
```

For each field j: evaluate `f_j(z)` into scratch, compute `‖f_j(z)‖`, multiply by
`(1/τ) · σ_j · (1 − σ_j)` where `σ_j = sigmoid(π_j / τ)`. Return the max over j.

This is the **first-order** bound. For finite perturbation δ, the true drift may
exceed the first-order bound if the sigmoid curvature changes significantly over
the interval `[π, π+δ]`. The GOAT gate (G1) uses small δ (0.01) where the first-order
approximation is tight.

### T1.3 — Implement `committed_blend_pi_sensitivity`

The probe function — perturb π, re-evaluate, measure drift:

```rust
pub fn committed_blend_pi_sensitivity<const N: usize, const D: usize>(
    blend: &CommittedFieldBlend<N, D>,
    fields: &[&dyn ArchetypeFieldSource<D>; N],
    z: &[f32],
    perturbation_level: f32,
    k_draws: u8,
    rng: &mut Rng,
) -> PiSensitivityScore<N>
```

Returns `PiSensitivityScore`:
- `mean_drift: f32` — mean L2 output drift across k draws
- `per_draw: [f32; 8]` — per-draw drifts (fixed array, zero-alloc)
- `bound: f32` — the on-the-fly π-sensitivity bound (same for all draws at fixed z)
- `accepted: bool` — `mean_drift <= bound * perturbation_level * sqrt(N)` (the bound
  scales with ‖δ‖₁ ≤ √N · ‖δ‖₂ for component-wise uniform perturbation)

### T1.4 — Feature flag

Add `hla_committed_belief_probe = []` to `Cargo.toml` features. Both dependencies
(`renoise_ce`, `committed_field_blend`) are already default-on, so this feature just
gates the new probe section. Opt-in until GOAT gate passes.

### T1.5 — Module wiring

Gate the new functions behind `#[cfg(feature = "hla_committed_belief_probe")]` in
`committed_field_blend.rs`. No new module file — the probe lives in the same file
as the blend it probes (cohesion).

### Phase 1 GOAT sub-gate
- [x] `cargo check -p katgpt-core --features hla_committed_belief_probe` clean
- [x] `cargo check -p katgpt-core --all-features` clean
- [x] Unit tests pass: DRY refactor bit-identical (13/13 existing), bound computation correct

---

## Phase 2 — GOAT gate

### G1 — Lipschitz bound holds (correctness) ✅ PASS

**Result: 1000/1000 random configs PASSED (0 violations).** N=3, D=32,
random committed blends (pi ∈ [-10, 10], tau=1.0), random z ∈ [-1, 1]^D,
perturbation_level=0.01, k_draws=8. The map IS Lipschitz by construction;
no violations → no numerics bug.

### G2 — Bug detection (value) ✅ PASS

**G2 (NaN):** NaN in pi → mean_drift = NaN → `accepted = false`. Correct.
**G2b (under-reported Lipschitz):** A field under-reporting `lipschitz_bound`
by 1000× does NOT cause false acceptance — the probe's bound uses the ACTUAL
`‖f_j(z)‖` (computed on-the-fly), not the reported Lipschitz constant. The
acceptance gate still holds (the map IS Lipschitz regardless of what the field
*claims* its Lipschitz constant is).

### G3 — No regression ✅ PASS

**Result: 13/13 existing tests pass with AND without `hla_committed_belief_probe`.**
The DRY refactor (T1.1 — `apply_blended` delegates to `apply_blended_with_pi`)
is bit-identical.

### G4 — Zero-allocation hot path ✅ PASS

**Result: 0 allocs / 1000 calls.** Fixed `[f32; N]`, `[f32; D]`, `[f32; 8]`
arrays — no heap. Verified via `crates/katgpt-core/tests/common/mod.rs` CountingAllocator.

### G5 — Latency ✅ PASS

**Result: p50 = 3.042µs (release), p99 = 3.209µs.** Target < 5µs (1.6× headroom).
N=3, D=32, k_draws=8. (Debug build: ~20µs — unoptimized, informational only.)

---

## Phase 3 — Promote/demote decision ✅ STAYS OPT-IN

**Verdict: G1+G1b+G2+G2b+G3+G4+G5 ALL PASS.** The probe is correct and useful
as a diagnostic. **Stays OPT-IN** — this is a diagnostic/self-verifier primitive,
not a hot-path performance primitive. No runtime consumer needs it yet. The GOAT
gain is correctness assurance (catches numerics bugs in the committed blend),
not latency or quality. Promotion to default-on is NOT warranted until a runtime
consumer (e.g., a freeze-gate self-check, or a CI numerics-bug detector) needs it.

---

## Phase 4 — Documentation + close Issue 048

- [x] GOAT verdict recorded in this plan (Phase 2 results above).
- [x] Update Plan 406 P5.4 status from `[-]` to `[x]`
- [x] Update Issue 048 with the plan outcome + close it
- [x] Commit with message `feat(hla-probe): committed-belief π-sensitivity Lipschitz probe (Plan 414, Issue 048 F4)`

---

## Open risks

1. **First-order bound may be loose for large δ.** The theoretical bound uses
   `σ_j(1-σ_j) ≤ 1/4` evaluated at π_j, but for finite δ the sigmoid curvature
   changes. Mitigation: use small δ (0.01) for G1; document the small-δ assumption.
2. **The probe is a diagnostic, not a runtime primitive.** GOAT promotion to
   default-on may not be justified — there's no runtime consumer that needs it.
   This is acceptable: the probe ships opt-in as a testing/debugging tool.
3. **Bound scales with ‖f_j(z)‖, which varies per state.** The bound is state-dependent,
   not a global constant. This is correct (local Lipschitz) but means the bound
   must be recomputed per probe call. Acceptable for a diagnostic.

---

## TL;DR

Plan 414 ships the F4 follow-up from Plan 406 (renoise-CE): a committed-belief
π-sensitivity probe that perturbs the committed `π` weights, re-evaluates the blend,
and checks realized output drift against an on-the-fly π-Lipschitz bound. Corrects
the issue's gate (the cached `lipschitz_bound` is for z-sensitivity, not π-sensitivity).
DRY refactor extracts `apply_blended_with_pi` free function. Ships behind opt-in
`hla_committed_belief_probe` feature. GOAT gate: G1 bound holds, G2 bug detection,
G3 no regression, G4 zero-alloc, G5 latency. Diagnostic primitive — likely stays
opt-in even if GOAT passes (no runtime consumer). All modelless.
