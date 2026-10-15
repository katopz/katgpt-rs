# Plan 627 — the `set_refine` operator: the one capability an independent option scorer structurally lacks

**Status:** FILED 2026-10-16 (substrate-first gate SATISFIED, see §1) — plan-first
for a new katgpt-core primitive; implementation unclaimed. Downstream of reflex
Issue 090 C2 (the Jeva fusion follow-up; the issue text is the seed spec). The
promotion decision stays gated on the MEASURED GOAT (§4) — nothing here promotes
or wires by default.

## 1. Substrate-first verdict (the gate this plan exists to record)

Vocabulary translation searched 3+ name variants across `katgpt-rs/**/*.rs`,
`.plans/`, `.research/` (2026-10-16, both boxes' canonical checkout):

| Variant | Result |
|---|---|
| `set_refine` / `set-refine` / `SetRefine` | ZERO hits |
| `option_interaction` / `set_dependent` / `preference_reversal` / `cross_option` / `option_conditioned` | ZERO hits in katgpt-rs (`option_cond` exists REFLEX-side — count-table terms keyed (qid, option); state-side signal, NOT option-interaction feedback) |
| `hopfield` / query-update / state-update / set-aware / decoy | **`cp_hopfield`** (Plan 567, opt-in) — the closest prior art |

**Adjacent substrate that EXISTS and will be consumed, not duplicated:**
- `exact_sigmoid` — the gate (never softmax; the C2 `w_i = σ(dot(x,k_i)/τ)` form).
- `cp_hopfield` (Plan 567) — the same FAMILY (set-conditioned state update over a
  stored pattern set: `recall_step` aggregates the set into the state), but a
  different job: associative-memory CAPACITY on CP^(d-1) via top-eigenvector
  alignment, opt-in, GOAT'd at N∈{8,64}. NOT a decision-option scoring operator —
  cited as prior art and a design warning (its machinery is orders heavier than
  the C2 operator needs).
- `spectral_pencil` / `basis_split` — direction-vector vocabulary (the k_i keys
  are plain embedded option rows; no learned directions needed at v1).
- Reflex engine seams the consumer side already ships: `Scratch::domains`,
  `semantic_route` (Bench 149), the `--rubric-probe` lane's route-agreement
  characterization (context==state-alone 59.7% — the measured gap a
  set-conditioned term would close or consciously not close).

**Verdict: the operator itself is NEW, small substrate.** ~100–150 LOC + types +
probe suite + benches. File in katgpt-core (this repo — modelless inference
primitive, no riir dep, upstream of everything). No parallel-system risk: nothing
in the workspace re-implements set-conditioned option scoring under another name.

## 2. The operator (flag-gated `set_refine`, inert by default)

Their Prop 1 (arXiv:2610.18773): an independent per-option scorer cannot REVERSE
an A-vs-B preference when the candidate set changes; Jeva's fix is one
candidate-conditioned attention block. The modelless analog to price:

```text
x' = x ⊕ g · Σ_i w_i · k_i        w_i = σ(dot(x, k_i) / τ)   (never softmax)
```

- `x` — the state embedding (the engine's state-alone vector, D-dim f32).
- `k_i` — the option embeddings (immutable; options never mutated).
- `g` — the gate scale (0.0 = OFF, byte-identical posture; the G3 identity).
- Permutation invariance by construction (B2): the Σ over options is
  order-symmetric — assert to bit-identity in the suite.
- Value-blind argmax preserved (C1): the operator reads scores/keys, never a
  value vector — assert the C1 covariance identities still hold on refined
  distributions.
- **Invariance scope is NOT A1–A3**: set-dependent reversal REQUIRES option
  interaction — a working `set_refine` deliberately BREAKS reflex A2 by design
  and may move A1/A3 wherever coverage reads refined scores. The exemption and
  its own gate suite are the issue's law; do NOT "fix" the A2 red.

## 3. The set-awareness probe suite (the gate A2 cannot provide)

A suite that REDS unless the scorer can reverse set-dependence — the capability
test, run at every posture:

- [ ] **S1 reversal capability**: constructible fixture where {A,B} picks A and
      {A,B,+decoys shaped like B} picks B (the Jeva Prop-1 demonstration, modelless
      edition) — the suite's existence proof; a scorer that cannot flip reds.
- [ ] **S2 permutation invariance to bit-identity**: refined scores identical
      under option permutation (the Σ symmetry), every fixture.
- [ ] **S3 monotone-gate check**: w_i ∈ (0,1), Σ-shaped response to dot(x,k_i)
      (sigmoid, never softmax — the denominator-free law; a softmax spelling reds).
- [ ] **S4 immutability**: k_i bit-identical before/after the call; x' bounded
      (‖x'‖ ≤ ‖x‖ + g·Σw_i·‖k_i‖ — no norm blowup).
- [ ] **S5 flag-off identity**: g=0 ⇒ byte-identical scores (G3's engine-side twin).

## 4. The GOAT gates (all four, `--release`, box state quoted)

- [ ] **G1** flag-on beats flag-off on a REAL decision suite beyond the
      conformal-naive floor (reflex harness A/B, paired, one frozen read —
      Report-the-Floor law).
- [ ] **G2** latency p99 ≤ the engine's 1 ms per decision set budget with the
      operator armed.
- [ ] **G3** flag-off hash-identity (byte-identical boots, the combo-lane law).
- [ ] **G4** alloc-free on the hot path (the canary-armed counting allocator).
- [ ] Only if ALL pass AND the gain is modelless: the promotion question goes to
      a proposal, never straight to default. Any miss ⇒ stays opt-in with the
      measured record (a negative is a valid landing — the Bench-109 shape).

## 5. Execution order (each its own session slice)

- [ ] T1 katgpt-core: `set_refine` module (feature `set_refine`, opt-in) —
      operator + types.rs + S1–S5 unit suite. Gate: `cargo test -p katgpt-core
      --features set_refine --lib`, clippy `-D warnings` default + all-features.
- [ ] T2 reflex: engine seam (flag-gated, default-off) + the harness A/B arm —
      NOT wired into any shipped posture; the lane refuses unless the feature is
      named (the birth-gate pair: `#![cfg]` + `[[test]]`/`[[bench]]`
      required-features in the same commit).
- [ ] T3 the GOAT run (§4) on the tranche suites; bench record + honest verdict.
- [ ] T4 proposal (or the measured negative) — owner-gated promotion decision.

## 6. Cross-references

- reflex `.issues/090` §C2 (the seed spec + the do-not-wire-blind law)
- `.research/` Jeva lineage: reflex Research 007 (rethink), riir-train Issue 632
- Plan 567 `cp_hopfield` (prior art; the heavy-machinery warning)
- Bench 149 `--rubric-probe` (the route-agreement characterization that motivates
  asking whether option-interaction buys anything at all)
