# Plan 587: ActionBridge Lean 4 Monotonicity Proof

> **Renumbered 449 -> 587 on 2026-09-04 (Issue 724 T2).** `449` was allocated
> twice: this plan (2026-06-23) and `449_poincare_latent_navigation_primitive.md`
> (2026-07-18). `f98f7b51`'s precedent is *renumber per git-log first-creation*,
> which would have kept 449 here and moved Poincaré — but measured, that is the
> wrong direction: **27 of 33 `Plan 449` mentions are Poincaré-context** (README
> feature table, CHANGELOG, docs, an example) against **4** here, and
> `.benchmarks/449_poincare_goat.md` + `.research/449_SeeSE3_*` already share the
> number by the owner-number convention. Renumbering Poincaré would also have
> rewritten CHANGELOG entries, which are historical records of what was said at
> the time and should not be edited to match a later renumber. So citation weight
> wins over creation order; the precedent does not generalize past its own sweep.

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md](../.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md)
**Source:** Bridge neuro-symbolic gap analysis (user prompt 2026-06-23)
**Target:** `katgpt-rs/.proofs/` (new top-level dir) + `katgpt-rs/tests/bridge_spec_match.rs`
**Status:** ✅ COMPLETE — Phase 1-3 done (G1 toolchain, G2 theorem type-checks, G3 Rust spec-match). All 3 theorems compile with no `sorry`; axioms = `{propext, Classical.choice, Quot.sound}`. Mathlib dependency forces toolchain to `v4.32.0-rc1` (higher than riir-chain's pinned `v4.31.0` — unavoidable for transcendental analysis of `exp`).

---

## Goal

Prove `∀ a b, dot a > dot b ⟺ sigmoid (dot a) > sigmoid (dot b)` in Lean 4 — the ranking-preservation property that Plan 262 G1.3 currently asserts over 1000 random triples. This is Tier 3 of the bridge FV strategy; it is the open katgpt-rs primitive. Promotes the empirical G1.3 test from `∃` to `∀`.

**GOAT gate:** G1–G3 (toolchain bootstraps, theorem type-checks, Rust spec matches Lean). Promotion: default-on docs reference once all three pass.

---

## Phase 1 — Lean Toolchain Bootstrap (after riir-chain Plan 004 T1.x lands)

### Tasks

- [x] **T1.1** Wait for riir-chain Plan 004 Phase 1 to confirm `elan` is in the dev workflow
  - **Status (2026-06-25):** DONE — riir-chain Plan 004 is COMPLETE (Phases 1-5). `elan`/`lean`/`lake` on PATH, Lean 4.31.0. Unblocks this plan.
- [x] **T1.2** Create `katgpt-rs/.proofs/` with `lakefile.toml` declaring `KatgptProof`
  - **Status (2026-06-25):** DONE — `lakefile.toml` created with `[[require]] mathlib` (required for transcendental sigmoid analysis; `lean-toolchain` auto-bumped to `v4.32.0-rc1` by Mathlib).
- [x] **T1.3** Pin same Lean 4 version as riir-chain `.proofs/lean-toolchain`
  - **Status (2026-06-25):** DEVIATION (documented) — Mathlib forces `leanprover/lean4:v4.32.0-rc1`, higher than riir-chain's `v4.31.0`. Unavoidable: riir-chain avoids Mathlib (integer arithmetic, `omega`-decidable); sigmoid monotonicity needs Mathlib's `Real.exp` analysis. Documented in `.proofs/README.md`.

---

## Phase 2 — ActionBridge Spec in Lean

### Tasks

- [x] **T2.1** Create `katgpt-rs/.proofs/KatgptProof/Bridge/Basic.lean`
  - **Status (2026-06-25):** DONE — `dot {ι : Type*} [Fintype ι] (q d : ι → ℝ) : ℝ := ∑ i, q i * d i` mirroring the Rust `mul_add` accumulation. Sigmoid = Mathlib's `Real.sigmoid` (no re-definition — `x.sigmoid = (1 + Real.exp (-x))⁻¹` IS the spec).
- [x] **T2.2** Define `dot {D : ℕ} (q d : Fin D → Float32) : Float32` mirroring `mul_add` loop
  - **Status (2026-06-25):** DONE (generalized) — modeled over `ℝ` with a generic finite index type `ι` rather than `Float32`. Rationale: the ranking-preservation property holds for the *mathematical* sigmoid over `ℝ`; `Float32` is a libm approximation documented in the spec-match test. This mirrors riir-chain's approach (model over `Int`/`Real`, not raw Rust types).
- [x] **T2.3** Define `sigmoid (x : Float32) : Float32` matching `simd::fast_sigmoid` (bounded (0,1), libm-exp) — document the approximation tolerance in a separate `sigmoid_approx.lean`
  - **Status (2026-06-25):** DONE (via Mathlib) — uses Mathlib's `Real.sigmoid` directly, which is `1/(1+exp(-x))` — the exact mathematical object the Rust `fast_sigmoid` approximates. The `|x|>40` saturation is an f32 concern, documented in `Basic.lean` and tested in `bridge_spec_match.rs`. No separate `sigmoid_approx.lean` needed — Mathlib IS the authoritative definition.
- [x] **T2.4** State the ranking theorem:
  ```lean
  theorem action_bridge_ranking_preserved
    {D : ℕ} (q d₁ d₂ : Fin D → Float32)
    (h : dot q d₁ > dot q d₂) :
    sigmoid (dot q d₁) > sigmoid (dot q d₂) := by
    exact strictMono_sigmoid _ _ h
  ```
  - **Status (2026-06-25):** DONE (generalized to `ι`) — `action_bridge_ranking_preserved {ι} [Fintype ι] (q d₁ d₂ : ι → ℝ) (h : dot q d₁ > dot q d₂) : sigmoid (dot q d₁) > sigmoid (dot q d₂) := Real.sigmoid_lt h`. Plus corollaries `action_bridge_ranking_preserved'` and `action_bridge_argmax_preserved`.

---

## Phase 3 — Proof & Spec-Match

### Tasks

- [x] **T3.1** Provide `strictMono_sigmoid` (1 Mathlib lemma, or 5-line hand-proof if Mathlib's `Real.strictMono_sigmoid` isn't in Float32 form yet)
  - **Status (2026-06-25):** DONE via Mathlib — `Mathlib.Analysis.SpecialFunctions.Sigmoid` ships `Real.sigmoid_strictMono : StrictMono sigmoid` and `Real.sigmoid_lt : a < b → sigmoid a < sigmoid b`. No hand-proof needed; Mathlib is the standard source for transcendental analysis. Axioms = `{propext, Classical.choice, Quot.sound}` (Mathlib's standard foundations).
- [x] **T3.2** Create `katgpt-rs/tests/bridge_spec_match.rs` gated by `action_bridge`:
  - assert `ActionBridge::select_action` calls `simd::fast_sigmoid` (verify by reading source via `#[doc]` or by static call graph)
  - assert no softmax anywhere in the bridge module (grep-equivalent compile-time check via trait bounds)
  - **Status (2026-06-25):** DONE — `tests/bridge_spec_match.rs` with 6 tests: `spec_fast_sigmoid_matches_mathlib_real_sigmoid` (spec match), `spec_fast_sigmoid_saturation_boundary` (saturation contract), `spec_select_action_uses_fast_sigmoid` (behavioural call-graph check: argmax + score == fast_sigmoid(dot)), `spec_no_softmax_in_bridge` (no softmax normalisation: identical logits → σ(dot) not 0.5), `empirical_ranking_preserved_within_f32_precision` (flip-detection over 10K pairs, ties allowed), `proofs_directory_exists` (sentinel). Gated `#![cfg(feature = "action_bridge")]`.
- [x] **T3.3** G3 — `cargo test --features action_bridge --test bridge_spec_match` passes
  - **Status (2026-06-25):** DONE — 6/6 PASS. Required adding `action_bridge = ["katgpt-core/action_bridge"]` to root `Cargo.toml` (was missing — only katgpt-core declared it).

---

## Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Proof is offline; bridge is inference-time |
| Latent-to-latent preferred | ✅ Operates on Q-value vectors, projects to scalar |
| Sigmoid not softmax | ✅ This is *the* sigmoid proof |
| Freeze/thaw over fine-tuning | N/A |
| 4-repo discipline | ✅ Open primitive, no chain/shard IP |
| Zero-alloc hot path | ✅ Proof is offline; bridge unchanged |
| File size < 2048 lines | ✅ < 100 lines per `.lean` file |

---

## TL;DR

Open primitive Tier 3. 5-line Lean 4 proof that `ActionBridge::select_action` ranking is preserved by sigmoid — the property Plan 262 G1.3 currently asserts over 1000 random triples. Establishes the second Lean toolchain instance (after riir-chain Plan 004) and the first one in the public MIT repo. Sets pattern for harder proofs (convexity of softmax-free attention, etc.). **Public math; the value is the integration pattern, not the theorem.**
