# Plan 293: ActionBridge Lean 4 Monotonicity Proof

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md](../.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md)
**Source:** Bridge neuro-symbolic gap analysis (user prompt 2026-06-23)
**Target:** `katgpt-rs/.proofs/` (new top-level dir) + `katgpt-rs/tests/bridge_spec_match.rs`
**Status:** Active — Phase 1 (P1, follows riir-chain Plan 004)

---

## Goal

Prove `∀ a b, dot a > dot b ⟺ sigmoid (dot a) > sigmoid (dot b)` in Lean 4 — the ranking-preservation property that Plan 262 G1.3 currently asserts over 1000 random triples. This is Tier 3 of the bridge FV strategy; it is the open katgpt-rs primitive. Promotes the empirical G1.3 test from `∃` to `∀`.

**GOAT gate:** G1–G3 (toolchain bootstraps, theorem type-checks, Rust spec matches Lean). Promotion: default-on docs reference once all three pass.

---

## Phase 1 — Lean Toolchain Bootstrap (after riir-chain Plan 004 T1.x lands)

### Tasks

- [ ] **T1.1** Wait for riir-chain Plan 004 Phase 1 to confirm `elan` is in the dev workflow
- [ ] **T1.2** Create `katgpt-rs/.proofs/` with `lakefile.toml` declaring `KatgptProof`
- [ ] **T1.3** Pin same Lean 4 version as riir-chain `.proofs/lean-toolchain`

---

## Phase 2 — ActionBridge Spec in Lean

### Tasks

- [ ] **T2.1** Create `katgpt-rs/.proofs/KatgptProof/Bridge/Basic.lean`
- [ ] **T2.2** Define `dot {D : ℕ} (q d : Fin D → Float32) : Float32` mirroring `mul_add` loop
- [ ] **T2.3** Define `sigmoid (x : Float32) : Float32` matching `simd::fast_sigmoid` (bounded (0,1), libm-exp) — document the approximation tolerance in a separate `sigmoid_approx.lean`
- [ ] **T2.4** State the ranking theorem:
  ```lean
  theorem action_bridge_ranking_preserved
    {D : ℕ} (q d₁ d₂ : Fin D → Float32)
    (h : dot q d₁ > dot q d₂) :
    sigmoid (dot q d₁) > sigmoid (dot q d₂) := by
    exact strictMono_sigmoid _ _ h
  ```

---

## Phase 3 — Proof & Spec-Match

### Tasks

- [ ] **T3.1** Provide `strictMono_sigmoid` (1 Mathlib lemma, or 5-line hand-proof if Mathlib's `Real.strictMono_sigmoid` isn't in Float32 form yet)
- [ ] **T3.2** Create `katgpt-rs/tests/bridge_spec_match.rs` gated by `action_bridge`:
  - assert `ActionBridge::select_action` calls `simd::fast_sigmoid` (verify by reading source via `#[doc]` or by static call graph)
  - assert no softmax anywhere in the bridge module (grep-equivalent compile-time check via trait bounds)
- [ ] **T3.3** G3 — `cargo test --features action_bridge --test bridge_spec_match` passes

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
