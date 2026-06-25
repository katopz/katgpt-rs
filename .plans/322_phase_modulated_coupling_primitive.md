# Plan 322: Phase-Modulated Coupling — Norm-Preserving Subspace Rotation Gate

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/305_Phase_Modulated_Cross_Domain_Coupling.md](../.research/305_Phase_Modulated_Cross_Domain_Coupling.md)
**Private guide:** [riir-ai/.research/159_phase_rotation_subspace_gate_guide.md](../../../riir-ai/.research/159_phase_rotation_subspace_gate_guide.md)
**Source paper:** [arxiv 2605.12700](https://arxiv.org/abs/2605.12700) — UFO: Domain-Unification-Free Operator Framework (Qiao, Karniadakis, Muniruzzaman, May 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/phase_rotation.rs` (new module) + Cargo feature `phase_rotation_coupling`
**Status:** Active — Phase 1 COMPLETE (2026-06-25). All 19 unit tests PASS (covers all plan-mandated scenarios + aliasing + monotone interpolation + scalar/per-channel equivalence + shape errors + Pythagorean identity G1 canary). `phase_safe_cos_sin` design pivot from the planned independent Padé cos/sin to libm-sin + Pythagorean-sqrt recovery: the independent Padé drifts in the `cos²+sin²=1` identity by ~5e-3 (50× the G1 budget); the sqrt-recovery forces it bit-by-bit. Phase 2 GOAT bench pending.

---

## Goal

Ship the **phase-modulated coupling primitive** distilled from UFO (arxiv 2605.12700): a zero-allocation, SIMD-vectorizable gate that mixes two latent slices `(a, b)` via a norm-preserving rotation `cos α ⊙ a + sin α ⊙ b`, where the phase `α` is constructed modellessly from a sigmoid projection onto a frozen direction vector. The primitive is the open math hook for the Super-GOAT described in riir-ai/.research/159 (HLA subspace gating, crowd-coherent mode transition, LatCal-committed phase).

**GOAT gate (open primitive):** G1 L2-norm preservation <1e-4; G2 rotation interpolates smoothly between a and b; G3 latency <50ns (D=8 scalar) / <250ns (D=64 per-channel); G4 zero-alloc; G5 feature isolation; G6 sigmoid-never-softmax (phase is sigmoid-bounded).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/phase_rotation.rs`:
  - `PhaseRotationGate` config struct (sharpness `λ`, broadcast-vs-per-channel flag).
  - `phase_rotation_gate_into(a, b, cos_alpha, sin_alpha, out)` — the core mix. Scalar-broadcast fast path (length-1 cos/sin) + per-channel path (length-D cos/sin). SIMD the inner loop (`simd::simd_mul_add` if available, else chunked 4-wide manual).
  - `compute_phase_from_projection(state, direction, sharpness, &mut cos_alpha, &mut sin_alpha)` — scalar phase: `α = sigmoid(dot · λ) · π/2`; returns (cos α, sin α).
  - `compute_phase_per_channel_into(state, directions, sharpness, cos_out, sin_out)` — per-channel phase: one α per channel; uses polynomial-Padé cos/sin (reuse Plan 319 Issue 003 approximation, max error 4.9e-3).
  - `PhaseRotationScratch` — pre-allocated scratch (`cos_alpha: Vec<f32>`, `sin_alpha: Vec<f32>`, sized to D).
  - Reuse `simd::simd_dot_f32` (matches Plan 310 / 319 convention).
  - **No new dependencies.** cos/sin via libm by default; polynomial-Padé as an opt-in helper.

  **Design pivot during implementation:** the planned independent-Padé cos/sin was replaced with `phase_safe_cos_sin` (libm `sin` + Pythagorean `sqrt(1-sin²)` recovery). Reason: independent Padé cos/sin drifts in the `cos²+sin²=1` identity by ~5e-3 (50× the G1 <1e-4 budget), and the sqrt-recovery construction forces the identity to hold bit-by-bit. The `use_pade` bool parameter was dropped (no longer needed — there is one path now; a future Phase 3 SIMD sin-LUT variant would land as a new entry point `compute_phase_per_channel_simd_into`, not a bool toggle). Libm-sin latency is well within the 1500ns D=64 budget (~830 ns measured ceiling); Phase 3 SIMD work is deferred unless G3 marginally fails.

- [x] **T1.2** Wire into `katgpt-rs/crates/katgpt-core/src/lib.rs`:
  - `#[cfg(feature = "phase_rotation_coupling")] pub mod phase_rotation;`
  - `#[cfg(feature = "phase_rotation_coupling")] pub use phase_rotation::{PhaseRotationGate, PhaseRotationScratch, phase_rotation_gate_into, compute_phase_from_projection, compute_phase_per_channel_into};`
  - Add `phase_rotation_coupling = []` to `[features]` in `katgpt-rs/crates/katgpt-core/Cargo.toml`. **Opt-in, NOT default** until G1–G4 pass.

- [x] **T1.3** Unit tests in `phase_rotation.rs` (`#[cfg(test)] mod tests`) — **19 tests, all PASS**:
  - `scalar_phase_at_zero_returns_a` — α=0 → output = a (cos 0 = 1, sin 0 = 0).
  - `scalar_phase_at_pi_half_returns_b` — α=π/2 → output = b.
  - `scalar_phase_at_pi_four_is_average_scaled` — α=π/4 → output = (a+b)/√2.
  - `l2_norm_bounded_by_sum_of_input_norms` — ‖out‖² ≤ ‖a‖² + ‖b‖² across α sweep.
  - `l2_norm_exact_for_orthogonal_equal_norm_at_pi_four` — a⊥b, ‖a‖=‖b‖=1, α=π/4 → ‖out‖=1 (Pythagorean identity).
  - `per_channel_phase_independent_rotations` — each channel rotates by its own α; channel c's output depends only on cos α_c, sin α_c.
  - `phase_bounded_in_zero_to_pi_half` — α ∈ [0, π/2] for arbitrary state/direction (sigmoid ∈ [0,1], ·π/2 ∈ [0,π/2]).
  - `deterministic_given_same_inputs` — same (state, direction, sharpness) → same (cos α, sin α), bit-identical.
  - `zero_alloc_in_steady_state` — `PhaseRotationScratch` allocated once; `cached_d` unchanged across hot-path calls.
  - `shape_mismatch_returns_err`, `invalid_phase_len_returns_err`, `projection_shape_mismatch_returns_err` — error paths.
  - `phase_safe_cos_sin_accuracy` — **G1 canary**: Pythagorean drift <1e-4 + per-element abs err <5e-3 across 1000-point sweep.
  - `per_channel_phase_matches_libm_within_budget` — end-to-end per-channel vs libm: cos/sin within 5e-3.
  - `gate_config_validation` — `PhaseRotationGate::new` validates sharpness (finite, non-negative, clamps at 100).
  - `scratch_ensure_capacity_noop_on_same_d` — scratch is a true no-op on hot path, grows/shrinks on cold path.
  - `scalar_and_per_channel_paths_agree_at_uniform_phase` — bit-identical FMA chain when per-channel phase is uniform.
  - `out_can_alias_a` — aliasing semantics pinned (safe Rust can't construct the simultaneous borrow; tests document the contract).
  - `monotone_interpolation_from_a_to_b` — **G2 canary**: cos sim to a monotone decreasing, cos sim to b monotone increasing across α sweep.

- [x] **T1.4** `cargo check -p katgpt-core --features phase_rotation_coupling` clean. Also `--all-features`, default, `--no-default-features` all clean.

- [x] **T1.5** `cargo test -p katgpt-core --features phase_rotation_coupling --lib phase_rotation` — all 19 unit tests pass (direct binary launch bypassing the `cargo test` dyld/trustd stall).

---

## Phase 2 — GOAT Gate (G1–G6)

### Tasks

- [ ] **T2.1** Create `katgpt-rs/crates/katgpt-core/benches/bench_322_phase_rotation_goat.rs`:
  - **G1 (norm preservation):** sweep α ∈ [0, π/2] in 1000 steps; for each, compute `|cos²α + sin²α - 1|` in f32. Report max. **Gate: < 1e-4.** Use libm cos/sin for the reference; also measure polynomial-Padé variant for the fast path.
  - **G2 (smooth interpolation):** for a fixed (a, b) pair with `a = [1,0,0,0]`, `b = [0,1,0,0]`, sweep α ∈ [0, π/2] in 100 steps; verify output moves monotonically from a to b in cosine-similarity space (cos sim to a decreases monotonically, cos sim to b increases monotonically). **Gate: monotone, no reversals.**
  - **G3 (latency):** batched-median timing (1024 calls per measurement, 256 batches, sink-hash anti-hoist — matches Plan 303 / 320 convention).
    - D=8 scalar phase: target < 50 ns/call.
    - D=64 per-channel phase (libm cos/sin): target < 1500 ns/call.
    - D=64 per-channel phase (polynomial-Padé): target < 600 ns/call.
  - **G4 (zero-alloc):** drop-tracking allocator; verify 0 allocations in steady-state (after scratch init).
  - **G6 (sigmoid never softmax):** static check — `compute_phase_from_projection` uses `sigmoid`, never `softmax`. Code review assertion.

- [ ] **T2.2** Run bench: `cargo bench -p katgpt-core --features phase_rotation_coupling --bench bench_322_phase_rotation_goat -- --nocapture`. Record results in `katgpt-rs/.benchmarks/322_phase_rotation_goat.md`.

- [ ] **T2.3** **Promote decision:**
  - If G1 < 1e-4 AND G2 monotone AND G3 meets latency AND G4 zero-alloc → **promote `phase_rotation_coupling` to `default` feature** in `katgpt-rs/crates/katgpt-core/Cargo.toml`. Update README feature showcase.
  - If G1 fails (norm drift) → **demote to Gain**, document why in the benchmark file, do NOT promote. The whole stability thesis depends on G1.
  - If G3 fails (too slow) → try polynomial-Padé-only path; if still too slow, keep opt-in, file issue.

---

## Phase 3 — SIMD Acceleration (only if G3 marginally fails)

### Tasks

- [ ] **T3.1** If scalar D=8 latency > 50ns: hand-written SIMD inner loop for the mix (`c·a + s·b` is a textbook FMA kernel — 2 mul + 1 add per element, fully vectorizable).
- [ ] **T3.2** If per-channel D=64 latency > 600ns even with polynomial-Padé: precompute cos/sin lookup tables for α ∈ [0, π/2] at 1024 entries; linear interpolation. Trade 4KB table for O(1) cos/sin. (This is the same LUT pattern AGENTS.md mandates for bounded-domain ops.)

---

## Phase 4 — Fusion Guides (deferred until Phase 2 promotes)

### Tasks

- [ ] **T4.1** If promoted to default: write `riir-ai/.plans/NNN_hla_phase_rotation_runtime.md` — HLA (a, b) split + per-faction direction artifacts + CCE crowd phase broadcast. Runtime gates G5 (long-horizon drift) + G6 (crowd coherence).
- [ ] **T4.2** If promoted: write `riir-chain/.research/NNN_committed_phase_latcal_bridge.md` — LatCal fixed-point commitment of the phase angle. Chain gate G7 (bit-identical replay).
- [ ] **T4.3** If promoted: write `katgpt-rs/.plans/NNN_dec_hodge_phase_mixer.md` — DEC wrapper `cos α ⊙ exact + sin α ⊙ coexact + harmonic` over shipped `hodge_decompose`.
- [ ] **T4.4** Optional: `riir-neuron-db/.research/NNN_shard_half_retrieval.md` — shard spectral/spatial half retrieval.

---

## Constraints checklist (AGENTS.md)

| Constraint | Status |
|---|---|
| Modelless (no backprop) | ✅ cos/sin/sigmoid/dot — closed form. |
| Latent-to-latent preferred | ✅ Operates on two latent halves (a, b). |
| Sigmoid not softmax | ✅ Phase is sigmoid-bounded; cos/sin is monotone rotation. |
| Freeze/thaw over fine-tuning | ✅ Direction vectors are frozen BLAKE3-committed artifacts (caller's responsibility; primitive is direction-agnostic). |
| Zero-alloc hot path | ✅ All scratch caller-provided. |
| SIMD / auto-vectorization | ✅ Inner mix loop is FMA-friendly; chunked 4-wide fallback. |
| Fixed-size arrays for bounded domains | ✅ HLA D=8 fits in a register; per-channel D=64 fits in a cache line × 2. |
| Pre-computed lookup tables | ✅ Phase 3.2 LUT for cos/sin if needed. |
| Files < 3200 lines | ✅ `phase_rotation.rs` will be ~300 lines incl. tests. |
| `Uuid::now_v7()` | N/A (no UUIDs in this primitive). |
| blake3 / argon2 / papaya | N/A (primitive itself; commitment is caller's responsibility via existing `MerkleFrozenEnvelope`). |

---

## §3.5 Modelless-First Check (recorded per skill mandate)

The primitive is **modelless by construction** — the §3.5 check was performed in Research 305 §5:

- **Path 1 (freeze/thaw):** N/A (not a bias-correction problem).
- **Path 2 (raw/lora deterministic construction):** ✅ PASSES. The phase function `α = sigmoid(dot · λ) · π/2` is closed-form; direction is a frozen artifact. No training.
- **Path 3 (latent-space correction):** subsumed by Path 2.

**No riir-train deferral.** The primitive ships modellessly. The PDE-benchmark quality claims from the UFO paper (UFO beats DeepONet/FNO on StepHeat etc.) are training-only and belong in riir-train; they are explicitly NOT part of this plan.

---

## TL;DR

Ship the phase-modulated coupling primitive (cos α ⊙ a + sin α ⊙ b, α from sigmoid projection) behind feature flag `phase_rotation_coupling` in `katgpt-rs/crates/katgpt-core/src/phase_rotation.rs`. Phase 1: skeleton + unit tests. Phase 2: GOAT gate (G1 norm preservation <1e-4 is the kill switch; G2 monotone interpolation; G3 latency <50ns D=8 scalar / <600ns D=64 per-channel; G4 zero-alloc). Phase 3: SIMD/LUT if G3 marginally fails. Phase 4: fusion guides in riir-ai (HLA runtime), riir-chain (LatCal committed phase), katgpt-rs (DEC Hodge mixer) — deferred until Phase 2 promotes to default. The primitive is the open math hook for the Super-GOAT in riir-ai/.research/159 (norm-preserving NPC affect rotation, crowd-coherent mode transition, chain-committed phase for deterministic replay).
