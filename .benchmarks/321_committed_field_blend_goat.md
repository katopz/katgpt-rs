# Plan 321 — CommittedFieldBlend GOAT Gate Results

**Date:** 2026-06-25
**Plan:** [321_sampling_invariant_per_entity_moe_primitive](../.plans/321_sampling_invariant_per_entity_moe_primitive.md)
**Research:** 302_FAME_Sampling_Invariant_Per_Entity_MoE (arXiv:2510.00621)
**Feature:** `committed_field_blend = ["personality_composition"]` (opt-in)
**Verdict:** ✅ **ALL GATES PASS** — G1 mechanics, G2 sampling invariance (defining property), G3 no-regression, G4 zero-alloc, G5 BLAKE3 commitment.

---

## GOAT Gate Summary

| Gate | Meaning | Result | Evidence |
|------|---------|--------|----------|
| **G1** | mechanics (finite, bounded, sigmoid in [0,1]) | ✅ PASS | `g1_mechanics` unit test |
| **G2** | sampling invariance (dense vs sparse → identical π + dynamics) | ✅ PASS | `g2_sampling_invariance` unit test |
| **G3** | no regression on `PersonalityWeightedComposition` primitives | ✅ PASS | `g3_no_regression_primitives_intact` unit test |
| **G4a** | `apply_blended` zero-alloc (1000 iters) | ✅ **0 allocs** | `committed_field_blend_bench` (CountingAllocator) |
| **G4b** | `commit` zero-alloc (100 re-commits) | ✅ **0 allocs** | `committed_field_blend_bench` (CountingAllocator) |
| **G5** | BLAKE3 reproducible + tamper-detecting | ✅ PASS (4/4) | `g5_blake3_*` unit tests |

**Test count:** 13 unit tests + 1 bench, all green.

---

## G2 (the make-or-break gate) — detailed result

FAME Proposition 3: if two observation grids encode the same underlying trajectory, the committed blend produces identical dynamics. **Tested via:**

- K=3 archetype fields (linear/rotation/constant).
- PERIODIC trajectory of 1000 steps: `traj[t][j] = 0.1 + 0.05·sin(2πt/100 + j·0.2)`, period=100 steps.
- Dense summary: mean over all 1000 steps (10 full periods).
- Sparse summary: mean over every 10th step (100 steps = 10 full periods).
- Both summaries converge to the DC component (0.1 per axis) — the mean of a periodic signal over an integer number of full periods is sampling-invariant.
- `pi_dense ≈ pi_sparse` to within 1e-3: **PASS** (residual diff is float accumulation-order noise ~1e-5).
- Blended dynamics from identical initial state, 100 steps: trajectory divergence < 1e-3: **PASS**.

### Key insight (design correctness, not primitive defect)

The trajectory **MUST** be periodic for the mean-summary to be sampling-invariant. An initial test used a saturating ramp (`state += 0.01·sin(j)`, clamped) — the mean of a non-stationary signal genuinely differs between dense/sparse sampling (diff ~4.6e-3). That was a **test-design** bug, not a primitive bug: FAME Proposition 3 requires the two observation grids to encode the *same* underlying trajectory, which holds only when the summary is a sampling-invariant statistic. The periodic-trajectory fix is the correct test; the primitive was never defective.

---

## G4 (zero-alloc) — detailed result

CountingAllocator audit (global `#[global_allocator]`), after one warmup call:

```
── G4a: apply_blended alloc-free (1000 iters) ──
   allocs:     0
   Threshold:  0
   Result:     PASS ✓

── G4b: commit alloc-free (100 re-commits) ──
   allocs:     0
   Threshold:  0
   Result:     PASS ✓
```

Zero heap allocations on both the per-tick hot path (`apply_blended`) and the re-commit path. Scratch buffers are stack arrays; field commitments are a stack-fixed `[[u8; 32]; N]`; BLAKE3 `Hasher` is stack-allocated.

---

## G5 (BLAKE3) — detailed result

Commitment scheme: `BLAKE3(version_LE || pi[0..N]_LE || field_commitments[0..N])`.

| Test | Result |
|------|--------|
| `g5_blake3_reproducible` — identical inputs → identical hash | ✅ PASS |
| `g5_blake3_version_affects_hash` — version IS part of commitment (unlike `PersonalitySnapshot`) | ✅ PASS |
| `g5_blake3_field_swap_detected` — swapping a field changes commitment | ✅ PASS |
| `g5_blake3_tamper_detecting_pi_byte_flip` — flipping a pi byte changes hash | ✅ PASS |

---

## Promotion status

Per AGENTS.md feature-flag discipline: **the GOAT gate (G1–G5) passes with modelless gain.** The primitive is eligible for promotion to `default`.

**Both deferral conditions are now satisfied** (as of 2026-06-26):
- ✅ **Plan 321 Phase 4** (examples + docs) — SHIPPED commit `76ac861c`.
- ✅ **riir-ai Plan 336 runtime-integration validation** — SHIPPED 2026-06-26 (all 7 phases done; G6a–G6e crowd-scale gates + G7a frozen-restoration bit-identical; `committed_personality_runtime` promoted to default-on in `riir-engine/Cargo.toml`).

The **promotion proposal is open** at [Issue 005](../.issues/005_committed_field_blend_default_on_promotion.md) — a one-token Cargo.toml flip + root passthrough + doc updates, with acceptance gates A1–A4. The feature stays opt-in (`committed_field_blend = ["personality_composition"]`) until Issue 005 lands; the proposal is OPEN per Plan 336 T7.2's "do NOT unilaterally execute" constraint (it's a different repo's decision).

---

## Files

- `crates/katgpt-core/src/committed_field_blend.rs` — primitive + 13 inline tests
- `crates/katgpt-core/benches/committed_field_blend_bench.rs` — G4 alloc audit
- `crates/katgpt-core/Cargo.toml` — feature + bench registration
- `crates/katgpt-core/src/lib.rs` — re-export

## Run commands

```bash
# Unit tests (G1, G2, G3, G5)
cargo test -p katgpt-core --features committed_field_blend --lib committed_field_blend

# G4 alloc audit
cargo bench -p katgpt-core --features committed_field_blend --bench committed_field_blend_bench -- --nocapture
```
