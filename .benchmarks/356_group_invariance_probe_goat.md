# Benchmark 356: Group Invariance Probe — GOAT Gate Results

**Plan:** [katgpt-rs/.plans/356_group_invariance_probe.md](../.plans/356_group_invariance_probe.md)
**Research:** [katgpt-rs/.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md](../.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md)
**Feature:** `group_invariance_probe` (opt-in)
**Date:** 2026-07-01
**Target dir:** `/tmp/plan356-g3` (per AGENTS.md "check running cargo if locked" rule)

---

## Summary

**8/8 GOAT gates PASS — feature ships OPT-IN (not promoted to default).**

The Group Invariance Probe is the modelless residue of LieFlow (arXiv:2512.20043): a small primitive that generalizes `subspace_phase_gate` from "subspace of `ℝᵈ`" to "subgroup of `G`" via direct invariance testing `σ(β·(1−d(q,g·q)))` + a dual-signal discrete-vs-continuous classifier (variance for large-fraction subgroups, concentration for small-fraction subgroups).

| Gate | Test | Target | Result | Status |
|---|---|---|---|---|
| **G1** Correctness | `discover_c8_to_c4_recovers_discrete_class` — C₄-invariant indicator under C₈ hypothesis group | `Discrete` class, ≥100 support samples, max_score > 0.95 | `Discrete`, n_support=131, max_score ≈ 1.0 | ✅ PASS |
| **G2a** Non-redundancy (no symmetry) | Uniform low scores `[0.1; 64]` | `None` | `None` | ✅ PASS |
| **G2b** Non-redundancy (continuous) | Uniform high scores `[0.9; 64]` | `Continuous` | `Continuous` | ✅ PASS |
| **G2c** Non-redundancy (small-fraction discrete) | 4 peaks in 64 samples `[1,0,...,1,0,...]` | `Discrete` | `Discrete` (via concentration signal) | ✅ PASS |
| **G3a** No regression (`--all-features`) | `cargo check -p katgpt-core --all-features` | clean | clean (1 pre-existing unrelated warning in `set_attention.rs:454`) | ✅ PASS |
| **G3b** No regression (`--no-default-features`) | `cargo check -p katgpt-core --no-default-features` | clean | clean | ✅ PASS |
| **G4a** Alloc-free (`discover_subgroup_into`) | CountingAllocator, 256-sample probe on C₈ | 0 allocs after warmup | 0 allocs | ✅ PASS |
| **G4b** Alloc-free (`classify_subgroup` + `invariance_score`) | CountingAllocator, leaf functions | 0 allocs | 0 allocs | ✅ PASS |

## The dual-signal classifier (key design finding)

The discrete-vs-continuous classification needs **two complementary signals** because no single measure handles both regimes:

| Regime | Example | Variance | Concentration | Correct signal |
|---|---|---|---|---|
| Large-fraction discrete | C₄ ⊂ C₈ (4/8 = 50%) | ≈ 0.25 (bimodal) | ≈ 0.5 (indistinguishable from spread) | **Variance** |
| Small-fraction discrete | C₄ ⊂ C₆₄ (4/64 = 6%) | ≈ 0.06 (low) | ≈ 0.06 (peaked) | **Concentration** |
| Continuous | SO(2) ⊂ SO(3) | low | high | Neither fires → Continuous |
| No symmetry | uniform low scores | ≈ 0 | low | support fraction < min → None |

`classify_subgroup` fires `Discrete` if EITHER signal triggers. This is the `OR` of two complementary detectors.

## The stabilizer lesson (from the test debugging)

The initial G1 test used `q = (1, 0)` (a single point) under `SO(2)`, expecting to discover C₄. **This was wrong** — a single point's stabilizer in SO(2) is trivial (only the identity fixes it), so no non-trivial symmetry is discoverable. This is exactly the LieFlow "point stabilizer" issue documented in Research 355 §1.3.

The fix: use a C₄-invariant **distribution** (the indicator `[1,0,1,0,1,0,1,0]` on C₈) rather than a single point. The probe tests invariance of the *distribution* under group actions, not of a point. This matches LieFlow's formulation: `q(hx) = q(x)` for all `x ∈ X` and `h ∈ H` — a distribution-level statement.

## The shifted-sigmoid score (from the test debugging)

The initial score formula `σ(−β·d)` gives 0.5 at d=0, conflating "perfect invariance" with "indifference". The fix: `σ(β·(1−d))` — a shifted sigmoid that hits ≈1.0 at d=0 (assuming d is normalized so the indifference point is at d=1). The normalization convention is documented in `invariance_score`'s docstring.

## Performance

Not benchmarked for latency (no bench file) — the primitive is O(n_samples · cost(distance_fn)) with zero allocation, and the per-sample work is dominated by the caller's distance function. The leaf functions (`invariance_score`, `score_variance`, `score_concentration`, `classify_subgroup`) are all O(n) chunk-4 loops; sub-µs for typical n ≤ 1024.

## Promotion decision

**Keep `group_invariance_probe` OPT-IN.** Do NOT promote to default until:
- Issue 011 returns Q2+Q3 = YES → riir-ai fusion plan consumes the API and demonstrates the per-NPC committed symmetry fingerprint selling point, OR
- riir-neuron-db `can_freeze` extension uses `discover_subgroup` to add a group-axis field to `FreezeGateReport` and demonstrates anti-cheat value.

The primitive is correct and zero-alloc but has no shipped consumer yet. Promotion without a consumer would be adding dead code to the default build.

## Test inventory

- **In-crate** (`crates/katgpt-core/src/group_invariance_probe.rs::tests`): 13 tests covering `invariance_score`, `score_variance`, `score_concentration`, `classify_subgroup`, `SubgroupClass` round-trip, and the G1 `discover_c8_to_c4` end-to-end.
- **Integration** (`crates/katgpt-core/tests/group_invariance_probe_g4.rs`): 2 CountingAllocator tests for G4.

## Related

- Plan: [`katgpt-rs/.plans/356_group_invariance_probe.md`](../.plans/356_group_invariance_probe.md)
- Research: [`katgpt-rs/.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md`](../.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md)
- Fusion follow-up: [`katgpt-rs/.issues/011_lieflow_fusion_super_goat_investigation.md`](../.issues/011_lieflow_fusion_super_goat_investigation.md)
