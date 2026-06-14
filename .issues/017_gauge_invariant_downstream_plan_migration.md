# Issue 017: Migrate Plans 094/201/233 to Consume `gauge_invariant` Primitive

**Source**: Plan 270 (`.plans/270_gauge_invariant_adapter_composition.md`) — Success Criteria "At least one downstream plan updated to use new primitive"
**Priority**: Medium
**Blocked**: No — primitive is default-ON and verified (GOAT 17/17 PASS)
**Depends**: Nothing in katgpt-rs. Cross-repo consumers (riir-ai, riir-train) need to pick up the new API.

## Summary

Plan 270 shipped three new modelless primitives in `katgpt-rs`, now promoted to **default-ON**:

- `ns_inv_sqrt_psd` (Newton-Schulz PSD inverse square root) — `src/newton_schulz.rs`
- `gauge_rebalance` (paper Algorithm 2) — `src/gauge_invariant.rs`
- `gauge_invariant_compose` (paper Algorithm 3) — `src/gauge_invariant.rs`

GOAT: **17/17 PASS** (gauge invariance under input rescaling within f32 ε, `AB^T` preserved, σ_max balance achieved, NS inv-sqrt roundtrip `P^{-1/2} P P^{-1/2} ≈ I`, throughput targets met).

The plan's Success Criteria explicitly calls for at least one downstream consumer to adopt the new primitive. This issue tracks that migration.

## Why This Is Tracked In katgpt-rs (Not The Consumer Repos)

The primitive lives in katgpt-rs and is the source of truth. The issue documents that the API is stable, default-ON, and ready for consumption. The actual migration edits happen in the consumer repos (riir-ai, riir-train) — they own their own plan files.

## Consumers & Migration Plan

### Plan 094 — TIES Merging (`riir-ai/crates/riir-gpu/src/merging.rs`)

**Current state**: TIES merging (Trim + Sign-Elect + Disjoint Merge at ρ=0.3) is implemented in riir-ai. Plan 094 T11 is marked `[x]`.

**Migration target**: When TIES is composed with other adapter-merge strategies (e.g., weighted average of TIES-merged adapters), the outer compose step should call `gauge_invariant_compose` instead of naive weighted sum. This eliminates magnitude drift when factor-pair gauges differ.

- [ ] **M1**: Audit `riir-ai/crates/riir-gpu/src/merging.rs` for multi-adapter compose call sites
- [ ] **M2**: Replace naive `Σ η_i · (A_i, B_i)` with `gauge_invariant_compose(&pairs, &mut out_a, &mut out_b)`
- [ ] **M3**: Add before/after benchmark — show `‖AB^T‖_F` stability improves on gauge-mismatched inputs

### Plan 201 — Rosetta Pruners (`katgpt-rs/src/pruners/`)

**Current state**: RosettaPruner is complete and uses cross-pruner *agreement* (not LoRA composition). Status: ✅ Implemented.

**Migration target**: **Likely N/A.** Rosetta mines universal constraint concepts from `ConstraintPruner` / `ScreeningPruner` outputs — it does not compose LoRA factor pairs. The `gauge_invariant` primitive is orthogonal.

- [ ] **M4**: Verify RosettaPruner has no latent adapter-composition code path that would benefit (quick grep). If none, close this sub-item as "not applicable".

### Plan 233 — Rosetta Cross-Game LoRA Alignment (`riir-train/.plans/233_rosetta_cross_game_lora_alignment.md`)

**Current state**: Lives in riir-train. Cross-game adapter alignment.

**Migration target**: When aligning adapters across games (Bomber ↔ Go ↔ Sudoku), the alignment step should rebalance factor pairs first via `gauge_rebalance`, then compose via `gauge_invariant_compose`. This ensures cross-game contributions are magnitude-comparable.

- [ ] **M5**: Audit `riir-train` cross-game alignment pipeline for adapter-compose call sites
- [ ] **M6**: Insert `gauge_rebalance` before each cross-game pair composition
- [ ] **M7**: Add regression test — verify cross-game merged adapter produces stable `‖AB^T‖_F` regardless of input factorization (paper Prop 1)

## Acceptance Criteria

- [ ] At least one of M1–M3 (Plan 094) **OR** M5–M7 (Plan 233) lands with a benchmark showing gauge-invariant compose produces more stable magnitudes than naive sum
- [ ] Plan 270 Success Criteria "At least one downstream plan updated" can be marked `[x]` with a commit reference

## Non-Goals

- Forcing migration where naive sum is provably equivalent (single-pair or pre-balanced inputs)
- Changing the public API of `gauge_invariant` — it's stable
- Promoting any consumer feature to default-on (that's each plan's own GOAT gate)

## Reference

- Plan 270: `.plans/270_gauge_invariant_adapter_composition.md` (✅ COMPLETE, default-ON)
- Primitive module: `src/gauge_invariant.rs` (default-ON)
- SparseTaskVector integration: `src/sparse_task_vector.rs::compose_gauge_invariant` (default-ON)
- GOAT proof: `tests/bench_270_gauge_invariant_goat.rs` (17/17 PASS)
- Paper: [LoRA-Muon (arXiv:2606.12921)](https://arxiv.org/pdf/2606.12921)
