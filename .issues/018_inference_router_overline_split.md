# Issue 018: `src/inference_router.rs` Exceeds 2048-Line Limit (2195 lines)

**Source**: Recurring AGENTS.md rule — "Keep files less than 2048 lines for `.rs` file as possible"
**Priority**: Medium (not blocking; refactor opportunity)
**Blocked**: No
**Depends**: Nothing — pure mechanical extraction

## Summary

`src/inference_router.rs` is **2195 lines** — 147 lines (≈7%) over the 2048-line ceiling in the user's `AGENTS.md` Rust rules. Build is clean, all functionality works; this is a maintainability/refactor concern only.

## Current State

```
$ wc -l src/inference_router.rs
2195 src/inference_router.rs
```

The file concentrates:
- `InferenceRouter` struct + impl (the central dispatch)
- `ComputeTarget` enum + `route_by_module_energy` (Plan 264 T4.1-T4.2)
- `ModuleEnergyProfile` (Plan 264 T4.3, `#[cfg(feature = "module_energy_route")]`)
- `TvpSignal`/`TvpConfig` fields + `tier_after_tvp` gate + `update_tvp()` (Plan 267 T9-T11, `#[cfg(feature = "thicket_variance_probe")]`)
- `~2500 lines` of inline `mod tests` (router tier-promotion/demotion tests, TVP integration tests, RV ablation)

## Proposed Split

Extract cohesive sub-systems into sibling files under `src/inference_router/` (or `src/router_*.rs` if keeping flat layout). Each extraction must be feature-gate-preserving and zero-cost when the relevant feature is off.

| New file | Contents | Est. lines moved |
|----------|----------|------------------|
| `src/router_compute_target.rs` | `ComputeTarget`, `route_by_module_energy`, `ModuleEnergyProfile` (Plan 264) | ~150 |
| `src/router_tvp.rs` | `TvpSignal`/`TvpConfig` wiring, `tier_after_tvp` gate, `update_tvp()` (Plan 267) | ~200 |
| `src/router_tests.rs` (or `tests/inference_router_*.rs`) | Inline `mod tests` block | ~800 |

After extraction, `inference_router.rs` should land at ≈1000-1100 lines (struct + core `forward()` + accessor methods), well under the 2048 ceiling.

## Constraints

- **Feature gates MUST be preserved** — every `#[cfg(feature = "...")]` block moves with its code.
- **Zero-cost when features off** — the disabled-feature path is currently `let tier_after_tvp = tier_after_critical` (single binding). Splitting must not introduce any codegen change.
- **Public API MUST NOT change** — `InferenceRouter`, `ComputeTarget`, `TvpSignal`, `TvpConfig` re-exported from the same paths. Use `pub use` in `src/lib.rs` if needed.
- **Tests MUST stay green** — `cargo test --lib --features thicket_variance_probe -- collapse_detector` (20 tests), `cargo test --lib --features module_energy_route -- inference_router` (existing router tests).

## Out of Scope

- Logic changes — pure mechanical move.
- New abstractions — only file extraction, no new traits/structs.
- Default-feature promotions.

## Verification

After the refactor:

```bash
# Must all pass with no regression
cargo build --lib
cargo test --lib
cargo test --lib --features thicket_variance_probe -- inference_router
cargo test --lib --features module_energy_route -- inference_router
cargo test --lib --features rv_gated_routing -- inference_router
```

## Why This Is an Issue, Not a Plan

Per user rule: *"Create issue at ./issues for optimization task, do not create plan."* This is a mechanical optimization, not a research/architecture task.

---

## TL;DR

`src/inference_router.rs` is 2195 lines (147 over ceiling). Extract `ComputeTarget`, `TvpSignal`/`TvpConfig`, and the inline `mod tests` into sibling files. Pure mechanical move; no logic change; preserve all `#[cfg]` gates; keep public API stable.
