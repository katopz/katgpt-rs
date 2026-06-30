# Issue 016 — `cargo check --no-default-features` broken by speculative re-export drift

## TL;DR

`cargo check --no-default-features --lib` on katgpt-rs root fails with 3 errors.
Root cause: `src/speculative/types.rs` L39-43 unconditionally re-exports
`EarlyStopGate`, `SpecCostSnapshot`, `StabilitySnapshot`, `RoutingOverlapSnapshot`
from `katgpt_core::speculative::types`, but in katgpt-core those types (and the
`DraftResult::{routing_overlap, cost_snapshot, stability}` fields) are gated
behind the `domain_latent` / `spec_cost_model` / `stability_metrics` features.
With `--no-default-features` all three features are off → the imports resolve
to nothing and the field initializers are missing the gated fields.

Discovered during Issue 355 Phase 2c (make `plotters` optional). Verified
pre-existing via `git stash` on baseline develop — the same 3 errors reproduce
without any Issue 355 changes. Phase 2c of Issue 355 is blocked on this until
resolved (the acceptance criterion is "`cargo check --no-default-features` is
clean on root").

## Reproduction

```bash
cd /git/katgpt-rs
CARGO_TARGET_DIR=/tmp/issue016 cargo check --no-default-features --lib
```

## Errors (verbatim)

```
error[E0432]: unresolved imports `katgpt_core::speculative::types::EarlyStopGate`,
              `katgpt_core::speculative::types::SpecCostSnapshot`,
              `katgpt_core::speculative::types::StabilitySnapshot`
   --> src/speculative/types.rs:40:77
    |
 40 |     BlockScores, BudgetAdaptation, DecodeStrategy, DraftEvent, DraftResult, EarlyStopGate,
    |                                                                             ^^^^^^^^^^^^^ no `EarlyStopGate` in `speculative::types`
 41 |     FlashPrefillConfig, PrefillMode, RejectionReason, RoutingOverlapSnapshot, SdeConfig,
    |                                         ^^^^^^^^^^^^^^^^^^^^^^^^                    ^ no `RoutingOverlapSnapshot`
 42 |     ScoreReduction, SpecCostSnapshot, StabilitySnapshot, TreeNode,
    |                   ^^^^^^^^^^^^^^^^ no `SpecCostSnapshot`
    |                                      ^^^^^^^^^^^^^^^^ no `StabilitySnapshot`

error[E0063]: missing field `routing_overlap` in initializer of `DraftResult`
   --> src/speculative/dflash.rs:425:5
    |
425 |     DraftResult {
    |     ^^^^^^^^^^^ missing `routing_overlap`

error[E0063]: missing field `routing_overlap` in initializer of `DraftResult`
   --> src/speculative/dflash.rs:468:5
    |
468 |     DraftResult {
    |     ^^^^^^^^^^^ missing `routing_overlap`
```

## Root cause (the drift)

`katgpt-core` gates the speculative substrate types behind features:

| Type / Field | Gate | File |
|---|---|---|
| `EarlyStopGate<P>` | `#[cfg(feature = "elf_sde")]` (per `katgpt-core/src/speculative/mod.rs` L35) | `katgpt-core/src/speculative/types.rs` L42 |
| `DraftResult::routing_overlap` | `#[cfg(feature = "domain_latent")]` | `katgpt-core/src/speculative/types.rs` L214 |
| `DraftResult::cost_snapshot: Option<SpecCostSnapshot>` | `#[cfg(feature = "spec_cost_model")]` | `katgpt-core/src/speculative/types.rs` L217 |
| `DraftResult::stability: Option<StabilitySnapshot>` | `#[cfg(feature = "stability_metrics")]` | `katgpt-core/src/speculative/types.rs` L220 |
| `RoutingOverlapSnapshot` | (transitively via `domain_latent`) | `katgpt-core/src/speculative/types.rs` L982 |
| `SpecCostSnapshot` | (transitively via `spec_cost_model`) | `katgpt-core/src/speculative/types.rs` L1002 |
| `StabilitySnapshot` | (transitively via `stability_metrics`) | `katgpt-core/src/speculative/types.rs` L114 |

The root crate (`katgpt-rs/src/speculative/types.rs` L39-43) re-exports all of
these **unconditionally**:

```rust
pub use katgpt_core::speculative::types::{
    BlockScores, BudgetAdaptation, DecodeStrategy, DraftEvent, DraftResult, EarlyStopGate,
    FlashPrefillConfig, PrefillMode, RejectionReason, RoutingOverlapSnapshot, SdeConfig,
    ScoreReduction, SpecCostSnapshot, StabilitySnapshot, TreeNode,
};
```

When `domain_latent` / `spec_cost_model` / `stability_metrics` / `elf_sde` are
OFF (as in `--no-default-features`), the names `EarlyStopGate`,
`RoutingOverlapSnapshot`, `SpecCostSnapshot`, `StabilitySnapshot` do not exist
in `katgpt_core::speculative::types` → the `pub use` fails. Separately,
`src/speculative/dflash.rs` constructs `DraftResult { ... }` literals at L425
and L468 without a `routing_overlap` field, which fails when `domain_latent`
is off (field absent).

## Fix sketch (TBD by implementer)

The root re-exports must mirror katgpt-core's gates. Two consistent approaches:

**Option A — gate each re-export:**
```rust
pub use katgpt_core::speculative::types::{
    BlockScores, BudgetAdaptation, DecodeStrategy, DraftEvent, DraftResult,
    FlashPrefillConfig, PrefillMode, RejectionReason, SdeConfig,
    ScoreReduction, TreeNode,
};
#[cfg(feature = "elf_sde")]
pub use katgpt_core::speculative::types::EarlyStopGate;
#[cfg(feature = "domain_latent")]
pub use katgpt_core::speculative::types::RoutingOverlapSnapshot;
#[cfg(feature = "spec_cost_model")]
pub use katgpt_core::speculative::types::SpecCostSnapshot;
#[cfg(feature = "stability_metrics")]
pub use katgpt_core::speculative::types::StabilitySnapshot;
```

**Option B — gate the whole `speculative::types` re-export module behind the
union of features** (coarser, simpler, but loses the always-on surface).

Plus the two `DraftResult` initializers in `src/speculative/dflash.rs` (L425,
L468) need either:
- A `..Default::default()` tail (requires `DraftResult: Default`), OR
- A `#[cfg(feature = "domain_latent")] routing_overlap: None,` line mirroring
  katgpt-core's gate, OR
- Restructuring so the initializers don't mention the gated fields directly.

## Acceptance

- [ ] `cargo check --no-default-features --lib` clean on root
- [ ] `cargo check --workspace` (default) still clean
- [ ] `cargo check --workspace --all-features` still clean
- [ ] `cargo check --features domain_latent --no-default-features --lib` clean (the gate fires)
- [ ] `cargo check --features spec_cost_model --no-default-features --lib` clean
- [ ] `cargo check --features stability_metrics --no-default-features --lib` clean
- [ ] `cargo check --features elf_sde --no-default-features --lib` clean (EarlyStopGate re-export fires)
- [ ] Unblocks Issue 355 Phase 2c

## Priority

P1 — blocks the `--no-default-features` acceptance gate (Issue 007 L258,
Issue 355 Phase 2c) and is the canonical "smallest reproducible feature-gate
drift" test case. Not a runtime bug (default + all-features both compile);
purely a feature-combination hygiene issue.

## Notes

- Verified pre-existing: `git stash` of Issue 355 Phase 2a changes on baseline
  `develop` reproduces the same 3 errors.
- The `katgpt-speculative` leaf crate (separate from `katgpt-core`'s
  speculative module) has its own `dd_tree.rs` that is unrelated to this drift.
- Related: `katgpt-rs/src/speculative/types.rs` L45-50 comment notes that
  "katgpt_core::speculative::types also exports feature-gated substrate types
  ... We don't re-export those here" — but `EarlyStopGate` / `SpecCostSnapshot`
  / `StabilitySnapshot` / `RoutingOverlapSnapshot` are conspicuously absent
  from that exclusion list. The intent was apparently to re-export them
  unconditionally; the implementation forgot the gates.
