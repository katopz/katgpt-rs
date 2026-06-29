# Plan 343 — Extract `katgpt-pruners` crate

- [x] Phase 0: scope discovery (dependency audit, feature inventory, cycle analysis)
- [x] Phase 1: create `crates/katgpt-pruners/` scaffold
- [x] Phase 2: `git mv` everything except `bomber/` (240 files, 20 subdirs)
- [x] Phase 3: rewrite `src/pruners/mod.rs` as back-compat shim (`pub use katgpt_pruners::*`)
- [x] Phase 4: fix moved files' internal `crate::` paths
- [x] Phase 5: feature-forward root → katgpt-pruners (~115 features)
- [x] Phase 6: handle non-bomber cross-crate edges
  - [x] `ThinkingMode` (collapse_detector) → local enum copy
  - [x] `ComputeTier` (thicket_variance_probe) → local enum copy + tier_to_kp bridge in router_tvp
  - [x] `TesConfig` → moved to `katgpt_pruners::tes_loop` (uses `BanditStrategy` which lives here)
  - [x] `freq_bandit::FrequencyBandit` → bfcp_lfu_shard integration gated out (stays in main crate)
  - [x] `residency_audit` test (bandit.rs) → deleted; relocate as integration test in follow-up
  - [x] `bomber_state.rs` → moved back to main crate (tightly coupled to bomber)
  - [x] `SdpgBanditPruner::from_replay` + `load_teacher_q_from_replay` → moved to main crate's `src/pruners/bomber/sdpg_helpers.rs` (depends on bomber's `ReplaySample`)
  - [x] `cna::bomber_pairs` → deleted from katgpt-pruners (dead code; zero consumers anywhere)
- [x] Phase 7: verify build (default + `--all-features`)
- [x] Phase 8: verify tests (`cargo test --lib` on both crates — 201 + 2177 passing)
- [-] Phase 9: relocate `test_goat_175_fusion_residency_audit_passes` as integration test (deferred)
- [-] Phase 10: move `ThinkingMode` + `ComputeTier` to `katgpt-core::traits` to eliminate duplicate enum copies (deferred follow-up)

## Outcome

`src/pruners/` (240 files, 20 subdirs) moved to `crates/katgpt-pruners/src/`. The
only thing that stayed behind is the `bomber/` subdirectory plus
`bomber_state.rs` (re-added inside bomber/) and `sdpg_helpers.rs` (new file with
the SDPG-from-replay constructor that depends on bomber's `ReplaySample`).

The `src/pruners/mod.rs` is now a 30-line back-compat shim that does
`pub use katgpt_pruners::*;` — so all 561 existing `crate::pruners::*`
import sites in `src/`, `examples/`, `tests/`, `benches/` continue to resolve
unchanged.

## Why bomber stayed

Bomber is the cycle epicenter: it depends on `crate::transformer::*`,
`crate::inference_router::InferenceRouter`, `crate::trigger_gate::*`,
`crate::types::*`, all of which are local to the main crate (some are re-export
shims from katgpt-transformer; others like inference_router/trigger_gate are
truly local). Moving bomber out would force either moving all of those too
(cascade) or breaking the dependency direction. Bomber's only real consumers
are the main crate's bomber feature, so leaving it in main crate has no
downstream cost.

## Follow-ups (non-blocking)

1. **Move `ThinkingMode` and `ComputeTier` to `katgpt-core::traits`.** Currently
   both have a duplicate definition: one in main crate (canonical, in
   `src/speculative/thinking_controller.rs` and `src/trigger_gate.rs`) and one
   in katgpt-pruners (`crates/katgpt-pruners/src/{collapse_detector,thicket_variance_probe}.rs`).
   The duplicates are bit-compatible via `#[repr(u8)]` and a `tier_to_kp` bridge,
   but DRY says we should consolidate. Low priority — the bridge is 7 lines.

2. **Relocate the deleted residency_audit test.** `test_goat_175_fusion_residency_audit_passes`
   was removed from `katgpt-pruners/src/bandit.rs` because it depended on the
   main crate's `crate::speculative::residency_audit` (test-only module). Should
   be re-added as an integration test in `katgpt-rs/tests/` that constructs
   `BanditPruner` via `katgpt_pruners::bandit::*` and audits via
   `katgpt_rs::speculative::residency_audit::*`.

3. **Crate-level doc + README** for `crates/katgpt-pruners/`.

## Validation

```bash
# Default features
CARGO_TARGET_DIR=/tmp/katgpt_pruners_extract cargo check

# All features
CARGO_TARGET_DIR=/tmp/katgpt_pruners_extract cargo check --all-features

# Main crate tests (2177 passing)
CARGO_TARGET_DIR=/tmp/katgpt_pruners_extract cargo test --lib

# New crate tests (201 passing with bandit feature)
CARGO_TARGET_DIR=/tmp/katgpt_pruners_extract cargo test -p katgpt-pruners --lib --features bandit
```
