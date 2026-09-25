# Issue 897 — katgpt-kv's KVarN `static_cal_tables` branches are gated on a feature katgpt-kv does not declare (compiled by nothing)

**Status:** OPEN — found 2026-09-25 while closing Issue 896 (HISTORY.md § Issue 896). Not fixed; it has no behavioural effect today, because the code is compiled nowhere.

## Finding (measured)

- `crates/katgpt-kv/src/kvarn/kv_cache.rs` (10 sites, 2 of them in tests) and `kvarn/eval.rs` (1 site) carry `#[cfg(feature = "static_cal_tables")]` / `#[cfg(not(...))]`. These gate `KVarNConfig::static_cal`, the struct field, and the static-calibration branch of `quantize_key_tile` / `quantize_val_tile`.
- `crates/katgpt-kv/Cargo.toml` declares no `static_cal_tables` feature: `cargo test -p katgpt-kv --features kvarn,static_cal_tables` fails with *"the package 'katgpt-kv' does not contain this feature"*. The root's `static_cal_tables` forwards to `katgpt-attn/static_cal_tables` only.
- The gated code names `crate::static_cal::StaticCalTable`. katgpt-kv has no `static_cal` module, so the branch would not compile if anything enabled it.
- The warning that would say so is silenced by `#![allow(unexpected_cfgs)]` in `crates/katgpt-kv/src/lib.rs:39`.
- So the Plan 227 Phase 1 "static cal replaces Sinkhorn in KVarN" path is dead code — presumably since the code moved into katgpt-kv (Issue 015 spin-out; not bisected). Every KVarN run uses the Sinkhorn / skip_varn arms.

## Tasks

- [ ] **T1 — decide:** either wire it (a `static_cal_tables = ["dep:katgpt-attn"?, …]` feature on katgpt-kv plus a real `StaticCalTable` path, subject to BOUNDARY.md), or delete the dead branches and the `KVarNConfig::static_cal` field.
- [ ] **T2 — remove or narrow `#![allow(unexpected_cfgs)]`** in katgpt-kv so the next undeclared-feature cfg is a warning.
