# katgpt-deprecated

Exiled primitives — dead stubs, off-topic research toys, GOAT-failed
mechanisms, and explicitly demoted primitives awaiting deletion. **Never
default-on.** Kept for regression comparison and GOAT-gate audit trail.

## Overview

This crate exists to make deletion safe and auditable, not to live forever.
Each item here carries a `TODO(deprecated): delete after ...` comment. The
membership table with citations lives in
``katgpt-rs/.docs/001_loser_sweep_audit.md``.

### 3-category rule (recap)

Only **category 3** (dead/failed) items live here. Categories 1 (pending)
and 2 (benchmark-loser kept for A/B) stay in their domain crates.

## Modules

| Module | Feature | Origin | Reason |
|---|---|---|---|
| `alien_sampler` | `alien_sampler` | `src/alien_sampler/` | Coherence × Availability Frontier Ranking (Plan 311). GOAT 2/4 PASS (initially 1/4, G3 closed via Rayon; G1+G2 are the demotion drivers), explicitly DEMOTED. |
| `feedback` | `feedback` | `src/feedback.rs` | Dead TTT feedback stub — `log::debug!` only, never HTTP POSTs (Plan 042). |
| `unit_distance` | `unit_distance` | `src/unit_distance/` | Number-theory Erdos unit-distance research toy (Plan 090). No inference role. |

## Feature flags

`default = []` — **always empty**. Nothing here is ever default-on.

| Feature | Description |
|---|---|
| `alien_sampler` | Coherence × Availability Frontier Ranking (Plan 311, DEMOTED). |
| `feedback` | Dead TTT feedback stub (Plan 042). |
| `unit_distance` | Erdos unit-distance research toy (Plan 090). |
| `sr2am_configurator` | Forwarding shim so `--all-features` doesn't break the `feedback` test struct literals that match `katgpt-types`' `sr2am_configurator`-gated fields. Mirrors `katgpt-speculative/Cargo.toml`. |

## Dependencies

- `katgpt-core` — `feedback.rs` uses `InferenceResult` (re-exported from
  `katgpt-core::types`).
- `postcard` — `feedback.rs` serializes `InferenceResult`.
- `log` — `feedback.rs` logs via `log::debug!`.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
