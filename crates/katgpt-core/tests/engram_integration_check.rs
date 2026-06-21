// Plan 299 — Engram integration compile check.
//
// This is an intentionally-empty integration test. Its sole purpose is to
// verify that the `engram` feature compiles when the crate is built with
// `--features engram` from the workspace root. The orchestrator wires the
// feature in `crates/katgpt-core/Cargo.toml` and the module export in
// `crates/katgpt-core/src/lib.rs`.
//
// Once the feature is wired, the full GOAT gate suite (Plan T7.3–T7.10)
// will live in `tests/bench_299_engram_goat.rs` (separate file, added by
// the orchestrator or a future task).
//
// Gated on `engram` so this file is a no-op when the feature is off.

#![cfg(feature = "engram")]

#[test]
fn engram_module_compiles() {
    // Re-exported from `katgpt_core::engram::*`. If the module doesn't
    // compile, this test (and the entire integration binary) won't build.
    // The body is empty because we have nothing to assert at runtime —
    // compilation success is the assertion.
    //
    // This function body references a public symbol so that
    // `--features engram` is observable even without further test code.
    let _ = katgpt_core::engram::K_MAX;
}
