//! Plan 298 — integration compile check for the `smear_classifier` feature.
//!
//! This file exists to ensure the feature-gated module compiles in the
//! `tests/` target context (separate crate compilation). It is intentionally
//! empty — Phase 2 (probe integration) and Phase 3 (GOAT gate G2 synthetic
//! harness) will add real integration tests here.
//!
//! Gated on `smear_classifier` so this file is a no-op when the feature is off.

#![cfg(feature = "smear_classifier")]

#[test]
fn compiles() {
    // If this compiles, the `smear_classifier` feature exposes its public
    // surface (SmearClass / SmearReport / SmearClassifier / CosineSmearClassifier)
    // from outside the crate. Phase 2 will exercise the real integration.
}
