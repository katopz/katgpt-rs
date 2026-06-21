//! Integration-test smoke check for the `personality_composition` public API
//! (Plan 297 Phase 1-3).
//!
//! Exercises the public surface that external consumers (riir-ai Plan 327)
//! will use:
//!
//! - `PersonalityWeightedComposition::<N, D>::new`
//! - `compose_into(&[&dyn LayerDirectionSource; N], &mut [f32], &mut [f32])`
//! - `drift(&[&dyn LayerDirectionSource; N], f32)`
//! - `PersonalitySnapshot::<N>::from_composition` + `verify_blake3`
//!
//! The full unit-test suite lives in
//! `katgpt-rs/crates/katgpt-core/src/personality_composition/tests.rs` —
//! this file is just a compile check against the public API (proves no
//! crate-internal symbols leaked into the surface).

#![cfg(test)]

#[cfg(not(feature = "personality_composition"))]
#[test]
fn compiles() {
    // Feature is off — module isn't even compiled. This stub exists so the
    // integration-test target stays valid (otherwise `cargo test` would
    // warn about an empty tests/ file).
}

#[cfg(feature = "personality_composition")]
#[test]
fn public_api_smoke() {
    use katgpt_core::personality_composition::{
        ArchetypeLabel, LayerDirectionSource, PersonalityConfig, PersonalitySnapshot,
        PersonalityWeightedComposition,
    };

    struct Echo<const D: usize>([f32; D]);
    impl<const D: usize> LayerDirectionSource for Echo<D> {
        fn direction<'a>(&self, scratch: &'a mut [f32]) -> &'a [f32] {
            let n = D.min(scratch.len());
            scratch[..n].copy_from_slice(&self.0[..n]);
            &scratch[..n]
        }
        fn recent_direction(&self) -> &[f32] {
            &self.0
        }
    }

    type K = PersonalityWeightedComposition<2, 4>;
    let kernel = K::new(PersonalityConfig::default(), [0.0, 0.0]);
    let l1 = Echo([1.0, 0.0, 0.0, 0.0]);
    let l2 = Echo([0.0, 1.0, 0.0, 0.0]);
    let layers: [&dyn LayerDirectionSource; 2] = [&l1, &l2];
    let mut scratch = [0.0f32; 4];
    let mut out = [0.0f32; 4];
    kernel.compose_into(&layers, &mut scratch, &mut out);
    // sigmoid(0/1) = 0.5 for both layers, belief = 1.0 default.
    assert!((out[0] - 0.5).abs() < 1e-6);
    assert!((out[1] - 0.5).abs() < 1e-6);

    let snap =
        PersonalitySnapshot::<2>::from_composition(&kernel, ArchetypeLabel::from_str("smoke"), 1);
    assert!(snap.verify_blake3());
}
