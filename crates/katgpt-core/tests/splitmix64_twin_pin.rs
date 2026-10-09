//! Issue 927 — the SplitMix64 twin-pin (the copy-gate convention's test half).
//!
//! The workspace carries two same-named `SplitMix64` stream types:
//! `katgpt_types::rng::SplitMix64` (the designated workspace home — reflex
//! Issue 071's substrate export, landed one day before the copy) and
//! `katgpt_assign::rng::SplitMix64` (a justified copy: katgpt-assign's
//! zero-dep standalone posture is a documented deliberate law, so the
//! "delegate" menu option was declined). This file is the one place both
//! crates are visible, so THIS test carries the pin the convention demands:
//! the stream cores are bit-identical from the same seed — any constant or
//! state-advance edit on either side alone fails here.
//!
//! Deliberately NOT pinned: `below(n)` — the twins' bounded draws differ by
//! design (types = 53-bit Lemire, assign = 64-bit Lemire — landed Plan 620
//! behavior; converging them would silently change every deterministic
//! sequence either side has pinned). Each `below` is pinned by its own
//! crate's reference tests.

#![cfg(feature = "assignment")]

use katgpt_core::assign::rng::SplitMix64 as AssignSplitMix64;
use katgpt_core::types::rng::SplitMix64 as TypesSplitMix64;

/// The twin streams are bit-identical from the same seed — the
/// divergence-failing pin of the copy-gate convention (Issue 927's menu
/// option 1). 10k draws per seed: any single-sided constant edit diverges
/// on the FIRST draw, so the loop length is cheap insurance, not the gate.
#[test]
fn splitmix64_twin_streams_bit_identical() {
    for seed in [0u64, 1, 42, 0xDEAD_BEEF_CAFE_F00D, u64::MAX / 2 + 1, u64::MAX] {
        let mut home = TypesSplitMix64::new(seed);
        let mut twin = AssignSplitMix64::new(seed);
        for i in 0..10_000u32 {
            assert_eq!(
                home.next_u64(),
                twin.next_u64(),
                "twin SplitMix64 streams diverged at draw {i} from seed {seed}"
            );
        }
    }
}
