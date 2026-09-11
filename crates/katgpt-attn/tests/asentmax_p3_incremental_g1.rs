//! Issue 747 P3 T3.2 — incremental decode entmax G1 gate: **bit-exact
//! parity with the full re-sort**.
//!
//! Lemma 3.1 (arXiv:2506.16640 App. D.1): below-threshold score additions
//! leave existing α-entmax probabilities *exactly* unchanged. Our
//! `IncrementalEntmax1p5` extends that to the whole stream state: after
//! EVERY push, `(probs, τ, |support|)` must equal a fresh
//! `entmax_1p5(&history[..])` bit-for-bit (`f32::to_bits`).
//!
//! Adversarial streams beyond plain random:
//! - **threshold-brushing**: scores exactly at τ, one ULP above, one ULP
//!   below — the `≤` vs `<` boundary of the fast path;
//! - **spike-after-plateau**: the support-SHRINK case (τ rises above the
//!   old plateau; former support members drop to exactly 0.0);
//! - **all-equal**: every push is a support-entry event (worst case for
//!   amortization) — parity must survive the rescan-shrink churn;
//! - **descending / ascending** monotone streams.

#![cfg(feature = "asentmax_schedule")]

use katgpt_attn::dash_attn::entmax::entmax_1p5;
use katgpt_attn::dash_attn::entmax_incremental::IncrementalEntmax1p5;

fn assert_parity(history: &[f32], inc: &IncrementalEntmax1p5, label: &str) {
    let (probs, tau) = entmax_1p5(history);
    assert_eq!(
        inc.len(),
        history.len(),
        "{label}: length diverges at n={}",
        history.len()
    );
    assert_eq!(
        inc.tau().to_bits(),
        tau.to_bits(),
        "{label}: tau bits diverge at n={} (inc={}, full={})",
        history.len(),
        inc.tau(),
        tau
    );
    for (i, (&a, &b)) in inc.probs().iter().zip(probs.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: probs[{i}] bits diverge at n={} (inc={a:e}, full={b:e})",
            history.len()
        );
    }
}

fn unit(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / ((1u64 << 24) as f32)
}

#[test]
fn p3_g1_random_stream_per_step_parity() {
    let mut state = 0x05EE_D747_u64;
    let mut inc = IncrementalEntmax1p5::new(2_048);
    let mut history: Vec<f32> = Vec::with_capacity(2_048);
    while history.len() < 2_048 {
        // Mixture: bulk N(0,1)-ish + occasional spikes — realistic decode
        // stream shape (spikes form the stable support, bulk brushes τ).
        let s = if unit(&mut state) < 0.02 {
            6.0 + unit(&mut state) * 2.0
        } else {
            unit(&mut state) * 8.0 - 4.0
        };
        history.push(s);
        inc.push(s);
        // Per-step parity at checkpoints (every 128 pushes — O(n²/128)
        // total, each check itself a fresh full re-sort).
        if history.len().is_multiple_of(128) {
            assert_parity(&history, &inc, "random");
        }
    }
    assert_parity(&history, &inc, "random");
}

#[test]
fn p3_g1_threshold_brushing_streams() {
    // Stream A: park a support, then push EXACTLY τ / τ±1ULP repeatedly.
    let mut inc = IncrementalEntmax1p5::new(512);
    let mut history = vec![7.0_f32, 6.5, 6.2];
    for &s in &history {
        inc.push(s);
    }
    assert_parity(&history, &inc, "brush:seed");

    for round in 0..160 {
        let tau = inc.tau();
        // Exactly at τ: below-or-equal fast path, no event.
        assert!(!inc.push(tau), "round {round}: at-τ must be a no-event");
        history.push(tau);
        // One ULP below: still fast path.
        let below = f32::from_bits(tau.to_bits() - 1);
        assert!(!inc.push(below), "round {round}: 1-ULP-below must be a no-event");
        history.push(below);
        // One ULP above: support-entry event (strictly greater than τ).
        let above = f32::from_bits(tau.to_bits() + 1);
        assert!(inc.push(above), "round {round}: 1-ULP-above must event");
        history.push(above);
        if history.len().is_multiple_of(32) {
            assert_parity(&history, &inc, "brush");
        }
    }
    assert_parity(&history, &inc, "brush");
}

#[test]
fn p3_g1_spike_after_plateau_shrinks_support_with_parity() {
    let mut inc = IncrementalEntmax1p5::new(64);
    let mut history: Vec<f32> = vec![5.0; 32];
    for &s in &history {
        inc.push(s);
    }
    assert_eq!(inc.support_size(), 32);
    assert_parity(&history, &inc, "plateau");

    // Rising staircase of spikes: each raises τ and can evict plateau
    // members — the rescan-shrink churn.
    for spike in [5.5_f32, 6.0, 6.5, 7.0, 7.5, 8.0] {
        history.push(spike);
        inc.push(spike);
        assert_parity(&history, &inc, "staircase");
    }
    assert!(
        inc.support_size() < 32,
        "spikes must have shrunk the plateau support (now {})",
        inc.support_size()
    );
    // Fallen plateau members are EXACTLY zero (not ε).
    let zeros = inc.probs().iter().filter(|&&p| p == 0.0).count();
    assert_eq!(zeros + inc.support_size(), inc.len());
    // Simplex holds.
    let sum: f32 = inc.probs().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
}

#[test]
fn p3_g1_all_equal_stream() {
    // Worst case: every equal-score push is a support entry; the support
    // never shrinks but every push rescans. Parity must hold throughout.
    let n = 1_024_usize;
    let mut inc = IncrementalEntmax1p5::new(n);
    let mut history = Vec::with_capacity(n);
    for i in 0..n {
        history.push(3.0);
        inc.push(3.0);
        if (i + 1) % 256 == 0 {
            assert_parity(&history, &inc, "all_equal");
        }
    }
    assert_eq!(inc.support_size(), n);
    let sum: f32 = inc.probs().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!((inc.probs()[0] - inc.probs()[n - 1]).abs() < 1e-9);
}

#[test]
fn p3_g1_monotone_streams() {
    // Descending: scores fall 0.01/step — a few early entries form the
    // support (~14: the slow slope keeps them above τ), then every push
    // is below τ — the pure fast-path stream. Parity + no events after
    // stabilization are the contract.
    let mut inc = IncrementalEntmax1p5::new(512);
    let mut history = Vec::with_capacity(512);
    let mut events_late = 0;
    for i in 0..512_usize {
        let s = 10.0 - i as f32 * 0.01;
        history.push(s);
        let event = inc.push(s);
        if i >= 64 && event {
            events_late += 1;
        }
    }
    assert_parity(&history, &inc, "descending");
    assert_eq!(events_late, 0, "descending stream must stabilize after ~14 entries");
    assert!((2..64).contains(&inc.support_size()));

    // Ascending: every push is a support entry, support shrinks as τ rises.
    let mut inc2 = IncrementalEntmax1p5::new(512);
    let mut history2 = Vec::with_capacity(512);
    for i in 0..512_usize {
        let s = -5.0 + i as f32 * 0.05;
        history2.push(s);
        inc2.push(s);
        if (i + 1) % 128 == 0 {
            assert_parity(&history2, &inc2, "ascending");
        }
    }
    assert_parity(&history2, &inc2, "ascending");
}
