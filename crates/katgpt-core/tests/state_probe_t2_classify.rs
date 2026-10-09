//! Plan 621 Phase 2 fixtures — T2.2 (planted-p G1) + T2.3 (the N-sweep
//! monotone-flip disclosure law). `#![cfg]` + the `required-features` row
//! (no green zero, the repo-birth gate discipline).
//!
//! The ensemble draws are seeded splitmix64-class LCG (no global RNG) with
//! exact binomial semantics: `k ~ Binomial(n, p)` by thresholding uniforms.
//! Research 609's 28/76 calibration split is CONTEXT, never a fixture
//! assertion (T2.3's own note).

#![cfg(feature = "state_probe")]

use katgpt_core::state_probe::{
    AdaptiveClassifier, AdaptiveDecision, ProbeClass, ProbeInput, classify, probe,
};

const Z: f64 = 1.959_963_984_540_054; // the two-sided 95% Wilson critical value
const EPSILON: f64 = 0.1;

/// Deterministic uniform [0,1) — seeded LCG, no global state.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 40) as f64 / (1u64 << 24) as f64
    }
    /// One binomial draw: `k ~ Binomial(n, p)` by thresholding.
    fn binomial(&mut self, n: u32, p: f64) -> u32 {
        let mut k = 0u32;
        for _ in 0..n {
            if self.unit() < p {
                k += 1;
            }
        }
        k
    }
}

/// The probe read at ensemble size `n` for a planted pass rate `p`.
fn probe_at(rng: &mut Lcg, n: u32, p: f64) -> katgpt_core::state_probe::ProbeEstimate {
    let k = rng.binomial(n, p);
    // Histogram: answers tracked as two classes (pass split across both) —
    // the entropy axis is not under test here; a 2-bin summary is enough.
    let (a, b) = (k / 2, k - k / 2);
    let hist = [a as u16, b as u16, 0u16, (n - k) as u16];
    probe(
        &ProbeInput {
            pass_count: k,
            n,
            histogram: &hist,
        },
        Z,
    )
}

// ── T2.2: planted p ∈ {0, 0.05, 0.2, 0.5} ───────────────────────────────

#[test]
fn planted_p0_declared_knowledge_like_within_n64() {
    let mut rng = Lcg(0x0621_0000);
    let mut log = [katgpt_core::state_probe::ClassFlip {
        at_n: 0,
        from: ProbeClass::Undetermined,
        to: ProbeClass::Undetermined,
        interval_width: 0.0,
    }; 16];
    let mut cl = AdaptiveClassifier::new(EPSILON, 64, &mut log);
    let mut verdict = AdaptiveDecision::Continue;
    for n in [8u32, 16, 32, 64] {
        verdict = cl.observe(&probe_at(&mut rng, n, 0.0), n);
    }
    assert_eq!(
        verdict,
        AdaptiveDecision::Settled(ProbeClass::KnowledgeLike),
        "p=0 must settle knowledge-like within N ≤ 64"
    );
    assert!(
        cl.flips()
            .iter()
            .any(|f| f.to == ProbeClass::KnowledgeLike),
        "the flip to knowledge-like must be DISCLOSED"
    );
    assert!(
        cl.flips().iter().all(|f| f.interval_width >= 0.0),
        "every flip carries a real interval width"
    );
}

#[test]
fn planted_p005_knowledge_like_by_n64_but_never_early() {
    // p=0.05 sits under ε=0.1: the direction is knowledge-like, but the
    // settle N is DRAW-DEPENDENT (E[k]=3.2 at N=64; a k=5 draw keeps
    // wilson_hi above the bar). Pin the seed-robust properties: never
    // Productive, undetermined at N=8 (the interval is wide), and — when
    // the seed settles — it settles KnowledgeLike, never the wrong class.
    let mut rng = Lcg(0x0621_0001);
    let mut log = [katgpt_core::state_probe::ClassFlip {
        at_n: 0,
        from: ProbeClass::Undetermined,
        to: ProbeClass::Undetermined,
        interval_width: 0.0,
    }; 16];
    let mut cl = AdaptiveClassifier::new(EPSILON, 64, &mut log);
    let mut first = None;
    let mut verdict = AdaptiveDecision::Continue;
    for n in [8u32, 16, 32, 64] {
        let d = cl.observe(&probe_at(&mut rng, n, 0.05), n);
        if first.is_none() {
            first = Some(d);
        }
        verdict = d;
    }
    assert_ne!(
        verdict,
        AdaptiveDecision::Settled(ProbeClass::Productive),
        "p=0.05 < ε must never settle productive"
    );
    assert_eq!(
        first,
        Some(AdaptiveDecision::Continue),
        "at N=8 the interval is wide — undetermined, keep sampling"
    );
    for f in cl.flips() {
        assert_ne!(f.to, ProbeClass::Productive, "no flip may land productive at p=0.05");
    }
}

#[test]
fn planted_p02_never_knowledge_like() {
    let mut rng = Lcg(0x0621_0002);
    for n in [8u32, 16, 32, 64, 128] {
        let e = probe_at(&mut rng, n, 0.2);
        let c = classify(&e, EPSILON);
        assert_ne!(
            c,
            ProbeClass::KnowledgeLike,
            "p=0.2 ≥ ε must NEVER classify knowledge-like (n={n}, wilson_hi={})",
            e.wilson_hi
        );
    }
    // …and by N=128 it is decisively productive.
    let mut rng = Lcg(0x0621_0002);
    let mut saw_productive = false;
    for n in [64u32, 128] {
        let e = probe_at(&mut rng, n, 0.2);
        if classify(&e, EPSILON) == ProbeClass::Productive {
            saw_productive = true;
        }
    }
    assert!(saw_productive, "p=0.2 should reach Productive at N ∈ {{64,128}}");
}

#[test]
fn planted_p05_either_but_disclosed() {
    // p=0.5: never knowledge-like; the class it settles into (Productive)
    // is reached THROUGH a disclosed flip, never silently.
    let mut rng = Lcg(0x0621_0003);
    let mut log = [katgpt_core::state_probe::ClassFlip {
        at_n: 0,
        from: ProbeClass::Undetermined,
        to: ProbeClass::Undetermined,
        interval_width: 0.0,
    }; 16];
    let mut cl = AdaptiveClassifier::new(EPSILON, 128, &mut log);
    for n in [8u32, 16, 32, 64, 128] {
        let _ = cl.observe(&probe_at(&mut rng, n, 0.5), n);
    }
    for f in cl.flips() {
        assert_ne!(f.to, ProbeClass::KnowledgeLike, "p=0.5 never flips knowledge-like");
        assert!(f.interval_width > 0.0, "flip at n={} carries its width", f.at_n);
    }
    assert_eq!(cl.settled_class(), Some(ProbeClass::Productive));
}

// ── T2.3: the reclassification N-sweep monotonicity law ─────────────────

#[test]
fn n_sweep_flips_are_monotone_and_disclosed() {
    // Every planted state's class sequence over N ∈ {8,16,32,64} must be
    // MONOTONE: at most one flip, and never a reversal (a state that reads
    // knowledge-like at some N cannot read productive at a larger N on this
    // fixture's draws, and vice versa).
    for (seed, p) in [
        (0x0621_0010u64, 0.0),
        (0x0621_0011, 0.05),
        (0x0621_0012, 0.2),
        (0x0621_0013, 0.5),
    ] {
        let mut rng = Lcg(seed);
        let mut prev: Option<ProbeClass> = None;
        let mut flips = 0usize;
        let mut log = [katgpt_core::state_probe::ClassFlip {
            at_n: 0,
            from: ProbeClass::Undetermined,
            to: ProbeClass::Undetermined,
            interval_width: 0.0,
        }; 16];
        let mut cl = AdaptiveClassifier::new(EPSILON, 64, &mut log);
        for n in [8u32, 16, 32, 64] {
            let d = cl.observe(&probe_at(&mut rng, n, p), n);
            let c = match d {
                AdaptiveDecision::Settled(c) => Some(c),
                AdaptiveDecision::Continue | AdaptiveDecision::CappedUndetermined => {
                    Some(ProbeClass::Undetermined)
                }
            };
            if let (Some(prev_c), Some(cur)) = (prev, c)
                && prev_c != cur
            {
                flips += 1;
                // MONOTONE: Undetermined is the ONLY origin of a flip.
                assert_eq!(
                    prev_c,
                    ProbeClass::Undetermined,
                    "p={p} n={n}: a DECISIVE class reverted — non-monotone"
                );
            }
            prev = c;
        }
        assert!(
            flips <= 1,
            "p={p}: {flips} flips across the sweep — flips must be one-directional"
        );
        // Disclosure law: the tracker's log matches the observed flips,
        // every entry carrying its width.
        assert_eq!(
            cl.flips().len(),
            flips,
            "p={p}: disclosed flip count must match observed"
        );
    }
}

#[test]
fn capped_undetermined_at_n_max_and_full_log_drops_loud() {
    // A straddling interval that cannot close inside n_max: the cap fires
    // and the caller's conservative reading applies. (Constructed directly —
    // the cap is interval geometry at n ≥ n_max, not a draw property.)
    let straddling = katgpt_core::state_probe::ProbeEstimate {
        v_hat: 0.12,
        h_mm: 0.0,
        wilson_lo: 0.02,
        wilson_hi: 0.40,
    };
    let mut log = [katgpt_core::state_probe::ClassFlip {
        at_n: 0,
        from: ProbeClass::Undetermined,
        to: ProbeClass::Undetermined,
        interval_width: 0.0,
    }; 16];
    let mut cl = AdaptiveClassifier::new(EPSILON, 8, &mut log);
    assert_eq!(cl.observe(&straddling, 4), AdaptiveDecision::Continue, "n < n_max: keep sampling");
    assert_eq!(
        cl.observe(&straddling, 8),
        AdaptiveDecision::CappedUndetermined,
        "n ≥ n_max: the cap, conservative reading"
    );
    // A FULL log drops further flips on the floor (documented in `new`):
    // with a 1-entry log and two flips pending, the second is dropped
    // silently-by-capacity — the caller's sizing duty, asserted here so
    // the behavior is pinned, not discovered. (Constructed directly: two
    // pending flips need a decisive→decisive sequence, which real draws
    // only produce through an Undetermined in between.)
    let productive = katgpt_core::state_probe::ProbeEstimate {
        v_hat: 0.45,
        h_mm: 0.0,
        wilson_lo: 0.30,
        wilson_hi: 0.60,
    };
    let kl = katgpt_core::state_probe::ProbeEstimate {
        v_hat: 0.0,
        h_mm: 0.0,
        wilson_lo: 0.0,
        wilson_hi: 0.05,
    };
    let und = katgpt_core::state_probe::ProbeEstimate {
        v_hat: 0.12,
        h_mm: 0.0,
        wilson_lo: 0.02,
        wilson_hi: 0.40,
    };
    let mut tiny = [katgpt_core::state_probe::ClassFlip {
        at_n: 0,
        from: ProbeClass::Undetermined,
        to: ProbeClass::Undetermined,
        interval_width: 0.0,
    }; 1];
    let mut cl = AdaptiveClassifier::new(EPSILON, 4096, &mut tiny);
    let _ = cl.observe(&und, 8);  // Undetermined — seeds `last`
    let _ = cl.observe(&kl, 16);  // flip 1: Und→KL — fills the 1-entry log
    let _ = cl.observe(&productive, 32); // flip 2: KL→Prod — DROPPED
    assert_eq!(cl.flips().len(), 1, "a full log drops further flips (documented)");
    assert_eq!(cl.flips()[0].to, ProbeClass::KnowledgeLike, "the RETAINED flip is the first");
    assert_eq!(cl.settled_class(), Some(ProbeClass::Productive), "the class still advances");
}
