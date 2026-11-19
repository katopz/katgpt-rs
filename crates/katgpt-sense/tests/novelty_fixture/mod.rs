//! Plan 639 T1.3 — the synthetic hidden-world fixture (the GOAT gate's only
//! judge; Phase 2 sweeps re-use it unchanged).
//!
//! Per SenseKind `m`, a hidden activation `h_t[m] ∈ [0, 1]` and an observed
//! activation `x_t[m] = max(h_t[m] + noise, 0)`. Noise is uniform with the
//! stated standard deviation, from a seeded xorshift64* stream per (case,
//! seed, kind) — bit-deterministic on every platform.
//!
//! Cases:
//! - (a) `Steady`        — constant hidden state + small noise;
//! - (b) `Steps`         — sustained steps at KNOWN ticks (`STEP_TICKS`);
//! - (c) `Spikes`        — constant hidden state + 1–2-tick noise bursts at
//!   KNOWN ticks (`SPIKE_TICKS`) — the hidden state never moves;
//! - (d) `Mixed`         — (b) + (c) + noise;
//! - (e) `HighNoise`     — sustained high noise over a LONG run (the
//!   channel-freeze case).

#![allow(dead_code)]

/// Ticks at which case (b)/(d) hidden states step.
pub const STEP_TICKS: [usize; 3] = [100, 200, 300];
/// Ticks at which case (c)/(d) bursts START (each lasts `SPIKE_LEN` ticks).
pub const SPIKE_TICKS: [usize; 4] = [60, 150, 250, 350];
/// Burst duration in ticks.
pub const SPIKE_LEN: usize = 2;
/// Burst amplitude (activation units, added on top of the hidden state).
pub const SPIKE_AMP: f32 = 0.8;
/// Ticks for cases (a)–(d).
pub const SHORT_TICKS: usize = 400;
/// Ticks for case (e).
pub const LONG_TICKS: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Case {
    Steady,
    Steps,
    Spikes,
    Mixed,
    HighNoise,
}

impl Case {
    pub const ALL: [Case; 5] = [
        Case::Steady,
        Case::Steps,
        Case::Spikes,
        Case::Mixed,
        Case::HighNoise,
    ];

    pub fn ticks(self) -> usize {
        match self {
            Case::HighNoise => LONG_TICKS,
            _ => SHORT_TICKS,
        }
    }

    fn noise_sd(self) -> f32 {
        match self {
            Case::Steady | Case::Steps | Case::Mixed => 0.05,
            Case::Spikes => 0.02,
            Case::HighNoise => 0.4,
        }
    }

    fn id(self) -> u64 {
        self as u64 + 1
    }
}

/// One fixture trace: hidden and observed activations per tick.
pub struct Trace {
    pub hidden: Vec<[f32; 6]>,
    pub observed: Vec<[f32; 6]>,
}

struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        // splitmix64 seed scramble so nearby seeds decorrelate.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    /// Uniform in [-1, 1).
    fn next_sym(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40; // 24 bits
        (r as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// Per-kind low/high hidden levels (distinct per kind so channels differ).
fn levels(m: usize) -> (f32, f32) {
    let lo = 0.10 + 0.02 * m as f32;
    (lo, lo + 0.45)
}

fn in_spike(t: usize) -> bool {
    SPIKE_TICKS.iter().any(|&s| t >= s && t < s + SPIKE_LEN)
}

/// Build one deterministic trace.
pub fn trace(case: Case, seed: u64) -> Trace {
    let n = case.ticks();
    let sd = case.noise_sd();
    // Uniform on [-a, a] has sd a/√3.
    let amp = sd * 3.0f32.sqrt();
    let mut rngs: Vec<XorShift> = (0..6)
        .map(|m| XorShift::new(case.id() * 1_000_003 + seed * 101 + m as u64))
        .collect();
    let mut hidden = Vec::with_capacity(n);
    let mut observed = Vec::with_capacity(n);
    for t in 0..n {
        let mut h = [0.0f32; 6];
        let mut x = [0.0f32; 6];
        for m in 0..6 {
            let (lo, hi) = levels(m);
            let steps_done = STEP_TICKS.iter().filter(|&&s| t >= s).count();
            h[m] = match case {
                Case::Steady | Case::Spikes | Case::HighNoise => lo,
                Case::Steps | Case::Mixed => {
                    if steps_done % 2 == 1 {
                        hi
                    } else {
                        lo
                    }
                }
            };
            let burst = match case {
                Case::Spikes | Case::Mixed if in_spike(t) => SPIKE_AMP,
                _ => 0.0,
            };
            x[m] = (h[m] + burst + amp * rngs[m].next_sym()).max(0.0);
        }
        hidden.push(h);
        observed.push(x);
    }
    Trace { hidden, observed }
}

/// The belief target a perfect tracker would hold: the kernel's own drive
/// map applied to the HIDDEN activation (`2σ(η·h) − 1`).
pub fn target_drive(h: f32, eta: f32) -> f32 {
    2.0 / (1.0 + (-eta * h).exp()) - 1.0
}
