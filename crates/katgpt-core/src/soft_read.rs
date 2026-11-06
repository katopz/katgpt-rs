//! Sigmoid-gated query-conditioned fixed-K soft read (Issue 985 T2/T3/T6,
//! the Research 656 Memorilla distillation's modelless floor form).
//!
//! The mechanism, no learned weights:
//!
//! - **Conditioned heads** — each slot k reads the store through its own
//!   deterministic view of the query:
//!   `q_k = normalize(z_q + β_k · ⟨z_q, d_k⟩ · d_k)`. Slot distinctness
//!   lives in the residual path (the paper's Eq. 3 note: a single-vector
//!   query gives all slots a shared update; β_k spans [0, 1] so slot 0 is
//!   the unconditioned plain head and slot K−1 the fully pulled view).
//! - **Gates** — `g_ik = σ(⟨q_k, z_i⟩/τ + b_k)`: sigmoid, never softmax
//!   (house law); `b_k` defaults to 0 and is the slot's selectivity dial.
//! - **Read (NORMALIZED — load-bearing)** — `M_k = Σ_i g_ik·z_i /
//!   (Σ_i g_ik + ε)` with ε = 1e-6 fixed. A sigmoid-gated SUM grows with
//!   store size; the gated weighted AVERAGE is bounded by the longest
//!   stored vector by construction — Memorilla's flat-cost-in-N property.
//!   ε ≪ any non-degenerate gate mass (gates live in (0,1), Σg up to N)
//!   yet keeps the denominator strictly positive; the shrinkage is ~1 part
//!   in 10⁶ at Σg = 1. When every gate underflows (b_k driven very
//!   negative, or τ tiny against anti-aligned rows), Σg ≪ ε and M_k
//!   collapses toward the zero vector — the CORRECT behaviour: "nothing
//!   relevant" must look like an empty memory (the zero-gate test pins
//!   exactly this).
//!
//! # Determinism law (Issue 940 — recorded at the code site per the
//! amendment)
//!
//! **The aggregate read is LOCAL-ONLY.** It is never committed, exported,
//! hashed, or synced. The fold below is sequential in store order as a
//! matter of construction, NOT as an Issue-940 determinism pin — no
//! reduction order or tie rule is owed across machines for a local-only
//! value. IF a future consumer ever syncs or commits M_k, the fold order
//! and tie rule MUST be pinned AT THAT POINT (sequential fold, one fixed
//! tie rule), never inherited by accident.
//!
//! # Posture
//!
//! Opt-in (`soft_read`). **MEASURED NEGATIVE (2026-11-06):** the
//! preregistered bar (≥ hard top-16 AND > the empty-memory floor) FAILED
//! at the amended GOAT cell — see `benches/bench_985_soft_read_goat.rs`
//! (report-only replication instrument) and Issue 985 for the numbers and
//! the mechanism finding (FPS anchors spread to the store's outskirts;
//! distinct slots without LEARNED placement do not condition reads to the
//! neighborhoods the raw query cannot rank). The grade is struck; the
//! feature stays opt-in for replication. The TRAINED lane (riir-train
//! Plan 488 R1) is NOT indicted — its bar already compares to hard top-16,
//! which the run confirms as the standing champion on the fixture class.

use crate::slot_bank::gate;
// Slot-posture vocabulary re-exported for consumers (the crate-internal
// gate fn is soft-read plumbing, not a public seam).
pub use crate::slot_bank::{SlotBank, SlotMode};

/// ε for the normalized read — fixed relative to the gate scale (gates in
/// (0,1); see the module doc for the semantics).
pub const READ_EPS: f32 = 1e-6;

/// The default gate temperature: unit-norm inner products spread by 1/τ;
/// 0.1 puts in-corpus gates near 1, orthogonal near 0.5, anti-aligned at
/// σ(−10) ≈ 5e-5.
pub const DEFAULT_TAU: f32 = 0.1;

/// Unit-normalize; a zero vector passes through (never NaN). Local twin of
/// `distance_abstain::unit` (that one is `pub(crate)` behind its feature;
/// this module is feature-independent by design — a duplicated 6-line
/// normalize beats a cross-feature edge).
fn unit<const D: usize>(v: [f32; D]) -> [f32; D] {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        v.map(|x| x * inv)
    } else {
        v
    }
}

/// Caller-owned scratch: one conditioned head per slot, reused across
/// reads — the zero-alloc steady state (T2). Build once, pass `&mut`.
pub struct SoftReadScratch<const D: usize, const K: usize> {
    /// The conditioned heads from the last [`SoftRead::read_into`] —
    /// read-only access for extraction consumers (Bench 985's per-slot
    /// top-gate row scan).
    pub heads: [[f32; D]; K],
}

impl<const D: usize, const K: usize> SoftReadScratch<D, K> {
    pub fn new() -> Self {
        Self {
            heads: [[0.0; D]; K],
        }
    }
}

impl<const D: usize, const K: usize> Default for SoftReadScratch<D, K> {
    fn default() -> Self {
        Self::new()
    }
}

/// The soft read over a persistent store. Build once per store epoch (the
/// slot bank is `BLAKE3(store)`-keyed — [`SlotBank::is_fresh`] is the cache
/// seam), read many.
pub struct SoftRead<const D: usize, const K: usize> {
    bank: SlotBank<D>,
    tau: f32,
    biases: [f32; K],
}

impl<const D: usize, const K: usize> SoftRead<D, K> {
    /// Preregistered default posture: farthest-point slots (on-manifold),
    /// τ = 0.1, b_k = 0.
    pub fn new(rows: &[[f32; D]]) -> Self {
        Self::with_mode(rows, SlotMode::FarthestPoint)
    }

    pub fn with_mode(rows: &[[f32; D]], mode: SlotMode) -> Self {
        Self {
            bank: SlotBank::new(rows, K, mode),
            tau: DEFAULT_TAU,
            biases: [0.0; K],
        }
    }

    /// Non-default τ (the zero-gate test uses a small τ to drive gates to
    /// genuine underflow; the GOAT posture stays at [`DEFAULT_TAU`]).
    pub fn with_tau(mut self, tau: f32) -> Self {
        assert!(
            tau.is_finite() && tau > 0.0,
            "τ must be finite and positive"
        );
        self.tau = tau;
        self
    }

    /// Per-slot gate biases (selectivity dials; default all-zero).
    pub fn with_biases(mut self, biases: [f32; K]) -> Self {
        self.biases = biases;
        self
    }

    pub fn bank(&self) -> &SlotBank<D> {
        &self.bank
    }

    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// The K slot reads for one query. Zero-alloc: every intermediate lives
    /// in `scratch` or on the stack; `rows` is borrowed. M_k is a gated
    /// weighted AVERAGE — bounded by the longest stored row by construction.
    ///
    /// Determinism: same (store, query, posture) → bit-identical reads on
    /// one machine. The fold order is construction-order sequential and
    /// LOCAL-ONLY — see the module doc's Issue-940 note before ever
    /// syncing this value.
    pub fn read_into(
        &self,
        q: &[f32; D],
        rows: &[[f32; D]],
        scratch: &mut SoftReadScratch<D, K>,
    ) -> [[f32; D]; K] {
        assert!(
            !rows.is_empty(),
            "soft read over an empty store — use the empty floor instead"
        );
        let q = unit(*q);
        // Conditioned heads (the residual path — slot distinctness).
        for k in 0..K {
            let d = self.bank.slot(k);
            let dot: f32 = q.iter().zip(d).map(|(a, b)| a * b).sum();
            let beta = self.bank.beta(k);
            scratch.heads[k] = {
                let mut h = q;
                for (x, dv) in h.iter_mut().zip(d) {
                    *x += beta * dot * dv;
                }
                unit(h)
            };
        }
        let mut reads = [[0.0f32; D]; K];
        for ((read_k, head_k), bias) in reads
            .iter_mut()
            .zip(scratch.heads.iter())
            .zip(self.biases.iter())
        {
            let head = head_k;
            let b = *bias;
            let mut num = [0.0f32; D];
            let mut den = 0.0f32;
            for r in rows {
                let dot: f32 = head.iter().zip(r).map(|(a, x)| a * x).sum();
                let g = gate(dot, self.tau, b);
                den += g;
                for (x, rv) in num.iter_mut().zip(r) {
                    *x += g * rv;
                }
            }
            // Normalized read: the weighted average, never the sum.
            let inv = 1.0 / (den + READ_EPS);
            for (x, rv) in read_k.iter_mut().zip(&num) {
                *x = rv * inv;
            }
        }
        reads
    }

    /// Best-slot answer cosine: `max_k cos(normalize(M_k), gold)` — the
    /// AGGREGATE-read semantic (the gated weighted average's direction).
    /// Structurally capped below an exact-row answer: a neighborhood
    /// average cannot reach cosine 1.0 even when the read is perfect —
    /// printed as the aggregate reference, never gated against hard's
    /// best-of-16 (see Bench 985's amendment 3).
    pub fn best_answer_cosine(reads: &[[f32; D]; K], gold: &[f32; D]) -> f32 {
        let gold = unit(*gold);
        let mut best = f32::NEG_INFINITY;
        for m in reads {
            let m = unit(*m);
            let c: f32 = m.iter().zip(&gold).map(|(a, b)| a * b).sum();
            if c > best {
                best = c;
            }
        }
        best
    }

    /// Best gated-row extraction: per slot, the TOP-GATE store row (argmax
    /// g_ik = argmax ⟨q_k, z_i⟩ — sigmoid monotonicity makes this
    /// gate-scale-free), scored by cosine to gold; best over slots. This
    /// is the modelless "consumer picks the useful token" semantic at the
    /// SAME candidate budget as hard top-K (one row per slot): the only
    /// difference from the discrete baseline is the conditioning. Must run
    /// AFTER [`Self::read_into`] with the same scratch (reuses the heads).
    /// Zero-alloc; deterministic.
    pub fn best_gated_row_cosine(
        &self,
        rows: &[[f32; D]],
        scratch: &SoftReadScratch<D, K>,
        gold: &[f32; D],
    ) -> f32 {
        let gold = unit(*gold);
        let mut best = f32::NEG_INFINITY;
        for head in &scratch.heads {
            // Top-gate row for this head = argmax cosine (monotone gate).
            let mut top_row = 0usize;
            let mut top_dot = f32::NEG_INFINITY;
            for (i, r) in rows.iter().enumerate() {
                let d: f32 = head.iter().zip(r).map(|(a, x)| a * x).sum();
                if d > top_dot {
                    top_dot = d;
                    top_row = i;
                }
            }
            let row = unit(rows[top_row]);
            let c: f32 = row.iter().zip(&gold).map(|(a, b)| a * b).sum();
            if c > best {
                best = c;
            }
        }
        best
    }
}

/// T6 — the CorpusDistanceGate second-signal composition. The max-sim path
/// is UNCHANGED (this composes the gate, never forks it): the existing
/// `CorpusDistanceGate` file is untouched, so feature-off byte-identity is
/// trivial. The added signal: `soft_confidence = max_k ⟨normalize(M_k),
/// q⟩` — how well SOME slot view reconstructs the query from the store
/// (in-corpus queries reconstruct well; OOD queries read spread).
#[cfg(feature = "distance_abstain")]
pub mod distance_soft {
    use super::{SlotMode, SoftRead, SoftReadScratch, unit};
    use crate::distance_abstain::CorpusDistanceGate;
    use std::cell::RefCell;

    pub struct DistanceSoftGate<const D: usize, const K: usize> {
        gate: CorpusDistanceGate<D>,
        soft: SoftRead<D, K>,
        scratch: RefCell<SoftReadScratch<D, K>>,
    }

    impl<const D: usize, const K: usize> DistanceSoftGate<D, K> {
        pub fn new(rows: &[[f32; D]], mid: f32, scale: f32) -> Self {
            Self {
                gate: CorpusDistanceGate::<D>::new(rows, mid, scale),
                soft: SoftRead::<D, K>::with_mode(rows, SlotMode::FarthestPoint),
                scratch: RefCell::new(SoftReadScratch::new()),
            }
        }

        /// The ORIGINAL max-similarity signal — verbatim composition over
        /// the untouched gate (the byte-identity seam).
        pub fn max_similarity(&self, q: &[f32; D]) -> f32 {
            self.gate.max_similarity(q)
        }

        pub fn abstain_confidence(&self, q: &[f32; D]) -> f32 {
            self.gate.abstain_confidence(q)
        }

        /// The query-conditioned second signal: best slot reconstruction
        /// cosine to the query itself.
        pub fn soft_confidence(&self, q: &[f32; D]) -> f32 {
            let mut scratch = self.scratch.borrow_mut();
            let reads = self.soft.read_into(q, self.gate_rows(), &mut scratch);
            let q = unit(*q);
            let mut best = f32::NEG_INFINITY;
            for m in &reads {
                let m = unit(*m);
                let c: f32 = m.iter().zip(&q).map(|(a, b)| a * b).sum();
                if c > best {
                    best = c;
                }
            }
            best
        }

        fn gate_rows(&self) -> &[[f32; D]] {
            // The gate's normalized rows ARE the store this gate reads.
            self.gate.rows()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot_bank::SplitMix64;

    const D: usize = 16;
    const K: usize = 4;

    fn seeded_store(seed: u64, n: usize) -> Vec<[f32; D]> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| unit(std::array::from_fn(|_| rng.next_sym())))
            .collect()
    }

    #[test]
    fn read_is_bounded_and_nan_free_at_every_store_size() {
        for n in [1usize, 16, 512] {
            let rows = seeded_store(985 + n as u64, n);
            let soft = SoftRead::<D, K>::new(&rows);
            let mut scratch = SoftReadScratch::new();
            let reads = soft.read_into(&rows[0], &rows, &mut scratch);
            for (k, m) in reads.iter().enumerate() {
                let mag: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    mag.is_finite() && mag <= 1.0 + 1e-3,
                    "N={n} slot {k}: read magnitude {mag} exceeds the longest stored row"
                );
            }
        }
    }

    #[test]
    fn replay_is_bit_identical() {
        let rows = seeded_store(986, 64);
        let a = SoftRead::<D, K>::new(&rows);
        let b = SoftRead::<D, K>::new(&rows);
        let mut s1 = SoftReadScratch::new();
        let mut s2 = SoftReadScratch::new();
        let q = seeded_store(987, 1).remove(0);
        let r1 = a.read_into(&q, &rows, &mut s1);
        let r2 = b.read_into(&q, &rows, &mut s2);
        assert_eq!(r1, r2, "(store, query) replay must be bit-identical");
    }

    #[test]
    fn zero_gates_collapse_the_read_toward_zero() {
        // b_k driven very negative: every gate underflows, Σg ≪ ε, and the
        // normalized read must return ≈ 0 — "nothing relevant looks like an
        // empty memory" (the read-side empty floor).
        let rows = seeded_store(988, 64);
        let soft = SoftRead::<D, K>::new(&rows).with_biases([-50.0; K]);
        let mut scratch = SoftReadScratch::new();
        let reads = soft.read_into(&rows[0], &rows, &mut scratch);
        for (k, m) in reads.iter().enumerate() {
            let mag: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                mag < 1e-4,
                "slot {k}: zero-gate read magnitude {mag} not ≈ 0"
            );
        }
    }
}
