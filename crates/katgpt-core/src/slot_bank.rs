//! Slot bank — K deterministic store-derived anchors (Issue 985 T1, the
//! Research 656 Memorilla distillation's modelless floor form).
//!
//! Memorilla's slots are Xavier-LEARNED; the paper's own residual design
//! note says slots need to be mutually DISTINCT, not pretrained. The
//! modelless floor replaces learning with deterministic construction:
//!
//! - **Farthest-point anchors** (default) — ON the data manifold: the first
//!   anchor is the row nearest the store centroid (deterministic start),
//!   each next anchor maximizes the min cosine-distance to the chosen set.
//!   A store with multi-hop / multi-cluster structure therefore spreads its
//!   anchors along that structure — which is exactly what lets a
//!   query-conditioned head read a neighborhood the raw query cannot rank.
//! - **Seeded orthonormal directions** — geometry-partitioning views
//!   independent of the store layout (the Hadamard-class constructor). For
//!   D a power of two, rows of the Walsh-Hadamard matrix selected by a
//!   BLAKE3-seeded index set; otherwise Gram-Schmidt over seeded draws.
//!
//! Determinism: every construction is keyed `BLAKE3(store bits + label)` —
//! the same store epoch reproduces byte-identical slots across runs and
//! platforms (G1). ANY append changes the store digest and therefore the
//! key: re-derivation is a FULL recompute, never incremental (pinned here —
//! an incremental slot update would drift from the digest-keyed determinism
//! law; full recompute at these sizes is microseconds).
//!
//! Coverage: `coverage_entropy` measures how evenly the slots tile the
//! store (rows assigned to their nearest slot; normalized Shannon entropy
//! of the assignment histogram). The even-tiling gate asserts a floor —
//! degenerate slot banks (all anchors in one cluster) are caught, not
//! inherited.
//!
//! Opt-in (`soft_read`); zero production consumers until Issue 985's gate
//! promotes one. Local-only: slots never leave the process.

use blake3::Hasher;

/// Unit-normalize; a zero vector passes through (never NaN).
fn unit<const D: usize>(v: [f32; D]) -> [f32; D] {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        v.map(|x| x * inv)
    } else {
        v
    }
}

/// Deterministic `SplitMix64` (the workspace idiom — no rand dep).
pub(crate) struct SplitMix64(u64);
impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub(crate) fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub(crate) fn next_sym(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

/// BLAKE3 domain digest over a label + the store's bit pattern. The slot
/// cache key: same store + label → same slots; any append flips the digest.
fn store_digest<const D: usize>(label: &str, rows: &[[f32; D]]) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(label.as_bytes());
    h.update(&(rows.len() as u64).to_le_bytes());
    for r in rows {
        for x in r {
            h.update(&x.to_bits().to_le_bytes());
        }
    }
    h.finalize().into()
}

/// Gate on a cosine at a per-slot bias: g = σ(cos/τ + b). The sigmoid
/// never softmax — house law.
#[inline]
pub(crate) fn gate(cos: f32, tau: f32, b: f32) -> f32 {
    crate::exact_sigmoid(cos / tau + b)
}

/// Slot construction mode (preregistered: the GOAT cell gates on
/// [`SlotMode::FarthestPoint`]; [`SlotMode::Orthonormal`] prints as the
/// comparison arm).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotMode {
    /// Farthest-point anchors over the store rows (on-manifold).
    FarthestPoint,
    /// BLAKE3-seeded orthonormal directions (geometry-partitioning).
    Orthonormal,
}

/// K deterministic slots over a store, plus the per-slot conditioning
/// schedule. `betas` spans [0, 1] linearly across slots — β=0 is the
/// unconditioned plain head (q_k = q), β=1 the fully slot-pulled view;
/// distinctness lives in the residual path per the paper's Eq. 3 note.
pub struct SlotBank<const D: usize> {
    slots: Vec<[f32; D]>,
    betas: Vec<f32>,
    digest: [u8; 32],
}

impl<const D: usize> SlotBank<D> {
    /// Build (and key) the bank for a store epoch. K > N is legal: FPS
    /// yields N manifold anchors and the remainder is seeded orthonormal
    /// fill (distinct slots, never repeats). An empty store refuses loud —
    /// a bank over nothing is a caller bug, not an empty read (the
    /// empty-memory floor lives in the soft read, not the bank).
    pub fn new(rows: &[[f32; D]], k: usize, mode: SlotMode) -> Self {
        assert!(
            !rows.is_empty(),
            "slot bank over an empty store — caller bug"
        );
        assert!(k > 0, "slot bank needs k >= 1");
        if mode == SlotMode::Orthonormal {
            assert!(k <= D, "orthonormal slots cap at D ({k} > {D})");
        }
        let digest = store_digest("soft_read:slot_bank:v1", rows);
        let mut slots = match mode {
            SlotMode::FarthestPoint => farthest_point_anchors(rows, k),
            SlotMode::Orthonormal => seeded_orthonormal(&digest, k),
        };
        // K > N fill: the manifold anchors are the first N; the remainder
        // is seeded Gram-Schmidt fill against the accepted set (distinct
        // slots, never repeats — a repeated slot would double-count one
        // neighborhood in the gated read).
        if slots.len() < k {
            let seed = u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"));
            let mut rng = SplitMix64::new(seed ^ 0xF11A_F11A_F11A_F11A);
            while slots.len() < k {
                let mut v: [f32; D] = std::array::from_fn(|_| rng.next_sym());
                for s in &slots {
                    let dot: f32 = s.iter().zip(v).map(|(a, b)| a * b).sum();
                    for (x, sv) in v.iter_mut().zip(s) {
                        *x -= dot * sv;
                    }
                }
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if n < 1e-4 {
                    continue; // degenerate draw — redraw (seeded, deterministic)
                }
                let inv = 1.0 / n;
                slots.push(v.map(|x| x * inv));
            }
        }
        for s in slots.iter_mut() {
            *s = unit(*s);
        }
        let betas = (0..k)
            .map(|i| {
                if k == 1 {
                    0.0
                } else {
                    i as f32 / (k - 1) as f32
                }
            })
            .collect();
        Self {
            slots,
            betas,
            digest,
        }
    }

    /// Slot count (may exceed the store size via orthonormal fill).
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The store digest this bank was derived from — the cache key.
    pub fn store_digest(&self) -> [u8; 32] {
        self.digest
    }

    /// True when this bank is still keyed to `rows` (the memoize-by-digest
    /// consumer seam: keep ONE bank per store epoch, rebuild on append;
    /// the check recomputes the store digest — cache-validity, never a hot
    /// path).
    pub fn is_fresh(&self, rows: &[[f32; D]]) -> bool {
        store_digest("soft_read:slot_bank:v1", rows) == self.digest
    }

    pub(crate) fn slot(&self, k: usize) -> &[f32; D] {
        &self.slots[k]
    }

    pub(crate) fn beta(&self, k: usize) -> f32 {
        self.betas[k]
    }

    /// All slots (read-only; tests and bench diagnostics).
    #[cfg(test)]
    pub(crate) fn slots(&self) -> &[[f32; D]] {
        &self.slots
    }

    /// Normalized Shannon entropy of the nearest-slot assignment histogram
    /// (rows → argmax cosine slot). 1.0 = perfectly even tiling; → 0 = all
    /// rows claimed by one slot. The even-tiling gate reads this.
    pub fn coverage_entropy(&self, rows: &[[f32; D]]) -> f32 {
        if rows.is_empty() || self.slots.is_empty() {
            return 0.0;
        }
        let mut counts = vec![0usize; self.slots.len()];
        for r in rows {
            let mut best = 0usize;
            let mut best_dot = f32::NEG_INFINITY;
            for (k, s) in self.slots.iter().enumerate() {
                let d: f32 = s.iter().zip(r).map(|(a, b)| a * b).sum();
                if d > best_dot {
                    best_dot = d;
                    best = k;
                }
            }
            counts[best] += 1;
        }
        let n = rows.len() as f32;
        let mut h = 0.0f32;
        for c in &counts {
            if *c > 0 {
                let p = *c as f32 / n;
                h -= p * p.ln();
            }
        }
        let used = counts.iter().filter(|c| **c > 0).count().max(1) as f32;
        h / used.ln()
    }
}

/// Farthest-point anchors: start at the row nearest the centroid
/// (deterministic), then repeatedly take the row maximizing the min
/// cosine-distance to the chosen set. O(K·N·D) — construction-time only.
fn farthest_point_anchors<const D: usize>(rows: &[[f32; D]], k: usize) -> Vec<[f32; D]> {
    let n = rows.len();
    let normalized: Vec<[f32; D]> = rows.iter().map(|r| unit(*r)).collect();
    let centroid: [f32; D] = {
        let mut c = [0.0f32; D];
        for r in &normalized {
            for (x, v) in c.iter_mut().zip(r) {
                *x += v;
            }
        }
        let inv = 1.0 / n as f32;
        c.map(|x| x * inv)
    };
    let centroid = unit(centroid);
    let dist = |a: &[f32; D], b: &[f32; D]| 1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    // First anchor: nearest the centroid.
    let mut chosen: Vec<usize> = Vec::with_capacity(k.min(n));
    let mut best0 = 0usize;
    let mut best0d = f32::INFINITY;
    for (i, r) in normalized.iter().enumerate() {
        let d = dist(r, &centroid);
        if d < best0d {
            best0d = d;
            best0 = i;
        }
    }
    chosen.push(best0);
    // Min-distance-to-chosen cache, updated per pick.
    let mut min_d: Vec<f32> = (0..n)
        .map(|i| dist(&normalized[i], &normalized[best0]))
        .collect();
    while chosen.len() < k.min(n) {
        let mut pick = 0usize;
        let mut pick_d = f32::NEG_INFINITY;
        for (i, d) in min_d.iter().enumerate() {
            if *d > pick_d {
                pick_d = *d;
                pick = i;
            }
        }
        chosen.push(pick);
        let last = &normalized[pick];
        for (md, r) in min_d.iter_mut().zip(&normalized) {
            let d = dist(r, last);
            if d < *md {
                *md = d;
            }
        }
    }
    chosen.into_iter().map(|i| normalized[i]).collect()
}

/// Seeded orthonormal directions: D power-of-two → Walsh-Hadamard rows at a
/// BLAKE3-seeded index set (exact ±1/√D orthogonality); otherwise
/// Gram-Schmidt over seeded draws (≤ 1e-5 orthogonality residual at these
/// sizes — construction-time only, never on the hot path).
fn seeded_orthonormal<const D: usize>(digest: &[u8; 32], k: usize) -> Vec<[f32; D]> {
    let seed = u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"));
    let mut rng = SplitMix64::new(seed ^ 0x5107_5107_5107_5107);
    let is_pow2 = D.is_power_of_two();
    let mut out: Vec<[f32; D]> = Vec::with_capacity(k);
    if is_pow2 {
        let scale = 1.0 / (D as f32).sqrt();
        let mut used: Vec<usize> = Vec::with_capacity(k);
        while out.len() < k {
            let idx = (rng.next_u64() as usize) % D;
            if used.contains(&idx) {
                continue;
            }
            used.push(idx);
            // Row `idx` of the natural-order Walsh-Hadamard matrix: entry j
            // = parity(popcount(idx & j)).
            let row: [f32; D] = std::array::from_fn(|j| {
                let par = ((idx & j).count_ones() & 1) as f32;
                (if par > 0.5 { -1.0 } else { 1.0 }) * scale
            });
            out.push(row);
        }
    } else {
        while out.len() < k {
            let mut v: [f32; D] = std::array::from_fn(|_| rng.next_sym());
            // Gram-Schmidt against the accepted set.
            for s in &out {
                let dot: f32 = s.iter().zip(v).map(|(a, b)| a * b).sum();
                for (x, sv) in v.iter_mut().zip(s) {
                    *x -= dot * sv;
                }
            }
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n < 1e-4 {
                continue; // degenerate draw — redraw (seeded stream, deterministic)
            }
            let inv = 1.0 / n;
            out.push(v.map(|x| x * inv));
        }
    }
    // Determinism sanity: the construction must never spin (k ≤ some sane
    // bound and D ≥ k reachability). Refuse loud instead of looping.
    assert!(out.len() == k, "seeded orthonormal construction fell short");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 16;

    fn seeded_rows(seed: u64, n: usize, clusters: usize) -> Vec<[f32; D]> {
        let mut rng = SplitMix64::new(seed);
        let mut rows = Vec::with_capacity(n);
        for c in 0..clusters {
            let center: [f32; D] =
                std::array::from_fn(|i| if (i + c) % 2 == 0 { 1.0 } else { -1.0 });
            for _ in 0..n / clusters {
                let row = unit(center.map(|x| x + 0.1 * rng.next_sym()));
                rows.push(row);
            }
        }
        rows
    }

    #[test]
    fn fps_anchors_are_deterministic_and_store_keyed() {
        let rows = seeded_rows(41, 64, 4);
        let a = SlotBank::<D>::new(&rows, 8, SlotMode::FarthestPoint);
        let b = SlotBank::<D>::new(&rows, 8, SlotMode::FarthestPoint);
        assert_eq!(a.slots(), b.slots(), "same store → byte-identical anchors");
        let mut rows2 = rows.clone();
        rows2[0][0] += 0.01; // an append-equivalent content change
        let c = SlotBank::<D>::new(&rows2, 8, SlotMode::FarthestPoint);
        assert_ne!(
            a.store_digest(),
            c.store_digest(),
            "content flip must rekey"
        );
    }

    #[test]
    fn fps_covers_clusters_evenly() {
        let rows = seeded_rows(42, 64, 4);
        let bank = SlotBank::<D>::new(&rows, 8, SlotMode::FarthestPoint);
        let h = bank.coverage_entropy(&rows);
        assert!(
            h >= 0.5,
            "even-tiling gate: 4 well-separated clusters with K=8 anchors must tile (H={h:.3})"
        );
    }

    #[test]
    fn k_above_n_fills_orthogonally_never_repeats() {
        let rows = seeded_rows(43, 8, 2);
        let bank = SlotBank::<D>::new(&rows, 16, SlotMode::FarthestPoint);
        assert_eq!(bank.len(), 16);
        let slots = bank.slots();
        for i in 0..slots.len() {
            for j in (i + 1)..slots.len() {
                let d: f32 = slots[i].iter().zip(slots[j]).map(|(a, b)| a * b).sum();
                // The first 8 are store rows (mutually non-orthogonal is
                // fine); the FILL slots must be distinct rows — exact
                // repeats are the failure mode this asserts against.
                assert!(
                    (d - 1.0).abs() > 1e-4,
                    "slots {i} and {j} are repeats (cos {d:.6})"
                );
            }
        }
    }

    #[test]
    fn orthonormal_directions_are_orthogonal() {
        let rows = seeded_rows(44, 16, 2);
        let bank = SlotBank::<D>::new(&rows, 8, SlotMode::Orthonormal);
        let slots = bank.slots();
        for i in 0..slots.len() {
            for j in (i + 1)..slots.len() {
                let d: f32 = slots[i].iter().zip(slots[j]).map(|(a, b)| a * b).sum();
                assert!(d.abs() < 1e-4, "slots {i},{j} not orthogonal (cos {d:.6})");
            }
        }
    }

    #[test]
    fn empty_store_refuses_loud() {
        let rows: Vec<[f32; D]> = Vec::new();
        let r = std::panic::catch_unwind(|| SlotBank::<D>::new(&rows, 4, SlotMode::FarthestPoint));
        assert!(r.is_err(), "empty-store bank must refuse");
    }
}
