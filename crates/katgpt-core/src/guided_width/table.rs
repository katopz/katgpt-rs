//! T5 — the success-SVD direction table: the μ≠0 fill on the belief host.
//!
//! **Fit (offline, setup-time):** outcome-weighted SVD over logged successful
//! displacements `Δh_i` — rows `√w_i · Δh_i`, factored by the shared one-sided
//! Jacobi kernel `thin_svd_into` (the same kernel `katgpt-canon`'s
//! `fit_joint_svd_pair` delegates to; see the module doc of
//! [`super`] for why the pair form is not called). The top-`r` right singular
//! vectors are the table's directions; each carries a sign-bias `b_j ∈ [0, 1]`
//! (|weighted mean cosine| of the logged rows with the direction, sign folded
//! into the direction) — `b_j` near 1 means successes move ONE way along
//! `d_j`, near 0 means they split ± (a multi-solution axis).
//!
//! **Freeze/thaw:** the table is immutable once built and BLAKE3-committed
//! over a domain tag, `d`, `r`, the direction bits and the bias bits
//! ([`DirectionTable::freeze`] / [`DirectionTable::thaw`]); a thaw whose bytes
//! do not reproduce the expected commitment is refused.
//!
//! **Beta posterior (runtime latent overlay):** [`DirectionPosterior`] keeps
//! per-direction success/failure counts and ranks directions by the
//! conservative ε-quantile [`crate::best_belief::best_belief_score`]. It is
//! NOT part of the commitment — a posterior is latent routing state, the
//! table is the frozen artefact.
//!
//! No training, no gradients: the fit is a deterministic factorisation of
//! logged data.

use crate::best_belief::best_belief_score;
use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

use super::types::MAX_DIRECTIONS;

const DOMAIN_TAG: &[u8] = b"katgpt.guided_width.direction_table.v1";
const MAGIC: [u8; 4] = *b"GWT1";

/// Why a [`DirectionTable::thaw`] was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThawError {
    /// Wrong magic / truncated / inconsistent lengths.
    Malformed,
    /// A direction or bias is non-finite, or `r > MAX_DIRECTIONS`.
    Invalid,
    /// The bytes do not reproduce the embedded or expected commitment.
    CommitmentMismatch,
}

/// A frozen, BLAKE3-committed table of unit guidance directions.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectionTable {
    d: usize,
    r: usize,
    dirs: Vec<f32>,
    bias: Vec<f32>,
    commitment: [u8; 32],
}

/// Reusable fit workspace (setup-time; grows on demand).
pub struct DirectionFitScratch {
    rows: Vec<f32>,
    svd: SvdResultScratch,
    work: SvdScratch,
    m_cap: usize,
    d_cap: usize,
}

impl DirectionFitScratch {
    /// Workspace for up to `max_rows` logged displacements of dimension `d`.
    pub fn with_capacity(max_rows: usize, d: usize) -> Self {
        let m = max_rows.max(d).max(1);
        Self {
            rows: Vec::with_capacity(m * d),
            svd: SvdResultScratch::with_capacity(m, d),
            work: SvdScratch::with_capacity(d, m),
            m_cap: m,
            d_cap: d,
        }
    }
}

fn commit(d: usize, r: usize, dirs: &[f32], bias: &[f32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(DOMAIN_TAG);
    h.update(&(d as u32).to_le_bytes());
    h.update(&(r as u32).to_le_bytes());
    for x in dirs {
        h.update(&x.to_bits().to_le_bytes());
    }
    for x in bias {
        h.update(&x.to_bits().to_le_bytes());
    }
    *h.finalize().as_bytes()
}

impl DirectionTable {
    /// Build from explicit parts: `dirs` is `r × d` (each row normalised to
    /// unit length here; a zero/non-finite row is refused), `bias[r]` is
    /// clamped to `[0, 1]`. `r` may be `0` (the empty table behaves exactly
    /// like no table). Returns `None` on any invalid input.
    pub fn from_parts(d: usize, mut dirs: Vec<f32>, mut bias: Vec<f32>) -> Option<Self> {
        if d == 0 || !dirs.len().is_multiple_of(d) {
            return None;
        }
        let r = dirs.len() / d;
        if r > MAX_DIRECTIONS || bias.len() != r {
            return None;
        }
        for row in dirs.chunks_mut(d) {
            let mut nsq = 0.0f32;
            for &x in row.iter() {
                nsq += x * x;
            }
            let n = nsq.sqrt();
            if !(n.is_finite() && n > 1e-12) {
                return None;
            }
            for x in row.iter_mut() {
                *x /= n;
            }
        }
        for b in bias.iter_mut() {
            if !b.is_finite() {
                return None;
            }
            *b = b.clamp(0.0, 1.0);
        }
        let commitment = commit(d, r, &dirs, &bias);
        Some(Self {
            d,
            r,
            dirs,
            bias,
            commitment,
        })
    }

    /// Outcome-weighted success-SVD fit over `m = weights.len()` logged
    /// displacements `deltas` (`m × d`, row-major). Rows with a non-positive
    /// or non-finite weight, or a zero / non-finite displacement, are
    /// skipped. Keeps up to `r` directions whose singular value is at least
    /// `1e-4 × σ₀`. Returns `None` when no usable row exists.
    pub fn fit(
        deltas: &[f32],
        weights: &[f32],
        d: usize,
        r: usize,
        scratch: &mut DirectionFitScratch,
    ) -> Option<Self> {
        let m = weights.len();
        if d == 0 || deltas.len() != m * d {
            return None;
        }
        let r = r.min(MAX_DIRECTIONS).min(d);
        // Weighted rows √w·Δh; zero-pad to ≥ d rows (the Jacobi kernel wants
        // m ≥ n; zero rows do not move the right singular vectors).
        scratch.rows.clear();
        let mut used = 0usize;
        let mut wsum = 0.0f32;
        for i in 0..m {
            let w = weights[i];
            let row = &deltas[i * d..(i + 1) * d];
            let nsq: f32 = row.iter().map(|x| x * x).sum();
            if !(w.is_finite() && w > 0.0 && nsq.is_finite() && nsq > 1e-24) {
                continue;
            }
            let sw = w.sqrt();
            scratch.rows.extend(row.iter().map(|x| sw * x));
            used += 1;
            wsum += w;
        }
        if used == 0 || r == 0 {
            return None;
        }
        let rows_n = used.max(d);
        scratch.rows.resize(rows_n * d, 0.0);
        if rows_n > scratch.m_cap || d > scratch.d_cap {
            scratch.svd = SvdResultScratch::with_capacity(rows_n, d);
            scratch.work = SvdScratch::with_capacity(d, rows_n);
            scratch.m_cap = rows_n;
            scratch.d_cap = d;
        }
        thin_svd_into(&scratch.rows, rows_n, d, &mut scratch.svd, &mut scratch.work);
        let s0 = scratch.svd.singular_value(0);
        if !(s0.is_finite() && s0 > 0.0) {
            return None;
        }
        let mut dirs = Vec::with_capacity(r * d);
        let mut bias = Vec::with_capacity(r);
        for j in 0..r.min(scratch.svd.len()) {
            if scratch.svd.singular_value(j) < 1e-4 * s0 {
                break;
            }
            let v = scratch.svd.right_singular_vector(j);
            // Sign-bias: weighted mean cosine of the successes with v.
            let mut acc = 0.0f32;
            for i in 0..m {
                let w = weights[i];
                let row = &deltas[i * d..(i + 1) * d];
                let nsq: f32 = row.iter().map(|x| x * x).sum();
                if !(w.is_finite() && w > 0.0 && nsq.is_finite() && nsq > 1e-24) {
                    continue;
                }
                let dot: f32 = row.iter().zip(v).map(|(a, b)| a * b).sum();
                acc += w * dot / nsq.sqrt();
            }
            let mean = acc / wsum;
            let sign = if mean < 0.0 { -1.0 } else { 1.0 };
            dirs.extend(v.iter().map(|x| sign * x));
            bias.push(mean.abs());
        }
        Self::from_parts(d, dirs, bias)
    }

    /// Latent dimension.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Number of directions.
    pub fn len(&self) -> usize {
        self.r
    }

    /// `true` for a zero-direction table.
    pub fn is_empty(&self) -> bool {
        self.r == 0
    }

    /// Unit direction `j`.
    pub fn direction(&self, j: usize) -> &[f32] {
        &self.dirs[j * self.d..(j + 1) * self.d]
    }

    /// Sign-bias of direction `j` in `[0, 1]`.
    pub fn bias(&self, j: usize) -> f32 {
        self.bias[j]
    }

    /// BLAKE3 commitment over the frozen content.
    pub fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Freeze into a self-describing envelope:
    /// `magic ‖ d:u32 ‖ r:u32 ‖ dirs:f32[r·d] ‖ bias:f32[r] ‖ commitment:[u8;32]`.
    pub fn freeze(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 8 + 4 * (self.dirs.len() + self.r) + 32);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(self.d as u32).to_le_bytes());
        out.extend_from_slice(&(self.r as u32).to_le_bytes());
        for x in &self.dirs {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        for x in &self.bias {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&self.commitment);
        out
    }

    /// Thaw a [`Self::freeze`] envelope. The content must reproduce the
    /// embedded commitment, and — when `expected` is given — equal it.
    /// Directions are taken bit-exactly (no renormalisation on thaw).
    pub fn thaw(bytes: &[u8], expected: Option<[u8; 32]>) -> Result<Self, ThawError> {
        if bytes.len() < 12 + 32 || bytes[..4] != MAGIC {
            return Err(ThawError::Malformed);
        }
        let rd = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        let d = rd(4) as usize;
        let r = rd(8) as usize;
        if d == 0 || r > MAX_DIRECTIONS {
            return Err(ThawError::Invalid);
        }
        let n_f = r * d + r;
        if bytes.len() != 12 + 4 * n_f + 32 {
            return Err(ThawError::Malformed);
        }
        let mut vals = Vec::with_capacity(n_f);
        for i in 0..n_f {
            let x = f32::from_bits(rd(12 + 4 * i));
            if !x.is_finite() {
                return Err(ThawError::Invalid);
            }
            vals.push(x);
        }
        let bias = vals.split_off(r * d);
        let dirs = vals;
        let mut embedded = [0u8; 32];
        embedded.copy_from_slice(&bytes[12 + 4 * n_f..]);
        let c = commit(d, r, &dirs, &bias);
        if c != embedded || expected.is_some_and(|e| e != c) {
            return Err(ThawError::CommitmentMismatch);
        }
        Ok(Self {
            d,
            r,
            dirs,
            bias,
            commitment: c,
        })
    }
}

/// Per-direction Beta posterior counts (runtime latent overlay on a frozen
/// [`DirectionTable`]). Fixed-size, zero-allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectionPosterior {
    succ: [u32; MAX_DIRECTIONS],
    fail: [u32; MAX_DIRECTIONS],
    r: usize,
}

impl DirectionPosterior {
    /// Uniform `Beta(1, 1)` prior over `r` directions (clamped to
    /// [`MAX_DIRECTIONS`]).
    pub fn new(r: usize) -> Self {
        Self {
            succ: [0; MAX_DIRECTIONS],
            fail: [0; MAX_DIRECTIONS],
            r: r.min(MAX_DIRECTIONS),
        }
    }

    /// Record one outcome for direction `j` (out of range: ignored).
    pub fn record(&mut self, j: usize, success: bool) {
        if j < self.r {
            if success {
                self.succ[j] = self.succ[j].saturating_add(1);
            } else {
                self.fail[j] = self.fail[j].saturating_add(1);
            }
        }
    }

    /// `(successes, failures)` of direction `j`.
    pub fn counts(&self, j: usize) -> (u32, u32) {
        (self.succ[j], self.fail[j])
    }

    /// Conservative ε-quantile score of direction `j`.
    pub fn score(&self, j: usize, epsilon: f32) -> f32 {
        best_belief_score(self.succ[j], self.fail[j], epsilon)
    }

    /// Write direction indices into `order[..r]`, best score first (stable:
    /// equal scores keep index order, so a fresh posterior ranks by SVD
    /// order). Returns the count written. Zero-allocation (insertion sort,
    /// `r ≤ 32`).
    pub fn rank_into(&self, epsilon: f32, order: &mut [u16]) -> usize {
        let r = self.r.min(order.len());
        let mut scores = [0.0f32; MAX_DIRECTIONS];
        for j in 0..r {
            order[j] = j as u16;
            let s = self.score(j, epsilon);
            scores[j] = if s.is_finite() { s } else { f32::NEG_INFINITY };
        }
        for i in 1..r {
            let mut k = i;
            while k > 0 && scores[order[k] as usize] > scores[order[k - 1] as usize] {
                order.swap(k, k - 1);
                k -= 1;
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_the_dominant_success_axis_with_its_sign() {
        let d = 6;
        let mut deltas = Vec::new();
        let mut w = Vec::new();
        for i in 0..40 {
            let t = (i as f32 * 0.37).sin() * 0.05;
            deltas.extend_from_slice(&[1.0, 0.5 + t, 0.0, -t, 0.0, 0.0]);
            w.push(1.0);
        }
        let mut s = DirectionFitScratch::with_capacity(40, d);
        let tab = DirectionTable::fit(&deltas, &w, d, 3, &mut s).unwrap();
        let d0 = tab.direction(0);
        let want = [1.0f32 / 1.25f32.sqrt(), 0.5 / 1.25f32.sqrt()];
        assert!((d0[0] - want[0]).abs() < 0.02 && (d0[1] - want[1]).abs() < 0.02, "{d0:?}");
        assert!(tab.bias(0) > 0.95, "one-sided successes ⇒ bias ≈ 1");
    }

    #[test]
    fn split_successes_have_low_bias() {
        let d = 4;
        let mut deltas = Vec::new();
        for i in 0..20 {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            deltas.extend_from_slice(&[s, 0.0, 0.01 * i as f32, 0.0]);
        }
        let w = vec![1.0f32; 20];
        let mut sc = DirectionFitScratch::with_capacity(20, d);
        let tab = DirectionTable::fit(&deltas, &w, d, 1, &mut sc).unwrap();
        assert!(tab.bias(0) < 0.1, "bias {}", tab.bias(0));
    }

    #[test]
    fn freeze_thaw_roundtrip_and_tamper_refusal() {
        let tab =
            DirectionTable::from_parts(3, vec![1.0, 0.0, 0.0, 0.0, 3.0, 4.0], vec![0.9, 0.2])
                .unwrap();
        assert!((tab.direction(1)[1] - 0.6).abs() < 1e-6);
        let bytes = tab.freeze();
        let back = DirectionTable::thaw(&bytes, Some(tab.commitment())).unwrap();
        assert_eq!(back, tab);
        let mut bad = bytes.clone();
        bad[14] ^= 1;
        assert_eq!(DirectionTable::thaw(&bad, None), Err(ThawError::CommitmentMismatch));
        assert_eq!(
            DirectionTable::thaw(&bytes, Some([7u8; 32])),
            Err(ThawError::CommitmentMismatch)
        );
        assert_eq!(DirectionTable::thaw(&bytes[..10], None), Err(ThawError::Malformed));
    }

    #[test]
    fn invalid_parts_are_refused() {
        assert!(DirectionTable::from_parts(2, vec![0.0, 0.0], vec![0.5]).is_none());
        assert!(DirectionTable::from_parts(2, vec![f32::NAN, 1.0], vec![0.5]).is_none());
        assert!(DirectionTable::from_parts(2, vec![1.0, 0.0], vec![]).is_none());
        let empty = DirectionTable::from_parts(2, vec![], vec![]).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn posterior_ranks_by_the_conservative_bound() {
        let mut p = DirectionPosterior::new(3);
        let mut order = [0u16; 3];
        p.rank_into(0.05, &mut order);
        assert_eq!(order, [0, 1, 2], "fresh posterior keeps SVD order");
        for _ in 0..6 {
            p.record(2, true);
            p.record(0, false);
        }
        p.rank_into(0.05, &mut order);
        assert_eq!(order, [2, 1, 0]);
    }
}
