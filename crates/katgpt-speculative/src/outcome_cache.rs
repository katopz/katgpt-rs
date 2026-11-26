//! JetSpec++ outcome-distribution speculation cache (Issue 976).
//!
//! Generalizes the shipped acceptance machinery from a **marginal** forecast
//! ([`crate::acceptance_forecast::AcceptanceForecast`], Bebop's `α ≈ a − b·H`)
//! to the joint **outcome distribution** of one speculative round
//!
//! ```text
//! P(k, a) = P(k) · p̂(a | k),   P(k) = Π_{i<k} c_i · (1 − c_k)   (k < n)
//!                               P(n) = Π_{i<n} c_i
//! ```
//!
//! (JetSpec++ Eq 6, exactly-k form — the `(1 − c_k)` stop factor makes the
//! `(k, a)` cells disjoint events that sum to one). `c_i` is the drafter's
//! per-position self-confidence, `k` the accepted-prefix length, `a` the bonus
//! token the verifier emits after it. `Π c_i` is one pass of the shipped
//! [`katgpt_core::cumprodsum::cumprodsum_scalar`] (`x = 0` → cumprod);
//! `p̂(a | k)` is a k-indexed bonus histogram held as a fixed-size all-`u32`
//! Pod ([`BonusHistogramPod`], 288 B), BLAKE3-committed at freeze and
//! KL-convergence-gated — freeze/thaw-legal by construction.
//!
//! # Cache semantics
//!
//! A cache cell `(k, a)` pre-materializes "accept `k`, then bonus `a`". A round
//! **hits** when the verifier's realized outcome is a materialized cell; on a
//! pipelined / multi-request posture the drafter is then freed for that round.
//! Which cells are pre-drafted never touches the verified output — speculative
//! decoding stays lossless whatever the cache holds. The fallback slot ("bonus
//! outside the frozen top-`A`") carries probability mass but names no token,
//! so it is never selectable: a realized fallback outcome is a miss for every
//! policy (fair by construction).
//!
//! # Selection (Eq 7 + Prop 2b)
//!
//! [`select_top_b`]: top-`B` cells by score — disjoint events make hit mass
//! additive, so sort-by-score is optimal for fixed `B` (Eq 7).
//! [`select_conformal`]: the same order truncated at the split-conformal
//! threshold `λ` (adaptive cache size, capped at `B`). Two fixed-allocation
//! baselines ship for the paired Prop 2b comparison:
//! [`static_top_b`] (top-`B` cells by calibration-window realized frequency —
//! the strongest legal fixed policy) and [`geometric_ladder`] (SSD-shape
//! geometric k-prior × per-k bonus mass).
//!
//! # Calibration
//!
//! [`conformal_threshold`]: finite-sample split-conformal order statistic of
//! the realized-cell scores, so `P(score(realized) ≥ λ) ≥ 1 − α` on
//! exchangeable rounds. [`ConfidenceRecalibrator`]: a deterministic per-position
//! binned (histogram-binning) map raw drafter confidence → empirical
//! acceptance, counted over the calibration window — the modelless degraded
//! mode of JetSpec++'s logistic recalibration (no gradient, no IRLS).
//!
//! # Determinism (Issue 940)
//!
//! Chain products are a sequential fold; every selection scans the canonical
//! `(k asc, slot asc)` grid and keeps strictly-greater scores, so ties resolve
//! to the FIRST index. Histogram masses fold in `u64` then one `f32` divide.
//! Nothing here crosses the sync boundary (local scheduling hint only).
//!
//! # Posture
//!
//! Feature `outcome_cache` (opt-in, default-OFF). No serving path is wired.
//! Gate battery: `tests/outcome_cache_trace_gates.rs` on a frozen,
//! BLAKE3-pinned teacher-forced trace (Bench 990). Promotion stays gated on
//! the Issue 976 posture triggers (a multi-request serving caller).

use katgpt_core::cumprodsum::cumprodsum_scalar;

/// Max drafted positions per round (`γ`).
pub const GAMMA: usize = 8;
/// Bonus-token slots tracked per `k` (the frozen top-`A` vocabulary).
pub const A_SLOTS: usize = 8;
/// `k` rows: accepted-count outcomes `0..=GAMMA`.
pub const K_SLOTS: usize = GAMMA + 1;
/// Columns per `k` row: `A_SLOTS` concrete tokens + one fallback slot.
pub const ROW: usize = A_SLOTS + 1;
/// Cells in the outcome grid (`k`-major).
pub const GRID: usize = K_SLOTS * ROW;
/// Laplace pseudo-count for bonus mass (`β`).
pub const SMOOTHING_BETA: f32 = 0.5;
/// Minimum tail mass a round-conditioned bonus row leaves for backoff onto
/// the frozen Pod (out-of-support bonus tokens keep a non-zero score).
pub const BACKOFF_ESCAPE: f32 = 0.05;

/// Grid cell index of `(k, slot)` (`slot == A_SLOTS` is the fallback).
#[inline]
#[must_use]
pub const fn cell(k: usize, slot: usize) -> usize {
    k * ROW + slot
}

/// `true` when `cell` is a fallback cell (never selectable).
#[inline]
#[must_use]
pub const fn is_fallback(cell: usize) -> bool {
    cell % ROW == A_SLOTS
}

// ── Freeze/thaw Pod ─────────────────────────────────────────────────────────

/// The freeze/thaw-legal bonus histogram: fixed-size, all-`u32`, 288 B.
///
/// - `top[slot]` — frozen bonus vocabulary shared by every `k` row
///   (window count desc, first-occurrence tie — Issue 940).
/// - `counts[k][slot]` — bonus-token counts given `k` accepted positions
///   (`slot == A_SLOTS` = outside `top`).
/// - `converged` — 1 when the KL gate passed at freeze, 0 when forced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BonusHistogramPod {
    /// Wire tag `'BHP1'`.
    pub magic: u32,
    /// Frozen bonus vocabulary (token ids).
    pub top: [u32; A_SLOTS],
    /// Per-`k` bonus counts; last column = fallback.
    pub counts: [[u32; ROW]; K_SLOTS],
    /// Calibration rounds folded in.
    pub rounds_seen: u32,
    /// 1 = KL gate passed at freeze; 0 = forced freeze.
    pub converged: u32,
    /// Number of valid entries in `top` (≤ `A_SLOTS`).
    pub top_len: u32,
    /// Reserved (zero) — keeps the Pod at 72 `u32`s.
    pub reserved: [u32; 1],
}

impl BonusHistogramPod {
    /// The `'BHP1'` magic.
    pub const MAGIC: u32 = u32::from_le_bytes(*b"BHP1");
    /// Wire length: 72 little-endian `u32`s.
    pub const BYTE_LEN: usize = (1 + A_SLOTS + K_SLOTS * ROW + 4) * 4;

    /// Zeroed Pod with the magic set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            magic: Self::MAGIC,
            top: [0; A_SLOTS],
            counts: [[0; ROW]; K_SLOTS],
            rounds_seen: 0,
            converged: 0,
            top_len: 0,
            reserved: [0; 1],
        }
    }

    fn words(&self) -> [u32; Self::BYTE_LEN / 4] {
        let mut w = [0u32; Self::BYTE_LEN / 4];
        let mut i = 0;
        w[i] = self.magic;
        i += 1;
        for &t in &self.top {
            w[i] = t;
            i += 1;
        }
        for row in &self.counts {
            for &c in row {
                w[i] = c;
                i += 1;
            }
        }
        for v in [
            self.rounds_seen,
            self.converged,
            self.top_len,
            self.reserved[0],
        ] {
            w[i] = v;
            i += 1;
        }
        debug_assert_eq!(i, w.len());
        w
    }

    /// Little-endian wire bytes.
    #[must_use]
    pub fn commit_bytes(&self) -> [u8; Self::BYTE_LEN] {
        let mut out = [0u8; Self::BYTE_LEN];
        for (chunk, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.words()) {
            chunk.copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Parse wire bytes; `None` on length/magic mismatch or a corrupt `top_len`.
    #[must_use]
    pub fn from_commit_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::BYTE_LEN {
            return None;
        }
        let mut it = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        let mut next = || it.next().unwrap_or(0);
        let magic = next();
        if magic != Self::MAGIC {
            return None;
        }
        let mut pod = Self::new();
        for t in &mut pod.top {
            *t = next();
        }
        for row in &mut pod.counts {
            for c in row {
                *c = next();
            }
        }
        pod.rounds_seen = next();
        pod.converged = next();
        pod.top_len = next();
        pod.reserved[0] = next();
        if pod.top_len as usize > A_SLOTS {
            return None;
        }
        Some(pod)
    }

    /// BLAKE3 digest over [`Self::commit_bytes`] — the freeze/thaw commitment.
    #[must_use]
    pub fn blake3_digest(&self) -> [u8; 32] {
        blake3::hash(&self.commit_bytes()).into()
    }

    /// `false` on any drift from a frozen digest (refuse loud at the caller).
    #[must_use]
    pub fn verify_against(&self, digest: &[u8; 32]) -> bool {
        &self.blake3_digest() == digest
    }

    /// Slot of `token` in the frozen vocabulary, else the fallback slot.
    #[inline]
    #[must_use]
    pub fn slot_of(&self, token: u32) -> usize {
        let len = self.top_len as usize;
        self.top[..len]
            .iter()
            .position(|&t| t == token)
            .unwrap_or(A_SLOTS)
    }

    /// Laplace-smoothed `p̂(slot | k)`; unused vocab slots carry zero mass.
    #[inline]
    #[must_use]
    pub fn mass_slot(&self, k: usize, slot: usize) -> f32 {
        let live = slot == A_SLOTS || slot < self.top_len as usize;
        if !live {
            return 0.0;
        }
        let total = self.counts[k].iter().fold(0u64, |a, &v| a + u64::from(v));
        let cols = self.top_len as f32 + 1.0;
        (self.counts[k][slot] as f32 + SMOOTHING_BETA) / (total as f32 + SMOOTHING_BETA * cols)
    }
}

impl Default for BonusHistogramPod {
    fn default() -> Self {
        Self::new()
    }
}

// ── Calibration-window builder ──────────────────────────────────────────────

/// Freeze verdict from [`HistogramBuilder::try_freeze`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FreezeVerdict {
    /// KL gate passed; Pod carries `converged = 1`.
    Converged(BonusHistogramPod, f32),
    /// KL gate failed (extend the window and retry); carries the mean KL.
    NotConverged(f32),
    /// Forced freeze after a KL failure; Pod carries `converged = 0`.
    Forced(BonusHistogramPod, f32),
    /// No data.
    Empty,
}

/// Calibration-window accumulator (offline posture — may allocate). The
/// FROZEN [`BonusHistogramPod`] is what serving carries.
#[derive(Debug, Default, Clone)]
pub struct HistogramBuilder {
    /// Realized `(k, bonus_token)` in round order.
    log: Vec<(u8, u32)>,
}

impl HistogramBuilder {
    /// Empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one realized outcome.
    pub fn observe(&mut self, k: usize, bonus_token: u32) {
        debug_assert!(k < K_SLOTS);
        self.log.push((k.min(GAMMA) as u8, bonus_token));
    }

    /// Rounds folded so far.
    #[must_use]
    pub fn rounds_seen(&self) -> usize {
        self.log.len()
    }

    /// Frozen vocabulary: window count desc, first-occurrence tie (Issue 940).
    fn top_vocab(&self) -> ([u32; A_SLOTS], u32) {
        let mut seen: Vec<(u32, u64)> = Vec::new();
        for &(_, t) in &self.log {
            match seen.iter_mut().find(|(s, _)| *s == t) {
                Some(e) => e.1 += 1,
                None => seen.push((t, 1)),
            }
        }
        // Stable sort keeps first-occurrence order among equal counts.
        seen.sort_by_key(|e| std::cmp::Reverse(e.1));
        let mut top = [0u32; A_SLOTS];
        let len = seen.len().min(A_SLOTS);
        for (dst, src) in top.iter_mut().zip(&seen) {
            *dst = src.0;
        }
        (top, len as u32)
    }

    fn fill(log: &[(u8, u32)], top: [u32; A_SLOTS], top_len: u32) -> BonusHistogramPod {
        let mut pod = BonusHistogramPod::new();
        pod.top = top;
        pod.top_len = top_len;
        for &(k, t) in log {
            let slot = pod.slot_of(t);
            pod.counts[k as usize][slot] += 1;
        }
        pod.rounds_seen = log.len() as u32;
        pod
    }

    /// Freeze, gated on KL convergence between the two halves of the window.
    ///
    /// Per `k` row, `KL(P_first ‖ P_second)` over the smoothed bonus masses
    /// (fallback included); the gate statistic is the mean weighted by the
    /// full window's row counts. `≤ kl_eps` → [`FreezeVerdict::Converged`];
    /// otherwise `force` decides between [`FreezeVerdict::Forced`] (recorded
    /// `converged = 0`) and [`FreezeVerdict::NotConverged`]. Never mutates.
    #[must_use]
    pub fn try_freeze(&self, kl_eps: f32, force: bool) -> FreezeVerdict {
        if self.log.len() < 2 {
            return FreezeVerdict::Empty;
        }
        let (top, top_len) = self.top_vocab();
        let full = Self::fill(&self.log, top, top_len);
        let half = self.log.len() / 2;
        let a = Self::fill(&self.log[..half], top, top_len);
        let b = Self::fill(&self.log[half..], top, top_len);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (k, counts_row) in full.counts[..K_SLOTS].iter().enumerate() {
            let w = counts_row.iter().fold(0u64, |s, &v| s + u64::from(v));
            if w == 0 {
                continue;
            }
            let mut kl = 0.0f64;
            for slot in 0..ROW {
                let p = f64::from(a.mass_slot(k, slot));
                let q = f64::from(b.mass_slot(k, slot));
                if p > 0.0 && q > 0.0 {
                    kl += p * (p / q).ln();
                }
            }
            num += w as f64 * kl;
            den += w as f64;
        }
        let mean_kl = (num / den) as f32;
        let mut pod = full;
        match (mean_kl <= kl_eps, force) {
            (true, _) => {
                pod.converged = 1;
                FreezeVerdict::Converged(pod, mean_kl)
            }
            (false, true) => FreezeVerdict::Forced(pod, mean_kl),
            (false, false) => FreezeVerdict::NotConverged(mean_kl),
        }
    }
}

// ── Confidence recalibration (modelless degraded mode) ──────────────────────

/// Bins per position in [`ConfidenceRecalibrator`].
pub const RECAL_BINS: usize = 10;

/// Deterministic histogram-binning recalibration of drafter confidence:
/// per draft position `i` and raw-confidence decile, the empirical
/// probability that position `i` is accepted GIVEN positions `< i` were.
/// Counted, not trained (freeze/thaw-legal fixed arrays; Beta(1,1) prior).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceRecalibrator {
    /// Accepted counts `[pos][bin]`.
    pub hit: [[u32; RECAL_BINS]; GAMMA],
    /// Trials `[pos][bin]` (position reached).
    pub trials: [[u32; RECAL_BINS]; GAMMA],
}

impl Default for ConfidenceRecalibrator {
    fn default() -> Self {
        Self {
            hit: [[0; RECAL_BINS]; GAMMA],
            trials: [[0; RECAL_BINS]; GAMMA],
        }
    }
}

impl ConfidenceRecalibrator {
    #[inline]
    fn bin(c: f32) -> usize {
        ((c.clamp(0.0, 1.0) * RECAL_BINS as f32) as usize).min(RECAL_BINS - 1)
    }

    /// Fold one calibration round: raw confidences + realized accepted count.
    pub fn observe(&mut self, conf: &[f32], accepted: usize) {
        let n = conf.len().min(GAMMA);
        for (i, &c) in conf.iter().enumerate().take(n.min(accepted + 1)) {
            let b = Self::bin(c);
            self.trials[i][b] += 1;
            if i < accepted {
                self.hit[i][b] += 1;
            }
        }
    }

    /// Recalibrated confidences written into `out[..conf.len().min(GAMMA)]`.
    pub fn apply(&self, conf: &[f32], out: &mut [f32; GAMMA]) {
        for (i, (&c, o)) in conf.iter().zip(out.iter_mut()).enumerate() {
            let b = Self::bin(c);
            *o = (self.hit[i][b] as f32 + 1.0) / (self.trials[i][b] as f32 + 2.0);
        }
    }
}

// ── Outcome distribution + scores ───────────────────────────────────────────

/// Exactly-`k` accepted-count distribution from per-position confidences.
///
/// Writes `pk[0..K_SLOTS]` (zeros beyond `n = conf.len().min(GAMMA)`), returns
/// `n`. Prefix products via `cumprodsum_scalar` (sequential FMA fold).
pub fn accept_count_distribution(conf: &[f32], pk: &mut [f32; K_SLOTS]) -> usize {
    let n = conf.len().min(GAMMA);
    *pk = [0.0; K_SLOTS];
    if n == 0 {
        pk[0] = 1.0;
        return 0;
    }
    let zeros = [0.0f32; GAMMA];
    let mut pre = [0.0f32; GAMMA];
    cumprodsum_scalar(&conf[..n], &zeros[..n], 1.0, &mut pre[..n]);
    pk[0] = 1.0 - conf[0];
    for k in 1..n {
        pk[k] = pre[k - 1] * (1.0 - conf[k]);
    }
    pk[n] = pre[n - 1];
    n
}

/// Forecast `E[k] = Σ k · P(k)` (sequential fold).
#[must_use]
pub fn expected_accepted(pk: &[f32; K_SLOTS]) -> f32 {
    pk.iter()
        .enumerate()
        .fold(0.0f32, |acc, (k, &p)| acc + k as f32 * p)
}

/// Joint outcome scores `P(k, a) = P(k) · p̂(a | k)` over the full grid.
pub fn outcome_scores(pk: &[f32; K_SLOTS], hist: &BonusHistogramPod, out: &mut [f32; GRID]) {
    for k in 0..K_SLOTS {
        for slot in 0..ROW {
            out[cell(k, slot)] = pk[k] * hist.mass_slot(k, slot);
        }
    }
}

// ── Per-round bonus rows (drafter residual) ─────────────────────────────────

/// Per-round bonus candidates: `tokens[k][slot]` with mass `mass[k][slot]`
/// (`len[k]` live slots). The ROUND-CONDITIONED counterpart of the frozen
/// Pod's context-free `p̂(a | k)` — JetSpec++ draws the bonus from the
/// drafter's distribution at position `k`; under lossless speculative
/// sampling that is the RESIDUAL (the rejected draft token excluded,
/// renormalised) for `k < n`, and the full successor row at `k = n`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BonusRows {
    /// Candidate tokens per `k`.
    pub tokens: [[u32; A_SLOTS]; K_SLOTS],
    /// Candidate mass per `k` (each live row sums to ≤ 1).
    pub mass: [[f32; A_SLOTS]; K_SLOTS],
    /// Live slots per `k`.
    pub len: [u8; K_SLOTS],
}

impl Default for BonusRows {
    fn default() -> Self {
        Self {
            tokens: [[0; A_SLOTS]; K_SLOTS],
            mass: [[0.0; A_SLOTS]; K_SLOTS],
            len: [0; K_SLOTS],
        }
    }
}

impl BonusRows {
    /// Fill row `k` from a drafter successor row (`succ`, `probs`, sorted by
    /// mass desc), excluding `rejected` (the draft token the verifier refused
    /// at `k`, `None` at `k = n`) and renormalising over the kept mass.
    /// Probabilities are true conditionals (a truncated row may sum to < 1);
    /// the residual divides by `1 − p(rejected)`, leaving the unnamed tail
    /// mass to the fallback cell.
    pub fn set_residual_row(
        &mut self,
        k: usize,
        succ: &[u32],
        probs: &[f32],
        rejected: Option<u32>,
    ) {
        let mut p_rej = 0.0f32;
        let mut n = 0usize;
        for (&t, &p) in succ.iter().zip(probs) {
            if Some(t) == rejected {
                p_rej = p;
                continue;
            }
            if n < A_SLOTS {
                self.tokens[k][n] = t;
                self.mass[k][n] = p;
                n += 1;
            }
        }
        let denom = (1.0 - p_rej).max(f32::MIN_POSITIVE);
        for m in &mut self.mass[k][..n] {
            *m = (*m / denom).min(1.0);
        }
        self.len[k] = n as u8;
    }

    /// Katz-style backoff onto the frozen Pod: row `k` keeps its named mass
    /// (scaled to leave at least [`BACKOFF_ESCAPE`]), and the remaining tail
    /// is spread over the Pod's context-free `p̂(a | k)` for the frozen
    /// vocabulary. The merged candidates are re-ranked by mass (desc, row
    /// tokens before Pod tokens on ties — Issue 940) and the top `A_SLOTS`
    /// kept. Stack-only.
    #[must_use]
    pub fn with_backoff(&self, pod: &BonusHistogramPod) -> Self {
        let mut out = Self::default();
        let top_len = pod.top_len as usize;
        for k in 0..K_SLOTS {
            let len = self.len[k] as usize;
            let named = self.mass[k][..len].iter().fold(0.0f32, |a, &m| a + m);
            let scale = if named > 1.0 - BACKOFF_ESCAPE {
                (1.0 - BACKOFF_ESCAPE) / named
            } else {
                1.0
            };
            let tail = 1.0 - named * scale;
            let mut tok = [0u32; 2 * A_SLOTS];
            let mut mass = [0.0f32; 2 * A_SLOTS];
            let mut m = 0usize;
            for slot in 0..len {
                tok[m] = self.tokens[k][slot];
                mass[m] = self.mass[k][slot] * scale;
                m += 1;
            }
            for slot in 0..top_len {
                let add = tail * pod.mass_slot(k, slot);
                let t = pod.top[slot];
                match tok[..m].iter().position(|&x| x == t) {
                    Some(i) => mass[i] += add,
                    None => {
                        tok[m] = t;
                        mass[m] = add;
                        m += 1;
                    }
                }
            }
            // Selection sort, strict > keeps the first index on ties.
            let mut used = [false; 2 * A_SLOTS];
            let mut n = 0usize;
            while n < A_SLOTS {
                let mut best = usize::MAX;
                let mut best_m = 0.0f32;
                for i in 0..m {
                    if !used[i] && mass[i] > best_m {
                        best_m = mass[i];
                        best = i;
                    }
                }
                if best == usize::MAX {
                    break;
                }
                used[best] = true;
                out.tokens[k][n] = tok[best];
                out.mass[k][n] = mass[best];
                n += 1;
            }
            out.len[k] = n as u8;
        }
        out
    }

    /// Slot of `token` in row `k`, else the fallback slot.
    #[inline]
    #[must_use]
    pub fn slot_of(&self, k: usize, token: u32) -> usize {
        let len = self.len[k] as usize;
        self.tokens[k][..len]
            .iter()
            .position(|&t| t == token)
            .unwrap_or(A_SLOTS)
    }
}

/// Round-conditioned joint scores `P(k) · p̂_round(a | k)`; the fallback cell
/// carries the residual mass not named by the row.
pub fn outcome_scores_rows(pk: &[f32; K_SLOTS], rows: &BonusRows, out: &mut [f32; GRID]) {
    *out = [0.0; GRID];
    for k in 0..K_SLOTS {
        let len = rows.len[k] as usize;
        let named = rows.mass[k][..len].iter().fold(0.0f32, |a, &m| a + m);
        for slot in 0..len {
            out[cell(k, slot)] = pk[k] * rows.mass[k][slot];
        }
        out[cell(k, A_SLOTS)] = pk[k] * (1.0 - named).max(0.0);
    }
}

// ── Selection ───────────────────────────────────────────────────────────────

/// Top-`b` selectable cells by score (desc, first-index tie), at most
/// `floor` excluded (cells with score `< floor` are skipped). Writes cell ids
/// into `out`, returns the count. `O(b · GRID)`, alloc-free.
fn select_core(scores: &[f32; GRID], b: usize, floor: f32, out: &mut [u16; GRID]) -> usize {
    let mut taken = [false; GRID];
    let mut n = 0;
    while n < b.min(GRID) {
        let mut best = usize::MAX;
        let mut best_s = f32::NEG_INFINITY;
        for (c, &s) in scores.iter().enumerate() {
            if taken[c] || is_fallback(c) || s <= 0.0 || s < floor {
                continue;
            }
            if s > best_s {
                best_s = s;
                best = c;
            }
        }
        if best == usize::MAX {
            break;
        }
        taken[best] = true;
        out[n] = best as u16;
        n += 1;
    }
    n
}

/// Per-round posterior selection: top-`b` cells by score (Eq 7).
pub fn select_top_b(scores: &[f32; GRID], b: usize, out: &mut [u16; GRID]) -> usize {
    select_core(scores, b, f32::NEG_INFINITY, out)
}

/// Conformal selection: top-`b_max` cells restricted to `score ≥ λ`.
pub fn select_conformal(
    scores: &[f32; GRID],
    lambda: f32,
    b_max: usize,
    out: &mut [u16; GRID],
) -> usize {
    select_core(scores, b_max, lambda, out)
}

/// Fixed allocation: top-`b` cells by calibration-window realized frequency
/// (`freq[cell]`). The strongest fixed policy the window can legally pick.
pub fn static_top_b(freq: &[u32; GRID], b: usize, out: &mut [u16; GRID]) -> usize {
    let mut scores = [0.0f32; GRID];
    for (s, &f) in scores.iter_mut().zip(freq) {
        *s = f as f32;
    }
    select_top_b(&scores, b, out)
}

/// Fixed allocation, SSD shape: geometric k-prior `r^k (1 − r)` (the last
/// row takes the tail `r^γ`) times the per-k bonus mass.
pub fn geometric_ladder(
    hist: &BonusHistogramPod,
    r: f32,
    b: usize,
    out: &mut [u16; GRID],
) -> usize {
    let mut pk = [0.0f32; K_SLOTS];
    let mut rk = 1.0f32;
    for (k, p) in pk.iter_mut().enumerate() {
        *p = if k == GAMMA { rk } else { rk * (1.0 - r) };
        rk *= r;
    }
    let mut scores = [0.0f32; GRID];
    outcome_scores(&pk, hist, &mut scores);
    select_top_b(&scores, b, out)
}

/// Split-conformal threshold: the `⌊(n+1)α⌋`-th smallest realized score
/// (1-based; `0.0` when that index is 0), so `P(score ≥ λ) ≥ 1 − α`. Sorts
/// `realized` in place (calibration posture).
pub fn conformal_threshold(realized: &mut [f32], alpha: f32) -> f32 {
    let n = realized.len();
    let idx = (((n + 1) as f32) * alpha).floor() as usize;
    if idx == 0 || n == 0 {
        return 0.0;
    }
    realized.sort_by(f32::total_cmp);
    realized[idx.min(n) - 1]
}

/// `true` when `cell` is among `sel`.
#[inline]
#[must_use]
pub fn hit(sel: &[u16], cell: usize) -> bool {
    sel.iter().any(|&c| c as usize == cell)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_distribution_sums_to_one() {
        let mut pk = [0.0; K_SLOTS];
        let n = accept_count_distribution(&[0.9, 0.5, 0.25], &mut pk);
        assert_eq!(n, 3);
        let s: f32 = pk.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "{s}");
        assert!((pk[0] - 0.1).abs() < 1e-6);
        assert!((pk[3] - 0.9 * 0.5 * 0.25).abs() < 1e-6);
        assert!((expected_accepted(&pk) - (0.9 + 0.45 + 0.1125)).abs() < 1e-5);
    }

    #[test]
    fn pod_roundtrip_and_digest() {
        let mut b = HistogramBuilder::new();
        for i in 0..200u32 {
            b.observe((i % 4) as usize, i % 3);
        }
        let FreezeVerdict::Converged(pod, kl) = b.try_freeze(0.05, false) else {
            panic!("stationary log must converge");
        };
        assert!(kl < 0.05);
        let bytes = pod.commit_bytes();
        let back = BonusHistogramPod::from_commit_bytes(&bytes).unwrap();
        assert_eq!(back, pod);
        let d = pod.blake3_digest();
        assert!(back.verify_against(&d));
        let mut drift = back;
        drift.counts[0][0] += 1;
        assert!(!drift.verify_against(&d));
        assert!(BonusHistogramPod::from_commit_bytes(&bytes[1..]).is_none());
    }

    #[test]
    fn nonstationary_log_refuses_unless_forced() {
        let mut b = HistogramBuilder::new();
        for _ in 0..100 {
            b.observe(1, 7);
        }
        for _ in 0..100 {
            b.observe(1, 9);
        }
        assert!(matches!(
            b.try_freeze(0.05, false),
            FreezeVerdict::NotConverged(_)
        ));
        let FreezeVerdict::Forced(pod, _) = b.try_freeze(0.05, true) else {
            panic!()
        };
        assert_eq!(pod.converged, 0);
    }

    #[test]
    fn ties_resolve_first_index_and_fallback_never_selected() {
        let mut s = [0.0f32; GRID];
        s[cell(0, A_SLOTS)] = 9.0; // fallback — unselectable
        s[cell(2, 1)] = 0.5;
        s[cell(1, 3)] = 0.5;
        let mut out = [0u16; GRID];
        let n = select_top_b(&s, 4, &mut out);
        assert_eq!(&out[..n], &[cell(1, 3) as u16, cell(2, 1) as u16]);
        let n = select_conformal(&s, 0.6, 4, &mut out);
        assert_eq!(n, 0);
    }

    #[test]
    fn residual_row_excludes_rejected_and_backoff_keeps_escape() {
        let mut rows = BonusRows::default();
        rows.set_residual_row(0, &[5, 6, 7], &[0.5, 0.3, 0.2], Some(5));
        assert_eq!(rows.len[0], 2);
        assert_eq!(rows.slot_of(0, 5), A_SLOTS);
        assert!((rows.mass[0][0] - 0.6).abs() < 1e-6 && (rows.mass[0][1] - 0.4).abs() < 1e-6);
        let mut b = HistogramBuilder::new();
        for i in 0..64u32 {
            b.observe(0, if i % 2 == 0 { 6 } else { 9 });
        }
        let FreezeVerdict::Converged(pod, _) = b.try_freeze(0.05, false) else {
            panic!()
        };
        let bo = rows.with_backoff(&pod);
        // Named mass (1.0) is scaled to leave the escape for the Pod's tokens.
        assert_eq!(
            bo.tokens[0][0], 6,
            "row + pod mass merge on the shared token"
        );
        assert_ne!(bo.slot_of(0, 9), A_SLOTS, "pod-only token joins the row");
        let named: f32 = bo.mass[0][..bo.len[0] as usize].iter().sum();
        assert!(named <= 1.0 + 1e-6 && named > 1.0 - BACKOFF_ESCAPE);
        let mut pk = [0.0; K_SLOTS];
        pk[0] = 1.0;
        let mut s = [0.0; GRID];
        outcome_scores_rows(&pk, &bo, &mut s);
        let total: f32 = s.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
    }

    #[test]
    fn conformal_threshold_order_statistic() {
        let mut r: Vec<f32> = (1..=19).map(|i| i as f32).collect();
        // (19+1)*0.1 = 2 → 2nd smallest.
        assert_eq!(conformal_threshold(&mut r, 0.1), 2.0);
        let mut few = [5.0f32; 3];
        assert_eq!(conformal_threshold(&mut few, 0.1), 0.0);
    }
}
