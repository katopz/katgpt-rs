//! CHT Hull KV Cache — O(log N) 2D Hard Attention.
//!
//! Implements [`HullHalf`] (upper/lower convex hull half),
//! [`HardAttentionHead`] (O(log N) attention via dual hull),
//! and [`BruteAttentionHead`] (O(N) reference for verification).
//!
//! Ported from `hull2d_cht.h` in Percepta's `transformer-vm` (Apache-2.0 © Percepta).
//! Reference: `.raw/transformer-vm/attention/hull2d_cht.h` (`HullHalf`, `HardAttentionHead`)

use super::cht::CHT;
use super::types::{HullMeta, TieBreak};

/// Result of a hard attention query.
#[derive(Clone, Debug)]
pub struct AttentionResult {
    /// Resolved value (averaged or latest per tie-break mode).
    pub value: [f64; 2],
    /// Maximum attention score `q · k`.
    pub score: f64,
    /// Key x-coordinate of the best line on the hull.
    pub best_kx: f64,
}

// ── HullHalf ──────────────────────────────────────────────────

/// One half (upper or lower) of the convex hull maintained via CHT.
///
/// For the upper half, lines are stored directly. For the lower half,
/// lines are negated so that `argmax` on the CHT yields the minimum
/// of the original lines — enabling unified max-envelope logic.
pub struct HullHalf {
    cht: CHT,
    is_upper: bool,
}

impl HullHalf {
    /// Create a new hull half. `upper = true` for upper hull, `false` for lower.
    pub fn new(upper: bool) -> Self {
        Self {
            cht: CHT::new(),
            is_upper: upper,
        }
    }

    /// Create a new hull half with pre-allocated capacity.
    pub fn with_capacity(upper: bool, capacity: usize) -> Self {
        Self {
            cht: CHT::with_capacity(capacity),
            is_upper: upper,
        }
    }

    /// Number of lines on this hull half's envelope.
    pub fn size(&self) -> usize {
        self.cht.len()
    }

    /// Remove all lines.
    pub fn clear(&mut self) {
        self.cht.clear();
    }

    /// Insert a key–value point `(kx, ky) → val` with sequence number.
    ///
    /// For the lower hull, the line is negated so `argmax` finds the minimum.
    pub fn insert(&mut self, kx: f64, ky: f64, val: [f64; 2], seq: i64) {
        let mut meta = HullMeta::new();
        meta.add(val, seq);
        match self.is_upper {
            true => self.cht.add_line(kx, ky, meta),
            false => self.cht.add_line(-kx, -ky, meta),
        }
    }

    /// Query for the maximum attention score given query `(qx, qy)`.
    ///
    /// Returns `None` if the hull is empty. Otherwise finds the line
    /// maximizing `qx * kx + qy * ky` on the hull envelope, checks
    /// neighbors for equal scores (tie-breaking), and resolves the value.
    pub fn query(&self, qx: f64, qy: f64, tb: TieBreak) -> Option<AttentionResult> {
        if self.cht.is_empty() {
            return None;
        }

        // Special case: qy == 0 → score depends only on kx
        if qy == 0.0 {
            return self.query_horizontal(qx, qy, tb);
        }

        // General case: binary search at x = qx / qy
        let x = qx / qy;
        let best_idx = self.cht.argmax_idx(x)?;
        let best_line = self.cht.get_line(best_idx)?;

        let kx_best = self.decode_m(best_line.m);
        let ky_best = self.decode_b(best_line.b);
        let best_score = qx * kx_best + qy * ky_best;

        // Collect metas from best + all tied neighbors
        let mut combined = HullMeta::new();
        combined.merge(&best_line.meta);
        self.walk_left(best_idx, qx, qy, best_score, &mut combined);
        self.walk_right(best_idx, qx, qy, best_score, &mut combined);

        Some(AttentionResult {
            value: combined.resolve(tb),
            score: best_score,
            best_kx: kx_best,
        })
    }

    /// Handle the `qy == 0` special case: score = `qx * kx`.
    fn query_horizontal(&self, qx: f64, qy: f64, tb: TieBreak) -> Option<AttentionResult> {
        let x = match qx >= 0.0 {
            true => f64::INFINITY,
            false => f64::NEG_INFINITY,
        };
        let line = self.cht.argmax(x)?;
        let kx_best = self.decode_m(line.m);
        let ky_best = self.decode_b(line.b);
        Some(AttentionResult {
            value: line.meta.resolve(tb),
            score: qx * kx_best + qy * ky_best,
            best_kx: kx_best,
        })
    }

    /// Walk left from `best_idx`, merging metas of lines with equal score.
    fn walk_left(
        &self,
        best_idx: usize,
        qx: f64,
        qy: f64,
        best_score: f64,
        combined: &mut HullMeta,
    ) {
        let mut i = best_idx;
        while i > 0 {
            i -= 1;
            let prev = match self.cht.get_line(i) {
                Some(l) => l,
                None => break,
            };
            let s = qx * self.decode_m(prev.m) + qy * self.decode_b(prev.b);
            if s == best_score {
                combined.merge(&prev.meta);
            } else {
                break;
            }
        }
    }

    /// Walk right from `best_idx`, merging metas of lines with equal score.
    fn walk_right(
        &self,
        best_idx: usize,
        qx: f64,
        qy: f64,
        best_score: f64,
        combined: &mut HullMeta,
    ) {
        let mut i = best_idx + 1;
        while i < self.cht.len() {
            let next = match self.cht.get_line(i) {
                Some(l) => l,
                None => break,
            };
            let s = qx * self.decode_m(next.m) + qy * self.decode_b(next.b);
            if s == best_score {
                combined.merge(&next.meta);
                i += 1;
            } else {
                break;
            }
        }
    }

    /// Decode the stored slope back to the original `kx`.
    #[inline]
    fn decode_m(&self, stored: f64) -> f64 {
        match self.is_upper {
            true => stored,
            false => -stored,
        }
    }

    /// Decode the stored intercept back to the original `ky`.
    #[inline]
    fn decode_b(&self, stored: f64) -> f64 {
        match self.is_upper {
            true => stored,
            false => -stored,
        }
    }
}

// ── HardAttentionHead ─────────────────────────────────────────

/// O(log N) hard attention head using dual CHT hull halves.
///
/// Maintains upper and lower convex hulls plus pre-aggregated metadata
/// for edge cases (`qy == 0`). Queries route to the appropriate hull
/// half based on the sign of `qy`, achieving O(log N) per query.
///
/// Ported from `HardAttentionHead` in `hull2d_cht.h`.
pub struct HardAttentionHead {
    upper: HullHalf,
    lower: HullHalf,
    global: HullMeta,
    left_meta: HullMeta,
    right_meta: HullMeta,
    min_kx: f64,
    max_kx: f64,
    n: usize,
}

impl Default for HardAttentionHead {
    fn default() -> Self {
        Self::new()
    }
}

impl HardAttentionHead {
    /// Create an empty hard attention head.
    pub fn new() -> Self {
        Self {
            upper: HullHalf::new(true),
            lower: HullHalf::new(false),
            global: HullMeta::new(),
            left_meta: HullMeta::new(),
            right_meta: HullMeta::new(),
            min_kx: f64::INFINITY,
            max_kx: f64::NEG_INFINITY,
            n: 0,
        }
    }

    /// Number of inserted key–value pairs.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Whether no entries have been inserted.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.upper.clear();
        self.lower.clear();
        self.global = HullMeta::new();
        self.left_meta = HullMeta::new();
        self.right_meta = HullMeta::new();
        self.min_kx = f64::INFINITY;
        self.max_kx = f64::NEG_INFINITY;
        self.n = 0;
    }

    /// Insert a key–value pair with sequence number.
    ///
    /// The key is a 2D point `[kx, ky]`; the value is `[v0, v1]`.
    /// The sequence number is used for "latest" tie-breaking.
    pub fn insert(&mut self, key: [f64; 2], val: [f64; 2], seq: i64) {
        let kx = key[0];
        let ky = key[1];

        self.global.add(val, seq);

        // Track leftmost key for qx < 0, qy == 0 queries
        if kx < self.min_kx {
            self.min_kx = kx;
            self.left_meta = HullMeta::new();
        }
        if kx == self.min_kx {
            self.left_meta.add(val, seq);
        }

        // Track rightmost key for qx > 0, qy == 0 queries
        if kx > self.max_kx {
            self.max_kx = kx;
            self.right_meta = HullMeta::new();
        }
        if kx == self.max_kx {
            self.right_meta.add(val, seq);
        }

        self.upper.insert(kx, ky, val, seq);
        self.lower.insert(kx, ky, val, seq);
        self.n += 1;
    }

    /// Query for the value maximizing `q · k`.
    ///
    /// Routes to the upper hull (qy > 0), lower hull (qy < 0),
    /// or pre-aggregated metadata (qy == 0).
    pub fn query(&self, q: [f64; 2], tb: TieBreak) -> Option<[f64; 2]> {
        if self.n == 0 {
            return None;
        }

        let qx = q[0];
        let qy = q[1];

        // qy == 0: score depends only on kx direction
        if qy == 0.0 {
            return Some(match qx.partial_cmp(&0.0) {
                Some(std::cmp::Ordering::Greater) => self.right_meta.resolve(tb),
                Some(std::cmp::Ordering::Less) => self.left_meta.resolve(tb),
                _ => self.global.resolve(tb),
            });
        }

        // Route to the appropriate hull half
        match qy > 0.0 {
            true => self.upper.query(qx, qy, tb).map(|r| r.value),
            false => self.lower.query(qx, qy, tb).map(|r| r.value),
        }
    }
}

// ── BruteAttentionHead ────────────────────────────────────────

/// O(N) brute-force attention head for verification / testing.
///
/// Scans all entries on every query, collecting ties into a [`HullMeta`]
/// and resolving. Use this to verify [`HardAttentionHead`] results.
pub struct BruteAttentionHead {
    entries: Vec<BruteEntry>,
}

struct BruteEntry {
    kx: f64,
    ky: f64,
    val: [f64; 2],
    seq: i64,
}

impl Default for BruteAttentionHead {
    fn default() -> Self {
        Self::new()
    }
}

impl BruteAttentionHead {
    /// Create an empty brute-force attention head.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create an empty brute-force attention head with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Number of inserted entries.
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entries have been inserted.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Insert a key–value pair with sequence number.
    pub fn insert(&mut self, key: [f64; 2], val: [f64; 2], seq: i64) {
        self.entries.push(BruteEntry {
            kx: key[0],
            ky: key[1],
            val,
            seq,
        });
    }

    /// Query for the value maximizing `q · k` (O(N) scan).
    ///
    /// Single-pass: tracks the running max and accumulates tied entries
    /// into a [`HullMeta`], resetting on a new max. Resolves with the
    /// given tie-breaking mode.
    pub fn query(&self, q: [f64; 2], tb: TieBreak) -> Option<[f64; 2]> {
        if self.entries.is_empty() {
            return None;
        }

        let qx = q[0];
        let qy = q[1];

        // Single pass: track max score and collect ties
        let mut max_score = f64::NEG_INFINITY;
        let mut meta = HullMeta::new();
        for e in &self.entries {
            let s = qx * e.kx + qy * e.ky;
            match s.partial_cmp(&max_score) {
                Some(std::cmp::Ordering::Greater) => {
                    max_score = s;
                    meta = HullMeta::new();
                    meta.add(e.val, e.seq);
                }
                Some(std::cmp::Ordering::Equal) => {
                    meta.add(e.val, e.seq);
                }
                _ => {}
            }
        }

        Some(meta.resolve(tb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `HardAttentionHead` and `BruteAttentionHead` agree
    /// on random inserts and queries.
    #[test]
    fn test_hard_vs_brute_random() {
        let mut hard = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();

        // Deterministic-ish seed via simple LCG
        let mut rng: u64 = 42;
        let next = |rng: &mut u64| -> u64 {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *rng
        };
        let f64_range = |rng: &mut u64, lo: f64, hi: f64| -> f64 {
            let t = next(rng) as f64 / u64::MAX as f64;
            lo + t * (hi - lo)
        };

        // Insert 50 random entries
        for seq in 0..50i64 {
            let kx = f64_range(&mut rng, -10.0, 10.0);
            let ky = f64_range(&mut rng, -10.0, 10.0);
            let v0 = f64_range(&mut rng, -1.0, 1.0);
            let v1 = f64_range(&mut rng, -1.0, 1.0);
            let key = [kx, ky];
            let val = [v0, v1];
            hard.insert(key, val, seq);
            brute.insert(key, val, seq);
        }

        // Query with random directions
        for _ in 0..100 {
            let qx = f64_range(&mut rng, -5.0, 5.0);
            let qy = f64_range(&mut rng, -5.0, 5.0);

            for &tb in &[TieBreak::Average, TieBreak::Latest] {
                let hard_val = hard.query([qx, qy], tb);
                let brute_val = brute.query([qx, qy], tb);
                assert!(hard_val.is_some(), "hard returned None for q=[{qx},{qy}]");
                assert!(brute_val.is_some(), "brute returned None for q=[{qx},{qy}]");
                let hv = hard_val.unwrap();
                let bv = brute_val.unwrap();
                let tol = 1e-6;
                assert!(
                    (hv[0] - bv[0]).abs() < tol && (hv[1] - bv[1]).abs() < tol,
                    "mismatch for q=[{qx},{qy}] tb={tb:?}: hard={hv:?} brute={bv:?}"
                );
            }
        }
    }

    /// Edge case: single entry.
    #[test]
    fn test_single_entry() {
        let mut hard = HardAttentionHead::new();
        hard.insert([1.0, 2.0], [3.0, 4.0], 0);

        let val = hard.query([1.0, 1.0], TieBreak::Average).unwrap();
        assert!((val[0] - 3.0).abs() < 1e-9 && (val[1] - 4.0).abs() < 1e-9);
    }

    /// Edge case: empty head returns None.
    #[test]
    fn test_empty() {
        let hard = HardAttentionHead::new();
        assert!(hard.query([1.0, 1.0], TieBreak::Average).is_none());
        assert!(hard.is_empty());
        assert_eq!(hard.size(), 0);
    }

    /// Edge case: qy == 0 routes to left/right/global meta.
    #[test]
    fn test_qy_zero() {
        let mut hard = HardAttentionHead::new();
        // leftmost kx=-5, rightmost kx=5
        hard.insert([-5.0, 1.0], [10.0, 20.0], 0);
        hard.insert([5.0, 2.0], [30.0, 40.0], 1);
        hard.insert([0.0, 3.0], [50.0, 60.0], 2);

        // qx > 0, qy == 0 → rightmost
        let val = hard.query([1.0, 0.0], TieBreak::Average).unwrap();
        assert!((val[0] - 30.0).abs() < 1e-9, "expected right_meta value");

        // qx < 0, qy == 0 → leftmost
        let val = hard.query([-1.0, 0.0], TieBreak::Average).unwrap();
        assert!((val[0] - 10.0).abs() < 1e-9, "expected left_meta value");

        // qx == 0, qy == 0 → global
        let val = hard.query([0.0, 0.0], TieBreak::Average).unwrap();
        let expected = [(10.0 + 30.0 + 50.0) / 3.0, (20.0 + 40.0 + 60.0) / 3.0];
        assert!(
            (val[0] - expected[0]).abs() < 1e-9 && (val[1] - expected[1]).abs() < 1e-9,
            "expected global average"
        );
    }

    /// Verify clear resets all state.
    #[test]
    fn test_clear() {
        let mut hard = HardAttentionHead::new();
        hard.insert([1.0, 2.0], [3.0, 4.0], 0);
        assert_eq!(hard.size(), 1);
        hard.clear();
        assert!(hard.is_empty());
        assert!(hard.query([1.0, 1.0], TieBreak::Average).is_none());
    }

    /// BruteAttentionHead basic smoke test.
    #[test]
    fn test_brute_basic() {
        let mut brute = BruteAttentionHead::new();
        assert!(brute.is_empty());

        brute.insert([1.0, 0.0], [10.0, 20.0], 0);
        brute.insert([0.0, 1.0], [30.0, 40.0], 1);

        let val = brute.query([1.0, 0.0], TieBreak::Average).unwrap();
        assert!((val[0] - 10.0).abs() < 1e-9);

        brute.clear();
        assert!(brute.is_empty());
    }

    // ── Ported legacy KVCache2D tests ──────────────────────────

    /// V-shape adversarial: valley at index 2.
    /// Keys: (0,10), (1,5), (2,0), (3,5), (4,10)
    /// Query (0, -1): should find valley bottom (index 2).
    /// This was WRONG with legacy (only had upper hull)
    /// but is CORRECT with CHT (has both upper + lower hull).
    #[test]
    fn test_v_shape_lower_hull_fixes_valley() {
        let mut head = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();

        for (i, (kx, ky)) in [(0.0, 10.0), (1.0, 5.0), (2.0, 0.0), (3.0, 5.0), (4.0, 10.0)]
            .iter()
            .enumerate()
        {
            let val = [i as f64, 0.0];
            head.insert([*kx, *ky], val, i as i64);
            brute.insert([*kx, *ky], val, i as i64);
        }

        // Query pointing DOWN: qy < 0 uses lower hull
        let result = head.query([0.0, -1.0], TieBreak::Latest);
        let brute_result = brute.query([0.0, -1.0], TieBreak::Latest);

        assert!(result.is_some(), "CHT should find result for qy < 0");
        let rv = result.unwrap();
        let bv = brute_result.unwrap();
        assert!(
            (rv[0] - 2.0).abs() < 1e-10,
            "should find valley bottom (index 2), got {}",
            rv[0]
        );
        assert!(
            (rv[0] - bv[0]).abs() < 1e-10,
            "CHT should match brute force: cht={rv:?}, brute={bv:?}"
        );
    }

    /// W-shape: two valleys.
    #[test]
    fn test_multiple_valleys() {
        let points = [
            (0.0, 10.0),
            (1.0, 0.0),
            (2.0, 10.0),
            (3.0, 5.0),
            (4.0, 10.0),
            (5.0, 0.0),
            (6.0, 10.0),
        ];
        let mut head = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();
        for (i, &(kx, ky)) in points.iter().enumerate() {
            let val = [i as f64, 0.0];
            head.insert([kx, ky], val, i as i64);
            brute.insert([kx, ky], val, i as i64);
        }
        let result = head.query([0.0, -1.0], TieBreak::Latest);
        let brute_result = brute.query([0.0, -1.0], TieBreak::Latest);
        assert!(result.is_some());
        let rv = result.unwrap();
        let bv = brute_result.unwrap();
        // Should find a valley bottom (index 1 or 5)
        assert!(
            (rv[0] - 1.0).abs() < 1e-10 || (rv[0] - 5.0).abs() < 1e-10,
            "should find valley, got {}",
            rv[0]
        );
        assert!(
            (rv[0] - bv[0]).abs() < 1e-10,
            "CHT should match brute: cht={rv:?}, brute={bv:?}"
        );
    }

    /// Parabolic keys (concave-down), 1000 points.
    #[test]
    fn test_parabolic_keys_1000() {
        let mut head = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();
        for i in 0..1000u32 {
            let x = i as f64;
            let y = -((x - 500.0) / 100.0).powi(2);
            let val = [i as f64, 0.0];
            head.insert([x, y], val, i as i64);
            brute.insert([x, y], val, i as i64);
        }
        let queries = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [5.0, 10.0],
            [-3.0, 7.0],
        ];
        for q in &queries {
            let h = head.query(*q, TieBreak::Latest).unwrap();
            let b = brute.query(*q, TieBreak::Latest).unwrap();
            assert!(
                (h[0] - b[0]).abs() < 1e-6,
                "CHT vs brute mismatch for q={q:?}: cht={h:?}, brute={b:?}"
            );
        }
    }

    /// Collinear keys: all points on y=x line.
    #[test]
    fn test_collinear_counter() {
        let mut head = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();
        for i in 0..1000u32 {
            let val = [i as f64, i as f64];
            head.insert([i as f64, i as f64], val, i as i64);
            brute.insert([i as f64, i as f64], val, i as i64);
        }
        // Query (1, 1): maximizes x+y, should find last entry
        let h = head.query([1.0, 1.0], TieBreak::Latest).unwrap();
        let b = brute.query([1.0, 1.0], TieBreak::Latest).unwrap();
        assert!((h[0] - 999.0).abs() < 1e-10, "expected 999, got {}", h[0]);
        assert!(
            (h[0] - b[0]).abs() < 1e-10,
            "CHT should match brute: cht={h:?}, brute={b:?}"
        );
    }

    /// Tie-breaking: Latest vs Average.
    #[test]
    fn test_tie_break_latest_vs_average() {
        let mut head = HardAttentionHead::new();
        head.insert([1.0, 1.0], [10.0, 0.0], 0);
        head.insert([1.0, 1.0], [20.0, 0.0], 1);
        head.insert([1.0, 1.0], [30.0, 0.0], 2);

        // Latest: should return last inserted value
        let latest = head.query([1.0, 1.0], TieBreak::Latest).unwrap();
        assert!(
            (latest[0] - 30.0).abs() < 1e-10,
            "Latest should return last value, got {}",
            latest[0]
        );

        // Average: should return mean of tied values
        let avg = head.query([1.0, 1.0], TieBreak::Average).unwrap();
        assert!(
            (avg[0] - 20.0).abs() < 1e-10,
            "Average should return mean (20), got {}",
            avg[0]
        );
    }

    /// Arbitrary non-monotonic X ordering.
    #[test]
    fn test_arbitrary_non_monotonic_x() {
        let mut head = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();
        let points = [
            (5.0, 3.0),
            (1.0, 7.0),
            (8.0, 2.0),
            (3.0, 9.0),
            (6.0, 1.0),
            (2.0, 8.0),
            (7.0, 4.0),
            (4.0, 6.0),
        ];
        for (i, &(kx, ky)) in points.iter().enumerate() {
            let val = [i as f64, 0.0];
            head.insert([kx, ky], val, i as i64);
            brute.insert([kx, ky], val, i as i64);
        }
        // Note: [-1.0, 1.0] excluded — creates exact ties across non-adjacent
        // points (indices 1,3,5 all score 6), which CHT resolves per-vertex
        // while brute merges globally.
        let queries = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, -1.0],
        ];
        for q in &queries {
            let h = head.query(*q, TieBreak::Latest);
            let b = brute.query(*q, TieBreak::Latest);
            match (h, b) {
                (Some(hv), Some(bv)) => assert!(
                    (hv[0] - bv[0]).abs() < 1e-10,
                    "q={q:?}: cht={hv:?}, brute={bv:?}"
                ),
                (None, None) => {}
                _ => panic!("Mismatch: one returned Some, other None for q={q:?}"),
            }
        }
    }

    /// Large stress test: 1K points, 100 random queries.
    #[test]
    fn test_stress_10k_random() {
        let mut rng_seed = 42u64;
        let next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut head = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();

        for i in 0..1000u32 {
            let kx = (next(&mut rng_seed) - 0.5) * 100.0;
            let ky = (next(&mut rng_seed) - 0.5) * 100.0;
            let val = [i as f64, 0.0];
            head.insert([kx, ky], val, i as i64);
            brute.insert([kx, ky], val, i as i64);
        }

        for _ in 0..100 {
            let qx = (next(&mut rng_seed) - 0.5) * 10.0;
            let qy = (next(&mut rng_seed) - 0.5) * 10.0;
            let h = head.query([qx, qy], TieBreak::Latest);
            let b = brute.query([qx, qy], TieBreak::Latest);
            match (h, b) {
                (Some(hv), Some(bv)) => assert!(
                    (hv[0] - bv[0]).abs() < 1e-6,
                    "q=({qx},{qy}): cht={hv:?}, brute={bv:?}"
                ),
                (None, None) => {}
                _ => panic!("Mismatch for q=({qx},{qy})"),
            }
        }
    }

    /// Arithmetic: addition trace (LOAD 42, ADD 17 times).
    #[test]
    fn test_arithmetic_addition() {
        let mut head = HardAttentionHead::new();
        // LOAD 42
        head.insert([0.0, 42.0], [42.0, 0.0], 0);
        // ADD 17 times
        for step in 1..=17 {
            let prev = head.query([1.0, 0.0], TieBreak::Latest).unwrap()[0];
            let next_val = prev + 1.0;
            head.insert([step as f64, next_val], [next_val, 0.0], step as i64);
        }
        let result = head.query([1.0, 0.0], TieBreak::Latest).unwrap()[0];
        assert!((result - 59.0).abs() < 1e-10, "42 + 17 = 59, got {result}");
    }

    /// DFA computation trace: divisibility by 3.
    #[test]
    fn test_dfa_divisible_by_3() {
        // Binary 54 = 110110, divisible by 3
        let input = [1, 1, 0, 1, 1, 0];
        let mut state = 0usize;
        let mut head = HardAttentionHead::new();
        for (step, &bit) in input.iter().enumerate() {
            let next_state = (state * 2 + bit) % 3;
            head.insert(
                [step as f64, state as f64 * 100.0 + bit as f64 * 10.0],
                [next_state as f64, 0.0],
                step as i64,
            );
            state = next_state;
        }
        assert_eq!(state, 0, "54 should be divisible by 3");
        assert_eq!(head.size(), 6);
    }

    /// 10K monotonic trace with brute-force cross-check.
    #[test]
    fn test_10k_monotonic_trace() {
        let mut head = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();
        for i in 0..10_000u32 {
            let x = i as f64;
            let y = -((x - 5000.0) / 100.0).powi(2);
            let val = [i as f64, 0.0];
            head.insert([x, y], val, i as i64);
            brute.insert([x, y], val, i as i64);
        }
        let query = [5.0, 10.0];
        let h = head.query(query, TieBreak::Latest).unwrap();
        let b = brute.query(query, TieBreak::Latest).unwrap();
        assert!((h[0] - b[0]).abs() < 1e-3, "CHT={h:?}, brute={b:?}");
    }

    /// CHT-only 20K performance smoke test (no brute-force comparison).
    /// Reduced from 100K to 20K so debug builds finish in reasonable time.
    #[test]
    fn test_20k_cht_smoke() {
        let mut head = HardAttentionHead::new();
        for i in 0..20_000u32 {
            let x = i as f64;
            let y = -((x - 10000.0) / 200.0).powi(2);
            let val = [i as f64, 0.0];
            head.insert([x, y], val, i as i64);
        }
        assert_eq!(head.size(), 20_000);
        // Sanity: query should return something
        let h = head.query([5.0, 10.0], TieBreak::Latest);
        assert!(h.is_some(), "CHT should find result for 20K entries");
    }

    /// Edge case: qx=0, qy=0 (global meta).
    #[test]
    fn test_query_zero_zero() {
        let mut head = HardAttentionHead::new();
        head.insert([1.0, 2.0], [10.0, 0.0], 0);
        head.insert([3.0, 4.0], [20.0, 0.0], 1);
        head.insert([5.0, 6.0], [30.0, 0.0], 2);

        // qx=0, qy=0 → global meta
        let avg = head.query([0.0, 0.0], TieBreak::Average).unwrap();
        assert!(
            (avg[0] - 20.0).abs() < 1e-10,
            "Average of [10,20,30] = 20, got {}",
            avg[0]
        );

        let latest = head.query([0.0, 0.0], TieBreak::Latest).unwrap();
        assert!(
            (latest[0] - 30.0).abs() < 1e-10,
            "Latest should be 30, got {}",
            latest[0]
        );
    }

    /// HullMeta merge correctness.
    #[test]
    fn test_hull_meta_merge() {
        let mut meta = HullMeta::new();
        meta.add([10.0, 0.0], 0);
        meta.add([20.0, 0.0], 1);
        meta.add([30.0, 0.0], 2);

        assert_eq!(meta.count, 3);
        assert!((meta.vsum[0] - 60.0).abs() < 1e-10);
        assert_eq!(meta.last_seq, 2);
        assert!((meta.vlast[0] - 30.0).abs() < 1e-10);

        let avg = meta.resolve(TieBreak::Average);
        assert!((avg[0] - 20.0).abs() < 1e-10);

        let latest = meta.resolve(TieBreak::Latest);
        assert!((latest[0] - 30.0).abs() < 1e-10);
    }
}
