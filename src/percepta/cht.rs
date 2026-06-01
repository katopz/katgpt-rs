//! Dynamic Convex Hull Trick (CHT) / LineContainer.
//!
//! Maintains the upper envelope of lines y = m*x + b, supporting:
//! - O(log h) amortized insert (`add_line`)
//! - O(log h) query via binary search on breakpoints (`argmax`)
//!
//! Uses a `Vec<Line>` sorted by slope with breakpoints enabling binary search
//! for the optimal line at any query point x.
//!
//! Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).
//! Reference: `.raw/transformer-vm/attention/hull2d_cht.h` (`_HullCHT`)

use super::types::HullMeta;

/// A line y = m*x + b with associated metadata and breakpoint.
///
/// The breakpoint `p` is the largest x-coordinate where this line
/// is optimal on the upper envelope. Lines are stored sorted by slope.
#[derive(Clone, Debug)]
pub struct Line {
    /// Slope of the line.
    pub m: f64,
    /// Y-intercept of the line.
    pub b: f64,
    /// Breakpoint: last x where this line is the best on the envelope.
    pub p: f64,
    /// Pre-aggregated value metadata for hull vertices.
    pub meta: HullMeta,
}

/// Dynamic CHT maintaining the max envelope of lines.
///
/// Internally uses a `Vec<Line>` sorted by ascending slope.
/// Among envelope lines, breakpoints are monotonically increasing,
/// enabling binary search for `argmax` queries.
pub struct CHT {
    lines: Vec<Line>,
}

impl Default for CHT {
    fn default() -> Self {
        Self::new()
    }
}

impl CHT {
    const INF: f64 = f64::INFINITY;
    const NEG_INF: f64 = f64::NEG_INFINITY;

    /// Create an empty CHT.
    pub fn new() -> Self {
        Self { lines: Vec::new() }
    }

    /// Create an empty CHT with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            lines: Vec::with_capacity(capacity),
        }
    }

    /// Whether the envelope is empty.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Number of lines on the envelope.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Remove all lines.
    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Evaluate line at x.
    #[inline]
    pub fn eval(line: &Line, x: f64) -> f64 {
        line.m * x + line.b
    }

    /// Compute the x-coordinate where two lines intersect.
    ///
    /// For lines with equal slope, returns `+INF` if the first dominates,
    /// or `−INF` otherwise.
    #[inline]
    fn line_intersect(a: &Line, b: &Line) -> f64 {
        if a.m == b.m {
            if a.b >= b.b {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            (b.b - a.b) / (a.m - b.m)
        }
    }

    /// Insert a line y = m*x + b with pre-aggregated metadata.
    ///
    /// Maintains the max envelope, removing dominated lines.
    /// Equal-slope lines are merged or the dominant one is kept.
    /// After insertion, all breakpoints are recomputed to ensure
    /// monotonically increasing ordering for binary-search queries.
    pub fn add_line(&mut self, m: f64, b: f64, meta: HullMeta) {
        let nl = Line {
            m,
            b,
            p: Self::NEG_INF,
            meta,
        };

        // Find insertion position by slope (lines sorted by ascending m).
        let pos = self.lines.partition_point(|l| l.m < m);

        // ── Handle equal-slope lines ─────────────────────────────
        if pos < self.lines.len() && self.lines[pos].m == m {
            if self.lines[pos].b == b {
                // Same line: merge metadata into existing.
                self.lines[pos].meta.merge(&nl.meta);
                return;
            } else if self.lines[pos].b >= b {
                // Existing intercept dominates for all x.
                return;
            } else {
                // New line dominates: remove existing.
                self.lines.remove(pos);
            }
        } else if pos > 0 && self.lines[pos - 1].m == m {
            if self.lines[pos - 1].b == b {
                self.lines[pos - 1].meta.merge(&nl.meta);
                return;
            } else if self.lines[pos - 1].b >= b {
                return;
            } else {
                self.lines.remove(pos - 1);
            }
        }

        // Recalculate insertion position after potential removals.
        let pos = self.lines.partition_point(|l| l.m < m);
        self.lines.insert(pos, nl);

        // ── Remove dominated interior lines ──────────────────────
        // An interior line i is dominated when its segment is empty:
        //   intersection(i−1, i) ≥ intersection(i, i+1).
        // First and last lines are never dominated (they extend to ±∞).
        let mut i = 1;
        while i + 1 < self.lines.len() {
            let p_left = Self::line_intersect(&self.lines[i - 1], &self.lines[i]);
            let p_right = Self::line_intersect(&self.lines[i], &self.lines[i + 1]);
            if p_left >= p_right {
                self.lines.remove(i);
                // Back up to recheck the previous pair after removal.
                if i > 1 {
                    i -= 1;
                }
            } else {
                i += 1;
            }
        }

        // ── Recompute all breakpoints ────────────────────────────
        // Ensures monotonically increasing breakpoints for binary search.
        for j in 0..self.lines.len().saturating_sub(1) {
            self.lines[j].p = Self::line_intersect(&self.lines[j], &self.lines[j + 1]);
        }
        if let Some(last) = self.lines.last_mut() {
            last.p = Self::INF;
        }
    }

    /// Query max at x, returning a reference to the best line.
    ///
    /// Uses binary search on breakpoints: finds the first line with `p >= x`.
    /// Returns `None` if the CHT is empty.
    pub fn argmax(&self, x: f64) -> Option<&Line> {
        if self.lines.is_empty() {
            return None;
        }
        // Binary search on breakpoints: find first line with p >= x.
        let idx = self.lines.partition_point(|l| l.p < x);
        if idx >= self.lines.len() {
            self.lines.last()
        } else {
            Some(&self.lines[idx])
        }
    }

    /// Query max at x, returning the index of the best line.
    ///
    /// Used by [`HullHalf`](super::hull::HullHalf) for neighbor walking
    /// to collect all lines with equal score for tie-breaking.
    pub fn argmax_idx(&self, x: f64) -> Option<usize> {
        if self.lines.is_empty() {
            return None;
        }
        let idx = self.lines.partition_point(|l| l.p < x);
        Some(if idx >= self.lines.len() {
            self.lines.len() - 1
        } else {
            idx
        })
    }

    /// Get the line at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn get_line(&self, idx: usize) -> Option<&Line> {
        self.lines.get(idx)
    }
}
