//! Shared types for the Percepta CHT Hull KV Cache.
//!
//! Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).
//! Reference: `.raw/transformer-vm/attention/hull2d_cht.h`

/// Tie-breaking mode for hard attention queries.
///
/// When multiple keys produce the same maximum dot product score,
/// this determines how the value is resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TieBreak {
    /// Return the average of all tied values.
    Average,
    /// Return the value with the highest sequence number (most recent).
    Latest,
}

/// Aggregated value metadata for hull vertices.
///
/// When multiple points map to the same hull vertex (same slope/intercept),
/// their values are merged into a single `HullMeta`. The `resolve` method
/// produces either the average or the latest value depending on tie-breaking mode.
///
/// Ported from `HullMeta` in `hull2d_cht.h`.
#[derive(Clone, Debug)]
pub struct HullMeta {
    /// Running sum of value pairs.
    pub vsum: [f64; 2],
    /// Most recent value by sequence number.
    pub vlast: [f64; 2],
    /// Highest sequence number seen.
    pub last_seq: i64,
    /// Number of merged points.
    pub count: usize,
}

impl Default for HullMeta {
    fn default() -> Self {
        Self {
            vsum: [0.0, 0.0],
            vlast: [0.0, 0.0],
            count: 0,
            last_seq: -1,
        }
    }
}

impl HullMeta {
    /// Create an empty `HullMeta`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge a new value with the given sequence number.
    #[inline]
    pub fn add(&mut self, val: [f64; 2], seq: i64) {
        self.vsum[0] += val[0];
        self.vsum[1] += val[1];
        self.count += 1;
        if seq > self.last_seq {
            self.last_seq = seq;
            self.vlast = val;
        }
    }

    /// Merge another `HullMeta` into this one.
    #[inline]
    pub fn merge(&mut self, other: &HullMeta) {
        self.vsum[0] += other.vsum[0];
        self.vsum[1] += other.vsum[1];
        self.count += other.count;
        if other.last_seq > self.last_seq {
            self.last_seq = other.last_seq;
            self.vlast = other.vlast;
        }
    }

    /// Resolve the aggregated value using the given tie-breaking mode.
    ///
    /// Returns `[0.0, 0.0]` if no values have been added.
    #[inline]
    pub fn resolve(&self, tb: TieBreak) -> [f64; 2] {
        if self.count == 0 {
            return [0.0, 0.0];
        }
        match tb {
            TieBreak::Latest => self.vlast,
            TieBreak::Average => {
                let inv = 1.0 / self.count as f64;
                [self.vsum[0] * inv, self.vsum[1] * inv]
            }
        }
    }

    /// Whether any values have been added.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// 2D vector for geometric attention operations.
///
/// Uses `f64` for consistency with the CHT implementation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Dot product — the core attention score in 2D.
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Z-component of cross product AB × AC.
    /// Positive = left turn, Negative = right turn, Zero = collinear.
    #[inline]
    pub fn cross_z(a: &Self, b: &Self, c: &Self) -> f64 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }
}

// ── Constants ──────────────────────────────────────────────────

/// Hard attention scaling constant.
/// Multiplies key coordinates to ensure hard (argmax) attention behavior.
pub const HARD_K: f64 = 1e6;

/// Large offset used in parabolic key encoding to separate
/// the tie-breaking term from the main key value.
pub const BIG: f64 = 1e12;

/// Default tolerance for floating-point comparisons.
pub const EPS: f64 = 1e-12;
