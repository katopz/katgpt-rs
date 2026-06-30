//! Ring buffers for the conformal residual pool and the seasonal forecaster.

/// A single-channel sorted ring buffer storing `(value, tick)` pairs, kept
/// sorted ascending by `value`.
///
/// Insertion is O(n) (linear scan + shift) rather than the planned O(log n)
/// binary-search insertion — the buffer is small (capacity ≤ 256 by default)
/// and the linear shift vectorizes well. If G2 latency fails, swap to binary
/// search + `ptr::copy` (still O(n) shift, but half the comparisons).
///
/// Layout: two parallel arrays `values: Vec<f32>` and `ticks: Vec<u64>`, plus
/// a `len` counter. When `len == capacity`, the oldest entry (lowest tick) is
/// evicted on push to make room.
pub struct SortedRing {
    values: Vec<f32>,
    ticks: Vec<u64>,
    len: usize,
    capacity: usize,
}

impl SortedRing {
    /// Construct an empty ring with the given per-bucket `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!(capacity >= 1);
        Self {
            values: Vec::with_capacity(capacity),
            ticks: Vec::with_capacity(capacity),
            len: 0,
            capacity,
        }
    }

    /// Current number of stored entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` iff no entries are stored.
    #[inline]
    #[allow(dead_code)] // public API for callers that introspect the pool
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sorted read of the `i`-th entry (0-indexed, ascending by value).
    ///
    /// Returns `(value, tick)`. Caller is responsible for `i < len`.
    #[inline]
    pub fn get_sorted(&self, i: usize) -> (f32, u64) {
        debug_assert!(i < self.len, "index {} out of bounds (len {})", i, self.len);
        (self.values[i], self.ticks[i])
    }

    /// Push `(value, tick)`, keeping `values` sorted ascending.
    ///
    /// If at capacity, evicts the entry with the smallest tick (oldest push).
    /// This is deterministic → bit-reproducible (G4).
    pub fn push(&mut self, value: f32, tick: u64) {
        if self.len < self.capacity {
            // Insert into the sorted position.
            let pos = self
                .values
                .binary_search_by(|v| v.partial_cmp(&value).unwrap_or(core::cmp::Ordering::Equal));
            let pos = match pos {
                Ok(p) | Err(p) => p,
            };
            self.values.insert(pos, value);
            self.ticks.insert(pos, tick);
            self.len += 1;
        } else {
            // At capacity: evict oldest-tick, then insert sorted.
            if let Some(evict_idx) = self.oldest_tick_index() {
                self.values.remove(evict_idx);
                self.ticks.remove(evict_idx);
                let pos = self.values.binary_search_by(|v| {
                    v.partial_cmp(&value).unwrap_or(core::cmp::Ordering::Equal)
                });
                let pos = match pos {
                    Ok(p) | Err(p) => p,
                };
                self.values.insert(pos, value);
                self.ticks.insert(pos, tick);
            }
        }
    }

    /// Index of the entry with the smallest tick (oldest). `None` if empty.
    fn oldest_tick_index(&self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        let mut best = 0usize;
        let mut best_tick = self.ticks[0];
        for i in 1..self.len {
            if self.ticks[i] < best_tick {
                best_tick = self.ticks[i];
                best = i;
            }
        }
        Some(best)
    }

    /// Clear all entries (keeps capacity).
    #[allow(dead_code)] // public API for callers that reset the pool
    pub fn clear(&mut self) {
        self.values.clear();
        self.ticks.clear();
        self.len = 0;
    }
}

/// Per-channel × per-horizon-bucket pool of sorted residual rings.
///
/// Layout: `n_channels × n_buckets` rings, each of `capacity` entries. Indexed
/// by `channel_bucket(channel, bucket)`.
pub struct ResidualRingBuffer {
    /// Flat array of rings: `rings[channel * n_buckets + bucket]`.
    rings: Vec<SortedRing>,
    /// Number of channels (dim 0).
    pub n_channels: usize,
    /// Number of horizon buckets (dim 1).
    pub n_buckets: usize,
}

impl ResidualRingBuffer {
    /// Construct a new pool with the given shape.
    pub fn new(n_channels: usize, n_buckets: usize, capacity: usize) -> Self {
        debug_assert!(n_channels >= 1);
        debug_assert!(n_buckets >= 1);
        debug_assert!(capacity >= 1);
        let total = n_channels
            .checked_mul(n_buckets)
            .expect("n_channels * n_buckets overflow");
        let mut rings = Vec::with_capacity(total);
        for _ in 0..total {
            rings.push(SortedRing::with_capacity(capacity));
        }
        Self {
            rings,
            n_channels,
            n_buckets,
        }
    }

    /// Flat index for `(channel, bucket)`.
    #[inline]
    fn flat(&self, channel: usize, bucket: usize) -> usize {
        debug_assert!(channel < self.n_channels, "channel {} oob", channel);
        debug_assert!(bucket < self.n_buckets, "bucket {} oob", bucket);
        channel * self.n_buckets + bucket
    }

    /// Push `(residual, tick)` into `(channel, bucket)`.
    #[inline]
    pub fn push(&mut self, residual: f32, channel: usize, bucket: usize, tick: u64) {
        let i = self.flat(channel, bucket);
        self.rings[i].push(residual, tick);
    }

    /// Read-only view of the `(channel, bucket)` ring.
    #[inline]
    pub fn channel_bucket(&self, channel: usize, bucket: usize) -> RingView<'_> {
        let i = self.flat(channel, bucket);
        RingView {
            inner: &self.rings[i],
        }
    }
}

/// Read-only view into one `SortedRing`.
pub struct RingView<'a> {
    inner: &'a SortedRing,
}

impl<'a> RingView<'a> {
    /// Number of stored entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Sorted read of the `i`-th entry.
    #[inline]
    pub fn get_sorted(&self, i: usize) -> (f32, u64) {
        self.inner.get_sorted(i)
    }
}

/// A simple FIFO ring buffer of `f32`, used by the seasonal forecaster's
/// history window. Not sorted.
pub struct RingBuffer {
    buf: Vec<f32>,
    head: usize,
    len: usize,
    capacity: usize,
}

impl RingBuffer {
    /// Construct an empty ring with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!(capacity >= 1);
        Self {
            buf: vec![0.0; capacity],
            head: 0,
            len: 0,
            capacity,
        }
    }

    /// Current number of stored entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` iff no entries are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` iff at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Push a value, evicting the oldest if at capacity.
    #[inline]
    pub fn push(&mut self, value: f32) {
        if self.len < self.capacity {
            let idx = (self.head + self.len) % self.capacity;
            self.buf[idx] = value;
            self.len += 1;
        } else {
            self.buf[self.head] = value;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Read the value at logical index `i` (0 = oldest). Returns `None` if
    /// `i >= len`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<f32> {
        if i >= self.len {
            return None;
        }
        let physical = (self.head + i) % self.capacity;
        Some(self.buf[physical])
    }

    /// Read the value at offset `back` from the most recent entry
    /// (`back = 0` = newest, `back = 1` = second-newest, ...).
    /// Returns `None` if `back >= len`.
    #[inline]
    pub fn back(&self, back: usize) -> Option<f32> {
        if back >= self.len {
            return None;
        }
        let i = self.len - 1 - back;
        self.get(i)
    }

    /// Clear all entries (keeps capacity).
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_ring_keeps_sorted_on_push() {
        let mut r = SortedRing::with_capacity(8);
        for &v in &[3.0_f32, 1.0, 4.0, 1.5, 2.0, 1.0, 3.5, 0.5] {
            r.push(v, 0);
        }
        // After 8 pushes capacity is full.
        let got: Vec<f32> = (0..r.len()).map(|i| r.get_sorted(i).0).collect();
        assert_eq!(got, vec![0.5, 1.0, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0]);
    }

    #[test]
    fn sorted_ring_evicts_oldest_tick_at_capacity() {
        let mut r = SortedRing::with_capacity(3);
        r.push(10.0, 1); // tick 1 — oldest
        r.push(20.0, 2);
        r.push(30.0, 3);
        // Now full. Push a 4th → evict tick 1.
        r.push(40.0, 4);
        let got: Vec<(f32, u64)> = (0..r.len()).map(|i| r.get_sorted(i)).collect();
        assert_eq!(got, vec![(20.0, 2), (30.0, 3), (40.0, 4)]);
    }

    #[test]
    fn ring_buffer_fifo() {
        let mut rb = RingBuffer::with_capacity(3);
        for &v in &[1.0_f32, 2.0, 3.0] {
            rb.push(v);
        }
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.get(0), Some(1.0));
        assert_eq!(rb.get(1), Some(2.0));
        assert_eq!(rb.get(2), Some(3.0));
        assert_eq!(rb.back(0), Some(3.0));
        assert_eq!(rb.back(1), Some(2.0));
        // Push past capacity → evict oldest.
        rb.push(4.0);
        assert_eq!(rb.get(0), Some(2.0));
        assert_eq!(rb.get(1), Some(3.0));
        assert_eq!(rb.get(2), Some(4.0));
        assert_eq!(rb.back(0), Some(4.0));
    }

    #[test]
    fn residual_pool_channel_bucket_isolation() {
        let mut pool = ResidualRingBuffer::new(2, 2, 4);
        // Channel 0, bucket 0.
        pool.push(1.0, 0, 0, 0);
        pool.push(2.0, 0, 0, 0);
        // Channel 1, bucket 1.
        pool.push(10.0, 1, 1, 0);
        pool.push(20.0, 1, 1, 0);
        // Channel 0 bucket 0 should NOT see channel 1's values.
        let v00 = pool.channel_bucket(0, 0);
        assert_eq!(v00.len(), 2);
        assert_eq!(v00.get_sorted(0).0, 1.0);
        assert_eq!(v00.get_sorted(1).0, 2.0);
        // Channel 1 bucket 0 should be empty.
        let v10 = pool.channel_bucket(1, 0);
        assert_eq!(v10.len(), 0);
        // Channel 1 bucket 1 should see its own values.
        let v11 = pool.channel_bucket(1, 1);
        assert_eq!(v11.len(), 2);
        assert_eq!(v11.get_sorted(0).0, 10.0);
    }
}
