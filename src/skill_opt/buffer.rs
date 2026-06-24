//! Rejected edit buffer — FIFO ring buffer storing negative examples for learning.

use super::gate::RejectedEdit;

/// Fixed-capacity FIFO buffer for rejected edits.
///
/// When full, the oldest entry is evicted to make room for new ones.
/// Uses a head-index ring buffer for O(1) push instead of `rotate_left`.
/// Supports JSONL serialization for persistence across sessions.
pub struct RejectedEditBuffer {
    edits: Vec<RejectedEdit>,
    head: usize, // index of the oldest entry
    len: usize,  // number of valid entries
    max_size: usize,
}

impl RejectedEditBuffer {
    /// Create a new buffer with the given maximum capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            edits: Vec::with_capacity(max_size),
            head: 0,
            len: 0,
            max_size,
        }
    }

    /// Push a rejected edit into the buffer. Evicts the oldest entry if full (FIFO).
    ///
    /// O(1) — writes at the next slot and advances the head pointer.
    pub fn push(&mut self, edit: RejectedEdit) {
        if self.max_size == 0 {
            return;
        }
        if self.len < self.max_size {
            // Still filling — grow the Vec up to max_size.
            if self.edits.len() < self.max_size {
                self.edits.push(edit);
            } else {
                let slot = (self.head + self.len) % self.max_size;
                self.edits[slot] = edit;
            }
            self.len += 1;
        } else {
            // Full — overwrite the oldest entry and advance head.
            self.edits[self.head] = edit;
            self.head = (self.head + 1) % self.max_size;
        }
    }

    /// Iterate over rejected edits in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &RejectedEdit> {
        let head = self.head;
        let len = self.len;
        let cap = self.max_size.max(1);
        (0..len).map(move |i| {
            let idx = (head + i) % cap;
            &self.edits[idx]
        })
    }

    /// Access the rejected edits as negative training examples.
    ///
    /// Returns entries in insertion order (oldest first).
    /// Collects into a Vec because the ring buffer may wrap around.
    pub fn as_negative_examples(&self) -> Vec<&RejectedEdit> {
        self.iter().collect()
    }

    /// Clear all stored rejected edits.
    pub fn clear(&mut self) {
        self.edits.clear();
        self.head = 0;
        self.len = 0;
    }

    /// Serialize the buffer to binary (length-prefixed postcard records).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.len * 64);
        for edit in self.iter() {
            let payload = postcard::to_allocvec(edit).unwrap_or_default();
            let len = payload.len() as u32;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(&payload);
        }
        buf
    }

    /// Deserialize binary (length-prefixed postcard records) back into a buffer.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut edits = Vec::new();
        let mut offset = 0usize;
        while offset + 4 <= data.len() {
            let len = u32::from_le_bytes(
                data[offset..offset + 4]
                    .try_into()
                    .map_err(|e: std::array::TryFromSliceError| e.to_string())?,
            ) as usize;
            offset += 4;
            if offset + len > data.len() {
                return Err("truncated record".into());
            }
            let edit: RejectedEdit = postcard::from_bytes(&data[offset..offset + len])
                .map_err(|e| format!("Parse error: {e}"))?;
            edits.push(edit);
            offset += len;
        }
        let max_size = edits.len().max(1);
        let len = edits.len();
        Ok(Self {
            edits,
            head: 0,
            len,
            max_size,
        })
    }
}
