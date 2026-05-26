//! Rejected edit buffer — FIFO ring buffer storing negative examples for learning.

use super::gate::RejectedEdit;

/// Fixed-capacity FIFO buffer for rejected edits.
///
/// When full, the oldest entry is evicted to make room for new ones.
/// Supports JSONL serialization for persistence across sessions.
pub struct RejectedEditBuffer {
    edits: Vec<RejectedEdit>,
    max_size: usize,
}

impl RejectedEditBuffer {
    /// Create a new buffer with the given maximum capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            edits: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Push a rejected edit into the buffer. Evicts the oldest entry if full (FIFO).
    pub fn push(&mut self, edit: RejectedEdit) {
        if self.edits.len() >= self.max_size {
            self.edits.remove(0);
        }
        self.edits.push(edit);
    }

    /// Access the rejected edits as negative training examples.
    pub fn as_negative_examples(&self) -> &[RejectedEdit] {
        &self.edits
    }

    /// Clear all stored rejected edits.
    pub fn clear(&mut self) {
        self.edits.clear();
    }

    /// Serialize the buffer to JSONL (one JSON object per line).
    pub fn to_jsonl(&self) -> String {
        self.edits
            .iter()
            .filter_map(|e| serde_json::to_string(e).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Deserialize a JSONL string back into a buffer.
    ///
    /// Returns an error if any line fails to parse.
    pub fn from_jsonl(data: &str) -> Result<Self, String> {
        let mut edits = Vec::new();
        for line in data.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let edit: RejectedEdit =
                serde_json::from_str(trimmed).map_err(|e| format!("JSON parse error: {e}"))?;
            edits.push(edit);
        }
        let max_size = edits.len().max(1);
        Ok(Self { edits, max_size })
    }
}
