//! Emotion vector inference — modelless observation from mid-layer activations.
//!
//! Projects residual-stream activations onto pre-computed emotion direction vectors
//! (valence, arousal, desperation, calm) during decode. Zero extra forward pass,
//! O(d) dot product per decode step.
//!
//! Based on Anthropic Transformer Circuits Thread, 2026 — emotion vectors causally
//! drive behavior (desperation → 14× reward hacking increase).
//!
//! # Architecture
//!
//! Direction vectors are pre-computed once at model load time and stored in
//! `EmotionDirections`. During decode, the mid-layer activation vector is projected
//! onto each direction via a simple dot product — no allocation, no extra forward pass.
//!
//! # Plan 162 — Phase 1: Infrastructure

/// Result of projecting activations onto all four emotion directions.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EmotionReading {
    /// Valence PC1 projection (positive = happy/calm, negative = desperate/angry).
    pub valence: f32,
    /// Arousal PC2 projection (positive = high arousal, negative = low arousal).
    pub arousal: f32,
    /// Desperation-specific direction projection.
    pub desperation: f32,
    /// Calm-specific direction projection.
    pub calm: f32,
}

impl std::fmt::Display for EmotionReading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "valence={:.3} arousal={:.3} desperation={:.3} calm={:.3}",
            self.valence, self.arousal, self.desperation, self.calm
        )
    }
}

/// Pre-computed emotion direction vectors for a given model.
///
/// Loaded once at model config time — zero allocation during decode.
/// Default vectors are zeros with the correct dimension; a real model
/// would load calibrated directions from checkpoint.
///
/// # Example
///
/// ```rust,ignore
/// let dirs = EmotionDirections::zeros(64);
/// let activation = vec![1.0f32; 64];
/// let reading = dirs.read_emotions(&activation);
/// assert_eq!(reading.valence, 0.0); // zero directions → zero projections
/// ```
pub struct EmotionDirections {
    /// Valence PC1 direction (positive = happy/calm, negative = desperate/angry).
    pub valence: Vec<f32>,
    /// Arousal PC2 direction (positive = high arousal, negative = low arousal).
    pub arousal: Vec<f32>,
    /// Desperation-specific direction.
    pub desperation: Vec<f32>,
    /// Calm-specific direction.
    pub calm: Vec<f32>,
}

impl EmotionDirections {
    /// Create zero-initialized direction vectors of the given dimension.
    ///
    /// Placeholder for when calibrated directions are not available.
    /// All projections onto zero vectors yield 0.0.
    pub fn zeros(d_model: usize) -> Self {
        Self {
            valence: vec![0.0; d_model],
            arousal: vec![0.0; d_model],
            desperation: vec![0.0; d_model],
            calm: vec![0.0; d_model],
        }
    }

    /// Create from explicit direction vectors.
    ///
    /// All four vectors must have the same length.
    /// Panics if lengths disagree.
    pub fn new(
        valence: Vec<f32>,
        arousal: Vec<f32>,
        desperation: Vec<f32>,
        calm: Vec<f32>,
    ) -> Self {
        let d = valence.len();
        assert_eq!(arousal.len(), d, "arousal direction dimension mismatch");
        assert_eq!(
            desperation.len(),
            d,
            "desperation direction dimension mismatch"
        );
        assert_eq!(calm.len(), d, "calm direction dimension mismatch");
        Self {
            valence,
            arousal,
            desperation,
            calm,
        }
    }

    /// Dimension of the direction vectors.
    pub fn dim(&self) -> usize {
        self.valence.len()
    }

    /// Project activation vector onto a direction — O(d) dot product, zero alloc.
    ///
    /// This is the core primitive: a single dot product between the mid-layer
    /// activation and a pre-computed emotion direction.
    ///
    /// Uses 4-wide chunked accumulation to help LLVM auto-vectorize.
    #[inline]
    pub fn project(activation: &[f32], direction: &[f32]) -> f32 {
        let len = activation.len().min(direction.len());
        let chunks = len / 4;

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let act = &activation[..len];
        let dir = &direction[..len];

        for i in 0..chunks {
            let base = i * 4;
            sum0 += act[base] * dir[base];
            sum1 += act[base + 1] * dir[base + 1];
            sum2 += act[base + 2] * dir[base + 2];
            sum3 += act[base + 3] * dir[base + 3];
        }

        let mut total = sum0 + sum1 + sum2 + sum3;

        for i in (chunks * 4)..len {
            total += act[i] * dir[i];
        }

        total
    }

    /// Project activations onto all four emotion directions.
    ///
    /// Returns an `EmotionReading` with all four projections.
    /// This is the hook point for the decode loop — call once per
    /// decode step with the mid-layer activation vector.
    ///
    /// If `activations` length != direction dimension, the shorter length
    /// is used (matching `Iterator::zip` behavior).
    pub fn read_emotions(&self, activations: &[f32]) -> EmotionReading {
        EmotionReading {
            valence: Self::project(activations, &self.valence),
            arousal: Self::project(activations, &self.arousal),
            desperation: Self::project(activations, &self.desperation),
            calm: Self::project(activations, &self.calm),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_directions_yield_zero_projections() {
        let dirs = EmotionDirections::zeros(4);
        let activation = vec![1.0, 2.0, 3.0, 4.0];
        let reading = dirs.read_emotions(&activation);
        assert_eq!(reading.valence, 0.0);
        assert_eq!(reading.arousal, 0.0);
        assert_eq!(reading.desperation, 0.0);
        assert_eq!(reading.calm, 0.0);
    }

    #[test]
    fn test_project_dot_product() {
        let activation = [1.0, 2.0, 3.0];
        let direction = [0.5, -1.0, 2.0];
        // 1*0.5 + 2*(-1.0) + 3*2.0 = 0.5 - 2.0 + 6.0 = 4.5
        let result = EmotionDirections::project(&activation, &direction);
        assert!((result - 4.5).abs() < 1e-6, "expected 4.5, got {result}");
    }

    #[test]
    fn test_project_empty() {
        let result = EmotionDirections::project(&[], &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_project_orthogonal() {
        let activation = [1.0, 0.0];
        let direction = [0.0, 1.0];
        let result = EmotionDirections::project(&activation, &direction);
        assert!((result).abs() < 1e-6, "expected 0.0, got {result}");
    }

    #[test]
    fn test_read_emotions_known_directions() {
        let dirs = EmotionDirections::new(
            vec![1.0, 0.0],  // valence: only reads first component
            vec![0.0, 1.0],  // arousal: only reads second component
            vec![1.0, 1.0],  // desperation: sum of both
            vec![1.0, -1.0], // calm: difference
        );
        let activation = vec![3.0, 5.0];
        let reading = dirs.read_emotions(&activation);

        assert!((reading.valence - 3.0).abs() < 1e-6);
        assert!((reading.arousal - 5.0).abs() < 1e-6);
        assert!((reading.desperation - 8.0).abs() < 1e-6);
        assert!((reading.calm - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_read_emotions_mismatched_length_uses_shorter() {
        let dirs = EmotionDirections::zeros(3);
        let activation = vec![1.0, 2.0]; // shorter than direction
        let reading = dirs.read_emotions(&activation);
        // zero directions → zero regardless
        assert_eq!(reading.valence, 0.0);
    }

    #[test]
    fn test_dim() {
        let dirs = EmotionDirections::zeros(128);
        assert_eq!(dirs.dim(), 128);
    }

    #[test]
    #[should_panic(expected = "arousal direction dimension mismatch")]
    fn test_new_dimension_mismatch_panics() {
        EmotionDirections::new(
            vec![1.0, 2.0],
            vec![1.0], // wrong length
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        );
    }

    #[test]
    fn test_emotion_reading_display() {
        let reading = EmotionReading {
            valence: 0.123,
            arousal: -0.456,
            desperation: 1.789,
            calm: 0.001,
        };
        let s = format!("{reading}");
        assert!(s.contains("valence=0.123"), "display missing valence: {s}");
        assert!(s.contains("arousal=-0.456"), "display missing arousal: {s}");
        assert!(
            s.contains("desperation=1.789"),
            "display missing desperation: {s}"
        );
        assert!(s.contains("calm=0.001"), "display missing calm: {s}");
    }

    #[test]
    fn test_emotion_reading_default() {
        let reading = EmotionReading::default();
        assert_eq!(reading.valence, 0.0);
        assert_eq!(reading.arousal, 0.0);
        assert_eq!(reading.desperation, 0.0);
        assert_eq!(reading.calm, 0.0);
    }
}
