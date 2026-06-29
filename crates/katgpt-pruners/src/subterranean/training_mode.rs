//! SubterraneanTrainingMode — training mode configuration for procedure compilation.
//!
//! Plan 110, T9: Full fine-tuning flag for compiling procedures into weights.
//!
//! Paper: arXiv:2605.22502 — "Procedural internalization requires modifying the
//! model's implicit state-tracking behavior — a deeper change than stylistic
//! alignment. LoRA fails to approach full fine-tuning on procedural tasks."
//!
//! This module provides the configuration type only. Actual full fine-tuning
//! implementation belongs in `riir-ai` (riir-gpu).

use std::fmt;

// ── SubterraneanTrainingMode ───────────────────────────────────

/// Training mode for compiling procedures into model weights.
///
/// The paper proves that procedural knowledge requires deeper model changes
/// than stylistic alignment. LoRA ranks 16–128 all fail to internalize
/// procedure graphs, while full fine-tuning achieves 87–98% of frontier quality.
///
/// # Choosing a mode
///
/// - [`Lora`](Self::Lora): Only for trivial procedures (< 10 paths).
///   Fast but insufficient for most real workflows.
/// - [`FullFineTune`]: Required for procedural internalization.
///   Paper's recommended default.
/// - [`Qlora`]: Middle ground — 4-bit quantization + full parameter update.
///   Reduces memory at ~2% quality cost.
///
/// # Example
///
/// ```ignore
/// use katgpt_rs::pruners::subterranean::training_mode::SubterraneanTrainingMode;
///
/// let mode = SubterraneanTrainingMode::default(); // FullFineTune
/// assert!(mode.is_full_finetune());
/// assert!(!mode.is_lora());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SubterraneanTrainingMode {
    /// Low-rank adaptation — works for style transfer, FAILS for procedures.
    ///
    /// Paper tested ranks 16, 32, 64, 128 — all fell short of full fine-tuning
    /// on procedure tasks. Only use for trivial procedures with < 10 unique paths.
    Lora {
        /// Rank of the low-rank decomposition.
        rank: usize,
    },

    /// Full parameter updates — REQUIRED for procedural internalization.
    ///
    /// The paper's recommended approach. Modifies the model's implicit
    /// state-tracking behavior, which is essential for procedure compilation.
    #[default]
    FullFineTune,

    /// QLoRA middle ground — quantized base + full parameter update.
    ///
    /// Uses NF4 quantization for the base model while performing full
    /// parameter updates. Reduces VRAM requirements by ~60% at the cost
    /// of ~2% quality degradation compared to full fine-tuning.
    Qlora {
        /// Quantization bits (typically 4).
        bits: u8,
    },
}

impl SubterraneanTrainingMode {
    /// Whether this mode uses LoRA (low-rank adaptation).
    pub fn is_lora(&self) -> bool {
        matches!(self, Self::Lora { .. })
    }

    /// Whether this mode uses full fine-tuning.
    pub fn is_full_finetune(&self) -> bool {
        matches!(self, Self::FullFineTune)
    }

    /// Whether this mode uses QLoRA (quantized LoRA).
    pub fn is_qlora(&self) -> bool {
        matches!(self, Self::Qlora { .. })
    }

    /// Estimated VRAM multiplier relative to base model size.
    ///
    /// - LoRA: ~1.1× (small adapter overhead)
    /// - FullFineTune: ~3.0× (optimizer states + gradients)
    /// - QLoRA: ~1.5× (quantized base + paged optimizer)
    pub fn vram_multiplier(&self) -> f64 {
        match self {
            Self::Lora { .. } => 1.1,
            Self::FullFineTune => 3.0,
            Self::Qlora { bits } => match bits {
                4 => 1.5,
                8 => 2.0,
                _ => 2.5,
            },
        }
    }

    /// Estimated quality fraction relative to frontier model.
    ///
    /// Paper: 87–98% for full fine-tuning, lower for LoRA.
    pub fn estimated_quality(&self, path_count: usize) -> f64 {
        let base_quality: f64 = match self {
            Self::FullFineTune => match path_count {
                0..=50 => 0.98,
                51..=200 => 0.95,
                201..=1000 => 0.92,
                1001..=5000 => 0.89,
                _ => 0.87,
            },
            Self::Lora { .. } => {
                // LoRA fails for procedures: paper shows significant quality drop
                match path_count {
                    0..=10 => 0.90,
                    11..=50 => 0.75,
                    51..=200 => 0.60,
                    _ => 0.45,
                }
            }
            Self::Qlora { .. } => {
                // QLoRA: ~2% below full fine-tuning
                let full = match path_count {
                    0..=50 => 0.98,
                    51..=200 => 0.95,
                    201..=1000 => 0.92,
                    1001..=5000 => 0.89,
                    _ => 0.87,
                };
                full - 0.02
            }
        };

        base_quality.clamp(0.0, 1.0)
    }

    /// Whether this mode is suitable for the given procedure complexity.
    ///
    /// Paper: LoRA fails for procedures with > 10 unique paths.
    pub fn suitable_for_path_count(&self, path_count: usize) -> bool {
        match self {
            Self::Lora { .. } => path_count < 10,
            Self::FullFineTune => true,
            Self::Qlora { .. } => true,
        }
    }

    /// Recommended training hours estimate multiplier.
    ///
    /// Full fine-tuning takes longer but produces better results.
    pub fn training_time_multiplier(&self) -> f64 {
        match self {
            Self::Lora { rank } => {
                // Higher rank = longer training, but still faster than full FT
                let base = 0.3;
                base + (*rank as f64 / 128.0) * 0.2
            }
            Self::FullFineTune => 1.0,
            Self::Qlora { bits } => match bits {
                4 => 0.8, // Slightly faster due to quantized ops
                _ => 0.9,
            },
        }
    }

    /// Human-readable description of this training mode.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Lora { .. } => "LoRA — fast but insufficient for procedures",
            Self::FullFineTune => "Full Fine-Tune — required for procedural internalization",
            Self::Qlora { .. } => "QLoRA — quantized fine-tune, reduced VRAM",
        }
    }

    /// Get the LoRA rank, if applicable.
    pub fn lora_rank(&self) -> Option<usize> {
        match self {
            Self::Lora { rank } => Some(*rank),
            _ => None,
        }
    }

    /// Get the QLoRA bits, if applicable.
    pub fn qlora_bits(&self) -> Option<u8> {
        match self {
            Self::Qlora { bits } => Some(*bits),
            _ => None,
        }
    }
}

impl fmt::Display for SubterraneanTrainingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lora { rank } => write!(f, "LoRA(rank={rank})"),
            Self::FullFineTune => write!(f, "FullFineTune"),
            Self::Qlora { bits } => write!(f, "QLoRA(bits={bits})"),
        }
    }
}

// ── TrainingBudget ─────────────────────────────────────────────

/// Estimated resource budget for a procedure compilation training run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingBudget {
    /// Training mode to use.
    pub mode: SubterraneanTrainingMode,
    /// Estimated training hours on a single A100.
    pub estimated_hours: f64,
    /// Estimated VRAM requirement in GB.
    pub vram_gb: f64,
    /// Whether this mode is suitable for the target procedure.
    pub suitable: bool,
    /// Estimated quality fraction (0.0–1.0).
    pub estimated_quality: f64,
}

impl TrainingBudget {
    /// Estimate training budget for a given mode and procedure complexity.
    ///
    /// `base_hours` is the baseline training time for full fine-tuning.
    /// `model_size_gb` is the model's parameter memory in GB.
    /// `path_count` is the number of unique paths in the procedure graph.
    pub fn estimate(
        mode: SubterraneanTrainingMode,
        base_hours: f64,
        model_size_gb: f64,
        path_count: usize,
    ) -> Self {
        let time_multiplier = mode.training_time_multiplier();
        let vram_multiplier = mode.vram_multiplier();

        Self {
            mode,
            estimated_hours: base_hours * time_multiplier,
            vram_gb: model_size_gb * vram_multiplier,
            suitable: mode.suitable_for_path_count(path_count),
            estimated_quality: mode.estimated_quality(path_count),
        }
    }
}

impl fmt::Display for TrainingBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.1}h, {:.0}GB VRAM, quality={:.0}%, suitable={}",
            self.mode,
            self.estimated_hours,
            self.vram_gb,
            self.estimated_quality * 100.0,
            self.suitable
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_full_finetune() {
        assert_eq!(
            SubterraneanTrainingMode::default(),
            SubterraneanTrainingMode::FullFineTune
        );
    }

    #[test]
    fn test_mode_classification() {
        let lora = SubterraneanTrainingMode::Lora { rank: 16 };
        let full = SubterraneanTrainingMode::FullFineTune;
        let qlora = SubterraneanTrainingMode::Qlora { bits: 4 };

        assert!(lora.is_lora());
        assert!(!lora.is_full_finetune());
        assert!(!lora.is_qlora());

        assert!(!full.is_lora());
        assert!(full.is_full_finetune());
        assert!(!full.is_qlora());

        assert!(!qlora.is_lora());
        assert!(!qlora.is_full_finetune());
        assert!(qlora.is_qlora());
    }

    #[test]
    fn test_vram_multiplier() {
        let lora = SubterraneanTrainingMode::Lora { rank: 32 };
        let full = SubterraneanTrainingMode::FullFineTune;
        let qlora_4 = SubterraneanTrainingMode::Qlora { bits: 4 };
        let qlora_8 = SubterraneanTrainingMode::Qlora { bits: 8 };

        assert!(lora.vram_multiplier() > 1.0);
        assert!(full.vram_multiplier() > lora.vram_multiplier());
        assert!(qlora_4.vram_multiplier() < full.vram_multiplier());
        assert!(qlora_8.vram_multiplier() > qlora_4.vram_multiplier());
    }

    #[test]
    fn test_lora_unsuitable_for_complex_procedures() {
        let lora = SubterraneanTrainingMode::Lora { rank: 128 };

        assert!(lora.suitable_for_path_count(5));
        assert!(!lora.suitable_for_path_count(10));
        assert!(!lora.suitable_for_path_count(100));
    }

    #[test]
    fn test_full_finetune_suitable_always() {
        let full = SubterraneanTrainingMode::FullFineTune;

        assert!(full.suitable_for_path_count(0));
        assert!(full.suitable_for_path_count(100));
        assert!(full.suitable_for_path_count(10000));
    }

    #[test]
    fn test_quality_bounds() {
        let full = SubterraneanTrainingMode::FullFineTune;

        let quality = full.estimated_quality(50);
        assert!(quality >= 0.87, "Quality should be >= 87%, got {quality}");
        assert!(quality <= 1.0, "Quality should be <= 100%, got {quality}");
    }

    #[test]
    fn test_lora_quality_drops_with_complexity() {
        let lora = SubterraneanTrainingMode::Lora { rank: 64 };

        let simple = lora.estimated_quality(5);
        let complex = lora.estimated_quality(100);

        assert!(
            simple > complex,
            "LoRA quality should drop with complexity: {simple} vs {complex}"
        );
    }

    #[test]
    fn test_qlora_slightly_below_full() {
        let full = SubterraneanTrainingMode::FullFineTune;
        let qlora = SubterraneanTrainingMode::Qlora { bits: 4 };

        let full_quality = full.estimated_quality(100);
        let qlora_quality = qlora.estimated_quality(100);

        assert!(
            qlora_quality < full_quality,
            "QLoRA should be slightly below full FT: {qlora_quality} vs {full_quality}"
        );
        assert!(
            (full_quality - qlora_quality - 0.02).abs() < 0.001,
            "QLoRA quality gap should be ~2%"
        );
    }

    #[test]
    fn test_display() {
        let lora = SubterraneanTrainingMode::Lora { rank: 32 };
        let full = SubterraneanTrainingMode::FullFineTune;
        let qlora = SubterraneanTrainingMode::Qlora { bits: 4 };

        assert_eq!(format!("{lora}"), "LoRA(rank=32)");
        assert_eq!(format!("{full}"), "FullFineTune");
        assert_eq!(format!("{qlora}"), "QLoRA(bits=4)");
    }

    #[test]
    fn test_description() {
        let full = SubterraneanTrainingMode::FullFineTune;
        assert!(full.description().contains("required"));
    }

    #[test]
    fn test_lora_rank_accessor() {
        let lora = SubterraneanTrainingMode::Lora { rank: 64 };
        let full = SubterraneanTrainingMode::FullFineTune;

        assert_eq!(lora.lora_rank(), Some(64));
        assert_eq!(full.lora_rank(), None);
    }

    #[test]
    fn test_qlora_bits_accessor() {
        let qlora = SubterraneanTrainingMode::Qlora { bits: 4 };
        let full = SubterraneanTrainingMode::FullFineTune;

        assert_eq!(qlora.qlora_bits(), Some(4));
        assert_eq!(full.qlora_bits(), None);
    }

    #[test]
    fn test_training_time_multiplier_ordering() {
        let lora = SubterraneanTrainingMode::Lora { rank: 16 };
        let full = SubterraneanTrainingMode::FullFineTune;
        let qlora = SubterraneanTrainingMode::Qlora { bits: 4 };

        assert!(lora.training_time_multiplier() < full.training_time_multiplier());
        assert!(qlora.training_time_multiplier() < full.training_time_multiplier());
    }

    #[test]
    fn test_training_budget_estimate() {
        let budget = TrainingBudget::estimate(
            SubterraneanTrainingMode::FullFineTune,
            3.0,  // base hours
            14.0, // model size GB (7B model)
            100,  // path count
        );

        assert_eq!(budget.mode, SubterraneanTrainingMode::FullFineTune);
        assert!(budget.estimated_hours > 0.0);
        assert!(budget.vram_gb > 14.0); // Full FT needs > model size
        assert!(budget.suitable);
        assert!(budget.estimated_quality > 0.87);
    }

    #[test]
    fn test_training_budget_lora_unsuitable() {
        let budget = TrainingBudget::estimate(
            SubterraneanTrainingMode::Lora { rank: 64 },
            3.0,
            14.0,
            500, // Complex procedure
        );

        assert!(!budget.suitable, "LoRA should be unsuitable for 500 paths");
    }

    #[test]
    fn test_training_budget_display() {
        let budget =
            TrainingBudget::estimate(SubterraneanTrainingMode::FullFineTune, 3.0, 14.0, 50);

        let display = format!("{budget}");
        assert!(display.contains("FullFineTune"));
        assert!(display.contains("GB"));
        assert!(display.contains("%"));
    }
}
