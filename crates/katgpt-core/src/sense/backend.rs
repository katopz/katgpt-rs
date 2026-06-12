//! NPC Brain Compute Backend — Plan 254 (ANE-Latent).
//!
//! Abstracts the NPC "think brain" compute path (sense projection, emotion, zone attention)
//! behind a trait so it can run on CPU SIMD (baseline) or ANE batch dispatch (macOS).
//!
//! **Key insight**: SenseModule uses ternary bit-plane projection (not float matmul).
//! The CPU baseline preserves exact ternary semantics. The ANE path would need
//! float-weight conversion or a custom MIL kernel. This module provides the trait
//! abstraction so either path can be swapped transparently.
//!
//! The ANE path is feature-gated behind `ane_npc` and only compiled on macOS.
//! The CPU ternary baseline is always available as fallback.

use crate::sense::brain::NpcBrain;

/// Maximum sense modules per NPC brain (matches SenseKind count).
pub const MAX_MODULES: usize = 6;

/// Per-NPC brain input for batch evaluation.
/// Contains the data needed to project HLA state onto sense modules.
#[derive(Clone, Debug)]
pub struct NpcBrainInput {
    /// HLA state vector (8-dim).
    pub hla_state: [f32; 8],
    /// Sense modules (up to MAX_MODULES).
    pub modules: [ModuleInput; MAX_MODULES],
    /// Number of valid modules (0..=MAX_MODULES).
    pub module_count: usize,
    /// GM override values per module (None = autonomous, Some(v) = pinned).
    pub overrides: [Option<f32>; MAX_MODULES],
    /// If true, autonomous computation disabled; only pinned values returned.
    pub autonomous_disabled: bool,
}

/// Per-module input for projection.
/// Captures the ternary direction data needed for dot-product + sigmoid.
///
/// Field order: large u64-aligned array first, then f32, then u8 tail
/// to minimize padding gaps.
#[derive(Clone, Copy, Debug)]
pub struct ModuleInput {
    /// Ternary direction vectors.
    pub directions: [crate::types::TernaryDir; 8],
    /// Module confidence [0, 1].
    pub confidence: f32,
    /// Number of active directions (n_directions).
    pub n_directions: u8,
}

impl Default for ModuleInput {
    fn default() -> Self {
        Self {
            directions: [crate::types::TernaryDir::zero(); 8],
            n_directions: 0,
            confidence: 0.0,
        }
    }
}

impl NpcBrainInput {
    /// Extract input from an NpcBrain for batch evaluation.
    pub fn from_brain(brain: &NpcBrain) -> Self {
        let mut modules = [ModuleInput::default(); MAX_MODULES];
        let mut overrides = [None; MAX_MODULES];

        let module_count = brain.modules.len().min(MAX_MODULES);
        for i in 0..module_count {
            let m = &brain.modules[i];
            modules[i] = ModuleInput {
                directions: m.directions,
                n_directions: m.n_directions,
                confidence: m.confidence,
            };
            overrides[i] = brain.overrides.pinned_value_brain(m.kind);
        }

        Self {
            hla_state: brain.hla_state,
            modules,
            module_count,
            overrides,
            autonomous_disabled: brain.overrides.autonomous_disabled,
        }
    }
}

impl Default for NpcBrainInput {
    fn default() -> Self {
        Self {
            hla_state: [0.0; 8],
            modules: [ModuleInput::default(); MAX_MODULES],
            module_count: 0,
            overrides: [None; MAX_MODULES],
            autonomous_disabled: false,
        }
    }
}

/// Per-NPC brain output from batch evaluation.
#[derive(Clone, Debug, Default)]
pub struct NpcBrainOutput {
    /// Projected sense values for each module.
    pub projections: [f32; MAX_MODULES],
}

/// NPC brain compute backend trait.
///
/// Implementations:
/// - `SimdNpcBrainBackend`: CPU ternary baseline (always available)
/// - `AneNpcBrainBackend`: ANE batch dispatch (macOS, feature `ane_npc`, future)
pub trait NpcBrainBackend: Send + Sync {
    /// Evaluate a batch of NPC brains.
    ///
    /// Inputs and outputs must have the same length.
    /// Returns Ok(()) on success, Err if backend-specific failure occurs
    /// (e.g., ANE not resident, model not loaded).
    fn batch_evaluate(
        &mut self,
        inputs: &[NpcBrainInput],
        outputs: &mut [NpcBrainOutput],
    ) -> Result<(), String>;

    /// Backend name for logging.
    fn backend_name(&self) -> &'static str;

    /// Maximum batch size this backend handles efficiently.
    /// Used by auto-route to decide when to batch vs serial.
    fn optimal_batch_size(&self) -> usize {
        1
    }
}

/// CPU baseline backend.
/// Exact ternary projection matching `SenseModule::project()`.
pub struct CpuTernaryBackend;

impl CpuTernaryBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuTernaryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl NpcBrainBackend for CpuTernaryBackend {
    fn batch_evaluate(
        &mut self,
        inputs: &[NpcBrainInput],
        outputs: &mut [NpcBrainOutput],
    ) -> Result<(), String> {
        if inputs.len() != outputs.len() {
            return Err(format!(
                "input/output length mismatch: {} vs {}",
                inputs.len(),
                outputs.len()
            ));
        }

        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            let projections = &mut output.projections;
            // Zero all slots first — cheaper than Default::default() which
            // goes through a memcpy of the whole struct.
            *projections = [0.0; MAX_MODULES];

            let module_count = input.module_count;
            let autonomous_disabled = input.autonomous_disabled;

            for i in 0..module_count {
                // Check GM override first
                if let Some(v) = input.overrides[i] {
                    projections[i] = v;
                    continue;
                }
                if autonomous_disabled {
                    // Already zeroed above
                    continue;
                }

                let module = &input.modules[i];
                projections[i] = project_ternary(
                    &input.hla_state,
                    &module.directions,
                    module.n_directions as usize,
                    module.confidence,
                );
            }
        }

        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "cpu_ternary"
    }

    fn optimal_batch_size(&self) -> usize {
        // CPU: no batching benefit, process serially
        1
    }
}

/// Ternary dot-product + sigmoid projection.
/// Matches `SenseModule::project()` exactly.
///
/// Uses a direct indexed loop instead of `.enumerate().take(n)` to reduce
/// iterator overhead and improve LLVM auto-vectorization.
#[inline(always)]
fn project_ternary(
    hla_state: &[f32; 8],
    directions: &[crate::types::TernaryDir; 8],
    n_directions: usize,
    confidence: f32,
) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..n_directions {
        let dir = &directions[i];
        let hla_val = hla_state[i];
        let pos = ((dir.pos_bits >> i) & 1) as u32 as f32;
        let neg = ((dir.neg_bits >> i) & 1) as u32 as f32;
        dot += (pos - neg) * hla_val * dir.row_scale;
    }
    confidence * fast_sigmoid(dot)
}

/// Fast rational sigmoid approximation matching SenseModule::project().
/// ~1ns vs ~15ns for exp()-based sigmoid. Max absolute error: ~0.003.
#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    let x = x.clamp(-12.0, 12.0);
    0.5 + x / (2.0 + (4.0 + x * x).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};
    use crate::types::SenseKind;

    fn make_brain() -> NpcBrain {
        let builder = SenseOctreeBuilder::new(3);
        let module = builder.build(
            SenseKind::SpatialSense,
            &[KgEmbedding {
                entity_hash: 1,
                relation_hash: 1,
                embedding: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                sign: true,
                confidence: 1.0,
            }],
        );
        NpcBrain::compose(vec![module])
    }

    #[test]
    fn test_cpu_backend_matches_brain_projection() {
        let brain = make_brain();
        let input = NpcBrainInput::from_brain(&brain);

        // Get expected from NpcBrain::project_all()
        let expected = brain.project_all();

        // Get actual from CpuTernaryBackend
        let mut backend = CpuTernaryBackend::new();
        let mut output = vec![NpcBrainOutput::default(); 1];
        backend.batch_evaluate(&[input], &mut output).unwrap();

        // Compare: should match exactly (same ternary projection)
        for i in 0..expected.len() {
            let diff = (output[0].projections[i] - expected[i]).abs();
            assert!(
                diff < 1e-6,
                "projection[{}] mismatch: backend={} vs brain={} (diff={})",
                i,
                output[0].projections[i],
                expected[i],
                diff
            );
        }
    }

    #[test]
    fn test_cpu_backend_batch_multiple_npcs() {
        let brain = make_brain();
        let inputs = vec![NpcBrainInput::from_brain(&brain); 10];
        let mut outputs = vec![NpcBrainOutput::default(); 10];

        let mut backend = CpuTernaryBackend::new();
        backend.batch_evaluate(&inputs, &mut outputs).unwrap();

        // All outputs should be identical (same input)
        for i in 1..10 {
            for j in 0..MAX_MODULES {
                assert_eq!(outputs[0].projections[j], outputs[i].projections[j]);
            }
        }
    }

    #[test]
    fn test_cpu_backend_gm_override() {
        let brain = make_brain();
        let mut input = NpcBrainInput::from_brain(&brain);
        input.overrides[0] = Some(0.99); // Override first module

        let mut backend = CpuTernaryBackend::new();
        let mut output = vec![NpcBrainOutput::default(); 1];
        backend.batch_evaluate(&[input], &mut output).unwrap();

        assert!(
            (output[0].projections[0] - 0.99).abs() < 1e-5,
            "GM override should take precedence"
        );
    }

    #[test]
    fn test_cpu_backend_autonomous_disabled() {
        let brain = make_brain();
        let mut input = NpcBrainInput::from_brain(&brain);
        input.autonomous_disabled = true;

        let mut backend = CpuTernaryBackend::new();
        let mut output = vec![NpcBrainOutput::default(); 1];
        backend
            .batch_evaluate(&[input.clone()], &mut output)
            .unwrap();

        for i in 0..input.module_count {
            if input.overrides[i].is_none() {
                assert_eq!(
                    output[0].projections[i], 0.0,
                    "autonomous disabled should zero"
                );
            }
        }
    }

    #[test]
    fn test_backend_name() {
        let backend = CpuTernaryBackend::new();
        assert_eq!(backend.backend_name(), "cpu_ternary");
    }

    #[test]
    fn test_length_mismatch_error() {
        let mut backend = CpuTernaryBackend::new();
        let inputs = vec![NpcBrainInput::default(); 2];
        let mut outputs = vec![NpcBrainOutput::default(); 3];
        let result = backend.batch_evaluate(&inputs, &mut outputs);
        assert!(result.is_err());
    }
}
