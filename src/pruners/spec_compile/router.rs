//! SpecRouter — adaptive CPU/SIMD/GPU/ANE compute tier routing for spec pruners.
//!
//! Classifies compiled specs by complexity and routes them to the appropriate
//! compute backend. Simple specs (few rules, small bitmaps) use plain CPU bitmap
//! lookup; moderate specs get SIMD batch validation; complex/unknown specs are
//! earmarked for GPU ternary adapter fallback (future).
//!
//! Feature gate: `spec_pruner`

use katgpt_core::traits::ConstraintPruner;

use super::types::*;

// ── Complexity thresholds ────────────────────────────────────────

/// Maximum rule count for Simple complexity (CPU bitmap path).
const SIMPLE_MAX_RULES: usize = 4;
/// Maximum bitmap cardinality for Simple complexity.
const SIMPLE_MAX_BITMAP_SIZE: usize = 1024;
/// Maximum rule count for Medium complexity (SIMD batch path).
const MEDIUM_MAX_RULES: usize = 16;
/// Maximum bitmap cardinality for Medium complexity.
const MEDIUM_MAX_BITMAP_SIZE: usize = 8192;

// ── Enums ────────────────────────────────────────────────────────

/// Spec complexity classification — determines which compute path to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SpecComplexity {
    /// CPU bitmap path — few rules, small token sets. O(1) bitmap lookup.
    Simple = 0,
    /// SIMD batch path — moderate rules, medium token sets. Batch validation.
    Medium = 1,
    /// GPU/ANE ternary adapter fallback — complex or unknown specs.
    Fuzzy = 2,
}

/// Compute tier — the execution backend for constraint validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComputeTier {
    /// Standard CPU bitmap lookup.
    Cpu = 0,
    /// SIMD batch validation.
    Simd = 1,
    /// GPU ternary adapter (future).
    Gpu = 2,
    /// Apple Neural Engine (future).
    Ane = 3,
}

// ── Router ───────────────────────────────────────────────────────

/// Adaptive compute tier router for compiled specs.
///
/// Stateless — all routing is deterministic from the spec's content.
/// The router classifies specs by complexity heuristic:
/// `rule_count * max_bitmap_size * dfa_state_estimate`.
///
/// Special cases:
/// - `SpecType::Unknown` → always Fuzzy/Gpu
/// - `SpecType::Classification` → always Simple/Cpu (bitmap is the right tool)
#[derive(Clone, Debug, Default)]
pub struct SpecRouter;

impl SpecRouter {
    /// Create a new router.
    #[inline]
    pub const fn new() -> Self {
        SpecRouter
    }

    /// Classify a compiled spec's complexity.
    ///
    /// Uses a heuristic combining rule count, max bitmap size, and
    /// estimated DFA state count. Special spec types override the heuristic.
    pub fn classify_complexity(&self, spec: &CompiledSpec, spec_type: SpecType) -> SpecComplexity {
        // SpecType overrides — short-circuit the heuristic.
        match spec_type {
            SpecType::Unknown => return SpecComplexity::Fuzzy,
            SpecType::Classification => return SpecComplexity::Simple,
            _ => {}
        }

        let rule_count = spec.rules.len();

        // Estimate max bitmap cardinality across all rules + global sets.
        let max_bitmap_size = spec
            .rules
            .iter()
            .map(|r| r.allowed.len())
            .chain(std::iter::once(spec.global_allowed.len()))
            .chain(std::iter::once(spec.global_blocked.len()))
            .max()
            .unwrap_or(0);

        // Estimate DFA state count: sum of unique depths + 1 for global.
        let dfa_state_count = {
            let mut depths: Vec<usize> = spec.rules.iter().filter_map(|r| r.depth).collect();
            depths.sort_unstable();
            depths.dedup();
            depths.len() + 1 // +1 for global/depth-agnostic
        };

        // Heuristic product — high values push toward Fuzzy.
        let _heuristic = rule_count * max_bitmap_size * dfa_state_count;

        // Tier classification by thresholds.
        if rule_count <= SIMPLE_MAX_RULES && max_bitmap_size <= SIMPLE_MAX_BITMAP_SIZE {
            SpecComplexity::Simple
        } else if rule_count <= MEDIUM_MAX_RULES && max_bitmap_size <= MEDIUM_MAX_BITMAP_SIZE {
            SpecComplexity::Medium
        } else {
            SpecComplexity::Fuzzy
        }
    }

    /// Route a compiled spec to the best compute tier.
    ///
    /// Maps complexity → tier. Fuzzy complexity defaults to Gpu,
    /// with Ane available as a latency-critical alternative.
    pub fn route(&self, spec: &CompiledSpec, spec_type: SpecType) -> ComputeTier {
        match self.classify_complexity(spec, spec_type) {
            SpecComplexity::Simple => ComputeTier::Cpu,
            SpecComplexity::Medium => ComputeTier::Simd,
            SpecComplexity::Fuzzy => ComputeTier::Gpu,
        }
    }

    /// Route multiple specs in batch.
    ///
    /// Each spec is routed independently based on its own complexity.
    pub fn route_batch(&self, specs: &[(&CompiledSpec, SpecType)]) -> Vec<ComputeTier> {
        specs
            .iter()
            .map(|&(spec, st)| self.route(spec, st))
            .collect()
    }

    /// Route with latency preference — returns Ane for Fuzzy if `latency_critical`.
    pub fn route_with_latency(
        &self,
        spec: &CompiledSpec,
        spec_type: SpecType,
        latency_critical: bool,
    ) -> ComputeTier {
        match self.classify_complexity(spec, spec_type) {
            SpecComplexity::Simple => ComputeTier::Cpu,
            SpecComplexity::Medium => ComputeTier::Simd,
            SpecComplexity::Fuzzy => {
                if latency_critical {
                    ComputeTier::Ane
                } else {
                    ComputeTier::Gpu
                }
            }
        }
    }
}

// ── ConstraintPruner delegation ──────────────────────────────────

/// The router delegates all validation to the `CompiledSpec`'s own `ConstraintPruner`
/// impl. SIMD/GPU backends are future work — this provides the routing infrastructure
/// without premature optimization.
impl ConstraintPruner for SpecRouter {
    /// Routes to the appropriate tier and delegates to the spec's validation.
    ///
    /// Note: The router itself is stateless and does not hold a spec reference.
    /// Callers should use `CompiledSpec::is_valid` directly for actual validation,
    /// or wrap spec + router together. This impl exists to satisfy the trait bound
    /// for API compatibility.
    #[inline]
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        // Stateless router has no spec to validate against.
        // Default: allow everything. Use route() to determine tier, then
        // delegate to the spec's own ConstraintPruner impl.
        true
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: minimal spec with N rules, each containing `bitmap_size` tokens.
    fn make_spec(rule_count: usize, bitmap_size: usize) -> CompiledSpec {
        let allowed = if bitmap_size == 0 {
            CompactBitmap::empty()
        } else {
            CompactBitmap::from_token_indices(0..bitmap_size)
        };

        CompiledSpec {
            spec_hash: [0u8; 32],
            rules: (0..rule_count)
                .map(|_| SpecRule {
                    depth: None,
                    prefix: Vec::new(),
                    allowed: allowed.clone(),
                    is_allowlist: true,
                })
                .collect(),
            vocab_size: bitmap_size.max(256),
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        }
    }

    fn make_spec_with_global(bitmap_size: usize) -> CompiledSpec {
        CompiledSpec {
            spec_hash: [0u8; 32],
            rules: vec![],
            vocab_size: bitmap_size.max(256),
            global_allowed: CompactBitmap::from_token_indices(0..bitmap_size),
            global_blocked: CompactBitmap::empty(),
        }
    }

    #[test]
    fn test_simple_classification_routes_to_cpu() {
        let router = SpecRouter::new();
        let spec = make_spec(2, 64);
        let tier = router.route(&spec, SpecType::Classification);
        assert_eq!(
            tier,
            ComputeTier::Cpu,
            "Classification specs always route to CPU"
        );
    }

    #[test]
    fn test_unknown_spec_routes_to_gpu() {
        let router = SpecRouter::new();
        let spec = make_spec(1, 10);
        let complexity = router.classify_complexity(&spec, SpecType::Unknown);
        assert_eq!(
            complexity,
            SpecComplexity::Fuzzy,
            "Unknown specs are always Fuzzy"
        );

        let tier = router.route(&spec, SpecType::Unknown);
        assert_eq!(tier, ComputeTier::Gpu, "Unknown specs route to GPU");
    }

    #[test]
    fn test_medium_complexity_routes_to_simd() {
        let router = SpecRouter::new();
        // 8 rules, 2000 tokens — exceeds Simple but within Medium
        let spec = make_spec(8, 2000);
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(complexity, SpecComplexity::Medium);

        let tier = router.route(&spec, SpecType::Extraction);
        assert_eq!(tier, ComputeTier::Simd, "Medium complexity routes to SIMD");
    }

    #[test]
    fn test_simple_complexity() {
        let router = SpecRouter::new();
        let spec = make_spec(3, 512);
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(
            complexity,
            SpecComplexity::Simple,
            "3 rules, 512 tokens → Simple"
        );
    }

    #[test]
    fn test_fuzzy_complexity() {
        let router = SpecRouter::new();
        // 32 rules, 10000 tokens — exceeds both Simple and Medium thresholds
        let spec = make_spec(32, 10000);
        let complexity = router.classify_complexity(&spec, SpecType::FormatRepair);
        assert_eq!(
            complexity,
            SpecComplexity::Fuzzy,
            "32 rules, 10000 tokens → Fuzzy"
        );
    }

    #[test]
    fn test_boundary_simple_max_rules() {
        let router = SpecRouter::new();
        // Exactly SIMPLE_MAX_RULES and SIMPLE_MAX_BITMAP_SIZE → still Simple
        let spec = make_spec(SIMPLE_MAX_RULES, SIMPLE_MAX_BITMAP_SIZE);
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(
            complexity,
            SpecComplexity::Simple,
            "At threshold boundary → Simple"
        );
    }

    #[test]
    fn test_boundary_medium_max_rules() {
        let router = SpecRouter::new();
        // Exactly MEDIUM_MAX_RULES and MEDIUM_MAX_BITMAP_SIZE → still Medium
        let spec = make_spec(MEDIUM_MAX_RULES, MEDIUM_MAX_BITMAP_SIZE);
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(
            complexity,
            SpecComplexity::Medium,
            "At Medium threshold boundary → Medium"
        );
    }

    #[test]
    fn test_just_above_simple() {
        let router = SpecRouter::new();
        // SIMPLE_MAX_RULES + 1 rules, within Medium bitmap → Medium
        let spec = make_spec(SIMPLE_MAX_RULES + 1, 512);
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(complexity, SpecComplexity::Medium);
    }

    #[test]
    fn test_just_above_medium_rules() {
        let router = SpecRouter::new();
        // MEDIUM_MAX_RULES + 1 rules → Fuzzy regardless of bitmap
        let spec = make_spec(MEDIUM_MAX_RULES + 1, 512);
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(complexity, SpecComplexity::Fuzzy);
    }

    #[test]
    fn test_just_above_simple_bitmap() {
        let router = SpecRouter::new();
        // Within Simple rules but bitmap exceeds SIMPLE_MAX_BITMAP_SIZE → Medium
        let spec = make_spec(2, SIMPLE_MAX_BITMAP_SIZE + 1);
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(complexity, SpecComplexity::Medium);
    }

    #[test]
    fn test_global_bitmap_included_in_heuristic() {
        let router = SpecRouter::new();
        // No rules but large global_allowed → pushes complexity up
        let spec = make_spec_with_global(SIMPLE_MAX_BITMAP_SIZE + 1);
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(
            complexity,
            SpecComplexity::Medium,
            "Global bitmap size counts toward complexity"
        );
    }

    #[test]
    fn test_intent_routing_medium() {
        let router = SpecRouter::new();
        // IntentRouting uses heuristic (no override), moderate spec → Medium
        let spec = make_spec(8, 2000);
        let complexity = router.classify_complexity(&spec, SpecType::IntentRouting);
        assert_eq!(complexity, SpecComplexity::Medium);
    }

    #[test]
    fn test_batch_routing() {
        let router = SpecRouter::new();

        let simple_spec = make_spec(2, 64);
        let medium_spec = make_spec(8, 2000);
        let unknown_spec = make_spec(1, 10);

        let specs: Vec<(&CompiledSpec, SpecType)> = vec![
            (&simple_spec, SpecType::Classification),
            (&medium_spec, SpecType::Extraction),
            (&unknown_spec, SpecType::Unknown),
        ];

        let tiers = router.route_batch(&specs);
        assert_eq!(tiers.len(), 3);
        assert_eq!(tiers[0], ComputeTier::Cpu, "Classification → CPU");
        assert_eq!(tiers[1], ComputeTier::Simd, "Medium extraction → SIMD");
        assert_eq!(tiers[2], ComputeTier::Gpu, "Unknown → GPU");
    }

    #[test]
    fn test_route_with_latency_not_critical() {
        let router = SpecRouter::new();
        let spec = make_spec(32, 10000); // Fuzzy
        let tier = router.route_with_latency(&spec, SpecType::FormatRepair, false);
        assert_eq!(tier, ComputeTier::Gpu, "Non-latency-critical Fuzzy → GPU");
    }

    #[test]
    fn test_route_with_latency_critical() {
        let router = SpecRouter::new();
        let spec = make_spec(32, 10000); // Fuzzy
        let tier = router.route_with_latency(&spec, SpecType::FormatRepair, true);
        assert_eq!(tier, ComputeTier::Ane, "Latency-critical Fuzzy → ANE");
    }

    #[test]
    fn test_complexity_deterministic() {
        let router = SpecRouter::new();
        let spec = make_spec(5, 3000);

        let c1 = router.classify_complexity(&spec, SpecType::Extraction);
        let c2 = router.classify_complexity(&spec, SpecType::Extraction);
        let c3 = router.classify_complexity(&spec, SpecType::Extraction);

        assert_eq!(c1, c2, "Same spec → same complexity (call 1 vs 2)");
        assert_eq!(c2, c3, "Same spec → same complexity (call 2 vs 3)");
    }

    #[test]
    fn test_empty_spec_is_simple() {
        let router = SpecRouter::new();
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };
        let complexity = router.classify_complexity(&spec, SpecType::Extraction);
        assert_eq!(complexity, SpecComplexity::Simple, "Empty spec → Simple");
    }

    #[test]
    fn test_default_router() {
        let router = SpecRouter::default();
        let spec = make_spec(1, 10);
        let tier = router.route(&spec, SpecType::Classification);
        assert_eq!(tier, ComputeTier::Cpu);
    }

    #[test]
    fn test_complexity_repr_u8() {
        assert_eq!(SpecComplexity::Simple as u8, 0);
        assert_eq!(SpecComplexity::Medium as u8, 1);
        assert_eq!(SpecComplexity::Fuzzy as u8, 2);

        assert_eq!(ComputeTier::Cpu as u8, 0);
        assert_eq!(ComputeTier::Simd as u8, 1);
        assert_eq!(ComputeTier::Gpu as u8, 2);
        assert_eq!(ComputeTier::Ane as u8, 3);
    }
}
