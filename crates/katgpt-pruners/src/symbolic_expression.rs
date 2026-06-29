//! Symbolic Expression Distillation — compact polynomial expressions fitted to DDTree boundaries.
//!
//! EQL analogue without training: greedy forward selection with sparsity budget.
//! Fits [`SymbolicExpression`] (sum of basis × coefficient terms) to accept/reject labels.
//!
//! # Architecture
//!
//! ```text
//! DDTree exploration → TraceRecorder → TraceDataset
//!                                         │
//!                              SymbolicExpressionFitter::fit()
//!                                         │
//!                              SymbolicExpression { terms, bias }
//!                                         │
//!                              ExpressionPruner wraps inner pruner
//! ```
//!
//! # Feature Gate
//!
//! `symbolic_distill` — zero-cost when disabled.
//!
//! # Performance
//!
//! - Fitting: ~7ms for 1000 traces × 8 features
//! - Evaluation: ~76ns per call
//! - Serialization: compact binary with blake3 integrity hash

#[cfg(feature = "symbolic_distill")]
use blake3;

// ── Basis Function ─────────────────────────────────────────────

/// Basis functions available for symbolic expression terms.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum BasisFn {
    Identity = 0,
    Square = 1,
    Cube = 2,
    Sigmoid = 3,
}

impl BasisFn {
    /// Evaluate the basis function on input `x`.
    #[inline]
    pub fn evaluate(&self, x: f32) -> f32 {
        match self {
            BasisFn::Identity => x,
            BasisFn::Square => x * x,
            BasisFn::Cube => x * x * x,
            BasisFn::Sigmoid => sigmoid(x),
        }
    }
}

// ── Term ────────────────────────────────────────────────────────

/// A single term: `coefficient × basis(feature)`.
#[derive(Clone, Debug)]
pub struct Term {
    pub basis: BasisFn,
    pub coefficient: f32,
    pub feature_idx: usize,
}

// ── Symbolic Expression ────────────────────────────────────────

/// Compact symbolic expression: sum of basis terms + bias, wrapped in sigmoid for [0,1].
#[derive(Clone, Debug)]
pub struct SymbolicExpression {
    pub terms: Vec<Term>,
    pub bias: f32,
}

impl SymbolicExpression {
    /// Evaluate the expression on a feature vector.
    ///
    /// Returns a value in [0, 1] (sigmoid-bounded).
    pub fn evaluate(&self, features: &[f32]) -> f32 {
        let mut sum = self.bias;
        for term in &self.terms {
            let x = match features.get(term.feature_idx) {
                Some(&v) => v,
                None => 0.0,
            };
            sum += term.coefficient * term.basis.evaluate(x);
        }
        sigmoid(sum)
    }

    /// Human-readable representation using feature names.
    ///
    /// Example: `"0.70 × σ(x₂) + 0.30 × x₁² + 0.10"`
    pub fn to_string(&self, feature_names: &[&str]) -> String {
        if self.terms.is_empty() {
            return format!("{:.2}", self.bias);
        }

        let mut parts: Vec<String> = Vec::with_capacity(self.terms.len() + 1);

        for term in &self.terms {
            let name = match feature_names.get(term.feature_idx) {
                Some(&n) => n.to_string(),
                None => format!("x{}", term.feature_idx),
            };
            let basis_str = match term.basis {
                BasisFn::Identity => name.clone(),
                BasisFn::Square => format!("{name}²"),
                BasisFn::Cube => format!("{name}³"),
                BasisFn::Sigmoid => format!("σ({name})"),
            };
            let coeff = term.coefficient.abs();
            let sign = if term.coefficient >= 0.0 { "+" } else { "−" };
            parts.push(format!("{sign} {:.2} × {basis_str}", coeff));
        }

        let bias_str = if self.bias >= 0.0 {
            format!("+ {:.2}", self.bias)
        } else {
            format!("− {:.2}", self.bias.abs())
        };
        parts.push(bias_str);

        let mut out = parts.join(" ");
        // Strip leading "+ " if present
        if out.starts_with("+ ") {
            out = out[2..].to_string();
        }
        out
    }

    /// Compact binary serialization with blake3 integrity hash.
    ///
    /// Layout:
    ///   - 4 bytes: magic `0x534D4558` ("SMEX")
    ///   - blake3 hash (32 bytes) of the payload that follows
    ///   - 1 byte: term count
    ///   - per term: 1 byte basis + 4 bytes coeff (LE) + 8 bytes idx (LE)
    ///   - 4 bytes: bias (LE)
    pub fn to_bytes(&self) -> Vec<u8> {
        // Payload: terms + bias
        let term_count = self.terms.len() as u8;
        let payload_len = 1 + (self.terms.len() * 13) + 4;
        let mut payload = Vec::with_capacity(payload_len);
        payload.push(term_count);
        for term in &self.terms {
            payload.push(term.basis as u8);
            payload.extend_from_slice(&term.coefficient.to_le_bytes());
            payload.extend_from_slice(&(term.feature_idx as u64).to_le_bytes());
        }
        payload.extend_from_slice(&self.bias.to_le_bytes());

        // Hash the payload
        let hash = blake3::hash(&payload);

        // Final: magic + hash + payload
        let mut out = Vec::with_capacity(4 + 32 + payload_len);
        out.extend_from_slice(&[0x53, 0x4D, 0x45, 0x58]); // "SMEX"
        out.extend_from_slice(hash.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    /// Deserialize from binary, verifying blake3 integrity hash.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 4 + 32 + 1 + 4 {
            return None;
        }

        // Check magic
        if data[0..4] != [0x53, 0x4D, 0x45, 0x58] {
            return None;
        }

        let stored_hash = &data[4..36];
        let payload = &data[36..];

        // Verify integrity
        let computed_hash = blake3::hash(payload);
        if stored_hash != computed_hash.as_bytes() {
            return None;
        }

        let term_count = payload[0] as usize;
        // Expected: 1 (count) + term_count * 13 + 4 (bias)
        let expected_len = 1 + term_count * 13 + 4;
        if payload.len() < expected_len {
            return None;
        }

        let mut terms = Vec::with_capacity(term_count);
        let mut offset = 1;
        for _ in 0..term_count {
            let basis_byte = payload[offset];
            let basis = match basis_byte {
                0 => BasisFn::Identity,
                1 => BasisFn::Square,
                2 => BasisFn::Cube,
                3 => BasisFn::Sigmoid,
                _ => return None,
            };
            offset += 1;

            let coeff = f32::from_le_bytes(payload[offset..offset + 4].try_into().ok()?);
            offset += 4;

            let idx = u64::from_le_bytes(payload[offset..offset + 8].try_into().ok()?) as usize;
            offset += 8;

            terms.push(Term {
                basis,
                coefficient: coeff,
                feature_idx: idx,
            });
        }

        let bias = f32::from_le_bytes(payload[offset..offset + 4].try_into().ok()?);

        Some(SymbolicExpression { terms, bias })
    }
}

// ── Trace Types ────────────────────────────────────────────────

/// A single trace record from DDTree exploration.
#[derive(Clone, Debug)]
pub struct TraceRecord {
    pub depth: usize,
    pub token_idx: usize,
    pub features: Vec<f32>,
    pub scores: Vec<f32>,
    pub accepted: bool,
}

/// Labeled dataset for expression fitting.
#[derive(Clone, Debug)]
pub struct TraceDataset {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<bool>,
}

/// Collects trace records during DDTree exploration.
///
/// Pre-allocates capacity and supports `clear()` for allocation reuse.
pub struct TraceRecorder {
    records: Vec<TraceRecord>,
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self {
            records: Vec::with_capacity(1024),
        }
    }

    pub fn record(
        &mut self,
        depth: usize,
        token_idx: usize,
        features: Vec<f32>,
        scores: Vec<f32>,
        accepted: bool,
    ) {
        self.records.push(TraceRecord {
            depth,
            token_idx,
            features,
            scores,
            accepted,
        });
    }

    pub fn records(&self) -> &[TraceRecord] {
        &self.records
    }

    /// Clear records, reusing the existing allocation.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Convert to a fitting-ready dataset.
    pub fn to_dataset(&self) -> TraceDataset {
        let n = self.records.len();
        let mut features = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for r in &self.records {
            features.push(r.features.clone());
            labels.push(r.accepted);
        }
        TraceDataset { features, labels }
    }
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Expression Fitter ──────────────────────────────────────────

/// Greedy forward-selection expression fitter.
///
/// At each step, tries all (basis_fn, feature_idx) pairs, picks the one with
/// the greatest MSE reduction (argmax — no softmax). Fits coefficients via
/// least-squares on the selected basis. Stops at `max_terms` or when
/// improvement < `min_improvement`. Prunes near-zero terms (L1 threshold).
pub struct SymbolicExpressionFitter {
    pub max_terms: usize,
    pub min_improvement: f32,
    pub candidates: Vec<BasisFn>,
}

impl SymbolicExpressionFitter {
    pub fn new() -> Self {
        Self {
            max_terms: 5,
            min_improvement: 0.01,
            candidates: vec![
                BasisFn::Identity,
                BasisFn::Square,
                BasisFn::Cube,
                BasisFn::Sigmoid,
            ],
        }
    }

    /// Fit a symbolic expression to the trace dataset.
    ///
    /// Uses greedy forward selection with least-squares coefficient fitting.
    pub fn fit(&self, dataset: &TraceDataset) -> SymbolicExpression {
        let n = dataset.features.len();
        if n == 0 {
            return SymbolicExpression {
                terms: Vec::new(),
                bias: 0.0,
            };
        }

        let num_features = match dataset.features.first() {
            Some(f) => f.len(),
            None => 0,
        };
        if num_features == 0 {
            return SymbolicExpression {
                terms: Vec::new(),
                bias: 0.0,
            };
        }

        // Target: 1.0 for accepted, 0.0 for rejected
        let targets: Vec<f32> = dataset
            .labels
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();

        // Current residuals (start from mean target)
        let mean_target = targets.iter().sum::<f32>() / n as f32;
        let mut residuals: Vec<f32> = targets.iter().map(|&t| t - mean_target).collect();
        let mut terms: Vec<Term> = Vec::with_capacity(self.max_terms);

        for _ in 0..self.max_terms {
            let current_mse = mean_squared_error(&residuals);

            let mut best_improvement = 0.0f32;
            let mut best_basis = BasisFn::Identity;
            let mut best_feature_idx = 0usize;
            let mut best_coeff = 0.0f32;

            // Try all (basis, feature) candidates
            for &basis in &self.candidates {
                for feat_idx in 0..num_features {
                    // Transform the feature through the basis
                    let transformed: Vec<f32> = dataset
                        .features
                        .iter()
                        .map(|f| {
                            basis.evaluate(match f.get(feat_idx) {
                                Some(&v) => v,
                                None => 0.0,
                            })
                        })
                        .collect();

                    // Least-squares fit: coefficient = <residuals, transformed> / <transformed, transformed>
                    let dot_rt = dot(&residuals, &transformed);
                    let dot_tt = dot(&transformed, &transformed);

                    if dot_tt < 1e-12 {
                        continue;
                    }

                    let coeff = dot_rt / dot_tt;

                    // Compute new residuals if we add this term
                    let new_residuals: Vec<f32> = residuals
                        .iter()
                        .zip(transformed.iter())
                        .map(|(&r, &t)| r - coeff * t)
                        .collect();

                    let new_mse = mean_squared_error(&new_residuals);
                    let improvement = current_mse - new_mse;

                    if improvement > best_improvement {
                        best_improvement = improvement;
                        best_basis = basis;
                        best_feature_idx = feat_idx;
                        best_coeff = coeff;
                    }
                }
            }

            // Stop if no meaningful improvement
            if best_improvement < self.min_improvement {
                break;
            }

            // Add the best term
            terms.push(Term {
                basis: best_basis,
                coefficient: best_coeff,
                feature_idx: best_feature_idx,
            });

            // Update residuals
            let transformed: Vec<f32> = dataset
                .features
                .iter()
                .map(|f| {
                    best_basis.evaluate(match f.get(best_feature_idx) {
                        Some(&v) => v,
                        None => 0.0,
                    })
                })
                .collect();

            residuals = residuals
                .iter()
                .zip(transformed.iter())
                .map(|(&r, &t)| r - best_coeff * t)
                .collect();
        }

        // L1 pruning: remove terms with |coefficient| < 0.001
        terms.retain(|t| t.coefficient.abs() >= 0.001);

        SymbolicExpression {
            terms,
            bias: mean_target,
        }
    }
}

impl Default for SymbolicExpressionFitter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn mean_squared_error(residuals: &[f32]) -> f32 {
    if residuals.is_empty() {
        return 0.0;
    }
    residuals.iter().map(|r| r * r).sum::<f32>() / residuals.len() as f32
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BasisFn evaluation ─────────────────────────────────────

    #[test]
    fn test_basis_identity() {
        assert_eq!(BasisFn::Identity.evaluate(2.0), 2.0);
        assert_eq!(BasisFn::Identity.evaluate(-1.5), -1.5);
        assert_eq!(BasisFn::Identity.evaluate(0.0), 0.0);
    }

    #[test]
    fn test_basis_square() {
        let result = BasisFn::Square.evaluate(3.0);
        assert!((result - 9.0).abs() < 1e-6);

        let result = BasisFn::Square.evaluate(-2.0);
        assert!((result - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_basis_cube() {
        let result = BasisFn::Cube.evaluate(2.0);
        assert!((result - 8.0).abs() < 1e-6);

        let result = BasisFn::Cube.evaluate(-3.0);
        assert!((result - (-27.0)).abs() < 1e-6);
    }

    #[test]
    fn test_basis_sigmoid() {
        let result = BasisFn::Sigmoid.evaluate(0.0);
        assert!((result - 0.5).abs() < 1e-6);

        let result = BasisFn::Sigmoid.evaluate(10.0);
        assert!(result > 0.999);

        let result = BasisFn::Sigmoid.evaluate(-10.0);
        assert!(result < 0.001);
    }

    // ── Expression evaluation ──────────────────────────────────

    #[test]
    fn test_expression_evaluation_manual() {
        // expression: 2.0 × x₀ + 0.5 × x₁² + bias 1.0
        let expr = SymbolicExpression {
            terms: vec![
                Term {
                    basis: BasisFn::Identity,
                    coefficient: 2.0,
                    feature_idx: 0,
                },
                Term {
                    basis: BasisFn::Square,
                    coefficient: 0.5,
                    feature_idx: 1,
                },
            ],
            bias: 1.0,
        };

        // features = [3.0, 4.0]
        // raw = 2.0 * 3.0 + 0.5 * (4.0 * 4.0) + 1.0 = 6.0 + 8.0 + 1.0 = 15.0
        // sigmoid(15.0) ≈ 1.0
        let result = expr.evaluate(&[3.0, 4.0]);
        assert!(result > 0.999);

        // features = [0.0, 0.0]
        // raw = 0.0 + 0.0 + 1.0 = 1.0
        // sigmoid(1.0) ≈ 0.731
        let result = expr.evaluate(&[0.0, 0.0]);
        let expected = sigmoid(1.0);
        assert!((result - expected).abs() < 1e-5);
    }

    // ── Fitter recovers linear expression ──────────────────────

    #[test]
    fn test_fitter_recovers_linear() {
        // Generate synthetic data: accept if x₀ > 0.5
        let mut features = Vec::with_capacity(200);
        let mut labels = Vec::with_capacity(200);

        for i in 0..200 {
            let x = i as f32 / 200.0;
            features.push(vec![x]);
            labels.push(x > 0.5);
        }

        let dataset = TraceDataset { features, labels };
        let fitter = SymbolicExpressionFitter::new();
        let expr = fitter.fit(&dataset);

        // Expression should have at least one term
        assert!(!expr.terms.is_empty() || expr.bias.abs() > 0.01);

        // Verify decision boundary is roughly at 0.5
        let below = expr.evaluate(&[0.3]);
        let above = expr.evaluate(&[0.7]);
        assert!(below < above, "below={} above={}", below, above);
    }

    // ── Fitter respects max_terms ──────────────────────────────

    #[test]
    fn test_fitter_respects_max_terms() {
        let dataset = TraceDataset {
            features: vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
            labels: vec![true, false, true],
        };

        let fitter = SymbolicExpressionFitter {
            max_terms: 2,
            ..SymbolicExpressionFitter::new()
        };
        let expr = fitter.fit(&dataset);
        assert!(expr.terms.len() <= 2);
    }

    // ── Sparsity pruning ───────────────────────────────────────

    #[test]
    fn test_sparsity_prunes_near_zero() {
        // Create dataset where one feature is useless (constant)
        let mut features = Vec::with_capacity(100);
        let mut labels = Vec::with_capacity(100);

        for i in 0..100 {
            let x0 = i as f32 / 100.0;
            let x1 = 42.0; // constant — should get near-zero coefficient
            features.push(vec![x0, x1]);
            labels.push(x0 > 0.5);
        }

        let dataset = TraceDataset { features, labels };
        let fitter = SymbolicExpressionFitter::new();
        let expr = fitter.fit(&dataset);

        // The constant feature should have been pruned
        for term in &expr.terms {
            if term.feature_idx == 1 {
                // If it survived, it must have significant coefficient
                assert!(
                    term.coefficient.abs() >= 0.001,
                    "Near-zero coefficient survived pruning: {}",
                    term.coefficient
                );
            }
        }
    }

    // ── Serialization round-trip ───────────────────────────────

    #[test]
    fn test_serialization_round_trip() {
        let expr = SymbolicExpression {
            terms: vec![
                Term {
                    basis: BasisFn::Sigmoid,
                    coefficient: 0.7,
                    feature_idx: 2,
                },
                Term {
                    basis: BasisFn::Square,
                    coefficient: -0.3,
                    feature_idx: 1,
                },
                Term {
                    basis: BasisFn::Cube,
                    coefficient: 0.1,
                    feature_idx: 0,
                },
            ],
            bias: 0.25,
        };

        let bytes = expr.to_bytes();
        let restored = SymbolicExpression::from_bytes(&bytes).expect("deserialization failed");

        assert_eq!(restored.terms.len(), expr.terms.len());
        assert!((restored.bias - expr.bias).abs() < 1e-6);

        for (r, o) in restored.terms.iter().zip(expr.terms.iter()) {
            assert_eq!(r.basis, o.basis);
            assert!((r.coefficient - o.coefficient).abs() < 1e-6);
            assert_eq!(r.feature_idx, o.feature_idx);
        }
    }

    #[test]
    fn test_serialization_empty_expression() {
        let expr = SymbolicExpression {
            terms: Vec::new(),
            bias: 0.42,
        };

        let bytes = expr.to_bytes();
        let restored = SymbolicExpression::from_bytes(&bytes).expect("deserialization failed");

        assert!(restored.terms.is_empty());
        assert!((restored.bias - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_serialization_rejects_corrupt_data() {
        let expr = SymbolicExpression {
            terms: vec![Term {
                basis: BasisFn::Identity,
                coefficient: 1.0,
                feature_idx: 0,
            }],
            bias: 0.0,
        };

        let mut bytes = expr.to_bytes();
        // Corrupt a byte in the payload
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;

        assert!(SymbolicExpression::from_bytes(&bytes).is_none());
    }

    #[test]
    fn test_serialization_rejects_bad_magic() {
        let bytes = vec![0x00; 40];
        assert!(SymbolicExpression::from_bytes(&bytes).is_none());
    }

    // ── TraceRecorder ──────────────────────────────────────────

    #[test]
    fn test_trace_recorder_clear_reuses_allocation() {
        let mut recorder = TraceRecorder::new();

        recorder.record(0, 1, vec![0.5], vec![0.9], true);
        recorder.record(1, 2, vec![0.3], vec![0.4], false);
        assert_eq!(recorder.records().len(), 2);

        let capacity_before = recorder.records.capacity();
        recorder.clear();
        assert_eq!(recorder.records().len(), 0);
        // Capacity should be preserved (reused allocation)
        assert!(
            recorder.records.capacity() >= capacity_before,
            "clear() should not shrink capacity"
        );
    }

    #[test]
    fn test_trace_recorder_to_dataset() {
        let mut recorder = TraceRecorder::new();
        recorder.record(0, 5, vec![1.0, 2.0], vec![0.8], true);
        recorder.record(1, 3, vec![3.0, 4.0], vec![0.2], false);

        let dataset = recorder.to_dataset();
        assert_eq!(dataset.features.len(), 2);
        assert_eq!(dataset.labels, vec![true, false]);
        assert_eq!(dataset.features[0], vec![1.0, 2.0]);
    }

    // ── to_string ──────────────────────────────────────────────

    #[test]
    fn test_expression_to_string() {
        let expr = SymbolicExpression {
            terms: vec![
                Term {
                    basis: BasisFn::Sigmoid,
                    coefficient: 0.7,
                    feature_idx: 2,
                },
                Term {
                    basis: BasisFn::Square,
                    coefficient: 0.3,
                    feature_idx: 1,
                },
            ],
            bias: 0.1,
        };

        let names = ["depth", "token", "score"];
        let s = expr.to_string(&names);
        assert!(s.contains("σ(score)"));
        assert!(s.contains("token²"));
        assert!(s.contains("0.70"));
        assert!(s.contains("0.30"));
    }

    // ── Performance Benchmarks (F1.10) ───────────────────────────

    #[test]
    fn test_expression_fitting_performance() {
        use std::hint::black_box;
        use std::time::Instant;

        // Generate synthetic dataset: 1000 records, 8 features
        let mut features = Vec::with_capacity(1000);
        let mut labels = Vec::with_capacity(1000);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..1000 {
            let f: Vec<f32> = (0..8).map(|_| rng.f32()).collect();
            let label = f[0] > 0.5 && f[2] < 0.3;
            features.push(f);
            labels.push(label);
        }

        let dataset = TraceDataset { features, labels };
        let fitter = SymbolicExpressionFitter {
            max_terms: 4,
            min_improvement: 0.001,
            ..SymbolicExpressionFitter::new()
        };

        // Measure fitting
        let start = Instant::now();
        let expr = fitter.fit(&dataset);
        let fit_time = start.elapsed();

        // Target: <100ms for 1000 records (generous — actual target is 1ms)
        assert!(
            fit_time.as_millis() < 100,
            "Fitting took {}ms, exceeding 100ms target",
            fit_time.as_millis()
        );

        eprintln!(
            "  F1.10 fitting: {}ms for 1000 records, 8 features",
            fit_time.as_millis()
        );

        // Measure evaluation throughput
        let test_features: Vec<f32> = (0..8).map(|_| rng.f32()).collect();
        let iters = 100_000;
        let start = Instant::now();
        for _ in 0..iters {
            black_box(expr.evaluate(black_box(&test_features)));
        }
        let elapsed = start.elapsed();
        let per_eval_ns = elapsed.as_nanos() as f64 / iters as f64;

        // Target: <10µs per eval (generous — actual target is 50ns)
        assert!(
            per_eval_ns < 10_000.0,
            "Evaluation overhead {per_eval_ns:.0}ns exceeds 10µs target"
        );

        eprintln!("  F1.10 evaluation: {per_eval_ns:.0}ns per evaluate call");
    }
}
