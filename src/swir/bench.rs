//! Benchmark harness for the SwiR GOAT gate (Plan 275 Phase 3, T3.2).
//!
//! This module provides the **engine-side** benchmark structure: traits that
//! abstract the "real model" + "real dataset" halves, plus a synthetic
//! reference implementation that validates the harness wiring on
//! deterministic data. riir-ai Plan 299 implements the traits against a real
//! Qwen3-1.7B model and the MATH500 dataset to produce the empirical G1
//! (accuracy) and G2 (efficiency) gates.
//!
//! # Why this split
//!
//! katgpt-rs is a modelless primitives library — no model loader, no
//! tokenizer, no KV cache (see `src/swir/README.md`). The paper's accuracy
//! and efficiency claims are empirical properties of the *combination*
//! (SwiR controller + real LLM); they cannot be measured in katgpt-rs alone.
//! But the **harness structure** — how to run two configs, what to measure,
//! how to sweep hyperparameters — is generic and ships here so riir-ai
//! doesn't reinvent it.
//!
//! # Usage (engine-side, synthetic validation)
//!
//! ```
//! use katgpt_rs::swir::bench::*;
//!
//! // Synthetic source — deterministic, exercises the harness wiring.
//! let source = SyntheticProblemSource::new(50, 42);
//! let backend = SyntheticDecodeBackend::new(42);
//! let result = run_benchmark(&source, &backend, BenchConfig::default());
//! println!("{}", result.summary());
//! ```
//!
//! # Usage (riir-ai side — real model)
//!
//! riir-ai Plan 299 implements `ProblemSource` over MATH500 and
//! `DecodeBackend` over candle/gguf, then calls the same `run_benchmark`.
//! The harness structure is identical; only the trait implementations differ.

use crate::swir::{SwiRConfig, SwiRController, StepAction, ThinkMode};

// ────────────────────────────────────────────────────────────────────────────
// Traits — the abstraction boundary between engine and fuel
// ────────────────────────────────────────────────────────────────────────────

/// Abstracts a benchmark problem source (MATH500, GSM8K, etc.).
///
/// The engine doesn't know what a "math problem" is — it just needs a
/// prompt string and a way to check whether a generated answer is correct.
/// riir-ai implements this over the real dataset + answer checker.
pub trait ProblemSource {
    /// Number of problems in the subset (e.g., 50 for a quick sweep, 500 for
    /// the full GOAT proof).
    fn len(&self) -> usize;

    /// Whether the source contains zero problems.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the prompt for problem `i`. The host is responsible for any
    /// prompt templating (e.g., wrapping in `<think>...</think>` for Qwen3).
    fn prompt(&self, i: usize) -> &str;

    /// Check whether `answer` is correct for problem `i`. The host implements
    /// the answer-extraction + ground-truth comparison (e.g., `\boxed{}`
    /// extraction for MATH500).
    fn is_correct(&self, i: usize, answer: &str) -> bool;
}

/// Abstracts the decode backend (real LLM, mock LLM, etc.).
///
/// The engine calls `decode_step` to advance the model one token. The backend
/// is responsible for the KV cache, the embedding matrix, and the softmax.
/// This is the trait that riir-ai implements over candle/gguf.
pub trait DecodeBackend {
    /// Dimensionality of the embedding vectors (for soft-embedding scratch).
    fn embedding_dim(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// The full embedding matrix, flattened as `[vocab * embedding_dim]`.
    /// Row `v` starts at index `v * embedding_dim`. Used for `soft_embedding`.
    fn embedding_matrix(&self) -> &[f32];

    /// Advance the decode loop one step. The backend receives the current
    /// "token" (either a concrete id for Explicit mode, or a soft-embedding
    /// slice for Latent mode), runs one forward pass, and writes the resulting
    /// probability distribution into `probs_out` (length = `vocab_size()`).
    ///
    /// Returns the argmax token id (for accuracy measurement — the "answer"
    /// token if this is the final step).
    fn decode_step(
        &mut self,
        token_id: Option<u32>,
        soft_embedding: Option<&[f32]>,
        probs_out: &mut [f32],
    ) -> u32;

    /// Reset the backend to a fresh state for a new problem.
    fn reset(&mut self, prompt: &str);
}

// ────────────────────────────────────────────────────────────────────────────
// Config + result types
// ────────────────────────────────────────────────────────────────────────────

/// Which configuration to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchMode {
    /// Baseline: vanilla thinking_cot, no SwiR mode switching. Always emits
    /// concrete tokens, terminates at `max_steps` or when the model emits
    /// the answer prefix naturally.
    Baseline,
    /// SwiR: drive the decode loop through `SwiRController::step()`. Switches
    /// between Explicit (concrete token) and Latent (soft embedding) modes
    /// based on entropy trends.
    Swir,
}

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Which mode to run (Baseline or SwiR).
    pub mode: BenchMode,
    /// SwiR hyperparameters (ignored for Baseline).
    pub swir_config: SwiRConfig,
    /// Maximum decode steps per problem.
    pub max_steps: u32,
    /// Temperature for answer extraction (0 = greedy).
    pub temperature: f32,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            mode: BenchMode::Swir,
            swir_config: SwiRConfig::default(),
            max_steps: 1024,
            temperature: 0.0,
        }
    }
}

/// Per-problem result.
#[derive(Debug, Clone)]
pub struct ProblemResult {
    /// Problem index.
    pub index: usize,
    /// Whether the generated answer was correct.
    pub correct: bool,
    /// Total decode steps used (Explicit + Latent).
    pub total_steps: u32,
    /// Steps spent in Latent mode (SwiR only; 0 for Baseline).
    pub latent_steps: u32,
    /// Steps spent in Explicit mode.
    pub explicit_steps: u32,
    /// Number of mode switches (SwiR only; 0 for Baseline).
    pub switches: u32,
    /// Wall-clock latency in nanoseconds.
    pub latency_ns: u64,
}

/// Aggregate result for a full benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Mode that was run.
    pub mode: BenchMode,
    /// Per-problem results.
    pub problems: Vec<ProblemResult>,
}

impl BenchResult {
    /// Accuracy = correct / total.
    pub fn accuracy(&self) -> f32 {
        if self.problems.is_empty() {
            return 0.0;
        }
        let correct = self.problems.iter().filter(|p| p.correct).count();
        correct as f32 / self.problems.len() as f32
    }

    /// Average total decode steps per problem.
    pub fn avg_steps(&self) -> f32 {
        if self.problems.is_empty() {
            return 0.0;
        }
        self.problems.iter().map(|p| p.total_steps).sum::<u32>() as f32
            / self.problems.len() as f32
    }

    /// Average wall-clock latency per problem (nanoseconds).
    pub fn avg_latency_ns(&self) -> u64 {
        if self.problems.is_empty() {
            return 0;
        }
        self.problems.iter().map(|p| p.latency_ns).sum::<u64>()
            / self.problems.len() as u64
    }

    /// Average number of mode switches per problem (SwiR only).
    pub fn avg_switches(&self) -> f32 {
        if self.problems.is_empty() {
            return 0.0;
        }
        self.problems.iter().map(|p| p.switches).sum::<u32>() as f32
            / self.problems.len() as f32
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "{:?}: accuracy={:.1}%, avg_steps={:.0}, avg_latency={:.1}ms, avg_switches={:.1}",
            self.mode,
            self.accuracy() * 100.0,
            self.avg_steps(),
            self.avg_latency_ns() as f64 / 1_000_000.0,
            self.avg_switches(),
        )
    }
}

/// Comparison between Baseline and SwiR runs — the GOAT gate verdict.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub baseline: BenchResult,
    pub swir: BenchResult,
}

impl ComparisonResult {
    /// G1 gate: accuracy delta (SwiR − Baseline), in percentage points.
    /// Target: ≥ +1.5pp (paper §4.1).
    pub fn accuracy_delta_pp(&self) -> f32 {
        (self.swir.accuracy() - self.baseline.accuracy()) * 100.0
    }

    /// G2 gate: token efficiency ratio (Baseline_steps / Swir_steps) at the
    /// achieved accuracy. Target: ≥ 1.3× (paper §4.2).
    pub fn token_efficiency_ratio(&self) -> f32 {
        if self.swir.avg_steps() > 0.0 {
            self.baseline.avg_steps() / self.swir.avg_steps()
        } else {
            0.0
        }
    }

    /// Latency ratio (Baseline / Swir). > 1.0 means SwiR is faster.
    pub fn latency_ratio(&self) -> f32 {
        if self.swir.avg_latency_ns() > 0 {
            self.baseline.avg_latency_ns() as f32 / self.swir.avg_latency_ns() as f32
        } else {
            0.0
        }
    }

    /// Human-readable GOAT gate verdict.
    pub fn verdict(&self) -> String {
        let d_pp = self.accuracy_delta_pp();
        let eff = self.token_efficiency_ratio();
        let g1_pass = d_pp >= 1.5;
        let g2_pass = eff >= 1.3;
        format!(
            "G1 accuracy delta: {d_pp:+.1}pp (target ≥ +1.5pp) — {}\n\
             G2 token efficiency: {eff:.2}× (target ≥ 1.3×) — {}\n\
             Latency ratio: {:.2}× ({})",
            if g1_pass { "✅ PASS" } else { "❌ FAIL" },
            if g2_pass { "✅ PASS" } else { "❌ FAIL" },
            self.latency_ratio(),
            if self.latency_ratio() >= 1.0 { "SwiR faster" } else { "SwiR slower" },
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// The benchmark runner
// ────────────────────────────────────────────────────────────────────────────

/// Run a benchmark over all problems in `source`, using `backend` for decoding.
///
/// This is the core T3.2 function. It drives the decode loop for each problem,
/// measuring accuracy, token count, latency, and (for SwiR mode) mode switches.
///
/// For `BenchMode::Baseline`: always emits concrete tokens (the argmax from the
/// backend's `decode_step`), terminates at `max_steps` or when the model emits
/// a natural answer prefix.
///
/// For `BenchMode::Swir`: drives through `SwiRController::step()`, which
/// switches between Explicit (concrete token) and Latent (soft embedding) modes.
pub fn run_benchmark(
    source: &dyn ProblemSource,
    backend: &mut dyn DecodeBackend,
    config: BenchConfig,
) -> BenchResult {
    let vocab = backend.vocab_size();
    let dim = backend.embedding_dim();
    let emb = backend.embedding_matrix().to_vec(); // owned copy for borrow ease

    let mut probs_buf: Vec<f32> = vec![0.0; vocab];
    let mut soft_buf: Vec<f32> = vec![0.0; dim];
    let mut results: Vec<ProblemResult> = Vec::with_capacity(source.len());

    for i in 0..source.len() {
        let prompt = source.prompt(i);
        backend.reset(prompt);

        let mut ctrl = SwiRController::new(config.swir_config);
        let mut current_token: u32 = 0;
        let mut answer_tokens = String::new();
        let mut switches = 0u32;
        let start = std::time::Instant::now();

        // Priming decode: feed the prompt (as token_id=None) to get the first
        // probability distribution. Without this, probs_buf is all zeros on
        // step 0 → entropy=0 → reference_entropy=0 → the controller never
        // sees entropy BELOW reference, so Latent→Explicit never fires.
        // The host (riir-ai) handles this naturally via its prompt-processing
        // pass; the bench harness must do the same.
        if config.mode == BenchMode::Swir {
            let _ = backend.decode_step(None, None, &mut probs_buf);
        }

        for step in 0..config.max_steps {
            let prev_mode = if config.mode == BenchMode::Swir {
                Some(ctrl.mode())
            } else {
                None
            };

            // Decide what to feed the backend this step.
            let (token_id, soft_emb, should_decode) = if config.mode == BenchMode::Swir {
                let entropy = {
                    // We need probs to compute entropy, but we haven't decoded
                    // yet this step. Use the previous step's probs (stored in
                    // probs_buf). On step 0, probs_buf is all zeros → entropy
                    // is 0 → controller starts in Latent (paper default).
                    crate::swir::shannon_entropy(&probs_buf)
                };
                match ctrl.step(entropy, step) {
                    StepAction::EmitToken(id) => (Some(id), None, true),
                    StepAction::EmitSoftEmbedding => {
                        // Compute soft embedding into scratch.
                        for x in soft_buf.iter_mut() {
                            *x = 0.0;
                        }
                        crate::swir::soft_embedding(&probs_buf, &emb, dim, &mut soft_buf);
                        (None, Some(&soft_buf[..]), true)
                    }
                    StepAction::InjectControlToken(_) => {
                        // Skip — the host would inject the control token here.
                        // For benchmark purposes, we just advance without
                        // decoding (the injection is a no-op for the backend).
                        (None, None, false)
                    }
                    StepAction::Terminate => break,
                }
            } else {
                // Baseline: always emit the previous argmax token.
                (Some(current_token), None, true)
            };

            if !should_decode {
                continue;
            }

            // Run one decode step.
            let argmax = backend.decode_step(token_id, soft_emb, &mut probs_buf);
            current_token = argmax;

            // Track switches for SwiR mode.
            if config.mode == BenchMode::Swir {
                let new_mode = ctrl.mode();
                if prev_mode == Some(ThinkMode::Latent) && new_mode == ThinkMode::Explicit {
                    switches += 1;
                }
            }

            // Accumulate answer tokens (simplified — real host would detokenize).
            answer_tokens.push_str(&format!(" {}", argmax));
        }

        let elapsed = start.elapsed();
        let stats = ctrl.stats();
        let (latent, explicit_s, total) = if config.mode == BenchMode::Swir {
            (stats.latent_steps, stats.explicit_steps, stats.latent_steps + stats.explicit_steps)
        } else {
            (0, config.max_steps, config.max_steps)
        };

        let correct = source.is_correct(i, &answer_tokens);
        results.push(ProblemResult {
            index: i,
            correct,
            total_steps: total,
            latent_steps: latent,
            explicit_steps: explicit_s,
            switches,
            latency_ns: elapsed.as_nanos() as u64,
        });
    }

    BenchResult {
        mode: config.mode,
        problems: results,
    }
}

/// Run a Pareto sweep over C_max values, comparing Baseline vs SwiR at each.
///
/// Produces a curve of (C_max, accuracy_delta, efficiency_ratio) tuples that
/// the host can plot. Paper Tab. 10 reports C_max=20 as the sweet spot; this
/// sweep verifies the same shape on the host's model.
pub fn run_pareto_sweep(
    source: &dyn ProblemSource,
    backend: &mut dyn DecodeBackend,
    c_max_values: &[u32],
    base_config: BenchConfig,
) -> Vec<ParetoPoint> {
    let mut points = Vec::with_capacity(c_max_values.len() + 1);

    // Baseline (C_max = ∞, effectively no SwiR switching).
    let baseline_config = BenchConfig {
        mode: BenchMode::Baseline,
        ..base_config
    };
    let baseline = run_benchmark(source, backend, baseline_config);
    points.push(ParetoPoint {
        c_max: u32::MAX,
        accuracy: baseline.accuracy(),
        avg_steps: baseline.avg_steps(),
        is_baseline: true,
    });

    for &c_max in c_max_values {
        let swir_config = SwiRConfig {
            c_max,
            ..base_config.swir_config
        };
        let config = BenchConfig {
            mode: BenchMode::Swir,
            swir_config,
            ..base_config
        };
        let swir = run_benchmark(source, backend, config);
        points.push(ParetoPoint {
            c_max,
            accuracy: swir.accuracy(),
            avg_steps: swir.avg_steps(),
            is_baseline: false,
        });
    }

    points
}

/// One point on the Pareto curve.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub c_max: u32,
    pub accuracy: f32,
    pub avg_steps: f32,
    pub is_baseline: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// Synthetic reference implementations — for harness validation
// ────────────────────────────────────────────────────────────────────────────

/// Synthetic problem source — deterministic, for harness validation.
///
/// Generates `n` fake "problems" with deterministic prompts and a simple
/// correctness check (answer must contain a specific token). This is NOT a
/// real benchmark — it just validates that the harness wiring is correct.
/// riir-ai replaces this with the real MATH500 loader.
pub struct SyntheticProblemSource {
    prompts: Vec<String>,
    answer_tokens: Vec<u32>,
}

impl SyntheticProblemSource {
    /// Create `n` synthetic problems with the given RNG seed.
    pub fn new(n: usize, seed: u64) -> Self {
        let mut state = seed.max(1);
        let mut prompts = Vec::with_capacity(n);
        let mut answer_tokens = Vec::with_capacity(n);
        for i in 0..n {
            // Simple LCG for deterministic variety.
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let answer_token = (state % 1000) as u32;
            prompts.push(format!("Problem {i}: generate token {answer_token}."));
            answer_tokens.push(answer_token);
        }
        Self {
            prompts,
            answer_tokens,
        }
    }
}

impl ProblemSource for SyntheticProblemSource {
    fn len(&self) -> usize {
        self.prompts.len()
    }

    fn prompt(&self, i: usize) -> &str {
        &self.prompts[i]
    }

    fn is_correct(&self, i: usize, answer: &str) -> bool {
        // Correct if the answer contains the target token number.
        answer.contains(&self.answer_tokens[i].to_string())
    }
}

/// Synthetic decode backend — deterministic, for harness validation.
///
/// Simulates a model that emits a predictable entropy schedule. This is NOT
/// a real LLM — it just validates the harness wiring. riir-ai replaces this
/// with candle/gguf.
pub struct SyntheticDecodeBackend {
    dim: usize,
    vocab: usize,
    embedding: Vec<f32>,
    step_counter: u32,
}

impl SyntheticDecodeBackend {
    pub fn new(_seed: u64) -> Self {
        let dim = 32;
        let vocab = 100;
        // Simple embedding matrix: token v has embedding[v*dim + d] = v as f32.
        let embedding: Vec<f32> = (0..(vocab * dim))
            .map(|i| ((i / dim) % vocab) as f32)
            .collect();
        Self {
            dim,
            vocab,
            embedding,
            step_counter: 0,
        }
    }
}

impl DecodeBackend for SyntheticDecodeBackend {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn embedding_matrix(&self) -> &[f32] {
        &self.embedding
    }

    fn decode_step(
        &mut self,
        _token_id: Option<u32>,
        _soft_embedding: Option<&[f32]>,
        probs_out: &mut [f32],
    ) -> u32 {
        // Deterministic entropy schedule: alternate between flat (high entropy)
        // and peaked (low entropy) in 4-step blocks. The first block is flat
        // so the controller's reference_entropy is set high, enabling
        // subsequent low-entropy steps to trigger Latent→Explicit switches.
        //
        // Schedule (step_counter increments before use, starts at 0):
        //   calls 1-4: flat  (H ≈ log(vocab))
        //   calls 5-8: peaked (H ≈ 0.01)
        //   calls 9-12: flat
        //   (repeats with period 8)
        self.step_counter += 1;
        let is_flat = ((self.step_counter - 1) % 8) < 4;
        for (i, p) in probs_out.iter_mut().enumerate() {
            if is_flat {
                *p = 1.0 / self.vocab as f32;
            } else {
                *p = if i == 0 { 0.999 } else { 0.001 / (self.vocab - 1) as f32 };
            }
        }
        0
    }

    fn reset(&mut self, _prompt: &str) {
        self.step_counter = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_source_generates_n_problems() {
        let src = SyntheticProblemSource::new(10, 42);
        assert_eq!(src.len(), 10);
        assert!(src.prompt(0).contains("Problem 0:"));
        assert!(src.prompt(9).contains("Problem 9:"));
    }

    #[test]
    fn test_synthetic_source_correctness_check() {
        let src = SyntheticProblemSource::new(1, 42);
        let target = src.answer_tokens[0];
        assert!(src.is_correct(0, &format!("answer is {target}")));
        assert!(!src.is_correct(0, "answer is 999999"));
    }

    #[test]
    fn test_benchmark_runs_baseline_mode() {
        let src = SyntheticProblemSource::new(5, 42);
        let mut backend = SyntheticDecodeBackend::new(42);
        let config = BenchConfig {
            mode: BenchMode::Baseline,
            max_steps: 16,
            ..Default::default()
        };
        let result = run_benchmark(&src, &mut backend, config);
        assert_eq!(result.problems.len(), 5);
        assert_eq!(result.mode, BenchMode::Baseline);
        // Baseline should have 0 latent steps and 0 switches.
        assert!(result.problems.iter().all(|p| p.latent_steps == 0));
        assert!(result.problems.iter().all(|p| p.switches == 0));
    }

    #[test]
    fn test_benchmark_runs_swir_mode() {
        let src = SyntheticProblemSource::new(5, 42);
        let mut backend = SyntheticDecodeBackend::new(42);
        let config = BenchConfig {
            mode: BenchMode::Swir,
            swir_config: SwiRConfig {
                w_e_to_l: 1, // fast switching for short test
                c_max: 4,
                ..Default::default()
            },
            max_steps: 64,
            ..Default::default()
        };
        let result = run_benchmark(&src, &mut backend, config);
        assert_eq!(result.problems.len(), 5);
        assert_eq!(result.mode, BenchMode::Swir);
        // SwiR mode should produce some latent steps (the schedule alternates).
        let total_latent: u32 = result.problems.iter().map(|p| p.latent_steps).sum();
        assert!(total_latent > 0, "SwiR mode should produce latent steps");
    }

    #[test]
    fn test_comparison_result_accuracy_delta() {
        let baseline = BenchResult {
            mode: BenchMode::Baseline,
            problems: vec![
                ProblemResult {
                    index: 0,
                    correct: true,
                    total_steps: 100,
                    latent_steps: 0,
                    explicit_steps: 100,
                    switches: 0,
                    latency_ns: 1_000_000,
                },
                ProblemResult {
                    index: 1,
                    correct: false,
                    total_steps: 100,
                    latent_steps: 0,
                    explicit_steps: 100,
                    switches: 0,
                    latency_ns: 1_000_000,
                },
            ],
        };
        let swir = BenchResult {
            mode: BenchMode::Swir,
            problems: vec![
                ProblemResult {
                    index: 0,
                    correct: true,
                    total_steps: 50,
                    latent_steps: 10,
                    explicit_steps: 40,
                    switches: 3,
                    latency_ns: 800_000,
                },
                ProblemResult {
                    index: 1,
                    correct: true,
                    total_steps: 60,
                    latent_steps: 15,
                    explicit_steps: 45,
                    switches: 4,
                    latency_ns: 900_000,
                },
            ],
        };
        let cmp = ComparisonResult { baseline, swir };
        // Baseline: 1/2 = 50%, SwiR: 2/2 = 100%. Delta = +50pp.
        assert_eq!(cmp.accuracy_delta_pp(), 50.0);
        // Token efficiency: baseline avg = 100, swir avg = 55. Ratio = 100/55 ≈ 1.82.
        assert!((cmp.token_efficiency_ratio() - 1.818).abs() < 0.01);
    }

    #[test]
    fn test_pareto_sweep_produces_baseline_plus_cmax_points() {
        let src = SyntheticProblemSource::new(3, 42);
        let mut backend = SyntheticDecodeBackend::new(42);
        let config = BenchConfig {
            max_steps: 32,
            swir_config: SwiRConfig {
                w_e_to_l: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let points = run_pareto_sweep(&src, &mut backend, &[4, 8, 16], config);
        // 1 baseline + 3 C_max points = 4 total.
        assert_eq!(points.len(), 4);
        assert!(points[0].is_baseline);
        assert_eq!(points[0].c_max, u32::MAX);
        assert!(!points[1].is_baseline);
        assert_eq!(points[1].c_max, 4);
        assert_eq!(points[3].c_max, 16);
    }

    #[test]
    fn test_bench_result_summary_formats_correctly() {
        let result = BenchResult {
            mode: BenchMode::Swir,
            problems: vec![ProblemResult {
                index: 0,
                correct: true,
                total_steps: 50,
                latent_steps: 10,
                explicit_steps: 40,
                switches: 3,
                latency_ns: 2_000_000,
            }],
        };
        let s = result.summary();
        assert!(s.contains("Swir"));
        assert!(s.contains("accuracy=100.0%"));
        assert!(s.contains("avg_switches=3.0"));
    }
}
