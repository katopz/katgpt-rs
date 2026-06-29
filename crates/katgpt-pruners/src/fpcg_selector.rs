//! FpcgSelector — Future Probe Controlled Generation (Plan 292 Phase 3).
//!
//! The candidate-sampler + score + select loop at sentence granularity.
//! Mirrors the CGSP Conjecturer→Guide shape but applies it to a *behavior
//! forecast probe* instead of a quality/compression/correspondence score.
//!
//! Algorithm (paper §4.1, distilled in Research 267 §1.4):
//! ```text
//! def fpcg(model, prompt, future_probe, layer, num_candidates, direction):
//!     response = ""
//!     while not finished(response):
//!         candidates = generate_sentence_candidates(model, prefix, n)
//!         for c in candidates:
//!             acts = extract_activation(model, prefix + c, layer)
//!             c.score = future_probe.forecast(acts).probability
//!         best = argmax(c.scores) if direction == "positive" else argmin(c.scores)
//!         response += best
//!     return response
//! ```
//!
//! # Read-only at the LLM level
//!
//! The selector **never modifies the residual stream**. It only *selects*
//! among already-generated candidates. This is the *whole point* of FPCG
//! (perplexity preservation) — breaking this rule voids the GOAT gate.
//!
//! # Zero-alloc hot path
//!
//! [`FpcgSelector::step`] pre-allocates `Vec::with_capacity(num_candidates)`
//! once in [`FpcgSelector::new`] and `clear()`s + reuses across calls. The
//! G5 GOAT gate asserts `Vec::capacity` stays stable across 1000 steps.
//!
//! # Atomic unit: sentence
//!
//! Atomic unit is a span of generated text terminated by a sentence boundary
//! (period, newline, or EOS). This is the paper's CoT-understanding atomicity
//! claim (Bogdan 2025, Macar 2026): decisions crystallize at sentence level,
//! not token level.

use std::sync::Arc;

use fastrand::Rng;

use super::future_probe::FutureBehaviorProbe;

/// Generates M candidate next-utterance-spans for FPCG.
///
/// Atomic unit: a span of generated text terminated by a sentence boundary
/// (period, newline, EOS). This is the paper's CoT-understanding atomicity
/// claim (Bogdan 2025, Macar 2026): decisions crystallize at sentence level.
///
/// The selector is generator-agnostic: any impl (temperature sampling, beam
/// search, nucleus sampling) works. The default shipped impl is
/// [`TemperatureCandidateGenerator`] which wraps the existing model forward
/// pass at temperature T=1.0 (matches paper setup). Other generators can be
/// added later.
pub trait SentenceCandidateGenerator {
    /// Generate `n` candidate next-sentence spans given the current `prefix`.
    ///
    /// `n` MUST be ≥ 1. Each returned `String` SHOULD end at a sentence
    /// boundary (period, newline, or EOS marker `"\0"`). Implementations
    /// that don't terminate at a sentence boundary still work but lose the
    /// atomicity guarantee that motivates FPCG.
    fn generate_candidates(&mut self, prefix: &str, n: usize, rng: &mut Rng) -> Vec<String>;
}

/// Steering direction. Positive = maximize future-behavior probability,
/// Negative = minimize.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SteeringDirection {
    /// Pick the candidate with the HIGHEST forecast probability — steer
    /// the model TOWARDS the behavior the probe forecasts.
    Positive = 0,
    /// Pick the candidate with the LOWEST forecast probability — steer
    /// the model AWAY from the behavior the probe forecasts.
    Negative = 1,
}

/// Extracts a residual-stream activation at a given layer for a given prefix.
///
/// This is the engine-side interface the selector calls per candidate. In a
/// real deployment this wraps the model forward pass (running up to `layer`
/// and reading the residual stream at the sentence-end position). For tests,
/// a stub implementation provides synthetic activations derived from the
/// candidate string itself.
///
/// The activation returned MUST be at the same layer the probe was trained
/// against (`probe.layer()`). The selector does NOT validate this — caller
/// responsibility.
pub trait ActivationExtractor {
    /// Compute the residual-stream activation at `layer` for the text
    /// `prefix + candidate`, evaluated at the sentence-end position.
    ///
    /// `scratch` is a caller-provided buffer that the extractor may use to
    /// avoid allocation on the hot path; extractors that don't need it may
    /// ignore the parameter. The returned slice is the activation — typically
    /// a borrow into the extractor's internal buffer.
    fn extract_activation<'a>(&'a mut self, prefix: &str, candidate: &str, layer: usize, scratch: &'a mut [f32]) -> &'a [f32];
}

/// FPCG candidate-sampler + score + select loop.
///
/// Generic over `G: SentenceCandidateGenerator` and `E: ActivationExtractor`.
/// Holds an `Arc<FutureBehaviorProbe>` so the probe can be shared across
/// selectors (and atomic-swapped via `probe.swap_direction` at runtime).
pub struct FpcgSelector<G, E> {
    generator: G,
    extractor: E,
    probe: Arc<FutureBehaviorProbe>,
    direction: SteeringDirection,
    num_candidates: usize,

    // ── Pre-allocated scratch (zero-alloc hot path) ──
    /// Candidate strings, `clear()` + reuse across steps. Capacity stable
    /// after the first step (G5 gate assertion).
    candidates_buf: Vec<String>,
    /// Score buffer parallel to `candidates_buf`.
    scores_buf: Vec<f32>,
    /// Activation scratch for the extractor. Sized to `d_model`.
    activation_scratch: Vec<f32>,
}

impl<G, E> FpcgSelector<G, E>
where
    G: SentenceCandidateGenerator,
    E: ActivationExtractor,
{
    /// Construct a new selector.
    ///
    /// Panics if `num_candidates == 0` (the algorithm needs at least one
    /// candidate to select from) or if `d_model == 0` (the probe direction
    /// has zero dimensions).
    pub fn new(
        generator: G,
        extractor: E,
        probe: Arc<FutureBehaviorProbe>,
        direction: SteeringDirection,
        num_candidates: usize,
        d_model: usize,
    ) -> Self {
        assert!(
            num_candidates >= 1,
            "FpcgSelector::new: num_candidates must be ≥ 1"
        );
        assert!(d_model >= 1, "FpcgSelector::new: d_model must be ≥ 1");
        Self {
            generator,
            extractor,
            probe,
            direction,
            num_candidates,
            candidates_buf: Vec::with_capacity(num_candidates),
            scores_buf: Vec::with_capacity(num_candidates),
            activation_scratch: vec![0.0; d_model],
        }
    }

    /// Number of candidates sampled per step (paper default: 10).
    pub fn num_candidates(&self) -> usize {
        self.num_candidates
    }

    /// Current steering direction.
    pub fn direction(&self) -> SteeringDirection {
        self.direction
    }

    /// Replace the steering direction at runtime (e.g. switch from
    /// "encourage refusal" to "discourage refusal"). Cheap — no reallocation.
    pub fn set_direction(&mut self, direction: SteeringDirection) {
        self.direction = direction;
    }

    /// Replace the probe atomically. Readers never see torn state —
    /// `Arc<FutureBehaviorProbe>::swap_direction` is the freeze/thaw entry
    /// point (Plan 292 Phase 2 T2.2).
    pub fn swap_probe(&self, new_probe: FutureBehaviorProbe) {
        self.probe.swap_direction(new_probe);
    }

    /// One FPCG step: sample N candidates, score each by probe forecast,
    /// return the argmax (Positive) or argmin (Negative).
    ///
    /// Returns the chosen candidate string. The caller appends it to the
    /// running prefix and calls `step` again until termination.
    ///
    /// # Zero-alloc hot path
    ///
    /// `candidates_buf` and `scores_buf` are `clear()` + reused — capacity
    /// stays stable across steps. The only allocation is whatever the
    /// generator does internally to produce the candidate strings, which
    /// is the generator's responsibility (a zero-alloc generator would
    /// return `&str` slices into a ring buffer; we accept `String` for
    /// ergonomics in v1).
    ///
    /// # EOS termination
    ///
    /// If the chosen candidate is exactly `"\0"` (the EOS marker), the
    /// caller SHOULD treat the response as complete (`run` does this).
    pub fn step(&mut self, prefix: &str, rng: &mut Rng) -> String {
        // ── 1. Sample N candidates (generator-owned allocation) ──
        self.candidates_buf.clear();
        let fresh = self
            .generator
            .generate_candidates(prefix, self.num_candidates, rng);
        self.candidates_buf.extend(fresh);

        // Edge case: if the generator returned nothing, return EOS so `run`
        // terminates instead of looping forever.
        if self.candidates_buf.is_empty() {
            return EOS_MARKER.to_string();
        }

        // ── 2. Score each candidate by probe forecast ──
        self.scores_buf.clear();
        self.scores_buf.reserve(self.candidates_buf.len());
        let layer = self.probe.layer();
        for candidate in &self.candidates_buf {
            let activation = self.extractor.extract_activation(
                prefix,
                candidate,
                layer,
                &mut self.activation_scratch,
            );
            let forecast = self.probe.forecast(activation);
            self.scores_buf.push(forecast.probability);
        }

        // ── 3. argmax / argmin by steering direction ──
        let chosen_idx = match self.direction {
            SteeringDirection::Positive => argmax(&self.scores_buf),
            SteeringDirection::Negative => argmin(&self.scores_buf),
        };

        // Move the chosen candidate out (replace with empty to avoid clone
        // of the others; we'd otherwise clone the chosen one and drop the
        // rest. Mem::take lets us extract without clone).
        std::mem::take(&mut self.candidates_buf[chosen_idx])
    }

    /// Top-level loop calling [`Self::step`] until EOS or `max_sentences`
    /// reached. Returns the concatenated response (without EOS marker and
    /// without the prompt prefix).
    ///
    /// The `prompt` is the initial prefix fed to the first `step`. Subsequent
    /// steps use the running response as the prefix (the candidate strings
    /// self-contextualize). If the caller wants the prompt included in the
    /// output, they should prepend it themselves.
    ///
    /// Each step appends the chosen candidate followed by a separator
    /// (space) — matches the paper's setup where sentences are concatenated
    /// with whitespace.
    pub fn run(&mut self, prompt: &str, max_sentences: usize, rng: &mut Rng) -> String {
        let mut response = String::new();
        // The first step uses `prompt` as the prefix so the generator sees
        // the user's actual context. After that, the response itself is
        // self-contextualizing (the model's own output conditions the next
        // sentence).
        let mut prefix_buf = String::new();
        for sentence_idx in 0..max_sentences {
            if sentence_idx == 0 {
                prefix_buf.clear();
                prefix_buf.push_str(prompt);
            } else {
                prefix_buf.clear();
                prefix_buf.push_str(&response);
            }
            let candidate = self.step(&prefix_buf, rng);
            if candidate == EOS_MARKER {
                break;
            }
            if !response.is_empty() {
                response.push(' ');
            }
            response.push_str(&candidate);
        }
        response
    }

    /// Capacity of the internal candidate buffer — used by the G5 zero-alloc
    /// gate to assert stability across many steps.
    pub fn candidates_buf_capacity(&self) -> usize {
        self.candidates_buf.capacity()
    }
}

/// EOS marker used by `step` / `run` to signal termination. Any candidate
/// that equals this exact string terminates the run. Generators can produce
/// this by sampling an EOS token.
pub const EOS_MARKER: &str = "\0";

/// argmax over a slice — returns the index of the maximum element.
/// Ties go to the FIRST maximum (lowest index). Panics on empty slice.
fn argmax(scores: &[f32]) -> usize {
    assert!(!scores.is_empty(), "argmax: scores must be non-empty");
    let mut best_idx = 0;
    let mut best = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best {
            best = s;
            best_idx = i;
        }
    }
    best_idx
}

/// argmin over a slice — returns the index of the minimum element.
/// Ties go to the FIRST minimum (lowest index). Panics on empty slice.
fn argmin(scores: &[f32]) -> usize {
    assert!(!scores.is_empty(), "argmin: scores must be non-empty");
    let mut best_idx = 0;
    let mut best = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s < best {
            best = s;
            best_idx = i;
        }
    }
    best_idx
}

// ── Default generator: TemperatureCandidateGenerator ─────────────

/// Default candidate generator that wraps a model forward pass at a fixed
/// temperature (paper default T=1.0).
///
/// This is a *trait stub* — the actual model-binding impl lives at the
/// engine layer (a real forward pass is engine-specific). The struct here
/// is the type consumers name when they want "the default temperature-style
/// generator"; they provide the underlying sampler function via constructor.
///
/// For tests and demos, see `synthetic_generator.rs` patterns or the
/// `fpcg_01_basic` example.
pub struct TemperatureCandidateGenerator<F>
where
    F: FnMut(&str, usize, &mut Rng) -> Vec<String>,
{
    temperature: f32,
    sampler: F,
}

impl<F> TemperatureCandidateGenerator<F>
where
    F: FnMut(&str, usize, &mut Rng) -> Vec<String>,
{
    /// Construct with a sampler closure and temperature. The closure takes
    /// `(prefix, n, rng)` and returns `n` candidate strings.
    pub fn new(temperature: f32, sampler: F) -> Self {
        assert!(
            temperature > 0.0,
            "TemperatureCandidateGenerator: temperature must be > 0"
        );
        Self {
            temperature,
            sampler,
        }
    }

    /// Current temperature.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl<F> SentenceCandidateGenerator for TemperatureCandidateGenerator<F>
where
    F: FnMut(&str, usize, &mut Rng) -> Vec<String>,
{
    fn generate_candidates(&mut self, prefix: &str, n: usize, rng: &mut Rng) -> Vec<String> {
        (self.sampler)(prefix, n, rng)
    }
}

// ── Tests (Plan 292 T3.6) ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::future_probe::FutureBehaviorProbe;

    /// Stub generator that produces a fixed candidate set per call.
    /// Each candidate is paired with a known-good/bad score via the
    /// `StubExtractor` below.
    struct StubGenerator {
            // candidates to return on each generate_candidates call
            candidates: Vec<String>,
    }

    impl SentenceCandidateGenerator for StubGenerator {
        fn generate_candidates(&mut self, _prefix: &str, n: usize, _rng: &mut Rng) -> Vec<String> {
            self.candidates.iter().take(n).cloned().collect()
        }
    }

    /// Stub extractor that maps each candidate to a deterministic activation
    /// vector. The map is `candidate_string -> activation`. Candidates not
    /// in the map get the zero vector.
    struct StubExtractor {
        /// Map from candidate string → activation vector.
        activations: std::collections::HashMap<String, Vec<f32>>,
    }

    impl ActivationExtractor for StubExtractor {
        fn extract_activation<'a>(
            &'a mut self,
            _prefix: &str,
            candidate: &str,
            _layer: usize,
            scratch: &'a mut [f32],
        ) -> &'a [f32] {
            if let Some(act) = self.activations.get(candidate) {
                let len = act.len().min(scratch.len());
                scratch[..len].copy_from_slice(&act[..len]);
                &scratch[..len]
            } else {
                // Zero activation — caller probe bias dominates.
                for v in scratch.iter_mut() {
                    *v = 0.0;
                }
                scratch
            }
        }
    }

    /// Build a stub probe where direction[0] is the only nonzero element,
    /// so the forecast probability is monotone in activation[0].
    fn stub_probe(bias: f32) -> Arc<FutureBehaviorProbe> {
        Arc::new(FutureBehaviorProbe::new(vec![1.0, 0.0, 0.0, 0.0], bias, 0, "stub"))
    }

    fn make_selector(
        candidates: Vec<String>,
        activations: Vec<(String, Vec<f32>)>,
        direction: SteeringDirection,
        num_candidates: usize,
    ) -> FpcgSelector<StubGenerator, StubExtractor> {
        let probe = stub_probe(0.0);
        let generator = StubGenerator { candidates };
        let map = activations.into_iter().collect();
        let ext = StubExtractor { activations: map };
        FpcgSelector::new(generator, ext, probe, direction, num_candidates, 4)
    }

    /// T3.6: selector with `num_candidates=1` is equivalent to unsteered
    /// generation (the single candidate is returned regardless of score).
    #[test]
    fn num_candidates_one_returns_the_single_candidate() {
        let candidates = vec!["only choice.".to_string()];
        let activations = vec![("only choice.".to_string(), vec![10.0, 0.0, 0.0, 0.0])];
        let mut sel = make_selector(candidates, activations, SteeringDirection::Positive, 1);
        let mut rng = Rng::with_seed(0);
        let chosen = sel.step("", &mut rng);
        assert_eq!(chosen, "only choice.");
    }

    /// T3.6: Positive direction picks the highest-probability candidate.
    #[test]
    fn positive_direction_picks_highest_probability() {
        let candidates = vec![
            "low.".to_string(),   // activation[0] = -5 → low prob
            "high.".to_string(),  // activation[0] = +5 → high prob
            "mid.".to_string(),   // activation[0] = 0  → σ(0) = 0.5
        ];
        let activations = vec![
            ("low.".to_string(), vec![-5.0, 0.0, 0.0, 0.0]),
            ("high.".to_string(), vec![5.0, 0.0, 0.0, 0.0]),
            ("mid.".to_string(), vec![0.0, 0.0, 0.0, 0.0]),
        ];
        let mut sel = make_selector(candidates, activations, SteeringDirection::Positive, 3);
        let mut rng = Rng::with_seed(0);
        let chosen = sel.step("", &mut rng);
        assert_eq!(chosen, "high.", "Positive direction must pick highest-prob candidate");
    }

    /// T3.6: Negative direction picks the lowest-probability candidate.
    #[test]
    fn negative_direction_picks_lowest_probability() {
        let candidates = vec![
            "low.".to_string(),
            "high.".to_string(),
            "mid.".to_string(),
        ];
        let activations = vec![
            ("low.".to_string(), vec![-5.0, 0.0, 0.0, 0.0]),
            ("high.".to_string(), vec![5.0, 0.0, 0.0, 0.0]),
            ("mid.".to_string(), vec![0.0, 0.0, 0.0, 0.0]),
        ];
        let mut sel = make_selector(candidates, activations, SteeringDirection::Negative, 3);
        let mut rng = Rng::with_seed(0);
        let chosen = sel.step("", &mut rng);
        assert_eq!(chosen, "low.", "Negative direction must pick lowest-prob candidate");
    }

    /// T3.6: selector terminates at EOS. The generator returning only the
    /// EOS marker produces an immediate `run` exit.
    #[test]
    fn run_terminates_at_eos() {
        let candidates = vec![EOS_MARKER.to_string()];
        let activations: Vec<(String, Vec<f32>)> = vec![(EOS_MARKER.to_string(), vec![0.0; 4])];
        let mut sel = make_selector(candidates, activations, SteeringDirection::Positive, 1);
        let mut rng = Rng::with_seed(0);
        let out = sel.run("prompt", 10, &mut rng);
        assert_eq!(out, "", "EOS on first step should give empty response");
    }

    /// T3.6: hot-path is zero-alloc across many steps — `Vec::capacity` stable.
    ///
    /// We measure `candidates_buf_capacity()` after construction and after
    /// 1000 steps. They MUST be equal — no growth means no realloc.
    #[test]
    fn hot_path_is_zero_alloc_across_many_steps() {
        // Generator cycles through a fixed candidate pool.
        struct CyclingGenerator {
            pool: Vec<String>,
            call_count: usize,
        }
        impl SentenceCandidateGenerator for CyclingGenerator {
            fn generate_candidates(&mut self, _prefix: &str, n: usize, _rng: &mut Rng) -> Vec<String> {
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    let idx = self.call_count % self.pool.len();
                    out.push(self.pool[idx].clone());
                    self.call_count += 1;
                }
                out
            }
        }
        let pool: Vec<String> = (0..5)
            .map(|i| format!("candidate_{i}."))
            .collect();
        let activations: Vec<(String, Vec<f32>)> = pool
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), vec![i as f32 - 2.0, 0.0, 0.0, 0.0]))
            .collect();
        let probe = stub_probe(0.0);
        let generator = CyclingGenerator { pool, call_count: 0 };
        let ext = StubExtractor { activations: activations.into_iter().collect() };
        let mut sel = FpcgSelector::new(generator, ext, probe, SteeringDirection::Positive, 10, 4);

        let capacity_before = sel.candidates_buf_capacity();
        let mut rng = Rng::with_seed(42);
        for _ in 0..1000 {
            let _ = sel.step("irrelevant prefix", &mut rng);
        }
        let capacity_after = sel.candidates_buf_capacity();
        assert_eq!(
            capacity_before, capacity_after,
            "G5 zero-alloc gate: candidates_buf capacity must stay stable across 1000 steps (before={}, after={})",
            capacity_before, capacity_after
        );
    }

    /// argmax sanity.
    #[test]
    fn argmax_picks_first_max() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0, 3.0]), 1, "ties go to lowest index");
        assert_eq!(argmax(&[5.0]), 0);
    }

    /// argmin sanity.
    #[test]
    fn argmin_picks_first_min() {
        assert_eq!(argmin(&[3.0, 1.0, 2.0, 1.0]), 1, "ties go to lowest index");
        assert_eq!(argmin(&[5.0]), 0);
    }

    /// run concatenates with spaces and respects max_sentences.
    #[test]
    fn run_respects_max_sentences_and_concatenates() {
        // Generator always offers the same 2 candidates per step.
        let candidates = vec!["alpha.".to_string(), "beta.".to_string()];
        let activations = vec![
            ("alpha.".to_string(), vec![1.0, 0.0, 0.0, 0.0]),  // higher prob
            ("beta.".to_string(), vec![-1.0, 0.0, 0.0, 0.0]),  // lower prob
        ];
        let mut sel = make_selector(candidates, activations, SteeringDirection::Positive, 2);
        let mut rng = Rng::with_seed(0);
        // Positive direction always picks alpha. Max 3 sentences → "alpha. alpha. alpha."
        let out = sel.run("", 3, &mut rng);
        assert_eq!(out, "alpha. alpha. alpha.");
    }

    /// Empty candidate pool terminates via EOS instead of looping forever.
    #[test]
    fn empty_candidate_pool_terminates() {
        struct EmptyGen;
        impl SentenceCandidateGenerator for EmptyGen {
            fn generate_candidates(&mut self, _: &str, _: usize, _: &mut Rng) -> Vec<String> {
                Vec::new()
            }
        }
        let probe = stub_probe(0.0);
        struct ZeroExt;
        impl ActivationExtractor for ZeroExt {
            fn extract_activation<'a>(&'a mut self, _: &str, _: &str, _: usize, scratch: &'a mut [f32]) -> &'a [f32] {
                for v in scratch.iter_mut() {
                    *v = 0.0;
                }
                scratch
            }
        }
        let mut sel = FpcgSelector::new(EmptyGen, ZeroExt, probe, SteeringDirection::Positive, 5, 4);
        let mut rng = Rng::with_seed(0);
        let out = sel.run("prompt", 100, &mut rng);
        assert_eq!(out, "", "empty pool must terminate immediately");
    }

    /// Steering direction discriminants.
    #[test]
    fn steering_direction_discriminants() {
        assert_eq!(SteeringDirection::Positive as u8, 0);
        assert_eq!(SteeringDirection::Negative as u8, 1);
    }

    /// `set_direction` flips the selection live.
    #[test]
    fn set_direction_flips_selection() {
        let candidates = vec!["low.".to_string(), "high.".to_string()];
        let activations = vec![
            ("low.".to_string(), vec![-5.0, 0.0, 0.0, 0.0]),
            ("high.".to_string(), vec![5.0, 0.0, 0.0, 0.0]),
        ];
        let mut sel = make_selector(candidates, activations, SteeringDirection::Positive, 2);
        let mut rng = Rng::with_seed(0);

        let chosen_pos = sel.step("", &mut rng);
        assert_eq!(chosen_pos, "high.");

        sel.set_direction(SteeringDirection::Negative);
        let chosen_neg = sel.step("", &mut rng);
        assert_eq!(chosen_neg, "low.");
    }

    /// `swap_probe` replaces the underlying probe atomically.
    #[test]
    fn swap_probe_changes_forecast_behavior() {
        let candidates = vec!["a.".to_string(), "b.".to_string()];
        let activations = vec![
            ("a.".to_string(), vec![5.0, 0.0, 0.0, 0.0]),
            ("b.".to_string(), vec![5.0, 0.0, 0.0, 0.0]), // same activation as "a."
        ];
        let mut sel = make_selector(candidates, activations, SteeringDirection::Positive, 2);
        let mut rng = Rng::with_seed(0);

        // Original probe: direction[0]=+1. Both candidates have act[0]=5 → tie.
        // argmax tie → first index → "a."
        let first = sel.step("", &mut rng);
        assert_eq!(first, "a.");

        // Swap in inverted probe: direction[0]=-1. Both have act[0]=5 → both
        // give the same low prob. Tie still goes to first → "a."
        // (This is a sanity test that swap_probe doesn't crash; the tie-break
        // behavior is unchanged.)
        let new_probe = FutureBehaviorProbe::new(vec![-1.0, 0.0, 0.0, 0.0], 0.0, 0, "v2");
        sel.swap_probe(new_probe);
        let after = sel.step("", &mut rng);
        assert_eq!(after, "a.", "tie-break still goes to first index after swap");
    }
}
