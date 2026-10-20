//! # katgpt-extract
//!
//! The modelless **extraction substrate** — the Fathom F0 floor
//! (Proposal 016 / Plan 628 Phase 0): typed field extraction from text
//! with zero weights, zero output tokens, µs–ms latency, and first-class
//! abstention. This crate ships the SUBSTRATE; the assembly/serving
//! engine is the consumer repo (Phase 1, owner-gated) — never here.
//!
//! | module | feature | what it ships |
//! |---|---|---|
//! | [`span_pointer`] | `span_pointer` | candidate-window span scoring: BM25/ngram lexical + hashed-feature embedding similarity + alias tables; sigmoid normalization; exact byte offsets; zero-alloc `score_into` |
//! | [`parsers`] | `typed_parsers` | deterministic typed parsers (number/date/time/datetime/duration/list); grammar design translated from Meta Duckling (Apache-2.0) + Microsoft Recognizers-Text (MIT); constraints by construction; `reference_date`-anchored |
//! | [`estimate`], [`forecast`] | `extraction_estimate` | conformal estimate (cross-sectional kNN conditioner) + forecast (time-series) heads over the Plan-340 substrate; Report-the-Floor harness plumbing |
//! | [`temper`] | `temper` | the calibration/abstention gate: per-field residual pools, sigmoid confidence, fused abstain, null-with-guess emission |
//!
//! Default-on: `extraction_estimate` (G1 floor comparison BEATS the
//! conformal-naive floor at --release, Plan 628 T0.6); `span_pointer`,
//! `typed_parsers` and `temper` are opt-in. `--no-default-features` is the
//! tested flag-OFF posture — it compiles to an empty crate. One dep:
//! `katgpt-core` (feature-gated, `default-features = false` — this crate
//! never drags core's default feature tree into a leaf build). Pure std +
//! katgpt-core: wasm-safe by construction, zero cross-repo edges (the
//! BOUNDARY.md Owns row).
//!
//! GOAT gates (Plan 628 T0.6) bind every promotion: G1 floor comparison
//! (estimate/forecast), G2 perf bar, G4 zero-alloc, G5 external
//! BLAKE3-pinned fixture vs a rule floor (spans).

// The G4 alloc-gate machinery (the ONE `#[global_allocator]` of the lib
// test binary; test-only). Plan 628 T0.6 consolidation — see the module doc.
#[cfg(test)]
pub(crate) mod alloc_gate;

#[cfg(feature = "extraction_estimate")]
pub mod estimate;
#[cfg(feature = "extraction_estimate")]
pub mod forecast;
#[cfg(feature = "typed_parsers")]
pub mod parsers;
#[cfg(feature = "span_pointer")]
pub mod span_pointer;
#[cfg(feature = "temper")]
pub mod temper;
