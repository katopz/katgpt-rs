//! Numeric-deviation contextualization probe (arXiv:2405.02803 "Is Flash
//! Attention Stable?", Golden et al., Meta FAIR + Harvard 2024) — Issue 697
//! Phase 1 ([`probe`]).
//!
//! The stack's kernel numeric gates pin hand-picked absolute/relative bands
//! (q8kv `5e-3`, parity `1e-2`/`2e-2`, the Bench-773 argmax+max_abs form).
//! This module ships the paper's alternative — a **contextualization
//! acceptance rule**: a variant's numeric deviation (the two-surface
//! [`DeviationReport`]: elementwise `max_diff` + 1-D Wasserstein) is accepted
//! iff it is dominated by **reference bands the system demonstrably
//! tolerates**:
//!
//! - **R1** — divergence between two draws of the same init distribution
//!   (`reference_r1_two_draws`). The system already tolerates this much
//!   deviation: re-seeding alone produces it.
//! - **R2** — divergence from a precision change, proxied by a
//!   quantize→dequant round-trip (`reference_r2_roundtrip`,
//!   `reference_r2_custom`). **Label: single-step lower bound.** A faithful
//!   trained R2 reference needs training runs; the shipped proxy captures
//!   ONE quantization event, so it lower-bounds real precision-change
//!   divergence. The label is pinned by `r2_lower_bound_label_tripwire`.
//!
//! # Scope limit
//!
//! **Scope limit: this protocol bounds DIVERGENCE SIMILARITY only — it is
//! NOT a training-stability proof.** The paper explicitly declines the
//! stability link ("ultimately linking this numeric deviation back to
//! training instability requires further investigation"); the stability
//! mechanism is owned by arXiv:2510.04212 (low-rank representation emergence
//! × biased rounding in low-precision Flash Attention). No API here may be
//! cited as evidence that a kernel/dtype is "training stable". Pinned by
//! `scope_limit_tripwire`.
//!
//! # The margin has no default
//!
//! [`accept`] takes `margin` as an EXPLICIT parameter — there is **no
//! default margin**, and none may be derived from the paper's headline
//! "2–5×" (that constant is context-specific to Meta's model / seq-len /
//! hardware; the durable extraction is the dominance rule, not the number).
//! Pinned by `margin_has_no_default_doc_truth`.
//!
//! # Substrate reuse (no duplication)
//!
//! The 1-D Wasserstein delegates to `crate::mag::transfer` (Plan 418): both
//! compute paths call the substrate's shared quantile-grid core
//! (`wasserstein1d_sorted_core` + `quantile_interp`, private to that
//! module) through `wasserstein1d_scalar_into`. The distance is therefore
//! the substrate's quantile-grid W1 definition (empirical quantile
//! functions compared on a common `max(m, n)`-point grid with linear
//! interpolation) — not the textbook order-statistic form. Identical
//! inputs give identical values by construction.
//!
//! # NaN / determinism policy
//!
//! - Non-finite tensor inputs are REJECTED at the boundary
//!   ([`NumericStabilityError::NonFinite`]) — garbage never becomes a
//!   report. Because inputs are validated finite before the metric runs,
//!   the substrate sort's comparator is exercised only on a total order →
//!   sorted-quantile determinism with no ambiguous NaN reordering on our
//!   paths (this is the documented policy; the substrate comparator itself
//!   is unchanged for its own MAG callers).
//! - `max_diff` may saturate to `+inf` for opposite-signed near-`f32::MAX`
//!   operands (finite inputs, overflowing difference). Non-finite REPORT
//!   values fail closed to `Reject` in [`accept`].
//! - No HashMap iteration, no wall-clock, no parallelism: every fn is a
//!   pure function of its arguments; equal inputs give bit-identical
//!   reports, and reports are symmetric (`compute(x, y)` ≡ `compute(y, x)`
//!   bit-exactly). Pinned by test.
//! - Sorting rides std's IN-PLACE unstable sort (deterministic algorithm,
//!   no merge-buffer allocation — that is what makes the hot path
//!   alloc-free). The metric reads only the sorted VALUE sequence, in which
//!   equal `f32`s are bit-identical and interchangeable, so results are
//!   bit-identical to a stable sort's; the in-test reference twin uses a
//!   stable `total_cmp` sort and pins the equality.
//!
//! # Zero-alloc convention
//!
//! Hot paths take caller scratch (`*_into`, grow-only `&mut Vec<f32>` sort
//! buffers — steady-state zero-alloc once capacity covers the largest
//! tensor, the Plan 418 pattern). Reference-band BUILDERS are cold path
//! (one allocation) by design. `accept` allocates nothing (pure
//! comparisons).
//!
//! Phase 1 ships T1.1–T1.5 + the T3.1 planted-deviation gate. Phase 2 — the
//! perturbable reference attention lab ([`lab`]: tile-shape / dim-order /
//! mantissa / seq-len knobs, the four ordering laws pinned as tests, and the
//! T3.2 `tol(S)` pinned schedule) — landed 2026-08-29. T3.3 (consumer
//! follow-ups) files on first consumption (substrate-first: the riir-ai gate
//! layer consumes; do not fork the probe). The riir-train side (drift probe
//! + divergence ledger) is riir-train Issue 492.
//!
//! Numeric-deviation contextualization probe (Issue 697 Phase 1 / Research
//! 515, arXiv:2405.02803 "Is Flash Attention Stable?"): the f64
//! mantissa-truncation format emulator + the two-surface DeviationReport
//! (elementwise max_diff + 1-D Wasserstein delegated to mag::transfer's
//! quantile-grid core) + the reference-band acceptance rule — R1 two-draw
//! init divergence, R2 quantize→dequant round-trip labeled a SINGLE-STEP
//! LOWER BOUND (doc-truth tripwired). The margin is an explicit caller
//! parameter: the paper's 2–5× is context-specific, never a default. Scope
//! limit: divergence similarity, NOT training stability (arXiv:2510.04212
//! owns the mechanism). Zero-alloc *_into hot paths; no new deps; NaN
//! rejected at the boundary. Opt-in — Phase 2 (perturbable attention lab) +
//! T3.2/T3.3 open; first consumer: riir-ai gate layer; riir-train: Issue 492.

pub mod lab;
pub mod probe;

pub use lab::{
    AxisSwap, LAB_SEED, LabConfig, PINNED_F64_MULTITILE_REL, PINNED_SPEARMAN_FLOOR, TOL_FIT_BAND,
    TOL_FIT_HASH_PINNED, TOL_FIT_PINNED, TOL_HEADROOM, TOL_TABLE_PINNED, TolFitInputs, band_at,
    deviation_report, fill_lab_inputs, lab_attention, lcg_next_f32, max_abs_diff_f64,
    max_rel_diff_f64, naive_attention_f64, run_pair, spearman_rho, tol_fit_inputs_hash,
};
pub use probe::{
    DeviationReport, F64_MANTISSA_BITS, NumericStabilityError, R2_LABEL, ReferenceBands, Verdict,
    accept, reference_r1_two_draws, reference_r2_custom, reference_r2_roundtrip,
    roundtrip_truncate_mantissa, roundtrip_truncate_mantissa_into, truncate_mantissa,
};
