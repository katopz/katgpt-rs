//! The Fathom extraction wire contract — Plan 628 T0.1, the katgpt-rs
//! public half of Proposal 016 (the extraction stack).
//!
//! The contract is the typed field-extraction vocabulary the F0 floor (and
//! every rung above it) speaks, mirroring the `decision_wire` shape one
//! module over: request = `state` + typed field specs, response = one
//! answer per field, abstention first-class.
//!
//! ## The vocabulary
//!
//! Nine field kinds — Clairic's class set adopted (Research 627 §2):
//! `text` (grounded spans), `number`/`date`/`time`/`datetime`/`duration`
//! (typed scalars, grammar-parsed), `list` (typed elements, input order),
//! `estimate` (cross-sectional conformal distribution) and `forecast`
//! (time-series per-step distribution). Each spec may carry `constraints`
//! (enforced by construction: integer/decimals/unit/range/element
//! kind/choices/grid/horizon) and free-text `instructions`.
//!
//! ## The disclosed F0 limit
//!
//! `instructions` are CARRIED but NOT READ by the modelless floor — a
//! weightless scorer cannot parse "the vendor's address, not the shipping
//! one" (Research 627 scoreboard LOSS row). What F0 reads is
//! schema/shape/`choices`/positional signal. Instruction-conditioned
//! extraction lands at F1 (dictionaries/prompt tables) and F2 (hosted
//! lane); the wire carries the field for every rung so no version bump is
//! owed when they arrive.
//!
//! ## Abstention + the guess law
//!
//! `null` is a first-class answer: `value: None` + `p_null` (the
//! calibrated probability the field is absent/unanswerable). A low-
//! confidence guess may ride a null as `guess: Some(..)` — Clairic's
//! honest null-with-guess adopted verbatim. The laws, fail-closed:
//! a guess never coexists with an answer, a guess never carries a span
//! (a grounded answer carries its exact input slice; a guess is by
//! definition ungrounded), and `estimate`/`forecast` answers never carry
//! spans (they are computed from evidence, not located in the text).
//!
//! ## Versioning — additive-only
//!
//! `contract_version` rides every request and is echoed on every response.
//! Absent = 1 (serde default). Evolution is ADDITIVE ONLY: new optional
//! fields join at the current version; any semantic break bumps the
//! version, and a decoder that does not know a version refuses loud
//! ([`ExtractionRequest::validate`] →
//! [`WireError::UnknownContractVersion`]) — never silently guesses. The
//! response echoes the request's version so a mismatched pair is
//! detectable at the edge.
//!
//! ## Scope
//!
//! WIRE ONLY — pure data + fail-closed structural validation. No engine
//! logic, no inference (the F0 floor lives in `katgpt-extract`; the
//! serving edge in the consumer). Serde (json + postcard) round-trips and
//! a golden JSON byte-pin ride in this module. Zero new deps — serde and
//! postcard are already non-optional. Opt-in (`extraction_wire`) per the
//! no-default-consumer rule — the consumer is the Proposal 016 Phase-1
//! assembly; promotion rides the GOAT gate.

use serde::{Deserialize, Serialize};

/// The contract version this module speaks. Additive-only evolution: this
/// bumps ONLY on a semantic break, and every bump refuses older/newer
/// peers loud (never a silent best-effort decode).
pub const CONTRACT_VERSION: u32 = 1;

fn default_contract_version() -> u32 {
    CONTRACT_VERSION
}

/// The nine typed field kinds (Clairic's class set, Research 627 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
pub enum FieldKind {
    /// Free text located as a span (names, addresses, titles).
    Text,
    /// A number (integer or decimal; unit via constraints).
    Number,
    /// A calendar date (ISO-8601 `YYYY-MM-DD` on the wire).
    Date,
    /// A wall-clock time (ISO-8601 `HH:MM[:SS]` on the wire).
    Time,
    /// A date+time (ISO-8601, `T` or space separator).
    DateTime,
    /// A duration (canonical unit + value on the wire).
    Duration,
    /// An ordered list of typed elements, in input order.
    List,
    /// A cross-sectional estimate: a conformal distribution conditioned on
    /// the state's evidence (comparables), not a located value.
    Estimate,
    /// A time-series forecast: per-step distributions over a horizon.
    Forecast,
}

impl FieldKind {
    /// The wire spelling (also the serde tag).
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Date => "date",
            Self::Time => "time",
            Self::DateTime => "datetime",
            Self::Duration => "duration",
            Self::List => "list",
            Self::Estimate => "estimate",
            Self::Forecast => "forecast",
        }
    }

    /// Parse the wire spelling (the unknown-field refusal surface for
    /// callers building requests from untyped config).
    #[inline]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "text" => Some(Self::Text),
            "number" => Some(Self::Number),
            "date" => Some(Self::Date),
            "time" => Some(Self::Time),
            "datetime" => Some(Self::DateTime),
            "duration" => Some(Self::Duration),
            "list" => Some(Self::List),
            "estimate" => Some(Self::Estimate),
            "forecast" => Some(Self::Forecast),
            _ => None,
        }
    }

    /// Kinds whose answers are COMPUTED from evidence rather than located
    /// in the text — they never carry spans.
    #[inline]
    pub fn is_computed(self) -> bool {
        matches!(self, Self::Estimate | Self::Forecast)
    }
}

/// The quantization grid an `estimate` answer is rendered onto (Clairic's
/// caller-grid law: the caller names the units/shape, the engine quantizes
/// onto it).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
pub enum GridKind {
    /// Evenly spaced steps (`step` in linear units).
    Linear,
    /// Evenly spaced steps in log space (`step` is a ratio, e.g. `1.1`).
    Log,
}

/// A quantization grid: kind + positive step.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    pub kind: GridKind,
    /// Linear units per step, or the log-space ratio per step. MUST be
    /// finite and `> 1.0` for [`GridKind::Log`], `> 0.0` for linear.
    pub step: f64,
}

/// Per-field constraints — enforced BY CONSTRUCTION (a parser that cannot
/// satisfy them abstains; the engine never clamps a wrong answer into
/// range). Always serialized (never skipped): postcard is positional — a
/// skipped field cannot be decoded back (the `ptg_functor_edges` law).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FieldConstraints {
    /// `number`: the value must parse as an integer (no fractional part).
    #[serde(default)]
    pub integer: bool,
    /// `number`: maximum decimal places (rejected, never rounded to).
    #[serde(default)]
    pub decimals: Option<u8>,
    /// `number`/`duration`: the required unit spelling (e.g. `"km"`,
    /// `"h"`); a value in any other unit abstains.
    #[serde(default)]
    pub unit: Option<String>,
    /// `number`/`estimate`: inclusive lower bound; outside → abstain.
    #[serde(default)]
    pub min: Option<f64>,
    /// `number`/`estimate`: inclusive upper bound; outside → abstain.
    #[serde(default)]
    pub max: Option<f64>,
    /// `list`: the element kind (validated: only legal when the field kind
    /// is [`FieldKind::List`]).
    #[serde(default)]
    pub element_kind: Option<FieldKind>,
    /// `text`: the F0-readable vocabulary — the closed set of acceptable
    /// answers the span scorer matches against state text. This is the
    /// schema signal instructions were supposed to carry (see the module
    /// doc's disclosed F0 limit).
    #[serde(default)]
    pub choices: Vec<String>,
    /// `estimate`: the grid answers quantize onto.
    #[serde(default)]
    pub grid: Option<Grid>,
    /// `forecast`: the horizon in steps (REQUIRED for forecast fields).
    #[serde(default)]
    pub horizon: Option<u32>,
}

/// One typed field spec. The answer space is part of the REQUEST (the
/// `decision_wire` law, extraction-flavored).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldSpec {
    /// Stable id; the response's answers echo it (the alignment pin).
    pub id: String,
    pub kind: FieldKind,
    /// What the field means — carried for every rung; NOT read by F0
    /// (the disclosed limit above). Always serialized.
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub constraints: FieldConstraints,
    /// Free-text conditioning ("the vendor's billing address"). Carried
    /// for F1/F2; NOT read by F0. Always serialized.
    #[serde(default)]
    pub instructions: Option<String>,
}

/// A byte-offset span into the request's `state` — `[start, end)`. The
/// provenance primitive: offsets are LITERAL input slices, so the answer
/// is grounded by construction (never a paraphrase, never a re-render).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: u32,
    /// Exclusive end byte offset. MUST satisfy `start <= end <=
    /// state.len()` and land on char boundaries.
    pub end: u32,
}

/// A duration in one canonical unit. Engines normalize ("90 minutes" →
/// unit `min`, value 90; "1.5 hours" → unit `h`, value 1.5 — unit choice
/// is the parser's; the wire validates the spelling, not the choice).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurationValue {
    /// Canonical spelling: `s` | `min` | `h` | `d` | `w` | `mo` | `y`.
    pub unit: String,
    /// Magnitude in `unit`s — finite, `>= 0`.
    pub value: f64,
}

/// The canonical duration units the wire accepts.
pub const DURATION_UNITS: [&str; 7] = ["s", "min", "h", "d", "w", "mo", "y"];

/// A cross-sectional estimate: the conformal distribution over the field's
/// value conditioned on the state's evidence. Quantiles ride ONE interval
/// level (the field's declared mass, e.g. 0.8); `p_below`/`p_above` are
/// the probability mass outside the caller's grid/units (Clairic's honest
/// out-of-grid disclosure).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimateDistribution {
    /// The point prediction (median of the distribution).
    pub point: f64,
    /// Lower quantile at the declared level.
    pub p10: f64,
    /// The median (== `point` by construction; carried so consumers read
    /// one shape).
    pub median: f64,
    /// Upper quantile at the declared level.
    pub p90: f64,
    /// Probability mass below the caller's grid/domain. `[0, 1]`.
    pub p_below: f64,
    /// Probability mass above the caller's grid/domain. `[0, 1]`.
    pub p_above: f64,
    /// The interval mass (`p90 - p10` coverage aim), `(0, 1]`.
    pub level: f32,
}

/// One forecast step: horizon `h` (1-based, strictly increasing across the
/// answer) + the point + the interval at the declared level.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForecastStep {
    /// Step index, 1-based (h=1 is the next step).
    pub h: u32,
    pub point: f64,
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
}

/// A time-series forecast: per-step conformal distributions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForecastDistribution {
    /// One step per horizon element; `h` strictly increasing from 1.
    pub steps: Vec<ForecastStep>,
    /// The interval mass, `(0, 1]`.
    pub level: f32,
}

/// A typed extracted value. Externally tagged (postcard-safe), lowercase
/// tags matching [`FieldKind::as_str`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldValue {
    Text(String),
    Number(f64),
    /// ISO-8601 date (`YYYY-MM-DD`).
    Date(String),
    /// ISO-8601 time (`HH:MM` or `HH:MM:SS`).
    Time(String),
    /// ISO-8601 datetime (`YYYY-MM-DDTHH:MM[:SS]`; space separator also
    /// legal on the wire).
    DateTime(String),
    Duration(DurationValue),
    List(Vec<FieldValue>),
    Estimate(EstimateDistribution),
    Forecast(ForecastDistribution),
}

/// One answer per field. `value: None` IS an answer (abstention
/// first-class); `guess` may ride a null with the engine's low-confidence
/// candidate disclosed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldAnswer {
    /// Echoes the request field's id (the alignment pin).
    pub id: String,
    /// Echoes the field's kind (the cheap kind-agreement check).
    pub kind: FieldKind,
    /// The answer — `None` = abstain (null is an answer).
    #[serde(default)]
    pub value: Option<FieldValue>,
    /// The grounded input slice for located answers. Laws: a span never
    /// rides a null; a span NEVER rides `estimate`/`forecast` (computed,
    /// not located); a located answer (`text`..`list`) ALWAYS carries one.
    #[serde(default)]
    pub span: Option<Span>,
    /// Calibrated probability the field is absent/unanswerable. `[0, 1]`.
    pub p_null: f32,
    /// The scalar confidence readout (the Bench-817 dispatch lives in the
    /// engine; this wire only carries the result). `[0, 1]`.
    pub confidence: f32,
    /// The disclosed low-confidence guess riding a null. Laws: never
    /// coexists with `value: Some`, never carries a span, kind must match.
    #[serde(default)]
    pub guess: Option<FieldValue>,
}

/// The extraction request: one state, ALL fields answered in one batch
/// (the `decision_wire` "all questions in the call" law — no per-field
/// round trips; the F0 batch is µs-class, so no multi-pass splitting).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractionRequest {
    /// Absent = 1 (serde default). Unknown versions refuse loud.
    #[serde(default = "default_contract_version")]
    pub contract_version: u32,
    /// The state under extraction — text/email/ticket/table-as-text/JSON.
    pub state: String,
    pub fields: Vec<FieldSpec>,
    /// The anchor for relative date/time expressions ("tomorrow", "next
    /// monday") — ISO-8601 date. Absent = the engine's clock, which
    /// breaks replay determinism: callers who care SET THIS.
    #[serde(default)]
    pub reference_date: Option<String>,
}

/// The extraction response: one answer per field, same order, version
/// echoed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractionResponse {
    /// Echoes the request's `contract_version` (the pair check).
    #[serde(default = "default_contract_version")]
    pub contract_version: u32,
    pub answers: Vec<FieldAnswer>,
}

/// Fail-closed structural validation errors. Copy so callers can hold the
/// verdict alongside the payload; `at` indexes into the request's
/// `fields` / the response's `answers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireError {
    /// The request/response names a version this module does not speak.
    UnknownContractVersion { got: u32, supported: u32 },
    /// The response's version ≠ the request's.
    VersionMismatch { request: u32, response: u32 },
    /// A field/answer id is empty (or whitespace).
    EmptyId { at: usize },
    /// Two fields share an id; `at` is the later occurrence.
    DuplicateFieldId { at: usize },
    /// Two constraints that cannot coexist (integer + decimals,
    /// `element_kind` off-list, `grid` on a non-estimate, `horizon` on a
    /// non-forecast).
    ConstraintConflict { at: usize },
    /// A constraint value is malformed (non-finite min/max, zero/negative
    /// step, log step ≤ 1, horizon 0, missing forecast horizon).
    BadConstraint { at: usize },
    /// A `choices` entry is empty/whitespace.
    EmptyChoice { at: usize },
    /// `reference_date` is present but not an ISO-8601 date.
    BadReferenceDate,
    /// Answer count ≠ field count.
    AnswerCountMismatch { answers: usize, fields: usize },
    /// `answers[at].id` ≠ `fields[at].id`.
    AnswerIdMismatch { at: usize },
    /// `answers[at].kind` ≠ `fields[at].kind`.
    AnswerKindMismatch { at: usize },
    /// The value variant disagrees with the field kind.
    ValueKindMismatch { at: usize },
    /// A numeric payload is NaN or infinite.
    ValueNotFinite { at: usize },
    /// The value violates `min`/`max`/`decimals`.
    ConstraintViolated { at: usize },
    /// Not an ISO-8601 date where one is required.
    BadDateFormat { at: usize },
    /// Not an ISO-8601 time where one is required.
    BadTimeFormat { at: usize },
    /// Not an ISO-8601 datetime where one is required.
    BadDateTimeFormat { at: usize },
    /// A duration unit outside [`DURATION_UNITS`], or a negative value.
    BadDuration { at: usize },
    /// A list element fails the declared `element_kind` (or nests a list).
    BadListElement { at: usize, index: usize },
    /// A distribution's quantiles are unordered, its level is outside
    /// `(0, 1]`, or its tails sum past 1.
    DistributionMalformed { at: usize },
    /// A forecast has no steps, unordered horizons, or a bad level.
    ForecastMalformed { at: usize },
    /// A span rode an abstention (`span: Some`, `value: None`).
    SpanWithoutValue { at: usize },
    /// A computed answer carried a span (estimate/forecast are located by
    /// evidence, not by text position).
    SpanOnComputedAnswer { at: usize },
    /// A located answer (non-computed kind) carried no span.
    ValueWithoutSpan { at: usize },
    /// A guess rode an answer (`guess: Some`, `value: Some`).
    GuessWithAnswer { at: usize },
    /// A guess carried a span — guesses are ungrounded by definition.
    GuessWithSpan { at: usize },
    /// `start > end` or `end > state.len()`.
    SpanOutOfBounds { at: usize },
    /// A span edge is not a char boundary of the state.
    SpanNotCharBoundary { at: usize },
    /// `p_null`/`confidence` is NaN or infinite.
    ProbabilityNotFinite { at: usize },
    /// `p_null`/`confidence` is outside `[0, 1]`.
    ProbabilityOutOfRange { at: usize },
}

/// Structural ISO-8601 date check: `YYYY-MM-DD`, plausible ranges. A
/// format gate, not a calendar (2026-02-30 passes; calendar truth is the
/// parser's job).
pub fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let digits = |r: &[u8]| r.iter().all(|c| c.is_ascii_digit());
    if !(digits(&b[0..4]) && digits(&b[5..7]) && digits(&b[8..10])) {
        return false;
    }
    // Plausible ranges: month 01-12, day 01-31 (a FORMAT gate, not a
    // calendar — 2026-02-30 passes; calendar truth is the parser's job).
    let month = u16::from(b[5] - b'0') * 10 + u16::from(b[6] - b'0');
    let day = u16::from(b[8] - b'0') * 10 + u16::from(b[9] - b'0');
    (1..=12).contains(&month) && (1..=31).contains(&day)
}

/// Structural ISO-8601 time check: `HH:MM` or `HH:MM:SS`, plausible
/// ranges.
pub fn is_iso_time(s: &str) -> bool {
    let b = s.as_bytes();
    let ok_hms = |h: u32, m: u32, sec: u32| h < 24 && m < 60 && sec < 60;
    let two = |b: &[u8]| -> Option<u32> {
        if b.len() == 2 && b.iter().all(|c| c.is_ascii_digit()) {
            Some((b[0] - b'0') as u32 * 10 + (b[1] - b'0') as u32)
        } else {
            None
        }
    };
    match b.len() {
        5 => {
            b[2] == b':'
                && match (two(&b[0..2]), two(&b[3..5])) {
                    (Some(h), Some(m)) => ok_hms(h, m, 0),
                    _ => false,
                }
        }
        8 => {
            b[2] == b':'
                && b[5] == b':'
                && match (two(&b[0..2]), two(&b[3..5]), two(&b[6..8])) {
                    (Some(h), Some(m), Some(sec)) => ok_hms(h, m, sec),
                    _ => false,
                }
        }
        _ => false,
    }
}

/// Structural ISO-8601 datetime check: date + (`T` or space) + time.
pub fn is_iso_datetime(s: &str) -> bool {
    let Some((d, t)) = s.split_once(['T', ' ']) else {
        return false;
    };
    is_iso_date(d) && is_iso_time(t)
}

impl FieldSpec {
    /// Structural self-check: id + kind-specific constraint sanity.
    pub fn validate(&self, at: usize) -> Result<(), WireError> {
        if self.id.trim().is_empty() {
            return Err(WireError::EmptyId { at });
        }
        let c = &self.constraints;
        // Pairwise-impossible constraints refuse at the WIRE (the engine
        // would silently honor one and drop the other).
        if c.integer && c.decimals.is_some() {
            return Err(WireError::ConstraintConflict { at });
        }
        if c.element_kind.is_some() && self.kind != FieldKind::List {
            return Err(WireError::ConstraintConflict { at });
        }
        if c.grid.is_some() && self.kind != FieldKind::Estimate {
            return Err(WireError::ConstraintConflict { at });
        }
        if c.horizon.is_some() && self.kind != FieldKind::Forecast {
            return Err(WireError::ConstraintConflict { at });
        }
        if let Some(min) = c.min
            && !min.is_finite()
        {
            return Err(WireError::BadConstraint { at });
        }
        if let Some(max) = c.max
            && !max.is_finite()
        {
            return Err(WireError::BadConstraint { at });
        }
        if let (Some(min), Some(max)) = (c.min, c.max)
            && min > max
        {
            return Err(WireError::BadConstraint { at });
        }
        if let Some(grid) = c.grid {
            let ok = match grid.kind {
                GridKind::Linear => grid.step.is_finite() && grid.step > 0.0,
                GridKind::Log => grid.step.is_finite() && grid.step > 1.0,
            };
            if !ok {
                return Err(WireError::BadConstraint { at });
            }
        }
        if self.kind == FieldKind::Forecast {
            match c.horizon {
                Some(h) if h >= 1 => {}
                _ => return Err(WireError::BadConstraint { at }),
            }
        }
        if let Some(unit) = &c.unit
            && unit.trim().is_empty()
        {
            return Err(WireError::BadConstraint { at });
        }
        if self.kind == FieldKind::Duration
            && let Some(unit) = &c.unit
            && !DURATION_UNITS.contains(&unit.as_str())
        {
            return Err(WireError::BadConstraint { at });
        }
        for choice in &c.choices {
            if choice.trim().is_empty() {
                return Err(WireError::EmptyChoice { at });
            }
        }
        Ok(())
    }
}

impl ExtractionRequest {
    /// Structural self-check: version, per-field validity, id uniqueness,
    /// reference-date format. The version refusal is the additive-only
    /// law's teeth (see the module doc).
    pub fn validate(&self) -> Result<(), WireError> {
        if self.contract_version != CONTRACT_VERSION {
            return Err(WireError::UnknownContractVersion {
                got: self.contract_version,
                supported: CONTRACT_VERSION,
            });
        }
        for (at, f) in self.fields.iter().enumerate() {
            f.validate(at)?;
        }
        for (at, f) in self.fields.iter().enumerate() {
            if let Some(dup) = self.fields.iter().position(|other| other.id == f.id)
                && dup != at
            {
                return Err(WireError::DuplicateFieldId { at });
            }
        }
        if let Some(rd) = &self.reference_date
            && !is_iso_date(rd)
        {
            return Err(WireError::BadReferenceDate);
        }
        Ok(())
    }
}

fn check_probabilities(at: usize, p_null: f32, confidence: f32) -> Result<(), WireError> {
    for p in [p_null, confidence] {
        if !p.is_finite() {
            return Err(WireError::ProbabilityNotFinite { at });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(WireError::ProbabilityOutOfRange { at });
        }
    }
    Ok(())
}

fn check_value_against_kind(
    at: usize,
    kind: FieldKind,
    v: &FieldValue,
    constraints: &FieldConstraints,
) -> Result<(), WireError> {
    let kind_ok = matches!(
        (kind, v),
        (FieldKind::Text, FieldValue::Text(_))
            | (FieldKind::Number, FieldValue::Number(_))
            | (FieldKind::Date, FieldValue::Date(_))
            | (FieldKind::Time, FieldValue::Time(_))
            | (FieldKind::DateTime, FieldValue::DateTime(_))
            | (FieldKind::Duration, FieldValue::Duration(_))
            | (FieldKind::List, FieldValue::List(_))
            | (FieldKind::Estimate, FieldValue::Estimate(_))
            | (FieldKind::Forecast, FieldValue::Forecast(_))
    );
    if !kind_ok {
        return Err(WireError::ValueKindMismatch { at });
    }
    match v {
        FieldValue::Text(s) => {
            if s.is_empty() {
                return Err(WireError::ValueKindMismatch { at });
            }
            if !constraints.choices.is_empty() && !constraints.choices.contains(s) {
                return Err(WireError::ConstraintViolated { at });
            }
        }
        FieldValue::Number(n) => {
            if !n.is_finite() {
                return Err(WireError::ValueNotFinite { at });
            }
            if constraints.integer && n.fract() != 0.0 {
                return Err(WireError::ConstraintViolated { at });
            }
            if let Some(d) = constraints.decimals {
                let scaled = n * 10f64.powi(i32::from(d));
                if (scaled - scaled.round()).abs() > 1e-9 {
                    return Err(WireError::ConstraintViolated { at });
                }
            }
            if let Some(min) = constraints.min
                && *n < min
            {
                return Err(WireError::ConstraintViolated { at });
            }
            if let Some(max) = constraints.max
                && *n > max
            {
                return Err(WireError::ConstraintViolated { at });
            }
        }
        FieldValue::Date(s) => {
            if !is_iso_date(s) {
                return Err(WireError::BadDateFormat { at });
            }
        }
        FieldValue::Time(s) => {
            if !is_iso_time(s) {
                return Err(WireError::BadTimeFormat { at });
            }
        }
        FieldValue::DateTime(s) => {
            if !is_iso_datetime(s) {
                return Err(WireError::BadDateTimeFormat { at });
            }
        }
        FieldValue::Duration(d) => {
            if !DURATION_UNITS.contains(&d.unit.as_str()) || !d.value.is_finite() || d.value < 0.0 {
                return Err(WireError::BadDuration { at });
            }
            if let Some(unit) = &constraints.unit
                && &d.unit != unit
            {
                return Err(WireError::ConstraintViolated { at });
            }
        }
        FieldValue::List(items) => {
            let element = constraints.element_kind;
            if let Some(ek) = element
                && matches!(
                    ek,
                    FieldKind::List | FieldKind::Estimate | FieldKind::Forecast
                )
            {
                // Nesting lists or embedding computed distributions in
                // a list is a schema error, not an extraction answer.
                return Err(WireError::BadConstraint { at });
            }
            for (index, item) in items.iter().enumerate() {
                let item_kind = element.unwrap_or(FieldKind::Text);
                check_value_against_kind(at, item_kind, item, &FieldConstraints::default())
                    .map_err(|_| WireError::BadListElement { at, index })?;
            }
        }
        FieldValue::Estimate(e) => check_estimate(at, e)?,
        FieldValue::Forecast(f) => {
            check_forecast(at, f)?;
            // The answered horizon must equal the DECLARED horizon — the
            // caller asked for exactly that many steps.
            if let Some(h) = constraints.horizon
                && f.steps.len() != h as usize
            {
                return Err(WireError::ForecastMalformed { at });
            }
        }
    }
    Ok(())
}

fn check_estimate(at: usize, e: &EstimateDistribution) -> Result<(), WireError> {
    for n in [e.point, e.p10, e.median, e.p90, e.p_below, e.p_above] {
        if !n.is_finite() {
            return Err(WireError::ValueNotFinite { at });
        }
    }
    if !(e.p10 <= e.median && e.median <= e.p90) {
        return Err(WireError::DistributionMalformed { at });
    }
    if e.p_below < 0.0 || e.p_above < 0.0 || e.p_below + e.p_above > 1.0 {
        return Err(WireError::DistributionMalformed { at });
    }
    if !(e.level > 0.0 && e.level <= 1.0) {
        return Err(WireError::DistributionMalformed { at });
    }
    Ok(())
}

fn check_forecast(at: usize, f: &ForecastDistribution) -> Result<(), WireError> {
    if f.steps.is_empty() {
        return Err(WireError::ForecastMalformed { at });
    }
    if !(f.level > 0.0 && f.level <= 1.0) {
        return Err(WireError::ForecastMalformed { at });
    }
    for (i, s) in f.steps.iter().enumerate() {
        for n in [s.point, s.p10, s.median, s.p90] {
            if !n.is_finite() {
                return Err(WireError::ValueNotFinite { at });
            }
        }
        if !(s.p10 <= s.median && s.median <= s.p90) {
            return Err(WireError::DistributionMalformed { at });
        }
        if s.h != i as u32 + 1 {
            return Err(WireError::ForecastMalformed { at });
        }
    }
    Ok(())
}

impl FieldAnswer {
    /// Structural check against the field this answer answers: kind
    /// agreement, value-kind agreement, span laws, guess laws,
    /// probability ranges. Span BOUNDS need the state — pass its byte
    /// length (`state_len`).
    pub fn validate_against(
        &self,
        field: &FieldSpec,
        state_len: usize,
        at: usize,
    ) -> Result<(), WireError> {
        if self.id != field.id {
            return Err(WireError::AnswerIdMismatch { at });
        }
        if self.kind != field.kind {
            return Err(WireError::AnswerKindMismatch { at });
        }
        check_probabilities(at, self.p_null, self.confidence)?;
        // The guess laws, before anything else can mask them.
        if let Some(g) = &self.guess {
            if self.value.is_some() {
                return Err(WireError::GuessWithAnswer { at });
            }
            if self.span.is_some() {
                return Err(WireError::GuessWithSpan { at });
            }
            check_value_against_kind(at, field.kind, g, &field.constraints)?;
        }
        let Some(v) = &self.value else {
            if self.span.is_some() {
                return Err(WireError::SpanWithoutValue { at });
            }
            return Ok(());
        };
        // Span laws: located answers MUST ground; computed answers MUST
        // NOT claim a location.
        if field.kind.is_computed() {
            if self.span.is_some() {
                return Err(WireError::SpanOnComputedAnswer { at });
            }
        } else if self.span.is_none() {
            return Err(WireError::ValueWithoutSpan { at });
        }
        if let Some(span) = self.span {
            let (start, end) = (span.start as usize, span.end as usize);
            if start > end || end > state_len {
                return Err(WireError::SpanOutOfBounds { at });
            }
            // The char-boundary half of the law needs the state STRING,
            // not its length — it lives in `ExtractionResponse::
            // validate_against`, which holds the request (and the state).
        }
        check_value_against_kind(at, field.kind, v, &field.constraints)
    }
}

// The char-boundary law needs the state string; a length alone cannot
// decide it. `validate_against` pins bounds only; the full law lives in
// [`ExtractionResponse::validate_against`], which takes the request (and
// therefore the state).

impl ExtractionResponse {
    /// Structural check against the request this response answers:
    /// version echo, count, id/kind alignment, per-answer validity
    /// including span bounds + char boundaries against the state.
    pub fn validate_against(&self, request: &ExtractionRequest) -> Result<(), WireError> {
        if self.contract_version != request.contract_version {
            return Err(WireError::VersionMismatch {
                request: request.contract_version,
                response: self.contract_version,
            });
        }
        if self.contract_version != CONTRACT_VERSION {
            return Err(WireError::UnknownContractVersion {
                got: self.contract_version,
                supported: CONTRACT_VERSION,
            });
        }
        if self.answers.len() != request.fields.len() {
            return Err(WireError::AnswerCountMismatch {
                answers: self.answers.len(),
                fields: request.fields.len(),
            });
        }
        for (at, (a, f)) in self.answers.iter().zip(request.fields.iter()).enumerate() {
            a.validate_against(f, request.state.len(), at)?;
            // The char-boundary half of the span law — needs the state.
            if let Some(span) = a.span {
                let (start, end) = (span.start as usize, span.end as usize);
                if !request.state.is_char_boundary(start)
                    || (end < request.state.len() && !request.state.is_char_boundary(end))
                {
                    return Err(WireError::SpanNotCharBoundary { at });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str, kind: FieldKind) -> FieldSpec {
        FieldSpec {
            id: id.to_string(),
            kind,
            description: None,
            constraints: FieldConstraints::default(),
            instructions: None,
        }
    }

    fn answer(
        id: &str,
        kind: FieldKind,
        value: Option<FieldValue>,
        span: Option<Span>,
    ) -> FieldAnswer {
        FieldAnswer {
            id: id.to_string(),
            kind,
            value,
            span,
            p_null: 0.01,
            confidence: 0.9,
            guess: None,
        }
    }

    fn canonical_pair() -> (ExtractionRequest, ExtractionResponse) {
        let request = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "Invoice from Acme Corp, total $1,234.50, due 2026-11-15.".to_string(),
            fields: vec![
                spec("vendor", FieldKind::Text),
                spec("total", FieldKind::Number),
                spec("due", FieldKind::Date),
            ],
            reference_date: Some("2026-10-20".to_string()),
        };
        let response = ExtractionResponse {
            contract_version: CONTRACT_VERSION,
            answers: vec![
                answer(
                    "vendor",
                    FieldKind::Text,
                    Some(FieldValue::Text("Acme Corp".to_string())),
                    Some(Span { start: 13, end: 22 }),
                ),
                answer(
                    "total",
                    FieldKind::Number,
                    Some(FieldValue::Number(1234.5)),
                    Some(Span { start: 30, end: 39 }),
                ),
                answer(
                    "due",
                    FieldKind::Date,
                    Some(FieldValue::Date("2026-11-15".to_string())),
                    Some(Span { start: 44, end: 54 }),
                ),
            ],
        };
        (request, response)
    }

    #[test]
    fn canonical_pair_validates() {
        let (req, res) = canonical_pair();
        req.validate().expect("canonical request validates");
        res.validate_against(&req)
            .expect("canonical response validates");
    }

    #[test]
    fn json_roundtrip_is_byte_identical() {
        let (req, res) = canonical_pair();
        let req_json = serde_json::to_string(&req).expect("serialize request");
        let req2: ExtractionRequest = serde_json::from_str(&req_json).expect("deserialize request");
        assert_eq!(req, req2);
        let res_json = serde_json::to_string(&res).expect("serialize response");
        let res2: ExtractionResponse =
            serde_json::from_str(&res_json).expect("deserialize response");
        assert_eq!(res, res2);
    }

    #[test]
    fn postcard_roundtrip_is_identical() {
        let (req, res) = canonical_pair();
        let req_bytes = postcard::to_allocvec(&req).expect("postcard request");
        let req2: ExtractionRequest = postcard::from_bytes(&req_bytes).expect("decode request");
        assert_eq!(req, req2);
        let res_bytes = postcard::to_allocvec(&res).expect("postcard response");
        let res2: ExtractionResponse = postcard::from_bytes(&res_bytes).expect("decode response");
        assert_eq!(res, res2);
    }

    /// The golden JSON byte-pin: field renames or shape drift change this
    /// string and red the test (the decision_wire convention).
    #[test]
    fn golden_request_json_byte_pin() {
        let mut req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "Total: $5".to_string(),
            fields: vec![FieldSpec {
                id: "total".to_string(),
                kind: FieldKind::Number,
                description: Some("the invoice total".to_string()),
                constraints: FieldConstraints {
                    min: Some(0.0),
                    ..FieldConstraints::default()
                },
                instructions: Some("the grand total, not the subtotal".to_string()),
            }],
            reference_date: None,
        };
        req.fields[0].constraints.unit = Some("usd".to_string());
        let json = serde_json::to_string(&req).expect("serialize");
        let golden = "{\"contract_version\":1,\"state\":\"Total: $5\",\"fields\":[{\"id\":\"total\",\"kind\":\"number\",\"description\":\"the invoice total\",\"constraints\":{\"integer\":false,\"decimals\":null,\"unit\":\"usd\",\"min\":0.0,\"max\":null,\"element_kind\":null,\"choices\":[],\"grid\":null,\"horizon\":null},\"instructions\":\"the grand total, not the subtotal\"}],\"reference_date\":null}";
        assert_eq!(
            json, golden,
            "the wire shape is v1-frozen; a drift here is a VERSION BUMP, not a test fix"
        );
    }

    #[test]
    fn absent_version_decodes_as_one() {
        let (req, _) = canonical_pair();
        let mut json = serde_json::to_value(&req).expect("to value");
        if let serde_json::Value::Object(ref mut map) = json {
            map.remove("contract_version");
        }
        let req2: ExtractionRequest = serde_json::from_value(json).expect("absent version = 1");
        assert_eq!(req2.contract_version, CONTRACT_VERSION);
    }

    #[test]
    fn unknown_version_refused() {
        let (mut req, _) = canonical_pair();
        req.contract_version = 99;
        assert_eq!(
            req.validate(),
            Err(WireError::UnknownContractVersion {
                got: 99,
                supported: 1
            })
        );
        // The additive-only law: the RESPONDER refusing to answer an
        // unknown version is the same teeth (validate_against pins the
        // echo before anything else).
        let (_, res) = canonical_pair();
        let req_ok = {
            let (r, _) = canonical_pair();
            r
        };
        let mut stale = res;
        stale.contract_version = 2;
        assert!(stale.validate_against(&req_ok).is_err());
    }

    #[test]
    fn version_mismatch_refused() {
        let (req, mut res) = canonical_pair();
        res.contract_version = 2;
        assert_eq!(
            res.validate_against(&req),
            Err(WireError::VersionMismatch {
                request: 1,
                response: 2
            })
        );
    }

    #[test]
    fn duplicate_field_id_refused() {
        let mut req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "x".to_string(),
            fields: vec![spec("a", FieldKind::Text), spec("a", FieldKind::Text)],
            reference_date: None,
        };
        req.fields[1].id = "a".to_string();
        assert_eq!(req.validate(), Err(WireError::DuplicateFieldId { at: 1 }));
    }

    #[test]
    fn constraint_conflicts_refused() {
        // integer + decimals cannot coexist.
        let mut s = spec("n", FieldKind::Number);
        s.constraints.integer = true;
        s.constraints.decimals = Some(2);
        assert_eq!(s.validate(0), Err(WireError::ConstraintConflict { at: 0 }));
        // element_kind off-list.
        let mut s = spec("t", FieldKind::Text);
        s.constraints.element_kind = Some(FieldKind::Text);
        assert_eq!(s.validate(0), Err(WireError::ConstraintConflict { at: 0 }));
        // grid off-estimate.
        let mut s = spec("n", FieldKind::Number);
        s.constraints.grid = Some(Grid {
            kind: GridKind::Linear,
            step: 1.0,
        });
        assert_eq!(s.validate(0), Err(WireError::ConstraintConflict { at: 0 }));
        // horizon off-forecast.
        let mut s = spec("n", FieldKind::Number);
        s.constraints.horizon = Some(3);
        assert_eq!(s.validate(0), Err(WireError::ConstraintConflict { at: 0 }));
        // forecast REQUIRES a horizon >= 1.
        assert_eq!(
            spec("f", FieldKind::Forecast).validate(0),
            Err(WireError::BadConstraint { at: 0 })
        );
        let mut s = spec("f", FieldKind::Forecast);
        s.constraints.horizon = Some(0);
        assert_eq!(s.validate(0), Err(WireError::BadConstraint { at: 0 }));
        // log grid step must exceed 1.
        let mut s = spec("e", FieldKind::Estimate);
        s.constraints.grid = Some(Grid {
            kind: GridKind::Log,
            step: 1.0,
        });
        assert_eq!(s.validate(0), Err(WireError::BadConstraint { at: 0 }));
    }

    #[test]
    fn span_laws_enforced() {
        let state = "hello world";
        // Located answer without a span refuses.
        let a = answer(
            "t",
            FieldKind::Text,
            Some(FieldValue::Text("hello".into())),
            None,
        );
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: state.to_string(),
            fields: vec![spec("t", FieldKind::Text)],
            reference_date: None,
        };
        assert_eq!(
            a.validate_against(&req.fields[0], state.len(), 0),
            Err(WireError::ValueWithoutSpan { at: 0 })
        );
        // Span on a null refuses.
        let a = answer("t", FieldKind::Text, None, Some(Span { start: 0, end: 5 }));
        assert_eq!(
            a.validate_against(&req.fields[0], state.len(), 0),
            Err(WireError::SpanWithoutValue { at: 0 })
        );
        // Out-of-bounds span refuses.
        let a = answer(
            "t",
            FieldKind::Text,
            Some(FieldValue::Text("hello".into())),
            Some(Span { start: 6, end: 99 }),
        );
        assert_eq!(
            a.validate_against(&req.fields[0], state.len(), 0),
            Err(WireError::SpanOutOfBounds { at: 0 })
        );
    }

    #[test]
    fn computed_answers_carry_no_span() {
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "repair quote".to_string(),
            fields: vec![spec("cost", FieldKind::Estimate)],
            reference_date: None,
        };
        let est = EstimateDistribution {
            point: 500.0,
            p10: 400.0,
            median: 500.0,
            p90: 650.0,
            p_below: 0.02,
            p_above: 0.03,
            level: 0.8,
        };
        let a = answer(
            "cost",
            FieldKind::Estimate,
            Some(FieldValue::Estimate(est)),
            Some(Span { start: 0, end: 5 }),
        );
        assert_eq!(
            a.validate_against(&req.fields[0], req.state.len(), 0),
            Err(WireError::SpanOnComputedAnswer { at: 0 })
        );
        let a = answer(
            "cost",
            FieldKind::Estimate,
            Some(FieldValue::Estimate(est)),
            None,
        );
        a.validate_against(&req.fields[0], req.state.len(), 0)
            .expect("estimate without span validates");
    }

    #[test]
    fn guess_laws_enforced() {
        let state = "maybe bob";
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: state.to_string(),
            fields: vec![spec("who", FieldKind::Text)],
            reference_date: None,
        };
        let f = &req.fields[0];
        // Guess WITH an answer refuses.
        let a = FieldAnswer {
            value: Some(FieldValue::Text("bob".into())),
            span: Some(Span { start: 6, end: 9 }),
            guess: Some(FieldValue::Text("alice".into())),
            ..answer("who", FieldKind::Text, None, None)
        };
        assert_eq!(
            a.validate_against(f, state.len(), 0),
            Err(WireError::GuessWithAnswer { at: 0 })
        );
        // Guess WITH a span refuses.
        let a = FieldAnswer {
            value: None,
            span: Some(Span { start: 6, end: 9 }),
            guess: Some(FieldValue::Text("bob".into())),
            ..answer("who", FieldKind::Text, None, None)
        };
        assert_eq!(
            a.validate_against(f, state.len(), 0),
            Err(WireError::GuessWithSpan { at: 0 })
        );
        // Null WITH guess validates (the honest null-with-guess).
        let a = FieldAnswer {
            value: None,
            span: None,
            guess: Some(FieldValue::Text("bob".into())),
            p_null: 0.7,
            ..answer("who", FieldKind::Text, None, None)
        };
        a.validate_against(f, state.len(), 0)
            .expect("null-with-guess validates");
    }

    #[test]
    fn estimate_ordering_enforced() {
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "e".to_string(),
            fields: vec![spec("e", FieldKind::Estimate)],
            reference_date: None,
        };
        let mut est = EstimateDistribution {
            point: 500.0,
            p10: 650.0,
            median: 500.0,
            p90: 400.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.8,
        };
        let a = answer(
            "e",
            FieldKind::Estimate,
            Some(FieldValue::Estimate(est)),
            None,
        );
        assert_eq!(
            a.validate_against(&req.fields[0], 1, 0),
            Err(WireError::DistributionMalformed { at: 0 })
        );
        est.p10 = 400.0;
        est.p90 = 650.0;
        est.level = 1.5;
        let a = answer(
            "e",
            FieldKind::Estimate,
            Some(FieldValue::Estimate(est)),
            None,
        );
        assert_eq!(
            a.validate_against(&req.fields[0], 1, 0),
            Err(WireError::DistributionMalformed { at: 0 })
        );
    }

    #[test]
    fn list_element_kinds_enforced() {
        let mut s = spec("items", FieldKind::List);
        s.constraints.element_kind = Some(FieldKind::Number);
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "3 apples, 5 oranges".to_string(),
            fields: vec![s],
            reference_date: None,
        };
        let f = &req.fields[0];
        let good = FieldValue::List(vec![FieldValue::Number(3.0), FieldValue::Number(5.0)]);
        let a = answer(
            "items",
            FieldKind::List,
            Some(good),
            Some(Span { start: 0, end: 19 }),
        );
        a.validate_against(f, req.state.len(), 0)
            .expect("typed list validates");
        let bad = FieldValue::List(vec![
            FieldValue::Number(3.0),
            FieldValue::Text("five".into()),
        ]);
        let a = answer(
            "items",
            FieldKind::List,
            Some(bad),
            Some(Span { start: 0, end: 19 }),
        );
        assert_eq!(
            a.validate_against(f, req.state.len(), 0),
            Err(WireError::BadListElement { at: 0, index: 1 })
        );
    }

    #[test]
    fn number_constraints_enforced_by_construction() {
        let mut s = spec("n", FieldKind::Number);
        s.constraints.integer = true;
        s.constraints.min = Some(0.0);
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "n".to_string(),
            fields: vec![s],
            reference_date: None,
        };
        let f = &req.fields[0];
        for (v, ok) in [(3.0, true), (3.5, false), (-1.0, false)] {
            let a = answer(
                "n",
                FieldKind::Number,
                Some(FieldValue::Number(v)),
                Some(Span { start: 0, end: 1 }),
            );
            let verdict = a.validate_against(f, 1, 0);
            assert_eq!(verdict.is_ok(), ok, "value {v}");
        }
        // decimals cap: 2 places accepts 1.25, refuses 1.255.
        let mut s = spec("d", FieldKind::Number);
        s.constraints.decimals = Some(2);
        let req2 = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "d".to_string(),
            fields: vec![s],
            reference_date: None,
        };
        let f = &req2.fields[0];
        let a = answer(
            "d",
            FieldKind::Number,
            Some(FieldValue::Number(1.25)),
            Some(Span { start: 0, end: 1 }),
        );
        a.validate_against(f, 1, 0).expect("1.25 fits 2 decimals");
        let a = answer(
            "d",
            FieldKind::Number,
            Some(FieldValue::Number(1.255)),
            Some(Span { start: 0, end: 1 }),
        );
        assert_eq!(
            a.validate_against(f, 1, 0),
            Err(WireError::ConstraintViolated { at: 0 })
        );
    }

    #[test]
    fn format_checks_and_choices_enforced() {
        assert!(is_iso_date("2026-11-15"));
        assert!(!is_iso_date("2026-13-15"));
        assert!(!is_iso_date("2026-11-1"));
        assert!(is_iso_time("14:30"));
        assert!(is_iso_time("14:30:59"));
        assert!(!is_iso_time("24:00"));
        assert!(!is_iso_time("14:60"));
        assert!(is_iso_datetime("2026-11-15T14:30"));
        assert!(is_iso_datetime("2026-11-15 14:30:00"));
        assert!(!is_iso_datetime("2026-11-15X14:30"));

        // choices: a value outside the closed set refuses (this is the
        // F0-readable schema signal, so its enforcement is the contract).
        let mut s = spec("city", FieldKind::Text);
        s.constraints.choices = vec!["Singapore".into(), "Tokyo".into()];
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "city".to_string(),
            fields: vec![s],
            reference_date: None,
        };
        let f = &req.fields[0];
        let a = answer(
            "city",
            FieldKind::Text,
            Some(FieldValue::Text("Singapore".into())),
            Some(Span { start: 0, end: 4 }),
        );
        a.validate_against(f, 4, 0)
            .expect("choice in set validates");
        let a = answer(
            "city",
            FieldKind::Text,
            Some(FieldValue::Text("Paris".into())),
            Some(Span { start: 0, end: 4 }),
        );
        assert_eq!(
            a.validate_against(f, 4, 0),
            Err(WireError::ConstraintViolated { at: 0 })
        );
    }

    #[test]
    fn duration_units_pinned() {
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "d".to_string(),
            fields: vec![spec("d", FieldKind::Duration)],
            reference_date: None,
        };
        let f = &req.fields[0];
        let a = answer(
            "d",
            FieldKind::Duration,
            Some(FieldValue::Duration(DurationValue {
                unit: "min".into(),
                value: 90.0,
            })),
            Some(Span { start: 0, end: 1 }),
        );
        a.validate_against(f, 1, 0)
            .expect("canonical duration validates");
        let a = answer(
            "d",
            FieldKind::Duration,
            Some(FieldValue::Duration(DurationValue {
                unit: "fortnight".into(),
                value: 1.0,
            })),
            Some(Span { start: 0, end: 1 }),
        );
        assert_eq!(
            a.validate_against(f, 1, 0),
            Err(WireError::BadDuration { at: 0 })
        );
    }

    #[test]
    fn forecast_shape_enforced() {
        let mut s = spec("fc", FieldKind::Forecast);
        s.constraints.horizon = Some(3);
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "s".to_string(),
            fields: vec![s],
            reference_date: None,
        };
        let f = &req.fields[0];
        let mk = |h: u32| ForecastStep {
            h,
            point: 1.0,
            p10: 0.9,
            median: 1.0,
            p90: 1.1,
        };
        let good = ForecastDistribution {
            steps: vec![mk(1), mk(2), mk(3)],
            level: 0.8,
        };
        let a = answer(
            "fc",
            FieldKind::Forecast,
            Some(FieldValue::Forecast(good)),
            None,
        );
        a.validate_against(f, 1, 0)
            .expect("1..h forecast validates");
        // Horizon gap refuses (h must be 1-based contiguous).
        let gapped = ForecastDistribution {
            steps: vec![mk(1), mk(3)],
            level: 0.8,
        };
        let a = answer(
            "fc",
            FieldKind::Forecast,
            Some(FieldValue::Forecast(gapped)),
            None,
        );
        assert_eq!(
            a.validate_against(f, 1, 0),
            Err(WireError::ForecastMalformed { at: 0 })
        );
        // Horizon overrun refuses (declared horizon 3, answered 2 — the
        // caller asked for steps the engine did not produce).
        let short = ForecastDistribution {
            steps: vec![mk(1), mk(2)],
            level: 0.8,
        };
        let a = answer(
            "fc",
            FieldKind::Forecast,
            Some(FieldValue::Forecast(short)),
            None,
        );
        assert_eq!(
            a.validate_against(f, 1, 0),
            Err(WireError::ForecastMalformed { at: 0 })
        );
    }

    #[test]
    fn count_and_id_alignment_enforced() {
        let (req, mut res) = canonical_pair();
        res.answers.pop();
        assert_eq!(
            res.validate_against(&req),
            Err(WireError::AnswerCountMismatch {
                answers: 2,
                fields: 3
            })
        );
        let (_, mut res) = canonical_pair();
        res.answers[1].id = "wrong".to_string();
        assert_eq!(
            res.validate_against(&req),
            Err(WireError::AnswerIdMismatch { at: 1 })
        );
        let (_, mut res) = canonical_pair();
        res.answers[0].kind = FieldKind::Number;
        assert_eq!(
            res.validate_against(&req),
            Err(WireError::AnswerKindMismatch { at: 0 })
        );
    }

    #[test]
    fn probability_ranges_enforced() {
        let (req, mut res) = canonical_pair();
        res.answers[0].p_null = f32::NAN;
        assert!(matches!(
            res.validate_against(&req),
            Err(WireError::ProbabilityNotFinite { at: 0 })
        ));
        let (_, mut res) = canonical_pair();
        res.answers[0].confidence = 1.5;
        assert!(matches!(
            res.validate_against(&req),
            Err(WireError::ProbabilityOutOfRange { at: 0 })
        ));
    }

    #[test]
    fn bad_reference_date_refused() {
        let mut req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: "s".to_string(),
            fields: vec![],
            reference_date: Some("10/20/2026".to_string()),
        };
        assert_eq!(req.validate(), Err(WireError::BadReferenceDate));
        req.reference_date = Some("2026-10-20".to_string());
        req.validate().expect("ISO reference date validates");
    }

    #[test]
    fn char_boundary_law_enforced_against_state() {
        let state = "héllo wörld"; // multi-byte chars: 'é' 2B, 'ö' 2B
        let req = ExtractionRequest {
            contract_version: CONTRACT_VERSION,
            state: state.to_string(),
            fields: vec![spec("t", FieldKind::Text)],
            reference_date: None,
        };
        // Byte 2 is INSIDE 'é' ('h' is byte 0, 'é' spans bytes 1-2) — in
        // bounds, so the per-answer bounds check passes; the char-boundary
        // refusal comes from the response check, which holds the state.
        let a = answer(
            "t",
            FieldKind::Text,
            Some(FieldValue::Text("é".into())),
            Some(Span { start: 2, end: 4 }),
        );
        a.validate_against(&req.fields[0], state.len(), 0)
            .expect("bounds alone pass at the answer level");
        let res = ExtractionResponse {
            contract_version: CONTRACT_VERSION,
            answers: vec![a],
        };
        assert_eq!(
            res.validate_against(&req),
            Err(WireError::SpanNotCharBoundary { at: 0 })
        );
    }
}
