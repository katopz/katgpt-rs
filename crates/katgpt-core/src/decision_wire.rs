//! Jev/laya-compatible decision wire contract — Plan 603 T1.2, the
//! katgpt-rs public half of Proposal 014 (the decision-engine arena).
//!
//! The contract is the THREE-TYPE vocabulary both arena lanes speak and
//! every published table reads:
//!
//! - **`choice`** — multiple choice; the option set is defined AT REQUEST
//!   TIME (laya's head law, Research 576 §1 — new schemas need no
//!   retraining). Optionally carries selection `criteria`.
//! - **`score`** — ordinal score over rubric `options`, lowest level first
//!   (SST-5's 1–5 is five options; laya's weakest axis).
//! - **`noul`** — yes/no, `TypeSafe`'s typed boolean (Research 562 §1:
//!   `Noul`(Bernoulli)/`Choice`/`Score`).
//!
//! Answers carry the typed outcome + calibrated per-option **probabilities**
//! + the scalar **confidence** readout. The confidence readout policy is
//!   INHERITED, not re-derived (Bench 817, caveat 6): label-entropy for
//!   narrow option sets, argmax-label-prob for wide ones — the dispatch
//!   lives in the engine; this wire only carries the result.
//!
//! Abstention: Jev CANNOT abstain (Research 562's recorded flaw — a forced
//! answer is a hallucination vector). This wire makes abstention a
//! first-class answer (`outcome: None`) while still allowing the
//! distribution + confidence to ride along, so the arena's risk–coverage /
//! selective-accuracy tables (T1.5, the T1.6 abstain arm) can be computed
//! from one response.
//!
//! Calibration + routing metadata ride the response: `Calibration` is the
//! G1 Report-the-Floor surface (a "calibrated" claim without naming the
//! method is unmarkable in the arena tables), and `Routing` records which
//! lane served (modelless / laya port / hybrid) plus the router's reason —
//! laya's own Router runs BEFORE the forward pass for the same reason
//! (script detection; Research 576 §1's 0.000-accuracy-at-0.952-confidence
//! Khmer result is why routing is not derivable from confidence).
//!
//! Scope: WIRE ONLY — pure data + fail-closed structural validation. No
//! engine logic, no inference, no allocation-sensitive hot path (the wire
//! is the cold edge; the engine's hot path in riir-reflex never allocates
//! per decision). Serde is already non-optional in this crate — zero new
//! deps. Opt-in (`decision_wire`) per the no-default-consumer rule
//! (`margin_gate` precedent) — the consumer is the Proposal 014 Phase-1
//! engine (riir-reflex); promotion to default rides the GOAT gate.
//!
//! Validation: [`DecisionRequest::validate`] + [`DecisionResponse::
//! validate_against`] are fail-closed structural checks (arity, ranges,
//! finiteness, id alignment, kind agreement). They pin the CONTRACT, not
//! model quality — a confidently-wrong answer validates fine (that is what
//! calibration + abstention are for).

use serde::{Deserialize, Serialize};

/// The three typed question kinds — laya's contract verbatim
/// (Research 576 §1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
pub enum QuestionKind {
    /// Multiple choice over `Question::options` (defined at request time).
    Choice,
    /// Ordinal score over rubric `Question::options`, lowest level first.
    Score,
    /// Yes/no — `TypeSafe`'s typed boolean (Research 562 §1).
    Noul,
}

impl QuestionKind {
    /// The wire spelling (also the serde tag).
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

/// One typed question. The answer space is part of the REQUEST — laya's
/// "answer space defined at request time" law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Question {
    /// Stable id; the response's answers echo it (the alignment pin).
    pub id: String,
    pub kind: QuestionKind,
    pub prompt: String,
    /// `choice`: the candidate answers, outcome-indexed.
    /// `score`: the ordinal rubric levels, LOWEST first.
    /// `noul`: MUST be empty — yes/no is implicit in the kind.
    ///
    /// Always serialized (never skipped): postcard is positional — a
    /// skipped field cannot be decoded back (the `ptg_functor_edges` law).
    #[serde(default)]
    pub options: Vec<String>,
    /// `choice` selection criteria (laya's "choice with criteria"); the
    /// rubric text for `score` may also ride here when it needs more than
    /// the level labels. Always serialized (`null` when absent).
    #[serde(default)]
    pub criteria: Option<String>,
}

impl Question {
    /// A `choice` question with explicit criteria.
    pub fn choice(
        id: impl Into<String>,
        prompt: impl Into<String>,
        options: Vec<String>,
        criteria: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: QuestionKind::Choice,
            prompt: prompt.into(),
            options,
            criteria,
        }
    }

    /// A `score` question over ordinal rubric levels (lowest first).
    pub fn score(id: impl Into<String>, prompt: impl Into<String>, rubric: Vec<String>) -> Self {
        Self {
            id: id.into(),
            kind: QuestionKind::Score,
            prompt: prompt.into(),
            options: rubric,
            criteria: None,
        }
    }

    /// A `noul` (yes/no) question.
    pub fn noul(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: QuestionKind::Noul,
            prompt: prompt.into(),
            options: Vec::new(),
            criteria: None,
        }
    }

    /// Length of the probability vector a well-formed answer to this
    /// question carries: `noul` carries one value (p(yes)); `choice` and
    /// `score` carry one per option.
    pub fn probability_arity(&self) -> usize {
        match self.kind {
            QuestionKind::Noul => 1,
            QuestionKind::Choice | QuestionKind::Score => self.options.len(),
        }
    }

    /// Structural self-check (position-tagged for actionable errors).
    pub fn validate(&self, at: usize) -> Result<(), WireError> {
        if self.prompt.trim().is_empty() {
            return Err(WireError::EmptyPrompt { at });
        }
        match self.kind {
            QuestionKind::Noul => {
                if !self.options.is_empty() {
                    return Err(WireError::NoulCarriesOptions { at });
                }
            }
            QuestionKind::Choice | QuestionKind::Score => {
                if self.options.len() < 2 {
                    return Err(WireError::TooFewOptions { at, n: self.options.len() });
                }
            }
        }
        Ok(())
    }
}

/// One typed decision outcome — the laya vocabulary, verbatim tags.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    /// The chosen option's index into `Question::options`.
    Choice { index: u32 },
    /// The chosen rubric level's index (lowest = 0).
    Score { level: u32 },
    /// The typed boolean.
    Noul { yes: bool },
}

/// One answer, aligned to its question by `question_id` echo AND by
/// position (`answers[i]` answers `questions[i]` — both are checked).
///
/// Abstention is `outcome: None` (Jev cannot abstain; the engine can).
/// An abstained answer may still carry the distribution + confidence —
/// that is what the risk–coverage tables read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    /// Echoes `Question::id`.
    pub question_id: String,
    /// `None` = abstained. Always serialized (`null` when abstained) —
    /// postcard is positional (the `ptg_functor_edges` law).
    pub outcome: Option<Outcome>,
    /// Calibrated per-option probabilities: `choice`/`score` carry
    /// `options.len()` values in option order; `noul` carries exactly one
    /// value, p(yes). Must be full-arity whenever `outcome` is present;
    /// optional when abstained (abstain-before-score serializes `[]`).
    #[serde(default)]
    pub probabilities: Vec<f32>,
    /// The confidence readout (Bench 817 policy, inherited — see module
    /// doc). Finite and within `[0, 1]`.
    pub confidence: f32,
}

impl Answer {
    /// A `choice` answer.
    pub fn choice(
        question_id: impl Into<String>,
        index: u32,
        probabilities: Vec<f32>,
        confidence: f32,
    ) -> Self {
        Self {
            question_id: question_id.into(),
            outcome: Some(Outcome::Choice { index }),
            probabilities,
            confidence,
        }
    }

    /// A `score` answer.
    pub fn score(
        question_id: impl Into<String>,
        level: u32,
        probabilities: Vec<f32>,
        confidence: f32,
    ) -> Self {
        Self {
            question_id: question_id.into(),
            outcome: Some(Outcome::Score { level }),
            probabilities,
            confidence,
        }
    }

    /// A `noul` answer; `p_yes` is the single probability carried.
    pub fn noul(question_id: impl Into<String>, yes: bool, p_yes: f32, confidence: f32) -> Self {
        Self {
            question_id: question_id.into(),
            outcome: Some(Outcome::Noul { yes }),
            probabilities: vec![p_yes],
            confidence,
        }
    }

    /// An abstained answer (no outcome, no distribution).
    pub fn abstain(question_id: impl Into<String>, confidence: f32) -> Self {
        Self {
            question_id: question_id.into(),
            outcome: None,
            probabilities: Vec::new(),
            confidence,
        }
    }

    /// Structural check against the question this answer claims to answer.
    pub fn validate_against(&self, question: &Question, at: usize) -> Result<(), WireError> {
        if self.question_id != question.id {
            return Err(WireError::AnswerIdMismatch { at });
        }
        if !self.confidence.is_finite() {
            return Err(WireError::ConfidenceNotFinite);
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(WireError::ConfidenceOutOfRange);
        }
        if let Some(outcome) = self.outcome {
            match (outcome, question.kind) {
                (Outcome::Choice { index }, QuestionKind::Choice) => {
                    if index as usize >= question.options.len() {
                        return Err(WireError::OutcomeOutOfRange {
                            at,
                            index,
                            options: question.options.len() as u32,
                        });
                    }
                }
                (Outcome::Score { level }, QuestionKind::Score) => {
                    if level as usize >= question.options.len() {
                        return Err(WireError::OutcomeOutOfRange {
                            at,
                            index: level,
                            options: question.options.len() as u32,
                        });
                    }
                }
                (Outcome::Noul { .. }, QuestionKind::Noul) => {}
                _ => return Err(WireError::OutcomeKindMismatch { at }),
            }
        }
        // Full-arity distribution REQUIRED when answered; OPTIONAL when
        // abstained (but if present it must still be well-formed).
        if self.outcome.is_some() || !self.probabilities.is_empty() {
            let want = question.probability_arity();
            if self.probabilities.len() != want {
                return Err(WireError::ProbabilityArityMismatch {
                    at,
                    got: self.probabilities.len(),
                    want,
                });
            }
            for (option, p) in self.probabilities.iter().enumerate() {
                if !p.is_finite() {
                    return Err(WireError::ProbabilityNotFinite { at, option });
                }
                if !(0.0..=1.0).contains(p) {
                    return Err(WireError::ProbabilityOutOfRange { at, option });
                }
            }
        }
        Ok(())
    }
}

/// The decision request: one state, ALL questions answered in one call
/// (laya's "all questions in the call" law — no per-question round trips).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    /// The state under decision — text/email/ticket/JSON (laya's state
    /// contract). The engine's corpus gates and routers read this.
    pub state: String,
    pub questions: Vec<Question>,
}

impl DecisionRequest {
    /// Structural self-check: per-question validity + id uniqueness
    /// (duplicate ids would make the answer-id echo ambiguous).
    pub fn validate(&self) -> Result<(), WireError> {
        for (at, q) in self.questions.iter().enumerate() {
            q.validate(at)?;
        }
        for (at, q) in self.questions.iter().enumerate() {
            if let Some(dup) = self.questions.iter().position(|other| other.id == q.id)
                && dup != at {
                    return Err(WireError::DuplicateQuestionId { at });
                }
        }
        Ok(())
    }
}

/// Which lane served the response — the arena's per-lane scoping axis
/// (caveat 3: every published table scopes determinism/latency claims to
/// the lane that produced them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
pub enum Lane {
    /// The katgpt-rs modelless lane (tables + corpus-is-the-model scoring).
    Modelless,
    /// The native-Rust laya port (Plan 603 T1.4).
    Laya,
    /// Both, composed (e.g. laya proposes, modelless gates/abstains).
    Hybrid,
    /// The CLM comparison lane (riir-reflex Issue 019) — the external
    /// Apache-2.0 Contrastive-LM reference served over HTTP, measured in
    /// the arena's comparison tables; never a default-on product lane.
    Clm,
}

/// Router verdict: which lane, and why it was picked (laya's Router runs
/// BEFORE the forward pass; Research 576 §1 — script/language detection is
/// a routing signal precisely because confidence cannot be trusted
/// cross-script).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Routing {
    pub lane: Lane,
    /// Why the router picked the lane (`null` when the engine records no
    /// reason). Always serialized — postcard is positional.
    pub reason: Option<String>,
}

/// The calibration applied to THIS response's probabilities — the G1
/// Report-the-Floor surface. An arena lane claiming "calibrated" without
/// this metadata is unmarkable; `method: "none"` + temperature `1.0` is
/// the honest raw posture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// `"none"` | `"temperature"` | other method ids as the engine grows
    /// them (per-(question-kind, option-count) refit is laya's shape,
    /// Research 576 §1: ECE 0.466→0.081).
    pub method: String,
    /// The temperature applied — `1.0` is identity (uncalibrated).
    pub temperature: f32,
}

impl Calibration {
    /// The honest raw posture.
    pub fn none() -> Self {
        Self {
            method: "none".to_string(),
            temperature: 1.0,
        }
    }
}

/// The decision response: one answer per question (same order), plus the
/// routing + calibration metadata the arena tables read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    /// `answers[i]` answers `questions[i]`; the id echo is checked too.
    pub answers: Vec<Answer>,
    pub routing: Routing,
    pub calibration: Calibration,
}

impl DecisionResponse {
    /// Structural check against the request this response answers:
    /// count, id alignment, per-answer validity.
    pub fn validate_against(&self, request: &DecisionRequest) -> Result<(), WireError> {
        if self.answers.len() != request.questions.len() {
            return Err(WireError::AnswerCountMismatch {
                answers: self.answers.len(),
                questions: request.questions.len(),
            });
        }
        for (at, (a, q)) in self.answers.iter().zip(request.questions.iter()).enumerate() {
            a.validate_against(q, at)?;
        }
        Ok(())
    }
}

/// Fail-closed structural validation errors. Copy so callers can hold the
/// verdict alongside the payload; positions (`at`) index into the
/// request's `questions` / the response's `answers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireError {
    /// Question prompt is empty (or whitespace).
    EmptyPrompt { at: usize },
    /// A `noul` question carried options — yes/no is implicit.
    NoulCarriesOptions { at: usize },
    /// A `choice`/`score` question carried fewer than two options.
    TooFewOptions { at: usize, n: usize },
    /// Two questions share an id; `at` is the later occurrence.
    DuplicateQuestionId { at: usize },
    /// Answer count ≠ question count.
    AnswerCountMismatch { answers: usize, questions: usize },
    /// `answers[at].question_id` ≠ `questions[at].id`.
    AnswerIdMismatch { at: usize },
    /// The outcome's index/level is past the option set.
    OutcomeOutOfRange { at: usize, index: u32, options: u32 },
    /// The outcome's kind disagrees with the question's kind.
    OutcomeKindMismatch { at: usize },
    /// Probability vector arity ≠ the question kind's requirement.
    ProbabilityArityMismatch { at: usize, got: usize, want: usize },
    /// A probability is NaN or infinite.
    ProbabilityNotFinite { at: usize, option: usize },
    /// A probability is outside `[0, 1]`.
    ProbabilityOutOfRange { at: usize, option: usize },
    /// Confidence is NaN or infinite.
    ConfidenceNotFinite,
    /// Confidence is outside `[0, 1]`.
    ConfidenceOutOfRange,
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::EmptyPrompt { at } => write!(f, "decision_wire: question {at} has an empty prompt"),
            WireError::NoulCarriesOptions { at } => {
                write!(f, "decision_wire: noul question {at} must not carry options")
            }
            WireError::TooFewOptions { at, n } => {
                write!(f, "decision_wire: choice/score question {at} needs ≥2 options, got {n}")
            }
            WireError::DuplicateQuestionId { at } => {
                write!(f, "decision_wire: duplicate question id at position {at}")
            }
            WireError::AnswerCountMismatch { answers, questions } => {
                write!(f, "decision_wire: {answers} answers for {questions} questions")
            }
            WireError::AnswerIdMismatch { at } => {
                write!(f, "decision_wire: answer {at} does not echo its question id")
            }
            WireError::OutcomeOutOfRange { at, index, options } => {
                write!(f, "decision_wire: answer {at} outcome index {index} out of range (options {options})")
            }
            WireError::OutcomeKindMismatch { at } => {
                write!(f, "decision_wire: answer {at} outcome kind disagrees with the question kind")
            }
            WireError::ProbabilityArityMismatch { at, got, want } => {
                write!(f, "decision_wire: answer {at} carries {got} probabilities, want {want}")
            }
            WireError::ProbabilityNotFinite { at, option } => {
                write!(f, "decision_wire: answer {at} probability {option} is not finite")
            }
            WireError::ProbabilityOutOfRange { at, option } => {
                write!(f, "decision_wire: answer {at} probability {option} outside [0, 1]")
            }
            WireError::ConfidenceNotFinite => write!(f, "decision_wire: confidence is not finite"),
            WireError::ConfidenceOutOfRange => write!(f, "decision_wire: confidence outside [0, 1]"),
        }
    }
}

impl std::error::Error for WireError {}

// ── Golden fixtures ─────────────────────────────────────────────────────
// Byte-pins for the wire shape: field order, serde tag spellings, and the
// ALWAYS-EMIT rule (no skip_serializing_if anywhere — postcard is
// positional and cannot decode a skipped field back; the ptg_functor_edges
// law). Absent values are `null` / `[]`, never missing. Values are dyadic
// (exact in f32, stable shortest-repr) so the pin is robust to formatter
// internals. A serde/serde_json change that moves these bytes is a WIRE
// BREAK and must be a deliberate fixture re-pin, never a silent drift.

#[cfg(test)]
const GOLDEN_REQUEST_JSON: &str = r#"{"state":"ticket 4711: login loop on Safari 17 after cookie purge","questions":[{"id":"q1","kind":"choice","prompt":"Which subsystem is at fault?","options":["auth","session","transport"],"criteria":"Pick the single subsystem whose defect best explains the report"},{"id":"q2","kind":"noul","prompt":"Is the report reproducible from the described steps?","options":[],"criteria":null},{"id":"q3","kind":"score","prompt":"Severity","options":["1","2","3","4","5"],"criteria":null}]}"#;

#[cfg(test)]
const GOLDEN_RESPONSE_JSON: &str = r#"{"answers":[{"question_id":"q1","outcome":{"choice":{"index":1}},"probabilities":[0.25,0.5,0.25],"confidence":0.5},{"question_id":"q2","outcome":null,"probabilities":[0.875],"confidence":0.875},{"question_id":"q3","outcome":{"score":{"level":3}},"probabilities":[0.125,0.125,0.25,0.25,0.25],"confidence":0.75}],"routing":{"lane":"modelless","reason":null},"calibration":{"method":"none","temperature":1.0}}"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn golden_request() -> DecisionRequest {
        DecisionRequest {
            state: "ticket 4711: login loop on Safari 17 after cookie purge".into(),
            questions: vec![
                Question::choice(
                    "q1",
                    "Which subsystem is at fault?",
                    vec!["auth".into(), "session".into(), "transport".into()],
                    Some("Pick the single subsystem whose defect best explains the report".into()),
                ),
                Question::noul("q2", "Is the report reproducible from the described steps?"),
                Question::score(
                    "q3",
                    "Severity",
                    vec!["1".into(), "2".into(), "3".into(), "4".into(), "5".into()],
                ),
            ],
        }
    }

    fn golden_response() -> DecisionResponse {
        DecisionResponse {
            answers: vec![
                Answer::choice("q1", 1, vec![0.25, 0.5, 0.25], 0.5),
                // Abstained AFTER scoring: the distribution still rides so
                // risk–coverage curves can be computed from one response.
                Answer {
                    question_id: "q2".into(),
                    outcome: None,
                    probabilities: vec![0.875],
                    confidence: 0.875,
                },
                Answer::score("q3", 3, vec![0.125, 0.125, 0.25, 0.25, 0.25], 0.75),
            ],
            routing: Routing {
                lane: Lane::Modelless,
                reason: None,
            },
            calibration: Calibration::none(),
        }
    }

    #[test]
    fn json_round_trip_preserves_request_and_response() {
        let req = golden_request();
        let resp = golden_response();
        let req_text = serde_json::to_string(&req).expect("serialize request");
        let resp_text = serde_json::to_string(&resp).expect("serialize response");
        let req_back: DecisionRequest = serde_json::from_str(&req_text).expect("parse request");
        let resp_back: DecisionResponse = serde_json::from_str(&resp_text).expect("parse response");
        assert_eq!(req_back, req, "request JSON round-trip must be lossless");
        assert_eq!(resp_back, resp, "response JSON round-trip must be lossless");
    }

    #[test]
    fn postcard_round_trip_preserves_request_and_response() {
        let req = golden_request();
        let resp = golden_response();
        let req_bytes = postcard::to_allocvec(&req).expect("postcard request");
        let resp_bytes = postcard::to_allocvec(&resp).expect("postcard response");
        let req_back: DecisionRequest = postcard::from_bytes(&req_bytes).expect("decode request");
        let resp_back: DecisionResponse =
            postcard::from_bytes(&resp_bytes).expect("decode response");
        assert_eq!(req_back, req, "postcard request round-trip must be lossless");
        assert_eq!(resp_back, resp, "postcard response round-trip must be lossless");
    }

    #[test]
    fn golden_request_json_is_byte_stable() {
        let text = serde_json::to_string(&golden_request()).expect("serialize request");
        assert_eq!(text, GOLDEN_REQUEST_JSON, "request wire bytes drifted — a serde/field change moved the wire shape; re-pin deliberately");
    }

    #[test]
    fn golden_response_json_is_byte_stable() {
        let text = serde_json::to_string(&golden_response()).expect("serialize response");
        assert_eq!(text, GOLDEN_RESPONSE_JSON, "response wire bytes drifted — a serde/field change moved the wire shape; re-pin deliberately");
    }

    #[test]
    fn validate_accepts_the_canonical_pair() {
        let req = golden_request();
        let resp = golden_response();
        req.validate().expect("canonical request must validate");
        resp.validate_against(&req).expect("canonical pair must validate");
    }

    #[test]
    fn request_validation_refuses_structural_violations() {
        let mut req = golden_request();
        // noul carrying options
        req.questions[1].options = vec!["yes".into(), "no".into()];
        assert_eq!(
            req.validate(),
            Err(WireError::NoulCarriesOptions { at: 1 }),
            "noul must refuse options"
        );
        // choice with a single option
        req = golden_request();
        req.questions[0].options = vec!["only".into()];
        assert_eq!(
            req.validate(),
            Err(WireError::TooFewOptions { at: 0, n: 1 }),
            "a one-option choice is degenerate"
        );
        // empty prompt
        req = golden_request();
        req.questions[2].prompt = "   ".into();
        assert_eq!(req.validate(), Err(WireError::EmptyPrompt { at: 2 }));
        // duplicate ids
        req = golden_request();
        req.questions[2].id = "q1".into();
        assert_eq!(req.validate(), Err(WireError::DuplicateQuestionId { at: 2 }));
    }

    #[test]
    fn response_validation_refuses_count_and_id_mismatch() {
        let req = golden_request();
        let mut resp = golden_response();
        resp.answers.pop();
        assert_eq!(
            resp.validate_against(&req),
            Err(WireError::AnswerCountMismatch {
                answers: 2,
                questions: 3
            })
        );
        let mut resp = golden_response();
        resp.answers[1].question_id = "qX".into();
        assert_eq!(resp.validate_against(&req), Err(WireError::AnswerIdMismatch { at: 1 }));
    }

    #[test]
    fn response_validation_refuses_bad_outcomes() {
        let req = golden_request();
        // index out of range
        let mut resp = golden_response();
        resp.answers[0].outcome = Some(Outcome::Choice { index: 3 });
        assert_eq!(
            resp.validate_against(&req),
            Err(WireError::OutcomeOutOfRange {
                at: 0,
                index: 3,
                options: 3
            })
        );
        // kind mismatch: a noul outcome against a choice question
        let mut resp = golden_response();
        resp.answers[0].outcome = Some(Outcome::Noul { yes: true });
        assert_eq!(
            resp.validate_against(&req),
            Err(WireError::OutcomeKindMismatch { at: 0 })
        );
    }

    #[test]
    fn response_validation_refuses_bad_distributions() {
        let req = golden_request();
        // arity mismatch
        let mut resp = golden_response();
        resp.answers[0].probabilities = vec![0.5, 0.5];
        assert_eq!(
            resp.validate_against(&req),
            Err(WireError::ProbabilityArityMismatch {
                at: 0,
                got: 2,
                want: 3
            })
        );
        // non-finite probability
        let mut resp = golden_response();
        resp.answers[0].probabilities = vec![0.5, f32::NAN, 0.5];
        assert_eq!(
            resp.validate_against(&req),
            Err(WireError::ProbabilityNotFinite { at: 0, option: 1 })
        );
        // probability outside [0, 1]
        let mut resp = golden_response();
        resp.answers[2].probabilities[0] = 1.5;
        assert_eq!(
            resp.validate_against(&req),
            Err(WireError::ProbabilityOutOfRange { at: 2, option: 0 })
        );
        // confidence out of range / non-finite
        let mut resp = golden_response();
        resp.answers[0].confidence = 1.5;
        assert_eq!(resp.validate_against(&req), Err(WireError::ConfidenceOutOfRange));
        let mut resp = golden_response();
        resp.answers[0].confidence = f32::INFINITY;
        assert_eq!(resp.validate_against(&req), Err(WireError::ConfidenceNotFinite));
    }

    #[test]
    fn abstained_answer_may_carry_or_omit_the_distribution() {
        let req = golden_request();
        // Abstain-before-score: empty distribution is legal.
        let mut resp = golden_response();
        resp.answers[1].probabilities.clear();
        resp.validate_against(&req)
            .expect("abstained with no distribution must validate");
        // Abstain-after-score: full-arity distribution rides along.
        let mut resp = golden_response();
        resp.answers[1].probabilities = vec![0.875];
        resp.validate_against(&req)
            .expect("abstained with distribution must validate");
        // …but a WRONG arity still refuses even when abstained.
        resp.answers[1].probabilities = vec![0.5, 0.5];
        assert_eq!(
            resp.validate_against(&req),
            Err(WireError::ProbabilityArityMismatch {
                at: 1,
                got: 2,
                want: 1
            })
        );
    }

    #[test]
    fn constructors_produce_self_consistent_answers() {
        let q_choice = Question::choice("c", "pick", vec!["a".into(), "b".into()], None);
        let q_score = Question::score("s", "rate", vec!["1".into(), "2".into()]);
        let q_noul = Question::noul("n", "yes or no");
        q_choice.validate(0).expect("choice question valid");
        q_score.validate(0).expect("score question valid");
        q_noul.validate(0).expect("noul question valid");
        Answer::choice("c", 0, vec![0.75, 0.25], 0.75)
            .validate_against(&q_choice, 0)
            .expect("choice constructor must self-validate");
        Answer::score("s", 1, vec![0.25, 0.75], 0.75)
            .validate_against(&q_score, 0)
            .expect("score constructor must self-validate");
        Answer::noul("n", true, 0.875, 0.875)
            .validate_against(&q_noul, 0)
            .expect("noul constructor must self-validate");
        Answer::abstain("n", 0.25)
            .validate_against(&q_noul, 0)
            .expect("abstain constructor must self-validate");
        assert_eq!(q_noul.probability_arity(), 1, "noul carries exactly p(yes)");
        assert_eq!(
            QuestionKind::Choice.as_str(),
            "choice",
            "wire spelling is the serde tag"
        );
    }
}
