//! PolicyEnvelope — ARG Step 1 Policy Pre-Check primitive.
//!
//! Distilled from ARG §2.1, §2.1 (Policy Manager PM-1). The PolicyEnvelope is
//! the *hard gate* envelope produced before any routing / retrieval / traversal
//! / action / write. Subsequent steps consume `PolicyConstraints` as hard
//! filters; they do NOT replace the Context Weaver (ARG §1 non-negotiable).
//!
//! Pure data + zero-alloc evaluation. No external services, no LLM. The
//! `PolicyConstraints` are passed by reference (caller owns the slice); the
//! envelope is a thin `(state, &constraints, response_mode)` record.

/// Stable label identifier — never recycled, never reused after `Removed`.
/// Distinct type from `u32` so it cannot be confused with taxonomy indices.
pub use super::taxonomy::LabelId;

/// `policy_state ∈ {ALLOW, ALLOW_WITH_REFOCUS, RESTRICT, BLOCK}`.
///
/// ARG §PM-1 outputs. The four states map to:
/// - `Allow` — proceed without modification.
/// - `AllowWithRefocus` — short refocus response + stop unless policy
///   explicitly allows continued processing under constraints.
/// - `Restrict` — proceed under `constraints`; may be forced into ABSTAIN/CLARIFY
///   depending on what is permitted.
/// - `Block` — immediate stop; refuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PolicyState {
    /// Proceed without modification.
    #[default]
    Allow = 0,
    /// Short refocus response + stop unless policy explicitly allows continued processing.
    AllowWithRefocus = 1,
    /// Proceed under constraints; may force ABSTAIN/CLARIFY.
    Restrict = 2,
    /// Immediate stop; refuse.
    Block = 3,
}

impl PolicyState {
    /// Returns `true` when this state mandates an immediate stop (ARG §2.1: `BLOCK → stop`).
    #[inline]
    pub fn must_stop(self) -> bool {
        matches!(self, PolicyState::Block)
    }

    /// Returns `true` when constraints MUST be applied to all downstream steps
    /// (ARG §2.1: `RESTRICT → proceed under constraints`).
    #[inline]
    pub fn requires_constraints(self) -> bool {
        matches!(self, PolicyState::Restrict | PolicyState::AllowWithRefocus)
    }
}

/// `response_mode ∈ {Normal, Prudent, Refocus, Refusal}`.
///
/// ARG §Policy IO §2.3 — consumed by the response shaper (private side).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResponseMode {
    #[default]
    Normal = 0,
    Prudent = 1,
    Refocus = 2,
    Refusal = 3,
}

/// Cross-cutting governance constraints — the *hard filters* applied to binding,
/// traversal, neighbor scoring, action execution, and writes.
///
/// ARG §PM-2 constraint injection. Caller owns the slices (`allowed_labels`,
/// `forbidden_labels`); this struct is a thin borrow.
#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyConstraints<'a> {
    /// Labels the request is permitted to route to (`N_scope` contribution).
    /// Empty slice = "no allowlist enforcement" (not "no labels allowed").
    pub allowed_labels: &'a [LabelId],
    /// Labels the request is forbidden from routing to. Always enforced.
    pub forbidden_labels: &'a [LabelId],
    /// Max traversal hops permitted.
    pub max_hops: u8,
    /// Max traversal depth permitted.
    pub max_depth: u8,
    /// Max complexity cap (caller-defined metric; e.g. token budget / shard count).
    pub max_complexity: u16,
}

impl<'a> PolicyConstraints<'a> {
    /// Returns `true` if `label` is explicitly forbidden.
    #[inline]
    pub fn is_forbidden(&self, label: LabelId) -> bool {
        // Linear scan is correct here: forbidden lists are typically small (≤32).
        // For larger lists, the caller should pre-build a sorted slice + binary search.
        self.forbidden_labels.iter().any(|&l| l == label)
    }

    /// Returns `true` if `label` is allowed under the allowlist.
    /// Empty allowlist = permissive (no allowlist enforcement).
    #[inline]
    pub fn is_allowed(&self, label: LabelId) -> bool {
        self.allowed_labels.is_empty() || self.allowed_labels.iter().any(|&l| l == label)
    }
}

/// The envelope produced by PM-1 and consumed by all subsequent ARG online steps.
#[derive(Clone, Copy, Debug)]
pub struct PolicyEnvelope<'a> {
    pub state: PolicyState,
    pub constraints: PolicyConstraints<'a>,
    pub response_mode: ResponseMode,
}

/// Decision returned by [`PolicyEnvelope::evaluate`]: whether to short-circuit
/// the pipeline and whether constraints must be threaded downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShouldProceed {
    /// Pipeline may continue. Constraints still apply if `state = Restrict`.
    Continue,
    /// Refocus response emitted; stop unless policy explicitly allows continuation.
    Refocus,
    /// Hard stop. Pipeline MUST refuse.
    Stop,
}

/// Per-request evaluation output.
#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyDecision {
    pub proceed: ShouldProceed,
    /// Whether downstream steps must enforce `constraints`.
    pub enforce_constraints: bool,
}

impl Default for ShouldProceed {
    #[inline]
    fn default() -> Self {
        ShouldProceed::Continue
    }
}

impl<'a> PolicyEnvelope<'a> {
    /// Evaluate the envelope against a request context (zero-alloc).
    ///
    /// `requested_label` is the candidate primary label from initial
    /// classification (Step 2). If `None`, allow/forbid checks are skipped
    /// (the constraints still apply via `requires_constraints`).
    #[inline]
    pub fn evaluate(&self, requested_label: Option<LabelId>) -> PolicyDecision {
        // Fast path: BLOCK is always Stop.
        if self.state.must_stop() {
            return PolicyDecision {
                proceed: ShouldProceed::Stop,
                enforce_constraints: false,
            };
        }

        // Label-level gates (if a primary label was provided).
        if let Some(label) = requested_label {
            // Forbidden label under any state forces Stop (policy gate is hard).
            if self.constraints.is_forbidden(label) {
                return PolicyDecision {
                    proceed: ShouldProceed::Stop,
                    enforce_constraints: false,
                };
            }
            // Allowlist enforcement: a non-allowed label under Restrict forces Refocus.
            if !self.constraints.is_allowed(label)
                && matches!(self.state, PolicyState::Restrict)
            {
                return PolicyDecision {
                    proceed: ShouldProceed::Refocus,
                    enforce_constraints: true,
                };
            }
        }

        match self.state {
            PolicyState::Allow => PolicyDecision {
                proceed: ShouldProceed::Continue,
                enforce_constraints: false,
            },
            PolicyState::AllowWithRefocus => PolicyDecision {
                proceed: ShouldProceed::Refocus,
                enforce_constraints: true,
            },
            PolicyState::Restrict => PolicyDecision {
                proceed: ShouldProceed::Continue,
                enforce_constraints: true,
            },
            PolicyState::Block => PolicyDecision {
                proceed: ShouldProceed::Stop,
                enforce_constraints: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lbl(n: u32) -> LabelId {
        LabelId::new(n)
    }

    #[test]
    fn allow_state_continues_without_constraints() {
        let env = PolicyEnvelope {
            state: PolicyState::Allow,
            constraints: PolicyConstraints::default(),
            response_mode: ResponseMode::Normal,
        };
        let d = env.evaluate(Some(lbl(7)));
        assert_eq!(d.proceed, ShouldProceed::Continue);
        assert!(!d.enforce_constraints);
    }

    #[test]
    fn block_state_always_stops() {
        let env = PolicyEnvelope {
            state: PolicyState::Block,
            constraints: PolicyConstraints::default(),
            response_mode: ResponseMode::Refusal,
        };
        let d = env.evaluate(Some(lbl(7)));
        assert_eq!(d.proceed, ShouldProceed::Stop);
        assert!(!d.enforce_constraints);
    }

    #[test]
    fn forbidden_label_forces_stop_even_under_allow() {
        let forbidden = [lbl(42)];
        let env = PolicyEnvelope {
            state: PolicyState::Allow,
            constraints: PolicyConstraints {
                forbidden_labels: &forbidden,
                ..PolicyConstraints::default()
            },
            response_mode: ResponseMode::Normal,
        };
        assert_eq!(env.evaluate(Some(lbl(42))).proceed, ShouldProceed::Stop);
        assert_eq!(env.evaluate(Some(lbl(7))).proceed, ShouldProceed::Continue);
    }

    #[test]
    fn allowlist_under_restrict_refocuses_on_violation() {
        let allowed = [lbl(1), lbl(2), lbl(3)];
        let env = PolicyEnvelope {
            state: PolicyState::Restrict,
            constraints: PolicyConstraints {
                allowed_labels: &allowed,
                ..PolicyConstraints::default()
            },
            response_mode: ResponseMode::Prudent,
        };
        // In-allowlist label under Restrict → Continue (with constraints).
        let d_ok = env.evaluate(Some(lbl(2)));
        assert_eq!(d_ok.proceed, ShouldProceed::Continue);
        assert!(d_ok.enforce_constraints);
        // Out-of-allowlist label under Restrict → Refocus.
        let d_no = env.evaluate(Some(lbl(99)));
        assert_eq!(d_no.proceed, ShouldProceed::Refocus);
        assert!(d_no.enforce_constraints);
    }

    #[test]
    fn empty_allowlist_is_permissive() {
        let env = PolicyEnvelope {
            state: PolicyState::Allow,
            constraints: PolicyConstraints {
                allowed_labels: &[],
                ..PolicyConstraints::default()
            },
            response_mode: ResponseMode::Normal,
        };
        // Empty allowlist means "no allowlist enforcement", not "nothing allowed".
        assert_eq!(env.evaluate(Some(lbl(7))).proceed, ShouldProceed::Continue);
        assert!(env.constraints.is_allowed(lbl(7)));
    }

    #[test]
    fn restrict_state_threads_constraints_downstream() {
        let env = PolicyEnvelope {
            state: PolicyState::Restrict,
            constraints: PolicyConstraints::default(),
            response_mode: ResponseMode::Prudent,
        };
        let d = env.evaluate(Some(lbl(7)));
        assert_eq!(d.proceed, ShouldProceed::Continue);
        assert!(d.enforce_constraints);
    }
}
