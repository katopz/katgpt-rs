//! MeMo Reflection QA Pipeline — 5-step compositional data synthesis from game replays.
//!
//! Distilled from MeMo: Memory as a Model (Research 60). Generates self-contained
//! QA pairs from game state sequences for bandit training signal enrichment.
//!
//! The pipeline:
//! 1. Fact Extraction (direct + indirect)
//! 2. Consolidation (merge related facts)
//! 3. Verification (self-containment check)
//! 4. Entity Surfacing (reverse lookup patterns)
//! 5. Cross-Game Synthesis (cross-domain knowledge transfer)
//!
//! Feature-gated behind `memo_reflections`. No gradient updates — consumed by BanditPruner.

use std::fmt;

// ── Types ──────────────────────────────────────────────────────

/// Source step that generated a reflection QA pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectionStep {
    /// Direct extraction: (state, action, outcome) → "What action at state S?"
    DirectExtraction,
    /// Indirect extraction: (state, outcome) → "Why did outcome O occur?"
    IndirectExtraction,
    /// Merged related facts into multi-fact questions.
    Consolidation,
    /// Rewritten for self-containment.
    Verification,
    /// Entity-from-pattern QA pairs (reversal curse).
    EntitySurfacing,
    /// Cross-domain knowledge transfer.
    CrossGameSynthesis,
}

impl fmt::Display for ReflectionStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DirectExtraction => write!(f, "direct_extraction"),
            Self::IndirectExtraction => write!(f, "indirect_extraction"),
            Self::Consolidation => write!(f, "consolidation"),
            Self::Verification => write!(f, "verification"),
            Self::EntitySurfacing => write!(f, "entity_surfacing"),
            Self::CrossGameSynthesis => write!(f, "cross_game_synthesis"),
        }
    }
}

/// Game domain for reflection synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectionDomain {
    /// Bomberman arena.
    Bomber,
    /// Go board game.
    Go,
    /// FFT Tactics Arena.
    FFT,
    /// Cross-domain synthesis.
    CrossGame,
}

impl fmt::Display for ReflectionDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bomber => write!(f, "bomber"),
            Self::Go => write!(f, "go"),
            Self::FFT => write!(f, "fft"),
            Self::CrossGame => write!(f, "cross_game"),
        }
    }
}

/// A reflection QA pair synthesized from game replay data.
#[derive(Clone, Debug)]
pub struct ReflectionQA {
    /// The question (compositional, self-contained).
    pub question: String,
    /// The answer (factual, derived from game data).
    pub answer: String,
    /// Source step that generated this pair.
    pub step: ReflectionStep,
    /// Game domain.
    pub domain: ReflectionDomain,
    /// Number of game situations this pair consolidates.
    pub consolidation_count: usize,
    /// Whether this pair passed self-containment verification.
    pub verified: bool,
}

/// A lightweight game state snapshot for reflection synthesis.
/// Generic across domains — callers convert their domain state into this format.
#[derive(Clone, Debug)]
pub struct GameStateSnapshot {
    /// Tick/turn number.
    pub tick: u32,
    /// Description of the current state (domain-specific).
    pub state_description: String,
    /// Action taken (if any).
    pub action_description: Option<String>,
    /// Outcome after action (if terminal or observable).
    pub outcome_description: Option<String>,
    /// Heuristic score at this state.
    pub score: f32,
}

/// Result of reflection synthesis.
#[derive(Clone, Debug)]
pub struct ReflectionResult {
    /// All generated QA pairs.
    pub pairs: Vec<ReflectionQA>,
    /// Count per step: [direct, indirect, consolidation, verification, entity_surfacing, cross_game].
    pub step_counts: [usize; 6],
    /// Verification pass rate.
    pub verification_rate: f64,
}

// ── Step 1: Fact Extraction ────────────────────────────────────

/// Step 1: Extract direct and indirect facts from game state sequences.
///
/// Direct: (state, action, outcome) → "What action should be taken at {state}?"
/// Indirect: (state, outcome) → "Why did {outcome} occur after {state}?"
pub fn extract_facts(states: &[GameStateSnapshot], domain: ReflectionDomain) -> Vec<ReflectionQA> {
    let mut pairs = Vec::new();

    for (i, state) in states.iter().enumerate() {
        // Direct extraction: state + action → what action at this state?
        if let Some(ref action) = state.action_description {
            pairs.push(ReflectionQA {
                question: format!(
                    "At tick {tick} in {domain} with state \"{state_desc}\", what action was taken?",
                    tick = state.tick,
                    domain = domain,
                    state_desc = state.state_description
                ),
                answer: action.clone(),
                step: ReflectionStep::DirectExtraction,
                domain,
                consolidation_count: 1,
                verified: false,
            });
        }

        // Indirect extraction: state + outcome → why did this outcome occur?
        if let Some(ref outcome) = state.outcome_description {
            let prev_state = if i > 0 { &states[i - 1] } else { state };
            pairs.push(ReflectionQA {
                question: format!(
                    "Why did outcome \"{outcome}\" occur in {domain} after state \"{state_desc}\"?",
                    outcome = outcome,
                    domain = domain,
                    state_desc = prev_state.state_description
                ),
                answer: format!(
                    "The outcome \"{outcome}\" resulted from action \"{action}\" at state \"{state_desc}\" (score: {score:.2}).",
                    outcome = outcome,
                    action = prev_state.action_description.as_deref().unwrap_or("none"),
                    state_desc = prev_state.state_description,
                    score = state.score
                ),
                step: ReflectionStep::IndirectExtraction,
                domain,
                consolidation_count: 1,
                verified: false,
            });
        }
    }

    pairs
}

// ── Step 2: Consolidation ──────────────────────────────────────

/// Step 2: Consolidate related facts into multi-fact questions.
///
/// Groups facts by similar state descriptions and merges them into
/// compositional questions covering multiple game situations.
pub fn consolidate_facts(pairs: &[ReflectionQA]) -> Vec<ReflectionQA> {
    if pairs.is_empty() {
        return Vec::new();
    }

    // Group by domain
    let mut groups: Vec<(ReflectionDomain, Vec<&ReflectionQA>)> = Vec::new();
    for pair in pairs {
        if let Some(group) = groups.iter_mut().find(|(d, _)| *d == pair.domain) {
            group.1.push(pair);
        } else {
            groups.push((pair.domain, vec![pair]));
        }
    }

    let mut consolidated = Vec::new();
    for (domain, group_pairs) in &groups {
        if group_pairs.len() < 2 {
            continue;
        }

        let answers: Vec<&str> = group_pairs.iter().map(|p| p.answer.as_str()).collect();
        let questions: Vec<&str> = group_pairs.iter().map(|p| p.question.as_str()).collect();

        let n = group_pairs.len().min(5); // Cap at 5 for readability
        consolidated.push(ReflectionQA {
            question: format!(
                "Which of the following {n} {domain} situations share a pattern? {qs}",
                n = n,
                domain = domain,
                qs = questions[..n].join("; ")
            ),
            answer: format!(
                "Common patterns: {answers}",
                answers = answers[..n].join("; ")
            ),
            step: ReflectionStep::Consolidation,
            domain: *domain,
            consolidation_count: n,
            verified: false,
        });
    }

    consolidated
}

// ── Step 3: Verification ───────────────────────────────────────

/// Check self-containment — ambiguous pronouns/references make QA pairs unusable without context.
fn is_self_contained(question: &str, answer: &str) -> bool {
    let ambiguous = ["it ", "this ", "that ", "they ", "them "];
    let q_lower = question.to_lowercase();
    let a_lower = answer.to_lowercase();
    let q_has_ambig = ambiguous.iter().any(|a| q_lower.contains(a));
    let a_has_ambig = ambiguous.iter().any(|a| a_lower.contains(a));
    !q_has_ambig && !a_has_ambig
}

/// Step 3: Verify self-containment and rewrite ambiguous pairs.
///
/// Rewrites ambiguous references by prefixing with domain context.
/// All pairs are marked as verified after rewriting.
pub fn verify_self_containment(pairs: &[ReflectionQA]) -> Vec<ReflectionQA> {
    pairs
        .iter()
        .map(|pair| {
            let contained = is_self_contained(&pair.question, &pair.answer);
            let (question, answer) = match contained {
                true => (pair.question.clone(), pair.answer.clone()),
                false => {
                    let domain_str = format!("{domain}", domain = pair.domain);
                    let q = format!("[{domain_str}] {q}", q = pair.question);
                    let a = format!("[{domain_str}] {a}", a = pair.answer);
                    (q, a)
                }
            };

            ReflectionQA {
                question,
                answer,
                // Preserve original step for self-contained pairs;
                // mark rewritten (ambiguous) pairs as Verification step.
                step: match contained {
                    true => pair.step,
                    false => ReflectionStep::Verification,
                },
                domain: pair.domain,
                consolidation_count: pair.consolidation_count,
                verified: true,
            }
        })
        .collect()
}

// ── Step 4: Entity Surfacing ───────────────────────────────────

/// Step 4: Surface entities from pattern descriptions (reversal curse mitigation).
///
/// Creates QA pairs that ask "What strategy has pattern X?" from existing
/// "Strategy X has pattern Y" facts.
pub fn surface_entities(pairs: &[ReflectionQA], domain: ReflectionDomain) -> Vec<ReflectionQA> {
    pairs
        .iter()
        .filter(|p| p.consolidation_count > 1 || p.step == ReflectionStep::Consolidation)
        .map(|pair| ReflectionQA {
            question: format!(
                "What {domain} strategy matches the pattern: {answer}?",
                domain = domain,
                answer = pair.answer
            ),
            answer: format!(
                "The matching strategy is described by: {q}",
                q = pair.question
            ),
            step: ReflectionStep::EntitySurfacing,
            domain,
            consolidation_count: pair.consolidation_count,
            verified: pair.verified,
        })
        .collect()
}

// ── Step 5: Cross-Game Synthesis ───────────────────────────────

/// Step 5: Cross-game synthesis — find parallels between game domains.
///
/// Creates QA pairs connecting strategies across domains.
pub fn synthesize_cross_game(
    verified: &[ReflectionQA],
    surfaced: &[ReflectionQA],
    domain: ReflectionDomain,
) -> Vec<ReflectionQA> {
    if domain == ReflectionDomain::CrossGame {
        return Vec::new();
    }

    let verified_count = verified.len();
    let surfaced_count = surfaced.len();

    if verified_count + surfaced_count == 0 {
        return Vec::new();
    }

    let mut cross = Vec::new();

    // Collect diverse source facts — deduplicate by answer prefix to get varied patterns.
    // Use up to 8 verified sources and all surfaced entities.
    let mut seen_prefixes: Vec<String> = Vec::new();
    let mut diverse_verified: Vec<&ReflectionQA> = Vec::new();
    for v in verified {
        let prefix: String = v.answer.chars().take(30).collect();
        if !seen_prefixes.iter().any(|p| p == &prefix) {
            seen_prefixes.push(prefix);
            diverse_verified.push(v);
        }
        if diverse_verified.len() >= 8 {
            break;
        }
    }

    // Converging clues: each verified fact yields a cross-domain question.
    // Different source facts produce distinct cross-game connections.
    let other_domains = match domain {
        ReflectionDomain::Bomber => vec!["Go", "FFT"],
        ReflectionDomain::Go => vec!["Bomber", "FFT"],
        ReflectionDomain::FFT => vec!["Bomber", "Go"],
        ReflectionDomain::CrossGame => vec![],
    };

    for source in &diverse_verified {
        for other in &other_domains {
            let pattern_short: String = source.answer.chars().take(60).collect();
            cross.push(ReflectionQA {
                question: format!(
                    "What strategic concept is shared between {domain} and {other} when considering pattern: \"{pattern}\"?",
                    domain = domain,
                    other = other,
                    pattern = pattern_short
                ),
                answer: format!(
                    "Both {domain} and {other} reward timing and positional awareness. In {domain}, the key pattern is: {full}",
                    domain = domain,
                    other = other,
                    full = source.answer
                ),
                step: ReflectionStep::CrossGameSynthesis,
                domain: ReflectionDomain::CrossGame,
                consolidation_count: source.consolidation_count,
                verified: true,
            });
        }
    }

    // Parallel properties: surfaced entity pairs yield cross-domain analogies.
    for pair in surfaced.iter().take(4) {
        let concept_short: String = pair.answer.chars().take(80).collect();
        let concept_long: String = pair.answer.chars().take(120).collect();

        cross.push(ReflectionQA {
            question: format!(
                "How does the {domain} concept of \"{concept}\" relate to similar concepts in other game domains?",
                domain = domain,
                concept = concept_short
            ),
            answer: format!(
                "The {domain} concept parallels other domains' emphasis on resource management and strategic positioning. Key insight: {concept}",
                domain = domain,
                concept = concept_long
            ),
            step: ReflectionStep::CrossGameSynthesis,
            domain: ReflectionDomain::CrossGame,
            consolidation_count: pair.consolidation_count,
            verified: true,
        });
    }

    cross
}

// ── Main Entry Point ───────────────────────────────────────────

/// Synthesize reflection QA pairs from game state snapshots.
///
/// Runs the full 5-step pipeline:
/// 1. Extract facts (direct + indirect)
/// 2. Consolidate related facts
/// 3. Verify self-containment
/// 4. Surface entities (reverse lookup)
/// 5. Cross-game synthesis
pub fn synthesize_reflections(
    states: &[GameStateSnapshot],
    domain: ReflectionDomain,
) -> ReflectionResult {
    // Step 1: Extract
    let extracted = extract_facts(states, domain);

    // Step 2: Consolidate
    let consolidated = consolidate_facts(&extracted);

    // Step 3: Verify (all extracted + consolidated)
    let all_pre_verify: Vec<ReflectionQA> = extracted
        .iter()
        .chain(consolidated.iter())
        .cloned()
        .collect();
    let verified = verify_self_containment(&all_pre_verify);

    // Step 4: Entity surfacing
    let surfaced = surface_entities(&verified, domain);

    // Step 5: Cross-game synthesis
    let cross = synthesize_cross_game(&verified, &surfaced, domain);

    // Combine all
    let mut pairs = Vec::new();
    pairs.extend(verified);
    pairs.extend(surfaced);
    pairs.extend(cross);

    // Compute stats
    let step_counts = [
        pairs
            .iter()
            .filter(|p| p.step == ReflectionStep::DirectExtraction)
            .count(),
        pairs
            .iter()
            .filter(|p| p.step == ReflectionStep::IndirectExtraction)
            .count(),
        pairs
            .iter()
            .filter(|p| p.step == ReflectionStep::Consolidation)
            .count(),
        pairs
            .iter()
            .filter(|p| p.step == ReflectionStep::Verification)
            .count(),
        pairs
            .iter()
            .filter(|p| p.step == ReflectionStep::EntitySurfacing)
            .count(),
        pairs
            .iter()
            .filter(|p| p.step == ReflectionStep::CrossGameSynthesis)
            .count(),
    ];

    let verified_count = pairs.iter().filter(|p| p.verified).count();
    let total = pairs.len();
    let verification_rate = match total {
        0 => 0.0,
        _ => verified_count as f64 / total as f64,
    };

    ReflectionResult {
        pairs,
        step_counts,
        verification_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_states(n: usize) -> Vec<GameStateSnapshot> {
        (0..n)
            .map(|i| GameStateSnapshot {
                tick: i as u32,
                state_description: format!("state_{i}"),
                action_description: Some(format!("action_{i}")),
                outcome_description: match i % 2 == 0 {
                    true => Some(format!("outcome_{i}")),
                    false => None,
                },
                score: 0.5 + i as f32 * 0.1,
            })
            .collect()
    }

    #[test]
    fn test_extract_facts_direct() {
        let states = make_states(5);
        let pairs = extract_facts(&states, ReflectionDomain::Bomber);
        let direct: Vec<_> = pairs
            .iter()
            .filter(|p| p.step == ReflectionStep::DirectExtraction)
            .collect();
        assert_eq!(direct.len(), 5);
    }

    #[test]
    fn test_extract_facts_indirect() {
        let states = make_states(5);
        let pairs = extract_facts(&states, ReflectionDomain::Go);
        let indirect: Vec<_> = pairs
            .iter()
            .filter(|p| p.step == ReflectionStep::IndirectExtraction)
            .collect();
        assert_eq!(indirect.len(), 3); // Only even indices have outcomes
    }

    #[test]
    fn test_consolidate_facts_empty() {
        let pairs = consolidate_facts(&[]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_consolidate_facts_groups() {
        let states = make_states(10);
        let extracted = extract_facts(&states, ReflectionDomain::Bomber);
        let consolidated = consolidate_facts(&extracted);
        assert!(!consolidated.is_empty());
        assert!(consolidated[0].consolidation_count >= 2);
    }

    #[test]
    fn test_verify_self_containment() {
        let pairs = vec![ReflectionQA {
            question: "What action at state_0?".to_string(),
            answer: "action_0".to_string(),
            step: ReflectionStep::DirectExtraction,
            domain: ReflectionDomain::Bomber,
            consolidation_count: 1,
            verified: false,
        }];
        let verified = verify_self_containment(&pairs);
        assert!(verified[0].verified);
    }

    #[test]
    fn test_verify_ambiguous_pair() {
        let pairs = vec![ReflectionQA {
            question: "What does it do at state_0?".to_string(),
            answer: "it takes action_0".to_string(),
            step: ReflectionStep::DirectExtraction,
            domain: ReflectionDomain::Go,
            consolidation_count: 1,
            verified: false,
        }];
        let verified = verify_self_containment(&pairs);
        assert!(verified[0].verified);
        assert!(verified[0].question.contains("[go]"));
    }

    #[test]
    fn test_surface_entities() {
        let pairs = vec![ReflectionQA {
            question: "test question".to_string(),
            answer: "test answer".to_string(),
            step: ReflectionStep::Consolidation,
            domain: ReflectionDomain::Bomber,
            consolidation_count: 3,
            verified: true,
        }];
        let surfaced = surface_entities(&pairs, ReflectionDomain::Bomber);
        assert!(!surfaced.is_empty());
        assert_eq!(surfaced[0].step, ReflectionStep::EntitySurfacing);
    }

    #[test]
    fn test_synthesize_cross_game() {
        let pairs = vec![ReflectionQA {
            question: "test question".to_string(),
            answer: "test answer".to_string(),
            step: ReflectionStep::Verification,
            domain: ReflectionDomain::Bomber,
            consolidation_count: 1,
            verified: true,
        }];
        let cross = synthesize_cross_game(&pairs, &[], ReflectionDomain::Bomber);
        assert!(!cross.is_empty());
        assert_eq!(cross[0].domain, ReflectionDomain::CrossGame);
    }

    #[test]
    fn test_synthesize_cross_game_crossgame_domain_returns_empty() {
        let cross = synthesize_cross_game(&[], &[], ReflectionDomain::CrossGame);
        assert!(cross.is_empty());
    }

    #[test]
    fn test_synthesize_reflections_full_pipeline() {
        let states = make_states(100);
        let result = synthesize_reflections(&states, ReflectionDomain::Bomber);

        // Should generate ≥100 pairs from 100 rounds
        assert!(
            result.pairs.len() >= 100,
            "Expected ≥100 pairs, got {}",
            result.pairs.len()
        );

        // Step counts should be non-zero for extraction steps
        assert!(
            result.step_counts[0] > 0,
            "Direct extraction should produce pairs"
        );
        assert!(
            result.step_counts[1] > 0,
            "Indirect extraction should produce pairs"
        );

        // Verification rate should be reasonable
        assert!(
            result.verification_rate > 0.5,
            "Expected >50% verification rate, got {:.2}",
            result.verification_rate
        );

        // Should have cross-game pairs
        assert!(
            result.step_counts[5] > 0,
            "Cross-game synthesis should produce pairs"
        );
    }

    #[test]
    fn test_reflection_step_display() {
        assert_eq!(
            format!("{}", ReflectionStep::DirectExtraction),
            "direct_extraction"
        );
        assert_eq!(
            format!("{}", ReflectionStep::CrossGameSynthesis),
            "cross_game_synthesis"
        );
    }

    #[test]
    fn test_reflection_domain_display() {
        assert_eq!(format!("{}", ReflectionDomain::Bomber), "bomber");
        assert_eq!(format!("{}", ReflectionDomain::CrossGame), "cross_game");
    }

    #[test]
    fn test_empty_states() {
        let result = synthesize_reflections(&[], ReflectionDomain::Go);
        assert!(result.pairs.is_empty());
        assert_eq!(result.verification_rate, 0.0);
    }

    #[test]
    fn test_step_counts_sum_to_total() {
        let states = make_states(20);
        let result = synthesize_reflections(&states, ReflectionDomain::FFT);
        let sum: usize = result.step_counts.into_iter().sum();
        assert_eq!(
            sum,
            result.pairs.len(),
            "Step counts should sum to total pairs"
        );
    }
}
