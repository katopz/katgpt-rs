//! Neuro-Symbolic RIIR Feedback Loop — rule extraction from translations (Plan 214 P4).
//!
//! Extracts translation rules from successful RIIR (Rewrite It In Rust) translations,
//! tracks Curator rule performance via bandit refinement, and classifies workload
//! routing for CPU vs async execution (P5).
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "coexplain_riir")]`.

use std::collections::HashMap;

// ── TranslationRule ─────────────────────────────────────────────────

/// A translation rule extracted from successful RIIR translations.
///
/// Each rule captures a DDTree path that led to compilable output,
/// with success/failure counts for bandit-based refinement.
#[derive(Debug, Clone)]
pub struct TranslationRule {
    /// DDTree path that led to compilable output.
    pub path: Vec<usize>,
    /// Number of times this path produced compilable output.
    pub successes: u32,
    /// Number of times this path failed to compile.
    pub failures: u32,
    /// Blake3 hash of the path for deduplication.
    pub path_hash: [u8; 32],
}

// ── Rule Extraction ─────────────────────────────────────────────────

/// Extract rules from successful and failed translations.
///
/// For each successful path, creates a `TranslationRule` with blake3 deduplication.
/// Failed paths that match a successful path increment the failure count.
///
/// # Arguments
///
/// * `successful_paths` — DDTree paths that produced compilable output
/// * `failed_paths`     — DDTree paths that failed to compile
///
/// # Returns
///
/// Deduplicated translation rules sorted by success count (descending).
pub fn extract_translation_rules(
    successful_paths: &[Vec<usize>],
    failed_paths: &[Vec<usize>],
) -> Vec<TranslationRule> {
    // Index by blake3 hash → (path, successes, failures)
    let mut rules: HashMap<[u8; 32], (Vec<usize>, u32, u32)> = HashMap::new();

    // Count successes
    for path in successful_paths {
        let hash = hash_path(path);
        let entry = rules.entry(hash).or_insert_with(|| (path.clone(), 0, 0));
        entry.1 += 1;
    }

    // Count failures
    for path in failed_paths {
        let hash = hash_path(path);
        if let Some(entry) = rules.get_mut(&hash) {
            entry.2 += 1;
        }
        // Ignore failed paths that never succeeded — no rule to create
    }

    // Build results sorted by success count descending
    let mut result: Vec<TranslationRule> = rules
        .into_iter()
        .map(|(path_hash, (path, successes, failures))| TranslationRule {
            path,
            successes,
            failures,
            path_hash,
        })
        .collect();

    result.sort_by(|a, b| b.successes.cmp(&a.successes));
    result
}

/// Compute blake3 hash of a DDTree path.
fn hash_path(path: &[usize]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for &idx in path {
        hasher.update(&idx.to_le_bytes());
    }
    let hash = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}

// ── CuratorRule ─────────────────────────────────────────────────────

/// Curator rule from external source (Curator marketplace, user, or auto-extracted).
#[derive(Debug, Clone)]
pub struct CuratorRule {
    /// Human-readable rule name.
    pub name: String,
    /// Rule definition in JSON format.
    pub rule: String,
    /// Source of the rule: "curator", "user", or "auto".
    pub source: String,
}

// ── CuratorIngestion ────────────────────────────────────────────────

/// Placeholder for Curator rule ingestion.
///
/// Accumulates pending rules until they are drained for processing.
/// Full implementation depends on riir-ai Curator API.
pub struct CuratorIngestion {
    /// Rules waiting to be processed.
    pub pending_rules: Vec<CuratorRule>,
}

impl CuratorIngestion {
    /// Create a new empty ingestion buffer.
    pub fn new() -> Self {
        Self {
            pending_rules: Vec::new(),
        }
    }

    /// Add a rule to the pending queue.
    pub fn ingest(&mut self, rule: CuratorRule) {
        self.pending_rules.push(rule);
    }

    /// Take all pending rules, leaving the buffer empty.
    pub fn drain(&mut self) -> Vec<CuratorRule> {
        std::mem::take(&mut self.pending_rules)
    }
}

impl Default for CuratorIngestion {
    fn default() -> Self {
        Self::new()
    }
}

// ── RuleBandit ──────────────────────────────────────────────────────

/// Track translation success rate per Curator rule.
///
/// Uses a simple (successes, failures) counter per rule name,
/// suitable for epsilon-greedy or UCB bandit strategies.
pub struct RuleBandit {
    /// Rule name → (successes, failures).
    pub stats: HashMap<String, (u32, u32)>,
}

impl RuleBandit {
    /// Create a new bandit with no tracked rules.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// Record a translation outcome for a rule.
    pub fn record(&mut self, rule_name: &str, success: bool) {
        let entry = self.stats.entry(rule_name.to_string()).or_insert((0, 0));
        if success {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    /// Compute success rate for a rule: successes / (successes + failures).
    ///
    /// Returns 0.0 if the rule has never been used.
    pub fn success_rate(&self, rule_name: &str) -> f32 {
        match self.stats.get(rule_name) {
            Some(&(s, f)) => {
                let total = s + f;
                match total {
                    0 => 0.0,
                    _ => s as f32 / total as f32,
                }
            }
            None => 0.0,
        }
    }

    /// Select the best-performing rule by success rate.
    ///
    /// Ties broken by total usage count (more data = more reliable).
    /// Returns `None` if no rules have been tracked.
    pub fn best_rule(&self) -> Option<String> {
        self.stats
            .iter()
            .filter(|(_, (s, f))| *s + *f > 0)
            .max_by(|a, b| {
                let rate_a = a.1.0 as f32 / (a.1.0 + a.1.1) as f32;
                let rate_b = b.1.0 as f32 / (b.1.0 + b.1.1) as f32;
                rate_a
                    .partial_cmp(&rate_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| (a.1.0 + a.1.1).cmp(&(b.1.0 + b.1.1)))
            })
            .map(|(name, _)| name.clone())
    }
}

impl Default for RuleBandit {
    fn default() -> Self {
        Self::new()
    }
}

// ── WorkloadRoute (P5) ──────────────────────────────────────────────

/// Route classification for CoExplain workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkloadRoute {
    /// CPU: lightweight O(1) bandit updates.
    Cpu,
    /// Async worker: WASM compilation.
    AsyncWorker,
}

/// Classify workload routing based on task type.
///
/// - CPU: bandit updates, TED-Lite (lightweight, O(1) per token)
/// - AsyncWorker: rule/WASM compilation (CPU-bound but infrequent)
pub fn classify_workload(task: &str) -> WorkloadRoute {
    match task {
        "bandit_update" | "ted_lite" => WorkloadRoute::Cpu,
        "rule_compile" | "wasm_compile" => WorkloadRoute::AsyncWorker,
        _ => WorkloadRoute::Cpu,
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_translation_rules_dedup() {
        let successful = vec![
            vec![0, 1, 2],
            vec![0, 1, 2], // duplicate
            vec![3, 4],
        ];
        let failed: Vec<Vec<usize>> = vec![];

        let rules = extract_translation_rules(&successful, &failed);
        assert_eq!(rules.len(), 2, "should deduplicate identical paths");

        // Most successful first
        assert_eq!(rules[0].path, vec![0, 1, 2]);
        assert_eq!(rules[0].successes, 2);
        assert_eq!(rules[1].path, vec![3, 4]);
        assert_eq!(rules[1].successes, 1);
    }

    #[test]
    fn test_extract_translation_rules_counts() {
        let successful = vec![vec![0, 1]];
        let failed = vec![
            vec![0, 1], // same path failed once
            vec![5, 6], // never succeeded → ignored
        ];

        let rules = extract_translation_rules(&successful, &failed);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].successes, 1);
        assert_eq!(rules[0].failures, 1);
    }

    #[test]
    fn test_curator_ingestion_drain() {
        let mut ingestion = CuratorIngestion::new();
        assert!(ingestion.pending_rules.is_empty());

        ingestion.ingest(CuratorRule {
            name: "rule_a".to_string(),
            rule: r#"{"action":"reject"}"#.to_string(),
            source: "curator".to_string(),
        });
        ingestion.ingest(CuratorRule {
            name: "rule_b".to_string(),
            rule: r#"{"action":"accept"}"#.to_string(),
            source: "user".to_string(),
        });
        assert_eq!(ingestion.pending_rules.len(), 2);

        let drained = ingestion.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].name, "rule_a");
        assert_eq!(drained[1].source, "user");
        assert!(ingestion.pending_rules.is_empty());

        // Second drain returns empty
        let drained2 = ingestion.drain();
        assert!(drained2.is_empty());
    }

    #[test]
    fn test_rule_bandit_success_rate() {
        let mut bandit = RuleBandit::new();

        // Unknown rule → 0.0
        assert_eq!(bandit.success_rate("unknown"), 0.0);

        bandit.record("rule_a", true);
        bandit.record("rule_a", true);
        bandit.record("rule_a", false);
        // 2/3 ≈ 0.667
        assert!((bandit.success_rate("rule_a") - 0.6667).abs() < 0.01);

        bandit.record("rule_b", false);
        bandit.record("rule_b", false);
        // 0/2 = 0.0
        assert_eq!(bandit.success_rate("rule_b"), 0.0);
    }

    #[test]
    fn test_rule_bandit_best_rule() {
        let mut bandit = RuleBandit::new();

        // No rules → None
        assert!(bandit.best_rule().is_none());

        bandit.record("rule_a", true);
        bandit.record("rule_a", false); // 50%

        bandit.record("rule_b", true);
        bandit.record("rule_b", true); // 100%

        bandit.record("rule_c", false);
        bandit.record("rule_c", false); // 0%

        assert_eq!(bandit.best_rule().as_deref(), Some("rule_b"));
    }

    #[test]
    fn test_classify_workload_cpu() {
        assert_eq!(classify_workload("bandit_update"), WorkloadRoute::Cpu);
        assert_eq!(classify_workload("ted_lite"), WorkloadRoute::Cpu);
        assert_eq!(classify_workload("unknown_task"), WorkloadRoute::Cpu);
    }

    #[test]
    fn test_classify_workload_async() {
        assert_eq!(
            classify_workload("rule_compile"),
            WorkloadRoute::AsyncWorker
        );
        assert_eq!(
            classify_workload("wasm_compile"),
            WorkloadRoute::AsyncWorker
        );
    }

    #[test]
    fn test_translation_rule_hash_deterministic() {
        let path = vec![1, 2, 3];
        let hash1 = hash_path(&path);
        let hash2 = hash_path(&path);
        assert_eq!(hash1, hash2, "same path must produce same hash");

        let different_path = vec![1, 2, 4];
        let hash3 = hash_path(&different_path);
        assert_ne!(
            hash1, hash3,
            "different paths must produce different hashes"
        );
    }

    #[test]
    fn test_extract_translation_rules_empty() {
        let rules = extract_translation_rules(&[], &[]);
        assert!(rules.is_empty());
    }
}
