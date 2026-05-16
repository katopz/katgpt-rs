//! Trial persistence for Heuristic Learning — JSONL episode history.
//!
//! Records every bandit episode to a JSONL file for offline analysis,
//! absorb-compress decisions, and regression testing.
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut log = TrialLog::new(Path::new("/tmp/trials.jsonl"))?;
//! log.append(&TrialRecord { episode: 0, arm: 2, reward: 1.0, ..Default::default() });
//! log.flush()?;
//!
//! let records = TrialLog::load(Path::new("/tmp/trials.jsonl"))?;
//! let summary = TrialLog::summary(&records);
//! ```

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Result, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::review_metrics::{ReviewMetrics, ReviewSummary};

// ── AnchorTrace (StepCodeReasoner Plan 054) ─────────────────────

/// Per-anchor verification trace for stepwise reward analysis.
///
/// Distilled from StepCodeReasoner's execution-trace anchors.
/// Each entry records what happened at one DDTree depth (one "anchor").
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AnchorTrace {
    /// Depth in DDTree (anchor position).
    pub depth: usize,
    /// Arm (token) selected at this depth.
    pub arm: usize,
    /// Flat binary reward (0.0 or 1.0).
    pub reward: f32,
    /// Shaped reward (reward × (1 + λ × future_accuracy)).
    pub shaped_reward: f32,
    /// Fraction of subsequent arms that were correct.
    pub future_accuracy: f32,
}

// ── Record ──────────────────────────────────────────────────────

/// A single episode record for persistent trial history.
///
/// Maps 1:1 with the HL article's `trials.jsonl` format.
/// Serialized as one JSON line per record.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TrialRecord {
    /// Episode index (0-based).
    pub episode: usize,
    /// Player/agent ID for multi-agent shared bandit (Issue 051 T4).
    /// Defaults to `0` when absent in legacy JSONL files.
    #[serde(default)]
    pub player_id: u32,
    /// Arm (action) selected.
    pub arm: usize,
    /// Observed reward ∈ [0.0, 1.0].
    pub reward: f32,
    /// Q-value estimate at time of selection.
    pub q_value: f32,
    /// Running sum of rewards through this episode.
    pub cumulative_reward: f32,
    /// Running sum of regret through this episode.
    pub cumulative_regret: f32,
    /// Configuration snapshot (strategy, env params, etc.).
    pub config: String,
    /// Free-form annotation (e.g., "compress triggered").
    pub note: String,
    /// Whether the base pruner (without review) was correct (Plan 036).
    pub base_correct: Option<bool>,
    /// Whether the reviewed decision was correct (Plan 036).
    pub reviewed_correct: Option<bool>,
    /// Per-anchor verification trace (StepCodeReasoner Plan 054).
    /// `None` for backward compatibility with existing logs.
    pub anchors: Option<Vec<AnchorTrace>>,
}

// ── Summary ─────────────────────────────────────────────────────

/// Aggregate statistics over a set of trial records.
#[derive(Clone, Debug)]
pub struct TrialSummary {
    /// Total number of episodes.
    pub total_episodes: usize,
    /// Arm with highest average reward.
    pub best_arm: usize,
    /// Average reward across all episodes.
    pub avg_reward: f32,
    /// Average regret across all episodes.
    pub avg_regret: f32,
}

// ── TrialLog ────────────────────────────────────────────────────

/// Append-only JSONL trial log for persistent episode history.
///
/// Each call to [`append`] writes one serialized [`TrialRecord`] as a JSON line.
/// Buffered for throughput — call [`flush`] to ensure durability.
pub struct TrialLog {
    writer: BufWriter<File>,
    path: std::path::PathBuf,
    count: usize,
    /// Optional review metrics for inference-time feedback tracking (Plan 036).
    review_metrics: Option<Arc<ReviewMetrics>>,
}

impl TrialLog {
    /// Create or append to a JSONL trial log at `path`.
    ///
    /// Creates the file if it doesn't exist. Appends if it does.
    pub fn new(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().append(true).create(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            path: path.to_path_buf(),
            count: 0,
            review_metrics: None,
        })
    }

    /// Enable review metrics tracking (builder pattern).
    ///
    /// When enabled, `append_with_review` updates both the log and metrics.
    /// The same `Arc<ReviewMetrics>` can be shared across components.
    pub fn with_review_metrics(mut self, metrics: Arc<ReviewMetrics>) -> Self {
        self.review_metrics = Some(metrics);
        self
    }

    /// Append a single trial record as one JSON line.
    ///
    /// Buffered — may not hit disk until [`flush`](Self::flush).
    pub fn append(&mut self, record: &TrialRecord) -> Result<()> {
        let line = serde_json::to_string(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()
    }

    /// Path of the JSONL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of records written so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Load all trial records from a JSONL file.
    ///
    /// Malformed lines are skipped with a warning to stderr.
    pub fn load(path: &Path) -> Result<Vec<TrialRecord>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<TrialRecord>(trimmed) {
                Ok(record) => records.push(record),
                Err(e) => eprintln!(
                    "trial_log: skipping malformed line {} in {}: {e}",
                    line_num + 1,
                    path.display()
                ),
            }
        }

        Ok(records)
    }

    /// Append a trial record with review classification.
    ///
    /// Writes the record to the JSONL log AND updates review metrics
    /// (if enabled via `with_review_metrics`).
    pub fn append_with_review(
        &mut self,
        record: &TrialRecord,
        base_correct: bool,
        reviewed_correct: bool,
    ) -> Result<()> {
        // Update review metrics if enabled
        if let Some(ref metrics) = self.review_metrics {
            metrics.record(base_correct, reviewed_correct);
        }
        self.append(record)
    }

    /// Read-only access to review metrics (if enabled).
    pub fn metrics(&self) -> Option<&ReviewMetrics> {
        self.review_metrics.as_ref().map(|arc| arc.as_ref())
    }

    /// Convenience: compute review metrics summary (if enabled).
    pub fn metrics_summary(&self) -> Option<ReviewSummary> {
        self.review_metrics.as_ref().map(|m| m.summary())
    }

    /// Compute aggregate summary over a slice of trial records.
    ///
    /// Returns default summary for empty input.
    pub fn summary(records: &[TrialRecord]) -> TrialSummary {
        if records.is_empty() {
            return TrialSummary {
                total_episodes: 0,
                best_arm: 0,
                avg_reward: 0.0,
                avg_regret: 0.0,
            };
        }

        let total_episodes = records.len();
        let avg_reward = records.iter().map(|r| r.reward).sum::<f32>() / total_episodes as f32;
        let avg_regret = records
            .iter()
            .map(|r| r.cumulative_regret)
            .next_back()
            .unwrap_or(0.0)
            / total_episodes as f32;

        // Find best arm by average reward
        let mut arm_counts: std::collections::HashMap<usize, (f32, usize)> =
            std::collections::HashMap::new();
        for r in records {
            let entry = arm_counts.entry(r.arm).or_insert((0.0, 0));
            entry.0 += r.reward;
            entry.1 += 1;
        }
        let best_arm = arm_counts
            .iter()
            .map(|(&arm, &(sum, count))| (arm, sum / count as f32))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arm, _)| arm)
            .unwrap_or(0);

        TrialSummary {
            total_episodes,
            best_arm,
            avg_reward,
            avg_regret,
        }
    }
}

// ── SharedTrialLog (Issue 051 T4: multi-writer support) ────────

/// Thread-safe wrapper around [`TrialLog`] for multi-agent shared bandit.
///
/// Multiple HLPlayers can write to the same JSONL file via cloned handles.
/// Gate: `#[cfg(feature = "bandit")]` — only needed for shared bandit mode.
///
/// # Usage
///
/// ```rust,ignore
/// let log = SharedTrialLog::new(TrialLog::new(path)?);
/// let h1 = log.clone_handle();
/// let h2 = log.clone_handle();
/// // h1 and h2 write concurrently to the same file
/// ```
#[cfg(feature = "bandit")]
pub struct SharedTrialLog {
    inner: Arc<Mutex<TrialLog>>,
}

#[cfg(feature = "bandit")]
impl SharedTrialLog {
    /// Wrap an existing [`TrialLog`] in a thread-safe handle.
    pub fn new(log: TrialLog) -> Self {
        Self {
            inner: Arc::new(Mutex::new(log)),
        }
    }

    /// Append a trial record (thread-safe).
    ///
    /// Blocks until the mutex is available.
    pub fn append(&self, record: &TrialRecord) -> Result<()> {
        self.inner.lock().unwrap().append(record)
    }

    /// Number of records written so far (thread-safe).
    pub fn count(&self) -> usize {
        self.inner.lock().unwrap().count()
    }

    /// Flush buffered writes to disk (thread-safe).
    pub fn flush(&self) -> Result<()> {
        self.inner.lock().unwrap().flush()
    }

    /// Clone a new handle referencing the same underlying log.
    ///
    /// Use this to distribute handles across threads/agents.
    pub fn clone_handle(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "microgpt_test_{name}_{pid}.jsonl",
            pid = std::process::id()
        ))
    }

    fn sample_record(episode: usize, arm: usize, reward: f32) -> TrialRecord {
        TrialRecord {
            episode,
            player_id: 0,
            arm,
            reward,
            q_value: 0.5,
            cumulative_reward: reward * episode as f32,
            cumulative_regret: (1.0 - reward) * episode as f32,
            config: "test".into(),
            note: format!("ep{episode}"),
            base_correct: None,
            reviewed_correct: None,
            anchors: None,
        }
    }

    #[test]
    fn test_roundtrip() {
        let path = temp_path("roundtrip");
        let _ = std::fs::remove_file(&path);

        // Write 3 records
        {
            let mut log = TrialLog::new(&path).unwrap();
            for i in 0..3 {
                log.append(&sample_record(i, i, 0.5)).unwrap();
            }
            log.flush().unwrap();
        }

        // Load back
        let records = TrialLog::load(&path).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0], sample_record(0, 0, 0.5));
        assert_eq!(records[2], sample_record(2, 2, 0.5));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_summary_aggregation() {
        let records = vec![
            sample_record(0, 0, 0.2),
            sample_record(1, 1, 0.9),
            sample_record(2, 1, 0.9),
            sample_record(3, 0, 0.1),
        ];

        let summary = TrialLog::summary(&records);
        assert_eq!(summary.total_episodes, 4);
        assert_eq!(summary.best_arm, 1); // arm 1 avg = 0.9 > arm 0 avg = 0.15
        assert!((summary.avg_reward - 0.525).abs() < 0.01);
    }

    #[test]
    fn test_empty_log() {
        let path = temp_path("empty");
        let _ = std::fs::remove_file(&path);

        // Write nothing
        {
            let mut log = TrialLog::new(&path).unwrap();
            log.flush().unwrap();
        }

        let records = TrialLog::load(&path).unwrap();
        assert!(records.is_empty());

        let summary = TrialLog::summary(&records);
        assert_eq!(summary.total_episodes, 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_count_tracks_appends() {
        let path = temp_path("count");
        let _ = std::fs::remove_file(&path);

        let mut log = TrialLog::new(&path).unwrap();
        assert_eq!(log.count(), 0);

        log.append(&sample_record(0, 0, 0.5)).unwrap();
        assert_eq!(log.count(), 1);

        log.append(&sample_record(1, 1, 0.8)).unwrap();
        assert_eq!(log.count(), 2);

        let _ = std::fs::remove_file(&path);
    }

    // ── AnchorTrace Tests (Plan 054, T7) ─────────────────────────

    #[test]
    fn test_anchor_trace_serialization() {
        let trace = AnchorTrace {
            depth: 2,
            arm: 5,
            reward: 1.0,
            shaped_reward: 1.15,
            future_accuracy: 0.5,
        };

        // Roundtrip through JSON
        let json = serde_json::to_string(&trace).unwrap();
        let deserialized: AnchorTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(trace, deserialized);
    }

    #[test]
    fn test_trial_record_with_anchors() {
        let anchors = vec![
            AnchorTrace {
                depth: 0,
                arm: 3,
                reward: 1.0,
                shaped_reward: 1.15,
                future_accuracy: 0.5,
            },
            AnchorTrace {
                depth: 1,
                arm: 7,
                reward: 0.0,
                shaped_reward: 0.0,
                future_accuracy: 0.0,
            },
        ];

        let record = TrialRecord {
            episode: 42,
            player_id: 0,
            arm: 7,
            reward: 0.0,
            q_value: 0.5,
            cumulative_reward: 10.0,
            cumulative_regret: 5.0,
            config: "stepcode".into(),
            note: "test".into(),
            base_correct: None,
            reviewed_correct: None,
            anchors: Some(anchors),
        };

        // Roundtrip through JSONL
        let path = temp_path("anchors");
        let _ = std::fs::remove_file(&path);

        {
            let mut log = TrialLog::new(&path).unwrap();
            log.append(&record).unwrap();
            log.flush().unwrap();
        }

        let loaded = TrialLog::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].episode, 42);
        assert!(loaded[0].anchors.is_some());
        let loaded_anchors = loaded[0].anchors.as_ref().unwrap();
        assert_eq!(loaded_anchors.len(), 2);
        assert_eq!(loaded_anchors[0].depth, 0);
        assert_eq!(loaded_anchors[0].shaped_reward, 1.15);
        assert_eq!(loaded_anchors[1].reward, 0.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_backward_compat_none_anchors() {
        // A JSON record without "anchors" field should load with anchors=None
        let json_line = r#"{"episode":0,"arm":1,"reward":0.5,"q_value":0.5,"cumulative_reward":0.0,"cumulative_regret":0.5,"config":"old","note":"","base_correct":null,"reviewed_correct":null}"#;

        let record: TrialRecord = serde_json::from_str(json_line).unwrap();
        assert_eq!(record.episode, 0);
        assert_eq!(record.arm, 1);
        assert!(record.anchors.is_none());
    }

    #[test]
    fn test_player_id_backward_compat() {
        // Legacy JSON without player_id should parse with default 0
        let json_line = r#"{"episode":5,"arm":2,"reward":0.8,"q_value":0.6,"cumulative_reward":4.0,"cumulative_regret":1.0,"config":"old","note":"","base_correct":null,"reviewed_correct":null}"#;
        let record: TrialRecord = serde_json::from_str(json_line).unwrap();
        assert_eq!(record.player_id, 0);

        // New JSON with player_id should parse correctly
        let json_with_id = r#"{"episode":5,"player_id":3,"arm":2,"reward":0.8,"q_value":0.6,"cumulative_reward":4.0,"cumulative_regret":1.0,"config":"new","note":"","base_correct":null,"reviewed_correct":null}"#;
        let record_id: TrialRecord = serde_json::from_str(json_with_id).unwrap();
        assert_eq!(record_id.player_id, 3);
    }

    // ── SharedTrialLog Tests (Issue 051 T4) ──────────────────────

    #[test]
    #[cfg(feature = "bandit")]
    fn test_shared_trial_log_multi_writer() {
        use std::sync::Arc;
        use std::thread;

        let path = temp_path("shared_multi_writer");
        let _ = std::fs::remove_file(&path);

        let log = Arc::new(super::SharedTrialLog::new(TrialLog::new(&path).unwrap()));

        let num_threads = 4usize;
        let records_per_thread = 50usize;

        let handles: Vec<_> = (0..num_threads)
            .map(|player_id| {
                let log_clone = Arc::clone(&log);
                thread::spawn(move || {
                    for i in 0..records_per_thread {
                        let record = TrialRecord {
                            episode: i,
                            player_id: player_id as u32,
                            arm: player_id,
                            reward: 0.5,
                            q_value: 0.5,
                            cumulative_reward: 0.0,
                            cumulative_regret: 0.0,
                            config: "multi".into(),
                            note: format!("player{player_id}_ep{i}"),
                            base_correct: None,
                            reviewed_correct: None,
                            anchors: None,
                        };
                        log_clone.append(&record).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        log.flush().unwrap();

        // Verify total count
        let total_expected = num_threads * records_per_thread;
        assert_eq!(log.count(), total_expected);

        // Load and verify records
        let records = TrialLog::load(&path).unwrap();
        assert_eq!(records.len(), total_expected);

        // Each player_id should appear exactly records_per_thread times
        for pid in 0..num_threads {
            let count = records.iter().filter(|r| r.player_id == pid as u32).count();
            assert_eq!(
                count, records_per_thread,
                "player_id {pid} expected {records_per_thread} records, got {count}"
            );
        }

        let _ = std::fs::remove_file(&path);
    }
}
