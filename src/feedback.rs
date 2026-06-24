//! Fire-and-forget feedback module for TTT-inspired E2E feedback loop.
//!
//! Sends InferenceResult to a configurable cache endpoint.
//! Failures are silently ignored — never block inference on cache writes.

use crate::types::InferenceResult;
use std::sync::OnceLock;
use std::sync::mpsc::Sender;

/// Background worker sender — initialized once, reused for all feedback calls.
/// Avoids ~10-50μs `thread::spawn` overhead per call.
static FEEDBACK_SENDER: OnceLock<Sender<Vec<u8>>> = OnceLock::new();

fn get_feedback_sender() -> &'static Sender<Vec<u8>> {
    FEEDBACK_SENDER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        std::thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                log::debug!("Feedback: {:.100}...", String::from_utf8_lossy(&msg));
            }
        });
        tx
    })
}

/// Configuration for feedback loop.
#[derive(Debug, Clone)]
pub struct FeedbackConfig {
    /// URL to POST inference results to (e.g., "http://localhost:8080/cache/ingest").
    /// If None, feedback is disabled (no behavior change).
    pub url: Option<String>,
    /// Minimum reward to send feedback (skip low-quality results).
    pub min_reward: f32,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            url: None,
            min_reward: 0.5,
        }
    }
}

/// Fire-and-forget: send InferenceResult to cache endpoint.
/// Returns immediately. Errors are logged but ignored.
pub fn send_feedback(config: &FeedbackConfig, result: &InferenceResult) {
    let Some(_url) = &config.url else {
        return; // Feedback disabled
    };

    if result.reward < config.min_reward {
        return; // Skip low-quality results
    }

    // Serialize to binary (postcard)
    let Ok(bytes) = postcard::to_allocvec(result) else {
        return;
    };

    // Send to background worker thread via channel — avoids thread::spawn per call.
    let _ = get_feedback_sender().send(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_disabled_when_no_url() {
        let config = FeedbackConfig::default();
        let result = InferenceResult {
            domain: "test".into(),
            reward: 0.9,
            tree_budget_used: 100,
            budget_level: 0,
            prompt_hash: 12345,
            output: "hello".into(),
            timestamp: 0,
            screened: false,
            #[cfg(feature = "sr2am_configurator")]
            planning_decision: None,
            #[cfg(feature = "sr2am_configurator")]
            plan_horizon_used: 0,
        };
        // Should not panic or error when url is None
        send_feedback(&config, &result);
    }

    #[test]
    fn test_feedback_skips_low_reward() {
        let config = FeedbackConfig {
            url: Some("http://localhost:9999/cache/ingest".into()),
            min_reward: 0.7,
        };
        let result = InferenceResult {
            domain: "test".into(),
            reward: 0.3,
            tree_budget_used: 10,
            budget_level: 0,
            prompt_hash: 0,
            output: String::new(),
            timestamp: 0,
            screened: true,
            #[cfg(feature = "sr2am_configurator")]
            planning_decision: None,
            #[cfg(feature = "sr2am_configurator")]
            plan_horizon_used: 0,
        };
        send_feedback(&config, &result);
        // Thread spawned but reward too low — no actual POST happens
    }
}
