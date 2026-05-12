//! Fire-and-forget feedback module for TTT-inspired E2E feedback loop.
//!
//! Sends InferenceResult to a configurable cache endpoint.
//! Failures are silently ignored — never block inference on cache writes.

use crate::types::InferenceResult;

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
    let Some(url) = &config.url else {
        return; // Feedback disabled
    };

    if result.reward < config.min_reward {
        return; // Skip low-quality results
    }

    // Serialize to JSON
    let Ok(json) = serde_json::to_string(result) else {
        return;
    };

    // Fire-and-forget: spawn a thread that makes the HTTP POST.
    // We don't use reqwest/tokio to avoid adding heavy deps.
    // In production, this would use the REST feature's client.
    let url = url.clone();
    std::thread::spawn(move || {
        // Placeholder: in production, use reqwest::blocking or the rest feature.
        // For now, log that we would send.
        log::debug!("Feedback POST to {url}: {:.100}...", json);
        let _ = (url, json); // Suppress unused warnings
    });
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
        };
        send_feedback(&config, &result);
        // Thread spawned but reward too low — no actual POST happens
    }
}
