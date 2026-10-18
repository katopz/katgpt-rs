//! Prompt-backend trait — abstraction over prompt→string inference services.
//!
//! Hoisted from `riir-game-sdk::gm::prompt` (Issue 580) so multiple consumers
//! (`riir-agents` Phase 2, `riir-game-sdk::gm::prompt`, any future caller)
//! share one trait contract. The trait is generic — a minimal "prompt in,
//! optional string out" surface. Use-case-specific mocks, parsers, and
//! production impls live in the consuming crates.
//!
//! ## Why this lives in katgpt-core
//!
//! katgpt-core already hosts abstract trait contracts that consumers plug
//! impls into (`QGradientOracle`, `NoGuidanceOracle` in [`crate::traits`];
//! `GameState`, `RolloutPolicy`). [`InferenceBackend`] follows the same
//! pattern: it's a contract, not a primitive. The trait itself performs no
//! inference and mutates no weights — it merely abstracts over an external
//! prompt→string call. This fits the "modelless inference primitives"
//! mandate (the trait is modelless; impls may or may not be).
//!
//! ## Why `&mut self`
//!
//! Production backends often own mutable state (HTTP client pools, rate
//! limiters, retry counters, in-flight caches). `&mut self` accommodates
//! these without forcing interior mutability. Mock / stateless impls ignore
//! the mutability. This matches the shape the SDK already ships, so its
//! existing `MockInferenceBackend` impl carries over unchanged.
//!
//! Prompt-backend trait — generic prompt→string inference contract (Issue 580).
//! Hoisted from riir-game-sdk::gm::prompt so multiple consumers (riir-agents
//! Phase 2, the SDK's gm::prompt module, future callers) share one trait.
//! Always-on: pure trait + zero-dep mock, no feature gate needed.

/// Trait for prompt→string inference backends.
///
/// Production implementations call an external service (HTTP API, local model
/// server, riir-engine forward pass wrapped in a prompt loop, etc.). The
/// contract is deliberately minimal: prompt in, optional string out.
/// Consumers own the response parsing (JSON, structured fields, free text) —
/// this trait does not prescribe a response format.
///
/// `&mut self` allows backends with mutable state (see module docs).
/// `Send + Sync` allows backends to be shared across threads.
///
/// # Object safety
///
/// The trait is object-safe: `Box<dyn InferenceBackend>` compiles. Consumers
/// store the backend as `Box<dyn InferenceBackend>` for pluggability (see
/// `PromptBridge` in `riir-game-sdk::gm::prompt` for the established
/// pattern).
pub trait InferenceBackend: Send + Sync {
    /// Generate a response from the given prompt.
    ///
    /// Returns `None` when the backend cannot produce a response (unparseable
    /// prompt, service error, empty result). Consumers decide how to handle
    /// `None` — typical policies: disable preview, fall back to a default,
    /// surface an error.
    fn generate(&mut self, prompt: &str) -> Option<String>;
}

/// Mock backend that returns a caller-supplied canned response.
///
/// Useful in tests + dev loops where a deterministic response is needed
/// without an external service. The response is supplied at construction;
/// every `generate` call returns the same value. For use-case-specific mocks
/// (keyword matching, structured JSON emission, fixture-table lookup),
/// consumers define their own [`InferenceBackend`] impl.
///
/// # Examples
///
/// ```
/// use katgpt_core::prompt_backend::{CannedResponseBackend, InferenceBackend};
///
/// let mut backend = CannedResponseBackend::new("hello");
/// assert_eq!(backend.generate("anything").as_deref(), Some("hello"));
/// ```
#[derive(Debug, Clone)]
pub struct CannedResponseBackend {
    response: Option<String>,
}

impl CannedResponseBackend {
    /// Create a backend that always returns this response.
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: Some(response.into()),
        }
    }

    /// Create a backend that always returns `None` (no response).
    ///
    /// Useful for testing the consumer's `None`-handling path.
    pub fn none() -> Self {
        Self { response: None }
    }
}

impl InferenceBackend for CannedResponseBackend {
    fn generate(&mut self, _prompt: &str) -> Option<String> {
        self.response.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canned_backend_returns_supplied_response() {
        let mut b = CannedResponseBackend::new("hello");
        assert_eq!(b.generate("anything").as_deref(), Some("hello"));
        // Returns the same response on every call (not just the first).
        assert_eq!(b.generate("different prompt").as_deref(), Some("hello"));
    }

    #[test]
    fn canned_none_backend_returns_none() {
        let mut b = CannedResponseBackend::none();
        assert_eq!(b.generate("anything"), None);
    }

    #[test]
    fn trait_is_object_safe() {
        // Box<dyn InferenceBackend> must compile (object safety check).
        let b: Box<dyn InferenceBackend> = Box::new(CannedResponseBackend::new("x"));
        drop(b);
    }

    #[test]
    fn custom_impl_works_via_trait_object() {
        struct Echo;
        impl InferenceBackend for Echo {
            fn generate(&mut self, prompt: &str) -> Option<String> {
                Some(format!("echo:{prompt}"))
            }
        }
        let mut b: Box<dyn InferenceBackend> = Box::new(Echo);
        assert_eq!(b.generate("hi").as_deref(), Some("echo:hi"));
    }

    #[test]
    fn canned_backend_is_send_sync() {
        // The Send + Sync bound on the trait must be satisfied.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CannedResponseBackend>();
    }
}
