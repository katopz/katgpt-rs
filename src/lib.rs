pub mod benchmark;
#[cfg(feature = "feedback")]
pub mod feedback;
#[cfg(feature = "hla_attention")]
pub mod hla;
pub mod percepta;
pub mod plot;
pub mod pruners;
pub mod speculative;
pub mod tokenizer;
pub mod transformer;
pub mod turboquant;
pub mod types;

#[cfg(debug_assertions)]
pub mod alloc;

/// Debug-only global allocator that tracks allocation count and bytes.
#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;

#[cfg(feature = "validator")]
pub mod validator;
