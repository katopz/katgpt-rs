//! Proof Sketch Evolution — Elo-rated population + global goal cache (Plan 128).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763).
//! Provides blake3-keyed global goal deduplication for DDTree verification,
//! reducing redundant constraint checks across draft branches by 3×.
//!
//! # Feature Gate
//!
//! Requires `proof_sketch_evolution` feature (depends on `bandit`).

pub mod dtree_goal_cache;
pub mod goal_cache;
pub mod parallelism;
pub mod plackett_luce;
pub mod sketch_population;
pub mod sketch_sampler;
pub mod sketch_types;

pub use dtree_goal_cache::{DTreeCacheSnapshot, DTreeGoalCache, encode_constraint_key};
pub use goal_cache::{GoalHash, GoalResult, ProofGoalCache, ProofGoalSnapshot};
pub use parallelism::{
    ParallelismGuard, SketchSelectionStrategy, select_strategy, should_use_population,
};
pub use plackett_luce::{GibbsScratch, PlackettLuceConfig, PlackettLuceRater};
pub use sketch_population::{EvictionReport, PopulationConfig, SketchPopulation};
pub use sketch_sampler::{
    DEFAULT_EPSILON, DEFAULT_EXPLORATION_C, SketchSampler, SketchSamplerConfig,
};
pub use sketch_types::{
    DEFAULT_ELO, DiversityHint, DiversityStrategy, ELO_SCALE, Goal, MAX_LESSONS, MAX_PENDING_GOALS,
    ProofState, SketchEntry, SketchId,
};
