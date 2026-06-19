//! ICT Distributional Branching-Point Detector — open, generic, MIT-licensed
//! primitives distilled from ICT (Feng et al., arxiv 2606.19771).
//!
//! Plan 294, Research 270. This module is the public adoption hook for the
//! ICT framework's modelless runtime primitives:
//!
//! - [`math`]: collision purity β(π), Rényi H₂, Shannon H₁, JS divergence.
//! - [`branching`]: critical-branching predicate + top-k% mask selector.
//! - [`detector`]: [`BranchingDetector`] — runtime structure consuming K
//!   candidate trajectories and emitting a branching mask + β + JS-uniqueness.
//! - [`types`]: [`BranchingReport`] output struct.
//! - [`bebop_upgrade`]: [`AcceptanceForecastH2`] — Bebop's H₁ → H₂ upgrade.
//!
//! ## What this is NOT
//!
//! - Not a game-specific module — no NPC semantics, no chain IP. The
//!   runtime fusion (CLR gating, HLA updates, KG emission, curiosity bursts)
//!   lives in `riir-ai` Plan 324.
//! - Not a training-time gradient mask (the paper's ICT selector is). This
//!   is the **inference-time** modelless drop-in.
//!
//! ## References
//!
//! - **Research 270** — design rationale + signatures:
//!   `katgpt-rs/.research/270_Beyond_Entropy_ICT_Distributional_Branching_Detector.md`
//! - **Plan 294** — implementation plan:
//!   `katgpt-rs/.plans/294_ict_branching_detector.md`
//! - **Private NPC guide (the moat):**
//!   `riir-ai/.research/142_Distributional_Branching_Point_NPC_Guide.md`
//! - **arxiv 2606.19771** — Beyond Entropy / ICT framework (Feng et al., 18 Jun 2026).
//!
//! ## Feature gate
//!
//! Entire module is behind `feature = "ict_branching"`, **default OFF**.
//! Promotion to default-on requires **both**:
//! - G3 (`bench_294_ict_g3.rs`): Spearman ρ(H₁, JS-uniqueness) < 0.5.
//! - G8 (riir-ai Plan 324): runtime fusion validated on real NPCs.
//!
//! G3 alone is necessary but not sufficient per Plan 294 §Phase 8 T8.4.

pub mod bebop_upgrade;
pub mod branching;
pub mod detector;
pub mod math;
pub mod types;

// Convenience re-exports at module root.
pub use bebop_upgrade::AcceptanceForecastH2;
pub use branching::{branching_point_mask, branching_point_mask_into, is_critical_branching};
pub use detector::BranchingDetector;
pub use math::{collision_purity, collision_purity_into, js_divergence, js_divergence_batch, renyi_h2, shannon_h1};
pub use types::BranchingReport;
