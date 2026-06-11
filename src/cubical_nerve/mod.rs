//! CubicalNerve — CAT(0) cubical complexes from game zone posets.
//!
//! The cubical nerve functor ⊞ (arXiv:2503.13663) sends a distributive
//! meet-semilattice L to a cubical set ⊞[L] whose geometric realization
//! is a CAT(0) cube complex. This guarantees unique geodesics (shortest
//! paths) for deterministic NPC navigation.
//!
//! Plan 252 Phase 3, Research 220.

mod cache;
mod cat0;
mod nerve;
mod poset;

pub use cache::{NerveCache, NerveFlowField, NavigationHint};
pub use cat0::{GeodesicPath, cat0_geodesic, is_cat0};
pub use nerve::{CubicalComplex, CubicalCube, cubical_nerve, cubical_nerve_with_threshold};
pub use poset::{DistributiveMeetSemilattice, ZoneId, ZonePoset};
