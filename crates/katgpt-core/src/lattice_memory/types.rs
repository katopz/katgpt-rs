//! `LatticeMemory` vocabulary types — decoupled structs, no logic deps.
//!
//! Plan 619 T1.1: a lattice of delta-rule cells over a preallocated flat
//! slab with RUNTIME `d_k`/`d_v` (the const-generic `DeltaMemoryState` shape
//! needs `generic_const_exprs`, unstable on the pinned toolchain; the flat
//! slab is the recorded alternative). Cell state = offset+length into the
//! slab — never a per-cell heap allocation.
//!
//! T1.5 adds the occupancy `warp` (a per-axis monotone Gaussian-CDF
//! remap — `super::cdf`) and T1.8 the `bump_kernel` tunable to the config:
//! both are addressing geometry, which is what this struct owns.
//!
//! Issue 938 T2.1: the geometry is generic over the axis count `A`
//! ([`LatticeConfigG`]); [`LatticeConfig`] is the shipped `A = 2` alias so
//! every Plan 619 call site compiles unchanged, and [`LatticeConfig3D`] is
//! the 3-axis remedy behind `lattice_memory_3d`.
//!
//! Name disambiguation (Plan 619): this is NOT `analytic_lattice` (Plan 330
//! transport operators — unrelated domain).

use super::cdf::CdfWarp;

/// One lattice cell's slab claim.
///
/// `offset`/`len` address the shared slab in `f32` elements; `len` is
/// `d_k * d_v` for every cell today (kept as a field for variable-shape
/// futures, exactly as the plan names it). `writes` counts delta-rule
/// updates — the forgetting axis (overwrite count drives recall decay).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LatticeCell {
    /// First slab element this cell owns.
    pub offset: u32,
    /// Slab elements this cell owns (`d_k * d_v` today).
    pub len: u32,
    /// Delta-rule updates applied to this cell (0 = never written).
    pub writes: u32,
}

impl LatticeCell {
    /// A cell that has never been written: no slab claim yet.
    #[must_use]
    pub const fn unwritten() -> Self {
        Self {
            offset: 0,
            len: 0,
            writes: 0,
        }
    }

    /// A cell is live once it has claimed slab space.
    #[must_use]
    pub const fn is_written(&self) -> bool {
        self.writes > 0
    }
}

/// Lattice geometry + addressing configuration, generic over the axis
/// count `A` (Issue 938 T2.1).
///
/// Built once, consumed by reference; the seeded E2LSH axes derive from it
/// (`seed` + the scalars → the shared `config_seed` discipline), so the same
/// configuration always produces the same lattice.
#[derive(Clone, Debug)]
pub struct LatticeConfigG<const A: usize> {
    /// Embedding dimension (the address space input width).
    pub d: usize,
    /// Cell key width (delta-rule state is `d_k * d_v`).
    pub d_k: usize,
    /// Cell value width (delta-rule state is `d_k * d_v`).
    pub d_v: usize,
    /// Cells per axis, row-major over axes (`grid[0]` rows × `grid[1]`
    /// columns × … at `A = 2`/`A = 3`).
    pub grid: [u32; A],
    /// Per-axis continuous-LSH width `w` (the quantization cell size in
    /// projection units).
    pub w: [f32; A],
    /// Deterministic seed — same config → same projections (Issue 809 law).
    pub seed: u64,
    /// Hard byte ceiling for the slab; `None` = unbounded (tests/diagnostics).
    /// The constructor refuses configurations over the budget (the T1.5 law,
    /// armed at birth because the refusal is cheap and the failure silent).
    pub byte_budget: Option<usize>,
    /// Per-axis monotone Gaussian-CDF occupancy remap (`None` = raw E2LSH
    /// coordinates). Fitted by [`super::sizer`] to the corpus's projected
    /// mean/spread: post-warp marginals are uniform, so the Poisson occupancy
    /// law the sizer solves against holds. Monotone ⇒ 1-D locality preserved.
    pub warp: Option<[CdfWarp; A]>,
    /// Bump kernel shape — the T1.8 tunable (tent is the shipped default;
    /// `Cos2` exists for the recorded A/B).
    pub bump_kernel: BumpKernel,
}

/// The shipped 2-axis lattice configuration (Plan 619) — every Plan 619
/// call site compiles unchanged through this alias.
pub type LatticeConfig = LatticeConfigG<2>;

/// The 3-axis configuration — Issue 938 T2's second capacity remedy
/// (`N_max ≈ 0.35·d_k·(3.76/R)^AXES` gains the exponent). Opt-in behind
/// `lattice_memory_3d`; the GOAT gates live in the bench, never here.
#[cfg(feature = "lattice_memory_3d")]
pub type LatticeConfig3D = LatticeConfigG<3>;

impl<const A: usize> LatticeConfigG<A> {
    /// Slab elements a written cell owns.
    #[must_use]
    pub const fn cell_len(&self) -> usize {
        self.d_k * self.d_v
    }

    /// Total grid cells (`L^A` for a uniform side).
    #[must_use]
    pub const fn cell_count(&self) -> usize {
        let mut n = 1_usize;
        let mut axis = 0;
        while axis < A {
            n *= self.grid[axis] as usize;
            axis += 1;
        }
        n
    }

    /// Slab bytes at full occupancy (every cell written).
    #[must_use]
    pub const fn slab_bytes(&self) -> usize {
        self.cell_count() * self.cell_len() * core::mem::size_of::<f32>()
    }
}

/// The per-axis bump kernel family (Plan 619 T1.8).
///
/// Both kernels live on the same 3-tap structure `{round(c)−1, round(c),
/// round(c)+1}` and normalize per axis; they differ only in shape. `Tent` is
/// the shipped default (hard ±1 support, direction-sensitive);
/// `Cos2` — cos²(π(δ−o)/4), truncated to the 3 taps — was the recorded
/// A/B alternative (it exists in the literature for SGD differentiability,
/// which this modelless primitive does not have); the T1.8 bench pins the
/// winner and this enum carries the losing arm for re-checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BumpKernel {
    /// Radius-1.5 tent partition of unity (shipped default).
    #[default]
    Tent,
    /// Truncated cos² window (the T1.8 A/B arm).
    Cos2,
}

/// A validation failure with the offending value named.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LatticeConfigError {
    /// A dimension or grid side was zero.
    ZeroDimension,
    /// A per-axis LSH width was zero or non-finite.
    BadWidth(usize),
    /// `slab_bytes()` exceeded `byte_budget`.
    OverBudget {
        /// Required slab bytes at full occupancy.
        required: usize,
        /// The configured ceiling.
        budget: usize,
    },
    /// [`LatticeMemory::sized`](super::LatticeMemory::sized) got a
    /// near-miss radius outside (0, 10] σ.
    BadNearMissRadius(f32),
    /// [`LatticeMemory::sized`](super::LatticeMemory::sized) refused an
    /// operating point whose per-cell crowding exceeds the delta-rule
    /// capacity (`0.35·d_k`): the remedy is named, not gestured at.
    OverCapacity {
        /// The requested item count.
        items: usize,
        /// The support-driven grid's cell count (`λ = items/cells`).
        cells: usize,
        /// The first-order capacity at the configured `d_k`.
        capacity_per_cell: f32,
        /// The `d_k` this operating point needs (`≥ λ/0.35`).
        required_d_k: usize,
    },
    /// [`LatticeMemory::sized`](super::LatticeMemory::sized) computed a
    /// support-driven grid whose cell count does not fit `usize` at this
    /// arity — the radius is absurdly small for the axis count (an
    /// adversarial input; every honest operating point sizes to a real
    /// grid).
    GridTooLarge {
        /// Cells per axis the law chose.
        side_per_axis: u32,
        /// The lattice axis count.
        axes: usize,
    },
    /// [`LatticeMemory::sized`](super::LatticeMemory::sized) got an empty
    /// sample set — the CDF warp has nothing to fit. Callers without corpus
    /// samples want the plain `new` constructor (raw coordinates, no warp).
    EmptySample,
}

impl core::fmt::Display for LatticeConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroDimension => write!(
                f,
                "lattice config: d, d_k, d_v and every grid side must be non-zero"
            ),
            Self::BadWidth(axis) => write!(
                f,
                "lattice config: axis {axis} width must be finite and > 0"
            ),
            Self::OverBudget { required, budget } => {
                write!(
                    f,
                    "lattice config: slab needs {required} B, byte_budget is {budget} B"
                )
            }
            Self::BadNearMissRadius(r) => {
                write!(
                    f,
                    "lattice sizer: near-miss radius must be in (0, 10] σ, got {r}"
                )
            }
            Self::OverCapacity {
                items,
                cells,
                capacity_per_cell,
                required_d_k,
            } => write!(
                f,
                "lattice sizer: {items} items over a {cells}-cell grid need λ = {:.1} per cell \
                 but d_k carries {:.1} — raise d_k to ≥ {required_d_k}, shrink items, or widen the radius",
                *items as f32 / *cells as f32,
                capacity_per_cell
            ),
            Self::GridTooLarge {
                side_per_axis,
                axes,
            } => write!(
                f,
                "lattice sizer: side {side_per_axis} per axis at {axes} axes overflows the \
                 cell count — widen the near-miss radius"
            ),
            Self::EmptySample => {
                write!(
                    f,
                    "lattice sizer: no samples to fit the occupancy warp — use `new` instead"
                )
            }
        }
    }
}

impl std::error::Error for LatticeConfigError {}

/// Serializable state snapshot of a [`LatticeMemory`](super::LatticeMemory)
/// — the freeze half of the Plan 199 T1.B consumer surface (`snapshot()` +
/// `commitment()` produce a BLAKE3-committed artifact; `restore()` is the
/// thaw).
///
/// Geometry (grid, `d_k`, `d_v`) is deliberately NOT carried: a snapshot
/// thaws into a lattice built from the same [`LatticeConfig`], and
/// [`super::LatticeMemory::restore`] refuses a mismatched shape by length
/// instead of guessing — the refusal names both sides' numbers.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LatticeSnapshot {
    /// Dense cell table (`grid[0] × … × grid[A−1]` entries; the slab claims).
    pub cells: Vec<LatticeCell>,
    /// Flat slab, `cell_count × d_k × d_v` f32 elements.
    pub slab: Vec<f32>,
    /// Next unclaimed slab element at snapshot time (allocations continue
    /// from here after a thaw — the bump cursor is part of the state).
    pub cursor: usize,
}

/// Why [`super::LatticeMemory::restore`] refused a snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LatticeSnapshotError {
    /// The snapshot's cell-table or slab length does not match the target
    /// lattice — a snapshot from a differently-shaped lattice.
    GeometryMismatch {
        expected_cells: usize,
        found_cells: usize,
        expected_slab: usize,
        found_slab: usize,
    },
    /// The snapshot's cursor points past its own slab (corrupt artifact).
    CursorOutOfBounds { cursor: usize, slab_len: usize },
    /// A written cell claims slab elements past the slab end (corrupt
    /// artifact) — refused here so the hot path's slab indexing stays
    /// unchecked-by-construction instead of panicking on a thawed lie.
    CellOutOfBounds {
        cell_index: usize,
        offset: u32,
        len: u32,
        slab_len: usize,
    },
}

impl std::fmt::Display for LatticeSnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GeometryMismatch {
                expected_cells,
                found_cells,
                expected_slab,
                found_slab,
            } => {
                write!(
                    f,
                    "lattice snapshot geometry mismatch: cells {found_cells} vs expected \
                     {expected_cells}, slab {found_slab} vs expected {expected_slab} — \
                     rebuild the lattice from the snapshot's own LatticeConfig"
                )
            }
            Self::CursorOutOfBounds { cursor, slab_len } => {
                write!(
                    f,
                    "lattice snapshot cursor {cursor} past its own slab length {slab_len} — corrupt snapshot"
                )
            }
            Self::CellOutOfBounds {
                cell_index,
                offset,
                len,
                slab_len,
            } => {
                write!(
                    f,
                    "lattice snapshot cell {cell_index} claims [{offset}, {}) past slab length {slab_len} — corrupt snapshot",
                    offset + len
                )
            }
        }
    }
}

impl std::error::Error for LatticeSnapshotError {}

/// Validation shared by every constructor.
pub fn validate<const A: usize>(config: &LatticeConfigG<A>) -> Result<(), LatticeConfigError> {
    if config.d == 0 || config.d_k == 0 || config.d_v == 0 {
        return Err(LatticeConfigError::ZeroDimension);
    }
    for &side in config.grid.iter() {
        if side == 0 {
            return Err(LatticeConfigError::ZeroDimension);
        }
    }
    for (axis, &w) in config.w.iter().enumerate() {
        if w <= 0.0 || !w.is_finite() {
            return Err(LatticeConfigError::BadWidth(axis));
        }
    }
    if let Some(budget) = config.byte_budget {
        let required = config.slab_bytes();
        if required > budget {
            return Err(LatticeConfigError::OverBudget { required, budget });
        }
    }
    Ok(())
}
