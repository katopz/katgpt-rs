//! LatticeMemory — E2LSH-addressed delta-rule cell lattice (Plan 619).
//!
//! The deterministic, modelless extraction of the Spotlight-class addressing
//! math (research note 233): a lattice of delta-rule cells addressed by
//! **continuous E2LSH per-axis coordinates** (`address`), bump-interpolated
//! over the 3^A neighborhood (`bump`), cells allocated on first write into a
//! preallocated flat slab (`types`) — the candidate fifth retrieval
//! complexity class (growing state + constant-neighborhood access + graded
//! near-miss recall) beside Raven O(1) / Engram O(1) / δ-Mem O(r) / PKM
//! O(√N).
//!
//! T1.4 arms the cells with the closed-form delta rule
//! `S ← (I − βkkᵀ)S + βkvᵀ` (rank-1 removal along the key + rank-1 add, two
//! `d_k·d_v` MAC passes, no gradient descent) and the per-cell
//! read-then-blend read `Σ φ_c·(S_cᵀ q)`. T1.5 adds the occupancy-warped
//! sizer (`sizer` + `cdf`); T1.6 the sigmoid-only scoring read. T1.9 arms
//! freeze/thaw — `snapshot`/`restore`/`commitment`, the Plan 199 T1.B
//! consumer surface (BLAKE3-committed artifact; whole-swap thaw).
//!
//! Issue 938: the lattice is **generic over the axis count** —
//! [`LatticeMemory`] is the shipped `A = 2` monomorphization (Plan 619,
//! bit-identical by construction: same draw order, same accumulation order,
//! same commitment byte stream), and [`LatticeMemory3D`] behind
//! `lattice_memory_3d` is the second capacity remedy for Bench 919's
//! NOT-PROMOTED floor (`N_max ≈ 0.35·d_k·(3.76/R)^AXES`). The sizer gains
//! the measured-erosion capacity (`capacity_per_cell_eroded`,
//! `sized_eroded`) and the overwrite lane gains the unblended
//! [`LatticeMemoryG::read_direct`].
//!
//! Opt-in (`feature = "lattice_memory"`) until the Plan 619 GOAT gate
//! (T1.7) passes; the gate designs live in the plan, never here. Name
//! disambiguation: NOT `analytic_lattice` (Plan 330 transport operators).
//!
//! LatticeMemory — E2LSH-addressed delta-rule cell lattice (Plan 619, research
//! note 233): continuous per-axis LSH addressing + 3-tap bump interpolation
//! + flat-slab delta cells. The candidate fifth retrieval complexity class.
//! NOT analytic_lattice (Plan 330 transport operators). Opt-in until the
//! Plan 619 T1.7 GOAT gate; consumer = riir-refine Plan 199.

mod address;
mod bump;
mod cdf;
mod sizer;
pub mod types;

use types::LatticeConfigG;

pub use address::{LshAxes, split_coordinate};
pub use bump::{AxisWeights, BumpLut, axis_weights, axis_weights_cos2, build_lut, weights_direct};
pub use cdf::CdfWarp;
pub use sizer::{capacity_per_cell, capacity_per_cell_eroded, grid_side_for, support_radius};
pub use types::{
    BumpKernel, LatticeCell, LatticeConfig, LatticeConfigError, LatticeSnapshot,
    LatticeSnapshotError, validate,
};

/// The shipped 2-axis lattice (Plan 619) — the default monomorphization.
/// Every Plan 619 call site compiles unchanged through this alias.
pub type LatticeMemory = LatticeMemoryG<2>;

/// The 3-axis lattice — Issue 938 T2's second capacity remedy. Opt-in
/// behind `lattice_memory_3d`; promotion is the bench's verdict (G1/G2/G5/G6),
/// never this alias.
#[cfg(feature = "lattice_memory_3d")]
pub type LatticeMemory3D = LatticeMemoryG<3>;

/// The 3-axis config alias (see [`types::LatticeConfig3D`]).
#[cfg(feature = "lattice_memory_3d")]
pub use types::LatticeConfig3D;

/// The 3-tap neighborhood weight row for one axis: `[prev, center, next]`.
const TAPS: usize = 3;

/// A query's lattice location: the clamped primary cell + the signed
/// fractional offsets the bump consumes.
struct Located<const A: usize> {
    /// The primary cell coordinate per axis (clamped into the grid).
    cell: [usize; A],
    delta: [f32; A],
}

/// The within-cell read query (T1.4).
///
/// `Row0` is `q = e₀` — only state row 0 contributes, which is exactly the
/// `write_value` overwrite lane; `Dense` is a caller-supplied key of width
/// `d_k`. An enum (not `&[f32]` + a flag) because the row-0 path must not
/// materialize an `e₀` buffer — the hot path allocates nothing.
#[derive(Clone, Copy)]
enum Query<'a> {
    Row0,
    Dense(&'a [f32]),
}

/// The lattice: seeded axes + bump table + dense cell table + flat slab,
/// generic over the axis count `A` (Issue 938 T2.1).
///
/// Constructed once via [`LatticeMemoryG::new`] (or the occupancy-fitted
/// [`LatticeMemoryG::sized`]); every hot-path read is a coordinate tuple →
/// 3^A dense-table lookups → slab rows blended in place — zero heap
/// allocation after construction (the G4 law, gated by
/// `tests/lattice_memory_gates.rs`).
pub struct LatticeMemoryG<const A: usize> {
    config: LatticeConfigG<A>,
    axes: LshAxes<A>,
    lut: BumpLut,
    /// Dense cell table, row-major over axes — unwritten cells carry
    /// `LatticeCell::unwritten()` (writes == 0).
    cells: Vec<LatticeCell>,
    /// Preallocated flat slab, `cell_count × d_k × d_v` elements. Cells claim
    /// regions from a bump cursor on first write; there is no cell
    /// deallocation, so no free list exists. State layout per cell: `d_k`
    /// rows of `d_v` (row-major) — row `i` is the value-channel weight for
    /// key component `k_i`.
    slab: Vec<f32>,
    /// Next unclaimed slab element.
    cursor: usize,
    /// `d_v` scratch for the delta rule's `r = kᵀS` read (T1.4) — allocated
    /// once here so `write_delta` never allocates.
    scratch: Vec<f32>,
}

impl<const A: usize> LatticeMemoryG<A> {
    /// Neighborhood taps across the whole product: `3^A`.
    const TAPS_POW: usize = TAPS.pow(A as u32);

    /// Construct from a validated config — one slab allocation, once.
    ///
    /// # Errors
    /// [`LatticeConfigError`] on zero dimensions, non-finite widths, or a
    /// slab over the configured `byte_budget`.
    pub fn new(config: LatticeConfigG<A>) -> Result<Self, LatticeConfigError> {
        validate(&config)?;
        let mut parts = Vec::with_capacity(A + 2);
        for &side in config.grid.iter() {
            parts.push(side as u64);
        }
        parts.push(config.d_k as u64);
        parts.push(config.d_v as u64);
        let axes = LshAxes::new(config.d, config.w, config.seed, &parts);
        let cells = vec![LatticeCell::unwritten(); config.cell_count()];
        let slab = vec![0.0_f32; config.cell_count() * config.cell_len()];
        let scratch = vec![0.0_f32; config.d_v];
        Ok(Self {
            config,
            axes,
            lut: build_lut(),
            cells,
            slab,
            cursor: 0,
            scratch,
        })
    }

    /// The validated config this lattice was built from.
    #[must_use]
    pub const fn config(&self) -> &LatticeConfigG<A> {
        &self.config
    }

    /// Cells claimed so far (occupancy readout — the T1.5 sizing axis).
    #[must_use]
    pub fn written_cells(&self) -> usize {
        self.cursor.checked_div(self.config.cell_len()).unwrap_or(0)
    }

    /// Slab elements claimed so far — bytes/4 is the evidence store's
    /// consumed size (consumer growth telemetry; also the exact value
    /// [`Self::snapshot`] freezes into [`LatticeSnapshot::cursor`]).
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// The clamped primary cell a query addresses — the occupancy/locality
    /// readout; the hot paths recompute it inline.
    #[must_use]
    pub fn primary_cell(&self, x: &[f32]) -> [usize; A] {
        self.locate(x).cell
    }

    /// Address `x` → primary cell + every axis' fractional offsets.
    ///
    /// The optional CDF warp (`T1.5`) remaps the raw coordinates first —
    /// monotone, so locality and the real fractional part survive. Cell
    /// coordinates clamp into the grid (queries may project outside it; the
    /// lattice's answers stay in-bounds — the read renormalizes over the
    /// clipped neighborhood).
    fn locate(&self, x: &[f32]) -> Located<A> {
        let mut c = self.axes.coordinates(x);
        if let Some(warp) = self.config.warp.as_ref() {
            for axis in 0..A {
                c[axis] = warp[axis].apply(c[axis]);
            }
        }
        let mut cell = [0_i64; A];
        let mut delta = [0.0_f32; A];
        for axis in 0..A {
            let (i, f) = split_coordinate(c[axis]);
            let m = i + i64::from(f > 0.5); // round(c): the primary
            cell[axis] = m;
            delta[axis] = c[axis] - m as f32;
        }
        let mut clamped = [0_usize; A];
        for axis in 0..A {
            clamped[axis] = cell[axis].clamp(0, i64::from(self.config.grid[axis]) - 1) as usize;
        }
        Located {
            cell: clamped,
            delta,
        }
    }

    /// Row-major flat index of a cell coordinate tuple (strides derived from
    /// the grid, never stored). For `A = 2` this is exactly the historical
    /// `row * grid[1] + col`.
    #[inline]
    fn cell_index(&self, cell: [usize; A]) -> usize {
        let mut idx = 0_usize;
        for (&c, &side) in cell.iter().zip(self.config.grid.iter()) {
            idx = idx * side as usize + c;
        }
        idx
    }

    /// **Delta-rule cell write** (T1.4): the cell `x` addresses gets
    ///
    /// ```text
    /// S ← (I − βkkᵀ)S + βkvᵀ      (the plan's S(I−βkkᵀ) + βkvᵀ in its
    ///                              row-vector spelling)
    /// ```
    ///
    /// evaluated closed-form in two `d_k·d_v` MAC passes — `r = kᵀS` (the
    /// state's read along the key), then the rank-1
    /// `S += β·k⊗(v − r)` — no gradient descent, no per-write heap (the
    /// scratch is a preallocated field). With β‖k‖² = 1 (unit keys, β = 1)
    /// the update removes the state along `k` EXACTLY: overwriting the same
    /// key replaces its association; near-collinear keys leave the
    /// measured Π-contracting interference the module tests document.
    ///
    /// `β` must be in `(0, 1]` — the contractive range (β > 1 amplifies old
    /// associations instead of removing them).
    ///
    /// # Panics
    /// On a `k`/`v` width mismatch, or `β` outside `(0, 1]`.
    pub fn write_delta(&mut self, x: &[f32], k: &[f32], v: &[f32], beta: f32) {
        assert_eq!(k.len(), self.config.d_k, "key width must be d_k");
        assert_eq!(v.len(), self.config.d_v, "value width must be d_v");
        assert!(
            beta > 0.0 && beta <= 1.0,
            "beta must be in (0, 1], got {beta}"
        );
        let located = self.locate(x);
        let idx = self.cell_index(located.cell);
        let len = self.config.cell_len();
        if !self.cells[idx].is_written() {
            let offset = self.cursor;
            self.cursor += len;
            self.cells[idx] = LatticeCell {
                offset: offset as u32,
                len: len as u32,
                writes: 0,
            };
        }
        let cell = self.cells[idx];
        let d_v = self.config.d_v;
        let base = cell.offset as usize;
        let (scratch, slab) = (&mut self.scratch, &mut self.slab);
        // Pass 1: r = kᵀS — the state's read along the key, before mutation.
        let r = &mut scratch[..d_v];
        r.fill(0.0);
        for (i, &ki) in k.iter().enumerate() {
            if ki == 0.0 {
                continue; // sparse-key fast skip (row 0 only ⇒ the e₀ lane)
            }
            let row = &slab[base + i * d_v..base + (i + 1) * d_v];
            for (rj, &s) in r.iter_mut().zip(row.iter()) {
                *rj += ki * s;
            }
        }
        // Pass 2: S += β·k⊗(v − r) — the rank-1 remove-along-k + add.
        for (i, &ki) in k.iter().enumerate() {
            let g = beta * ki;
            if g == 0.0 {
                continue;
            }
            let row = &mut slab[base + i * d_v..base + (i + 1) * d_v];
            for (sj, (&vj, &rj)) in row.iter_mut().zip(v.iter().zip(r.iter())) {
                *sj += g * (vj - rj);
            }
        }
        self.cells[idx].writes += 1;
    }

    /// Write `v` as THE value of the cell `x` addresses — the overwrite lane.
    ///
    /// This is `write_delta(x, e₀, v, 1.0)` evaluated exactly: `r` is state
    /// row 0, then row 0 `+= (v − r)` ⇒ row 0 = `v` — kept specialized (no
    /// `e₀` buffer, one memcpy) so the simple association path stays the
    /// cheapest. First write claims the cell's slab region; re-writes
    /// overwrite the claimed region in place (the slab never grows).
    pub fn write_value(&mut self, x: &[f32], v: &[f32]) {
        assert_eq!(v.len(), self.config.d_v, "value width must be d_v");
        let located = self.locate(x);
        let idx = self.cell_index(located.cell);
        let len = self.config.cell_len();
        if !self.cells[idx].is_written() {
            let offset = self.cursor;
            self.cursor += len;
            self.cells[idx] = LatticeCell {
                offset: offset as u32,
                len: len as u32,
                writes: 0,
            };
        }
        let cell = self.cells[idx];
        let d_v = self.config.d_v;
        let dst = &mut self.slab[cell.offset as usize..cell.offset as usize + d_v];
        dst.copy_from_slice(v);
        self.cells[idx].writes += 1;
    }

    /// The φ-blended read at `x` with the row-0 (overwrite-lane) query —
    /// [`read_cells`](Self::read_cells) with `q = e₀`, no `e₀` buffer.
    pub fn read_blend(&self, x: &[f32], out: &mut [f32]) {
        self.read_impl(x, Query::Row0, out);
    }

    /// The unblended primary-cell read of the overwrite lane (Issue 938
    /// T4.1): `write_value` + `read_direct` is a BIT-IDENTICAL round trip.
    ///
    /// The tent's exact-hit center mass is 0.6 per axis (and 0.6^A for the
    /// blended read at an exact hit — 0.36 at `A = 2`, 0.216 at `A = 3`), so
    /// a POSITIONAL consumer — data placed by address, not keyed by content —
    /// must read through this lane and not [`read_blend`](Self::read_blend).
    /// mLattice found the same boundary independently (their MAD Copy = 0.003
    /// content lane forced `write_cell`/`read_cell` into their API).
    ///
    /// Returns whether the primary cell was written; an unwritten cell
    /// zero-fills `out` and returns `false`. Zero-alloc (one memcpy).
    pub fn read_direct(&self, x: &[f32], out: &mut [f32]) -> bool {
        assert_eq!(out.len(), self.config.d_v, "out width must be d_v");
        out.fill(0.0);
        let located = self.locate(x);
        let idx = self.cell_index(located.cell);
        let cell = self.cells[idx];
        if !cell.is_written() {
            return false;
        }
        let base = cell.offset as usize;
        let d_v = self.config.d_v;
        out.copy_from_slice(&self.slab[base..base + d_v]);
        true
    }

    /// The φ-blended delta-rule read at `x`: `Σ_cells φ(cell)·(S_cᵀ q)`,
    /// written into `out` (caller-owned — the hot path allocates nothing).
    ///
    /// Per-cell read-then-blend (the T1.4 law): each written cell in the 3^A
    /// neighborhood contributes `φ_c·(S_cᵀ q)` by row-wise direct
    /// accumulation — `3^A × d_k` row MACs at `d_v` wide — and no cross-cell
    /// state is ever materialized (sum-then-read would need a `d_k·d_v`
    /// scratch AND read every row of every cell).
    pub fn read_cells(&self, x: &[f32], q: &[f32], out: &mut [f32]) {
        assert_eq!(q.len(), self.config.d_k, "query width must be d_k");
        self.read_impl(x, Query::Dense(q), out);
    }

    /// [`read_cells`](Self::read_cells) plus the **sigmoid-only content
    /// score** (T1.6): `σ(λ·(2·wf − 1))` where `wf` is the fraction of the
    /// query's in-grid kernel mass landing on WRITTEN cells.
    ///
    /// This is the routing/content factorization the consumer needs in one
    /// pass: the read IS the content, the score IS the routing confidence —
    /// full coverage → `σ(λ)`, empty support → `σ(−λ)`, half → 0.5. Sigmoid,
    /// never softmax (the project law — gated by the module's grep test);
    /// the shared `crate::sigmoid` is the substrate kernel.
    ///
    /// `λ` must be finite and `> 0` (it is a slope, not a probability).
    #[must_use]
    pub fn read_scored(&self, x: &[f32], q: &[f32], out: &mut [f32], lambda: f32) -> f32 {
        assert_eq!(q.len(), self.config.d_k, "query width must be d_k");
        assert!(
            lambda > 0.0 && lambda.is_finite(),
            "lambda must be finite > 0"
        );
        let wf = self.read_impl(x, Query::Dense(q), out);
        crate::sigmoid(lambda * (2.0 * wf - 1.0))
    }

    /// Serializable state snapshot — the freeze half of the Plan 199 T1.B
    /// consumer surface. Pairs with [`Self::commitment`]: the artifact and
    /// its BLAKE3 digest travel together, and [`Self::restore`] is the thaw
    /// (whole-artifact swap semantics — never a blend with current state).
    ///
    /// Batch-path by design: this clones the cell table and the slab, which
    /// is exactly what a freeze is. The G4 zero-alloc law covers the hot
    /// write/read path only, and this is neither.
    #[must_use]
    pub fn snapshot(&self) -> LatticeSnapshot {
        LatticeSnapshot {
            cells: self.cells.clone(),
            slab: self.slab.clone(),
            cursor: self.cursor,
        }
    }

    /// Thaw: replace the current state with a snapshot taken from a lattice
    /// of the SAME geometry (built from the same `LatticeConfig`). After a
    /// successful restore, reads are bitwise
    /// what the source lattice's reads were at snapshot time, and later
    /// writes continue from the snapshot's slab cursor.
    ///
    /// # Errors
    /// [`LatticeSnapshotError::GeometryMismatch`] when the snapshot came
    /// from a differently-shaped lattice;
    /// [`LatticeSnapshotError::CursorOutOfBounds`] or
    /// [`LatticeSnapshotError::CellOutOfBounds`] when the snapshot is
    /// corrupt — refused here so a thawed lie can never reach the hot path's
    /// slab indexing.
    pub fn restore(&mut self, snapshot: &LatticeSnapshot) -> Result<(), LatticeSnapshotError> {
        if snapshot.cells.len() != self.cells.len() || snapshot.slab.len() != self.slab.len() {
            return Err(LatticeSnapshotError::GeometryMismatch {
                expected_cells: self.cells.len(),
                found_cells: snapshot.cells.len(),
                expected_slab: self.slab.len(),
                found_slab: snapshot.slab.len(),
            });
        }
        if snapshot.cursor > snapshot.slab.len() {
            return Err(LatticeSnapshotError::CursorOutOfBounds {
                cursor: snapshot.cursor,
                slab_len: snapshot.slab.len(),
            });
        }
        let slab_len = snapshot.slab.len();
        for (idx, cell) in snapshot.cells.iter().enumerate() {
            let end = u64::from(cell.offset) + u64::from(cell.len);
            if end > slab_len as u64 {
                return Err(LatticeSnapshotError::CellOutOfBounds {
                    cell_index: idx,
                    offset: cell.offset,
                    len: cell.len,
                    slab_len,
                });
            }
        }
        self.cells.clone_from(&snapshot.cells);
        self.slab.clone_from(&snapshot.slab);
        self.cursor = snapshot.cursor;
        Ok(())
    }

    /// BLAKE3 commitment over the full state: a geometry header (every grid
    /// side in axis order, then `d_k`, `d_v`, lengths, cursor), the cell
    /// table, then every slab element — all little-endian, so commitments
    /// are comparable across machines. Byte-identical state commits
    /// identically; any written byte moves the commitment. The `A = 2`
    /// header byte stream is exactly the pre-generic one.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for side in self.config.grid {
            hasher.update(&side.to_le_bytes());
        }
        hasher.update(&(self.config.d_k as u64).to_le_bytes());
        hasher.update(&(self.config.d_v as u64).to_le_bytes());
        hasher.update(&(self.cells.len() as u64).to_le_bytes());
        hasher.update(&(self.slab.len() as u64).to_le_bytes());
        hasher.update(&(self.cursor as u64).to_le_bytes());
        for cell in &self.cells {
            hasher.update(&cell.offset.to_le_bytes());
            hasher.update(&cell.len.to_le_bytes());
            hasher.update(&cell.writes.to_le_bytes());
        }
        // 64 KiB of f32 per update — no slab-sized copy, no per-element call.
        let mut buf = [0u8; 4 * 16_384];
        for chunk in self.slab.chunks(16_384) {
            let (buf_rows, _) = buf.as_chunks_mut::<4>();
            for (dst, src) in buf_rows.iter_mut().zip(chunk.iter()) {
                dst.copy_from_slice(&src.to_le_bytes());
            }
            hasher.update(&buf[..chunk.len() * 4]);
        }
        *hasher.finalize().as_bytes()
    }

    /// The shared read body — blends the neighborhood into `out` and returns
    /// the written-mass fraction of the in-grid kernel mass (the T1.6 score
    /// input; `0.0` for an all-empty support).
    ///
    /// The neighborhood is the 3^A product over per-axis 3-tap rows, visited
    /// in row-major tap order (axis 0 most significant) — at `A = 2` exactly
    /// the historical `(row, col)` double loop, so the mass/out accumulation
    /// ORDER (and therefore every float result) is unchanged.
    fn read_impl(&self, x: &[f32], query: Query<'_>, out: &mut [f32]) -> f32 {
        assert_eq!(out.len(), self.config.d_v, "out width must be d_v");
        out.fill(0.0);
        let located = self.locate(x);
        let w_axis = self.neighborhood_weights(located.delta);
        let taps_pow = Self::TAPS_POW;
        let mut tap_stride = [0_usize; A];
        let mut s = 1_usize;
        for axis in (0..A).rev() {
            tap_stride[axis] = s;
            s *= TAPS;
        }
        let d_v = self.config.d_v;
        let d_k = self.config.d_k;
        // The kernel's in-grid mass — data-independent shape walk, computed
        // first; the written share rides the same walk (cell-table reads
        // only).
        let mut mass = 0.0_f32;
        let mut written_mass = 0.0_f32;
        for tap in 0..taps_pow {
            if let Some((phi, cell_idx)) = self.tap_location(tap, &located, &w_axis, &tap_stride) {
                mass += phi;
                if self.cells[cell_idx].is_written() {
                    written_mass += phi;
                }
            }
        }
        if mass <= f32::EPSILON {
            return 0.0;
        }
        for tap in 0..taps_pow {
            let Some((phi, cell_idx)) = self.tap_location(tap, &located, &w_axis, &tap_stride)
            else {
                continue;
            };
            let phi = phi / mass;
            let cell = self.cells[cell_idx];
            if !cell.is_written() {
                continue;
            }
            let base = cell.offset as usize;
            match query {
                Query::Row0 => {
                    let src = &self.slab[base..base + d_v];
                    for (o, &s) in out.iter_mut().zip(src.iter()) {
                        *o += phi * s;
                    }
                }
                Query::Dense(q) => {
                    for (i, &qi) in q.iter().take(d_k).enumerate() {
                        if qi == 0.0 {
                            continue;
                        }
                        let w = phi * qi;
                        let row = &self.slab[base + i * d_v..base + (i + 1) * d_v];
                        for (o, &s) in out.iter_mut().zip(row.iter()) {
                            *o += w * s;
                        }
                    }
                }
            }
        }
        written_mass / mass
    }

    /// One neighborhood tap: the per-axis tap index decomposition, its
    /// product weight, and the in-bounds check — `None` when any axis'
    /// tap falls outside the grid (the read renormalizes over the clipped
    /// neighborhood). The weight product multiplies in axis order, which at
    /// `A = 2` is exactly the historical `wr * wc`.
    #[inline]
    fn tap_location(
        &self,
        tap: usize,
        located: &Located<A>,
        w_axis: &[[f32; TAPS]; A],
        tap_stride: &[usize; A],
    ) -> Option<(f32, usize)> {
        let mut weight = 1.0_f32;
        let mut flat = 0_usize;
        for axis in 0..A {
            let t = (tap / tap_stride[axis]) % TAPS;
            weight *= w_axis[axis][t];
            let c = located.cell[axis] as i64 + t as i64 - 1;
            if c < 0 || c >= i64::from(self.config.grid[axis]) {
                return None;
            }
            flat = flat * self.config.grid[axis] as usize + c as usize;
        }
        Some((weight, flat))
    }

    /// Per-axis 3-tap weights, dispatched on the configured kernel (T1.8's
    /// tunable).
    #[must_use]
    fn neighborhood_weights(&self, delta: [f32; A]) -> [[f32; TAPS]; A] {
        let mut w = [[0.0_f32; TAPS]; A];
        for (axis, row) in w.iter_mut().enumerate() {
            *row = match self.config.bump_kernel {
                BumpKernel::Tent => {
                    let (a, b, c) = axis_weights(&self.lut, delta[axis]);
                    [a, b, c]
                }
                BumpKernel::Cos2 => {
                    let (a, b, c) = axis_weights_cos2(delta[axis]);
                    [a, b, c]
                }
            };
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::LatticeMemory;
    use super::cdf::CdfWarp;
    use super::types::{BumpKernel, LatticeConfig, LatticeConfigError};

    fn config(grid: [u32; 2], w: [f32; 2]) -> LatticeConfig {
        LatticeConfig {
            d: 16,
            d_k: 4,
            d_v: 4,
            grid,
            w,
            seed: 42,
            byte_budget: None,
            warp: None,
            bump_kernel: BumpKernel::Tent,
        }
    }

    /// A unit key along `i`, `d_k`-wide, allocated per call (test-only).
    fn unit_key(i: usize, d_k: usize) -> Vec<f32> {
        let mut k = vec![0.0_f32; d_k];
        k[i] = 1.0;
        k
    }

    #[test]
    fn constructor_refuses_zero_and_over_budget() {
        let bad = config([0, 4], [1.0, 1.0]);
        assert!(matches!(
            LatticeMemory::new(bad),
            Err(LatticeConfigError::ZeroDimension)
        ));
        let mut big = config([4, 4], [1.0, 1.0]);
        big.byte_budget = Some(16); // slab needs 16·4·4·4 = 1024 B
        assert!(matches!(
            LatticeMemory::new(big),
            Err(LatticeConfigError::OverBudget { .. })
        ));
    }

    #[test]
    fn write_then_exact_read_returns_the_value() {
        let mut lat = LatticeMemory::new(config([64, 64], [2.0, 2.0])).unwrap();
        let v = [0.25, -0.5, 0.75, 1.0];
        lat.write_value(&[0.1; 16], &v);
        let mut out = [0.0_f32; 4];
        lat.read_blend(&[0.1; 16], &mut out);
        // With only the stored cell written, the read is k·v — DIRECTION
        // preserved (every component the same factor). k = the stored cell's
        // renormalized φ ≤ 1 (the in-grid-normalized weights stay a partition
        // of unity; interior queries cap at center_max² = 0.36, grid-edge
        // queries renormalize over the clipped neighborhood and may carry
        // more). Retrieval ranks on direction; the dilution gradient is what
        // G5 contrasts against Engram's cliff.
        let k = out[0] / v[0];
        assert!(k > 0.0 && k <= 1.0 + 1e-6, "read factor {k} outside (0, 1]");
        for (o, &s) in out.iter().zip(v.iter()) {
            assert!((o - k * s).abs() < 1e-6, "direction broken: {o} vs {k}·{s}");
        }
    }

    #[test]
    fn near_miss_recovers_through_neighbor_weight() {
        // T1.2's property: a query whose PRIMARY cell differs from the
        // stored key's still recovers the stored value — graded, not a
        // cliff. Sweep past a cell boundary; recall must stay positive
        // while neighborhood overlap remains and decay with distance.
        let mut lat = LatticeMemory::new(config([64, 64], [1.0, 1.0])).unwrap();
        let key = [0.0_f32; 16];
        lat.write_value(&key, &[1.0; 4]);
        // Find the first primary-cell crossing on a monotone sweep.
        let mut boundary = None;
        let mut prev_cell = lat.primary_cell(&key)[0];
        for step in 1..4096_u32 {
            let off = step as f32 * 0.001;
            let mut q = key;
            q[0] = off;
            let row = lat.primary_cell(&q)[0];
            if row != prev_cell {
                boundary = Some(off);
                break;
            }
            prev_cell = row;
        }
        let boundary = boundary.expect("a 1.0-width lattice must have a boundary within the sweep");
        let mut out = [0.0_f32; 4];
        let mut graded = Vec::new();
        for k in 1..=8_u32 {
            let off = boundary + k as f32 * 0.02;
            let mut q = key;
            q[0] = off;
            lat.read_blend(&q, &mut out);
            let recall = out[0]; // v was all-ones → the blended mass IS recall
            assert!(
                recall > 0.0,
                "near-miss at +{k}·0.02 past the boundary reads ZERO — cliff, not graded"
            );
            graded.push(recall);
        }
        // Graded = decaying with distance, not flat or inverted.
        assert!(
            graded[0] > graded[graded.len() - 1],
            "recall must decay with distance: {graded:?}"
        );
    }

    #[test]
    fn unwritten_lattice_reads_zero() {
        let lat = LatticeMemory::new(config([8, 8], [2.0, 2.0])).unwrap();
        let mut out = [1.0_f32; 4]; // poisoned — must be overwritten to 0
        lat.read_blend(&[0.3; 16], &mut out);
        for o in out {
            assert_eq!(o, 0.0);
        }
    }

    #[test]
    fn same_config_same_lattice() {
        let a = LatticeMemory::new(config([16, 16], [2.0, 2.0])).unwrap();
        let b = LatticeMemory::new(config([16, 16], [2.0, 2.0])).unwrap();
        let mut oa = [0.0_f32; 4];
        let mut ob = [0.0_f32; 4];
        // (reads on unwritten lattices — the axes agreement is what's pinned)
        a.read_blend(&[0.2; 16], &mut oa);
        b.read_blend(&[0.2; 16], &mut ob);
        assert_eq!(oa, ob);
    }

    // ── T1.4: the delta rule ──────────────────────────────────────────────

    #[test]
    fn delta_overwrite_removes_along_k() {
        // The forgetting law: with unit keys and β = 1 the update removes
        // the state along k EXACTLY — overwriting (k, v_a) with (k, v_b)
        // leaves the v_b component as the read's whole direction and the
        // v_a component at float noise. Assertions are RATIO-based: the
        // read blends the 3×3 neighborhood, so the absolute scale is the
        // primary cell's φ (≈ 0.36 interior) — the law is in the direction.
        let mut lat = LatticeMemory::new(config([32, 32], [2.0, 2.0])).unwrap();
        let x = [0.4_f32; 16];
        let k = [0.5_f32, 0.5, 0.5, 0.5]; // unit: ‖k‖² = 1
        let v_a = [1.0_f32, 0.0, 0.0, 0.0];
        let v_b = [0.0_f32, 1.0, 0.0, 0.0]; // ⊥ v_a
        lat.write_delta(&x, &k, &v_a, 1.0);
        let mut out = [0.0_f32; 4];
        lat.read_cells(&x, &k, &mut out);
        let a_scale = out[0];
        assert!(
            a_scale > 0.2 && out[1].abs() < 1e-4 * a_scale,
            "first association must be the read's whole direction: {out:?}"
        );
        lat.write_delta(&x, &k, &v_b, 1.0);
        lat.read_cells(&x, &k, &mut out);
        let b_scale = out[1];
        assert!(
            b_scale > 0.2,
            "new value must dominate after overwrite: {out:?}"
        );
        let old_recall = out[0] / b_scale;
        assert!(
            old_recall.abs() < 1e-4,
            "old value must be REMOVED along k (recall ≈ 0 relative), got {old_recall}"
        );
        // The exact-arithmetic identity: after two same-key β=1 writes the
        // state is k·v_bᵀ alone — every other component is float noise.
        assert!((out[2] / b_scale).abs() < 1e-4 && (out[3] / b_scale).abs() < 1e-4);
    }

    #[test]
    fn near_collinear_interference_documented() {
        // W near-collinear overwrites: the first value's survival matches
        // the exact single-step anchor, and after the full chain the v₁
        // association is destroyed — every later read sits at the sin²θ
        // noise scale, ≪ the original. (NOT asserted: per-step monotonicity
        // of the k₁-component — the (I − kkᵀ) products contract the channel
        // vector's NORM, but its k₁-component can rotate; the measured
        // recalls below ARE the documented interference record.)
        let d_k = 4;
        let mut cfg = config([16, 16], [2.0, 2.0]);
        cfg.d_k = d_k;
        let mut lat = LatticeMemory::new(cfg).unwrap();
        let x = [0.2_f32; 16];
        // k₁ = e₀; overwrites at angle θ around it.
        let k1 = unit_key(0, d_k);
        let theta = 0.1_f32;
        let v1 = [1.0_f32, 0.0, 0.0, 0.0];
        lat.write_delta(&x, &k1, &v1, 1.0);
        let mut recall = Vec::new();
        let mut out = [0.0_f32; 4];
        lat.read_cells(&x, &k1, &mut out);
        recall.push(out[0]);
        let mut first_cos = 1.0_f32; // cos(k₁, k₂) — the anchor's cosine
        for w in 1..8_usize {
            let mut kw = unit_key(0, d_k);
            kw[1] = theta * (w as f32 * 0.7).sin(); // deterministic, < 1
            let n = (kw[0] * kw[0] + kw[1] * kw[1]).sqrt();
            kw[0] /= n;
            kw[1] /= n;
            if w == 1 {
                first_cos = kw[0]; // k₁ = e₀ ⇒ k₁ᵀk_w = kw[0]
            }
            // Orthogonal marker values keep the v₁ channel separable.
            let mut vw = [0.0_f32; 4];
            vw[1 + (w % 3)] = 1.0;
            lat.write_delta(&x, &kw, &vw, 1.0);
            lat.read_cells(&x, &k1, &mut out);
            recall.push(out[0]);
        }
        // Anchor: after the FIRST overwrite the v₁ survival factor is the
        // read RATIO recall[1]/recall[0] (the read blends, so absolutes are
        // φ-scaled) and equals 1 − (k₁ᵀk₂)² exactly (k₁ᵀ(I − k₂k₂ᵀ)k₁ with
        // unit vectors).
        let survival_1 = recall[1] / recall[0];
        let expected = 1.0 - first_cos * first_cos;
        assert!(
            (survival_1 - expected).abs() < 1e-3,
            "first-overwrite survival {survival_1} must ≈ 1 − (k₁ᵀk₂)² = {expected}"
        );
        // Destroyed: every post-overwrite read is ≤ 2% of the original
        // association (measured ~0.5%), vs the 100% before the chain.
        for w in 1..recall.len() {
            assert!(
                recall[w].abs() <= 0.02 * recall[0],
                "overwrite {w} left recall {:+.5} above the noise scale (original {:.5}): {recall:?}",
                recall[w],
                recall[0]
            );
        }
        // And the chain's FINAL read is no larger than the first step's —
        // the interference stays bounded at the sin²θ scale, it does not
        // re-grow.
        assert!(
            recall[recall.len() - 1].abs() <= recall[1].abs() * 3.0,
            "final recall {:+.5} escaped the first-step scale {:+.5}",
            recall[recall.len() - 1],
            recall[1]
        );
    }

    #[test]
    fn write_value_is_the_row0_delta_lane() {
        // The two write paths agree on the lane they share: write_value
        // followed by a dense e₀ read == write_value + read_blend.
        let mut lat = LatticeMemory::new(config([16, 16], [2.0, 2.0])).unwrap();
        let x = [0.6_f32; 16];
        let v = [0.5_f32, -0.25, 1.0, 0.0];
        lat.write_value(&x, &v);
        let e0 = unit_key(0, 4);
        let mut dense = [0.0_f32; 4];
        let mut blend = [0.0_f32; 4];
        lat.read_cells(&x, &e0, &mut dense);
        lat.read_blend(&x, &mut blend);
        for (a, b) in dense.iter().zip(blend.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "row-0 paths disagree: {dense:?} vs {blend:?}"
            );
        }
    }

    // ── Issue 938 T4: the positional-copy boundary ────────────────────────

    #[test]
    fn read_direct_is_the_exact_write_value_round_trip() {
        // The T4.2 law: write_value + read_direct is BIT-IDENTICAL, while
        // the blended read at the same address is the graded dilution (the
        // tent's ≈ 0.6 exact-hit center mass, renormalized over the 3×3
        // support). A positional consumer must use read_direct.
        let mut lat = LatticeMemory::new(config([32, 32], [2.0, 2.0])).unwrap();
        let x = [0.6_f32; 16];
        let v = [0.5_f32, -0.25, 1.0, 0.0];
        lat.write_value(&x, &v);
        let mut direct = [0.0_f32; 4];
        assert!(
            lat.read_direct(&x, &mut direct),
            "written cell must read true"
        );
        assert_eq!(direct, v, "read_direct must be the exact round trip");
        // The blended read is GRADED, not exact: direction preserved (every
        // component scaled by one factor — the written cell's normalized
        // kernel mass), strictly below 1.0. The 0.6 exact-hit center mass is
        // a per-axis bump property (pinned in bump.rs); the blended factor
        // at an arbitrary query is that mass to the A-th power, renormalized
        // over the in-grid support — geometry-dependent, so the contract
        // asserted here is direction + dilution, not a constant.
        let mut blend = [0.0_f32; 4];
        lat.read_blend(&x, &mut blend);
        let g = blend[0] / v[0];
        assert!(
            g > 0.0 && g <= 1.0 + 1e-6,
            "blend factor {g} outside (0, 1]"
        );
        for (b, &s) in blend.iter().zip(v.iter()) {
            assert!(
                (b - g * s).abs() < 1e-6,
                "blend broke direction: {b} vs {g}·{s}"
            );
        }
        assert!(
            (g - 1.0).abs() > 1e-6,
            "blend must be DILUTED, not exact (g = {g})"
        );
        // Unwritten cell: zeros + false.
        let mut out = [9.0_f32; 4]; // poisoned
        assert!(
            !lat.read_direct(&[0.99; 16], &mut out),
            "unwritten must read false"
        );
        assert!(
            out.iter().all(|&o| o == 0.0),
            "unwritten read must zero-fill"
        );
    }

    #[test]
    fn read_direct_survives_the_freeze_thaw_round_trip() {
        // The unblended lane is state like any other: snapshot/restore must
        // preserve it bitwise (the Plan 199 consumer surface, T4 rider).
        let mut lat = LatticeMemory::new(config([16, 16], [2.0, 2.0])).unwrap();
        let x = [0.35_f32; 16];
        let v = [3.5_f32, -1.25, 0.0, 7.0];
        lat.write_value(&x, &v);
        let snap = lat.snapshot();
        let mut before = [0.0_f32; 4];
        lat.read_direct(&x, &mut before);
        lat.restore(&snap).expect("restore");
        let mut after = [0.0_f32; 4];
        assert!(lat.read_direct(&x, &mut after));
        assert_eq!(
            before, after,
            "thaw must preserve the unblended lane bitwise"
        );
    }

    // ── T1.6: sigmoid-only scoring ────────────────────────────────────────

    #[test]
    fn scoring_is_sigmoid_shaped_in_written_coverage() {
        // Full written support → score near σ(λ); empty → near σ(−λ); the
        // score is in (0, 1) always and rises with coverage. "Full" needs a
        // DENSE region — one write leaves the query's 3×3 support mostly
        // unwritten by construction (the sparsity attenuation IS the law).
        let mut lat = LatticeMemory::new(config([16, 16], [1.0, 1.0])).unwrap();
        let x = [0.35_f32; 16];
        let k = unit_key(0, 4);
        let mut out = [0.0_f32; 4];
        let empty = lat.read_scored(&x, &k, &mut out, 4.0);
        assert!(
            empty < 0.05,
            "empty support must score ≈ σ(−4), got {empty}"
        );
        // Dense coverage: 1000 jittered writes around x fill the cells of
        // x's neighborhood (16×16 grid, w=1 — the jitter stays within a few
        // cells of x on every axis).
        let mut rng = fastrand::Rng::with_seed(0x619_0616);
        for _ in 0..1000 {
            let mut q = x;
            for v in q.iter_mut() {
                *v += rng.f32() * 0.8 - 0.4;
            }
            lat.write_value(&q, &[1.0; 4]);
        }
        let full = lat.read_scored(&x, &k, &mut out, 4.0);
        assert!(
            full > 0.9,
            "densely-written support must score ≈ σ(+4), got {full}"
        );
        assert!(empty < full && (0.0..=1.0).contains(&empty) && (0.0..=1.0).contains(&full));
        // λ scales the sharpness: a small λ pulls both ends toward 0.5.
        let soft = lat.read_scored(&x, &k, &mut out, 0.2);
        assert!(soft > empty && soft < full && (soft - 0.5).abs() < 0.06);
    }

    #[test]
    fn sigmoid_only_law_grep_gate() {
        // The project law (sigmoid, never the banned normalizer) as a grep
        // gate over every file in this module, production AND test code.
        // Comment-stripped first (documentation may DISCUSS the law); the
        // needle is assembled from parts so this gate's own source cannot
        // trip it.
        let needle = format!("soft{}", "max");
        for (file, src) in [
            ("mod.rs", include_str!("mod.rs")),
            ("address.rs", include_str!("address.rs")),
            ("bump.rs", include_str!("bump.rs")),
            ("cdf.rs", include_str!("cdf.rs")),
            ("sizer.rs", include_str!("sizer.rs")),
            ("types.rs", include_str!("types.rs")),
        ] {
            let code = strip_comments(src).to_ascii_lowercase();
            assert!(
                !code.contains(&needle),
                "{file} names the banned normalizer in CODE — the project law is sigmoid-only"
            );
        }
    }

    /// Naive comment stripper for the grep gate: drops `//`-to-EOL and
    /// `/* */` regions, leaves everything else. String literals are not
    /// tracked (a `"//"` inside a string would clip the rest of a line —
    /// harmless here: the gate only ever needs to NOT see the word, and a
    /// clipped line cannot manufacture one).
    fn strip_comments(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut chars = src.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '/' if chars.peek() == Some(&'/') => {
                    for n in chars.by_ref() {
                        if n == '\n' {
                            out.push('\n');
                            break;
                        }
                    }
                }
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    let mut closed = false;
                    while let Some(n) = chars.next() {
                        if n == '*' && chars.peek() == Some(&'/') {
                            chars.next();
                            closed = true;
                            break;
                        }
                    }
                    assert!(closed, "unterminated block comment in module source");
                }
                _ => out.push(c),
            }
        }
        out
    }

    // ── T1.5 smoke: the warp rides the config end-to-end ─────────────────

    #[test]
    fn warped_lattice_stays_local_and_graded() {
        // A hand-fitted warp on an identity-ish projection: locality and
        // graded near-miss must survive the remap (the sizer's end-to-end
        // histogram gate lives in sizer.rs).
        let mut cfg = config([64, 64], [1.0, 1.0]);
        cfg.warp = Some([
            CdfWarp {
                mean: 0.0,
                inv_std: 1.0,
                cells: 64.0,
            },
            CdfWarp {
                mean: 0.0,
                inv_std: 1.0,
                cells: 64.0,
            },
        ]);
        let mut lat = LatticeMemory::new(cfg).unwrap();
        let key = [0.5_f32; 16];
        lat.write_value(&key, &[1.0; 4]);
        let cell = lat.primary_cell(&key);
        // A small perturbation must stay within ±1 cell on both axes.
        let mut q = key;
        q[3] += 0.01;
        let cell2 = lat.primary_cell(&q);
        assert!(cell2[0].abs_diff(cell[0]) <= 1 && cell2[1].abs_diff(cell[1]) <= 1);
        // And the read must still be graded, not a cliff.
        let mut out = [0.0_f32; 4];
        lat.read_blend(&q, &mut out);
        assert!(
            out.iter().sum::<f32>() > 0.0,
            "warped near-miss read a cliff"
        );
    }
}
