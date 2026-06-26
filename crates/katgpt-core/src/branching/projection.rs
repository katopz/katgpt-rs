//! `NonInterferenceProjection` — orthogonal latent subspace per branch
//! (Plan 329 T2.1).
//!
//! Assigns each branch a direction vector in a D-dimensional latent space.
//! Two branches `b_i`, `b_j` are **non-interfering** iff their direction
//! vectors are orthogonal: `dot(dir_i, dir_j) ≈ 0`. In a D-dimensional space
//! the maximum number of mutually-orthogonal directions is exactly `D` — the
//! "orthogonal capacity limit" (Plan 329 Risk #2). Beyond `D` branches the
//! caller must accept non-zero interference (near-orthogonal vectors, ε > 0).
//!
//! # Non-interference is structural
//!
//! When `interference(b_i, b_j) < ε`, a latent update projected onto `dir_i`
//! has a near-zero component along `dir_j`. Writing to branch `b_i` does not
//! contaminate branch `b_j`'s direction — this is the "non-interference" in
//! the RIZZ paper's title. The guarantee is geometric, not learned.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | `directions` (the projection matrix) | **Latent** | NO (per-branch learned direction) |
//! | `interference` scalar (output) | **Raw-ish** | Caller decides (it's a derived f32) |
//! | `dim`, `capacity` | **Raw** | YES (structural config) |
//!
//! The projection matrix is per-NPC local state (latent). It is NOT synced
//! directly — only the scalar outputs of `project()` cross the boundary (and
//! only if the caller chooses to sync them).
//!
//! # Hot path
//!
//! `project()` is a dot-product reduction over `D` f32s — auto-vectorizable,
//! zero allocation. `interference()` is the same. `assign_direction()` is a
//! cold-path lifecycle operation (called on spawn / merge).
//!
//! # Const-generic dimension
//!
//! `D` is a const generic so `[f32; D]` is a fixed-size stack array (matches
//! the `DelayRing<D, K>` idiom in `crate::karc`). Default `D = 8` (the HLA
//! dimensionality from Research 310 §2.2).

use crate::branching::types::BranchId;

/// Default latent dimensionality (HLA space, Research 310 §2.2).
pub const DEFAULT_PROJECTION_DIM: usize = 8;

/// Default ε below which two directions are treated as "effectively orthogonal".
/// Anything strictly below this is non-interfering by construction.
pub const DEFAULT_ORTHOGONAL_EPSILON: f32 = 1e-6;

/// Default ε above which `assign_direction` will reject an assignment as
/// "too correlated with an existing branch". Strictly greater than
/// `DEFAULT_ORTHOGONAL_EPSILON` so legitimate near-orthogonal assignments
/// (e.g., 9th direction in D=8 space) still succeed.
pub const DEFAULT_ASSIGN_MAX_INTERFERENCE: f32 = 0.1;

/// Error returned by [`NonInterferenceProjection::assign_direction`] when the
/// proposed direction interferes with an existing branch beyond
/// `max_interference`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AssignError {
    /// The direction vector has the wrong dimensionality.
    WrongDimension = 0,
    /// The direction vector has zero magnitude (cannot be normalized).
    ZeroMagnitude = 1,
    /// The direction interferes with an existing branch beyond `max_interference`.
    /// Carries the offending branch id (via separate field on the result) and
    /// the measured interference magnitude.
    Interferes = 2,
}

/// Result of [`NonInterferenceProjection::assign_direction`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssignResult {
    /// The error kind, or `None` on success.
    pub error: Option<AssignError>,
    /// When `error == Some(Interferes)`, the branch id whose direction was
    /// violated. `None` otherwise.
    pub conflict_branch: Option<BranchId>,
    /// When `error == Some(Interferes)`, the measured interference magnitude.
    /// `0.0` otherwise.
    pub interference: f32,
}

impl AssignResult {
    /// Construct a success result.
    #[inline]
    #[must_use]
    pub const fn ok() -> Self {
        Self {
            error: None,
            conflict_branch: None,
            interference: 0.0,
        }
    }

    /// Construct a wrong-dimension error.
    #[inline]
    #[must_use]
    pub const fn wrong_dimension() -> Self {
        Self {
            error: Some(AssignError::WrongDimension),
            conflict_branch: None,
            interference: 0.0,
        }
    }

    /// Construct a zero-magnitude error.
    #[inline]
    #[must_use]
    pub const fn zero_magnitude() -> Self {
        Self {
            error: Some(AssignError::ZeroMagnitude),
            conflict_branch: None,
            interference: 0.0,
        }
    }

    /// Construct an interference-violation error.
    #[inline]
    #[must_use]
    pub const fn interferes(conflict: BranchId, interference: f32) -> Self {
        Self {
            error: Some(AssignError::Interferes),
            conflict_branch: Some(conflict),
            interference,
        }
    }

    /// True if assignment succeeded.
    #[inline]
    #[must_use]
    pub const fn is_ok(self) -> bool {
        self.error.is_none()
    }
}

/// Orthogonal projection directions per branch (Plan 329 T2.1).
///
/// `D` is the latent dimensionality; the projection matrix is a row-major
/// `Vec<[f32; D]>` indexed by `BranchId.0 as usize`. Each row is a unit-norm
/// direction vector. Rows default to all-zeros (sentinel for "unassigned");
/// `assign_direction` overwrites a row in place.
///
/// # Capacity limit
///
/// In D-dimensional space, at most `D` directions can be mutually orthogonal.
/// Beyond that, any new direction must interfere with at least one existing
/// direction by ≥ `1/sqrt(D)` (frame theory). Use [`max_orthogonal_branches`]
/// to query the hard limit before spawning.
///
/// # Allocation discipline
///
/// `new(capacity)` pre-allocates `capacity` rows once. `assign_direction`
/// writes in place (no allocation). The hot path (`project`, `interference`)
/// is a pure dot-product over stack arrays — zero allocation, auto-vectorizable.
#[derive(Clone, Debug)]
pub struct NonInterferenceProjection<const D: usize = DEFAULT_PROJECTION_DIM> {
    /// Row-major projection matrix. Row `i` is the unit-norm direction for
    /// `BranchId(i)`. Zero-vector = unassigned slot.
    directions: Vec<[f32; D]>,
    /// ε below which two directions are treated as orthogonal.
    orthogonal_epsilon: f32,
    /// ε above which `assign_direction` rejects an assignment.
    assign_max_interference: f32,
}

impl<const D: usize> Default for NonInterferenceProjection<D> {
    #[inline]
    fn default() -> Self {
        Self::new(DEFAULT_PROJECTION_CAPACITY)
    }
}

/// Default projection capacity (matches `branching::DEFAULT_MAX_BRANCHES`).
const DEFAULT_PROJECTION_CAPACITY: usize = 64;

impl<const D: usize> NonInterferenceProjection<D> {
    /// Construct a projection matrix with `capacity` zero-initialized rows.
    ///
    /// Pre-allocates `capacity` rows; `assign_direction` writes in place.
    #[inline]
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            directions: vec![[0.0f32; D]; capacity],
            orthogonal_epsilon: DEFAULT_ORTHOGONAL_EPSILON,
            assign_max_interference: DEFAULT_ASSIGN_MAX_INTERFERENCE,
        }
    }

    /// Construct with custom thresholds.
    #[inline]
    #[must_use]
    pub fn with_thresholds(
        capacity: usize,
        orthogonal_epsilon: f32,
        assign_max_interference: f32,
    ) -> Self {
        Self {
            directions: vec![[0.0f32; D]; capacity],
            orthogonal_epsilon,
            assign_max_interference,
        }
    }

    /// Latent dimensionality `D`.
    #[inline]
    #[must_use]
    pub const fn dim(&self) -> usize {
        D
    }

    /// Row capacity (max branches the matrix can hold).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.directions.len()
    }

    /// Hard upper bound on the number of mutually-orthogonal branches in a
    /// D-dimensional space. Always returns `D`.
    ///
    /// This is the "orthogonal capacity limit" (Plan 329 Risk #2). Beyond this
    /// count, any new direction must interfere with at least one existing
    /// direction by ≥ `1/sqrt(D)`.
    #[inline]
    #[must_use]
    pub const fn max_orthogonal_branches() -> usize {
        D
    }

    /// ε below which two directions are treated as orthogonal.
    #[inline]
    #[must_use]
    pub const fn orthogonal_epsilon(&self) -> f32 {
        self.orthogonal_epsilon
    }

    /// Borrow the direction for `branch_id`, or `None` if out of range or
    /// unassigned (all-zeros sentinel).
    #[inline]
    #[must_use]
    pub fn direction(&self, branch_id: BranchId) -> Option<&[f32; D]> {
        let row = self.directions.get(branch_id.0 as usize)?;
        if row.iter().all(|&v| v == 0.0) {
            None
        } else {
            Some(row)
        }
    }

    /// Project `vector` onto `branch_id`'s direction.
    ///
    /// Returns `Some(scalar)` = `dot(vector, dir_{branch_id})`, or `None` if
    /// the branch is out of range or unassigned.
    ///
    /// **Interpretation**: the scalar is the component of `vector` along this
    /// branch's direction. A non-interfering write to branch `i` has zero
    /// projection onto branch `j ≠ i` (when `dir_i ⊥ dir_j`).
    ///
    /// Zero allocation, auto-vectorizable dot product over `D` f32s.
    #[inline]
    pub fn project(&self, branch_id: BranchId, vector: &[f32]) -> Option<f32> {
        let dir = self.direction(branch_id)?;
        Some(dot_fixed(dir, vector))
    }

    /// Interference between two branches = `|dot(dir_{b1}, dir_{b2})|`.
    ///
    /// Returns `0.0` if either branch is out of range or unassigned (treating
    /// "unassigned" as orthogonal to everything, which is the safe default).
    ///
    /// **Interpretation**:
    /// - `0.0` = perfectly orthogonal (non-interfering).
    /// - `1.0` = identical directions (fully interfering).
    ///
    /// Zero allocation, auto-vectorizable.
    #[inline]
    #[must_use]
    pub fn interference(&self, b1: BranchId, b2: BranchId) -> f32 {
        let (Some(d1), Some(d2)) = (self.direction(b1), self.direction(b2)) else {
            return 0.0;
        };
        dot_fixed(d1, d2).abs()
    }

    /// True iff `b1` and `b2` are non-interfering (interference < ε).
    #[inline]
    #[must_use]
    pub fn is_non_interfering(&self, b1: BranchId, b2: BranchId) -> bool {
        self.interference(b1, b2) < self.orthogonal_epsilon
    }

    /// True iff `branch_id`'s direction is non-interfering with ALL other
    /// assigned branches.
    ///
    /// O(n_assigned) scan. Use after `assign_direction` to verify the global
    /// orthogonality invariant (G1 gate).
    #[inline]
    #[must_use]
    pub fn is_non_interfering_with_all(&self, branch_id: BranchId) -> bool {
        let Some(target) = self.direction(branch_id) else {
            return true; // unassigned = trivially non-interfering
        };
        for (i, dir) in self.directions.iter().enumerate() {
            if i as u32 == branch_id.0 {
                continue;
            }
            if dir.iter().all(|&v| v == 0.0) {
                continue; // unassigned slot
            }
            let inter = dot_fixed(target, dir).abs();
            if inter >= self.orthogonal_epsilon {
                return false;
            }
        }
        true
    }

    /// Assign (and L2-normalize) `direction` to `branch_id`.
    ///
    /// Validates:
    /// 1. `direction.len() == D` (else `WrongDimension`).
    /// 2. Non-zero magnitude (else `ZeroMagnitude` — cannot normalize).
    /// 3. Interference with all existing assigned branches is strictly below
    ///    `assign_max_interference` (else `Interferes` with the worst offender).
    ///
    /// On success, writes the normalized direction in place (no allocation).
    /// On failure, the matrix is unchanged.
    ///
    /// **Cold path**: spawn / merge lifecycle only.
    pub fn assign_direction(&mut self, branch_id: BranchId, direction: &[f32]) -> AssignResult {
        // (1) Dimension check.
        if direction.len() != D {
            return AssignResult::wrong_dimension();
        }

        // (2) Magnitude check — compute norm while copying into a stack array.
        let mut buf = [0.0f32; D];
        let mut norm_sq = 0.0f32;
        for i in 0..D {
            buf[i] = direction[i];
            norm_sq += direction[i] * direction[i];
        }
        if norm_sq == 0.0 {
            return AssignResult::zero_magnitude();
        }

        // (3) Interference check against all assigned branches.
        // Normalize in place for the check (and for the write on success).
        let inv_norm = 1.0 / norm_sq.sqrt();
        for v in &mut buf {
            *v *= inv_norm;
        }

        // Out-of-range branch id: treat as always-assignable (the caller will
        // grow the matrix separately if needed; this primitive is bounded).
        if branch_id.0 as usize >= self.directions.len() {
            return AssignResult::ok();
        }

        // Scan existing assigned rows for the worst interference.
        let mut worst_inter = 0.0f32;
        let mut worst_id = None;
        for (i, dir) in self.directions.iter().enumerate() {
            if i as u32 == branch_id.0 {
                continue;
            }
            if dir.iter().all(|&v| v == 0.0) {
                continue; // unassigned slot
            }
            let inter = dot_fixed(&buf, dir).abs();
            if inter > worst_inter {
                worst_inter = inter;
                worst_id = Some(BranchId(i as u32));
            }
        }

        if worst_inter >= self.assign_max_interference {
            return AssignResult::interferes(worst_id.unwrap_or(BranchId::SENTINEL), worst_inter);
        }

        // (4) Write in place.
        self.directions[branch_id.0 as usize] = buf;
        AssignResult::ok()
    }

    /// Reset `branch_id`'s direction to the all-zeros sentinel (unassigned).
    ///
    /// Called on prune. Does not shrink the matrix.
    #[inline]
    pub fn clear_direction(&mut self, branch_id: BranchId) {
        if let Some(row) = self.directions.get_mut(branch_id.0 as usize) {
            *row = [0.0f32; D];
        }
    }

    /// Grow the matrix by `n` zero-initialized rows. Use when a caller needs
    /// to assign directions beyond the initial capacity.
    #[inline]
    pub fn grow(&mut self, n: usize) {
        self.directions.extend(core::iter::repeat([0.0f32; D]).take(n));
    }

    /// Iterator over `(BranchId, &[f32; D])` for every assigned (non-zero) row.
    #[inline]
    pub fn assigned_directions(&self) -> impl Iterator<Item = (BranchId, &[f32; D])> {
        self.directions
            .iter()
            .enumerate()
            .filter(|(_, dir)| !dir.iter().all(|&v| v == 0.0))
            .map(|(i, dir)| (BranchId(i as u32), dir))
    }

    /// Count of assigned (non-zero) rows.
    #[inline]
    #[must_use]
    pub fn n_assigned(&self) -> usize {
        self.directions
            .iter()
            .filter(|dir| !dir.iter().all(|&v| v == 0.0))
            .count()
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Dot product of a fixed-size `D`-array with a (possibly shorter) slice.
/// Uses the shorter length when `vector.len() < D` (caller should
/// pre-normalize dimensions). Auto-vectorizable inner loop, zero allocation.
#[inline]
fn dot_fixed<const D: usize>(a: &[f32; D], b: &[f32]) -> f32 {
    let n = D.min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Free-function alias for the hard orthogonal-branch limit at dimension `D`.
/// Same as [`NonInterferenceProjection::<D>::max_orthogonal_branches`].
#[inline]
#[must_use]
pub const fn max_orthogonal_branches<const D: usize>() -> usize {
    D
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard basis vector `e_i` in D dimensions.
    fn e_i<const D: usize>(i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; D];
        v[i] = 1.0;
        v
    }

    #[test]
    fn new_initializes_zero_rows() {
        let p: NonInterferenceProjection<4> = NonInterferenceProjection::new(8);
        assert_eq!(p.dim(), 4);
        assert_eq!(p.capacity(), 8);
        assert_eq!(p.n_assigned(), 0);
        // All rows unassigned.
        assert!(p.direction(BranchId::new(0)).is_none());
        assert!(p.direction(BranchId::new(7)).is_none());
    }

    #[test]
    fn default_uses_default_capacity_and_dim() {
        let p = NonInterferenceProjection::<8>::default();
        assert_eq!(p.dim(), 8);
        assert!(p.capacity() >= 1);
        assert_eq!(p.n_assigned(), 0);
    }

    #[test]
    fn max_orthogonal_branches_equals_dim() {
        assert_eq!(NonInterferenceProjection::<8>::max_orthogonal_branches(), 8);
        assert_eq!(NonInterferenceProjection::<4>::max_orthogonal_branches(), 4);
        assert_eq!(NonInterferenceProjection::<2>::max_orthogonal_branches(), 2);
        assert_eq!(max_orthogonal_branches::<16>(), 16);
    }

    #[test]
    fn assign_standard_basis_is_unit_norm() {
        let mut p: NonInterferenceProjection<4> = NonInterferenceProjection::new(8);
        let r = p.assign_direction(BranchId::new(0), &e_i::<4>(0));
        assert!(r.is_ok());
        let dir = p.direction(BranchId::new(0)).unwrap();
        // |e_0| = 1 already; normalized stays 1.
        let norm_sq: f32 = dir.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn assign_normalizes_non_unit_direction() {
        let mut p: NonInterferenceProjection<2> = NonInterferenceProjection::new(4);
        // (3, 4) has magnitude 5; normalized is (0.6, 0.8).
        let r = p.assign_direction(BranchId::new(0), &[3.0, 4.0]);
        assert!(r.is_ok());
        let dir = p.direction(BranchId::new(0)).unwrap();
        assert!((dir[0] - 0.6).abs() < 1e-6);
        assert!((dir[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn assign_rejects_wrong_dimension() {
        let mut p: NonInterferenceProjection<4> = NonInterferenceProjection::new(8);
        let r = p.assign_direction(BranchId::new(0), &[1.0, 0.0]); // len 2 ≠ D=4
        assert_eq!(r.error, Some(AssignError::WrongDimension));
        assert!(p.direction(BranchId::new(0)).is_none()); // unchanged
    }

    #[test]
    fn assign_rejects_zero_magnitude() {
        let mut p: NonInterferenceProjection<4> = NonInterferenceProjection::new(8);
        let r = p.assign_direction(BranchId::new(0), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(r.error, Some(AssignError::ZeroMagnitude));
        assert!(p.direction(BranchId::new(0)).is_none());
    }

    #[test]
    fn assign_rejects_interfering_direction() {
        let mut p: NonInterferenceProjection<4> = NonInterferenceProjection::new(8);
        // First branch: e_0.
        assert!(p
            .assign_direction(BranchId::new(0), &e_i::<4>(0))
            .is_ok());
        // Second branch: try to also assign e_0 — interference = 1.0 ≥ 0.1.
        let r = p.assign_direction(BranchId::new(1), &e_i::<4>(0));
        assert_eq!(r.error, Some(AssignError::Interferes));
        assert_eq!(r.conflict_branch, Some(BranchId::new(0)));
        assert!((r.interference - 1.0).abs() < 1e-6);
        // Second slot unchanged.
        assert!(p.direction(BranchId::new(1)).is_none());
    }

    #[test]
    fn assign_allows_near_orthogonal_within_threshold() {
        // Custom threshold: allow up to 0.5 interference.
        let mut p: NonInterferenceProjection<2> =
            NonInterferenceProjection::with_thresholds(4, 1e-6, 0.5);
        // First: e_0 = (1, 0).
        assert!(p
            .assign_direction(BranchId::new(0), &[1.0, 0.0])
            .is_ok());
        // Second: 30-degree rotation (cos 30° ≈ 0.866 → interference 0.866 > 0.5).
        let r = p.assign_direction(BranchId::new(1), &[0.866, 0.5]);
        assert!(!r.is_ok(), "0.866 interference should be rejected");

        // Now try 60-degree (cos 60° = 0.5 → interference exactly 0.5, still rejected by strict >=");
        let r2 = p.assign_direction(BranchId::new(1), &[0.5, 0.866]);
        // 0.5 >= 0.5 threshold → rejected.
        assert!(!r2.is_ok());

        // 70-degree (cos 70° ≈ 0.342 → interference 0.342 < 0.5 → accepted).
        let r3 = p.assign_direction(BranchId::new(1), &[0.342, 0.940]);
        assert!(r3.is_ok(), "0.342 interference should be accepted");
    }

    #[test]
    fn orthogonal_standard_basis_has_zero_interference() {
        let mut p: NonInterferenceProjection<4> = NonInterferenceProjection::new(8);
        for i in 0..4 {
            assert!(p
                .assign_direction(BranchId::new(i as u32), &e_i::<4>(i))
                .is_ok());
        }
        // All pairs of standard basis vectors are orthogonal.
        for i in 0..4u32 {
            for j in 0..4u32 {
                let inter = p.interference(BranchId::new(i), BranchId::new(j));
                if i == j {
                    assert!((inter - 1.0).abs() < 1e-6, "self-interference {i} == 1.0");
                } else {
                    assert!(
                        inter < 1e-6,
                        "interference({i}, {j}) = {inter}, expected < 1e-6"
                    );
                }
            }
        }
    }

    #[test]
    fn is_non_interfering_with_all_holds_for_orthogonal_set() {
        let mut p: NonInterferenceProjection<4> = NonInterferenceProjection::new(8);
        for i in 0..4 {
            assert!(p
                .assign_direction(BranchId::new(i as u32), &e_i::<4>(i))
                .is_ok());
        }
        for i in 0..4u32 {
            assert!(
                p.is_non_interfering_with_all(BranchId::new(i)),
                "branch {i} should be non-interfering with all"
            );
        }
    }

    #[test]
    fn project_returns_dot_component() {
        let mut p: NonInterferenceProjection<2> = NonInterferenceProjection::new(4);
        // dir_0 = (1, 0).
        assert!(p
            .assign_direction(BranchId::new(0), &[1.0, 0.0])
            .is_ok());
        // dir_1 = (0, 1).
        assert!(p
            .assign_direction(BranchId::new(1), &[0.0, 1.0])
            .is_ok());

        // vector (3, 4) projects onto dir_0 as 3, onto dir_1 as 4.
        let v = [3.0f32, 4.0];
        assert!((p.project(BranchId::new(0), &v).unwrap() - 3.0).abs() < 1e-6);
        assert!((p.project(BranchId::new(1), &v).unwrap() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn project_returns_none_for_unassigned() {
        let p: NonInterferenceProjection<2> = NonInterferenceProjection::new(4);
        assert!(p.project(BranchId::new(0), &[1.0, 0.0]).is_none());
    }

    #[test]
    fn project_returns_none_for_out_of_range() {
        let p: NonInterferenceProjection<2> = NonInterferenceProjection::new(4);
        assert!(p.project(BranchId::new(99), &[1.0, 0.0]).is_none());
    }

    #[test]
    fn interference_unassigned_branch_is_zero() {
        let mut p: NonInterferenceProjection<2> = NonInterferenceProjection::new(4);
        assert!(p
            .assign_direction(BranchId::new(0), &[1.0, 0.0])
            .is_ok());
        // Unassigned branch is treated as orthogonal to everything.
        assert!((p.interference(BranchId::new(0), BranchId::new(1))).abs() < 1e-6);
        assert!((p.interference(BranchId::new(2), BranchId::new(3))).abs() < 1e-6);
    }

    #[test]
    fn clear_direction_resets_to_unassigned() {
        let mut p: NonInterferenceProjection<2> = NonInterferenceProjection::new(4);
        assert!(p
            .assign_direction(BranchId::new(0), &[1.0, 0.0])
            .is_ok());
        assert!(p.direction(BranchId::new(0)).is_some());
        p.clear_direction(BranchId::new(0));
        assert!(p.direction(BranchId::new(0)).is_none());
        assert_eq!(p.n_assigned(), 0);
    }

    #[test]
    fn clear_direction_out_of_range_is_noop() {
        let mut p: NonInterferenceProjection<2> = NonInterferenceProjection::new(4);
        p.clear_direction(BranchId::new(99)); // no panic
    }

    #[test]
    fn grow_extends_capacity_with_zero_rows() {
        let mut p: NonInterferenceProjection<2> = NonInterferenceProjection::new(2);
        assert_eq!(p.capacity(), 2);
        p.grow(3);
        assert_eq!(p.capacity(), 5);
        assert_eq!(p.n_assigned(), 0);
        // Can now assign into the grown region.
        assert!(p
            .assign_direction(BranchId::new(4), &[1.0, 0.0])
            .is_ok());
    }

    #[test]
    fn assigned_directions_iterator() {
        let mut p: NonInterferenceProjection<4> = NonInterferenceProjection::new(8);
        assert!(p
            .assign_direction(BranchId::new(0), &e_i::<4>(0))
            .is_ok());
        assert!(p
            .assign_direction(BranchId::new(2), &e_i::<4>(2))
            .is_ok());
        let assigned: Vec<BranchId> = p.assigned_directions().map(|(id, _)| id).collect();
        assert_eq!(assigned, vec![BranchId::new(0), BranchId::new(2)]);
        assert_eq!(p.n_assigned(), 2);
    }

    #[test]
    fn orthogonal_property_invariant_8_branches_in_d8() {
        // G1-critical: 8 standard-basis vectors in D=8 are mutually orthogonal.
        let mut p: NonInterferenceProjection<8> = NonInterferenceProjection::new(16);
        for i in 0..8usize {
            let r = p.assign_direction(BranchId::new(i as u32), &e_i::<8>(i));
            assert!(r.is_ok(), "assign {i} failed: {r:?}");
        }
        // Verify all i≠j pairs have interference < 1e-6.
        for i in 0..8u32 {
            for j in 0..8u32 {
                if i == j {
                    continue;
                }
                let inter = p.interference(BranchId::new(i), BranchId::new(j));
                assert!(
                    inter < 1e-6,
                    "Pair ({i},{j}) interference {inter} violates orthogonality"
                );
            }
        }
        // And every branch is non-interfering with all others.
        for i in 0..8u32 {
            assert!(p.is_non_interfering_with_all(BranchId::new(i)));
        }
    }

    #[test]
    fn ninth_direction_in_d8_must_interfere() {
        // Frame-theory: 9 mutually-orthogonal directions in D=8 is impossible.
        // Any 9th must interfere with at least one existing direction by ≥ 1/sqrt(D).
        let mut p: NonInterferenceProjection<8> = NonInterferenceProjection::new(16);
        // Assign 8 standard basis vectors.
        for i in 0..8usize {
            assert!(p
                .assign_direction(BranchId::new(i as u32), &e_i::<8>(i))
                .is_ok());
        }
        // Try to assign a uniform vector (interferes equally with all 8).
        let uniform = vec![1.0f32 / 8.0f32.sqrt(); 8];
        let r = p.assign_direction(BranchId::new(8), &uniform);
        // Interference with each basis vector = 1/sqrt(8) ≈ 0.354 > 0.1 threshold.
        assert_eq!(r.error, Some(AssignError::Interferes));
        assert!((r.interference - (1.0 / 8.0f32.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn assign_out_of_range_branch_is_ok() {
        // Out-of-range branch ids are treated as always-assignable; the caller
        // is responsible for growing the matrix. This keeps the primitive
        // composable with banks that grow dynamically.
        let mut p: NonInterferenceProjection<2> = NonInterferenceProjection::new(4);
        let r = p.assign_direction(BranchId::new(99), &[1.0, 0.0]);
        assert!(r.is_ok());
        // But the assignment was a no-op on the matrix.
        assert_eq!(p.n_assigned(), 0);
    }

    #[test]
    fn assign_result_constructors() {
        let ok = AssignResult::ok();
        assert!(ok.is_ok());
        assert!(ok.error.is_none());

        let wd = AssignResult::wrong_dimension();
        assert_eq!(wd.error, Some(AssignError::WrongDimension));

        let zm = AssignResult::zero_magnitude();
        assert_eq!(zm.error, Some(AssignError::ZeroMagnitude));

        let inter = AssignResult::interferes(BranchId::new(3), 0.42);
        assert_eq!(inter.error, Some(AssignError::Interferes));
        assert_eq!(inter.conflict_branch, Some(BranchId::new(3)));
        assert!((inter.interference - 0.42).abs() < 1e-6);
    }

    #[test]
    fn with_thresholds_custom_epsilon() {
        let p: NonInterferenceProjection<2> =
            NonInterferenceProjection::with_thresholds(4, 0.01, 0.2);
        assert!((p.orthogonal_epsilon() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn is_non_interfering_uses_epsilon() {
        let mut p: NonInterferenceProjection<2> =
            NonInterferenceProjection::with_thresholds(4, 0.01, 0.5);
        // Two directions 5° apart: cos 5° ≈ 0.996. Interference 0.996.
        // With ε=0.01, 0.996 >= 0.01 → NOT non-interfering.
        let d1 = [1.0f32, 0.0];
        let d2 = [0.996, 0.087]; // ~5° rotation
        assert!(p.assign_direction(BranchId::new(0), &d1).is_ok());
        // Manually set second direction to bypass the assign-interference gate.
        p.directions[1] = d2;
        assert!(!p.is_non_interfering(BranchId::new(0), BranchId::new(1)));
    }
}
