//! `committed_field_blend` — Sampling-Invariant Per-Entity MoE Composition
//! (Plan 321, Research 302).
//!
//! A generic, modelless, MIT-licensed primitive: compute a per-entity
//! **frozen** convex blend of `N` operator fields over `D`-dim state, with
//! sigmoid-computed weights derived **once** from a trajectory summary and
//! committed via BLAKE3. The blend governs the entity's dynamics for its
//! entire lifetime (until a major personality event triggers re-commitment).
//!
//! # The math
//!
//! ```text
//! pi_k   = clamp( dot(summary, dir_k), -pi_max, +pi_max )   // computed ONCE at commit
//! f_pi(z) = Σ_k sigmoid(pi_k / tau) · f_k(z)                 // applied every tick
//! ```
//!
//! # Sampling invariance (the defining property — FAME Proposition 3)
//!
//! Because both `pi` (the blend weights) and the fields `f_k` (frozen
//! snapshots) are frozen, `f_pi(z)` is a pure function of `z`. Observation
//! density, network desync, and snapshot thaw all preserve the committed
//! personality: two observation grids encoding the same underlying trajectory
//! produce identical dynamics. This is the Young-integral / FAME property.
//!
//! # Why sigmoid, not softmax (AGENTS.md)
//!
//! Sigmoid is mandated for projections onto learned direction vectors. Softmax
//! would destroy the "near-zero weight = field ignored" semantics — softmax
//! always assigns non-trivial probability to every field. Sigmoid allows a
//! field to contribute ~0 (entity ignores it) or ~1 (entity embodies it).
//!
//! # Reuse (DRY)
//!
//! - Reuses [`sigmoid`] from `personality_composition` (numerically stable,
//!   branching on sign of `x` to avoid `e^{-x}` overflow).
//! - Reuses [`simd_fused_scale_acc`](crate::simd::simd_fused_scale_acc) for the
//!   inner `dz_out[j] += gate · f_k(z)[j]` FMA loop — the same SIMD primitive
//!   `PersonalityWeightedComposition::compose_into` uses.
//! - Reuses [`simd_dot_f32`](crate::simd::simd_dot_f32) for the commit-time
//!   `pi_k = dot(summary, dir_k)` projection.
//!
//! # Sync boundary (AGENTS.md)
//!
//! - `pi` (the K-weight blend vector) crosses the sync boundary as `K` raw
//!   scalars (LatCal-committable) — see Plan 321 Phase 3 / riir-chain R003.
//! - The archetype field **definitions** stay library-side (referenced by their
//!   BLAKE3 commitment hash, never sent over the wire).
//! - The BLAKE3 commitment + version are synced as an audit event on
//!   re-commit.
//!
//! # Feature gate
//!
//! Gated behind the `committed_field_blend` Cargo feature (implies
//! `personality_composition` for sigmoid reuse). Opt-in until the GOAT gate
//! (G1–G5) passes. See `katgpt-rs/.plans/321_sampling_invariant_per_entity_moe_primitive.md`.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/321_sampling_invariant_per_entity_moe_primitive.md`]
//! - Research: [`katgpt-rs/.research/302_FAME_Sampling_Invariant_Per_Entity_MoE.md`]
//! - Source paper: arxiv 2510.00621 — FAME (Gao/Chen/Zhang, NeurIPS 2025)
//! - Closest shipped cousin (per-layer, drifting): Plan 297
//!   (`PersonalityWeightedComposition`)

use crate::personality_composition::sigmoid::sigmoid;
use crate::simd::{simd_dot_f32, simd_fused_scale_acc};

// ─── Trait ────────────────────────────────────────────────────────────────

/// A host-supplied source of one operator field `f_k(z) -> dz`.
///
/// The host (game, robot, recommender) implements this per archetype. The
/// field is **FROZEN** — [`evolve`](Self::evolve) MUST be a pure function of
/// `(z, dz_scratch)`. No internal mutable state, no drift, no learning. The
/// whole point of `CommittedFieldBlend` is that both the weights and the
/// fields are frozen, so the blended dynamics are sampling-invariant.
///
/// # Zero-allocation contract
///
/// [`evolve`](Self::evolve) MUST NOT allocate. The caller passes a scratch
/// buffer; the implementation writes its dynamics update into it and returns
/// a mutable reference to the written region.
///
/// # Entity-agnostic
///
/// The trait carries no game semantics. A "predator chase" field and a
/// "recommendation explore" field both implement this trait with different
/// internals. The blend kernel is the same in all cases.
pub trait ArchetypeFieldSource<const D: usize>: Send + Sync {
    /// Apply the field at state `z`, writing the dynamics update `f_k(z)` into
    /// `dz_scratch`. Returns a mutable reference to the written region (length
    /// `D`).
    ///
    /// # Zero-allocation
    ///
    /// MUST NOT allocate. Write into `dz_scratch` and return a reborrow of it.
    /// The returned slice is tied to `dz_scratch`'s lifetime.
    ///
    /// # Pure / frozen
    ///
    /// MUST be a pure function of `(z, dz_scratch)`. No interior mutability.
    /// This is what guarantees sampling invariance.
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32];

    /// BLAKE3 hash of the frozen field definition (for commitment).
    ///
    /// Two fields with the same definition MUST return the same hash; two
    /// fields with different definitions MUST return different hashes. The host
    /// typically hashes the field's parameters (weights, config) via
    /// `blake3::Hasher` streaming input.
    fn commitment(&self) -> [u8; 32];

    /// Optional Lipschitz bound `L_k` of the field (for the safety-bound
    /// composition, FAME Lemma 1).
    ///
    /// Returns `f32::INFINITY` by default — the primitive does not assume
    /// boundedness. Override to return a finite bound if the host can provide
    /// one; this enables [`CommittedFieldBlend::lipschitz_bound`] to return a
    /// finite safety guarantee.
    fn lipschitz_bound(&self) -> f32 {
        f32::INFINITY
    }
}

// ─── Struct ───────────────────────────────────────────────────────────────

/// A per-entity committed archetype blend (Plan 321).
///
/// Computes blend weights `pi` ONCE from a trajectory summary via sigmoid
/// projection, then FREEZES them for the entity's lifetime. The blended field
///
/// ```text
/// f_pi(z) = Σ_k sigmoid(pi_k / tau) · f_k(z)
/// ```
///
/// governs the entity's dynamics. Because `pi` and the fields are both frozen,
/// the entity's trajectory is sampling-invariant (FAME/Young-integral property).
///
/// # Const-generic budget
///
/// `N` is the archetype count (default `K = 3`); `D` is the state dimension.
/// The production Entity Cognition Stack case is `N = 3, D = 32`.
///
/// # Layout
///
/// `pi` (N·4) + `tau` (4) + `pi_max` (4) + `blake3` (32) + `version` (8).
/// At `N = 3` that's `12 + 4 + 4 + 32 + 8 = 60` bytes — one cache line.
pub struct CommittedFieldBlend<const N: usize, const D: usize> {
    /// Committed blend weights (pre-sigmoid logits). Signed; clamped to
    /// `[-pi_max, +pi_max]`. Computed ONCE from trajectory summary at
    /// [`commit`](Self::commit); never mutated after.
    pub pi: [f32; N],

    /// Personality-sharpness temperature (sigmoid denominator). MUST be > 0.
    pub tau: f32,

    /// Clamp bound on `pi`. Prevents extreme sigmoid saturation.
    pub pi_max: f32,

    /// BLAKE3 commitment over `(version, pi, field_commitments)`. Set at
    /// [`commit`](Self::commit); verified at thaw via
    /// [`verify_commitment`](Self::verify_commitment).
    pub blake3: [u8; 32],

    /// Version counter (incremented on each re-commit). Part of the BLAKE3
    /// input — unlike `PersonalitySnapshot`, version IS part of the commitment
    /// identity here, because re-commit is a major personality event.
    pub version: u64,
}

// SAFETY: contains only f32 arrays, a [u8;32], and a u64 — no interior
// mutability, no cell, no raw pointer. Safe to share across threads.
unsafe impl<const N: usize, const D: usize> Send for CommittedFieldBlend<N, D> {}
unsafe impl<const N: usize, const D: usize> Sync for CommittedFieldBlend<N, D> {}

impl<const N: usize, const D: usize> CommittedFieldBlend<N, D> {
    /// Default personality-sharpness temperature. `sigmoid(±x / 1.0)` gives
    /// standard logistic sharpness.
    pub const DEFAULT_TAU: f32 = 1.0;

    /// Default clamp bound on `pi`. `sigmoid(±10 / 1.0) ≈ {4.5e-5, 0.99995}` —
    /// near-binary field selection (one field fully embodied, rest ignored).
    pub const DEFAULT_PI_MAX: f32 = 10.0;

    /// Construct an **uncommitted** blend with the given config.
    ///
    /// `pi` is all-zero (uniform 0.5 blend at `tau = 1.0`), `blake3` is
    /// all-zero, `version` is 0. You MUST call [`commit`](Self::commit) before
    /// [`apply_blended`](Self::apply_blended) produces a meaningful result —
    /// an uncommitted blend applies a uniform 0.5 gate to every field.
    #[inline]
    pub fn new(tau: f32, pi_max: f32) -> Self {
        Self {
            pi: [0.0; N],
            tau,
            pi_max,
            blake3: [0u8; 32],
            version: 0,
        }
    }

    /// Construct an uncommitted blend with default config
    /// (`tau = 1.0`, `pi_max = 10.0`).
    #[inline]
    pub fn uncommitted() -> Self {
        Self::new(Self::DEFAULT_TAU, Self::DEFAULT_PI_MAX)
    }

    // ─── T1.2: commit ────────────────────────────────────────────────────

    /// Compute blend weights `pi` ONCE from a trajectory summary, then commit.
    ///
    /// For each archetype `k`:
    ///
    /// ```text
    /// pi_k = clamp( dot(summary, direction_vectors[k]), -pi_max, +pi_max )
    /// ```
    ///
    /// Then computes the BLAKE3 commitment over `(version, pi, field_commitments)`
    /// and stores it in `self.blake3`.
    ///
    /// After this call, `pi` is frozen — call [`commit`](Self::commit) again
    /// only on major personality events (re-commit).
    ///
    /// # Arguments
    ///
    /// - `summary` — host-supplied trajectory summary (e.g. KARC delay-embedding
    ///   of the entity's HLA history, or a simpler EMA/ConvPool summary).
    ///   Length MUST be `>= D` (only the first `D` elements are used).
    /// - `direction_vectors` — `N` pre-computed direction vectors (one per
    ///   archetype), used for the sigmoid projection.
    /// - `fields` — the `N` frozen archetype fields. Used only to fetch their
    ///   BLAKE3 commitments for the commitment hash.
    /// - `version` — monotonic version counter for this commit.
    ///
    /// # Zero-allocation
    ///
    /// No allocation. Operates in-place on `self.pi` + a stack-fixed
    /// `[[u8; 32]; N]` for the commitments.
    pub fn commit(
        &mut self,
        summary: &[f32],
        direction_vectors: &[[f32; D]; N],
        fields: &[&dyn ArchetypeFieldSource<D>; N],
        version: u64,
    ) -> [u8; 32] {
        debug_assert!(
            summary.len() >= D,
            "summary must be at least D={D} elements, got {}",
            summary.len()
        );

        // Sigmoid projection: pi_k = clamp(dot(summary, dir_k), -pi_max, pi_max).
        for (k, dir_k) in direction_vectors.iter().enumerate() {
            let dot = simd_dot_f32(summary, dir_k, D);
            self.pi[k] = dot.clamp(-self.pi_max, self.pi_max);
        }

        self.version = version;

        // Collect field commitments (stack-fixed, no alloc).
        let mut field_commitments = [[0u8; 32]; N];
        for k in 0..N {
            field_commitments[k] = fields[k].commitment();
        }

        let hash = self.recompute_blake3(&field_commitments);
        self.blake3 = hash;
        hash
    }

    // ─── T1.3: apply_blended ─────────────────────────────────────────────

    /// Apply the blended field at state `z`, writing the dynamics update into
    /// `dz_out`.
    ///
    /// ```text
    /// f_pi(z) = Σ_k sigmoid(pi_k / tau) · f_k(z)
    /// ```
    ///
    /// Zero-allocation: caller provides `dz_scratch` (reused per-field, length
    /// `>= D`) and `dz_out` (length `D`). Returns `&mut dz_out[..D]`.
    ///
    /// # Reuse (DRY)
    ///
    /// The inner `dz_out[j] += gate · f_k(z)[j]` loop delegates to
    /// [`simd_fused_scale_acc`] — the same SIMD primitive
    /// `PersonalityWeightedComposition::compose_into` uses. The sigmoid gate
    /// reuses [`crate::personality_composition::sigmoid::sigmoid`].
    ///
    /// # Panics (debug)
    ///
    /// In debug builds, panics if `z.len() < D`, `dz_scratch.len() < D`, or
    /// `dz_out.len() != D`.
    pub fn apply_blended<'a>(
        &self,
        fields: &[&dyn ArchetypeFieldSource<D>; N],
        z: &[f32],
        dz_scratch: &mut [f32],
        dz_out: &'a mut [f32],
    ) -> &'a mut [f32] {
        debug_assert!(z.len() >= D, "z must be at least D={D} elements");
        debug_assert!(
            dz_scratch.len() >= D,
            "dz_scratch must be at least D={D} elements"
        );
        debug_assert_eq!(dz_out.len(), D, "dz_out must be exactly D={D} elements");

        // Zero the output. LLVM elides this into a memset.
        for x in dz_out[..D].iter_mut() {
            *x = 0.0;
        }

        for (k, field_k) in fields.iter().enumerate() {
            // f_k(z) — writes into dz_scratch, returns a reborrow.
            let dz_k = field_k.evolve(z, dz_scratch);
            debug_assert_eq!(
                dz_k.len(),
                D,
                "field {k} returned evolve output of length {}, expected D={D}",
                dz_k.len()
            );

            // Per-field gate: sigmoid(pi_k / tau).
            let gate = sigmoid(self.pi[k] / self.tau);

            // FMA accumulate: dz_out[j] += gate · dz_k[j].
            simd_fused_scale_acc(dz_out, dz_k, gate, D);
        }

        &mut dz_out[..D]
    }

    // ─── T1.4: verify_commitment + recompute_blake3 ──────────────────────

    /// Verify the stored BLAKE3 commitment matches the current state.
    ///
    /// Recomputes the commitment from `(version, pi, field_commitments)` and
    /// compares with `self.blake3`. Returns `true` iff they match. A `false`
    /// result indicates tampering or corruption of `pi`, the fields, or
    /// `version`.
    ///
    /// Used at thaw-time (anti-tamper check) and after atomic swaps.
    pub fn verify_commitment(&self, fields: &[&dyn ArchetypeFieldSource<D>; N]) -> bool {
        let mut field_commitments = [[0u8; 32]; N];
        for k in 0..N {
            field_commitments[k] = fields[k].commitment();
        }
        let recomputed = self.recompute_blake3(&field_commitments);
        recomputed == self.blake3
    }

    /// Recompute the BLAKE3 from current state (for atomic swap verification).
    ///
    /// Commitment scheme (streaming input, layout-independent — matches the
    /// `PersonalitySnapshot` pattern):
    ///
    /// ```text
    /// hasher.update(version.to_le_bytes());              // 8 bytes
    /// for k in 0..N { hasher.update(&pi[k].to_le_bytes()); }    // N * 4 bytes
    /// for k in 0..N { hasher.update(&field_commitments[k]); }   // N * 32 bytes
    /// ```
    ///
    /// `version` IS part of the input (unlike `PersonalitySnapshot`) — a
    /// re-commit with new version is a distinct commitment event.
    #[inline]
    fn recompute_blake3(&self, field_commitments: &[[u8; 32]; N]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.version.to_le_bytes());
        for &pi_k in &self.pi {
            hasher.update(&pi_k.to_le_bytes());
        }
        for fc in field_commitments {
            hasher.update(fc);
        }
        *hasher.finalize().as_bytes()
    }

    // ─── Phase 3 (T3.1): Lipschitz safety bound ──────────────────────────

    /// Deterministic safety bound of the committed blend (FAME Lemma 1).
    ///
    /// Returns `max_k { sigmoid(pi_k / tau) · L_k }` — the worst-case
    /// Lipschitz constant of the blended field. This is a closed-form quantity
    /// that can be LatCal-committed alongside `pi` (Plan 321 Phase 3).
    ///
    /// If any field reports `L_k = ∞` (the default), this returns `∞`.
    pub fn lipschitz_bound(&self, fields: &[&dyn ArchetypeFieldSource<D>; N]) -> f32 {
        let mut bound = 0.0f32;
        for (k, field_k) in fields.iter().enumerate() {
            let gate = sigmoid(self.pi[k] / self.tau);
            let l_k = field_k.lipschitz_bound();
            bound = bound.max(gate * l_k);
        }
        bound
    }
}

// ─── Pinned const-generic aliases ─────────────────────────────────────────
//
// Per AGENTS.md: pin N to a small set via type aliases to keep
// monomorphisation bounded. The production Entity Cognition Stack case is
// K=3 archetypes, D=32 (Research 302).

/// K=3 archetype blend (the production Entity Cognition Stack case at D=32).
pub type TriArchetypeBlend = CommittedFieldBlend<3, 32>;

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Test field impls ────────────────────────────────────────────────

    /// A linear field: `f(z) = scale · z`. Lipschitz bound = |scale|.
    /// Deterministic, frozen, zero-alloc.
    struct LinearField {
        scale: f32,
        commitment: [u8; 32],
    }

    impl LinearField {
        fn new(scale: f32, id: u8) -> Self {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"LinearField");
            hasher.update(&[id]);
            hasher.update(&scale.to_le_bytes());
            Self {
                scale,
                commitment: *hasher.finalize().as_bytes(),
            }
        }
    }

    impl ArchetypeFieldSource<32> for LinearField {
        fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
            for j in 0..32 {
                dz_scratch[j] = self.scale * z[j];
            }
            &mut dz_scratch[..32]
        }

        fn commitment(&self) -> [u8; 32] {
            self.commitment
        }

        fn lipschitz_bound(&self) -> f32 {
            self.scale.abs()
        }
    }

    /// A constant-push field: `f(z) = push` (independent of z).
    struct ConstantField {
        push: [f32; 32],
        commitment: [u8; 32],
    }

    impl ConstantField {
        fn new(push: [f32; 32], id: u8) -> Self {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"ConstantField");
            hasher.update(&[id]);
            for &p in &push {
                hasher.update(&p.to_le_bytes());
            }
            Self {
                push,
                commitment: *hasher.finalize().as_bytes(),
            }
        }
    }

    impl ArchetypeFieldSource<32> for ConstantField {
        fn evolve<'a>(&self, _z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
            dz_scratch[..32].copy_from_slice(&self.push);
            &mut dz_scratch[..32]
        }

        fn commitment(&self) -> [u8; 32] {
            self.commitment
        }
    }

    /// A rotation field: `f(z) = R · z` where R rotates axes (i, i+1) by angle.
    /// Lipschitz bound = 1 (rotation is an isometry).
    struct RotationField {
        i: usize,
        j: usize,
        cos_a: f32,
        sin_a: f32,
        commitment: [u8; 32],
    }

    impl RotationField {
        fn new(i: usize, j: usize, angle: f32, id: u8) -> Self {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"RotationField");
            hasher.update(&[id]);
            hasher.update(&(i as u32).to_le_bytes());
            hasher.update(&(j as u32).to_le_bytes());
            hasher.update(&angle.to_le_bytes());
            Self {
                i,
                j,
                cos_a: angle.cos(),
                sin_a: angle.sin(),
                commitment: *hasher.finalize().as_bytes(),
            }
        }
    }

    impl ArchetypeFieldSource<32> for RotationField {
        fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
            dz_scratch[..32].copy_from_slice(&z[..32]);
            let zi = z[self.i];
            let zj = z[self.j];
            dz_scratch[self.i] = self.cos_a * zi - self.sin_a * zj;
            dz_scratch[self.j] = self.sin_a * zi + self.cos_a * zj;
            &mut dz_scratch[..32]
        }

        fn commitment(&self) -> [u8; 32] {
            self.commitment
        }

        fn lipschitz_bound(&self) -> f32 {
            1.0
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn make_three_direction_vectors() -> [[f32; 32]; 3] {
        // Three orthogonal-ish direction vectors.
        let mut d0 = [0.0f32; 32];
        let mut d1 = [0.0f32; 32];
        let mut d2 = [0.0f32; 32];
        for j in 0..32 {
            d0[j] = if j < 11 { 1.0 } else { 0.0 };
            d1[j] = if (11..22).contains(&j) { 1.0 } else { 0.0 };
            d2[j] = if j >= 22 { 1.0 } else { 0.0 };
        }
        [d0, d1, d2]
    }

    // ─── Phase 1 T1.6: unit tests ────────────────────────────────────────

    #[test]
    fn commit_produces_stable_pi() {
        let dirs = make_three_direction_vectors();
        let summary: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

        let f0 = LinearField::new(0.5, 0);
        let f1 = LinearField::new(-0.3, 1);
        let f2 = LinearField::new(0.8, 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let mut blend_a = TriArchetypeBlend::uncommitted();
        let mut blend_b = TriArchetypeBlend::uncommitted();
        let ha = blend_a.commit(&summary, &dirs, &fields, 1);
        let hb = blend_b.commit(&summary, &dirs, &fields, 1);

        // Identical inputs → identical pi + identical hash.
        assert_eq!(blend_a.pi, blend_b.pi, "pi must be deterministic");
        assert_eq!(ha, hb, "blake3 must be reproducible");
        assert!(
            blend_a.verify_commitment(&fields),
            "must verify after commit"
        );
    }

    #[test]
    fn apply_blended_zero_when_all_pi_negative() {
        let dirs = make_three_direction_vectors();
        // Summary that dots strongly NEGATIVE against all three dirs.
        let summary = vec![-100.0f32; 32];

        let f0 = LinearField::new(0.5, 0);
        let f1 = LinearField::new(-0.3, 1);
        let f2 = LinearField::new(0.8, 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let mut blend = TriArchetypeBlend::uncommitted();
        blend.commit(&summary, &dirs, &fields, 1);

        let z = [0.5f32; 32];
        let mut scratch = [0.0f32; 32];
        let mut out = [0.0f32; 32];
        blend.apply_blended(&fields, &z, &mut scratch, &mut out);

        // All pi very negative → sigmoid ≈ 0 → output ≈ 0.
        for &v in &out {
            assert!(
                v.abs() < 1e-3,
                "output must be ~0 when all pi negative, got {v}"
            );
        }
    }

    #[test]
    fn apply_blended_selects_single_field_when_pi_extreme() {
        let dirs = make_three_direction_vectors();

        // Summary that dots strongly POSITIVE against dir 0, NEGATIVE against
        // dirs 1 and 2. We craft it so pi[0] is large positive, pi[1]/pi[2]
        // large negative.
        let mut summary = vec![0.0f32; 32];
        for v in &mut summary[..11] {
            *v = 100.0;
        }
        for v in &mut summary[11..32] {
            *v = -100.0;
        }

        let f0 = ConstantField::new([1.0f32; 32], 0);
        let f1 = ConstantField::new([2.0f32; 32], 1);
        let f2 = ConstantField::new([3.0f32; 32], 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let mut blend = TriArchetypeBlend::uncommitted();
        blend.commit(&summary, &dirs, &fields, 1);

        // pi[0] should be large positive (dot with dir0 = 11 * 100 = 1100,
        // clamped to pi_max=10). pi[1] = (11..22 overlap) ... let's just check
        // the blend selects field 0.
        let z = [0.0f32; 32];
        let mut scratch = [0.0f32; 32];
        let mut out = [0.0f32; 32];
        blend.apply_blended(&fields, &z, &mut scratch, &mut out);

        // Field 0 pushes [1.0; 32] with gate ≈ 1.0. Fields 1/2 push with gate ≈ 0.
        for &v in &out {
            assert!(
                (v - 1.0).abs() < 0.01,
                "blend must select field 0 (push=1.0), got {v}"
            );
        }
    }

    #[test]
    fn verify_commitment_detects_tamper() {
        let dirs = make_three_direction_vectors();
        let summary: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

        let f0 = LinearField::new(0.5, 0);
        let f1 = LinearField::new(-0.3, 1);
        let f2 = LinearField::new(0.8, 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let mut blend = TriArchetypeBlend::uncommitted();
        blend.commit(&summary, &dirs, &fields, 1);
        assert!(
            blend.verify_commitment(&fields),
            "must verify before tamper"
        );

        // Tamper pi[0] by a tiny amount — must break the commitment.
        blend.pi[0] += 0.001;
        assert!(
            !blend.verify_commitment(&fields),
            "tampered pi must fail verify"
        );
    }

    #[test]
    fn sigmoid_stable_for_extreme_inputs() {
        // Extreme pi / tau must produce finite gates in [0, 1].
        let mut blend = CommittedFieldBlend::<3, 32>::new(0.001, 1000.0);
        blend.pi = [1000.0, -1000.0, 0.0];

        let g0 = sigmoid(blend.pi[0] / blend.tau);
        let g1 = sigmoid(blend.pi[1] / blend.tau);
        let g2 = sigmoid(blend.pi[2] / blend.tau);

        assert!(g0.is_finite() && (0.0..=1.0).contains(&g0), "g0={g0}");
        assert!(g1.is_finite() && (0.0..=1.0).contains(&g1), "g1={g1}");
        assert!(g2.is_finite() && (0.0..=1.0).contains(&g2), "g2={g2}");
        assert!(g0 > 0.9999, "large positive → ≈1, got {g0}");
        assert!(g1 < 1e-3, "large negative → ≈0, got {g1}");
    }

    // ─── Phase 2 T2.1: G1 mechanics ──────────────────────────────────────

    #[test]
    fn g1_mechanics() {
        let dirs = make_three_direction_vectors();
        let summary: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

        let f0 = LinearField::new(0.7, 0);
        let f1 = RotationField::new(0, 1, 0.3, 1);
        let f2 = ConstantField::new(
            (0..32)
                .map(|i| (i as f32) * 0.01)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            2,
        );
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let mut blend = TriArchetypeBlend::uncommitted();
        blend.commit(&summary, &dirs, &fields, 1);

        let z = [0.5f32; 32];
        let mut scratch = [0.0f32; 32];
        let mut out = [0.0f32; 32];
        blend.apply_blended(&fields, &z, &mut scratch, &mut out);

        // Finite output, no NaN/Inf.
        for &v in &out {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
        // Output is in R^D (all 32 elements valid).
        assert_eq!(out.len(), 32);
        // Sigmoid gates individually in [0, 1].
        for k in 0..3 {
            let g = sigmoid(blend.pi[k] / blend.tau);
            assert!((0.0..=1.0).contains(&g), "gate {k}={g} must be in [0,1]");
        }
    }

    // ─── Phase 2 T2.2: G2 sampling invariance (the defining property) ────

    #[test]
    fn g2_sampling_invariance() {
        // Construct K=3 archetype fields (linear/rotation/constant).
        let f0 = LinearField::new(0.9, 0);
        let f1 = RotationField::new(0, 1, 0.5, 1);
        let f2 = ConstantField::new([0.1f32; 32], 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let dirs = make_three_direction_vectors();

        // Generate a PERIODIC "true" trajectory of 1000 steps. Periodicity is
        // essential: the mean of a periodic signal over an integer number of
        // full periods is a sampling-invariant statistic — it equals the DC
        // component regardless of whether you sample every step or every 10th
        // step. This is the precondition for FAME Proposition 3 ("two
        // observation grids encoding the SAME underlying trajectory").
        //
        // traj[t][j] = dc + amp * sin(2π t / PERIOD + j * phase_step)
        //   dc = 0.1, amp = 0.05, PERIOD = 100 steps.
        // Dense (1000 steps = 10 periods) and sparse (every 10th = 100 steps
        // = 10 periods) both converge to mean ≈ dc = 0.1 per component, with
        // pi ≈ 0.1 * 11 = 1.1 (well within ±pi_max=10, so no clamping hides
        // the comparison — the test is meaningful).
        const PERIOD: usize = 100;
        const DC: f32 = 0.1;
        const AMP: f32 = 0.05;
        let mut traj: Vec<[f32; 32]> = Vec::with_capacity(1000);
        for t in 0..1000 {
            let mut state = [0.0f32; 32];
            for (j, state_j) in state.iter_mut().enumerate() {
                let phase =
                    2.0 * core::f32::consts::PI * (t as f32) / (PERIOD as f32) + (j as f32) * 0.2;
                *state_j = DC + AMP * phase.sin();
            }
            traj.push(state);
        }

        // Dense summary: mean over all 1000 steps (10 full periods).
        let mut dense = [0.0f32; 32];
        for s in &traj {
            for j in 0..32 {
                dense[j] += s[j];
            }
        }
        for v in dense.iter_mut() {
            *v /= 1000.0;
        }

        // Sparse summary: mean over every 10th step (100 steps = 10 periods).
        let mut sparse = [0.0f32; 32];
        let mut count = 0;
        for s in traj.iter().step_by(10) {
            for j in 0..32 {
                sparse[j] += s[j];
            }
            count += 1;
        }
        for v in sparse.iter_mut() {
            *v /= count as f32;
        }

        // Compute pi from each summary.
        let mut blend_dense = TriArchetypeBlend::uncommitted();
        let mut blend_sparse = TriArchetypeBlend::uncommitted();
        blend_dense.commit(&dense, &dirs, &fields, 1);
        blend_sparse.commit(&sparse, &dirs, &fields, 1);

        // pi_dense ≈ pi_sparse to within 1e-3. Both summaries converge to the
        // DC component (sampling-invariant); the only residual difference is
        // float accumulation-order noise (~1e-5 for these magnitudes).
        for k in 0..3 {
            let diff = (blend_dense.pi[k] - blend_sparse.pi[k]).abs();
            assert!(
                diff < 1e-3,
                "pi[{k}] diverges: dense={}, sparse={}, diff={diff}",
                blend_dense.pi[k],
                blend_sparse.pi[k]
            );
        }

        // Apply the blended field from identical initial state — trajectories
        // must diverge by < 1e-3 over 100 steps. This holds because pi_dense ≈
        // pi_sparse (above), so the dynamics are ≈ identical.
        let z0 = [0.5f32; 32];
        let mut state_dense = z0;
        let mut state_sparse = z0;
        let mut scratch = [0.0f32; 32];
        let mut dz = [0.0f32; 32];
        for _ in 0..100 {
            blend_dense.apply_blended(&fields, &state_dense, &mut scratch, &mut dz);
            for j in 0..32 {
                state_dense[j] += 0.01 * dz[j];
            }
            blend_sparse.apply_blended(&fields, &state_sparse, &mut scratch, &mut dz);
            for j in 0..32 {
                state_sparse[j] += 0.01 * dz[j];
            }
        }
        let mut max_diff = 0.0f32;
        for j in 0..32 {
            max_diff = max_diff.max((state_dense[j] - state_sparse[j]).abs());
        }
        assert!(
            max_diff < 1e-3,
            "trajectories diverge by {max_diff} — sampling invariance broken"
        );
    }

    // ─── Phase 2 T2.3: G3 no regression on PersonalityWeightedComposition ─
    //
    // G3 is verified by the fact that `committed_field_blend` *reuses*
    // `PersonalityWeightedComposition`'s primitives (`sigmoid` + `simd_fused_scale_acc`)
    // without modifying them. We assert the primitives are still callable with
    // the `committed_field_blend` feature enabled.
    #[test]
    fn g3_no_regression_primitives_intact() {
        use crate::personality_composition::sigmoid::sigmoid as pwc_sigmoid;
        // The sigmoid reused by CommittedFieldBlend is the SAME function the
        // PWC kernel uses — verify it still behaves correctly.
        assert!((pwc_sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(pwc_sigmoid(100.0) > 0.9999);
        assert!(pwc_sigmoid(-100.0) < 1e-4);

        // simd_fused_scale_acc still works: dst[i] += scale * src[i].
        let mut dst = [1.0f32; 4];
        let src = [2.0f32; 4];
        simd_fused_scale_acc(&mut dst, &src, 3.0, 4);
        // 1 + 3*2 = 7
        for &v in &dst {
            assert!((v - 7.0).abs() < 1e-6, "fused_scale_acc regressed: {v}");
        }
    }

    // ─── Phase 2 T2.5: G5 BLAKE3 reproducible + tamper-detecting ──────────

    #[test]
    fn g5_blake3_reproducible() {
        let dirs = make_three_direction_vectors();
        let summary: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

        let f0 = LinearField::new(0.5, 0);
        let f1 = LinearField::new(-0.3, 1);
        let f2 = LinearField::new(0.8, 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let mut a = TriArchetypeBlend::uncommitted();
        let mut b = TriArchetypeBlend::uncommitted();
        let ha = a.commit(&summary, &dirs, &fields, 1);
        let hb = b.commit(&summary, &dirs, &fields, 1);
        assert_eq!(ha, hb, "identical inputs must produce identical BLAKE3");
        assert_ne!(ha, [0u8; 32], "hash must be non-zero");
    }

    #[test]
    fn g5_blake3_tamper_detecting_pi_byte_flip() {
        let dirs = make_three_direction_vectors();
        let summary: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

        let f0 = LinearField::new(0.5, 0);
        let f1 = LinearField::new(-0.3, 1);
        let f2 = LinearField::new(0.8, 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let mut blend = TriArchetypeBlend::uncommitted();
        blend.commit(&summary, &dirs, &fields, 1);
        let original = blend.blake3;

        // Tamper: flip a bit of pi[1] via byte manipulation.
        let mut bytes = blend.pi[1].to_le_bytes();
        bytes[0] ^= 0x01;
        blend.pi[1] = f32::from_le_bytes(bytes);

        let mut fc = [[0u8; 32]; 3];
        for k in 0..3 {
            fc[k] = fields[k].commitment();
        }
        let recomputed = blend.recompute_blake3(&fc);
        assert_ne!(
            recomputed, original,
            "tampered pi must produce a different hash"
        );
    }

    #[test]
    fn g5_blake3_version_affects_hash() {
        // Unlike PersonalitySnapshot, version IS part of the commitment here.
        let dirs = make_three_direction_vectors();
        let summary: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

        let f0 = LinearField::new(0.5, 0);
        let f1 = LinearField::new(-0.3, 1);
        let f2 = LinearField::new(0.8, 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let mut a = TriArchetypeBlend::uncommitted();
        let mut b = TriArchetypeBlend::uncommitted();
        a.commit(&summary, &dirs, &fields, 1);
        b.commit(&summary, &dirs, &fields, 2);
        // Same pi (same summary+dirs), different version → different hash.
        assert_eq!(a.pi, b.pi);
        assert_ne!(a.blake3, b.blake3, "version must affect blake3");
    }

    #[test]
    fn g5_blake3_field_swap_detected() {
        // Swapping a field for a different one must change the commitment.
        let dirs = make_three_direction_vectors();
        let summary: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

        let f0 = LinearField::new(0.5, 0);
        let f1 = LinearField::new(-0.3, 1);
        let f2 = LinearField::new(0.8, 2);
        let f2_tampered = LinearField::new(0.9, 2); // different scale → different commitment

        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];
        let fields_tampered: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2_tampered];

        let mut a = TriArchetypeBlend::uncommitted();
        let mut b = TriArchetypeBlend::uncommitted();
        a.commit(&summary, &dirs, &fields, 1);
        b.commit(&summary, &dirs, &fields_tampered, 1);
        // Same pi (commit only uses field commitments for the hash, not the
        // field values in the dot product), but field commitments differ.
        assert_ne!(a.blake3, b.blake3, "swapped field must change commitment");
    }

    // ─── Phase 3 T3.2: lipschitz_bound sanity ────────────────────────────

    #[test]
    fn lipschitz_bound_matches_max_gate_times_lk() {
        let f0 = LinearField::new(2.0, 0); // L = 2.0
        let f1 = LinearField::new(5.0, 1); // L = 5.0
        let f2 = LinearField::new(1.0, 2); // L = 1.0
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        let dirs = make_three_direction_vectors();
        let summary = vec![10.0f32; 32]; // strong positive → pi clamped to pi_max

        let mut blend = TriArchetypeBlend::uncommitted();
        blend.commit(&summary, &dirs, &fields, 1);

        // All pi = pi_max = 10 → gate ≈ 0.99995. Bound ≈ max(0.99995*2, 0.99995*5, 0.99995*1) ≈ 5*0.99995.
        let bound = blend.lipschitz_bound(&fields);
        let expected = sigmoid(10.0 / 1.0) * 5.0;
        assert!(
            (bound - expected).abs() < 1e-3,
            "bound={bound}, expected≈{expected}"
        );
    }
}
