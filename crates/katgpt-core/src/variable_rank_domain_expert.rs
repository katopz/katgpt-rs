//! Variable-Rank Domain Expert Clusters (Plan 558, Research 453).
//!
//! A thin composition layer over [`CommittedFieldBlend<N, D>`](crate::committed_field_blend)
//! that applies LatentMoE's transferable principle to per-NPC cognition.
//!
//! # The principle
//!
//! LatentMoE ([arXiv:2601.18089](https://arxiv.org/abs/2601.18089), NVIDIA 2026-01)
//! proves that for MoE inference, **different tasks have different intrinsic
//! feature ranks**. Compressing the hidden state to the task's rank `ℓ` and
//! scaling expert count by `α = D_full / ℓ` preserves total compute
//! `K × D = K' × ℓ` while boosting archetype diversity via combinatorial
//! sparsity. The paper itself is PASS (training architecture, not modelless
//! math) — this module distills the *transferable principle* into a modelless
//! per-NPC cognition primitive.
//!
//! Applied to per-NPC cognition: movement needs ~8 dims, combat ~16,
//! quest/social ~32. A uniform `CommittedFieldBlend<3, 32>` wastes 24 of 32
//! dimensions on movement decisions. Compressing movement to `ℓ=8` and
//! scaling `K` from 3 to 12 gives 4× more movement archetype diversity at the
//! same `K × D = 96` compute budget.
//!
//! # Plan 230 mitigation — guided projection, not blind JL/PCA
//!
//! Plan 230 (Shard Embedding Projection) tried blind JL/PCA projection to
//! `m=8` and FAILED — it violated the Johnson-Lindenstrauss lower bound by
//! 200× (needs `m ≥ 554` for `ε=0.5, n=100`). This module mitigates that
//! cautionary flag with **guided projection**: we select semantically-relevant
//! dimensions per domain by index (a zero-cost gather), not a random
//! projection matrix. No information loss within the selected subspace; no
//! JL bound to violate.
//!
//! The PoC (Research 453 §4) confirmed guided projection does NOT collapse
//! archetype diversity: per-domain entropy reached 97–99.7% of `log₂(K')`.
//!
//! # Composition
//!
//! - Reuses [`CommittedFieldBlend<N, D>`](crate::committed_field_blend) (Plan 321,
//!   DEFAULT-ON) — the per-domain blend is just `CommittedFieldBlend<K_d, L_d>`
//!   instantiated at the domain's variable rank.
//! - The domain gate ([`pick_domain`]) uses the same dot-projection-onto-
//!   direction-vectors pattern as Plan 309's latent steering.
//! - Future: the host-supplied `domain_directions` can be mined unsupervised
//!   by MAG (Plan 418).
//!
//! # Honest scope caveat
//!
//! This is a bandwidth/diversity optimization (Q2 in Research 453 novelty gate
//! was conditional), not a new capability class. NPCs do the same things,
//! just more efficiently. The Super-GOAT tier (requires Q2 = YES) is out of
//! reach; this targets GOAT only.
//!
//! # Status
//!
//! Opt-in — Plan 558 GOAT gate pending. Promotion to default-on requires
//! release-mode latency ≤1.0× baseline at 10K NPCs.

use crate::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};

// ────────────────────────────────────────────────────────────────────────────
// Primitive 1: Domain Gate — deterministic dot-product routing
// ────────────────────────────────────────────────────────────────────────────

/// Pick ONE domain from `N` candidates by `argmax(activity · domain_directions)`.
///
/// Modelless (no learned router, no softmax — pure argmax on host-supplied
/// activity vector × host-supplied direction matrix). Ties are broken by
/// lowest index (deterministic).
///
/// Sibling to `latent_steering::LatentSteeringVector` (Plan 309) — both use
/// dot-projection onto pre-computed direction vectors. The gate just routes;
/// the steering vector adjusts state.
///
/// # Const generics
///
/// - `N`: number of domain candidates (e.g. 3 for move/combat/quest).
/// - `A`: activity vector dimension.
///
/// # Zero allocation
///
/// Scores are kept on the stack in a fixed `[f32; N]` array.
#[inline]
pub fn pick_domain<const N: usize, const A: usize>(
    activity: &[f32; A],
    domain_directions: &[[f32; A]; N],
) -> usize {
    let mut scores = [0.0f32; N];
    for (d, dir) in domain_directions.iter().enumerate() {
        let mut s = 0.0f32;
        for i in 0..A {
            s += activity[i] * dir[i];
        }
        scores[d] = s;
    }
    // argmax with deterministic tie-break (lowest index wins)
    let (best, _) = (0..N)
        .map(|d| (d, scores[d]))
        .max_by(|(d1, s1), (d2, s2)| {
            // Strict > so ties pick the lower index (first one wins on equal).
            s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal)
                .then(d2.cmp(d1)) // lower d wins on tie → reverse the d comparison
        })
        .unwrap_or((0, scores[0]));
    best
}

/// Pick the top `k` domains by gate affinity, in DESCENDING score order.
///
/// The `k`-selection generalisation of [`pick_domain`]. The difference that
/// matters is not the return shape but the arithmetic: a caller that wants
/// `k` domains and only has `pick_domain` must call it `k` times, and
/// `pick_domain` recomputes **all `N` dot products on every call** — so a
/// top-`k` selection costs `k·N·A` multiply-adds plus one masking write per
/// pick. This scores each domain **once** (`N·A`) and then runs `k` argmax
/// passes over `N` scalars.
///
/// Measured on the first consumer (`riir-games` strategy-MoE family gate,
/// `N = 5`, `A = 8`, riir-ai Bench 861): the repeated-`pick_domain` shape
/// costs 11.7 ns at k=1 rising to 73.6 ns at k=5 — superlinear in `k`
/// because each pick re-does the whole score vector.
///
/// # Semantics — identical to `pick_domain` where they overlap
///
/// - Ties break to the **lower index**, so `k = 1` with
///   `min_score = f32::NEG_INFINITY` selects exactly what `pick_domain`
///   selects.
/// - `min_score` is a **strict** floor: a domain enters the selection only
///   if its affinity is `> min_score`. Pass `f32::NEG_INFINITY` for pure
///   argmax semantics (`pick_domain`'s), or `0.0` for the positive-affinity
///   floor a sparse gate wants — which also makes zero-padded rows in an
///   over-sized direction matrix drop out on their own, with no separate
///   padding check.
/// - A domain scoring exactly `f32::NEG_INFINITY` is never selected. This is
///   the one divergence from `pick_domain`, which would return it as the
///   argmax of an all-`-inf` matrix; such a matrix has no meaningful winner.
///
/// Returns the number of `(domain_index, score)` pairs written to `out`,
/// which is `min(k, N)` unless the floor cuts the selection short.
///
/// # Modelless
///
/// Dot products + argmax. No softmax, no learned router, no allocation —
/// scores live in a fixed `[f32; N]` on the stack, exactly as in
/// [`pick_domain`].
///
/// # Example
///
/// ```
/// use katgpt_core::variable_rank_domain_expert::pick_domains_top_k;
///
/// // Three domains; the activity vector leans on dim 0, then dim 1.
/// let dirs = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]];
/// let activity = [0.9, 0.4];
/// let mut out = [(0usize, 0.0f32); 3];
///
/// // Positive-affinity floor: the third domain is negative and never enters.
/// let n = pick_domains_top_k(&activity, &dirs, 3, 0.0, &mut out);
/// assert_eq!(n, 2);
/// assert_eq!(out[0].0, 0); // 0.9 beats 0.4
/// assert_eq!(out[1].0, 1);
/// ```
#[inline]
pub fn pick_domains_top_k<const N: usize, const A: usize>(
    activity: &[f32; A],
    domain_directions: &[[f32; A]; N],
    k: usize,
    min_score: f32,
    out: &mut [(usize, f32); N],
) -> usize {
    // Score every domain ONCE — this is the whole point of the function.
    let mut scores = [0.0f32; N];
    for (d, dir) in domain_directions.iter().enumerate() {
        let mut s = 0.0f32;
        for i in 0..A {
            s += activity[i] * dir[i];
        }
        scores[d] = s;
    }

    let k = k.min(N);
    let mut count = 0usize;
    while count < k {
        // Strict `>` against the running best gives lowest-index-wins ties,
        // matching `pick_domain`; seeding it with `min_score` folds the
        // floor into the same comparison instead of a second pass.
        let mut best = usize::MAX;
        let mut best_score = min_score;
        for (d, &score) in scores.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best = d;
            }
        }
        if best == usize::MAX {
            break;
        }
        out[count] = (best, best_score);
        scores[best] = f32::NEG_INFINITY; // mask: never wins again
        count += 1;
    }
    count
}

// ────────────────────────────────────────────────────────────────────────────
// Primitive 2: Guided Projection — zero-cost dimension gather
// ────────────────────────────────────────────────────────────────────────────

/// Select the semantically-relevant `L` dimensions from a `D`-dim state into
/// an `L`-dim latent. This is the Plan 230 mitigation: instead of a random
/// JL/PCA projection matrix, we select known-relevant dimensions by index.
///
/// # Contract
///
/// `indices` MUST be sorted ascending + unique. This is enforced by
/// `debug_assert!` (elided in release builds for zero cost). Production code
/// supplies compile-time-known index arrays (e.g. `MOVE_DIMS = [0,1,2,3,4,5,6,7]`).
///
/// # Zero allocation
///
/// Pure gather — `L` indexed loads from `z_full` into `z_out`.
///
/// # Const generics
///
/// - `D`: full state dimension.
/// - `L`: projected latent dimension (the domain's intrinsic rank).
#[inline]
pub fn project_guided<const D: usize, const L: usize>(
    z_full: &[f32; D],
    indices: &[usize; L],
    z_out: &mut [f32; L],
) {
    debug_assert!(L == 0 || indices[0] < D, "projection index out of bounds");
    if L > 0 {
        debug_assert!(indices[L - 1] < D, "projection index out of bounds");
    }
    #[cfg(debug_assertions)]
    {
        for i in 1..L {
            debug_assert!(
                indices[i] > indices[i - 1],
                "projection indices must be strictly ascending + unique"
            );
        }
    }
    for i in 0..L {
        z_out[i] = z_full[indices[i]];
    }
}

/// Inverse of [`project_guided`]: scatter an `L`-dim update back into a `D`-dim
/// output. Dims not in `indices` are NOT zeroed by this call — the caller is
/// responsible for zeroing `dz_out_full` before invoking if zeroed non-domain
/// dims are desired (the typical pattern).
#[inline]
pub fn scatter_guided<const D: usize, const L: usize>(
    dz_proj: &[f32; L],
    indices: &[usize; L],
    dz_out_full: &mut [f32; D],
) {
    debug_assert!(L == 0 || indices[0] < D, "scatter index out of bounds");
    if L > 0 {
        debug_assert!(indices[L - 1] < D, "scatter index out of bounds");
    }
    for i in 0..L {
        dz_out_full[indices[i]] = dz_proj[i];
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Primitive 3: Variable-Rank Router — heterogeneous-rank dispatch
// ────────────────────────────────────────────────────────────────────────────

/// Result of a single [`VariableRankRouter::tick`] call. Small `Copy` struct
/// for caller introspection (entropy bookkeeping, debug overlays).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingVerdict {
    /// Which domain was selected by the gate (0..DOMAINS).
    pub domain: usize,
    /// The winning archetype within that domain's cluster (0..K_d).
    pub winner: usize,
}

/// Object-safe wrapper around a per-domain [`CommittedFieldBlend<K, L>`] +
/// its frozen archetype fields. Required because heterogeneous const-generics
/// (`K_d, L_d` differ per domain) cannot be stored in a single array without
/// type erasure.
///
/// The trait-object dispatch is one virtual call per NPC per tick — negligible
/// vs the blend work (`K × L` multiply-adds). If Plan 558 G2 perf gate fails
/// because of this, the escape hatch is macro-generated per-domain-count
/// monomorphization (documented in Plan 558 §Honest Risks).
pub trait ErasedCluster: Send + Sync {
    /// Apply the blended field at projected state `z_proj` (length = this
    /// cluster's `L`), writing the dynamics update into `dz_out` (same length).
    /// Returns the winning archetype index for caller entropy bookkeeping.
    ///
    /// The caller-supplied scratch buffer must be at least `L` elements.
    fn apply_blended(&self, z_proj: &[f32], scratch: &mut [f32], dz_out: &mut [f32]) -> usize;

    /// BLAKE3 commitment of the underlying blend (anti-tamper).
    fn commitment(&self) -> [u8; 32];

    /// This cluster's latent rank `L` (the projection dimension).
    fn latent_dim(&self) -> usize;

    /// This cluster's expert count `K`.
    fn expert_count(&self) -> usize;

    /// Override the committed pi weights for this tick. The host calls this
    /// when simulating per-entity committed personalities (each NPC has its
    /// own pi vector; the router dispatches to the cluster, the cluster uses
    /// the host-supplied pi for that entity).
    ///
    /// `pi` length must equal `expert_count()`. Trailing entries are ignored
    /// if shorter; panic (debug only) if longer than the cluster's K.
    fn override_pi(&mut self, pi: &[f32]);
}

/// Concrete `ErasedCluster` holder wrapping a `CommittedFieldBlend<K, L>` +
/// its `K` frozen archetype fields.
///
/// The archetype fields are stored as trait objects (`[Box<dyn ArchetypeFieldSource<L>>; K]`)
/// so the holder can own heterogeneous field implementations. Host code
/// constructs these once at startup.
pub struct ClusterHolder<const K: usize, const L: usize> {
    blend: CommittedFieldBlend<K, L>,
    fields: [Box<dyn ArchetypeFieldSource<L>>; K],
}

impl<const K: usize, const L: usize> ClusterHolder<K, L> {
    /// This cluster's latent rank `L` (the projection dimension).
    pub const LATENT_DIM: usize = L;

    /// This cluster's expert count `K`.
    pub const EXPERT_COUNT: usize = K;

    /// Construct from an owned blend + boxed archetype fields.
    pub fn new(
        blend: CommittedFieldBlend<K, L>,
        fields: [Box<dyn ArchetypeFieldSource<L>>; K],
    ) -> Self {
        Self { blend, fields }
    }

    /// Mutable access to the underlying blend (for commit/recommit).
    pub fn blend_mut(&mut self) -> &mut CommittedFieldBlend<K, L> {
        &mut self.blend
    }

    /// Apply the blended field at projected state `z_proj` (length `L`),
    /// writing the dynamics update into `dz_out` (same length). Returns the
    /// winning archetype index for caller entropy bookkeeping.
    ///
    /// **Inherent method** — callable without trait-object dispatch. This is
    /// the zero-vtable path used by the [`variable_rank_router_static!`] macro
    /// router. Same logic as [`ErasedCluster::apply_blended`].
    ///
    /// The caller-supplied scratch buffer must be at least `L` elements.
    #[inline]
    pub fn apply_direct(&self, z_proj: &[f32], scratch: &mut [f32], dz_out: &mut [f32]) -> usize {
        debug_assert!(z_proj.len() >= L, "z_proj too short: {} < {}", z_proj.len(), L);
        debug_assert!(scratch.len() >= L, "scratch too short: {} < {}", scratch.len(), L);
        debug_assert!(dz_out.len() >= L, "dz_out too short: {} < {}", dz_out.len(), L);
        let fields_ref: [&dyn ArchetypeFieldSource<L>; K] =
            std::array::from_fn(|i| self.fields[i].as_ref());
        let z_slice = &z_proj[..L];
        let scratch_slice = &mut scratch[..L];
        let dz_slice = &mut dz_out[..L];
        self.blend
            .apply_blended(&fields_ref, z_slice, scratch_slice, dz_slice);

        // Winning archetype = highest pi (sigmoid monotonicity).
        let mut winner = 0usize;
        let mut best = self.blend.pi[0];
        for k in 1..K {
            if self.blend.pi[k] > best {
                best = self.blend.pi[k];
                winner = k;
            }
        }
        winner
    }

    /// Override the committed pi weights for this tick. **Inherent method** —
    /// zero-vtable path used by the [`variable_rank_router_static!`] macro router.
    /// Same logic as [`ErasedCluster::override_pi`].
    ///
    /// `pi` length must equal `EXPERT_COUNT` (`K`).
    #[inline]
    pub fn override_pi_direct(&mut self, pi: &[f32]) {
        debug_assert!(pi.len() >= K, "override_pi slice too short: {} < {}", pi.len(), K);
        self.blend.pi[..K].copy_from_slice(&pi[..K]);
    }
}

impl<const K: usize, const L: usize> ErasedCluster for ClusterHolder<K, L> {
    fn apply_blended(&self, z_proj: &[f32], scratch: &mut [f32], dz_out: &mut [f32]) -> usize {
        // DRY: delegate to the inherent method (Issue 189 T2).
        self.apply_direct(z_proj, scratch, dz_out)
    }

    fn commitment(&self) -> [u8; 32] {
        self.blend.blake3
    }

    fn latent_dim(&self) -> usize {
        L
    }

    fn expert_count(&self) -> usize {
        K
    }

    fn override_pi(&mut self, pi: &[f32]) {
        // DRY: delegate to the inherent method (Issue 189 T2).
        self.override_pi_direct(pi);
    }
}

/// A variable-rank router owning one [`ErasedCluster`] per domain + the
/// per-domain projection indices + the domain gate directions.
///
/// Generic over:
/// - `DOMAINS`: number of domains (e.g. 3 for move/combat/quest).
/// - `D_FULL`: full state dimension (e.g. 32 — the HLA state size).
/// - `A`: activity vector dimension (the gate's input).
///
/// Per NPC per tick, [`tick`](Self::tick):
/// 1. Picks a domain via [`pick_domain`].
/// 2. Projects the full state to that domain's rank via [`project_guided`].
/// 3. Applies the domain's blend via `ErasedCluster::apply_blended`.
/// 4. Scatters the `L`-dim update back to the full `D` dims via [`scatter_guided`]
///    (dims not in the projection mask retain their caller-set value —
///    callers typically zero `dz_out_full` first).
pub struct VariableRankRouter<const DOMAINS: usize, const D_FULL: usize, const A: usize> {
    /// One type-erased cluster per domain.
    clusters: [Box<dyn ErasedCluster>; DOMAINS],
    /// Per-domain projection indices into the full `D_FULL` state.
    projection_indices: [Vec<usize>; DOMAINS],
    /// Domain gate direction matrix: `domain_directions[d]` scores domain `d`.
    domain_directions: [[f32; A]; DOMAINS],
}

impl<const DOMAINS: usize, const D_FULL: usize, const A: usize>
    VariableRankRouter<DOMAINS, D_FULL, A>
{
    /// Construct from owned clusters + per-domain projection indices + gate
    /// directions.
    ///
    /// # Panics (debug only)
    ///
    /// Debug-asserts that each domain's projection indices are in bounds
    /// (`< D_FULL`) and strictly ascending. Release builds skip the check.
    pub fn new(
        clusters: [Box<dyn ErasedCluster>; DOMAINS],
        projection_indices: [Vec<usize>; DOMAINS],
        domain_directions: [[f32; A]; DOMAINS],
    ) -> Self {
        for (d, idx) in projection_indices.iter().enumerate() {
            debug_assert!(
                idx.len() == clusters[d].latent_dim(),
                "domain {} projection indices length {} != cluster latent_dim {}",
                d,
                idx.len(),
                clusters[d].latent_dim()
            );
            #[cfg(debug_assertions)]
            {
                for i in 1..idx.len() {
                    debug_assert!(
                        idx[i] > idx[i - 1],
                        "domain {d} projection indices must be strictly ascending"
                    );
                }
                if let Some(&last) = idx.last() {
                    debug_assert!(last < D_FULL, "domain {d} index {last} >= D_FULL {D_FULL}");
                }
            }
        }
        Self {
            clusters,
            projection_indices,
            domain_directions,
        }
    }

    /// Route + apply for one NPC this tick.
    ///
    /// - `z_full`: the NPC's full `D_FULL` state.
    /// - `activity`: the NPC's `A`-dim activity vector.
    /// - `scratch_full`: caller-provided scratch, at least `D_FULL` elements
    ///   (used for the projected latent + blend scratch).
    /// - `dz_out_full`: the output dynamics update, length `D_FULL`. Caller
    ///   should zero this first if zeroed non-domain dims are desired.
    ///
    /// Returns the [`RoutingVerdict`] (which domain won + which archetype).
    ///
    /// # Zero allocation
    ///
    /// All work happens in the caller-supplied `scratch_full` + `dz_out_full`.
    /// The router does NOT allocate in the hot path.
    pub fn tick(
        &self,
        z_full: &[f32; D_FULL],
        activity: &[f32; A],
        scratch_full: &mut [f32],
        dz_out_full: &mut [f32; D_FULL],
    ) -> RoutingVerdict {
        let domain = pick_domain::<DOMAINS, A>(activity, &self.domain_directions);
        let cluster = &self.clusters[domain];
        let l = cluster.latent_dim();

        // Project full state to domain rank, then split scratch into 3
        // non-overlapping regions (z_proj, blend_scratch, dz_proj). We use
        // chained split_at_mut because `l` is a runtime value — the borrow
        // checker can't prove non-overlap from range arithmetic alone.
        let idx: &[usize] = &self.projection_indices[domain];
        let (z_proj_region, rest) = scratch_full.split_at_mut(l);
        let (blend_scratch_region, dz_proj_region) = rest.split_at_mut(l);
        let z_proj = &mut z_proj_region[..l];
        let blend_scratch = &mut blend_scratch_region[..l];
        let dz_proj = &mut dz_proj_region[..l];
        for i in 0..l {
            z_proj[i] = z_full[idx[i]];
        }

        // Apply blend at variable rank.
        let winner = cluster.apply_blended(z_proj, blend_scratch, dz_proj);

        // Scatter back to full D. Caller is responsible for zeroing dz_out_full
        // before this call if zeroed non-domain dims are desired.
        for i in 0..l {
            dz_out_full[idx[i]] = dz_proj[i];
        }

        RoutingVerdict { domain, winner }
    }

    /// Borrow the cluster at domain `d` mutably (for commit/recommit / pi override).
    pub fn cluster_mut(&mut self, domain: usize) -> &mut dyn ErasedCluster {
        self.clusters[domain].as_mut()
    }

    /// Borrow the projection indices for domain `d`.
    pub fn projection_indices(&self, domain: usize) -> &[usize] {
        &self.projection_indices[domain]
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Monomorphization escape hatch — the `variable_rank_router_static!` macro
// (Issue 189 T2). Generates a router struct with typed cluster fields +
// match-based dispatch (no `Box<dyn>`, no vtable). The ergonomic dynamic
// `VariableRankRouter` stays for consumers who need runtime domain-count
// flexibility; this macro is the zero-vtable fast path.
// ────────────────────────────────────────────────────────────────────────────

/// Generate a variable-rank router struct with **monomorphized dispatch** —
/// no `Box<dyn ErasedCluster>`, no vtable. Each domain becomes a typed
/// `ClusterHolder<K, L>` field; the gate dispatches via a `match` on the
/// domain index (CPU-predictable jump table, not an indirect call).
///
/// This is the Issue 189 T2 escape hatch for eliminating the ~50 ns vtable
/// tax that makes the dynamic [`VariableRankRouter`] 2× slower than the
/// uniform baseline. The trade-off: the domain count is fixed at compile
/// time (each macro invocation produces a concrete struct).
///
/// # Syntax
///
/// ```text
/// variable_rank_router_static! {
///     $(#[doc = $struct_doc])*
///     $vis:vis struct $name:ident
///         < $domains:literal, $d_full:literal, $a:literal >;
///
///     $idx:literal => $field:ident : $cluster_ty:ty => $indices:expr;
///     ...
/// }
/// ```
///
/// - `$domains`: domain count (e.g. `3` for move/combat/quest).
/// - `$d_full`: full state dimension (e.g. `32`).
/// - `$a`: activity vector dimension (the gate's input).
/// - Per domain: explicit `0..N` index, field name, cluster type
///   (`ClusterHolder<K, L>`), and projection indices array.
///
/// The macro generates: the struct, a `new()` constructor, a `tick()`
/// method (same API as [`VariableRankRouter::tick`]), and an
/// `override_cluster_pi()` method (zero-vtable pi override).
///
/// # Example
///
/// ```
/// use katgpt_core::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
/// use katgpt_core::variable_rank_domain_expert::ClusterHolder;
/// use katgpt_core::variable_rank_router_static;
///
/// variable_rank_router_static! {
///     /// 2-domain test router: move (L=4) + combat (L=2).
///     pub struct TestRouter<2, 4, 2>;
///
///     0 => move_cluster:   ClusterHolder<4, 4> => [0, 1, 2, 3];
///     1 => combat_cluster: ClusterHolder<2, 2> => [0, 1];
/// }
/// ```
///
/// # Contract
///
/// Domain indices MUST be `0..N` contiguous (debug_asserted). Projection
/// indices must be in bounds (`< D_FULL`) and match the cluster's `L`.
#[macro_export]
macro_rules! variable_rank_router_static {
    (
        $(#[doc = $struct_doc:literal])*
        $vis:vis struct $name:ident
        < $domains:literal, $d_full:literal, $a:literal >;

        $( $idx:literal => $field:ident : $cluster_ty:ty => $indices:expr );+ $(;)?
    ) => {
        $(#[doc = $struct_doc])*
        $vis struct $name {
            $( $field: $cluster_ty, )+
            domain_directions: [[f32; $a]; $domains],
        }

        impl $name {
            /// Construct from owned clusters + domain gate directions.
            ///
            /// # Panics (debug only)
            ///
            /// Debug-asserts that domain indices are `0..N` contiguous.
            pub fn new(
                $( $field: $cluster_ty, )+
                domain_directions: [[f32; $a]; $domains],
            ) -> Self {
                #[cfg(debug_assertions)]
                {
                    let mut expected: usize = 0;
                    $(
                        debug_assert_eq!($idx, expected, "domain index must be 0..N contiguous");
                        expected += 1;
                    )+
                    debug_assert_eq!(expected, $domains, "domain count must match <DOMAINS>");
                }
                Self {
                    $( $field, )+
                    domain_directions,
                }
            }

            /// Override committed pi for domain `d`. Direct field access —
            /// **zero vtable dispatch** (unlike [`$crate::variable_rank_domain_expert::VariableRankRouter::cluster_mut`]
            /// which goes through `&mut dyn ErasedCluster`).
            #[inline]
            pub fn override_cluster_pi(&mut self, domain: usize, pi: &[f32]) {
                match domain {
                    $( $idx => self.$field.override_pi_direct(pi), )+
                    _ => unreachable!("domain {} out of range 0..{}", domain, $domains),
                }
            }

            /// Route + apply for one NPC this tick. **Zero vtable dispatch** —
            /// all cluster calls are monomorphized inherent method calls.
            ///
            /// Same API contract as
            /// [`VariableRankRouter::tick`].
            pub fn tick(
                &self,
                z_full: &[f32; $d_full],
                activity: &[f32; $a],
                scratch_full: &mut [f32],
                dz_out_full: &mut [f32; $d_full],
            ) -> $crate::variable_rank_domain_expert::RoutingVerdict {
                let domain = $crate::variable_rank_domain_expert::pick_domain::<$domains, $a>(
                    activity,
                    &self.domain_directions,
                );
                let winner = match domain {
                    $( $idx => {
                        let cluster = &self.$field;
                        let proj: &[usize] = &$indices;
                        let l = proj.len();
                        let (z_proj_region, rest) = scratch_full.split_at_mut(l);
                        let (blend_scratch_region, dz_proj_region) = rest.split_at_mut(l);
                        let z_proj = &mut z_proj_region[..l];
                        let blend_scratch = &mut blend_scratch_region[..l];
                        let dz_proj = &mut dz_proj_region[..l];
                        for i in 0..l {
                            z_proj[i] = z_full[proj[i]];
                        }
                        let w = cluster.apply_direct(z_proj, blend_scratch, dz_proj);
                        for i in 0..l {
                            dz_out_full[proj[i]] = dz_proj[i];
                        }
                        w
                    } )+
                    _ => unreachable!("domain {} out of range 0..{}", domain, $domains),
                };
                $crate::variable_rank_domain_expert::RoutingVerdict { domain, winner }
            }
        }
    };
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── G1-mechanics: pick_domain ───────────────────────────────────────────

    // ─── pick_domains_top_k: EQUIVALENCE to the repeated-pick_domain shape ───

    /// The shape `pick_domains_top_k` replaces: call `pick_domain`, record it,
    /// zero the winning row so it cannot win again, repeat. Kept here as the
    /// reference implementation so the equivalence claim is checked against
    /// real code rather than against prose.
    fn reference_repeated_pick_domain<const N: usize, const A: usize>(
        activity: &[f32; A],
        directions: &[[f32; A]; N],
        k: usize,
        live: usize,
    ) -> Vec<(usize, f32)> {
        let mut work = *directions;
        let mut picked: Vec<usize> = Vec::new();
        let mut out: Vec<(usize, f32)> = Vec::new();
        for _ in 0..k.min(N) {
            let idx = pick_domain::<N, A>(activity, &work);
            if idx >= live || picked.contains(&idx) {
                break;
            }
            let mut score = 0.0f32;
            for i in 0..A {
                score += activity[i] * work[idx][i];
            }
            if score <= 0.0 {
                break;
            }
            picked.push(idx);
            out.push((idx, score));
            work[idx] = [0.0; A];
        }
        out
    }

    #[test]
    fn top_k_reproduces_repeated_pick_domain_over_a_swept_corpus() {
        // 8 rows with 5 "live" families + 3 zero-padded ones — the exact
        // shape riir-games' strategy-MoE gate uses, including the padding
        // rows whose separate `idx >= live` check the floor now subsumes.
        const N: usize = 8;
        const A: usize = 8;
        const LIVE: usize = 5;

        let mut checked = 0usize;
        for seed in 0..64u32 {
            let mut directions = [[0.0f32; A]; N];
            for (r, row) in directions.iter_mut().enumerate().take(LIVE) {
                for (i, v) in row.iter_mut().enumerate() {
                    // Deterministic spread including NEGATIVE directions, so
                    // the gate floor is exercised in both directions.
                    let h = (seed as usize * 31 + r * 7 + i * 3) % 13;
                    *v = (h as f32 - 6.0) / 6.0;
                }
            }
            for a_seed in 0..8u32 {
                let mut activity = [0.0f32; A];
                for (i, v) in activity.iter_mut().enumerate() {
                    let h = (a_seed as usize * 17 + i * 5) % 11;
                    *v = (h as f32 - 5.0) / 5.0;
                }
                for k in 0..=N {
                    let want = reference_repeated_pick_domain::<N, A>(
                        &activity,
                        &directions,
                        k,
                        LIVE,
                    );
                    let mut got = [(0usize, 0.0f32); N];
                    let n = pick_domains_top_k::<N, A>(
                        &activity,
                        &directions,
                        k,
                        0.0,
                        &mut got,
                    );
                    assert_eq!(
                        n,
                        want.len(),
                        "count diverged at seed={seed} a_seed={a_seed} k={k}"
                    );
                    for (i, w) in want.iter().enumerate() {
                        assert_eq!(
                            got[i].0, w.0,
                            "index diverged at seed={seed} a_seed={a_seed} k={k} slot={i}"
                        );
                        assert_eq!(
                            got[i].1.to_bits(),
                            w.1.to_bits(),
                            "score not BIT-identical at seed={seed} a_seed={a_seed} k={k} slot={i}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        // Non-vacuity: the sweep must actually have run, and must have
        // produced at least one non-empty selection (an all-negative corpus
        // would make every comparison trivially `0 == 0`).
        assert_eq!(checked, 64 * 8 * (N + 1), "sweep did not cover the corpus");
        let mut any_selected = false;
        for seed in 0..64u32 {
            let mut directions = [[0.0f32; A]; N];
            for (r, row) in directions.iter_mut().enumerate().take(LIVE) {
                for (i, v) in row.iter_mut().enumerate() {
                    let h = (seed as usize * 31 + r * 7 + i * 3) % 13;
                    *v = (h as f32 - 6.0) / 6.0;
                }
            }
            let activity = [0.4f32; A];
            let mut got = [(0usize, 0.0f32); N];
            if pick_domains_top_k::<N, A>(&activity, &directions, 5, 0.0, &mut got) > 0 {
                any_selected = true;
                break;
            }
        }
        assert!(any_selected, "corpus never selected anything — the equivalence would be vacuous");
    }

    #[test]
    fn top_k_at_k1_no_floor_matches_pick_domain() {
        // With the floor disabled, k=1 IS pick_domain — including on a
        // matrix whose every score is negative, where a positive floor
        // would (correctly) select nothing.
        let activity = [0.5f32, 0.25];
        let directions: [[f32; 2]; 3] = [[-1.0, -1.0], [-0.5, -2.0], [-2.0, -0.1]];
        let want = pick_domain::<3, 2>(&activity, &directions);
        let mut out = [(0usize, 0.0f32); 3];
        let n = pick_domains_top_k::<3, 2>(
            &activity,
            &directions,
            1,
            f32::NEG_INFINITY,
            &mut out,
        );
        assert_eq!(n, 1);
        assert_eq!(out[0].0, want);

        let m = pick_domains_top_k::<3, 2>(&activity, &directions, 1, 0.0, &mut out);
        assert_eq!(m, 0, "the positive floor admits nothing from an all-negative matrix");
    }

    #[test]
    fn top_k_orders_descending_and_breaks_ties_by_lowest_index() {
        let activity = [1.0f32, 1.0];
        // Three domains tie at 1.0; one scores 2.0.
        let directions: [[f32; 2]; 4] =
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let mut out = [(0usize, 0.0f32); 4];
        let n = pick_domains_top_k::<4, 2>(&activity, &directions, 4, 0.0, &mut out);
        assert_eq!(n, 4);
        assert_eq!(out[0], (2, 2.0), "highest first");
        assert_eq!(out[1].0, 0, "1.0 tie → lowest index first");
        assert_eq!(out[2].0, 1);
        assert_eq!(out[3], (3, 1.0));
        for w in out.windows(2) {
            assert!(w[0].1 >= w[1].1, "scores must be descending");
        }
    }

    #[test]
    fn top_k_k_zero_and_k_over_n_are_clamped() {
        let activity = [1.0f32, 1.0];
        let directions: [[f32; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
        let mut out = [(0usize, 0.0f32); 2];
        assert_eq!(
            pick_domains_top_k::<2, 2>(&activity, &directions, 0, 0.0, &mut out),
            0
        );
        assert_eq!(
            pick_domains_top_k::<2, 2>(&activity, &directions, 99, 0.0, &mut out),
            2,
            "k is clamped to N, never reads past `out`"
        );
    }


    #[test]
    fn g1_pick_domain_argmax_picks_expected_winner() {
        // 3 domains, A=3. Domain 1 should win (highest dot product).
        let activity = [0.8, 0.1, 0.1];
        let directions: [[f32; 3]; 3] = [
            [0.1, 0.9, 0.9], // dot = 0.08 + 0.09 + 0.09 = 0.26
            [0.9, 0.1, 0.1], // dot = 0.72 + 0.01 + 0.01 = 0.74  ← winner
            [0.3, 0.3, 0.3], // dot = 0.24 + 0.03 + 0.03 = 0.30
        ];
        let d = pick_domain::<3, 3>(&activity, &directions);
        assert_eq!(d, 1, "domain 1 should win with dot=0.74");
    }

    #[test]
    fn g1_pick_domain_ties_broken_by_lowest_index() {
        // Two domains with equal score → lowest index wins.
        let activity = [0.5, 0.5];
        let directions: [[f32; 2]; 3] = [
            [1.0, 0.0], // dot = 0.5
            [1.0, 0.0], // dot = 0.5  ← tie, but index 1 > 0 so 0 wins
            [0.0, 1.0], // dot = 0.5  ← also tie
        ];
        let d = pick_domain::<3, 2>(&activity, &directions);
        assert_eq!(d, 0, "ties broken by lowest index");
    }

    #[test]
    fn g1_pick_domain_single_domain() {
        let activity = [0.42];
        let directions: [[f32; 1]; 1] = [[1.0]];
        let d = pick_domain::<1, 1>(&activity, &directions);
        assert_eq!(d, 0);
    }

    // ─── G2-mechanics: project_guided + scatter_guided ──────────────────────

    #[test]
    fn g2_project_guided_gathers_selected_dims() {
        let z_full = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let indices = [0usize, 2, 5, 7];
        let mut z_out = [0.0f32; 4];
        project_guided::<8, 4>(&z_full, &indices, &mut z_out);
        assert_eq!(z_out, [0.0, 2.0, 5.0, 7.0]);
    }

    #[test]
    fn g2_project_guided_identity_reproduces_full_state() {
        let z_full = [1.0f32, 2.0, 3.0, 4.0];
        let indices = [0usize, 1, 2, 3];
        let mut z_out = [0.0f32; 4];
        project_guided::<4, 4>(&z_full, &indices, &mut z_out);
        assert_eq!(z_out, z_full);
    }

    #[test]
    fn g2_scatter_guided_writes_back_to_correct_positions() {
        let dz_proj = [10.0f32, 20.0, 30.0];
        let indices = [1usize, 4, 7];
        let mut dz_out_full = [0.0f32; 8];
        scatter_guided::<8, 3>(&dz_proj, &indices, &mut dz_out_full);
        // Non-written positions stay zero (caller's responsibility to pre-zero).
        assert_eq!(dz_out_full[0], 0.0);
        assert_eq!(dz_out_full[1], 10.0);
        assert_eq!(dz_out_full[4], 20.0);
        assert_eq!(dz_out_full[7], 30.0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "projection indices must be strictly ascending")]
    fn g2_project_guided_unsorted_indices_panic_in_debug() {
        let z_full = [0.0f32; 8];
        let indices = [0usize, 5, 2, 7]; // not ascending
        let mut z_out = [0.0f32; 4];
        project_guided::<8, 4>(&z_full, &indices, &mut z_out);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "projection index out of bounds")]
    fn g2_project_guided_oob_index_panic_in_debug() {
        let z_full = [0.0f32; 4];
        let indices = [0usize, 5]; // 5 >= D=4
        let mut z_out = [0.0f32; 2];
        project_guided::<4, 2>(&z_full, &indices, &mut z_out);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "projection indices must be strictly ascending")]
    fn g2_project_guided_duplicate_indices_panic_in_debug() {
        let z_full = [0.0f32; 8];
        let indices = [0usize, 0, 2]; // duplicate 0 → not strictly ascending
        let mut z_out = [0.0f32; 3];
        project_guided::<8, 3>(&z_full, &indices, &mut z_out);
    }

    // ─── G3-mechanics: project + scatter round-trip ─────────────────────────

    #[test]
    fn g3_project_scatter_round_trip_preserves_values() {
        let z_full = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let indices = [0usize, 2, 4];
        let mut z_proj = [0.0f32; 3];
        project_guided::<5, 3>(&z_full, &indices, &mut z_proj);
        assert_eq!(z_proj, [10.0, 30.0, 50.0]);
        let mut recovered = [0.0f32; 5];
        scatter_guided::<5, 3>(&z_proj, &indices, &mut recovered);
        // Non-selected positions stay zero; selected positions recover.
        assert_eq!(recovered[0], 10.0);
        assert_eq!(recovered[1], 0.0); // not in mask
        assert_eq!(recovered[2], 30.0);
        assert_eq!(recovered[3], 0.0); // not in mask
        assert_eq!(recovered[4], 50.0);
    }

    // ─── G1-router: VariableRankRouter dispatch ─────────────────────────────
    //
    // Uses a minimal 2-domain fixture (move <4,4> + combat <2,4>) at D_FULL=4,
    // A=2 — small enough to verify by hand, exercises the full dispatch path.

    /// A trivially-simple archetype field for the router fixture: returns
    /// `direction · dot(z, direction)` scaled by the field's index. This
    /// mirrors the PoC's DirectionField but boxed for ErasedCluster storage.
    struct FixtureField<const D: usize> {
        direction: [f32; D],
        blake3: [u8; 32],
        scale: f32,
    }

    impl<const D: usize> ArchetypeFieldSource<D> for FixtureField<D> {
        fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
            let dot: f32 = (0..D).map(|i| z[i] * self.direction[i]).sum();
            for (i, slot) in dz_scratch.iter_mut().enumerate().take(D) {
                *slot = self.direction[i] * dot * self.scale;
            }
            &mut dz_scratch[..D]
        }
        fn commitment(&self) -> [u8; 32] {
            self.blake3
        }
        fn lipschitz_bound(&self) -> f32 {
            self.scale
        }
    }

    fn make_fixture_field<const D: usize>(seed: usize, scale: f32) -> Box<FixtureField<D>> {
        let mut direction = [0.0f32; D];
        for (i, slot) in direction.iter_mut().enumerate() {
            let x = (seed * 37 + i * 13) as f32;
            *slot = ((x * 0.1).sin() + (x * 0.07).cos()) * 0.5;
        }
        let norm: f32 = direction.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in direction.iter_mut() {
            *v /= norm;
        }
        let mut blake3 = [0u8; 32];
        for (i, b) in blake3.iter_mut().enumerate() {
            *b = ((seed * 251 + i) & 0xFF) as u8;
        }
        Box::new(FixtureField {
            direction,
            blake3,
            scale,
        })
    }

    fn make_two_domain_router() -> VariableRankRouter<2, 4, 2> {
        // Domain 0: "move" — L=4 (no projection, full state), K=4 archetypes.
        let mut move_blend = CommittedFieldBlend::<4, 4>::uncommitted();
        move_blend.pi = [0.5, -0.3, 0.8, 0.1];
        move_blend.tau = 1.0;
        let move_fields: [Box<dyn ArchetypeFieldSource<4>>; 4] = [
            make_fixture_field::<4>(100, 1.0),
            make_fixture_field::<4>(200, 1.0),
            make_fixture_field::<4>(300, 1.0),
            make_fixture_field::<4>(400, 1.0),
        ];
        let move_cluster = Box::new(ClusterHolder::<4, 4>::new(move_blend, move_fields));

        // Domain 1: "combat" — L=2 (project to dims [0,1]), K=2 archetypes.
        let mut combat_blend = CommittedFieldBlend::<2, 2>::uncommitted();
        combat_blend.pi = [0.6, -0.2];
        combat_blend.tau = 1.0;
        let combat_fields: [Box<dyn ArchetypeFieldSource<2>>; 2] = [
            make_fixture_field::<2>(500, 1.0),
            make_fixture_field::<2>(600, 1.0),
        ];
        let combat_cluster = Box::new(ClusterHolder::<2, 2>::new(combat_blend, combat_fields));

        // Gate directions: domain 0 wins when activity[0] high; domain 1 when activity[1] high.
        let domain_directions: [[f32; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
        let projection_indices: [Vec<usize>; 2] = [vec![0, 1, 2, 3], vec![0, 1]];

        VariableRankRouter::<2, 4, 2>::new(
            [move_cluster, combat_cluster],
            projection_indices,
            domain_directions,
        )
    }

    #[test]
    fn g1_router_dispatch_move_domain() {
        let router = make_two_domain_router();
        let z = [1.0f32, 2.0, 3.0, 4.0];
        let activity = [0.9, 0.1]; // domain 0 (move) wins
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        let verdict = router.tick(&z, &activity, &mut scratch, &mut dz_out);
        assert_eq!(verdict.domain, 0, "high activity[0] → domain 0");
    }

    #[test]
    fn g1_router_dispatch_combat_domain() {
        let router = make_two_domain_router();
        let z = [1.0f32, 2.0, 3.0, 4.0];
        let activity = [0.1, 0.9]; // domain 1 (combat) wins
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        let verdict = router.tick(&z, &activity, &mut scratch, &mut dz_out);
        assert_eq!(verdict.domain, 1, "high activity[1] → domain 1");
    }

    #[test]
    fn g2_router_scatter_back_zeros_non_projected_dims() {
        // Combat domain projects to dims [0,1]. After tick, dz_out[2] and
        // dz_out[3] should stay zero (caller pre-zeroed; scatter only writes
        // projected dims).
        let router = make_two_domain_router();
        let z = [1.0f32, 2.0, 3.0, 4.0];
        let activity = [0.1, 0.9]; // combat
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4]; // pre-zeroed
        router.tick(&z, &activity, &mut scratch, &mut dz_out);
        // Dims 0,1 written by the blend; dims 2,3 NOT in combat's mask stay zero.
        assert!(
            dz_out[2] == 0.0 && dz_out[3] == 0.0,
            "non-projected dims must stay zero, got {dz_out:?}"
        );
        // Dim 0,1 should have non-trivial output (the blend wrote something).
        // We can't assert exact values without re-deriving the blend, but we
        // can assert they're finite.
        assert!(dz_out[0].is_finite() && dz_out[1].is_finite());
    }

    #[test]
    fn g3_router_no_nan_across_random_inputs() {
        // 10K random inputs through the router — no NaN, no panic.
        let router = make_two_domain_router();
        let mut rng_state = 0x1234_5678_9abc_def0u64;
        for _ in 0..10_000 {
            // xorshift64
            rng_state ^= rng_state >> 13;
            rng_state ^= rng_state << 7;
            rng_state ^= rng_state >> 17;
            let mut z = [0.0f32; 4];
            for slot in z.iter_mut() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *slot = ((rng_state >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0;
            }
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let activity = [
                ((rng_state >> 11) as f32 / (1u64 << 53) as f32),
                ((rng_state >> 21) as f32 / (1u64 << 53) as f32),
            ];
            let mut scratch = [0.0f32; 16];
            let mut dz_out = [0.0f32; 4];
            let verdict = router.tick(&z, &activity, &mut scratch, &mut dz_out);
            assert!(verdict.domain < 2);
            for v in dz_out.iter() {
                assert!(v.is_finite(), "NaN in dz_out: {dz_out:?}");
            }
        }
    }

    // ─── G1-macro: variable_rank_router_static! dispatch ───────────────────
    //
    // The same 2-domain fixture as the dynamic router tests, but using the
    // monomorphized macro router (Issue 189 T2). Verifies the macro generates
    // the same dispatch behavior with zero-vtable path.

    variable_rank_router_static! {
        /// 2-domain test router: move (K=4, L=4) + combat (K=2, L=2).
        struct StaticRouter2<2, 4, 2>;

        0 => move_cluster:   ClusterHolder<4, 4> => [0, 1, 2, 3];
        1 => combat_cluster: ClusterHolder<2, 2> => [0, 1];
    }

    fn make_two_domain_static_router() -> StaticRouter2 {
        // Domain 0: "move" — L=4 (no projection), K=4 archetypes.
        let mut move_blend = CommittedFieldBlend::<4, 4>::uncommitted();
        move_blend.pi = [0.5, -0.3, 0.8, 0.1];
        move_blend.tau = 1.0;
        let move_fields: [Box<dyn ArchetypeFieldSource<4>>; 4] = [
            make_fixture_field::<4>(100, 1.0),
            make_fixture_field::<4>(200, 1.0),
            make_fixture_field::<4>(300, 1.0),
            make_fixture_field::<4>(400, 1.0),
        ];
        let move_cluster = ClusterHolder::<4, 4>::new(move_blend, move_fields);

        // Domain 1: "combat" — L=2 (project to dims [0,1]), K=2 archetypes.
        let mut combat_blend = CommittedFieldBlend::<2, 2>::uncommitted();
        combat_blend.pi = [0.6, -0.2];
        combat_blend.tau = 1.0;
        let combat_fields: [Box<dyn ArchetypeFieldSource<2>>; 2] = [
            make_fixture_field::<2>(500, 1.0),
            make_fixture_field::<2>(600, 1.0),
        ];
        let combat_cluster = ClusterHolder::<2, 2>::new(combat_blend, combat_fields);

        StaticRouter2::new(
            move_cluster,
            combat_cluster,
            [[1.0, 0.0], [0.0, 1.0]],
        )
    }

    #[test]
    fn g1_macro_router_dispatch_move_domain() {
        let router = make_two_domain_static_router();
        let z = [1.0f32, 2.0, 3.0, 4.0];
        let activity = [0.9, 0.1]; // domain 0 (move) wins
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        let verdict = router.tick(&z, &activity, &mut scratch, &mut dz_out);
        assert_eq!(verdict.domain, 0, "high activity[0] → domain 0");
    }

    #[test]
    fn g1_macro_router_dispatch_combat_domain() {
        let router = make_two_domain_static_router();
        let z = [1.0f32, 2.0, 3.0, 4.0];
        let activity = [0.1, 0.9]; // domain 1 (combat) wins
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        let verdict = router.tick(&z, &activity, &mut scratch, &mut dz_out);
        assert_eq!(verdict.domain, 1, "high activity[1] → domain 1");
    }

    #[test]
    fn g1_macro_router_scatter_back_zeros_non_projected_dims() {
        // Combat domain projects to dims [0,1]. After tick, dz_out[2] and
        // dz_out[3] should stay zero (caller pre-zeroed).
        let router = make_two_domain_static_router();
        let z = [1.0f32, 2.0, 3.0, 4.0];
        let activity = [0.1, 0.9]; // combat
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4]; // pre-zeroed
        router.tick(&z, &activity, &mut scratch, &mut dz_out);
        assert!(
            dz_out[2] == 0.0 && dz_out[3] == 0.0,
            "non-projected dims must stay zero, got {dz_out:?}"
        );
        assert!(dz_out[0].is_finite() && dz_out[1].is_finite());
    }

    #[test]
    fn g1_macro_router_override_cluster_pi() {
        // Verify override_cluster_pi works — override domain 0's pi, check
        // the winner changes. With pi=[0.5,-0.3,0.8,0.1], winner=2 (0.8 is
        // highest). After override to [0.1,0.1,0.1,0.9], winner=3.
        let mut router = make_two_domain_static_router();
        let z = [1.0f32, 2.0, 3.0, 4.0];
        let activity = [0.9, 0.1]; // domain 0 (move)

        // Before override — winner should be 2 (pi[2]=0.8 is highest).
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        let v1 = router.tick(&z, &activity, &mut scratch, &mut dz_out);
        assert_eq!(v1.domain, 0);
        assert_eq!(v1.winner, 2, "pi=[0.5,-0.3,0.8,0.1] → winner=2");

        // Override pi for domain 0 — now winner should be 3 (pi[3]=0.9).
        router.override_cluster_pi(0, &[0.1, 0.1, 0.1, 0.9]);
        let mut scratch2 = [0.0f32; 16];
        let mut dz_out2 = [0.0f32; 4];
        let v2 = router.tick(&z, &activity, &mut scratch2, &mut dz_out2);
        assert_eq!(v2.domain, 0);
        assert_eq!(v2.winner, 3, "overridden pi=[0.1,0.1,0.1,0.9] → winner=3");
    }

    #[test]
    fn g1_macro_router_no_nan_across_10k_inputs() {
        // Port of g3_router_no_nan_across_random_inputs to the macro router.
        let router = make_two_domain_static_router();
        let mut rng_state = 0x1234_5678_9abc_def0u64;
        for _ in 0..10_000 {
            rng_state ^= rng_state >> 13;
            rng_state ^= rng_state << 7;
            rng_state ^= rng_state >> 17;
            let mut z = [0.0f32; 4];
            for slot in z.iter_mut() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *slot = ((rng_state >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0;
            }
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let activity = [
                ((rng_state >> 11) as f32 / (1u64 << 53) as f32),
                ((rng_state >> 21) as f32 / (1u64 << 53) as f32),
            ];
            let mut scratch = [0.0f32; 16];
            let mut dz_out = [0.0f32; 4];
            let verdict = router.tick(&z, &activity, &mut scratch, &mut dz_out);
            assert!(verdict.domain < 2);
            for v in dz_out.iter() {
                assert!(v.is_finite(), "NaN in dz_out: {dz_out:?}");
            }
        }
    }

    #[test]
    fn g1_macro_router_bit_identical_to_dynamic() {
        // The macro router must produce the SAME results as the dynamic
        // VariableRankRouter — same math, only the dispatch path differs.
        // This is the G1 parity gate (Issue 189 risk: "G1 regression from
        // monomorphization").
        let dyn_router = make_two_domain_router();
        let static_router = make_two_domain_static_router();

        let mut rng_state = 0xabcd_ef01_2345_6789u64;
        for i in 0..500 {
            rng_state ^= rng_state >> 13;
            rng_state ^= rng_state << 7;
            rng_state ^= rng_state >> 17;
            let mut z = [0.0f32; 4];
            for slot in z.iter_mut() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *slot = ((rng_state >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0;
            }
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let activity = [
                ((rng_state >> 11) as f32 / (1u64 << 53) as f32),
                ((rng_state >> 21) as f32 / (1u64 << 53) as f32),
            ];

            let mut s1 = [0.0f32; 16];
            let mut dz1 = [0.0f32; 4];
            let v1 = dyn_router.tick(&z, &activity, &mut s1, &mut dz1);

            let mut s2 = [0.0f32; 16];
            let mut dz2 = [0.0f32; 4];
            let v2 = static_router.tick(&z, &activity, &mut s2, &mut dz2);

            assert_eq!(v1, v2, "verdict mismatch at iter {i}: {v1:?} vs {v2:?}");
            assert_eq!(
                dz1, dz2,
                "dz_out bit-mismatch at iter {i}: {dz1:?} vs {dz2:?}"
            );
        }
    }
}
