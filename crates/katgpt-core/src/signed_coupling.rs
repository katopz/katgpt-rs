//! Signed-coupling opinion dynamics — Glauber update on a signed social graph
//! plus the three crowd order parameters.
//!
//! > **Source:** El, Paeng, Dinc, Su, Erdogan, Pappu, Ye, Zhao, Ganguli & Zou,
//! > *Physics of Agents: Statistical Mechanics Predicts Collective Behavior of
//! > AI Agents*, [arXiv:2608.16578](https://arxiv.org/abs/2608.16578), 2026.
//! > Distilled in `.research/497`; implemented per `.issues/680`.
//!
//! # The primitive
//!
//! Each of `N` entities holds a binary stance `s_i ∈ {−1, +1}`. Ties between
//! entities are **signed and typed**: `J_ij = +1` (concordant — allies),
//! `J_ij = −1` (discordant — rivals), `J_ij = 0` (no tie, simply absent from
//! the graph). One synchronous Glauber (heat-bath) step gives the probability
//! that `i` holds `+1` next tick:
//!
//! ```text
//! h_i = β⁺·Σ_j J⁺_ij·s_j  +  β⁻·Σ_j J⁻_ij·s_j  +  β₀·Σ_j |J_ij|·s_j  +  g_i
//! P(s_i = +1) = σ(h_i)
//! ```
//!
//! with `J⁺ = max(J, 0)`, `J⁻ = min(J, 0)`, and three separate couplings:
//! `β⁺` (how hard allies pull), `β⁻` (how hard rivals push), `β₀` (how hard
//! *mere connection* pulls, regardless of tie type). `g_i` is the intrinsic
//! field — the entity's own disposition, in our stack a direction-vector
//! dot-product `w·φ_i` (personality × question), which is why nothing here
//! needs training: **the caller authors the couplings.**
//!
//! Because `J⁻_ij = −1` on a discordant tie, its contribution is `−β⁻·s_j` —
//! rivals push you *away* from their stance. Collapsing the three sums into
//! one pass over the neighbor list gives the shipped kernel:
//!
//! ```text
//! h_i = Σ_{j ∈ N(i)} w[sign(J_ij)] · s_j  +  g_i
//! w[+] = β⁺ + β₀        w[−] = β₀ − β⁻
//! ```
//!
//! — a two-entry weight table. Better still, the two channels can be summed
//! *separately* and weighted once per node, because the `|J|` channel is not
//! independent: with `P = Σ_{concordant} s_j` and `D = Σ_{discordant} s_j`, the
//! third sum is just `P + D`, so
//!
//! ```text
//! h_i = (β⁺ + β₀)·P  +  (β₀ − β⁻)·D  +  g_i
//! ```
//!
//! The shipped inner loop is therefore two conditional adds per edge — **no
//! multiply and no table load in the edge loop at all**, two multiplies per
//! *node*.
//!
//! Measured (Bench 672 G2, median pairwise ratio over 9 interleaved rounds):
//! **0.97–1.02× the naive three-accumulator form** — parity to a couple of
//! percent, at ~1.8 ns per edge from N=32 to N=1024. The first draft weighted
//! each edge through a 2-entry table *inside* the edge loop and measured
//! **1.5× slower** than the naive form; the per-node hoist is what recovered
//! it. Worth knowing before "optimizing" this loop again.
//!
//! # Order parameters (the crowd-level half)
//!
//! - [`net_opinion`] `n = mean(s)` — which way the crowd leans (−1…+1).
//! - [`crowd_conviction`] `c = mean(s²)` — how *hard* it holds any stance at
//!   all, direction-blind. With pure `±1` states `c ≡ 1`; it becomes
//!   informative on the magnitude-weighted path (see below).
//! - [`SusceptibilityAccumulator`] `χ = N·Var_t(|n|)` — the response of the
//!   crowd to its own fluctuations, accumulated over ticks. Its **peak over a
//!   temperature sweep locates the critical social temperature `T_c`**, which
//!   is the paper's central diagnostic.
//!
//! The three regimes the paper names fall out of `(|n|, c)`: **indifference**
//! (both low), **polarization** (`|n|` low, `c` high — two committed camps
//! cancelling), **consensus** (both high).
//!
//! # `±1` states vs magnitude-weighted states
//!
//! The kernel reads `states` as plain `f32`, so callers have two options:
//!
//! - **Discrete** `s_j ∈ {−1, +1}` — the paper's form. Sample each tick with
//!   [`sample_states_into`]. `crowd_conviction` is then identically 1.0 and
//!   only `n` moves.
//! - **Magnitude-weighted** `s_j ∈ [−1, +1]` — a stance *held with strength*
//!   (e.g. `2·σ(x) − 1` of an emotion projection, or a running mean of recent
//!   samples). Then `c = mean(s²)` genuinely separates polarization (two
//!   committed camps, `c → 1`) from indifference (everyone wavering, `c → 0`),
//!   which is what makes the regime split legible at a single tick instead of
//!   only over a trajectory. This is the recommended consumer path.
//!
//! Both are the same code; nothing in the kernel assumes `|s| = 1`.
//!
//! # Latent vs raw boundary (per `AGENTS.md` §"Latent vs Raw Space Rules")
//!
//! Everything here is **semantic/think-brain**: stances, couplings, and the
//! local fields `h_i` are computed locally and never synced. Only crowd
//! *summaries* (`n`, `c`, `χ` per zone) may cross the sync boundary, under the
//! flock-centroid precedent (a summary is not per-entity truth). If a crowd
//! decision commits through chain, the resulting **event** is a raw TxDelta;
//! the dynamics that produced it stay latent. Never sync `h_i`.
//!
//! # Honesty note (UQ)
//!
//! `σ(h_i)` is a Bernoulli parameter of a **dynamics rule**, not a calibrated
//! forecast. Any future claim about prediction *quality* (the paper reports
//! 75–86% balanced accuracy against real LLM crowds) is a UQ-bearing claim and
//! owes the conformal-naive floor per `AGENTS.md` §"Report the Floor" — the
//! primitive itself makes no such claim.
//!
//! # Vocabulary collision (load-bearing for future greps)
//!
//! "Conviction" means two different things in this stack, and both greps must
//! land here:
//!
//! | name | crate | meaning |
//! |---|---|---|
//! | [`crowd_conviction`] | this module | crowd **order parameter** `mean(s²)` — how hard the crowd holds stances |
//! | Sheaf-ADMM `conviction` | `katgpt-dec::sheaf_admm`, `riir-agents::multi_agent` | per-agent/per-dim **resistance** in the consensus quadratic — how hard one agent holds its ground |
//!
//! They compose rather than compete: a Sheaf conviction vector is a natural
//! source for this module's per-agent intrinsic field `g_i`.
//!
//! Issue 680: Signed-Coupling Opinion Dynamics — Glauber (heat-bath) update on
//! a SIGNED social graph plus the three crowd order parameters, distilled from
//! "Physics of Agents" (El et al., arXiv:2608.16578; Research 497). The kernel
//! is CLR set attention's sibling — the same σ(gated weighted sum) shape, but
//! with signed, tie-typed couplings on a stance instead of unsigned relevance
//! weights: h_i = β⁺Σ J⁺s + β⁻Σ J⁻s + β₀Σ|J|s + g_i, collapsed to one
//! branch-free O(edges) pass over a CSR row. Ships the three reducers nothing
//! else in the stack had: net_opinion (mean), crowd_conviction (mean of
//! squares — genuinely new), and the χ = N·Var_t(|n|) susceptibility
//! accumulator whose peak over a temperature sweep locates the critical social
//! temperature. Pure modelless (the paper's only gradient descent fits ~19
//! scalars to real LLM transitions; a game crowd AUTHORS its couplings, so the
//! paper's fitted ranges become designer-facing defaults). OPT-IN — promotion
//! waits on a production consumer, the CLR precedent.

use crate::sigmoid;

/// Paper-fitted range for the concordant coupling `β⁺` (Table 2, four models).
pub const PAPER_BETA_PLUS_RANGE: (f32, f32) = (0.9, 2.4);
/// Paper-fitted range for the discordant coupling `β⁻`.
pub const PAPER_BETA_MINUS_RANGE: (f32, f32) = (0.2, 1.1);
/// Paper-fitted range for the mere-connection coupling `β₀`.
pub const PAPER_BETA_ZERO_RANGE: (f32, f32) = (0.6, 1.0);
/// Paper-fitted range for the truth gap `β_T⁺ − β_F⁺` (correct neighbors pull
/// harder than wrong ones on the concordant channel).
pub const PAPER_TRUTH_GAP_RANGE: (f32, f32) = (0.1, 0.3);

/// Why a [`SignedGraph`] could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedGraphError {
    /// An endpoint index was `>= n_nodes`. Carries `(edge_index, bad_node)`.
    NodeOutOfRange(usize, u32),
    /// A tie sign was `0`. "No tie" is expressed by *omitting* the edge — a
    /// stored zero would still cost a neighbor-list slot and contribute `β₀`
    /// through `|J_ij|`, which is exactly wrong. Carries the edge index.
    ZeroSign(usize),
    /// A self-loop `i → i`. The energy `E = −½ΣΣ J_ij s_i s_j` has no diagonal;
    /// a self-tie is an intrinsic field, so pass it through `intrinsic` (`g_i`)
    /// instead. Carries the edge index.
    SelfLoop(usize),
}

impl core::fmt::Display for SignedGraphError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NodeOutOfRange(e, n) => write!(f, "edge {e}: node {n} out of range"),
            Self::ZeroSign(e) => write!(f, "edge {e}: zero sign — omit the edge instead"),
            Self::SelfLoop(e) => write!(f, "edge {e}: self-loop — use the intrinsic field g_i"),
        }
    }
}

impl std::error::Error for SignedGraphError {}

/// Row-compressed signed adjacency (CSR): one contiguous neighbor list plus a
/// parallel sign list, `u32` indices, **no heap traffic after construction**.
///
/// `J⁺` and `J⁻` share one row rather than living in two lists: the kernel
/// needs both channels *and* the `|J|` channel on the same visit, so one pass
/// over `(neighbor, sign)` pairs touches each cache line once instead of
/// three times.
#[derive(Debug, Clone)]
pub struct SignedGraph {
    /// Row starts, length `n_nodes + 1`.
    offsets: Box<[u32]>,
    /// Neighbor indices, grouped by row.
    neighbors: Box<[u32]>,
    /// Tie sign per entry of `neighbors`: `+1` concordant, `−1` discordant.
    signs: Box<[i8]>,
}

impl SignedGraph {
    /// Build from **symmetric** ties — the paper's form (`J_ij = J_ji`, the
    /// energy has no preferred direction). Each input edge lands in both rows.
    ///
    /// Duplicate edges are *not* deduplicated: they accumulate, which is the
    /// documented way to express a tie of weight 2. `|J_ij| ∈ {0, 1}` in the
    /// paper, so pass each pair once for paper-faithful behavior.
    pub fn from_edges(n_nodes: usize, edges: &[(u32, u32, i8)]) -> Result<Self, SignedGraphError> {
        Self::build(n_nodes, edges, true)
    }

    /// Build from **directed** ties — `i` listens to `j` without `j` listening
    /// back. Not the paper's form (the energy formulation needs symmetry), but
    /// the natural one for asymmetric social influence: a recruit weighs the
    /// veteran's stance, not the reverse. Each input edge `(i, j, sign)` lands
    /// in row `i` only.
    pub fn from_directed_edges(
        n_nodes: usize,
        edges: &[(u32, u32, i8)],
    ) -> Result<Self, SignedGraphError> {
        Self::build(n_nodes, edges, false)
    }

    fn build(
        n_nodes: usize,
        edges: &[(u32, u32, i8)],
        symmetric: bool,
    ) -> Result<Self, SignedGraphError> {
        // Pass 1: validate + count degrees.
        let mut degrees = vec![0u32; n_nodes];
        for (e, &(i, j, sign)) in edges.iter().enumerate() {
            match () {
                _ if i as usize >= n_nodes => return Err(SignedGraphError::NodeOutOfRange(e, i)),
                _ if j as usize >= n_nodes => return Err(SignedGraphError::NodeOutOfRange(e, j)),
                _ if sign == 0 => return Err(SignedGraphError::ZeroSign(e)),
                _ if i == j => return Err(SignedGraphError::SelfLoop(e)),
                _ => {}
            }
            degrees[i as usize] += 1;
            if symmetric {
                degrees[j as usize] += 1;
            }
        }

        // Prefix sum → row offsets.
        let mut offsets = vec![0u32; n_nodes + 1];
        let mut acc = 0u32;
        for (slot, &d) in offsets.iter_mut().skip(1).zip(degrees.iter()) {
            acc += d;
            *slot = acc;
        }
        let n_entries = acc as usize;

        // Pass 2: scatter. `cursor` walks each row as it fills.
        let mut neighbors = vec![0u32; n_entries];
        let mut signs = vec![0i8; n_entries];
        let mut cursor: Vec<u32> = offsets[..n_nodes].to_vec();
        let mut push = |row: u32, other: u32, sign: i8| {
            let at = cursor[row as usize] as usize;
            neighbors[at] = other;
            signs[at] = sign;
            cursor[row as usize] += 1;
        };
        for &(i, j, sign) in edges {
            push(i, j, sign);
            if symmetric {
                push(j, i, sign);
            }
        }

        Ok(Self {
            offsets: offsets.into_boxed_slice(),
            neighbors: neighbors.into_boxed_slice(),
            signs: signs.into_boxed_slice(),
        })
    }

    /// Number of entities (`N`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether the graph holds no entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total stored `(node, neighbor)` entries — twice the undirected edge
    /// count for a symmetric graph.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.neighbors.len()
    }

    /// Number of ties incident to `i`.
    #[must_use]
    pub fn degree(&self, i: usize) -> usize {
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }

    /// Row `i` as `(neighbors, signs)` — always equal length.
    #[must_use]
    pub fn row(&self, i: usize) -> (&[u32], &[i8]) {
        let (lo, hi) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        (&self.neighbors[lo..hi], &self.signs[lo..hi])
    }
}

/// The three couplings of the Glauber update.
///
/// [`Default`] is the **midpoint of each paper-fitted range** (see
/// [`PAPER_BETA_PLUS_RANGE`] and siblings). The paper's own finding is
/// `β⁺ > β⁻` in every model × dataset cell — concordant ties outweigh
/// discordant ones, which is why crowds drift toward consensus rather than
/// deadlock. Keep that ordering unless you *want* a polarizing crowd.
///
/// # The `β₀` vs `β⁻` trap (measured, Bench 672 G1a)
///
/// A discordant tie's net weight is `β₀ − β⁻`, so **`β₀ > β⁻` makes rivals
/// attractive**: mere connection outweighs rivalry and the crowd converges no
/// matter how frustrated the graph is. At these defaults (the range midpoints)
/// that weight is `0.8 − 0.65 = +0.15` — consensus-biased by construction,
/// which is faithful to the paper but surprising if you expected a two-block
/// graph to polarize. Polarization needs `β⁻ > β₀`, reachable *inside* the
/// fitted ranges at the corner `β⁺ = 0.9`, `β⁻ = 1.1`, `β₀ = 0.6`
/// (net `−0.5`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Couplings {
    /// `β⁺` — pull from concordant (ally) ties.
    pub beta_plus: f32,
    /// `β⁻` — push from discordant (rival) ties. Enters `h` negatively.
    pub beta_minus: f32,
    /// `β₀` — pull from *mere connection*, tie type ignored.
    pub beta_zero: f32,
}

impl Default for Couplings {
    fn default() -> Self {
        Self {
            beta_plus: midpoint(PAPER_BETA_PLUS_RANGE),
            beta_minus: midpoint(PAPER_BETA_MINUS_RANGE),
            beta_zero: midpoint(PAPER_BETA_ZERO_RANGE),
        }
    }
}

fn midpoint((lo, hi): (f32, f32)) -> f32 {
    0.5 * (lo + hi)
}

impl Couplings {
    /// Rescale every coupling by `1/t` — the **social-temperature dial**.
    ///
    /// Temperature enters the Glauber rule only as an overall scale on the
    /// field (`h/T`), so one scalar moves a crowd between apathy (high `T`:
    /// couplings vanish, everyone follows their own `g_i`) and a mob (low `T`:
    /// couplings dominate, the crowd orders itself). The paper's crowds all sit
    /// **below** `T_c`, which is why conviction builds.
    ///
    /// `t <= 0` is clamped to [`f32::MIN_POSITIVE`] — zero temperature is the
    /// deterministic limit and would divide by zero here; callers wanting hard
    /// argmax should threshold the probability instead.
    #[must_use]
    pub fn at_social_temperature(&self, t: f32) -> Self {
        let inv = 1.0 / t.max(f32::MIN_POSITIVE);
        Self {
            beta_plus: self.beta_plus * inv,
            beta_minus: self.beta_minus * inv,
            beta_zero: self.beta_zero * inv,
        }
    }

    /// Per-channel weights `[w(+1), w(−1)]`, indexed by the sign bit — the
    /// collapse that lets the kernel weight each channel once per *node*
    /// instead of once per edge (`w(−1) = β₀ − β⁻`, so a negative value means
    /// rivals genuinely repel; see the type-level note).
    #[must_use]
    #[inline]
    pub fn edge_weights(&self) -> [f32; 2] {
        [
            self.beta_zero + self.beta_plus,
            self.beta_zero - self.beta_minus,
        ]
    }
}

/// The paper's truth-asymmetric 5-coupling split: the concordant *and*
/// discordant channels each fork on whether the neighbor holds the **correct**
/// stance (`κ_j`).
///
/// Both forks are truth-seeking, and in opposite directions — the paper's §6
/// finding: correct neighbors pull *harder* along the concordant channel
/// (`β_T⁺ > β_F⁺`), and wrong neighbors push *harder* along the discordant one
/// (`β_F⁻ > β_T⁻`). In game terms this is the closed-form of "ask the veteran,
/// not the tourist": hints from NPCs who actually cleared the content spread
/// with a stronger coupling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformedCouplings {
    /// `β_T⁺` — concordant tie, neighbor is correct.
    pub beta_true_plus: f32,
    /// `β_F⁺` — concordant tie, neighbor is wrong.
    pub beta_false_plus: f32,
    /// `β_T⁻` — discordant tie, neighbor is correct.
    pub beta_true_minus: f32,
    /// `β_F⁻` — discordant tie, neighbor is wrong.
    pub beta_false_minus: f32,
    /// `β₀` — mere connection, indifferent to both tie type and correctness.
    pub beta_zero: f32,
}

impl Default for InformedCouplings {
    fn default() -> Self {
        Self::from_couplings(&Couplings::default(), midpoint(PAPER_TRUTH_GAP_RANGE))
    }
}

impl InformedCouplings {
    /// Split a symmetric [`Couplings`] into the 5-coupling form by opening a
    /// `gap` between the informed and uninformed forks, centered on the base
    /// values (so the *average* neighbor keeps the base coupling):
    ///
    /// ```text
    /// β_T⁺ = β⁺ + gap/2     β_F⁺ = β⁺ − gap/2
    /// β_T⁻ = β⁻ − gap/2     β_F⁻ = β⁻ + gap/2
    /// ```
    ///
    /// A `gap` of 0 reproduces the base couplings exactly on every channel.
    #[must_use]
    pub fn from_couplings(base: &Couplings, gap: f32) -> Self {
        let half = 0.5 * gap;
        Self {
            beta_true_plus: base.beta_plus + half,
            beta_false_plus: base.beta_plus - half,
            beta_true_minus: base.beta_minus - half,
            beta_false_minus: base.beta_minus + half,
            beta_zero: base.beta_zero,
        }
    }

    /// Per-`(sign, κ)` edge weights, indexed `sign_bit << 1 | informed_bit`:
    /// `[+/wrong, +/correct, −/wrong, −/correct]`.
    #[must_use]
    #[inline]
    pub fn edge_weights(&self) -> [f32; 4] {
        [
            self.beta_zero + self.beta_false_plus,
            self.beta_zero + self.beta_true_plus,
            self.beta_zero - self.beta_false_minus,
            self.beta_zero - self.beta_true_minus,
        ]
    }
}

/// Sign bit of a tie as a LUT index: `0` for `+1`, `1` for `−1`. Arithmetic
/// shift, no branch.
#[inline]
fn sign_index(sign: i8) -> usize {
    ((sign >> 7) & 1) as usize
}

/// One synchronous Glauber step: `out_probs[i] = σ(h_i)`.
///
/// `O(entries)`, zero-allocation — writes only into `out_probs`, which the
/// caller owns and reuses across ticks.
///
/// `intrinsic` is the per-entity field `g_i` (personality × question, in our
/// stack a direction-vector dot product). Pass all-zeros for a pure
/// social-pressure crowd.
///
/// # Panics
///
/// If `states`, `intrinsic`, or `out_probs` disagree with `graph.len()`. The
/// check is once per call, not per element.
pub fn signed_coupling_update_into(
    graph: &SignedGraph,
    states: &[f32],
    couplings: &Couplings,
    intrinsic: &[f32],
    out_probs: &mut [f32],
) {
    let n = graph.len();
    assert_eq!(states.len(), n, "states length must equal graph node count");
    assert_eq!(
        intrinsic.len(),
        n,
        "intrinsic length must equal graph node count"
    );
    assert_eq!(
        out_probs.len(),
        n,
        "out_probs length must equal graph node count"
    );

    let [w_plus, w_minus] = couplings.edge_weights();
    for i in 0..n {
        let (neighbors, signs) = graph.row(i);
        // Two channel sums, no multiply and no table load in the edge loop.
        let mut concordant = 0.0f32;
        let mut discordant = 0.0f32;
        for (&j, &sign) in neighbors.iter().zip(signs) {
            let s = states[j as usize];
            if sign > 0 {
                concordant += s;
            } else {
                discordant += s;
            }
        }
        out_probs[i] = sigmoid(w_plus * concordant + w_minus * discordant + intrinsic[i]);
    }
}

/// Truth-asymmetric Glauber step — the paper's 5-coupling variant.
///
/// `informed[j]` is the indicator `κ_j`: `true` when `j` holds the stance that
/// is actually correct. On subjective questions there is no such thing, and
/// the caller should use [`signed_coupling_update_into`] instead of passing an
/// all-`false` slice (which is *not* equivalent — it drives every tie onto the
/// `β_F` fork).
///
/// A sibling function rather than a flag on the base call, deliberately: the
/// weight table has a different shape, and a bool parameter would put a branch
/// in the inner loop.
///
/// # Panics
///
/// If `states`, `informed`, `intrinsic`, or `out_probs` disagree with
/// `graph.len()`.
pub fn signed_coupling_update_informed_into(
    graph: &SignedGraph,
    states: &[f32],
    informed: &[bool],
    couplings: &InformedCouplings,
    intrinsic: &[f32],
    out_probs: &mut [f32],
) {
    let n = graph.len();
    assert_eq!(states.len(), n, "states length must equal graph node count");
    assert_eq!(
        informed.len(),
        n,
        "informed length must equal graph node count"
    );
    assert_eq!(
        intrinsic.len(),
        n,
        "intrinsic length must equal graph node count"
    );
    assert_eq!(
        out_probs.len(),
        n,
        "out_probs length must equal graph node count"
    );

    let w = couplings.edge_weights();
    for i in 0..n {
        let (neighbors, signs) = graph.row(i);
        // Four channel sums — (sign × κ) — weighted once per node, same shape
        // as the two-channel base kernel.
        let mut sums = [0.0f32; 4];
        for (&j, &sign) in neighbors.iter().zip(signs) {
            let j = j as usize;
            sums[(sign_index(sign) << 1) | usize::from(informed[j])] += states[j];
        }
        let h = w[0] * sums[0] + w[1] * sums[1] + w[2] * sums[2] + w[3] * sums[3] + intrinsic[i];
        out_probs[i] = sigmoid(h);
    }
}

/// Draw the next discrete stance from a probability slice:
/// `out_states[i] = if uniforms[i] < probs[i] { +1 } else { −1 }`.
///
/// The caller supplies the uniforms, so this stays RNG-free, deterministic,
/// and zero-allocation — seed addressability lives with the consumer, and a
/// replayable rollout is just a replayable uniform stream.
///
/// # Panics
///
/// If the three slices are not the same length.
pub fn sample_states_into(probs: &[f32], uniforms: &[f32], out_states: &mut [f32]) {
    assert_eq!(
        probs.len(),
        uniforms.len(),
        "uniforms must match probs length"
    );
    assert_eq!(
        probs.len(),
        out_states.len(),
        "out_states must match probs length"
    );
    for ((p, u), s) in probs.iter().zip(uniforms).zip(out_states.iter_mut()) {
        *s = if u < p { 1.0 } else { -1.0 };
    }
}

/// Net opinion `n = mean(s)` — which way the crowd leans, in `[−1, +1]` for
/// bounded stances. `0.0` for an empty crowd.
#[must_use]
pub fn net_opinion(states: &[f32]) -> f32 {
    match states.len() {
        0 => 0.0,
        n => states.iter().sum::<f32>() / n as f32,
    }
}

/// Crowd conviction `c = mean(s²)` — how hard the crowd holds stances at all,
/// direction-blind. `0.0` for an empty crowd.
///
/// Identically `1.0` on the discrete `±1` path; informative on the
/// magnitude-weighted path, where it separates **polarization** (`|n|` low,
/// `c` high — two committed camps) from **indifference** (`|n|` low, `c` low —
/// nobody committed). Nothing else in the stack ships this reducer; the
/// similarly-named Sheaf-ADMM `conviction` is a different quantity (see the
/// module docs).
#[must_use]
pub fn crowd_conviction(states: &[f32]) -> f32 {
    match states.len() {
        0 => 0.0,
        n => states.iter().map(|s| s * s).sum::<f32>() / n as f32,
    }
}

/// Verdict margin `m = |Σ w_i·v_i| / Σ w_i` with `v_i = sign(s_i)` — the
/// **one-snapshot crowd-manipulability forecast** (Research 499 /
/// arXiv:2608.12645 §6.2: baseline jury majority strength is the single
/// best predictor of which verdicts flip under pressure, |ρ| = 0.59).
///
/// This is the *forecast* companion to the two shipped *measurement*
/// order parameters: [`crowd_conviction`] (instantaneous, direction-blind)
/// and [`SusceptibilityAccumulator`] (needs a time series before it can say
/// anything). The margin needs ONE vote reading — `O(n)`, zero alloc — and
/// answers a different question: not "how committed is this crowd now" but
/// "how far could a persuader move it". The conviction-gated attack that
/// consumes this reading (`pressure ∝ 1 − m`) was PoC-validated on this
/// substrate (riir-ai Issue 745: gated ρ(margin, flips) = **−0.65** vs the
/// paper's −0.59; gated pressure leaves margin > 0.8 crowds unmoved at
/// 0.016 flip-fraction while uniform pressure converts them at 0.804).
///
/// `weights` are per-entity reliability scores — [`crate::set_attention::
/// clr_reliability_scores`] is the intended producer (a reliable scout's
/// vote counts more), but any non-negative slice works; pass all-ones for
/// the unweighted majority reading. Degenerate inputs return `0.0` (empty
/// crowd, zero weight total), matching the module's reducer convention.
///
/// # Panics
///
/// If `weights` and `states` disagree in length (two-slice contract — the
/// single-slice reducers above have nothing to mismatch).
///
/// [`crate::set_attention::clr_reliability_scores`]: crate::set_attention::clr_reliability_scores
#[must_use]
pub fn verdict_margin(weights: &[f32], states: &[f32]) -> f32 {
    assert_eq!(
        weights.len(),
        states.len(),
        "verdict_margin: weights and states must be the same length"
    );
    let sum_w: f32 = weights.iter().sum();
    if sum_w <= 0.0 {
        return 0.0;
    }
    let signed: f32 = weights
        .iter()
        .zip(states)
        .map(|(w, s)| w * s.signum())
        .sum();
    (signed / sum_w).abs()
}

/// Running `Var_t(|n|)` over ticks → susceptibility `χ = N · Var_t(|n|)`.
///
/// Welford in `f64`, so a long rollout does not lose the variance to
/// cancellation. Fixed size, `Copy`, no heap — one of these per crowd per
/// temperature is free.
///
/// **`T_c` is located offline.** The critical temperature is the `argmax` of
/// `χ` over a temperature sweep (the paper uses 41 log-spaced points × 500
/// steps); that is a bench/example workload, not a tick-rate path. What ships
/// as a runtime primitive is the accumulator — a live "how twitchy is this
/// crowd" reading.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SusceptibilityAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
}

impl SusceptibilityAccumulator {
    /// Empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold in one tick's net opinion. Takes `n` (signed); the accumulator
    /// tracks `|n|`, per the paper's `χ = N·Var_t(|n(t)|)` — the absolute value
    /// keeps a crowd that flips its majority from reading as maximally
    /// susceptible purely because of the sign change.
    pub fn observe(&mut self, net_opinion: f32) {
        let x = f64::from(net_opinion.abs());
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    /// Ticks observed.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Mean `|n|` so far — the crowd's typical lean magnitude.
    #[must_use]
    pub fn mean_abs_net(&self) -> f32 {
        self.mean as f32
    }

    /// Sample variance of `|n|` (Bessel-corrected). `0.0` before two ticks.
    #[must_use]
    pub fn variance(&self) -> f32 {
        match self.count {
            0 | 1 => 0.0,
            c => (self.m2 / (c - 1) as f64) as f32,
        }
    }

    /// Susceptibility `χ = N · Var_t(|n|)` for a crowd of `n_agents`.
    #[must_use]
    pub fn susceptibility(&self, n_agents: usize) -> f32 {
        n_agents as f32 * self.variance()
    }

    /// Drop the history (e.g. after a burn-in, or on a new temperature).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-clique of allies plus one rival tie, used across the shape tests.
    fn small_graph() -> SignedGraph {
        SignedGraph::from_edges(4, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, -1)]).unwrap()
    }

    #[test]
    fn csr_rows_hold_every_symmetric_tie() {
        let g = small_graph();
        assert_eq!(g.len(), 4);
        // 4 undirected edges → 8 directed entries.
        assert_eq!(g.entry_count(), 8);
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 2);
        let (nb, sg) = g.row(0);
        // Row 0 holds the concordant 0-1 tie and the discordant 0-3 tie.
        let mut pairs: Vec<(u32, i8)> = nb.iter().copied().zip(sg.iter().copied()).collect();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![(1, 1), (3, -1)]);
    }

    #[test]
    fn directed_edges_land_in_one_row_only() {
        let g = SignedGraph::from_directed_edges(3, &[(0, 1, 1), (0, 2, -1)]).unwrap();
        assert_eq!(g.entry_count(), 2);
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 0);
        assert_eq!(g.degree(2), 0);
    }

    #[test]
    fn malformed_edges_are_rejected() {
        assert_eq!(
            SignedGraph::from_edges(2, &[(0, 5, 1)]).unwrap_err(),
            SignedGraphError::NodeOutOfRange(0, 5)
        );
        assert_eq!(
            SignedGraph::from_edges(2, &[(0, 1, 0)]).unwrap_err(),
            SignedGraphError::ZeroSign(0)
        );
        assert_eq!(
            SignedGraph::from_edges(2, &[(1, 1, 1)]).unwrap_err(),
            SignedGraphError::SelfLoop(0)
        );
    }

    #[test]
    fn edge_weight_lut_matches_the_three_sum_form() {
        let c = Couplings {
            beta_plus: 1.5,
            beta_minus: 0.5,
            beta_zero: 0.8,
        };
        let w = c.edge_weights();
        // Concordant: β⁺·(+1) + β₀·|+1| = 2.3. Discordant: β⁻·(−1) + β₀ = 0.3.
        assert!((w[sign_index(1)] - 2.3).abs() < 1e-6);
        assert!((w[sign_index(-1)] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn update_matches_the_explicit_three_sum_reference() {
        let g = small_graph();
        let c = Couplings {
            beta_plus: 1.2,
            beta_minus: 0.4,
            beta_zero: 0.7,
        };
        let states = [1.0, -1.0, 1.0, -1.0];
        let intrinsic = [0.1, -0.2, 0.3, 0.0];
        let mut probs = [0.0f32; 4];
        signed_coupling_update_into(&g, &states, &c, &intrinsic, &mut probs);

        // Reference: the three separate sums of the paper's equation.
        for i in 0..4 {
            let (nb, sg) = g.row(i);
            let mut plus = 0.0;
            let mut minus = 0.0;
            let mut zero = 0.0;
            for (&j, &s) in nb.iter().zip(sg) {
                let sj = states[j as usize];
                match s {
                    1 => plus += sj,
                    -1 => minus += -sj, // J⁻ = −1
                    _ => unreachable!(),
                }
                zero += sj; // |J| = 1
            }
            let h = c.beta_plus * plus + c.beta_minus * minus + c.beta_zero * zero + intrinsic[i];
            assert!(
                (probs[i] - sigmoid(h)).abs() < 1e-6,
                "node {i}: kernel {} vs reference {}",
                probs[i],
                sigmoid(h)
            );
        }
    }

    #[test]
    fn isolated_node_follows_only_its_intrinsic_field() {
        let g = SignedGraph::from_edges(2, &[]).unwrap();
        let mut probs = [0.0f32; 2];
        signed_coupling_update_into(
            &g,
            &[1.0, -1.0],
            &Couplings::default(),
            &[2.0, 0.0],
            &mut probs,
        );
        assert!((probs[0] - sigmoid(2.0)).abs() < 1e-6);
        // g_i = 0 and no ties → maximally undecided. A softmax could not
        // express this; the sigmoid gives exactly 0.5.
        assert!((probs[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn zero_truth_gap_reproduces_the_base_couplings() {
        let g = small_graph();
        let base = Couplings {
            beta_plus: 1.3,
            beta_minus: 0.6,
            beta_zero: 0.9,
        };
        let split = InformedCouplings::from_couplings(&base, 0.0);
        let states = [1.0, -1.0, 1.0, 1.0];
        let intrinsic = [0.2, 0.0, -0.1, 0.4];

        let mut plain = [0.0f32; 4];
        let mut informed_true = [0.0f32; 4];
        let mut informed_false = [0.0f32; 4];
        signed_coupling_update_into(&g, &states, &base, &intrinsic, &mut plain);
        signed_coupling_update_informed_into(
            &g,
            &states,
            &[true; 4],
            &split,
            &intrinsic,
            &mut informed_true,
        );
        signed_coupling_update_informed_into(
            &g,
            &states,
            &[false; 4],
            &split,
            &intrinsic,
            &mut informed_false,
        );
        // At gap = 0 both forks collapse onto the base weights, so the
        // informed kernel agrees with the plain one regardless of κ. Not
        // asserted bit-identical: the two kernels sum a different number of
        // channels (4 vs 2), so the float accumulation order differs.
        for i in 0..4 {
            assert!(
                (plain[i] - informed_true[i]).abs() < 1e-6,
                "node {i} (κ=true)"
            );
            assert!(
                (plain[i] - informed_false[i]).abs() < 1e-6,
                "node {i} (κ=false)"
            );
        }
    }

    #[test]
    fn correct_neighbors_pull_harder_than_wrong_ones() {
        // One concordant tie. The neighbor holds +1 either way; the only
        // difference is whether that stance is the correct one.
        let g = SignedGraph::from_edges(2, &[(0, 1, 1)]).unwrap();
        let split = InformedCouplings::from_couplings(&Couplings::default(), 0.3);
        let states = [-1.0, 1.0];
        let intrinsic = [0.0, 0.0];
        let mut with_truth = [0.0f32; 2];
        let mut without = [0.0f32; 2];
        signed_coupling_update_informed_into(
            &g,
            &states,
            &[false, true],
            &split,
            &intrinsic,
            &mut with_truth,
        );
        signed_coupling_update_informed_into(
            &g,
            &states,
            &[false, false],
            &split,
            &intrinsic,
            &mut without,
        );
        assert!(
            with_truth[0] > without[0],
            "an informed ally must pull harder: {} vs {}",
            with_truth[0],
            without[0]
        );
    }

    #[test]
    fn social_temperature_scales_the_field_not_the_shape() {
        let c = Couplings::default();
        let hot = c.at_social_temperature(100.0);
        let g = small_graph();
        let states = [1.0, 1.0, 1.0, 1.0];
        let mut probs = [0.0f32; 4];
        signed_coupling_update_into(&g, &states, &hot, &[0.0; 4], &mut probs);
        // High temperature ⇒ couplings vanish ⇒ every entity is undecided.
        for p in probs {
            assert!(
                (p - 0.5).abs() < 0.05,
                "hot crowd should be indifferent, got {p}"
            );
        }
        // Cold ⇒ the same all-+1 crowd locks in.
        let cold = c.at_social_temperature(0.1);
        signed_coupling_update_into(&g, &states, &cold, &[0.0; 4], &mut probs);
        for p in probs {
            assert!(p > 0.99, "cold crowd should commit, got {p}");
        }
        // t <= 0 must not divide by zero.
        assert!(c.at_social_temperature(0.0).beta_plus.is_finite());
    }

    #[test]
    fn order_parameters_separate_the_three_regimes() {
        // Magnitude-weighted stances (the recommended consumer path).
        let indifference = [0.05, -0.03, 0.02, -0.04];
        let polarization = [1.0, 1.0, -1.0, -1.0];
        let consensus = [0.95, 1.0, 0.9, 1.0];

        assert!(net_opinion(&indifference).abs() < 0.1);
        assert!(crowd_conviction(&indifference) < 0.01);

        assert!(net_opinion(&polarization).abs() < 1e-6);
        assert!(crowd_conviction(&polarization) > 0.99);

        assert!(net_opinion(&consensus) > 0.9);
        assert!(crowd_conviction(&consensus) > 0.8);

        // Empty crowd is defined, not a NaN.
        assert_eq!(net_opinion(&[]), 0.0);
        assert_eq!(crowd_conviction(&[]), 0.0);
    }

    #[test]
    fn susceptibility_is_welford_variance_times_n() {
        let mut acc = SusceptibilityAccumulator::new();
        assert_eq!(acc.variance(), 0.0);
        acc.observe(0.2);
        // One sample has no variance yet.
        assert_eq!(acc.variance(), 0.0);
        for x in [-0.4, 0.6, -0.8] {
            acc.observe(x);
        }
        // |n| stream: 0.2, 0.4, 0.6, 0.8 → mean 0.5, sample var 0.0666…
        assert_eq!(acc.count(), 4);
        assert!((acc.mean_abs_net() - 0.5).abs() < 1e-6);
        assert!((acc.variance() - 0.066_666_7).abs() < 1e-5);
        assert!((acc.susceptibility(32) - 32.0 * acc.variance()).abs() < 1e-5);

        acc.reset();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn sampling_thresholds_uniforms_against_probabilities() {
        let probs = [0.9, 0.1, 0.5];
        let uniforms = [0.5, 0.5, 0.5];
        let mut states = [0.0f32; 3];
        sample_states_into(&probs, &uniforms, &mut states);
        // 0.5 < 0.9 → +1; 0.5 !< 0.1 → −1; ties resolve to −1 (strict <).
        assert_eq!(states, [1.0, -1.0, -1.0]);
    }

    // ── verdict_margin (the one-snapshot manipulability forecast) ──

    #[test]
    fn verdict_margin_uniform_weights_matches_abs_net_opinion() {
        // 6 for, 2 against on ±1 states → |mean(s)| = 0.5.
        let states = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0];
        let weights = [1.0; 8];
        assert!((verdict_margin(&weights, &states) - net_opinion(&states).abs()).abs() < 1e-6);
        assert!((verdict_margin(&weights, &states) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn verdict_margin_uses_stance_sign_not_magnitude() {
        // Graded stances: only the sign votes.
        let states = [0.9, 0.05, -0.8, -0.3];
        let weights = [1.0; 4];
        assert!((verdict_margin(&weights, &states) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn verdict_margin_weighted_asymmetry_dominates() {
        // 7 agents at −1 with weight 1 + 1 reliable scout at +1 with weight 9
        // → the scout's vote dominates: margin = |(9 − 7)/16| = 0.125.
        let states = [-1.0; 7];
        let weights = [1.0; 7];
        let states = {
            let mut s = states.to_vec();
            s.push(1.0);
            s
        };
        let weights = {
            let mut w = weights.to_vec();
            w.push(9.0);
            w
        };
        assert!((verdict_margin(&weights, &states) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn verdict_margin_degenerate_inputs_return_zero() {
        // Empty crowd + zero weight total both read 0.0 (the reducer
        // convention) — not a division error.
        assert_eq!(verdict_margin(&[], &[]), 0.0);
        let states = [1.0, -1.0];
        assert_eq!(verdict_margin(&[0.0, 0.0], &states), 0.0);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn verdict_margin_length_mismatch_panics() {
        let _ = verdict_margin(&[1.0, 1.0], &[1.0]);
    }
}
