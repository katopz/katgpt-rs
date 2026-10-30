//! Walk-proximity context ranker — the katgpt-core lift of the primitive
//! proven in riir-neuron-db Issue 640 (RT-J distill, arXiv:2610.17867).
//!
//! **Embedding-free relational-context ranking**: W short non-backtracking
//! random walks of length K from a query node over any graph the caller can
//! expose; rank visited nodes by visit count, then recency, then hash. The
//! mechanism — walk-visit-count proximity ranks relational context better
//! than embedding-cosine and beats plain BFS — was proven by the ndb T1 PoC
//! (riir-neuron-db bench 516: walk 0.965 p@10 vs cosine 0.492 dense,
//! transfers to the sparse-typed arm, strictly beats BFS), and the latency
//! surface is priced in riir-neuron-db bench 517 (defaults 0.07–0.20 ms/call,
//! FLAT in N up to 100k nodes — the walk visits W×K nodes regardless of
//! graph size, 50–147× under the 10 ms offline budget).
//!
//! The math is standalone over the [`WalkGraphView`] trait — the ranker
//! touches hashes + ticks through two methods and knows nothing about the
//! store behind them. Store adapters are consumer-side (riir-neuron-db owns
//! the NS-graph and KG-triple adapters; its `walk_rank` feature re-exports
//! this module at the same paths).
//!
//! # OFFLINE ONLY
//!
//! W×K steps of neighbor enumeration + hashing cost 0.07–0.20 ms/call at the
//! defaults (riir-neuron-db bench 517) — over a sub-50 µs online-query budget.
//! Call it from offline/batch retrieval only.
//!
//! # Design laws carried from the ndb landing
//!
//! - **Temporal guard filters BEFORE sampling** (T1-proven): an inadmissible
//!   neighbor never enters a choice set, so a future-dated insertion provably
//!   cannot perturb an as-of walk (the guard is enforced in the sampler, not
//!   at proposal time). Dangling hashes (no node → no tick) are skipped.
//! - **Determinism is total**: per-walk rng seeds derive from
//!   `BLAKE3(seed ‖ walk_index)`; the rank sort is a total order
//!   (count desc, tick desc, hash asc); same seed + same graph →
//!   byte-identical output.
//! - **No allocation per step**: the counts map and neighbor/candidate scratch
//!   buffers are created once per call and reused across steps; they still
//!   grow with the visited set (the G4 test bounds a defaults call at ≤ 16
//!   allocations against 256 steps).
//! - **Ranking is BY structure** (the load-bearing finding): re-scoring
//!   visited nodes by cosine collapses the walk into pool-widening — the ndb
//!   T1 measured that composite ≡ flat cosine exactly. This ranker ranks by
//!   VISIT COUNT; do not "improve" it back into a cosine sort.
//!
//! Modelless: BLAKE3 + splitmix64 + xorshift64 + integer counting — no
//! weights, no gradients, no training dependency.

use std::collections::{HashMap as StdHashMap, HashSet};

/// Neighbor + tick source for the walk sampler — the ONLY graph access the
/// ranker needs.
pub trait WalkGraphView {
    /// Append `hash`'s neighbors (ALL directions — the caller owns the edge
    /// semantics) into `out`. `out` is caller-owned scratch: cleared by the
    /// sampler, appended here, never reallocated per step. Duplicates are
    /// legal and act as visit weighting (relation-rich pairs rank higher).
    fn neighbors_of(&self, hash: &[u8; 32], out: &mut Vec<[u8; 32]>);

    /// Commit tick of `hash`. `None` = unknown/dangling hash — the sampler
    /// never walks into it (cannot verify an as-of cut, and a visit cannot be
    /// recency-ranked). An atemporal substrate may return `Some(0)` for
    /// known nodes (existence check only; the recency tiebreak then
    /// degenerates to hash-ascending, which keeps the total order).
    fn tick_of(&self, hash: &[u8; 32]) -> Option<u64>;
}

/// Walk-mode configuration. Defaults are the proven headline posture
/// (W=32, K=8 — the riir-neuron-db T1/T2 landing).
///
/// The T1 sensitivity sweep (riir-neuron-db bench 516) priced the budget:
/// the paper-faithful half-budget (W=16, K=8) retains 0.954 p@10 (−0.013
/// vs headline); the quarter-budget (W=8, K=4) drops 0.083 — recorded, do
/// not ship. Cost is linear in steps (riir-neuron-db bench 517: 3.4×/6.2×
/// cheaper respectively) — there is no latency reason to leave the defaults.
#[derive(Clone, Debug)]
pub struct WalkRankConfig {
    /// Number of walks from the query (W).
    pub walks: u32,
    /// Steps per walk (K).
    pub walk_len: u32,
    /// Temporal admissibility cut: a step may only traverse to a node whose
    /// tick ≤ `as_of`. `None` = no cut (every real node admissible).
    pub as_of: Option<u64>,
    /// Master seed — per-walk seeds derive from `BLAKE3(seed ‖ walk_index)`.
    /// Same seed + same graph ⇒ byte-identical ranked output.
    pub seed: u64,
}

impl Default for WalkRankConfig {
    fn default() -> Self {
        Self {
            walks: 32,
            walk_len: 8,
            as_of: None,
            seed: 0,
        }
    }
}

/// One ranked context candidate: a node the walks visited, with its visit
/// count and tick. Ranking: `visit_count` desc, then `tick` desc (recency),
/// then `node_hash` asc (total order — determinism).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalkRankedHit {
    pub node_hash: [u8; 32],
    pub visit_count: u32,
    pub tick: u64,
}

/// Stage-2 budgeted context builder configuration: consume ranked seeds in
/// order, each seed joins the context then expands with a bounded structural
/// traversal, shared nodes accounted once, until `max_nodes` fills.
#[derive(Clone, Copy, Debug)]
pub struct ContextBudget {
    /// Node budget for the built context (the query is never included).
    pub max_nodes: usize,
    /// Per-seed traversal depth (depth bound on the structural BFS seeded by
    /// each ranked hit). `0` = the ranked seeds themselves only.
    pub seed_depth: u8,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_nodes: 64,
            seed_depth: 2,
        }
    }
}

/// splitmix64 finalizer — mixes the BLAKE3-derived per-walk seed before it
/// drives the xorshift.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xorshift64 (Marsaglia) — the same shape the ndb T1 PoC used.
struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        Self(seed | 1) // nonzero
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

/// Per-walk rng seed: `BLAKE3(seed_le ‖ walk_index_le)[..8]`, splitmixed.
fn walk_seed(master: u64, walk_index: u32) -> u64 {
    let mut buf = [0u8; 12];
    buf[..8].copy_from_slice(&master.to_le_bytes());
    buf[8..].copy_from_slice(&walk_index.to_le_bytes());
    let digest = blake3::hash(&buf);
    let mut lo = [0u8; 8];
    lo.copy_from_slice(&digest.as_bytes()[..8]);
    splitmix64(u64::from_le_bytes(lo))
}

/// Stage 1 — visit-count ranker. W non-backtracking walks of length K from
/// `query`; rank visited nodes by count desc, tick desc, hash asc. The query
/// itself is never returned.
///
/// OFFLINE ONLY. Creates its scratch once per call and reuses it across steps.
pub fn walk_rank<V: WalkGraphView>(
    view: &V,
    query: &[u8; 32],
    cfg: &WalkRankConfig,
) -> Vec<WalkRankedHit> {
    let mut counts: StdHashMap<[u8; 32], u32> = StdHashMap::new();
    let mut neighbors: Vec<[u8; 32]> = Vec::new();
    let mut cands: Vec<[u8; 32]> = Vec::new();

    for w in 0..cfg.walks {
        let mut rng = XorShift::new(walk_seed(cfg.seed, w));
        let mut cur = *query;
        let mut prev: Option<[u8; 32]> = None;
        for _ in 0..cfg.walk_len {
            neighbors.clear();
            view.neighbors_of(&cur, &mut neighbors);
            cands.clear();
            for nb in neighbors.drain(..) {
                // Non-backtracking + guard + dangling — all filtered BEFORE
                // sampling (the T1-proven order: an inadmissible candidate
                // never perturbs the choice set, so future-dated insertions
                // cannot change an as-of walk).
                if Some(nb) == prev {
                    continue;
                }
                let tick = match view.tick_of(&nb) {
                    None => continue, // dangling — never walked into
                    Some(t) => t,
                };
                if let Some(cut) = cfg.as_of
                    && tick > cut
                {
                    continue;
                }
                cands.push(nb);
            }
            if cands.is_empty() {
                break;
            }
            let nxt = cands[rng.range(cands.len())];
            *counts.entry(nxt).or_insert(0) += 1;
            prev = Some(cur);
            cur = nxt;
        }
    }

    // Rank: count desc, tick desc, hash asc — a total order, so the output is
    // byte-identical across re-runs (no HashMap iteration order leaks).
    let mut ranked: Vec<([u8; 32], u32)> = counts.into_iter().collect();
    ranked.retain(|&(h, _)| h != *query);
    let ticks: StdHashMap<[u8; 32], u64> = ranked
        .iter()
        .filter_map(|&(h, _)| view.tick_of(&h).map(|t| (h, t)))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(ticks.get(&b.0).cmp(&ticks.get(&a.0)))
            .then(a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .map(|(h, c)| WalkRankedHit {
            tick: ticks
                .get(&h)
                .copied()
                .expect("visited nodes always have ticks"),
            node_hash: h,
            visit_count: c,
        })
        .collect()
}

/// Stage 2 — budgeted context builder. Consumes [`walk_rank`]'s output in
/// order; each hit JOINS the context (the ranked hits are the top candidates)
/// and seeds a bounded structural BFS of `seed_depth`; shared nodes are
/// accounted once (set semantics); stops at `max_nodes`. Returns the context
/// in discovery order (deterministic given the ranked order and the view's
/// stable neighbor order).
///
/// The query is never part of its own context — it is the caller's anchor.
/// It can be a walk-rank hit only as the walk's start, which [`walk_rank`]
/// already excludes.
pub fn budgeted_context<V: WalkGraphView>(
    view: &V,
    ranked: &[WalkRankedHit],
    query: &[u8; 32],
    budget: &ContextBudget,
) -> Vec<[u8; 32]> {
    let mut taken: HashSet<[u8; 32]> = HashSet::new();
    let mut context: Vec<[u8; 32]> = Vec::with_capacity(budget.max_nodes.min(1024));
    let mut frontier: Vec<([u8; 32], u8)> = Vec::new();
    let mut neighbors: Vec<[u8; 32]> = Vec::new();

    for hit in ranked {
        if context.len() >= budget.max_nodes {
            break;
        }
        // The seed itself is a context candidate (guard against a duplicate
        // insert if an earlier seed's expansion already took it).
        if !taken.contains(&hit.node_hash) {
            taken.insert(hit.node_hash);
            context.push(hit.node_hash);
        }
        // Bounded BFS from the seed — discovery order via a cursor'd Vec.
        frontier.clear();
        frontier.push((hit.node_hash, 0u8));
        let mut cursor = 0usize;
        while cursor < frontier.len() && context.len() < budget.max_nodes {
            let (cur, depth) = frontier[cursor];
            cursor += 1;
            if depth >= budget.seed_depth {
                continue;
            }
            neighbors.clear();
            view.neighbors_of(&cur, &mut neighbors);
            for nb in neighbors.drain(..) {
                if nb == *query || taken.contains(&nb) || view.tick_of(&nb).is_none() {
                    continue;
                }
                taken.insert(nb);
                context.push(nb);
                if context.len() >= budget.max_nodes {
                    break;
                }
                frontier.push((nb, depth + 1));
            }
        }
    }
    context
}

// ────────────────────────── tests ──────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test hash with an embedded index in `hash[..8]` for assertions.
    fn h(index: u64) -> [u8; 32] {
        let mut z = [0u8; 32];
        z[..8].copy_from_slice(&index.to_le_bytes());
        z
    }

    /// Synthetic view: adjacency + ticks in plain maps — the toy-graph shape
    /// the ndb T1 PoC used, shrunk to the fixture scale. Fixture: A(t=100) →
    /// B(110), C(120); C sibling-lists B; B → D(130); D sibling-lists E(105),
    /// plus a far-future F(200) under A for the guard arm. Edge registration
    /// is BIDIRECTIONAL (the ndb `ExperienceGraph::insert` semantics: listed
    /// edges are traversable both ways).
    struct ToyView {
        adj: StdHashMap<[u8; 32], Vec<[u8; 32]>>,
        ticks: StdHashMap<[u8; 32], u64>,
    }

    impl ToyView {
        fn link(&mut self, a: [u8; 32], b: [u8; 32]) {
            self.adj.entry(a).or_default().push(b);
            self.adj.entry(b).or_default().push(a);
        }
    }

    impl WalkGraphView for ToyView {
        fn neighbors_of(&self, hash: &[u8; 32], out: &mut Vec<[u8; 32]>) {
            if let Some(nbs) = self.adj.get(hash) {
                out.extend_from_slice(nbs);
            }
        }

        fn tick_of(&self, hash: &[u8; 32]) -> Option<u64> {
            self.ticks.get(hash).copied()
        }
    }

    /// A(t=100); B(110), C(120); B→D(130); D lists E(105); F(200, future).
    fn build() -> ToyView {
        let mut v = ToyView {
            adj: StdHashMap::new(),
            ticks: StdHashMap::new(),
        };
        for (n, t) in [
            (h(1), 100u64),
            (h(2), 110),
            (h(3), 120),
            (h(4), 130),
            (h(5), 105),
        ] {
            v.ticks.insert(n, t);
        }
        v.ticks.insert(h(6), 200); // F — future
        // A—B, A—C, C—B (sibling listing), B—D, D—E, A—F.
        v.link(h(1), h(2));
        v.link(h(1), h(3));
        v.link(h(3), h(2));
        v.link(h(2), h(4));
        v.link(h(4), h(5));
        v.link(h(1), h(6));
        v
    }

    /// Determinism: same seed + same view ⇒ byte-identical ranked output;
    /// the query itself never ranks; the walk reaches the graph.
    #[test]
    fn deterministic_reruns() {
        let view = build();
        let cfg = WalkRankConfig {
            walks: 64,
            walk_len: 8,
            as_of: None,
            seed: 42,
        };
        let query = h(1);
        let r1 = walk_rank(&view, &query, &cfg);
        let r2 = walk_rank(&view, &query, &cfg);
        assert_eq!(r1, r2, "same seed must re-run byte-identically");
        assert!(!r1.is_empty(), "walks must reach the graph");
        assert!(r1.iter().all(|hit| hit.node_hash != query));
    }

    /// Temporal guard: zero hits past the as-of cut; a future-dated edge
    /// leaves the as-of ranking byte-identical (guard filters BEFORE
    /// sampling); non-vacuous (the ranking is non-empty).
    #[test]
    fn temporal_guard_leak_free_and_insert_invariant() {
        let mut view = build();
        // A second future node G(300) under B, added AFTER the ranking is
        // captured — the invariant: the as-of ranking is unchanged.
        let cfg = WalkRankConfig {
            walks: 64,
            walk_len: 8,
            as_of: Some(130),
            seed: 7,
        };
        let query = h(1);
        let r1 = walk_rank(&view, &query, &cfg);
        assert!(!r1.is_empty(), "as-of walks must reach the graph");
        view.ticks.insert(h(7), 300);
        view.link(h(2), h(7));
        let r2 = walk_rank(&view, &query, &cfg);
        for hit in &r2 {
            assert!(
                hit.tick <= 130,
                "TEMPORAL LEAK: hit tick {} > as-of 130",
                hit.tick
            );
        }
        assert!(
            r2.iter()
                .all(|hit| hit.node_hash != h(6) && hit.node_hash != h(7))
        );
        assert_eq!(
            r1, r2,
            "future-dated insertions must not perturb the as-of ranking"
        );
    }

    /// Budget accounting: exact under a large budget (every reachable node,
    /// each exactly once), capped under a small one, query never included.
    #[test]
    fn budget_accounting_exact_and_capped() {
        let view = build();
        let cfg = WalkRankConfig {
            walks: 64,
            walk_len: 8,
            as_of: None,
            seed: 9,
        };
        let query = h(1);
        let ranked = walk_rank(&view, &query, &cfg);
        assert!(!ranked.is_empty());

        // Exact: the always-reached set {B, C, D, E} is taken, once each. F is
        // not asserted: its only neighbor is the query, which the stage-2 BFS
        // skips, so F enters the context only if a walk happened to rank it.
        let budget = ContextBudget {
            max_nodes: 64,
            seed_depth: 2,
        };
        let ctx = budgeted_context(&view, &ranked, &query, &budget);
        let mut uniq = ctx.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(ctx.len(), uniq.len(), "no duplicates in the context");
        for expected in [h(2), h(3), h(4), h(5)] {
            assert!(
                ctx.contains(&expected),
                "reachable node {expected:?} must be taken"
            );
        }
        assert!(
            !ctx.contains(&query),
            "query never appears in its own context"
        );

        // Capped: max_nodes 2 → exactly 2 nodes.
        let small = ContextBudget {
            max_nodes: 2,
            seed_depth: 2,
        };
        let ctx2 = budgeted_context(&view, &ranked, &query, &small);
        assert_eq!(ctx2.len(), 2, "budget cap binds exactly");
    }

    /// Dangling neighbors (no tick behind them) are never walked into and
    /// never rank — and the walk from a dangling key produces nothing.
    #[test]
    fn dangling_neighbors_skipped() {
        let mut view = build();
        // A dangling listing: node B lists h(999), which has no tick.
        view.adj.entry(h(2)).or_default().push(h(999));
        let cfg = WalkRankConfig {
            walks: 32,
            walk_len: 8,
            as_of: None,
            seed: 3,
        };
        let ranked = walk_rank(&view, &h(1), &cfg);
        assert!(
            ranked.iter().all(|hit| hit.node_hash != h(999)),
            "dangling hash must never rank"
        );
        // The dangling key itself as the query: no adjacency behind it →
        // nothing to walk → empty ranking.
        assert!(walk_rank(&view, &h(999), &cfg).is_empty());
    }

    /// G4 (alloc): the scratch is ONCE-PER-CALL, never per-step — one
    /// stage-1 + one stage-2 call must allocate O(config), not O(W·K).
    /// Runs wherever the counting allocator exists (the Issue-741 gate:
    /// debug builds, or the profile-free `alloc_tracking` opt-in).
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    #[test]
    fn alloc_per_call_is_bounded_not_per_step() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};
        let view = build();
        let cfg = WalkRankConfig::default();
        let q = h(1);
        let warm = walk_rank(&view, &q, &cfg); // warm maps outside the window
        assert!(!warm.is_empty());
        let budget = ContextBudget::default();

        reset_alloc_stats();
        let ranked = walk_rank(&view, &q, &cfg);
        let (stage1, _) = get_alloc_stats();
        reset_alloc_stats();
        let _ = budgeted_context(&view, &ranked, &q, &budget);
        let (stage2, _) = get_alloc_stats();

        // 256 walk steps at defaults: per-step allocation would be 256+.
        // The once-per-call law lands in the low teens even with map growth.
        assert!(
            stage1 <= 16,
            "stage-1 must allocate once-per-call, got {stage1} (steps = 256)"
        );
        assert!(
            stage2 <= 16,
            "stage-2 must allocate once-per-call, got {stage2}"
        );
    }

    /// The hub-dominance mechanism (the ndb T1 finding, fixture-scale): a
    /// node every cycle crosses strictly dominates on visit COUNT — the
    /// ranking is by structure, never by a tiebreak accident.
    #[test]
    fn hub_dominates_on_visit_count() {
        let mut view = ToyView {
            adj: StdHashMap::new(),
            ticks: StdHashMap::new(),
        };
        // entry a→b; b fans to c AND d; both back-edge to b.
        for (n, t) in [(h(10), 1u64), (h(11), 2), (h(12), 3), (h(13), 4)] {
            view.ticks.insert(n, t);
        }
        view.link(h(10), h(11)); // a—b
        view.link(h(11), h(12)); // b—c
        view.link(h(11), h(13)); // b—d
        let cfg = WalkRankConfig {
            walks: 64,
            walk_len: 8,
            as_of: None,
            seed: 17,
        };
        let ranked = walk_rank(&view, &h(10), &cfg);
        assert_eq!(ranked[0].node_hash, h(11), "the hub must rank first");
        let count_of = |hh: [u8; 32]| {
            ranked
                .iter()
                .find(|hit| hit.node_hash == hh)
                .map(|hit| hit.visit_count)
                .unwrap_or(0)
        };
        let b = count_of(h(11));
        assert!(
            b > count_of(h(12)) && b > count_of(h(13)),
            "the hub must dominate on visit count, not tiebreak"
        );
    }
}
