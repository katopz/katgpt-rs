//! Chance-node PUCT for single-player stochastic games (katgpt-rs Issue 892
//! T3) — the moka trick (Bench 205: 74% → 98% vs greedy Moka) transplanted
//! from two-player Go to games with dice / piece draws.
//!
//! # The recipe (what transfers from Bench 205, what changes)
//!
//! * **Policy prior pruned to `top_k`, value at the leaf, no rollouts,
//!   most-visited root.** Unchanged from moka. UCB1 without a prior measured
//!   ≈0% there (`katgpt-core::mcts` is that variant and is untouched here).
//! * **Prior = sigmoid, never softmax** (owner rule). Over the kept `top_k`
//!   raw scores `s_i` with mean `m`: `p_i = σ((s_i − m)/T) / Σ_kept σ(·)`.
//!   `T = prior_temp` when `> 0`, else the population std of the kept scores
//!   (scale-free default — the caller's score units never have to be known).
//! * **Value squashed into (0,1), NO sign flip** (single player — the moka
//!   negamax flip is exactly the Q-sign bug class this must not inherit):
//!   `v01 = σ((v − ref)/scale)`, `ref` = the ROOT state's `value()` (so 0.5
//!   means "as good as where we stand now"; a deeper state that improved on
//!   the root reads > 0.5), `scale = value_scale` when `> 0`, else the std of
//!   the root's kept prior scores (correct when prior scores and values share
//!   a scale — the 1-ply-eval prior every board game adapter here uses).
//!   A **terminal** state (`is_terminal()` — loss / top-out) backs up `0.0`.
//!   `Q = mean backed-up v01`.
//! * **Chance nodes SAMPLE one outcome per simulation, proportional to its
//!   probability**, from a caller-seeded `fastrand::Rng` (deterministic for a
//!   fixed seed; never the unseeded global). Chosen over full expansion
//!   because full expansion multiplies every simulation's cost by the chance
//!   branching factor (7 for a Tetris bag) while the mean backup of sampled
//!   outcomes already converges to `Σ p_i Q_i` — the expectation, not the
//!   max, which is the property the G1 gamble test pins. Sampling also keeps
//!   the "one decision expansion per simulation" cost model of moka: a chance
//!   node expands for free (its outcome children are created, not evaluated)
//!   and the same simulation continues into the sampled child.
//! * **Root decision:** most-visited child; ties → higher Q → lower original
//!   action index.
//!
//! # Contract ([`ChanceGame`])
//!
//! A state is either a *decision* state (the player moves) or a *chance*
//! state (the environment draws) — the searcher tracks which, the game only
//! answers the question asked of it:
//! * decision: [`actions`](ChanceGame::actions) → `(action, raw prior score)`
//!   in the game's canonical order (the "original index" the result
//!   reports); [`apply`](ChanceGame::apply) → the post-decision (chance)
//!   state. No actions + `is_terminal()` ⇒ loss (0). No actions and NOT
//!   terminal ⇒ a static leaf (end of game with a score): its squashed
//!   `value()` is backed up on every visit.
//! * chance: [`outcomes`](ChanceGame::outcomes) → `(outcome, probability)`
//!   (a single outcome with `p = 1` for a known draw, e.g. a revealed preview
//!   piece); [`resolve`](ChanceGame::resolve) → the next decision state.
//!   `is_terminal()` is consulted when a chance state is first expanded;
//!   no outcomes ⇒ static leaf.
//!
//! # Allocation
//!
//! The searcher owns an arena (`Vec` of nodes, children contiguous) plus
//! scratch buffers, all `clear()`-reused across [`ChancePuct::search`]
//! calls: after the first search at a given budget the search loop itself
//! allocates nothing (pinned by the G4 arm of `bench_892_chance_puct_goat`
//! on a heap-free game). A game whose `actions`/`apply` allocate still does.

use crate::exact_sigmoid;

/// A single-player stochastic game, as the chance-node PUCT sees it.
pub trait ChanceGame: Clone {
    /// A decision edge. Copy so the arena can hold it inline.
    type Action: Copy;
    /// A chance edge.
    type Outcome: Copy;

    /// `true` iff this state is a LOSS (backs up 0). Consulted for decision
    /// states with no actions, and for chance states when first expanded.
    fn is_terminal(&self) -> bool;
    /// Raw leaf value, caller scale (higher = better). Squashed by the
    /// searcher; never negated.
    fn value(&self) -> f32;
    /// Decision state: push every legal `(action, raw prior score)`, in the
    /// game's canonical order. `out` arrives cleared.
    fn actions(&self, out: &mut Vec<(Self::Action, f32)>);
    /// Decision state + action → post-decision (chance) state.
    fn apply(&self, action: Self::Action) -> Self;
    /// Chance state: push every `(outcome, probability)`; probabilities need
    /// not be normalised (sampling divides by their sum). `out` arrives
    /// cleared.
    fn outcomes(&self, out: &mut Vec<(Self::Outcome, f32)>);
    /// Chance state + outcome → next decision state.
    fn resolve(&self, outcome: Self::Outcome) -> Self;
}

/// Search configuration. Defaults are the moka Bench 205 row
/// (`c_puct = 1.5`, `top_k = 8`) at budget 400.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChancePuctConfig {
    /// Simulations per decision; each expands at most one decision node.
    pub budget: u32,
    /// Exploration constant in `Q + c·P·√N/(1+n)`.
    pub c_puct: f32,
    /// Keep only the `top_k` highest-prior actions at every decision node.
    pub top_k: usize,
    /// Prior sigmoid temperature; `<= 0` ⇒ std of the kept scores.
    pub prior_temp: f32,
    /// Value squash scale; `<= 0` ⇒ std of the root's kept prior scores.
    pub value_scale: f32,
}

impl Default for ChancePuctConfig {
    fn default() -> Self {
        Self {
            budget: 400,
            c_puct: 1.5,
            top_k: 8,
            prior_temp: 0.0,
            value_scale: 0.0,
        }
    }
}

/// The root decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChancePuctPick<A> {
    /// The chosen action.
    pub action: A,
    /// Its index in the root's [`ChanceGame::actions`] order.
    pub index: usize,
    /// Its visit count (0 when the root had a single legal action or the
    /// budget was 0 — the highest-prior action is returned unsearched).
    pub visits: u32,
    /// Its mean backed-up value in (0,1) (0.5 when unvisited).
    pub q: f32,
}

/// One root child, for diagnostics / tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootStat {
    /// Index in the root's action order.
    pub index: usize,
    /// Normalised sigmoid prior.
    pub prior: f32,
    pub visits: u32,
    /// Mean backed-up value (0.5 when unvisited).
    pub q: f32,
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Decision,
    Chance,
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Fresh,
    Expanded,
    /// Loss — backs up 0.
    Terminal,
    /// Static leaf — backs up `leaf_value`.
    Leaf,
}

struct Node<G: ChanceGame> {
    state: G,
    /// Edge action (decision children only).
    action: Option<G::Action>,
    total: f32,
    /// Prior (child of a decision node) or probability (child of a chance
    /// node).
    weight: f32,
    leaf_value: f32,
    visits: u32,
    first_child: u32,
    n_children: u32,
    /// Original action index (decision children).
    index: u32,
    kind: Kind,
    status: Status,
}

impl<G: ChanceGame> Node<G> {
    #[inline]
    fn q_or(&self, fpu: f32) -> f32 {
        if self.visits == 0 {
            fpu
        } else {
            self.total / self.visits as f32
        }
    }
}

/// Reusable chance-node PUCT searcher (arena + scratch owned, reused across
/// searches).
pub struct ChancePuct<G: ChanceGame> {
    cfg: ChancePuctConfig,
    nodes: Vec<Node<G>>,
    path: Vec<u32>,
    raw_actions: Vec<(G::Action, f32)>,
    ranked: Vec<(u32, G::Action, f32)>,
    raw_outcomes: Vec<(G::Outcome, f32)>,
    child_states: Vec<G>,
    value_ref: f32,
    value_scale: f32,
}

#[inline]
fn std_dev(xs: impl Iterator<Item = f32> + Clone) -> f32 {
    let mut n = 0u32;
    let mut sum = 0.0f64;
    for x in xs.clone() {
        n += 1;
        sum += x as f64;
    }
    if n == 0 {
        return 0.0;
    }
    let mean = sum / n as f64;
    let mut var = 0.0f64;
    for x in xs {
        let d = x as f64 - mean;
        var += d * d;
    }
    (var / n as f64).sqrt() as f32
}

/// A scale below this (all kept scores equal) falls back to 1.0.
const MIN_SCALE: f32 = 1e-6;

impl<G: ChanceGame> ChancePuct<G> {
    pub fn new(cfg: ChancePuctConfig) -> Self {
        Self {
            cfg,
            nodes: Vec::new(),
            path: Vec::new(),
            raw_actions: Vec::new(),
            ranked: Vec::new(),
            raw_outcomes: Vec::new(),
            child_states: Vec::new(),
            value_ref: 0.0,
            value_scale: 1.0,
        }
    }

    pub fn config(&self) -> &ChancePuctConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: ChancePuctConfig) {
        self.cfg = cfg;
    }

    /// Nodes in the last search's tree.
    pub fn tree_size(&self) -> usize {
        self.nodes.len()
    }

    /// The last search's root children, in prior-rank order.
    pub fn root_stats(&self) -> impl Iterator<Item = RootStat> + '_ {
        let (first, n) = match self.nodes.first() {
            Some(r) if r.status == Status::Expanded => {
                (r.first_child as usize, r.n_children as usize)
            }
            _ => (0, 0),
        };
        self.nodes[first..first + n].iter().map(|c| RootStat {
            index: c.index as usize,
            prior: c.weight,
            visits: c.visits,
            q: c.q_or(0.5),
        })
    }

    /// Search from a DECISION state. `None` ⇔ the root has no legal action
    /// (terminal / leaf — nothing to choose).
    pub fn search(
        &mut self,
        root: &G,
        rng: &mut fastrand::Rng,
    ) -> Option<ChancePuctPick<G::Action>> {
        self.nodes.clear();
        self.value_ref = root.value();
        self.value_scale = 1.0;
        self.nodes.push(Node {
            state: root.clone(),
            action: None,
            total: 0.0,
            weight: 1.0,
            leaf_value: 0.0,
            visits: 0,
            first_child: 0,
            n_children: 0,
            index: 0,
            kind: Kind::Decision,
            status: Status::Fresh,
        });
        let v_root = self.expand_decision(0, true);
        if self.nodes[0].status != Status::Expanded {
            return None;
        }
        let root_node = &mut self.nodes[0];
        root_node.visits = 1;
        root_node.total = v_root;

        let first = self.nodes[0].first_child as usize;
        let n = self.nodes[0].n_children as usize;
        let mut best = first; // highest prior (children are prior-ranked)
        if n > 1 && self.cfg.budget > 0 {
            for _ in 0..self.cfg.budget {
                self.simulate(rng);
            }
        } else {
            let b = &self.nodes[best];
            return Some(ChancePuctPick {
                action: b.action.expect("decision child carries its action"),
                index: b.index as usize,
                visits: b.visits,
                q: b.q_or(0.5),
            });
        }

        // Most-visited; ties → higher Q → lower original index.
        for c in first + 1..first + n {
            let (a, b) = (&self.nodes[c], &self.nodes[best]);
            let better = match a.visits.cmp(&b.visits) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => {
                    let (qa, qb) = (a.q_or(0.5), b.q_or(0.5));
                    match qa.total_cmp(&qb) {
                        std::cmp::Ordering::Greater => true,
                        std::cmp::Ordering::Less => false,
                        std::cmp::Ordering::Equal => a.index < b.index,
                    }
                }
            };
            if better {
                best = c;
            }
        }
        let b = &self.nodes[best];
        Some(ChancePuctPick {
            action: b.action.expect("decision child carries its action"),
            index: b.index as usize,
            visits: b.visits,
            q: b.q_or(0.5),
        })
    }

    /// Expand a Fresh decision node; returns its squashed value (0 for a
    /// loss). Sets `status`.
    fn expand_decision(&mut self, idx: usize, is_root: bool) -> f32 {
        let Self {
            cfg,
            nodes,
            raw_actions,
            ranked,
            child_states,
            value_ref,
            value_scale,
            ..
        } = self;
        let st = &nodes[idx].state;
        raw_actions.clear();
        st.actions(raw_actions);
        if raw_actions.is_empty() {
            let (status, v) = if st.is_terminal() {
                (Status::Terminal, 0.0)
            } else {
                let v = exact_sigmoid((st.value() - *value_ref) / *value_scale);
                (Status::Leaf, v)
            };
            let node = &mut nodes[idx];
            node.status = status;
            node.leaf_value = v;
            return v;
        }
        ranked.clear();
        ranked.extend(
            raw_actions
                .iter()
                .enumerate()
                .map(|(i, &(a, s))| (i as u32, a, s)),
        );
        // Score desc, original index asc — deterministic under ties/NaN.
        ranked.sort_unstable_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)));
        ranked.truncate(cfg.top_k.max(1));

        let k = ranked.len() as f32;
        let mean = ranked.iter().map(|r| r.2).sum::<f32>() / k;
        let sd = std_dev(ranked.iter().map(|r| r.2));
        if is_root {
            *value_scale = if cfg.value_scale > 0.0 {
                cfg.value_scale
            } else if sd > MIN_SCALE {
                sd
            } else {
                1.0
            };
        }
        let temp = if cfg.prior_temp > 0.0 {
            cfg.prior_temp
        } else if sd > MIN_SCALE {
            sd
        } else {
            1.0
        };
        let mut z = 0.0f32;
        for r in ranked.iter_mut() {
            r.2 = exact_sigmoid((r.2 - mean) / temp);
            z += r.2;
        }
        let v = exact_sigmoid((st.value() - *value_ref) / *value_scale);

        child_states.clear();
        child_states.extend(ranked.iter().map(|r| st.apply(r.1)));
        let first = nodes.len() as u32;
        for (r, s) in ranked.iter().zip(child_states.drain(..)) {
            nodes.push(Node {
                state: s,
                action: Some(r.1),
                total: 0.0,
                weight: r.2 / z,
                leaf_value: 0.0,
                visits: 0,
                first_child: 0,
                n_children: 0,
                index: r.0,
                kind: Kind::Chance,
                status: Status::Fresh,
            });
        }
        let node = &mut nodes[idx];
        node.first_child = first;
        node.n_children = ranked.len() as u32;
        node.status = Status::Expanded;
        v
    }

    /// Expand a Fresh chance node (creates outcome children, evaluates
    /// nothing). Sets `status`.
    fn expand_chance(&mut self, idx: usize) {
        let Self {
            nodes,
            raw_outcomes,
            child_states,
            value_ref,
            value_scale,
            ..
        } = self;
        let st = &nodes[idx].state;
        if st.is_terminal() {
            nodes[idx].status = Status::Terminal;
            nodes[idx].leaf_value = 0.0;
            return;
        }
        raw_outcomes.clear();
        st.outcomes(raw_outcomes);
        if raw_outcomes.is_empty() {
            let v = exact_sigmoid((st.value() - *value_ref) / *value_scale);
            nodes[idx].status = Status::Leaf;
            nodes[idx].leaf_value = v;
            return;
        }
        child_states.clear();
        child_states.extend(raw_outcomes.iter().map(|&(o, _)| st.resolve(o)));
        let first = nodes.len() as u32;
        for (&(_, p), s) in raw_outcomes.iter().zip(child_states.drain(..)) {
            nodes.push(Node {
                state: s,
                action: None,
                total: 0.0,
                weight: p.max(0.0),
                leaf_value: 0.0,
                visits: 0,
                first_child: 0,
                n_children: 0,
                index: 0,
                kind: Kind::Decision,
                status: Status::Fresh,
            });
        }
        let node = &mut nodes[idx];
        node.first_child = first;
        node.n_children = raw_outcomes.len() as u32;
        node.status = Status::Expanded;
    }

    #[inline]
    fn select_puct(&self, idx: usize) -> usize {
        let p = &self.nodes[idx];
        let sqrt_n = (p.visits.max(1) as f32).sqrt();
        let fpu = p.q_or(0.5); // first-play urgency = parent's mean value
        let first = p.first_child as usize;
        let mut best = first;
        let mut best_s = f32::NEG_INFINITY;
        for c in first..first + p.n_children as usize {
            let ch = &self.nodes[c];
            let s = ch.q_or(fpu) + self.cfg.c_puct * ch.weight * sqrt_n / (1.0 + ch.visits as f32);
            if s > best_s {
                best_s = s;
                best = c;
            }
        }
        best
    }

    #[inline]
    fn sample_outcome(&self, idx: usize, rng: &mut fastrand::Rng) -> usize {
        let p = &self.nodes[idx];
        let first = p.first_child as usize;
        let n = p.n_children as usize;
        let total: f32 = self.nodes[first..first + n].iter().map(|c| c.weight).sum();
        let mut u = rng.f32() * total;
        for c in first..first + n {
            u -= self.nodes[c].weight;
            if u < 0.0 {
                return c;
            }
        }
        first + n - 1
    }

    fn simulate(&mut self, rng: &mut fastrand::Rng) {
        self.path.clear();
        let mut cur = 0usize;
        self.path.push(0);
        let v = loop {
            let node = &self.nodes[cur];
            match (node.status, node.kind) {
                (Status::Terminal, _) => break 0.0,
                (Status::Leaf, _) => break node.leaf_value,
                (Status::Fresh, Kind::Decision) => break self.expand_decision(cur, false),
                (Status::Fresh, Kind::Chance) => {
                    // Free expansion — continue into a sampled outcome in
                    // this same simulation.
                    self.expand_chance(cur);
                    continue;
                }
                (Status::Expanded, Kind::Decision) => cur = self.select_puct(cur),
                (Status::Expanded, Kind::Chance) => cur = self.sample_outcome(cur, rng),
            }
            self.path.push(cur as u32);
        };
        for &i in &self.path {
            let n = &mut self.nodes[i as usize];
            n.visits += 1;
            n.total += v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy game: decision states carry a menu; chance states carry an
    /// outcome table of static leaf values.
    #[derive(Clone, Debug)]
    enum Toy {
        /// `(prior score, outcome table)` per action.
        Menu(Vec<(f32, Vec<(f32, f32)>)>),
        /// Post-decision: `(leaf value, probability)`.
        After(Vec<(f32, f32)>),
        /// Static leaf with a value.
        Leaf(f32),
        /// A loss.
        Dead,
    }

    impl ChanceGame for Toy {
        type Action = usize;
        type Outcome = f32; // NaN encodes Dead

        fn is_terminal(&self) -> bool {
            matches!(self, Toy::Dead)
        }
        fn value(&self) -> f32 {
            match self {
                Toy::Leaf(v) => *v,
                _ => 0.0,
            }
        }
        fn actions(&self, out: &mut Vec<(usize, f32)>) {
            if let Toy::Menu(m) = self {
                out.extend(m.iter().enumerate().map(|(i, (s, _))| (i, *s)));
            }
        }
        fn apply(&self, a: usize) -> Self {
            match self {
                Toy::Menu(m) => Toy::After(m[a].1.clone()),
                _ => unreachable!(),
            }
        }
        fn outcomes(&self, out: &mut Vec<(f32, f32)>) {
            if let Toy::After(t) = self {
                out.extend(t.iter().copied());
            }
        }
        fn resolve(&self, o: f32) -> Self {
            if o.is_nan() { Toy::Dead } else { Toy::Leaf(o) }
        }
    }

    fn cfg(budget: u32) -> ChancePuctConfig {
        ChancePuctConfig {
            budget,
            value_scale: 1.0,
            ..Default::default()
        }
    }

    fn pick(game: &Toy, c: ChancePuctConfig, seed: u64) -> ChancePuctPick<usize> {
        let mut s = ChancePuct::new(c);
        s.search(game, &mut fastrand::Rng::with_seed(seed)).unwrap()
    }

    /// G1 — expectation, not max. Safe = 0 (p=1 chance). Gamble: max +6
    /// but EV −3.6 (0.2·6 − 0.8·6). The prior FAVOURS the gamble (score 3
    /// vs 0), so the search must overturn it. A max-backup or a flipped Q
    /// sign picks the gamble; a correct expectation picks Safe.
    #[test]
    fn gamble_is_judged_by_expectation_not_max() {
        let game = Toy::Menu(vec![
            (0.0, vec![(0.0, 1.0)]),
            (3.0, vec![(6.0, 0.2), (-6.0, 0.8)]),
        ]);
        for seed in 1..=5 {
            let p = pick(&game, cfg(400), seed);
            assert_eq!(p.index, 0, "seed {seed}: picked the gamble, q={}", p.q);
        }
    }

    /// Two-sided arm of the above: flip the gamble's odds (EV +3.6) and
    /// the same search must take it — the chance weighting decides, not a
    /// hard-coded preference for the safe action.
    #[test]
    fn favourable_gamble_is_taken() {
        let game = Toy::Menu(vec![
            (3.0, vec![(0.0, 1.0)]),
            (0.0, vec![(6.0, 0.8), (-6.0, 0.2)]),
        ]);
        for seed in 1..=5 {
            assert_eq!(pick(&game, cfg(400), seed).index, 1, "seed {seed}");
        }
    }

    /// Terminal = loss (0): an action whose every outcome dies loses to a
    /// mediocre leaf even when the prior loves it. And a root with no
    /// actions returns None.
    #[test]
    fn terminal_is_a_loss() {
        let game = Toy::Menu(vec![
            (-1.0, vec![(-2.0, 1.0)]),
            (5.0, vec![(f32::NAN, 0.5), (f32::NAN, 0.5)]),
        ]);
        let p = pick(&game, cfg(200), 7);
        assert_eq!(p.index, 0);
        let mut s = ChancePuct::new(cfg(50));
        assert!(
            s.search(&Toy::Dead, &mut fastrand::Rng::with_seed(1))
                .is_none()
        );
        assert!(
            s.search(&Toy::Leaf(3.0), &mut fastrand::Rng::with_seed(1))
                .is_none()
        );
        // Dead children were visited and backed up exactly 0.
        let s2 = {
            let mut s = ChancePuct::new(cfg(200));
            s.search(&game, &mut fastrand::Rng::with_seed(7));
            s.root_stats().collect::<Vec<_>>()
        };
        let dead = s2.iter().find(|r| r.index == 1).unwrap();
        assert!(dead.visits > 0);
        assert_eq!(dead.q, 0.0);
    }

    /// Top-k pruning is real: 20 actions, top_k 8 ⇒ 8 root children, and
    /// the best-VALUE action ranked last by prior is unreachable.
    #[test]
    fn top_k_prunes_low_prior_actions() {
        let mut menu: Vec<(f32, Vec<(f32, f32)>)> = (0..20)
            .map(|i| (20.0 - i as f32, vec![(0.0, 1.0)]))
            .collect();
        menu[19].1 = vec![(10.0, 1.0)]; // best value, worst prior
        let game = Toy::Menu(menu);
        let mut s = ChancePuct::new(cfg(300));
        let p = s.search(&game, &mut fastrand::Rng::with_seed(3)).unwrap();
        let stats: Vec<_> = s.root_stats().collect();
        assert_eq!(stats.len(), 8);
        assert!(stats.iter().all(|r| r.index < 8));
        assert_ne!(p.index, 19);
        // Sigmoid priors are normalised over the kept set and rank-ordered.
        let z: f32 = stats.iter().map(|r| r.prior).sum();
        assert!((z - 1.0).abs() < 1e-5);
        assert!(stats.windows(2).all(|w| w[0].prior >= w[1].prior));
        // Widening top_k makes it reachable — and chosen.
        let wide = ChancePuctConfig {
            top_k: 20,
            ..cfg(600)
        };
        assert_eq!(pick(&game, wide, 3).index, 19);
    }

    /// The returned child is the most visited; exact ties fall to higher Q
    /// then lower original index.
    #[test]
    fn most_visited_root_with_tie_break() {
        let game = Toy::Menu(vec![
            (1.0, vec![(0.5, 1.0)]),
            (1.0, vec![(0.2, 1.0)]),
            (1.0, vec![(0.9, 1.0)]),
        ]);
        let mut s = ChancePuct::new(cfg(300));
        let p = s.search(&game, &mut fastrand::Rng::with_seed(1)).unwrap();
        let max_v = s.root_stats().map(|r| r.visits).max().unwrap();
        assert_eq!(p.visits, max_v);
        assert_eq!(p.index, 2);
        // budget 3 over three equal priors: one visit each ⇒ highest Q wins.
        assert_eq!(pick(&game, cfg(3), 1).index, 2);
        // Identical twins: visits and Q tie ⇒ lower original index.
        let twins = Toy::Menu(vec![(1.0, vec![(0.3, 1.0)]), (1.0, vec![(0.3, 1.0)])]);
        assert_eq!(pick(&twins, cfg(2), 1).index, 0);
        // Budget 0 ⇒ the top-prior action, unsearched.
        let p0 = pick(&game, cfg(0), 1);
        assert_eq!((p0.index, p0.visits), (0, 0));
    }

    /// Same seed ⇒ identical visit vector and pick; the searcher is reusable.
    #[test]
    fn deterministic_under_seed_and_reuse() {
        let game = Toy::Menu(vec![
            (0.0, vec![(1.0, 0.5), (-1.0, 0.5)]),
            (0.5, vec![(2.0, 0.3), (-0.5, 0.7)]),
            (0.2, vec![(0.1, 1.0)]),
        ]);
        let run = |s: &mut ChancePuct<Toy>, seed| {
            let p = s
                .search(&game, &mut fastrand::Rng::with_seed(seed))
                .unwrap();
            (p, s.root_stats().collect::<Vec<_>>())
        };
        let mut a = ChancePuct::new(cfg(500));
        let mut b = ChancePuct::new(cfg(500));
        let ra = run(&mut a, 42);
        let _ = run(&mut b, 9); // dirty the arena first
        assert_eq!(ra, run(&mut b, 42));
        assert_eq!(ra, run(&mut a, 42));
    }
}
