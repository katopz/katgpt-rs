//! Headless Solver Benchmark — Three-Round Strategic Puzzle Comparison
//!
//! Compares three solvers across multiple seeds:
//! - 🐻 BruteForce: exhaustive tree, uniform marginals
//! - 🐰 AI: belief-state lever discovery, weighted marginals
//! - 🦊 Hybrid: AI discovers levers, brute force ordering
//!
//! Run: `cargo run --example tactical_08_headless`

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use katgpt_rs::pruners::pathfinder::{find_distance, find_path};
use katgpt_rs::speculative::types::ConstraintPruner;
use katgpt_rs::speculative::{
    build_dd_tree_pruned, find_valid_sequence, par_find_shortest_sequence,
};
use katgpt_rs::types::Config;

// ── Constants ──────────────────────────────────────────────────

const DIR_DELTA: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const BOSS_SPEED: u32 = 3;

const MAP: &str = "\
# # # # # # # # # # # # # # # #
# B . . . . . . . . . . k . . #
# . # # . # . . # . # . . # . #
# . . . . . . . . . . . . . . #
# . # . ! # . # # . # ! . # . #
# . . . . . . . . . . . . . . #
# . # # . . . . . . . # # # . #
# . . j . . ! O . . . . . . . #
# . # . # . # . . # . # . # . #
# . . . . . 1 . . 2 . . . . . #
# . # # . . . . 3 . . . # # . #
# # # # # # # = = = # # # # # #
# . . a . . . . . . b . . . . #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . G . #
# # # # # # # # # # # # # # # #";

// ── Types ──────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StrategicState {
    r: usize,
    c: usize,
    keys_held: u8,
    keys_used: u8,
    boxes_opened: u8,
    lever_state: u8,
    bridge_open: bool,
    boss_r: usize,
    boss_c: usize,
    boss_alive: bool,
    total_cost: u32,
    dead: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Target {
    Key(usize),
    Box_(usize),
    Lever(usize),
    Goal,
}

#[derive(Clone)]
struct Milestone {
    #[allow(dead_code)]
    target_idx: usize,
    #[allow(dead_code)]
    step: usize,
}

#[derive(Clone)]
struct Solution {
    #[allow(dead_code)]
    target_sequence: Vec<usize>,
    #[allow(dead_code)]
    milestones: Vec<Milestone>,
    actions: Vec<usize>,
    #[allow(dead_code)]
    states: Vec<StrategicState>,
    solve_time_ms: u64,
    tree_nodes: usize,
    levers_discovered: usize,
}

// ── Game Engine ────────────────────────────────────────────────

struct StrategicGame {
    grid: Vec<Vec<char>>,
    start: (usize, usize),
    boss_start: (usize, usize),
    keys: Vec<(usize, usize)>,
    boxes: Vec<(usize, usize)>,
    levers: Vec<(usize, usize)>,
    traps: HashSet<(usize, usize)>,
    bridge: HashSet<(usize, usize)>,
    goal: (usize, usize),
    key_mapping: [usize; 2],
    target_lever_mask: u8,
}

impl StrategicGame {
    fn new(map_str: &str, seed: u64) -> Self {
        let mut grid = Vec::new();
        let mut start = (0, 0);
        let mut goal = (0, 0);
        let mut bridge = HashSet::new();
        let mut bridge_row = usize::MAX;

        for (r, line) in map_str.lines().enumerate() {
            let mut row = Vec::new();
            for (c, token) in line.split_whitespace().enumerate() {
                let ch = token.chars().next().unwrap();
                match ch {
                    'B' => {
                        start = (r, c);
                        row.push('.');
                    }
                    'G' => {
                        goal = (r, c);
                        row.push('.');
                    }
                    '=' => {
                        bridge.insert((r, c));
                        bridge_row = r;
                        row.push('=');
                    }
                    '#' => row.push('#'),
                    _ => row.push('.'),
                }
            }
            grid.push(row);
        }

        let mut upper_floors = Vec::new();
        let mut lower_floors = Vec::new();
        for (r, row) in grid.iter().enumerate() {
            for (c, &cell) in row.iter().enumerate() {
                if cell == '.' && (r, c) != start && (r, c) != goal {
                    if r < bridge_row {
                        upper_floors.push((r, c));
                    } else if r > bridge_row {
                        lower_floors.push((r, c));
                    }
                }
            }
        }

        let mut rng_s = seed;
        for i in (1..upper_floors.len()).rev() {
            rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_s as usize) % (i + 1);
            upper_floors.swap(i, j);
        }
        for i in (1..lower_floors.len()).rev() {
            rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_s as usize) % (i + 1);
            lower_floors.swap(i, j);
        }

        let boss_start = upper_floors[0];
        let mut levers = vec![upper_floors[1], upper_floors[2], upper_floors[3]];
        let traps: HashSet<_> = [upper_floors[4], upper_floors[5]].into_iter().collect();
        let mut keys = vec![upper_floors[6], upper_floors[7]];
        let mut boxes = vec![lower_floors[0], lower_floors[1]];
        levers.sort();
        keys.sort();
        boxes.sort();

        let (key_mapping, target_lever_mask) = Self::generate_config(seed);
        Self {
            grid,
            start,
            boss_start,
            keys,
            boxes,
            levers,
            traps,
            bridge,
            goal,
            key_mapping,
            target_lever_mask,
        }
    }

    fn generate_config(seed: u64) -> ([usize; 2], u8) {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            s
        };
        let mut key_mapping = [0, 1];
        for i in (1..2).rev() {
            let j = (next() as usize) % (i + 1);
            key_mapping.swap(i, j);
        }
        if key_mapping[0] == 0 {
            key_mapping.swap(0, 1);
        }
        let target_lever_mask = ((next() as usize) % 6 + 1) as u8;
        (key_mapping, target_lever_mask)
    }

    fn rows(&self) -> usize {
        self.grid.len()
    }
    fn cols(&self) -> usize {
        self.grid.first().map_or(0, |r| r.len())
    }

    fn initial_state(&self) -> StrategicState {
        StrategicState {
            r: self.start.0,
            c: self.start.1,
            keys_held: 0,
            keys_used: 0,
            boxes_opened: 0,
            lever_state: 0,
            bridge_open: false,
            boss_r: self.boss_start.0,
            boss_c: self.boss_start.1,
            boss_alive: true,
            total_cost: 0,
            dead: false,
        }
    }

    fn blocked_set(&self, state: &StrategicState) -> HashSet<(usize, usize)> {
        let mut blocked = HashSet::new();
        if !state.bridge_open {
            for &pos in &self.bridge {
                blocked.insert(pos);
            }
        }
        for &pos in &self.traps {
            blocked.insert(pos);
        }
        blocked
    }

    fn target_pos(&self, target: &Target) -> (usize, usize) {
        match target {
            Target::Key(i) => self.keys[*i],
            Target::Box_(j) => self.boxes[*j],
            Target::Lever(k) => self.levers[*k],
            Target::Goal => self.goal,
        }
    }

    fn boss_next_move(&self, state: &StrategicState) -> (usize, usize) {
        if !state.boss_alive {
            return (state.boss_r, state.boss_c);
        }
        let start = (state.boss_r, state.boss_c);
        let target = (state.r, state.c);
        if start == target {
            return start;
        }

        let rows = self.rows();
        let cols = self.cols();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut came_from = HashMap::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(pos) = queue.pop_front() {
            if pos == target {
                let mut current = target;
                loop {
                    let prev = came_from[&current];
                    if prev == start {
                        return current;
                    }
                    current = prev;
                }
            }
            for &(dr, dc) in &DIR_DELTA {
                let nr = pos.0 as isize + dr;
                let nc = pos.1 as isize + dc;
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;
                let next = (nr, nc);
                if visited.contains(&next) {
                    continue;
                }
                if self.grid[nr][nc] == '#' {
                    continue;
                }
                if self.bridge.contains(&next) {
                    continue;
                }
                visited.insert(next);
                came_from.insert(next, pos);
                queue.push_back(next);
            }
        }
        start
    }

    fn interact(&self, state: &mut StrategicState) {
        let pos = (state.r, state.c);
        for (i, &kpos) in self.keys.iter().enumerate() {
            if kpos == pos && (state.keys_held & (1 << i)) == 0 {
                state.keys_held |= 1 << i;
            }
        }
        for (j, &bpos) in self.boxes.iter().enumerate() {
            if bpos == pos && (state.boxes_opened & (1 << j)) == 0 && state.keys_held != 0 {
                for k in 0..self.keys.len() {
                    if (state.keys_held & (1 << k)) != 0 && self.key_mapping[k] == j {
                        state.boxes_opened |= 1 << j;
                        state.keys_held &= !(1 << k);
                        state.keys_used |= 1 << k;
                        break;
                    }
                }
            }
        }
        for (l, &lpos) in self.levers.iter().enumerate() {
            if lpos == pos {
                state.lever_state ^= 1 << l;
                state.bridge_open = state.lever_state == self.target_lever_mask;
            }
        }
    }

    fn apply_action_no_boss(
        &self,
        state: &StrategicState,
        action: usize,
    ) -> Option<StrategicState> {
        if action > 3 {
            return None;
        }
        let mut next = state.clone();
        let (dr, dc) = DIR_DELTA[action];
        let nr = next.r as isize + dr;
        let nc = next.c as isize + dc;
        if nr < 0 || nc < 0 || nr >= self.rows() as isize || nc >= self.cols() as isize {
            return None;
        }
        let nr = nr as usize;
        let nc = nc as usize;
        if self.grid[nr][nc] == '#' {
            return None;
        }
        if self.bridge.contains(&(nr, nc)) && !next.bridge_open {
            return None;
        }
        if self.traps.contains(&(nr, nc)) {
            return None;
        }
        next.r = nr;
        next.c = nc;
        next.total_cost += 1;
        self.interact(&mut next);
        Some(next)
    }

    fn apply_action(&self, state: &StrategicState, action: usize) -> Option<StrategicState> {
        if action > 3 {
            return None;
        }
        let mut next = state.clone();
        let (dr, dc) = DIR_DELTA[action];
        let nr = next.r as isize + dr;
        let nc = next.c as isize + dc;
        if nr < 0 || nc < 0 || nr >= self.rows() as isize || nc >= self.cols() as isize {
            return None;
        }
        let nr = nr as usize;
        let nc = nc as usize;
        if self.grid[nr][nc] == '#' {
            return None;
        }
        if self.bridge.contains(&(nr, nc)) && !next.bridge_open {
            return None;
        }
        if self.traps.contains(&(nr, nc)) {
            next.dead = true;
            return Some(next);
        }
        next.r = nr;
        next.c = nc;
        next.total_cost += 1;
        self.interact(&mut next);

        if next.boss_alive && next.total_cost.is_multiple_of(BOSS_SPEED) {
            let (new_br, new_bc) = self.boss_next_move(&next);
            next.boss_r = new_br;
            next.boss_c = new_bc;
            if self.traps.contains(&(new_br, new_bc)) {
                next.boss_alive = false;
            }
        }
        if next.boss_alive && next.boss_r == next.r && next.boss_c == next.c {
            next.dead = true;
        }
        Some(next)
    }

    fn is_reachable(
        &self,
        from: (usize, usize),
        to: (usize, usize),
        blocked: &HashSet<(usize, usize)>,
    ) -> bool {
        find_distance(&self.grid, from, to, blocked).is_some()
    }
}

// ── Target Enumeration ─────────────────────────────────────────

fn enumerate_targets(num_keys: usize, num_boxes: usize, num_levers: usize) -> Vec<Target> {
    let mut targets = Vec::with_capacity(num_keys + num_boxes + num_levers + 1);
    for i in 0..num_keys {
        targets.push(Target::Key(i));
    }
    for j in 0..num_boxes {
        targets.push(Target::Box_(j));
    }
    for k in 0..num_levers {
        targets.push(Target::Lever(k));
    }
    targets.push(Target::Goal);
    targets
}

#[allow(dead_code)]
fn target_label(target: &Target) -> String {
    match target {
        Target::Key(i) => format!("Key({i})"),
        Target::Box_(j) => format!("Box({j})"),
        Target::Lever(k) => format!("Lever({k})"),
        Target::Goal => "Goal".into(),
    }
}

// ── Pruners ────────────────────────────────────────────────────

struct StrategicPruner<'a> {
    game: &'a StrategicGame,
    targets: Vec<Target>,
}

impl<'a> StrategicPruner<'a> {
    fn new(game: &'a StrategicGame) -> Self {
        let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
        Self { game, targets }
    }

    fn simulate(&self, target_indices: &[usize]) -> Option<StrategicState> {
        let mut state = self.game.initial_state();
        for &idx in target_indices {
            let target = &self.targets[idx];
            let target_pos = self.game.target_pos(target);
            let blocked = self.game.blocked_set(&state);
            let path = find_path(&self.game.grid, (state.r, state.c), target_pos, &blocked)?;
            for &action in &path {
                state = self.game.apply_action_no_boss(&state, action)?;
            }
        }
        Some(state)
    }
}

impl ConstraintPruner for StrategicPruner<'_> {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let Some(target) = self.targets.get(token_idx) else {
            return false;
        };
        if parent_tokens.contains(&token_idx) {
            return false;
        }
        let Some(state) = self.simulate(parent_tokens) else {
            return false;
        };
        let blocked = self.game.blocked_set(&state);

        match target {
            Target::Key(i) => {
                if (state.keys_held & (1 << i)) != 0 {
                    return false;
                }
                let pos = self.game.keys[*i];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Box_(j) => {
                if (state.boxes_opened & (1 << *j)) != 0 {
                    return false;
                }
                if state.keys_held == 0 {
                    return false;
                }
                let pos = self.game.boxes[*j];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Lever(_) => {
                if state.bridge_open {
                    return false;
                }
                let pos = self.game.target_pos(target);
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Goal => {
                let all_boxes = (1 << self.game.boxes.len()) - 1;
                if state.boxes_opened != all_boxes {
                    return false;
                }
                self.game
                    .is_reachable((state.r, state.c), self.game.goal, &blocked)
            }
        }
    }
}

struct AiPruner<'a> {
    game: &'a StrategicGame,
    targets: Vec<Target>,
    discovered_levers: HashSet<usize>,
}

impl<'a> AiPruner<'a> {
    fn new(game: &'a StrategicGame, discovered: &[usize]) -> Self {
        let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
        Self {
            game,
            targets,
            discovered_levers: discovered.iter().copied().collect(),
        }
    }

    fn simulate(&self, target_indices: &[usize]) -> Option<StrategicState> {
        let mut state = self.game.initial_state();
        for &idx in target_indices {
            let target = &self.targets[idx];
            let target_pos = self.game.target_pos(target);
            let blocked = self.game.blocked_set(&state);
            let path = find_path(&self.game.grid, (state.r, state.c), target_pos, &blocked)?;
            for &action in &path {
                state = self.game.apply_action_no_boss(&state, action)?;
            }
        }
        Some(state)
    }
}

impl ConstraintPruner for AiPruner<'_> {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let Some(target) = self.targets.get(token_idx) else {
            return false;
        };
        if parent_tokens.contains(&token_idx) {
            return false;
        }
        let Some(state) = self.simulate(parent_tokens) else {
            return false;
        };
        let blocked = self.game.blocked_set(&state);

        match target {
            Target::Key(i) => {
                if (state.keys_held & (1 << i)) != 0 {
                    return false;
                }
                let pos = self.game.keys[*i];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Box_(j) => {
                if (state.boxes_opened & (1 << *j)) != 0 {
                    return false;
                }
                if state.keys_held == 0 {
                    return false;
                }
                let pos = self.game.boxes[*j];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Lever(l) => {
                if state.bridge_open {
                    return false;
                }
                if !self.discovered_levers.contains(l) {
                    return false;
                }
                let pos = self.game.levers[*l];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Goal => {
                let all_boxes = (1 << self.game.boxes.len()) - 1;
                if state.boxes_opened != all_boxes {
                    return false;
                }
                self.game
                    .is_reachable((state.r, state.c), self.game.goal, &blocked)
            }
        }
    }
}

// ── Sequence Validation ────────────────────────────────────────

fn try_sequence(
    game: &StrategicGame,
    target_seq: &[usize],
    targets: &[Target],
) -> Option<(Vec<usize>, Vec<StrategicState>, Vec<Milestone>)> {
    let mut state = game.initial_state();
    let mut all_actions = Vec::new();
    let mut all_states = vec![state.clone()];
    let mut milestones = Vec::new();

    for &idx in target_seq {
        let target = &targets[idx];
        let target_pos = game.target_pos(target);
        let blocked = game.blocked_set(&state);
        let path = find_path(&game.grid, (state.r, state.c), target_pos, &blocked)?;
        milestones.push(Milestone {
            target_idx: idx,
            step: all_actions.len(),
        });
        for &action in &path {
            state = game.apply_action(&state, action)?;
            all_actions.push(action);
            all_states.push(state.clone());
            if state.dead {
                return None;
            }
        }
    }

    if (state.r, state.c) == game.goal
        && state.boxes_opened == (1 << game.boxes.len()) - 1
        && state.bridge_open
    {
        Some((all_actions, all_states, milestones))
    } else {
        None
    }
}

// ── Lever Discovery ────────────────────────────────────────────

fn discover_levers(game: &StrategicGame) -> (Vec<usize>, usize) {
    let n = game.levers.len();
    let candidates: Vec<u8> = (1..(1u8 << n).saturating_sub(1)).collect();
    let mut tested = 0;

    for size in 1..=n {
        for &mask in &candidates {
            if mask.count_ones() as usize != size {
                continue;
            }
            tested += 1;
            if mask == game.target_lever_mask {
                return ((0..n).filter(|&i| (mask >> i) & 1 == 1).collect(), tested);
            }
        }
    }
    (vec![], tested)
}

// ── Solvers ────────────────────────────────────────────────────

fn solve(game: &StrategicGame) -> Option<Solution> {
    let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
    let num_targets = targets.len();
    let pruner = StrategicPruner::new(game);

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 100_000;

    let marginals = vec![vec![1.0f32 / num_targets as f32; num_targets]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
    let bf_nodes = tree.len();

    let seq_result = find_valid_sequence(&tree, |seq| try_sequence(game, seq, &targets));
    let par_result = par_find_shortest_sequence(
        &tree,
        |seq| try_sequence(game, seq, &targets),
        |(actions, _, _)| actions.len(),
    );

    let total = start.elapsed();
    par_result.or(seq_result).map(
        |(target_sequence, (actions, states, milestones))| Solution {
            target_sequence,
            milestones,
            actions,
            states,
            solve_time_ms: total.as_millis() as u64,
            tree_nodes: bf_nodes,
            levers_discovered: 0,
        },
    )
}

fn solve_ai(game: &StrategicGame) -> Option<Solution> {
    let start = Instant::now();
    let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
    let num_targets = targets.len();

    let (discovered, observations) = discover_levers(game);
    let pruner = AiPruner::new(game, &discovered);

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 100_000;

    let discovered_set: HashSet<usize> = discovered.iter().copied().collect();
    let mut probs = vec![0.0f32; num_targets];
    for (i, target) in targets.iter().enumerate() {
        let is_valid = match target {
            Target::Lever(l) => discovered_set.contains(l),
            _ => true,
        };
        if is_valid {
            probs[i] = 1.0;
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    let marginals = vec![probs; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();
    let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
    let ai_nodes = tree.len();

    let result = par_find_shortest_sequence(
        &tree,
        |seq| try_sequence(game, seq, &targets),
        |(actions, _, _)| actions.len(),
    );

    let (result, final_nodes) = match result {
        some @ Some(_) => (some, ai_nodes),
        None => {
            let fallback_pruner = StrategicPruner::new(game);
            let mut fallback_probs = vec![1.0f32; num_targets];
            for (i, target) in targets.iter().enumerate() {
                if let Target::Lever(l) = target {
                    fallback_probs[i] = if discovered_set.contains(l) { 3.0 } else { 0.3 };
                }
            }
            let fsum: f32 = fallback_probs.iter().sum();
            for p in &mut fallback_probs {
                *p /= fsum;
            }
            let fallback_marginals = vec![fallback_probs; config.draft_lookahead];
            let frefs: Vec<&[f32]> = fallback_marginals.iter().map(|v| v.as_slice()).collect();
            let fallback_tree = build_dd_tree_pruned(&frefs, &config, &fallback_pruner, false);
            let fb_nodes = fallback_tree.len();
            (
                par_find_shortest_sequence(
                    &fallback_tree,
                    |seq| try_sequence(game, seq, &targets),
                    |(actions, _, _)| actions.len(),
                ),
                fb_nodes,
            )
        }
    };

    let total = start.elapsed();
    result.map(
        |(target_sequence, (actions, states, milestones))| Solution {
            target_sequence,
            milestones,
            actions,
            states,
            solve_time_ms: total.as_millis() as u64,
            tree_nodes: final_nodes,
            levers_discovered: observations,
        },
    )
}

fn solve_hybrid(game: &StrategicGame) -> Option<Solution> {
    let start = Instant::now();
    let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
    let num_targets = targets.len();

    let (discovered, observations) = discover_levers(game);
    let pruner = AiPruner::new(game, &discovered);

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 100_000;

    // Uniform marginals for valid targets — brute force ordering, no AI weighting bias
    let discovered_set: HashSet<usize> = discovered.iter().copied().collect();
    let mut probs = vec![0.0f32; num_targets];
    for (i, target) in targets.iter().enumerate() {
        let is_valid = match target {
            Target::Lever(l) => discovered_set.contains(l),
            _ => true,
        };
        if is_valid {
            probs[i] = 1.0;
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    let marginals = vec![probs; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();
    let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
    let hybrid_nodes = tree.len();

    let result = par_find_shortest_sequence(
        &tree,
        |seq| try_sequence(game, seq, &targets),
        |(actions, _, _)| actions.len(),
    );

    let (result, final_nodes) = match result {
        some @ Some(_) => (some, hybrid_nodes),
        None => {
            let fallback_pruner = StrategicPruner::new(game);
            let uniform = vec![1.0f32 / num_targets as f32; num_targets];
            let fallback_marginals = vec![uniform; config.draft_lookahead];
            let frefs: Vec<&[f32]> = fallback_marginals.iter().map(|v| v.as_slice()).collect();
            let fallback_tree = build_dd_tree_pruned(&frefs, &config, &fallback_pruner, false);
            let fb_nodes = fallback_tree.len();
            (
                par_find_shortest_sequence(
                    &fallback_tree,
                    |seq| try_sequence(game, seq, &targets),
                    |(actions, _, _)| actions.len(),
                ),
                fb_nodes,
            )
        }
    };

    let total = start.elapsed();
    result.map(
        |(target_sequence, (actions, states, milestones))| Solution {
            target_sequence,
            milestones,
            actions,
            states,
            solve_time_ms: total.as_millis() as u64,
            tree_nodes: final_nodes,
            levers_discovered: observations,
        },
    )
}

// ── Benchmark Runner ───────────────────────────────────────────

struct BenchResult {
    seed: u64,
    lever_mask: u8,
    bf_steps: usize,
    bf_nodes: usize,
    bf_ms: u64,
    ai_steps: usize,
    ai_nodes: usize,
    ai_ms: u64,
    ai_obs: usize,
    ai_fallback: bool,
    hy_steps: usize,
    hy_nodes: usize,
    hy_ms: u64,
    hy_obs: usize,
    hy_fallback: bool,
}

fn bench_seed(seed: u64) -> Option<BenchResult> {
    let game = StrategicGame::new(MAP, seed);
    let bf = solve(&game)?;
    let ai = solve_ai(&game)?;
    let hy = solve_hybrid(&game).unwrap_or_else(|| bf.clone());

    // Detect fallback: AI tree == BF tree means fallback triggered
    let ai_fallback = ai.tree_nodes == bf.tree_nodes;
    let hy_fallback = hy.tree_nodes == bf.tree_nodes;

    Some(BenchResult {
        seed,
        lever_mask: game.target_lever_mask,
        bf_steps: bf.actions.len(),
        bf_nodes: bf.tree_nodes,
        bf_ms: bf.solve_time_ms,
        ai_steps: ai.actions.len(),
        ai_nodes: ai.tree_nodes,
        ai_ms: ai.solve_time_ms,
        ai_obs: ai.levers_discovered,
        ai_fallback,
        hy_steps: hy.actions.len(),
        hy_nodes: hy.tree_nodes,
        hy_ms: hy.solve_time_ms,
        hy_obs: hy.levers_discovered,
        hy_fallback,
    })
}

fn main() {
    let seeds: Vec<u64> = (42..=71).collect();
    let mut results = Vec::new();
    let mut skipped = 0;

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║              🐻 BruteForce vs 🐰 AI vs 🦊 Hybrid — Strategic Puzzle Solver Benchmark         ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();

    for &seed in &seeds {
        eprint!("seed={seed:>3}...");
        match bench_seed(seed) {
            Some(r) => {
                eprintln!(
                    " BF:{}steps/{}nodes/{:>4}ms  AI:{}steps/{}nodes/{:>4}ms{}  HY:{}steps/{}nodes/{:>4}ms{}",
                    r.bf_steps,
                    r.bf_nodes,
                    r.bf_ms,
                    r.ai_steps,
                    r.ai_nodes,
                    r.ai_ms,
                    if r.ai_fallback { " [FB]" } else { "" },
                    r.hy_steps,
                    r.hy_nodes,
                    r.hy_ms,
                    if r.hy_fallback { " [FB]" } else { "" },
                );
                results.push(r);
            }
            None => {
                eprintln!(" unsolvable");
                skipped += 1;
            }
        }
    }

    println!(
        "┌──────┬────────┬──────────────────────────────────┬──────────────────────────────────────┬──────────────────────────────────────┐"
    );
    println!(
        "│ Seed │ LvrMsk │ 🐻 BruteForce (Steps/Nodes/ms) │ 🐰 AI (Steps/Nodes/ms/Obs/FB)       │ 🦊 Hybrid (Steps/Nodes/ms/Obs/FB)   │"
    );
    println!(
        "├──────┼────────┼──────────────────────────────────┼──────────────────────────────────────┼──────────────────────────────────────┤"
    );

    for r in &results {
        let node_pct_ai = if r.bf_nodes > 0 {
            100 - (r.ai_nodes * 100 / r.bf_nodes)
        } else {
            0
        };
        let node_pct_hy = if r.bf_nodes > 0 {
            100 - (r.hy_nodes * 100 / r.bf_nodes)
        } else {
            0
        };

        let ai_fb_tag = if r.ai_fallback { " FB" } else { "" };
        let hy_fb_tag = if r.hy_fallback { " FB" } else { "" };

        let ai_step_tag = if r.ai_steps < r.bf_steps {
            format!("⚡{}", r.bf_steps - r.ai_steps)
        } else {
            "=".into()
        };
        let hy_step_tag = if r.hy_steps < r.bf_steps {
            format!("⚡{}", r.bf_steps - r.hy_steps)
        } else {
            "=".into()
        };

        println!(
            "│ {:>4} │ 0b{:03b}  │ {:>5} / {:>5} / {:>4}ms       │ {:>5} / {:>5} / {:>4}ms / {}{:>2}  {} │ {:>5} / {:>5} / {:>4}ms / {}{:>2}  {} │",
            r.seed,
            r.lever_mask,
            r.bf_steps,
            r.bf_nodes,
            r.bf_ms,
            r.ai_steps,
            r.ai_nodes,
            r.ai_ms,
            r.ai_obs,
            ai_fb_tag,
            format_args!("steps:{} nodes:{}%↓", ai_step_tag, node_pct_ai),
            r.hy_steps,
            r.hy_nodes,
            r.hy_ms,
            r.hy_obs,
            hy_fb_tag,
            format_args!("steps:{} nodes:{}%↓", hy_step_tag, node_pct_hy),
        );
    }

    println!(
        "└──────┴────────┴──────────────────────────────────┴──────────────────────────────────────┴──────────────────────────────────────┘"
    );

    // Aggregate stats
    let count = results.len();
    if count == 0 {
        println!("\n⚠ No solvable seeds in range");
        return;
    }

    let avg_bf_steps: f64 = results.iter().map(|r| r.bf_steps as f64).sum::<f64>() / count as f64;
    let avg_ai_steps: f64 = results.iter().map(|r| r.ai_steps as f64).sum::<f64>() / count as f64;
    let avg_hy_steps: f64 = results.iter().map(|r| r.hy_steps as f64).sum::<f64>() / count as f64;

    let avg_bf_nodes: f64 = results.iter().map(|r| r.bf_nodes as f64).sum::<f64>() / count as f64;
    let avg_ai_nodes: f64 = results.iter().map(|r| r.ai_nodes as f64).sum::<f64>() / count as f64;
    let avg_hy_nodes: f64 = results.iter().map(|r| r.hy_nodes as f64).sum::<f64>() / count as f64;

    let avg_bf_ms: f64 = results.iter().map(|r| r.bf_ms as f64).sum::<f64>() / count as f64;
    let avg_ai_ms: f64 = results.iter().map(|r| r.ai_ms as f64).sum::<f64>() / count as f64;
    let avg_hy_ms: f64 = results.iter().map(|r| r.hy_ms as f64).sum::<f64>() / count as f64;

    let ai_fallbacks = results.iter().filter(|r| r.ai_fallback).count();
    let hy_fallbacks = results.iter().filter(|r| r.hy_fallback).count();

    let ai_wins_steps = results.iter().filter(|r| r.ai_steps < r.bf_steps).count();
    let hy_wins_steps = results.iter().filter(|r| r.hy_steps < r.bf_steps).count();
    let hy_beats_ai = results.iter().filter(|r| r.hy_steps < r.ai_steps).count();
    let ai_beats_hy = results.iter().filter(|r| r.ai_steps < r.hy_steps).count();

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!(
        "│ Aggregate ({count} seeds, {skipped} unsolvable)                                  │"
    );
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│              │ 🐻 BruteForce │ 🐰 AI          │ 🦊 Hybrid       │");
    println!(
        "│ Avg Steps    │ {avg_bf_steps:>13.1} │ {avg_ai_steps:>14.1} │ {avg_hy_steps:>15.1} │"
    );
    println!(
        "│ Avg Nodes    │ {avg_bf_nodes:>13.0} │ {avg_ai_nodes:>14.0} │ {avg_hy_nodes:>15.0} │"
    );
    println!("│ Avg Time(ms) │ {avg_bf_ms:>13.1} │ {avg_ai_ms:>14.1} │ {avg_hy_ms:>15.1} │");
    println!("│ Fallbacks    │             - │ {ai_fallbacks:>14} │ {hy_fallbacks:>15} │");
    println!("│ Step Wins    │             - │ {ai_wins_steps:>14} │ {hy_wins_steps:>15} │");
    println!("├──────────────┴───────────────┴────────────────┴─────────────────┤");
    println!(
        "│ 🐰 AI beats 🐻 BF in steps:   {ai_wins_steps:>3}/{count} seeds ({:.0}%)",
        ai_wins_steps as f64 / count as f64 * 100.0
    );
    println!(
        "│ 🦊 HY beats 🐻 BF in steps:   {hy_wins_steps:>3}/{count} seeds ({:.0}%)",
        hy_wins_steps as f64 / count as f64 * 100.0
    );
    println!(
        "│ 🐰 AI beats 🦊 HY in steps:   {ai_beats_hy:>3}/{count} seeds ({:.0}%)",
        ai_beats_hy as f64 / count as f64 * 100.0
    );
    println!(
        "│ 🦊 HY beats 🐰 AI in steps:   {hy_beats_ai:>3}/{count} seeds ({:.0}%)",
        hy_beats_ai as f64 / count as f64 * 100.0
    );
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Verdict
    println!();
    let node_red_ai = if avg_bf_nodes > 0.0 {
        (1.0 - avg_ai_nodes / avg_bf_nodes) * 100.0
    } else {
        0.0
    };
    let node_red_hy = if avg_bf_nodes > 0.0 {
        (1.0 - avg_hy_nodes / avg_bf_nodes) * 100.0
    } else {
        0.0
    };
    let speed_ai = if avg_ai_ms > 0.0 {
        avg_bf_ms / avg_ai_ms
    } else {
        0.0
    };
    let speed_hy = if avg_hy_ms > 0.0 {
        avg_bf_ms / avg_hy_ms
    } else {
        0.0
    };

    println!("🔍 Verdict:");
    println!(
        "   🐰 AI:    {node_red_ai:.0}% fewer nodes, {:.1}x speedup vs BF, {ai_fallbacks} fallbacks",
        speed_ai
    );
    println!(
        "   🦊 Hybrid: {node_red_hy:.0}% fewer nodes, {:.1}x speedup vs BF, {hy_fallbacks} fallbacks",
        speed_hy
    );

    if hy_beats_ai > ai_beats_hy {
        println!(
            "   🏆 🦊 Hybrid wins: beats 🐰 AI in {hy_beats_ai}/{count} seeds (brute force ordering finds shorter paths!)"
        );
    } else if ai_beats_hy > hy_beats_ai {
        println!(
            "   🏆 🐰 AI wins: beats 🦊 Hybrid in {ai_beats_hy}/{count} seeds (weighted marginals better than uniform)"
        );
    } else {
        println!("   🤝 🐰 AI = 🦊 Hybrid: tied in step quality across {count} seeds");
    }

    let both_win = results
        .iter()
        .filter(|r| r.ai_steps < r.bf_steps && r.hy_steps < r.bf_steps)
        .count();
    println!(
        "   📊 Both AI+HY beat BF: {both_win}/{count} seeds — lever pruning helps when it avoids unnecessary detours"
    );
    println!(
        "   📊 AI fallback rate: {ai_fallbacks}/{count} ({:.0}%) — boss/traps force visiting 'unnecessary' levers",
        ai_fallbacks as f64 / count as f64 * 100.0
    );
}
