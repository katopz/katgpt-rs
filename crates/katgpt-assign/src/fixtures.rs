//! Deterministic instance generators (tests + benchmarks share these;
//! also handy for consumers evaluating the solver on known shapes).
//!
//! All generators are seeded through [`crate::rng::SplitMix64`] — same
//! seed, same instance, on every node.

use crate::rng::SplitMix64;
use crate::types::{ContainerId, Problem, Spec};

/// A perfectly-balanceable instance with NEAR-UNIFORM parts: each of
/// `num_containers` groups has `parts_per_group` parts of size `base ±
/// {0,1}` (perturbed in ± pairs so each group sums to exactly
/// `parts_per_group · base`), plus two unit parts. Near-uniform parts are
/// the realistic shard-workload shape AND make the equal-sum partition
/// reachable by diff-1 swaps from any nearby state (arbitrary-part
/// partition closure can require triple moves — a Phase 2 move type;
/// that family is covered by `random_instance`'s vs-baseline gates). With
/// `Spec::capacity(0, parts_per_group · base)` the optimum has spread 0
/// and zero violation — the known-optimum G1/G2 fixture. `initial` is
/// deliberately bad: every object starts in container 0.
pub fn perfect_balance(
    seed: u64,
    num_containers: usize,
    base: i64,
    parts_per_group: usize,
) -> Problem {
    assert!(base >= 2, "base >= 2 keeps ±1 perturbations positive");
    let group_total = base * parts_per_group as i64;
    let mut rng = SplitMix64::new(seed);
    let mut demands: Vec<i64> = Vec::with_capacity(num_containers * (parts_per_group + 2));
    for _ in 0..num_containers {
        let mut parts = vec![base; parts_per_group];
        // ±1 perturbation pairs keep the group sum invariant while
        // creating diff-1 neighbours (the swap-closure currency).
        let pairs = 1 + rng.below(parts_per_group / 2);
        for _ in 0..pairs {
            let j = rng.below(parts_per_group);
            let k = rng.below(parts_per_group);
            if j != k && parts[j] < base + 1 && parts[k] > base - 1 {
                parts[j] += 1;
                parts[k] -= 1;
            }
        }
        demands.extend_from_slice(&parts);
        // The two unit parts (fine granularity at the partition tail).
        demands.extend_from_slice(&[1, 1]);
    }
    // Deterministic object-order shuffle.
    for i in (1..demands.len()).rev() {
        let j = rng.below(i + 1);
        demands.swap(i, j);
    }
    let n = demands.len();
    Problem {
        num_objects: n,
        num_containers,
        num_dims: 1,
        demands,
        initial: vec![0 as ContainerId; n],
        specs: vec![
            // +2: the two unit parts ride inside the group (each group
            // sums to parts·base + 2).
            Spec::capacity(0, group_total + 2),
            Spec::Balance { dim: 0, weight: 1 },
        ],
    }
}

/// An arbitrary random instance: `num_dims` dims, demands in `1..=max_d`,
/// round-robin initial assignment, one capacity spec per dim (limit =
/// ceil(3/2 · mean) — feasible but requiring repair), balance + movement
/// goals. The G2 workhorse (greedy comparison, lower-bound reporting).
pub fn random_instance(
    seed: u64,
    num_objects: usize,
    num_containers: usize,
    num_dims: usize,
    max_d: i64,
) -> Problem {
    let mut rng = SplitMix64::new(seed);
    let mut demands = Vec::with_capacity(num_objects * num_dims);
    for _ in 0..num_objects * num_dims {
        demands.push(1 + rng.below(max_d as usize) as i64);
    }
    let initial: Vec<ContainerId> = (0..num_objects)
        .map(|o| (o % num_containers) as ContainerId)
        .collect();
    let mut specs = Vec::with_capacity(num_dims + 2);
    for d in 0..num_dims {
        let total: i128 = demands[d..]
            .iter()
            .step_by(num_dims)
            .map(|&x| x as i128)
            .sum();
        let mean = (total / num_objects as i128) as i64;
        // Container-level limit: mean load × objects-per-container × slack.
        let per_container_objects = num_objects.div_ceil(num_containers) as i64;
        let limit = (((mean * per_container_objects * 3) + 1) / 2).max(1);
        specs.push(Spec::capacity(d, limit));
    }
    for d in 0..num_dims {
        specs.push(Spec::Balance { dim: d, weight: 1 });
    }
    // NOTE: deliberately NO MinimizeMovement spec — construction
    // baselines (FFD) ignore `initial`, so a movement term would compare
    // a repair-started solver against a from-scratch greedy unfairly.
    // The movement axis has its own fixture (`overpacked_one_bin`).
    Problem {
        num_objects,
        num_containers,
        num_dims,
        demands,
        initial,
        specs,
    }
}

/// The riir-chain cluster shape (Issue 164 — the primary consumer's bench
/// instance): objects = map shards with two integer load dims (NPC density,
/// tick cost — block-skewed across map ids, the realistic shape: town maps
/// carry most NPCs), containers = nodes with one capacity per dim
/// (katgpt-assign specs are uniform-limit, so heterogeneous node classes
/// are modeled by sizing the limit at the SMALL class — the binding
/// constraint), initial = round-robin (the naive hand assignment the
/// operator would write by hand). Specs: both capacities + both balances +
/// movement (weight 3 — a rebalance that teleports the whole world is
/// worse than a slightly unbalanced stable one, exactly the issue's
/// framing).
pub fn shard_topology(seed: u64, num_shards: usize, num_nodes: usize) -> Problem {
    assert!(num_nodes >= 2);
    let mut rng = SplitMix64::new(seed);
    let mut demands = Vec::with_capacity(num_shards * 2);
    for m in 0..num_shards {
        // Block skew: within each 16-map block the early ids are hot
        // (towns), the tail cold (wilderness) — a 10:1 head-to-tail ratio.
        let hot = 1000 / (1 + m % 16) as i64;
        let npc = 10 + hot / 4 + rng.below(20) as i64;
        let tick = 5 + hot / 8 + rng.below(12) as i64;
        demands.push(npc);
        demands.push(tick);
    }
    let total_npc: i128 = demands.iter().step_by(2).map(|&x| x as i128).sum();
    let total_tick: i128 = demands[1..].iter().step_by(2).map(|&x| x as i128).sum();
    // Uniform capacity at the small-node class with ~20% slack: the
    // even-share × objects-per-node × 1.2 bound.
    let per_node_objects = num_shards.div_ceil(num_nodes) as i128;
    let ram_cap = (total_npc * per_node_objects * 12 / 10 / num_nodes as i128).max(1) as i64;
    let tick_cap = (total_tick * per_node_objects * 12 / 10 / num_nodes as i128).max(1) as i64;
    // Round-robin initial (the hand assignment).
    let initial: Vec<ContainerId> = (0..num_shards)
        .map(|m| (m % num_nodes) as ContainerId)
        .collect();
    Problem {
        num_objects: num_shards,
        num_containers: num_nodes,
        num_dims: 2,
        demands,
        initial,
        specs: vec![
            Spec::capacity(0, ram_cap),
            Spec::capacity(1, tick_cap),
            Spec::Balance { dim: 0, weight: 1 },
            Spec::Balance { dim: 1, weight: 1 },
            Spec::MinimizeMovement { weight: 3 },
        ],
    }
}

/// A movement-dominated repair instance: container 0 holds one fat object
/// (demand `limit`) plus `excess_objects` unit stragglers — over capacity
/// by exactly `excess_objects` — and every other container is EMPTY.
/// Known optimum: move the fat object out (one move) ⇒ violation 0,
/// movement 1, folded objective 1 (any feasible assignment moves ≥ 1
/// object, and movement 1 with zero violation is achievable). The unit
/// stragglers sit FIRST in object order, so an index-order greedy repair
/// that only moves overflowing objects one at a time pays `excess_objects`
/// moves where the solver must find the single fat move.
pub fn overpacked_one_bin(num_containers: usize, limit: i64, excess_objects: usize) -> Problem {
    assert!(num_containers >= 2, "need a spare container to move into");
    assert!(excess_objects as i64 <= limit);
    // Container 0: `excess_objects` unit stragglers + one fat object —
    // stragglers first so index-order greed pays excess moves.
    let mut demands: Vec<i64> = vec![1; excess_objects];
    demands.push(limit);
    let n = demands.len();
    Problem {
        num_objects: n,
        num_containers,
        num_dims: 1,
        demands,
        initial: vec![0 as ContainerId; n],
        specs: vec![
            Spec::capacity(0, limit),
            Spec::MinimizeMovement { weight: 1 },
        ],
    }
}

/// The Issue-1211 two-basin landscape: N unit objects all start packed in
/// container 0 (over capacity by exactly one object — the shallow decoy),
/// capacity permits them spread over `num_containers`; the GOAL layer is
/// MinimizeMovement (weight W_move) fighting a strong Balance pull
/// (weight W_bal). The shallow basin = "move the one overflow object,
/// leave the rest" (low movement, high imbalance); the planted optimum =
/// "spread evenly" (higher movement, near-zero imbalance). With
/// W_bal·imbalance > W_move·(spread moves − 1), the even spread is the
/// planted optimum while the one-move repair is an adversarial local
/// minimum: every single-object move from it is accepted only if it
/// improves balance enough to cover its movement cost — the decoy's
/// neighbors are all downhill or level, so first-improvement descent
/// stalls there unless the tie-break trajectory (seeded) walks the
/// plateau out.
pub fn two_basin(n: usize, containers: usize, w_bal: i64, w_move: i64) -> Problem {
    Problem {
        num_objects: n,
        num_containers: containers,
        num_dims: 1,
        demands: vec![1; n],
        initial: vec![0 as ContainerId; n],
        specs: vec![
            Spec::capacity(0, (n - 1) as i64), // over capacity by exactly 1
            Spec::Balance {
                dim: 0,
                weight: w_bal,
            },
            Spec::MinimizeMovement { weight: w_move },
        ],
    }
}

/// Greedy construction baseline (balance-aware, index order): objects in
/// index order, each into the least-loaded container (total demand across
/// all dims; ties: lowest index) that keeps non-negative residual capacity
/// in EVERY capacity dim; if nothing fits, the container with the largest
/// min-residual (least violation) — mirroring [`first_fit_decreasing`]'s
/// fallback. Deterministic. The second G2 outside baseline: where FFD is
/// capacity-first, this arm is balance-first — it bounds what a 10-line
/// constructive heuristic reaches WITHOUT search, so a search lane's GOAT
/// row cannot hide behind a weak comparator.
pub fn greedy_least_loaded(problem: &Problem) -> Vec<ContainerId> {
    let b = problem.num_containers;
    let odims = problem.num_dims;
    let caps: Vec<(usize, i64)> = problem
        .specs
        .iter()
        .filter_map(|s| match *s {
            Spec::Capacity { dim, limit, .. } => Some((dim, limit)),
            _ => None,
        })
        .collect();
    let mut load = vec![0i64; b * odims];
    let mut out = vec![0 as ContainerId; problem.num_objects];
    let demand = |o: u32, d: usize| problem.demands[o as usize * odims + d];
    for o in 0..problem.num_objects as u32 {
        let min_residual = |c: usize| -> i64 {
            let mut mr = i64::MAX;
            for &(dim, limit) in &caps {
                let r = limit - (load[c * odims + dim] + demand(o, dim));
                mr = mr.min(r);
            }
            if mr == i64::MAX {
                mr = 0; // no capacity specs: every container fits
            }
            mr
        };
        let total_load = |c: usize| -> i64 { (0..odims).map(|d| load[c * odims + d]).sum() };
        let mut chosen: Option<usize> = None;
        let mut best_load = i64::MAX;
        for c in 0..b {
            if min_residual(c) >= 0 {
                let l = total_load(c);
                if l < best_load {
                    best_load = l;
                    chosen = Some(c);
                }
            }
        }
        let chosen = match chosen {
            Some(c) => c,
            None => {
                // Nothing fits: least-violating container (max residual).
                let mut best_c = 0;
                let mut best_r = i64::MIN;
                for c in 0..b {
                    let r = min_residual(c);
                    if r > best_r {
                        best_r = r;
                        best_c = c;
                    }
                }
                best_c
            }
        };
        for d in 0..odims {
            load[chosen * odims + d] += demand(o, d);
        }
        out[o as usize] = chosen as ContainerId;
    }
    out
}

/// Greedy construction baseline (classic first-fit-decreasing): objects
/// by total demand descending (ties: lowest index), each into the FIRST
/// container (index order) with non-negative residual capacity in EVERY
/// capacity dim; if nothing fits, the container with the largest
/// min-residual (least violation). Deterministic. The G2 outside
/// baseline (workspace precedent: riir-rag's greedy knapsack packer).
pub fn first_fit_decreasing(problem: &Problem) -> Vec<ContainerId> {
    let b = problem.num_containers;
    let odims = problem.num_dims;
    let caps: Vec<(usize, i64)> = problem
        .specs
        .iter()
        .filter_map(|s| match *s {
            Spec::Capacity { dim, limit, .. } => Some((dim, limit)),
            _ => None,
        })
        .collect();
    let mut order: Vec<u32> = (0..problem.num_objects as u32).collect();
    let total = |o: u32| -> i64 {
        (0..odims)
            .map(|d| problem.demands[o as usize * odims + d])
            .sum()
    };
    order.sort_unstable_by_key(|&o| (std::cmp::Reverse(total(o)), o));
    let mut load = vec![0i64; b * odims];
    let mut out = vec![0 as ContainerId; problem.num_objects];
    let demand = |o: u32, d: usize| problem.demands[o as usize * odims + d];
    for &o in &order {
        let min_residual = |c: usize| -> i64 {
            let mut mr = i64::MAX;
            for &(dim, limit) in &caps {
                let r = limit - (load[c * odims + dim] + demand(o, dim));
                mr = mr.min(r);
            }
            if mr == i64::MAX {
                mr = 0; // no capacity specs: every container equal
            }
            mr
        };
        let mut chosen = None;
        for c in 0..b {
            if min_residual(c) >= 0 {
                chosen = Some(c);
                break;
            }
        }
        let chosen = match chosen {
            Some(c) => c,
            None => {
                // Nothing fits: least-violating container (max residual).
                let mut best_c = 0;
                let mut best_r = i64::MIN;
                for c in 0..b {
                    let r = min_residual(c);
                    if r > best_r {
                        best_r = r;
                        best_c = c;
                    }
                }
                best_c
            }
        };
        for d in 0..odims {
            load[chosen * odims + d] += demand(o, d);
        }
        out[o as usize] = chosen as ContainerId;
    }
    out
}
