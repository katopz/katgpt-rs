//! Lodestar — Completion-Distance Pruning (Plan 207, Research 183).
//!
//! GOAT proof for the core claim: **one precomputed integer per automaton state**
//! — the shortest-accepting-distance `d(s)` — simultaneously delivers
//!   (A) a budget-aware masking *guarantee* (TRUNCPROOF),
//!   (B) jump-ahead *speed* on singular paths (SGLang compressed-FSM), and
//!   (C) A\* ordering + *termination* (Knaster–Tarski / XML fixed-point).
//!
//! It does so with ZERO model training — pure inference-time symbolic machinery.
//!
//! Run: `cargo run --release --example lodestar_demo --features lodestar`
//!
//! The grammar is a forced 3-token header followed by a (possibly nested) array
//! of numbers — `H H [ ... ]`. A "draft model" that loves opening brackets is the
//! adversary: under a tight token budget, **naive masking** keeps nesting and runs
//! out of budget before it can close → truncated, INVALID output. **Lodestar**
//! masks any token whose successor cannot still complete within the remaining
//! budget, so it is *guaranteed* to emit a complete, valid, in-budget output —
//! while jump-ahead collapses the forced header from 3 steps to 1.

// ── Vocabulary ──────────────────────────────────────────────────
const OPEN: usize = 0; // [
const CLOSE: usize = 1; // ]
const NUM: usize = 2; // a number literal
const COMMA: usize = 3; // ,
const HDR: usize = 4; // forced header filler
const VOCAB: usize = 5;
const TOK_NAME: [&str; VOCAB] = ["[", "]", "N", ",", "H"];

/// Max nesting depth the grammar allows.
const MAX_DEPTH: usize = 3;

// ── Automaton ───────────────────────────────────────────────────
//
// States (flat indices):
//   0,1,2          = header H0,H1,H2  (each forces one token)
//   3 + (d-1)*2    = (depth d, Value)   — expecting a value: NUM or nested `[`
//   3 + (d-1)*2 +1 = (depth d, More)    — after a value: `,` or `]`
//   ACCEPT         = complete top-level array
const HLEN: usize = 3;
const fn s_value(d: usize) -> usize {
    HLEN + (d - 1) * 2
}
const fn s_more(d: usize) -> usize {
    HLEN + (d - 1) * 2 + 1
}
const ACCEPT: usize = HLEN + MAX_DEPTH * 2;
const N_STATES: usize = ACCEPT + 1;
const START: usize = 0;

/// Transition function δ(state, token) → Some(next) if the token is legal here.
/// This is exactly what any `ConstraintPruner::is_valid` already encodes — we
/// just make the successor explicit so distances can be precomputed.
fn delta(state: usize, token: usize) -> Option<usize> {
    match state {
        0 if token == HDR => Some(1),
        1 if token == HDR => Some(2),
        2 if token == OPEN => Some(s_value(1)),
        _ if state == ACCEPT => None,
        _ => {
            // body states
            for d in 1..=MAX_DEPTH {
                if state == s_value(d) {
                    return match token {
                        NUM => Some(s_more(d)),
                        OPEN if d < MAX_DEPTH => Some(s_value(d + 1)),
                        _ => None,
                    };
                }
                if state == s_more(d) {
                    return match token {
                        COMMA => Some(s_value(d)),
                        CLOSE => Some(if d == 1 { ACCEPT } else { s_more(d - 1) }),
                        _ => None,
                    };
                }
            }
            None
        }
    }
}

// ── (A) The lodestar: shortest-accepting-distance ───────────────
//
// d(s) = 0 for accepting states; else 1 + min over legal tokens of d(δ(s,t)).
// Computed by fixpoint relaxation (reverse-BFS equivalent) — runs ONCE.
// u32::MAX marks a state from which no accepting state is reachable.
fn precompute_distances() -> [u32; N_STATES] {
    let mut d = [u32::MAX; N_STATES];
    d[ACCEPT] = 0;
    // Bellman-style relaxation; converges in ≤ N_STATES passes.
    let mut changed = true;
    while changed {
        changed = false;
        for s in 0..N_STATES {
            for t in 0..VOCAB {
                if let Some(ns) = delta(s, t)
                    && d[ns] != u32::MAX
                {
                    let cand = d[ns] + 1;
                    if cand < d[s] {
                        d[s] = cand;
                        changed = true;
                    }
                }
            }
        }
    }
    d
}

// ── (B) Singular-span length for jump-ahead ─────────────────────
//
// From `state`, count consecutive states that have exactly one legal token,
// following that forced token, until we hit a real branch (or ACCEPT).
// These spans can be emitted in a single prefill step.
fn singular_span_len(mut state: usize) -> u32 {
    let mut len = 0u32;
    loop {
        let legal: Vec<usize> = (0..VOCAB).filter(|&t| delta(state, t).is_some()).collect();
        if legal.len() != 1 {
            return len;
        }
        state = delta(state, legal[0]).unwrap();
        len += 1;
        if state == ACCEPT {
            return len;
        }
    }
}

// ── Draft "model" — an adversary that loves opening brackets ────
//
// Weights over the vocab. NOT a trained model — a fixed inference-time prior,
// deliberately biased toward `[` to create the budget trap.
const DRAFT_WEIGHT: [f32; VOCAB] = [
    /*OPEN */ 5.0, /*CLOSE*/ 1.0, /*NUM */ 2.0, /*COMMA*/ 2.0,
    /*HDR */ 1.0,
];

/// Tiny deterministic LCG so trials are reproducible without external crates.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32)
    }
}

/// Weighted choice among `legal` tokens by draft weight (renormalized).
fn sample_legal(legal: &[usize], rng: &mut Lcg) -> usize {
    let total: f32 = legal.iter().map(|&t| DRAFT_WEIGHT[t]).sum();
    let mut r = rng.next_f32() * total;
    for &t in legal {
        r -= DRAFT_WEIGHT[t];
        if r <= 0.0 {
            return t;
        }
    }
    *legal.last().unwrap()
}

// ── Decoders ────────────────────────────────────────────────────

struct Run {
    valid_complete: bool, // reached ACCEPT within budget?
    steps: usize,         // decode steps (jump-ahead counts a span as 1 step)
}

/// NON-THINKING baseline: mask only by *legality* (`is_valid`), ignore budget.
/// This is the naive constrained-decoding everyone ships. It can paint itself
/// into a corner: open brackets it cannot close before the budget runs out.
fn decode_naive(budget: usize, rng: &mut Lcg) -> Run {
    let mut state = START;
    let mut tokens = 0;
    let mut steps = 0;
    while tokens < budget {
        if state == ACCEPT {
            return Run {
                valid_complete: true,
                steps,
            };
        }
        let legal: Vec<usize> = (0..VOCAB).filter(|&t| delta(state, t).is_some()).collect();
        if legal.is_empty() {
            break;
        }
        let t = sample_legal(&legal, rng);
        state = delta(state, t).unwrap();
        tokens += 1;
        steps += 1;
    }
    Run {
        valid_complete: state == ACCEPT,
        steps,
    }
}

/// THINKING: Lodestar. Same draft, but
///   (A) prune any token whose successor cannot complete within budget_remaining,
///   (B) emit the forced singular header span in ONE step (jump-ahead),
///   (C) (implicit here) the budget mask guarantees termination at ACCEPT.
fn decode_lodestar(budget: usize, dist: &[u32; N_STATES], rng: &mut Lcg) -> Run {
    let mut state = START;
    let mut tokens = 0;
    let mut steps = 0;
    while tokens < budget {
        if state == ACCEPT {
            return Run {
                valid_complete: true,
                steps,
            };
        }

        // (B) Jump-ahead: collapse a forced singular span into one step if it fits.
        let span = singular_span_len(state);
        if span >= 1 && (tokens + span as usize) <= budget {
            // Walk the forced span (deterministic) in a single decode step.
            for _ in 0..span {
                let legal: Vec<usize> = (0..VOCAB).filter(|&t| delta(state, t).is_some()).collect();
                state = delta(state, legal[0]).unwrap();
                tokens += 1;
            }
            steps += 1; // the whole span = 1 prefill step
            continue;
        }

        // (A) Budget-aware mask: keep token t only if 1 + d(δ(s,t)) ≤ budget_remaining.
        let budget_remaining = (budget - tokens) as u32;
        let legal: Vec<usize> = (0..VOCAB)
            .filter(|&t| match delta(state, t) {
                Some(ns) => dist[ns] != u32::MAX && dist[ns] < budget_remaining,
                None => false,
            })
            .collect();
        if legal.is_empty() {
            // By the admissibility guarantee this is unreachable when the budget
            // was feasible at the root; break defensively.
            break;
        }
        let t = sample_legal(&legal, rng);
        state = delta(state, t).unwrap();
        tokens += 1;
        steps += 1;
    }
    Run {
        valid_complete: state == ACCEPT,
        steps,
    }
}

fn main() {
    let dist = precompute_distances();

    println!("== Lodestar — Completion-Distance Pruning (Plan 207 / Research 183) ==\n");

    // Show the precomputed lodestar (the one integer per state).
    println!("Shortest-accepting-distance d(s) — precomputed ONCE, reverse-BFS:");
    print!("  START(H0)={}  H1={}  H2={}", dist[0], dist[1], dist[2]);
    for d in 1..=MAX_DEPTH {
        print!(
            "  (d{d},Val)={}  (d{d},More)={}",
            dist[s_value(d)],
            dist[s_more(d)]
        );
    }
    println!("  ACCEPT={}", dist[ACCEPT]);
    println!(
        "  forced-header singular span from START = {} tokens → jump-ahead emits in 1 step\n",
        singular_span_len(START)
    );

    // ── Budget sweep (deterministic seed) ──────────────────────
    println!("Budget sweep (draft adversary biased toward '['):");
    println!(
        "  {:>6} | {:>14} {:>6} | {:>16} {:>6} {:>6}",
        "budget", "naive valid?", "steps", "lodestar valid?", "steps", "saved"
    );
    println!(
        "  {:-<6}-+-{:-<14}-{:-<6}-+-{:-<16}-{:-<6}-{:-<6}",
        "", "", "", "", "", ""
    );
    let feasible = dist[START] as usize; // min tokens for ANY valid output
    for budget in [feasible, feasible + 2, feasible + 4, feasible + 8, 30] {
        let mut rng_a = Lcg(0x1234_5678);
        let mut rng_b = Lcg(0x1234_5678); // same seed → same draft stream
        let a = decode_naive(budget, &mut rng_a);
        let b = decode_lodestar(budget, &dist, &mut rng_b);
        let saved = a.steps as i64 - b.steps as i64;
        println!(
            "  {:>6} | {:>14} {:>6} | {:>16} {:>6} {:>+6}",
            budget,
            yn(a.valid_complete),
            a.steps,
            yn(b.valid_complete),
            b.steps,
            saved
        );
    }

    // ── Aggregate rate over many sampled trials at a tight budget ──
    let tight = feasible + 3;
    let trials: usize = 5000;
    let (mut naive_ok, mut lode_ok) = (0u32, 0u32);
    let (mut naive_steps, mut lode_steps) = (0u64, 0u64);
    for s in 0..trials {
        let mut ra = Lcg(0xA5A5 ^ (s as u64).wrapping_mul(2654435761));
        let mut rb = Lcg(0xA5A5 ^ (s as u64).wrapping_mul(2654435761));
        let a = decode_naive(tight, &mut ra);
        let b = decode_lodestar(tight, &dist, &mut rb);
        naive_ok += a.valid_complete as u32;
        lode_ok += b.valid_complete as u32;
        naive_steps += a.steps as u64;
        lode_steps += b.steps as u64;
    }

    println!("\nThinking vs non-thinking — {trials} trials at tight budget = {tight} tokens:");
    println!("  {:<26} {:>12} {:>14}", "", "non-thinking", "Lodestar");
    println!(
        "  {:<26} {:>11.1}% {:>13.1}%",
        "valid-&-complete rate",
        100.0 * naive_ok as f64 / trials as f64,
        100.0 * lode_ok as f64 / trials as f64
    );
    println!(
        "  {:<26} {:>12.2} {:>14.2}",
        "avg decode steps",
        naive_steps as f64 / trials as f64,
        lode_steps as f64 / trials as f64
    );

    // ── Verdict ────────────────────────────────────────────────
    let correctness_win = lode_ok as usize == trials && (naive_ok as usize) < trials;
    let speed_win = lode_steps < naive_steps;
    println!("\nGOAT verdict:");
    println!(
        "  (A) budget guarantee : Lodestar {} valid-in-budget vs naive {} → {}",
        pct(lode_ok, trials),
        pct(naive_ok, trials),
        if correctness_win {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );
    println!(
        "  (B) jump-ahead speed : avg steps {:.2} vs {:.2} → {}",
        lode_steps as f64 / trials as f64,
        naive_steps as f64 / trials as f64,
        if speed_win { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "  overall              : {}",
        if correctness_win && speed_win {
            "GOAT PASS — gain, no model training"
        } else {
            "see above"
        }
    );

    // Inline self-checks (cheap, run every invocation).
    run_invariant_checks(&dist);
}

fn yn(b: bool) -> &'static str {
    if b { "✅ valid" } else { "❌ truncated" }
}
fn pct(ok: u32, n: usize) -> String {
    format!("{:.1}%", 100.0 * ok as f64 / n as f64)
}

/// Invariants from the proof plan (Plan 207 T12): admissibility, monotonicity,
/// and the budget guarantee. Panics on violation — keeps the demo honest.
fn run_invariant_checks(dist: &[u32; N_STATES]) {
    // Admissibility + consistency: d(s) ≤ 1 + d(δ(s,t)) for every legal edge.
    for s in 0..N_STATES {
        for (t, _) in TOK_NAME.iter().enumerate() {
            if let Some(ns) = delta(s, t)
                && dist[ns] != u32::MAX
                && dist[s] != u32::MAX
            {
                assert!(
                    dist[s] <= 1 + dist[ns],
                    "consistency violated at state {s} token {}",
                    TOK_NAME[t]
                );
            }
        }
    }
    // Monotone descent: every reachable non-accept state has a token strictly
    // reducing the distance (guarantees a finite path to ACCEPT).
    for s in 0..N_STATES {
        if s == ACCEPT || dist[s] == u32::MAX {
            continue;
        }
        let has_descent = (0..VOCAB).any(|t| match delta(s, t) {
            Some(ns) => dist[ns] != u32::MAX && dist[ns] + 1 == dist[s],
            None => false,
        });
        assert!(has_descent, "no monotone descent from state {s}");
    }
    // Budget guarantee: at feasible budget, Lodestar always lands on ACCEPT.
    let feasible = dist[START] as usize;
    for s in 0..200u64 {
        let mut rng = Lcg(0xDEAD ^ s.wrapping_mul(11400714819323198485));
        let r = decode_lodestar(feasible + 1, dist, &mut rng);
        assert!(r.valid_complete, "budget guarantee violated on seed {s}");
    }
    println!("\n[invariants] admissibility + monotone-descent + budget-guarantee: all PASS ✅");
}
