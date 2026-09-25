//! tetris_06_rulebook_arena — Issue 892 T2/T4: the ruliology arena over the
//! strategy RULEBOOK (`common/tetris_rulebook.rs`).
//!
//! Subcommands (all seeded, all deterministic):
//! - `anchor` — the rulebook engine must REPRODUCE Bench 891's ply2-shaped
//!   exactly (same pieces/lines per seed) before any rule is measured.
//! - `enum` — ruliology enumeration: every subset of the owner's rules
//!   (9-1 well, tetris bonus, flat top, hole-cover/downstack, hold-I,
//!   preview, hold queue) on top of the classic base, plus leave-one-out
//!   over the classic base; ranked, with the Pareto front on
//!   (survival, points, complexity).
//! - `climb` — delta-gated self-evolve over the genome (rule flips + weight
//!   nudges per mode), accept only strict improvement ≥ δ on TRAIN seeds;
//!   the champion is then scored on HELD-OUT seeds against Bench 891.
//! - `eval <genome-line>` — score one genome (held-out seeds).
//!
//! Run: `cargo run --release --example tetris_06_rulebook_arena -- <cmd> [opts]`
//! opts: `--games N --cap N --rows N --fill PCT --seed0 S --depth D --beam K`
//! climb: `--iters N --delta10 D --test-seed0 S --test-games N --no-hold
//! --fitness pieces|points`

use katgpt_tetris::sim as tetris_sim;
use katgpt_tetris::lookahead as tetris_lookahead;
use katgpt_tetris::rulebook as tetris_rulebook;

use std::time::Instant;
use tetris_lookahead::{apply, garbage_board, pick, Bag, Player};
use tetris_rulebook::{
    play_game, selftest, toggleable, GameStats, Genome, Physics, RuleId, MODES, N_MODES,
    RULES,
};
use tetris_sim::{landing_options_with, Board, DropRule};

#[derive(Clone, Copy)]
struct Cfg {
    games: u64,
    cap: usize,
    rows: usize,
    fill: u64,
    seed0: u64,
}

impl Cfg {
    fn start(&self, seed: u64) -> Board {
        if self.rows > 0 {
            garbage_board(seed, self.rows, self.fill)
        } else {
            Board::empty()
        }
    }
    fn seeds(&self) -> impl Iterator<Item = u64> + '_ {
        self.seed0..self.seed0 + self.games
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Score {
    games: u64,
    survived: u64,
    pieces: u64,
    lines: u64,
    points: u64,
    tetrises: u64,
    holds: u64,
    modes: [u64; N_MODES],
}

impl Score {
    fn add(&mut self, s: &GameStats, cap: usize) {
        self.games += 1;
        self.survived += u64::from(s.pieces == cap);
        self.pieces += s.pieces as u64;
        self.lines += s.lines as u64;
        self.points += s.points;
        self.tetrises += s.tetrises as u64;
        self.holds += s.holds as u64;
        for m in 0..N_MODES {
            self.modes[m] += s.mode_counts[m] as u64;
        }
    }
    /// Scalar fitness: survival first (pieces is its continuous proxy),
    /// points as the tiebreak — or points first (`--fitness points`, the
    /// scoring regime) with pieces as the tiebreak.
    fn fitness(&self) -> f64 {
        let g = self.games.max(1) as f64;
        if POINTS_FITNESS.load(std::sync::atomic::Ordering::Relaxed) {
            self.points as f64 / g + 1e-3 * self.pieces as f64 / g
        } else {
            self.pieces as f64 / g + 1e-3 * self.points as f64 / g
        }
    }
    fn row(&self) -> String {
        let g = self.games.max(1) as f64;
        let tot: u64 = self.modes.iter().sum::<u64>().max(1);
        format!(
            "{:>5} {:>7.0} {:>7.1} {:>8.0} {:>6.2} {:>5.1} {:>4.0}/{:>2.0}/{:>2.0}%",
            format!("{}/{}", self.survived, self.games),
            self.pieces as f64 / g,
            self.lines as f64 / g,
            self.points as f64 / g,
            self.tetrises as f64 / g,
            self.holds as f64 / g,
            100.0 * self.modes[0] as f64 / tot as f64,
            100.0 * self.modes[1] as f64 / tot as f64,
            100.0 * self.modes[2] as f64 / tot as f64,
        )
    }
}

static POINTS_FITNESS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// `--no-hold`: the laya arena has no hold queue — the climb must not use it.
static NO_HOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const HEADER: &str = "surv   pieces   lines   points  tetr  hold  mode b/d/s";

/// Parallel map over independent work items (scoped threads, no deps).
fn par_map<T: Sync, R: Send + Default + Clone>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n_threads = std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1))
        .unwrap_or(4)
        .min(items.len().max(1));
    let mut out = vec![R::default(); items.len()];
    let chunk = items.len().div_ceil(n_threads);
    std::thread::scope(|s| {
        for (ins, outs) in items.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let f = &f;
            s.spawn(move || {
                for (i, o) in ins.iter().zip(outs.iter_mut()) {
                    *o = f(i);
                }
            });
        }
    });
    out
}

fn score_genome(g: &Genome, cfg: &Cfg) -> Score {
    let seeds: Vec<u64> = cfg.seeds().collect();
    let stats = par_map(&seeds, |&s| play_game(g, s, cfg.cap, cfg.start(s)));
    let mut sc = Score::default();
    for st in &stats {
        sc.add(st, cfg.cap);
    }
    sc
}

/// Score many genomes; parallel over (genome, seed) pairs.
fn score_many(gs: &[Genome], cfg: &Cfg) -> Vec<Score> {
    let seeds: Vec<u64> = cfg.seeds().collect();
    let work: Vec<(usize, u64)> = (0..gs.len())
        .flat_map(|i| seeds.iter().map(move |&s| (i, s)))
        .collect();
    let stats = par_map(&work, |&(i, s)| play_game(&gs[i], s, cfg.cap, cfg.start(s)));
    let mut out = vec![Score::default(); gs.len()];
    for ((i, _), st) in work.iter().zip(&stats) {
        out[*i].add(st, cfg.cap);
    }
    out
}

// ── anchor ───────────────────────────────────────────────────────────────

/// Bench 891's ply2-shaped game loop, verbatim semantics (for the anchor).
fn play_891(seed: u64, cfg: &Cfg) -> (usize, u32) {
    let mut bag = Bag::new(seed);
    let mut board = cfg.start(seed);
    let mut next = bag.draw();
    let (mut pieces, mut lines) = (0usize, 0u32);
    while pieces < cfg.cap {
        let cur = next;
        next = bag.draw();
        let Some(i) = pick(&board, cur, next, Player::Ply2Shaped) else {
            break;
        };
        let opts = landing_options_with(&board, cur, DropRule::FromTop);
        let (b1, l1) = apply(&board, &opts[i].cells);
        board = b1;
        lines += l1;
        pieces += 1;
    }
    (pieces, lines)
}

fn cmd_anchor(cfg: &Cfg) {
    let g = Genome::bench891_ply2_shaped();
    println!("anchor genome {} = {}", g.id(), g.to_line());
    let seeds: Vec<u64> = cfg.seeds().collect();
    let pairs = par_map(&seeds, |&s| {
        let a = play_891(s, cfg);
        let b = play_game(&g, s, cfg.cap, cfg.start(s));
        (a, (b.pieces, b.lines))
    });
    let mut ok = 0;
    for (s, (a, b)) in seeds.iter().zip(&pairs) {
        let m = a == b;
        ok += usize::from(m);
        println!(
            "  seed {s:>3}: bench891 {:>4} pcs {:>4} lines | rulebook {:>4} pcs {:>4} lines {}",
            a.0,
            a.1,
            b.0,
            b.1,
            if m { "✓" } else { "✗ MISMATCH" }
        );
    }
    assert_eq!(ok, seeds.len(), "the rulebook engine must reproduce Bench 891 ply2-shaped");
    println!("anchor ✓ {ok}/{} seeds identical", seeds.len());
}

// ── enum ─────────────────────────────────────────────────────────────────

/// The owner's rules (the enumeration axis) and the classic base.
const OWNER: [RuleId; 7] = [
    RuleId::NineOneWell,
    RuleId::TetrisBonus,
    RuleId::FlatTop,
    RuleId::HoleCover,
    RuleId::HoldI,
    RuleId::NextPreview,
    RuleId::HoldQueue,
];
const CLASSIC: [RuleId; 7] = [
    RuleId::Lines,
    RuleId::RowTrans,
    RuleId::ColTrans,
    RuleId::Holes,
    RuleId::Wells,
    RuleId::MaxHeight,
    RuleId::DeepWellUrgency,
];

fn subset_label(g: &Genome) -> String {
    let on: Vec<&str> = OWNER
        .iter()
        .filter(|&&id| g.on(id))
        .map(|&id| RULES[id as usize].key)
        .collect();
    if on.is_empty() { "(classic only)".into() } else { on.join("+") }
}

fn cmd_enum(cfg: &Cfg, depth: u8, beam: u8) {
    let base = {
        let mut g = Genome::full(Physics::FromTop);
        g.depth = depth;
        g.beam = beam;
        g
    };
    let mut gs: Vec<Genome> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for mask in 0u32..(1 << OWNER.len()) {
        let mut g = base.clone();
        for (k, &id) in OWNER.iter().enumerate() {
            g.set(id, mask & (1 << k) != 0);
        }
        labels.push(subset_label(&g));
        gs.push(g);
    }
    // Leave-one-out over the classic base (full owner set on).
    for &id in &CLASSIC {
        let mut g = base.clone();
        g.set(id, false);
        labels.push(format!("ALL − {}", RULES[id as usize].key));
        gs.push(g);
    }
    let t = Instant::now();
    let scores = score_many(&gs, cfg);
    let secs = t.elapsed().as_secs_f64();
    let mut order: Vec<usize> = (0..gs.len()).collect();
    order.sort_by(|&a, &b| scores[b].fitness().total_cmp(&scores[a].fitness()));
    println!(
        "enum: {} genomes × {} games, {:.1}s wall (depth {depth}, beam {beam})",
        gs.len(),
        cfg.games,
        secs
    );
    println!("{:<4} {:<44} {:>3}  {HEADER}", "rank", "rules (owner set on top of classic)", "cx");
    for (rank, &i) in order.iter().enumerate() {
        println!(
            "{:<4} {:<44} {:>3}  {}",
            rank + 1,
            labels[i],
            gs[i].complexity(),
            scores[i].row()
        );
    }
    // Pareto front on (survived, points, −complexity).
    println!("\nPareto front (survived ↑, points ↑, complexity ↓):");
    for &i in &order {
        let dominated = (0..gs.len()).any(|j| {
            let (a, b) = (&scores[j], &scores[i]);
            let ge = a.survived >= b.survived
                && a.points >= b.points
                && gs[j].complexity() <= gs[i].complexity();
            let gt = a.survived > b.survived
                || a.points > b.points
                || gs[j].complexity() < gs[i].complexity();
            j != i && ge && gt
        });
        if !dominated {
            println!("  {:<44} cx {:>2}  {}  id {}", labels[i], gs[i].complexity(), scores[i].row(), gs[i].id());
        }
    }
    // Marginal value of each owner rule: mean fitness with vs without.
    println!("\nmarginal effect per owner rule (mean over the 2^6 contexts of the other rules):");
    for (k, &id) in OWNER.iter().enumerate() {
        let (mut dp, mut dpts, mut dt, mut n) = (0.0, 0.0, 0.0, 0.0);
        for mask in 0u32..(1 << OWNER.len()) {
            if mask & (1 << k) == 0 {
                let (a, b) = (&scores[(mask | (1 << k)) as usize], &scores[mask as usize]);
                let g = a.games.max(1) as f64;
                dp += (a.pieces as f64 - b.pieces as f64) / g;
                dpts += (a.points as f64 - b.points as f64) / g;
                dt += (a.tetrises as f64 - b.tetrises as f64) / g;
                n += 1.0;
            }
        }
        println!(
            "  {:<12} {:>+8.1} pieces/g {:>+8.0} points/g {:>+6.2} tetrises/g   ({})",
            RULES[id as usize].key,
            dp / n,
            dpts / n,
            dt / n,
            RULES[id as usize].source
        );
    }
    let best = &gs[order[0]];
    println!("\nbest genome {} :: {}", best.id(), best.to_line());
}

// ── climb (delta-gated self-evolve) ──────────────────────────────────────

fn mutate(g: &Genome, rng: &mut fastrand::Rng) -> (Genome, String) {
    let mut m = g.clone();
    let no_hold = NO_HOLD.load(std::sync::atomic::Ordering::Relaxed);
    let tog: Vec<RuleId> = toggleable(Physics::FromTop)
        .into_iter()
        .filter(|&id| !(no_hold && matches!(id, RuleId::HoldQueue | RuleId::HoldI)))
        .collect();
    match rng.u32(0..10) {
        0 | 1 => {
            let id = tog[rng.usize(0..tog.len())];
            let on = !m.on(id);
            m.set(id, on);
            (m, format!("{} {}", if on { "enable" } else { "disable" }, RULES[id as usize].key))
        }
        2 => {
            let d: i16 = if rng.bool() { 1 } else { -1 };
            m.survive_h = (m.survive_h as i16 + d).clamp(6, 19) as u8;
            let what = format!("survive_h → {}", m.survive_h);
            (m, what)
        }
        _ => {
            let weighted: Vec<RuleId> = RULES
                .iter()
                .filter(|r| r.kind == tetris_rulebook::RuleKind::BoardWeight && m.on(r.id))
                .map(|r| r.id)
                .collect();
            let id = weighted[rng.usize(0..weighted.len())];
            let mode = MODES[rng.usize(0..N_MODES)];
            let w = &mut m.w[id as usize][mode as usize];
            let f = if rng.bool() { 1.25 } else { 0.8 };
            *w = if *w == 0.0 { if rng.bool() { 1.0 } else { -1.0 } } else { *w * f };
            // Keep weights readable (4 significant decimals) so the genome
            // line stays short and exact.
            *w = (*w * 1e4).round() / 1e4;
            let what = format!("w[{}][{}] → {}", RULES[id as usize].key, mode.name(), *w);
            (m, what)
        }
    }
}

fn cmd_climb(train: &Cfg, test: &Cfg, iters: u32, delta: f64, depth: u8, beam: u8) {
    let mut g = Genome::full(Physics::FromTop);
    g.depth = depth;
    g.beam = beam;
    if NO_HOLD.load(std::sync::atomic::Ordering::Relaxed) {
        g.set(RuleId::HoldQueue, false);
        g.set(RuleId::HoldI, false);
    }
    let mut best = score_genome(&g, train);
    println!("climb start {} fitness {:.1} :: {HEADER}\n  {}", g.id(), best.fitness(), best.row());
    let mut rng = fastrand::Rng::with_seed(892);
    let mut accepted = 0;
    for it in 1..=iters {
        let (cand, what) = mutate(&g, &mut rng);
        let sc = score_genome(&cand, train);
        if sc.fitness() >= best.fitness() + delta {
            accepted += 1;
            println!(
                "  it {it:>3} ACCEPT {:<34} {:.1} → {:.1}  {}",
                what,
                best.fitness(),
                sc.fitness(),
                sc.row()
            );
            g = cand;
            best = sc;
        }
    }
    println!("climb: {accepted}/{iters} accepted; champion {} :: {}", g.id(), g.to_line());
    println!("\nHELD-OUT ({} seeds from {}): {HEADER}", test.games, test.seed0);
    let anchor = Genome::bench891_ply2_shaped();
    let full = {
        let mut f = Genome::full(Physics::FromTop);
        f.depth = depth;
        f.beam = beam;
        if NO_HOLD.load(std::sync::atomic::Ordering::Relaxed) {
            f.set(RuleId::HoldQueue, false);
            f.set(RuleId::HoldI, false);
        }
        f
    };
    for (name, gg) in [("bench891 ply2-shaped", &anchor), ("rulebook full (defaults)", &full), ("climb champion", &g)] {
        println!("  {:<26} {}", name, score_genome(gg, test).row());
    }
}

fn cmd_eval(line: &str, cfg: &Cfg) {
    let g = Genome::from_line(line).expect("unparseable genome line");
    let t = Instant::now();
    let sc = score_genome(&g, cfg);
    println!("eval {} ({:.1}s)\n  {HEADER}\n  {}", g.id(), t.elapsed().as_secs_f64(), sc.row());
}

fn main() {
    selftest();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().unwrap_or_else(|| "anchor".into());
    let opt = |k: &str, d: u64| -> u64 {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let cfg = Cfg {
        games: opt("--games", 10),
        cap: opt("--cap", 500) as usize,
        rows: opt("--rows", 16) as usize,
        fill: opt("--fill", 75),
        seed0: opt("--seed0", 1),
    };
    let depth = opt("--depth", 2) as u8;
    let flag = |k: &str| args.iter().any(|a| a == k);
    POINTS_FITNESS.store(
        args.iter()
            .position(|a| a == "--fitness")
            .and_then(|i| args.get(i + 1))
            .is_some_and(|v| v == "points"),
        std::sync::atomic::Ordering::Relaxed,
    );
    NO_HOLD.store(flag("--no-hold"), std::sync::atomic::Ordering::Relaxed);
    let beam = opt("--beam", 6) as u8;
    println!(
        "== tetris_06_rulebook_arena :: {cmd} :: games {} cap {} garbage {} rows @ {}% seeds {}.. ==",
        cfg.games, cfg.cap, cfg.rows, cfg.fill, cfg.seed0
    );
    match cmd.as_str() {
        "anchor" => cmd_anchor(&cfg),
        "enum" => cmd_enum(&cfg, depth, beam),
        "climb" => {
            let test = Cfg {
                seed0: opt("--test-seed0", 101),
                games: opt("--test-games", 20),
                ..cfg
            };
            cmd_climb(&cfg, &test, opt("--iters", 40) as u32, opt("--delta10", 20) as f64 / 10.0, depth, beam);
        }
        "eval" => {
            let line = args.get(1).expect("eval needs a genome line (quote it)");
            cmd_eval(line, &cfg);
        }
        other => panic!("unknown command {other:?} (anchor | enum | climb | eval)"),
    }
}
