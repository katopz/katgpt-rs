//! tetris_08_puct_goat — Issue 892 T3 G3: the generic chance-node PUCT
//! (`katgpt_core::chance_puct`, the moka trick) vs the Bench 891 champion
//! `Player::Ply2Shaped` (depth-2 exhaustive over cur × next) on IDENTICAL
//! seeds, under garbage-start survival regimes.
//!
//! Both players use the SAME evaluation — `eval_board(board_features(b, l),
//! shaped = true)` — so the only delta is the search: depth-2 exhaustive
//! (max over the KNOWN next piece, no chance) vs PUCT (sigmoid top-k prior =
//! the 1-ply eval, value leaf = the eval, chance nodes over the 7-bag
//! remainder, most-visited root).
//!
//! Games run in parallel across seeds (one worker per seed, `threads`
//! workers); ms/decision is per-worker wall time and so carries the box's
//! load — the box state is printed beside it.
//!
//! Run: `cargo run --release --features chance_puct --example tetris_08_puct_goat
//!       [-- <games=20> <cap=1000> <threads=8> <budgets=100,400,1600> <regimes=16:75,18:75>]`

use katgpt_tetris::lookahead as tetris_lookahead;
#[path = "common/tetris_puct.rs"]
mod tetris_puct;
use katgpt_tetris::sim as tetris_sim;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tetris_lookahead::{
    Bag, LINES_SCORE, Player, apply, board_features, eval_board, garbage_board, pick,
};
use tetris_puct::{PuctCfg, pick_puct};
use tetris_sim::{Board, DropRule, landing_options_with};

#[derive(Clone, Copy)]
enum Arm {
    Ply2,
    Puct(u32),
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Ply2 => "ply2-shaped (Bench 891 champion)".into(),
            Arm::Puct(b) => format!("puct b{b} (c1.5 k8)"),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Game {
    points: u64,
    lines: u32,
    pieces: usize,
    decide_ns: u128,
}

fn eval(b: &Board, l: u32) -> f64 {
    eval_board(&board_features(b, l), true)
}

fn play(seed: u64, arm: Arm, cap: usize, rows: usize, fill: u64) -> Game {
    let mut bag = Bag::new(seed);
    let mut board = garbage_board(seed, rows, fill);
    let mut rng = fastrand::Rng::with_seed(seed ^ 0x05EE_D892);
    let cfg = match arm {
        Arm::Puct(b) => PuctCfg {
            budget: b,
            ..PuctCfg::default()
        },
        Arm::Ply2 => PuctCfg::default(),
    };
    let mut next = bag.draw();
    let mut g = Game::default();
    while g.pieces < cap {
        let cur = next;
        next = bag.draw();
        let t = Instant::now();
        let choice = match arm {
            Arm::Ply2 => pick(&board, cur, next, Player::Ply2Shaped),
            Arm::Puct(_) => pick_puct(&board, cur, next, bag.remaining(), &cfg, &mut rng, &eval),
        };
        g.decide_ns += t.elapsed().as_nanos();
        let Some(i) = choice else { break };
        let options = landing_options_with(&board, cur, DropRule::FromTop);
        let (b1, l1) = apply(&board, &options[i].cells);
        board = b1;
        g.lines += l1;
        g.points += LINES_SCORE[l1.min(4) as usize];
        g.pieces += 1;
    }
    g
}

fn box_state() {
    let run = |cmd: &str, args: &[&str]| {
        std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_else(|| "n/a".into())
    };
    let vm = run("vm_stat", &[]);
    let free = vm
        .lines()
        .find(|l| l.starts_with("Pages free"))
        .unwrap_or("n/a")
        .to_owned();
    println!(
        "box: loadavg {} · {free} (16 KiB pages) · swap {}",
        run("sysctl", &["-n", "vm.loadavg"]),
        run("sysctl", &["-n", "vm.swapusage"])
    );
}

fn run_arm(
    arm: Arm,
    games: usize,
    cap: usize,
    rows: usize,
    fill: u64,
    threads: usize,
) -> Vec<Game> {
    let out = Mutex::new(vec![Game::default(); games]);
    let next_seed = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let k = next_seed.fetch_add(1, Ordering::Relaxed);
                    if k >= games {
                        break;
                    }
                    let g = play(k as u64 + 1, arm, cap, rows, fill);
                    out.lock().unwrap()[k] = g;
                }
            });
        }
    });
    out.into_inner().unwrap()
}

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let games: usize = arg(1).and_then(|a| a.parse().ok()).unwrap_or(20);
    let cap: usize = arg(2).and_then(|a| a.parse().ok()).unwrap_or(1000);
    let threads: usize = arg(3).and_then(|a| a.parse().ok()).unwrap_or(8);
    let budgets: Vec<u32> = arg(4)
        .unwrap_or_else(|| "100,400,1600".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let regimes: Vec<(usize, u64)> = arg(5)
        .unwrap_or_else(|| "16:75,18:75".into())
        .split(',')
        .filter_map(|s| {
            let (r, f) = s.split_once(':')?;
            Some((r.parse().ok()?, f.parse().ok()?))
        })
        .collect();

    println!(
        "== tetris_08_puct_goat — chance-node PUCT vs depth-2 exhaustive (Issue 892 T3 G3) =="
    );
    println!(
        "eval: eval_board(board_features(b, l), shaped) for BOTH players · physics FromTop · seeded 7-bag · seeds 1..={games} · cap {cap} · {threads} workers"
    );
    box_state();

    let mut arms = vec![Arm::Ply2];
    arms.extend(budgets.iter().map(|&b| Arm::Puct(b)));

    for &(rows, fill) in &regimes {
        println!();
        println!("-- regime: garbage {rows} rows @ {fill}% fill --");
        println!(
            "{:<34} {:>9} {:>9} {:>8} {:>9} {:>11} {:>14}",
            "player", "survived", "pieces/g", "lines/g", "points/g", "ms/decision", "pieces W/T/L"
        );
        let mut base: Option<Vec<Game>> = None;
        for &arm in &arms {
            let t = Instant::now();
            let res = run_arm(arm, games, cap, rows, fill, threads);
            let wall = t.elapsed().as_secs_f64();
            let surv = res.iter().filter(|g| g.pieces == cap).count();
            let pieces: usize = res.iter().map(|g| g.pieces).sum();
            let lines: u32 = res.iter().map(|g| g.lines).sum();
            let points: u64 = res.iter().map(|g| g.points).sum();
            let ns: u128 = res.iter().map(|g| g.decide_ns).sum();
            let decisions = pieces + res.iter().filter(|g| g.pieces < cap).count();
            let ms = ns as f64 / 1e6 / decisions.max(1) as f64;
            let wtl = match &base {
                None => "—".to_string(),
                Some(b) => {
                    let (mut w, mut tie, mut l) = (0, 0, 0);
                    for (x, y) in res.iter().zip(b) {
                        match x.pieces.cmp(&y.pieces) {
                            std::cmp::Ordering::Greater => w += 1,
                            std::cmp::Ordering::Equal => tie += 1,
                            std::cmp::Ordering::Less => l += 1,
                        }
                    }
                    format!("{w}/{tie}/{l}")
                }
            };
            println!(
                "{:<34} {:>9} {:>9.1} {:>8.1} {:>9.0} {:>11.2} {:>14}   ({wall:.0}s wall)",
                arm.name(),
                format!("{surv}/{games}"),
                pieces as f64 / games as f64,
                lines as f64 / games as f64,
                points as f64 / games as f64,
                ms,
                wtl
            );
            if base.is_none() {
                base = Some(res);
            }
        }
    }
    println!();
    box_state();
    println!("W/T/L = per-seed pieces placed vs the ply2 row (same seed, same bag, same garbage).");
}
