//! tetris_09_site_walk — record the HYBRID rulebook champion's game as a
//! reflex-site arena replay walk (katgpt-rs Issue 892, Bench 892).
//!
//! The piece stream is the reflex-site's own, ported bit-exactly:
//!
//! - `--stream bag` (default) — `assets/games/tetris_view.js` `PieceBag` over
//!   `new Rng(seed)`: when the queue is empty, copy `PIECES`
//!   (`I O T S Z J L`, == `Piece::ALL`), Fisher-Yates
//!   `for i = 6..1: j = rng.u32Below(i+1)`, then `pop()` from the END. The
//!   live arena boards' stream (`TetrisBoard.reset(607)`).
//! - `--stream uniform` — `scripts/record_demo_walks.mjs`
//!   `PIECES[rng.u32Below(7)]`: the stream every RECORDED demo walk
//!   (`tetris_walk`, `tetris_head_walk`, `tetris_raw_walk`,
//!   `tetris_python_walk`) was recorded on.
//!
//! `u32Below(n)` is fastrand 2.4.1 `gen_mod_u32(n)` (Lemire + rejection) =
//! `fastrand::Rng::u32(..n)`; `new Rng(seed)` is `Rng::with_seed(seed)`.
//! `--dump-stream N` prints the first N pieces for the JS parity proof.
//!
//! The game: empty board, `DropRule::FromTop` (v3), preview = the next
//! piece, hold OFF (`hold_ready: false, held: None`), depth-3 bag support =
//! the pieces left in the current 7-bag after `next` was drawn (`bag`), or
//! empty = uniform over all seven (`uniform` — the EXACT chance model of an
//! i.i.d. stream).
//!
//! Output (stdout or `--out`): `{walk, summary, info}` — walk rows are the
//! site's `tetris_walk` shape `[state_sentence, probs[], piece, board_rows
//! (BEFORE placement), chosen_index, per_option_ms[]]`. probs = sigmoid((v −
//! max v)/scale), scale = population std of the option root values (floor
//! 1e-9) — the pick reads 0.5, everything else below (sigmoid, never
//! softmax). per_option_ms = the decision's wall time / option count.
//!
//! Run:
//! ```sh
//! cargo run --release --example tetris_09_site_walk -- \
//!     [--seed 607] [--cap 300] [--stream bag|uniform] [--out walk.json]
//! cargo run --release --example tetris_09_site_walk -- --dump-stream 280
//! ```
//! Merge + verify: `reflex-site/scripts/merge_rulebook_walk.mjs walk.json`.

use katgpt_tetris::lookahead as tetris_lookahead;
use katgpt_tetris::rulebook as tetris_rulebook;
use katgpt_tetris::sim as tetris_sim;

use serde_json::{Value, json};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tetris_lookahead::{LINES_SCORE, apply};
use tetris_rulebook::{Decision, Genome, View, decide, decide_scored};
use tetris_sim::{Board, DropRule, Piece, landing_options_with, render_state_sentence};

/// The champion's pinned genome id (Bench 892).
const HYBRID_ID: &str = "68cae9d382014662";

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    Bag,
    Uniform,
}

/// The reflex-site piece stream (see the module docs).
struct SiteStream {
    rng: fastrand::Rng,
    kind: StreamKind,
    /// PieceBag queue; draws pop from the END (JS `Array.pop`).
    queue: Vec<Piece>,
}

impl SiteStream {
    fn new(seed: u64, kind: StreamKind) -> Self {
        Self {
            rng: fastrand::Rng::with_seed(seed),
            kind,
            queue: Vec::with_capacity(7),
        }
    }

    fn next(&mut self) -> Piece {
        match self.kind {
            StreamKind::Uniform => Piece::ALL[self.rng.u32(..7) as usize],
            StreamKind::Bag => {
                if self.queue.is_empty() {
                    self.queue.extend_from_slice(&Piece::ALL);
                    for i in (1..7usize).rev() {
                        let j = self.rng.u32(..(i as u32 + 1)) as usize;
                        self.queue.swap(i, j);
                    }
                }
                self.queue.pop().expect("refilled above")
            }
        }
    }

    /// The support of the next unseen piece, in draw order. Empty ⇒ uniform
    /// over all seven (a fresh bag, or the i.i.d. stream).
    fn support(&self) -> Vec<Piece> {
        match self.kind {
            StreamKind::Uniform => Vec::new(),
            StreamKind::Bag => self.queue.iter().rev().copied().collect(),
        }
    }
}

struct Args {
    seed: u64,
    cap: usize,
    kind: StreamKind,
    out: Option<String>,
    dump: Option<usize>,
}

fn parse_args() -> Args {
    let mut a = Args {
        seed: 607,
        cap: 300,
        kind: StreamKind::Bag,
        out: None,
        dump: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let val = || argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--seed" => a.seed = val().parse().expect("--seed u64"),
            "--cap" => a.cap = val().parse().expect("--cap usize"),
            "--out" => a.out = Some(val()),
            "--dump-stream" => a.dump = Some(val().parse().expect("--dump-stream usize")),
            "--stream" => {
                a.kind = match val().as_str() {
                    "bag" => StreamKind::Bag,
                    "uniform" => StreamKind::Uniform,
                    other => panic!("--stream bag|uniform, got {other:?}"),
                }
            }
            other => panic!("unknown arg {other:?}"),
        }
        i += 2;
    }
    a
}

/// Round to `dp` decimals (the recorded walks carry 4-decimal probs, 3-decimal ms).
fn round_to(x: f64, dp: i32) -> f64 {
    let k = 10f64.powi(dp);
    (x * k).round() / k
}

/// First strict maximum (the site's `argmax`: lowest index on ties).
fn first_argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

/// Sigmoid-mapped probs, rounded as coarsely as possible (4 dp) while the
/// site's argmax over them still equals `chosen` — escalating precision on
/// the rare row where rounding would tie an earlier option with the pick.
fn probs_for(values: &[f64], chosen: usize) -> Vec<f64> {
    let n = values.len() as f64;
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let scale = var.sqrt().max(1e-9);
    let raw: Vec<f64> = values
        .iter()
        .map(|v| 1.0 / (1.0 + (-(v - max) / scale).exp()))
        .collect();
    for dp in [4, 6, 9, 12] {
        let r: Vec<f64> = raw.iter().map(|&p| round_to(p, dp)).collect();
        if first_argmax(&r) == chosen {
            return r;
        }
    }
    assert_eq!(first_argmax(&raw), chosen, "unrounded probs lost the pick");
    raw
}

/// `YYYY-MM-DDTHH:MM:SS.mmmZ` (UTC) without a date crate.
fn iso_now() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock");
    let secs = d.as_secs() as i64;
    let (days, sod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        sod / 3600,
        sod % 3600 / 60,
        sod % 60,
        d.subsec_millis()
    )
}

/// `<short hostname> · <os>/<arch>` in the site recorder's node spelling.
fn host() -> String {
    let name = std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        o => o,
    };
    let arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        "x86_64" => "x64",
        a => a,
    };
    format!("{name} · {os}/{arch}")
}

/// Box load at record time (`uptime`'s load averages) — a latency figure
/// without its box state is not a measurement.
fn box_load() -> String {
    std::process::Command::new("uptime")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.split("load average")
                .nth(1)
                .map(|t| t.trim_start_matches(['s', ':', ' ']).trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    let a = parse_args();
    if let Some(n) = a.dump {
        let mut s = SiteStream::new(a.seed, a.kind);
        let pieces: String = (0..n).map(|_| s.next().id()).collect();
        println!("{pieces}");
        return;
    }

    let g = Genome::champion_hybrid();
    assert_eq!(g.id(), HYBRID_ID, "hybrid champion genome drifted");
    let mut stream = SiteStream::new(a.seed, a.kind);
    let mut board = Board::empty();
    let mut next = stream.next();
    let (mut score, mut lines, mut tetrises, mut pieces) = (0u64, 0u32, 0u32, 0usize);
    let mut topped_out = false;
    let mut walk: Vec<Value> = Vec::with_capacity(a.cap);
    let mut decision_ms: Vec<f64> = Vec::with_capacity(a.cap);

    while pieces < a.cap {
        let cur = next;
        next = stream.next();
        let support = stream.support();
        let opts = landing_options_with(&board, cur, DropRule::FromTop);
        if opts.is_empty() {
            topped_out = true;
            break;
        }
        let view = View {
            board: &board,
            cur,
            next,
            held: None,
            hold_ready: false,
            bag_remaining: &support,
        };
        let t0 = Instant::now();
        let scored = decide_scored(&g, &view);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        // Hold is off: exactly one root per landing option, in option order.
        assert_eq!(scored.len(), opts.len(), "root arity vs landing options");
        assert!(
            scored
                .iter()
                .enumerate()
                .all(|(i, (d, _))| !d.use_hold && d.index == i)
        );
        let values: Vec<f64> = scored.iter().map(|(_, v)| *v).collect();
        let chosen = first_argmax(&values);
        assert_eq!(
            decide(&g, &view),
            Some(Decision {
                use_hold: false,
                index: chosen
            }),
            "decide_scored argmax != decide at piece {pieces}"
        );
        decision_ms.push(ms);
        let per_opt = round_to(ms / opts.len() as f64, 3);
        walk.push(json!([
            render_state_sentence(&board, cur),
            probs_for(&values, chosen),
            cur.id(),
            board.to_strings(),
            chosen,
            vec![per_opt; opts.len()],
        ]));
        let (nb, l) = apply(&board, &opts[chosen].cells);
        board = nb;
        lines += l;
        tetrises += u32::from(l == 4);
        score += LINES_SCORE[l.min(4) as usize];
        pieces += 1;
    }

    let mut sorted = decision_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let p50 = sorted.get(sorted.len() / 2).map(|&m| round_to(m, 3));
    let stream_desc = match a.kind {
        StreamKind::Bag => format!(
            "PieceBag(new Rng({})) — assets/games/tetris_view.js 7-bag, the live-board stream",
            a.seed
        ),
        StreamKind::Uniform => format!(
            "PIECES[new Rng({}).u32Below(7)] — scripts/record_demo_walks.mjs, the recorded-walk stream",
            a.seed
        ),
    };
    let out = json!({
        "walk": walk,
        "summary": {
            "score": score,
            "lines": lines,
            "pieces": pieces,
            "abstains": 0,
            "decisions": decision_ms.len(),
            "p50_ms": p50,
            "tetrises": tetrises,
            "topped_out": topped_out,
        },
        "info": {
            "seed": a.seed,
            "stream": stream_desc,
            "cap": a.cap,
            "genome_id": g.id(),
            "genome": g.to_line(),
            "recorded_at": iso_now(),
            "host": host(),
            "load_avg": box_load(),
        },
    });
    let text = serde_json::to_string(&out).expect("serialize");
    match &a.out {
        Some(p) => std::fs::write(p, text).expect("write --out"),
        None => println!("{text}"),
    }
    eprintln!(
        "tetris_09_site_walk seed {} ({}): {pieces} pieces · {lines} lines · {tetrises} tetrises · {score} pts · topped_out {topped_out} · p50 {:?} ms/decision",
        a.seed,
        if a.kind == StreamKind::Bag {
            "bag"
        } else {
            "uniform"
        },
        p50
    );
}
