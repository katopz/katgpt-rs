//! tetris_07_laya_h2h — the live laya-vs-candidate Tetris head-to-head on
//! IDENTICAL seeds (katgpt-rs Issue 892 T5, Bench 892).
//!
//! Every player plays the same game per seed: the same seeded 7-bag
//! (`Bag::new(seed)`), the same start board (empty or
//! `garbage_board(seed, rows, fill)`), the same physics (`DropRule::FromTop`,
//! the v3 real hard drop) and the same guideline scoring (40/100/300/1200).
//!
//! **A player is one function** — `fn(&Board, cur, next) -> Option<usize>`,
//! an index into `landing_options_with(board, cur, FromTop)` (`None` = top
//! out). Plugging a new candidate is ONE match arm in [`player_by_name`].
//!
//! **The laya player** reproduces the reflex-site arena's laya lane
//! byte-for-byte (`reflex-site/assets/arena.js` `scoreOptions` +
//! `assets/games/tetris.js` `buildTurn`): per option, one `POST /decide`
//! with `state` = the v3 spot sentence ALONE (`render_spot_sentence` — the
//! spot sentence IS the state), one `noul` question `SPOT_QUESTION`, header
//! `X-Reflex-Lane: laya`; P(clean) = `answers[0].probabilities[0]`; the pick
//! is the first argmax over non-null p's (lowest index on ties). The arena
//! fires 6 concurrent requests per turn; the engine's laya actor serializes
//! forwards anyway. laya never sees the next piece — that is its arena
//! contract; the candidates get the preview.
//!
//! Transport: raw HTTP/1.1 over `std::net::TcpStream` (the engine answers
//! `Connection: close`); no new deps. Run the engine first:
//!
//! ```sh
//! # riir-reflex (any checkout; weights cache under ~/.cache/riir-reflex/laya)
//! CARGO_TARGET_DIR=/tmp/reflex-892-h2h cargo build --release --bin reflex \
//!     --features modelless,laya-riir,laya-riir-metal
//! RIIR_REFLEX_LAYA=1 RIIR_REFLEX_BIND=127.0.0.1:7392 /tmp/reflex-892-h2h/release/reflex
//! # katgpt-rs
//! cargo run --release --example tetris_07_laya_h2h -- \
//!     --player laya,ply1-classic,ply2-shaped --games 10 --cap 500 \
//!     [--garbage-rows 10 --garbage-fill 75] [--url 127.0.0.1:7392]
//! # protocol check: replay the arena's recorded laya (Rust) walk
//! cargo run --release --example tetris_07_laya_h2h -- \
//!     --verify-walk ../reflex-site/arena/demo_oracle.json
//! ```

#[path = "common/tetris_sim.rs"]
mod tetris_sim;
#[path = "common/tetris_lookahead.rs"]
mod tetris_lookahead;
#[path = "common/tetris_rulebook.rs"]
mod tetris_rulebook;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tetris_lookahead::{apply, garbage_board, pick, Bag, Player, LINES_SCORE};
use tetris_sim::{
    landing_options_with, outcome_features, render_spot_sentence, render_state_sentence, Board,
    DropRule, Piece, SPOT_QUESTION,
};

// ── The player registry (a new candidate = one arm) ──────────────────────

/// Index into `landing_options_with(board, cur, FromTop)`; `None` = top out.
type PlayerFn = fn(&Board, Piece, Piece) -> Option<usize>;

fn player_by_name(name: &str) -> Option<PlayerFn> {
    Some(match name {
        "laya" => laya_pick,
        "ply1-classic" => |b, c, n| pick(b, c, n, Player::Ply1Classic),
        "ply1-shaped" => |b, c, n| pick(b, c, n, Player::Ply1Shaped),
        "ply2-shaped" => |b, c, n| pick(b, c, n, Player::Ply2Shaped),
        // Issue 892 T4: the rulebook SCORE champion (9-1 stack + tetris
        // bonus + flat top, climbed on points, no hold — the arena has none).
        "rulebook-points" => |b, c, n| {
            static G: OnceLock<tetris_rulebook::Genome> = OnceLock::new();
            let g = G.get_or_init(tetris_rulebook::Genome::champion_points);
            tetris_rulebook::pick_no_hold(g, b, c, n)
        },
        // Same genome at depth 3 (beam 6 prior-pruned, the moka trick). The
        // harness signature carries no bag state, so the piece after the
        // preview is taken as uniform over a fresh bag here (the arena's
        // game loop uses the exact 7-bag remainder).
        "rulebook-points-d3" => |b, c, n| {
            static G: OnceLock<tetris_rulebook::Genome> = OnceLock::new();
            let g = G.get_or_init(|| {
                let mut g = tetris_rulebook::Genome::champion_points();
                g.depth = 3;
                g
            });
            tetris_rulebook::pick_no_hold(g, b, c, n)
        },
        _ => return None,
    })
}

const PLAYER_NAMES: &[&str] = &[
    "laya",
    "ply1-classic",
    "ply1-shaped",
    "ply2-shaped",
    "rulebook-points",
    "rulebook-points-d3",
];

// ── The laya client (the arena's laya lane, over raw HTTP/1.1) ───────────

struct LayaCfg {
    addr: String,
    concurrency: usize,
}

static LAYA: OnceLock<LayaCfg> = OnceLock::new();
/// Per-forward round-trip accounting (µs sum, count) — the per-option cost,
/// distinct from the per-decision wall the game loop measures.
static LAYA_RT_US: AtomicU64 = AtomicU64::new(0);
static LAYA_RT_N: AtomicU64 = AtomicU64::new(0);

/// One `POST /decide` with the arena's laya body; returns P(clean), or
/// `None` on an abstain (the arena's `outcome == null` rule).
fn laya_decide(addr: &str, state: &str) -> Result<Option<f64>, String> {
    let body = serde_json::json!({
        "state": state,
        "questions": [{ "id": "q0", "kind": "noul", "prompt": SPOT_QUESTION, "options": [] }],
    })
    .to_string();
    let t0 = Instant::now();
    let mut s = TcpStream::connect(addr).map_err(|e| format!("connect {addr}: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(120))).ok();
    let req = format!(
        "POST /decide HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nX-Reflex-Lane: laya\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut resp = Vec::with_capacity(1024);
    s.read_to_end(&mut resp).map_err(|e| e.to_string())?;
    LAYA_RT_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
    LAYA_RT_N.fetch_add(1, Ordering::Relaxed);
    let text = String::from_utf8_lossy(&resp);
    let (head, json) = text.split_once("\r\n\r\n").ok_or("malformed HTTP response")?;
    let status = head.lines().next().unwrap_or("");
    if !status.contains(" 200 ") {
        return Err(format!("{status}: {json}"));
    }
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let a = &v["answers"][0];
    if a.is_null() {
        return Err("no answer".into());
    }
    if a["outcome"].is_null() || a["outcome"]["noul"].is_null() {
        return Ok(None);
    }
    Ok(a["probabilities"][0].as_f64())
}

/// P(clean) for every option sentence, `concurrency` requests in flight
/// (the arena's worker pool). A transport error panics — a laya game played
/// on errors would be an unlabelled random game, never a laya row.
fn laya_score_all(sentences: &[String]) -> Vec<Option<f64>> {
    let cfg = LAYA.get().expect("laya client not configured");
    let next = AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<f64>>> =
        (0..sentences.len()).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..cfg.concurrency.min(sentences.len()).max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= sentences.len() {
                    break;
                }
                let p = laya_decide(&cfg.addr, &sentences[i])
                    .unwrap_or_else(|e| panic!("laya /decide failed: {e}"));
                *slots[i].lock().unwrap() = p;
            });
        }
    });
    slots.into_iter().map(|s| s.into_inner().unwrap()).collect()
}

/// First argmax over the non-null p's (lowest index on ties) — the arena's
/// `argmax`. `None` iff every option abstained.
fn first_argmax(ps: &[Option<f64>]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, p) in ps.iter().enumerate() {
        let Some(p) = p else { continue };
        match best {
            Some(b) if ps[b].unwrap() >= *p => {}
            _ => best = Some(i),
        }
    }
    best
}

fn spot_sentences(board: &Board, cur: Piece) -> Vec<String> {
    landing_options_with(board, cur, DropRule::FromTop)
        .iter()
        .map(|p| render_spot_sentence(board, p, &outcome_features(board, p)))
        .collect()
}

/// The laya player: the v3 option sentences only, argmax P(clean).
fn laya_pick(board: &Board, cur: Piece, _next: Piece) -> Option<usize> {
    let sentences = spot_sentences(board, cur);
    if sentences.is_empty() {
        return None;
    }
    let ps = laya_score_all(&sentences);
    // All-abstain never happens on the laya lane (it always answers); the
    // arena would fall back to a labelled random spot — refuse instead.
    Some(first_argmax(&ps).expect("laya abstained on every option"))
}

// ── The generic game loop ────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
struct Game {
    pieces: usize,
    lines: u32,
    points: u64,
    decide_us: u128,
}

fn play(seed: u64, player: PlayerFn, cap: usize, garbage_rows: usize, fill_pct: u64) -> Game {
    let mut bag = Bag::new(seed);
    let mut board = if garbage_rows > 0 {
        garbage_board(seed, garbage_rows, fill_pct)
    } else {
        Board::empty()
    };
    let mut next = bag.draw();
    let mut g = Game::default();
    while g.pieces < cap {
        let cur = next;
        next = bag.draw();
        let t0 = Instant::now();
        let choice = player(&board, cur, next);
        g.decide_us += t0.elapsed().as_micros();
        let Some(i) = choice else { break }; // topped out
        let options = landing_options_with(&board, cur, DropRule::FromTop);
        let (b1, l1) = apply(&board, &options[i].cells);
        board = b1;
        g.lines += l1;
        g.points += LINES_SCORE[l1.min(4) as usize];
        g.pieces += 1;
    }
    g
}

// ── Protocol check: replay the arena's recorded laya walk ────────────────

fn piece_of(id: &str) -> Piece {
    *Piece::ALL
        .iter()
        .find(|p| p.id() == id)
        .unwrap_or_else(|| panic!("unknown piece {id}"))
}

/// Re-trace `tetris_walk` (the arena's recorded laya (Rust) game): for each
/// turn, re-render the state + option sentences from the recorded board,
/// ask the live engine, and compare p's and the argmax with the recording;
/// then apply OUR pick and require the NEXT recorded board.
fn verify_walk(path: &str) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&text).expect("demo oracle json");
    let walk = v["tetris_walk"].as_array().expect("tetris_walk");
    let meta = &v["_meta"]["sources"]["tetris_laya"]["summary"];
    println!("recorded laya (Rust) walk: {} turns · summary {meta}", walk.len());
    let (mut state_ok, mut pick_ok, mut board_ok, mut n) = (0usize, 0usize, 0usize, 0usize);
    let mut max_dp = 0.0f64;
    let (mut lines, mut points) = (0u32, 0u64);
    for (t, rec) in walk.iter().enumerate() {
        let rows: Vec<&str> = rec[3]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        let board = Board::from_strings(&rows);
        let piece = piece_of(rec[2].as_str().unwrap());
        state_ok += usize::from(render_state_sentence(&board, piece) == rec[0].as_str().unwrap());
        let rec_ps: Vec<f64> = rec[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_f64().unwrap())
            .collect();
        let sentences = spot_sentences(&board, piece);
        assert_eq!(sentences.len(), rec_ps.len(), "turn {t}: option count");
        let ps = laya_score_all(&sentences);
        for (a, b) in ps.iter().zip(&rec_ps) {
            max_dp = max_dp.max((a.unwrap() - b).abs());
        }
        let ours = first_argmax(&ps).unwrap();
        let rec_pick = rec[4].as_u64().unwrap() as usize;
        pick_ok += usize::from(ours == rec_pick);
        let options = landing_options_with(&board, piece, DropRule::FromTop);
        let (b1, l1) = apply(&board, &options[ours].cells);
        lines += l1;
        points += LINES_SCORE[l1.min(4) as usize];
        if let Some(nxt) = walk.get(t + 1) {
            let want: Vec<String> = nxt[3]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_str().unwrap().to_string())
                .collect();
            board_ok += usize::from(b1.to_strings() == want);
        }
        if ours != rec_pick {
            println!("  turn {t}: pick {ours} vs recorded {rec_pick}");
        }
        n += 1;
    }
    println!(
        "state sentence match {state_ok}/{n} · argmax match {pick_ok}/{n} · next-board match {board_ok}/{} · max |Δp| {max_dp:.4} (recording rounds to 4 dp) · re-traced {} pieces {lines} lines {points} pts",
        n.saturating_sub(1),
        n
    );
}

// ── CLI ──────────────────────────────────────────────────────────────────

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let addr = arg(&args, "--url").unwrap_or("127.0.0.1:7392").to_string();
    let concurrency: usize = arg(&args, "--concurrency").map_or(6, |s| s.parse().unwrap());
    LAYA.set(LayaCfg { addr, concurrency }).ok();

    if let Some(path) = arg(&args, "--verify-walk") {
        verify_walk(path);
        return;
    }

    let games: u64 = arg(&args, "--games").map_or(10, |s| s.parse().unwrap());
    let seed0: u64 = arg(&args, "--seed-start").map_or(1, |s| s.parse().unwrap());
    let cap: usize = arg(&args, "--cap").map_or(500, |s| s.parse().unwrap());
    let rows: usize = arg(&args, "--garbage-rows").map_or(0, |s| s.parse().unwrap());
    let fill: u64 = arg(&args, "--garbage-fill").map_or(75, |s| s.parse().unwrap());
    let names: Vec<&str> = arg(&args, "--player")
        .unwrap_or("laya,ply1-classic,ply2-shaped")
        .split(',')
        .collect();
    let players: Vec<(&str, PlayerFn)> = names
        .iter()
        .map(|n| {
            let f = player_by_name(n)
                .unwrap_or_else(|| panic!("unknown player {n}; known: {PLAYER_NAMES:?}"));
            (*n, f)
        })
        .collect();
    let seeds: Vec<u64> = (seed0..seed0 + games).collect();
    let start = if rows > 0 {
        format!("garbage {rows} rows @ {fill}% fill")
    } else {
        "empty board".to_string()
    };
    println!(
        "h2h: seeds {seed0}..={} · {start} · cap {cap} · FromTop · 7-bag · 40/100/300/1200",
        seed0 + games - 1
    );

    // results[player][seed]
    let mut results: Vec<Vec<Game>> = Vec::with_capacity(players.len());
    for (name, f) in &players {
        let rt0 = (LAYA_RT_US.load(Ordering::Relaxed), LAYA_RT_N.load(Ordering::Relaxed));
        let mut row = Vec::with_capacity(seeds.len());
        for &seed in &seeds {
            let g = play(seed, *f, cap, rows, fill);
            eprintln!(
                "  {name:<14} seed {seed:>3}: {:>4} pieces {:>4} lines {:>6} pts",
                g.pieces, g.lines, g.points
            );
            row.push(g);
        }
        let dn = LAYA_RT_N.load(Ordering::Relaxed) - rt0.1;
        if dn > 0 {
            let dus = LAYA_RT_US.load(Ordering::Relaxed) - rt0.0;
            println!(
                "  {name}: {dn} /decide round-trips, mean {:.1} ms/forward",
                dus as f64 / dn as f64 / 1000.0
            );
        }
        results.push(row);
    }

    println!(
        "\n| player | games | survived (cap {cap}) | pieces/g | lines/g | points/g | ms/decision |"
    );
    println!("|---|---|---|---|---|---|---|");
    for ((name, _), row) in players.iter().zip(&results) {
        let n = row.len() as f64;
        let pieces: usize = row.iter().map(|g| g.pieces).sum();
        let us: u128 = row.iter().map(|g| g.decide_us).sum();
        println!(
            "| {name} | {} | {}/{} | {:.1} | {:.1} | {:.0} | {:.3} |",
            row.len(),
            row.iter().filter(|g| g.pieces == cap).count(),
            row.len(),
            pieces as f64 / n,
            row.iter().map(|g| g.lines as f64).sum::<f64>() / n,
            row.iter().map(|g| g.points as f64).sum::<f64>() / n,
            us as f64 / 1000.0 / pieces.max(1) as f64,
        );
    }

    print!("\n| seed |");
    for (name, _) in &players {
        print!(" {name} pieces/lines/pts |");
    }
    println!(" most pieces | most points |");
    print!("|---|");
    for _ in &players {
        print!("---|");
    }
    println!("---|---|");
    let mut wins_p = vec![0usize; players.len()];
    let mut wins_s = vec![0usize; players.len()];
    let winner = |key: &dyn Fn(&Game) -> u64, s: usize| -> Option<usize> {
        let best = results.iter().map(|r| key(&r[s])).max()?;
        let at: Vec<usize> = (0..results.len())
            .filter(|&p| key(&results[p][s]) == best)
            .collect();
        (at.len() == 1).then(|| at[0])
    };
    for (s, seed) in seeds.iter().enumerate() {
        print!("| {seed} |");
        for row in &results {
            let g = row[s];
            print!(" {}/{}/{} |", g.pieces, g.lines, g.points);
        }
        let wp = winner(&|g: &Game| g.pieces as u64, s);
        let ws = winner(&|g: &Game| g.points, s);
        if let Some(w) = wp {
            wins_p[w] += 1;
        }
        if let Some(w) = ws {
            wins_s[w] += 1;
        }
        let label = |w: Option<usize>| w.map_or("tie", |w| players[w].0);
        println!(" {} | {} |", label(wp), label(ws));
    }
    println!("\nsole per-seed wins (ties excluded):");
    for (p, (name, _)) in players.iter().enumerate() {
        println!("  {name}: most pieces {} · most points {}", wins_p[p], wins_s[p]);
    }
}
