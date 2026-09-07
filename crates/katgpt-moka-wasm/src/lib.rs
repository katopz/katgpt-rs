//! Minimal standalone WASM build of the Moka v1 port (Plan 565).
//!
//! Exists purely to answer one measurement question: is our native Rust
//! port of Moka's own network, compiled to WASM and run in a real browser,
//! faster than the real deployed Moka (which the same plan measured directly
//! at ~6.4 ms/move median in real Chrome — see `.docs/06_game_arenas/
//! go_arena.md`)? No `katgpt-core` dependency (so no ahash/getrandom wasm32
//! backend friction a browser deployment has no reason to carry), no game
//! engine beyond what's needed to generate realistic self-play positions.
//!
//! API shape deliberately mirrors the real Moka JS package
//! (`createGameState`/`encodeStudentFeatures`/`getLegalMoves`/`playMove`/
//! `selectHighestLegalMove`, from `github.com/millionco/moka`'s
//! `src/game.ts`) so the browser benchmark harness can drive both sides
//! with the same self-play loop shape.

pub mod board;
pub mod moka;
pub mod moka_int8;
pub mod puct;

// Issue 565 / Research 463: research hooks for the quant-error-LoRA PoC.
// Native-only (never enabled for the WASM browser build). Exposes weight
// accessors + a corrected forward pass for the riir-poc bench.
#[cfg(feature = "research")]
pub mod research;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmGame {
    board: board::Board,
    history: Vec<Option<(usize, usize)>>,
    /// Persistent, fixed-length, never-resized after construction — its
    /// address in wasm linear memory is therefore stable across every
    /// `encode_features_ptr` call. JS wraps that address in ONE
    /// `Float32Array` view (see `bench_ours.html`) instead of receiving a
    /// fresh marshalled array on every call.
    features_buf: Vec<f32>,
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            board: board::Board::new(),
            history: Vec::new(),
            features_buf: vec![0.0; moka::INPUT_ELEMENT_COUNT],
        }
    }

    /// Legal board-point move indices (0..81). Pass is always additionally
    /// legal but not listed here, matching the real Moka JS `getLegalMoves`.
    pub fn legal_moves(&self) -> Vec<u32> {
        self.board.legal_moves().into_iter().map(|i| i as u32).collect()
    }

    pub fn play(&mut self, idx: u32) {
        let idx = idx as usize;
        self.board.play(idx);
        self.history.push(Some((idx / board::SIZE, idx % board::SIZE)));
    }

    pub fn pass(&mut self) {
        self.board.pass();
        self.history.push(None);
    }

    /// Moka's 12-plane `9*9*12` HWC feature tensor for the current position,
    /// marshalled out as a fresh array on every call — the baseline API,
    /// kept for the A/B comparison against `encode_features_ptr`.
    pub fn encode_features(&self) -> Vec<f32> {
        let mut out = vec![0.0; moka::INPUT_ELEMENT_COUNT];
        moka::encode_features_into(&self.board, &self.history, &mut out);
        out
    }

    /// Writes the same tensor into `features_buf` and returns a pointer to
    /// it — zero-copy on the wasm side. JS reads the result through a
    /// `Float32Array` view over `wasm_memory()` instead of a marshalled
    /// return value.
    pub fn encode_features_ptr(&mut self) -> *const f32 {
        moka::encode_features_into(&self.board, &self.history, &mut self.features_buf);
        self.features_buf.as_ptr()
    }
}

impl Default for WasmGame {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
pub struct WasmMoka {
    weights: moka::MokaWeights,
    scratch: moka::MokaScratch,
    /// Persistent 83-float (`POLICY_MOVES` + value) output buffer — same
    /// stable-address idea as `WasmGame::features_buf`, for `infer_ptr`.
    out_buf: Vec<f32>,
}

#[wasm_bindgen]
impl WasmMoka {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            weights: moka::MokaWeights::load(),
            scratch: moka::MokaScratch::new(),
            out_buf: vec![0.0; moka::POLICY_MOVES + 1],
        }
    }

    /// Returns 83 floats: 82 policy logits (81 board points + pass at index
    /// 81), then the value estimate as the last element. Baseline API —
    /// `features` is marshalled INTO wasm on every call (a real bulk copy of
    /// the JS array into a temporary wasm buffer) and the `Vec<f32>` return
    /// is marshalled back OUT (allocate in wasm, copy to a fresh JS array,
    /// free the wasm buffer) — two real copies per call, kept for the A/B
    /// comparison against `infer_ptr`.
    pub fn infer(&mut self, features: &[f32]) -> Vec<f32> {
        let (policy, value) = moka::forward_with_scratch(&self.weights, features, &mut self.scratch);
        let mut out = policy.to_vec();
        out.push(value);
        out
    }

    /// Zero-copy variant: `features_ptr` must already point at
    /// `INPUT_ELEMENT_COUNT` valid floats already sitting in wasm linear
    /// memory (e.g. `WasmGame::encode_features_ptr()`'s return value — same
    /// memory space, so handing the pointer straight through costs nothing).
    /// Writes the result into this instance's persistent `out_buf` and
    /// returns a pointer to it, so JS never allocates or copies on either
    /// side of this call.
    ///
    /// # Safety
    /// `features_ptr` must be non-null and point to at least
    /// `INPUT_ELEMENT_COUNT` valid, initialized `f32`s for the lifetime of
    /// this call. Caller-enforced — this is the entire point of the raw
    /// pointer API (skip wasm-bindgen's safe-but-copying slice marshalling).
    pub unsafe fn infer_ptr(&mut self, features_ptr: *const f32) -> *const f32 {
        let features = unsafe { std::slice::from_raw_parts(features_ptr, moka::INPUT_ELEMENT_COUNT) };
        let (policy, value) = moka::forward_with_scratch(&self.weights, features, &mut self.scratch);
        self.out_buf[..moka::POLICY_MOVES].copy_from_slice(&policy);
        self.out_buf[moka::POLICY_MOVES] = value;
        self.out_buf.as_ptr()
    }
}

/// Exposes the WASM linear memory as a JS `WebAssembly.Memory` object, so
/// the browser harness can wrap `Float32Array` views directly over it —
/// the standard wasm-bindgen pattern for zero-copy JS↔wasm data sharing.
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

impl Default for WasmMoka {
    fn default() -> Self {
        Self::new()
    }
}

/// PUCT search player — combines the policy head (exploration prior) with
/// the value head (leaf evaluation) in an AlphaZero-style MCTS. This is the
/// Issue 204 port of `GoPuctMokaPlayer` from `katgpt-pruners`, adapted to
/// this crate's standalone `Board`. Exposed so the browser harness can
/// measure the combined PUCT+WASM latency that the prior doc sidestepped.
#[wasm_bindgen]
pub struct WasmPuctPlayer {
    inner: puct::PuctPlayer,
}

#[wasm_bindgen]
impl WasmPuctPlayer {
    /// `budget` = MCTS simulations per move (50/100/200 are the configs from
    /// Bench 205). `c_puct` = exploration constant (1.5 default, 2.5 high).
    /// `top_k` = beam width over policy priors (8 default).
    #[wasm_bindgen(constructor)]
    pub fn new(budget: usize, c_puct: f32, top_k: usize) -> Self {
        Self {
            inner: puct::PuctPlayer::new(budget, c_puct, top_k),
        }
    }

    /// Run a PUCT search from the given board position and return the chosen
    /// move: `Some(idx)` for a placement at flat board index `idx`, or `None`
    /// for a pass. The board is consumed as a flat array of 81 cells where
    /// 0=empty, 1=black, 2=white, plus `to_play` (0=black, 1=white) and an
    /// optional `ko_point` (flat index, or 255 for none).
    ///
    /// Returns `(u32, usize)`: `(move_or_255, nodes_evaluated)`. `move_or_255`
    /// is 255 for pass, otherwise the flat board index to play.
    pub fn search(&mut self, cells: &[u8], to_play: u8, ko_point: u32, consecutive_passes: u8) -> u32 {
        let board = decode_board(cells, to_play, ko_point, consecutive_passes);
        match self.inner.select_move(&board) {
            Some(idx) => idx as u32,
            None => 255, // sentinel for pass
        }
    }

    /// Number of value-head forward passes the most recent `search` performed.
    /// Useful for the bench harness to confirm the budget knob is actually
    /// controlling simulation count (sanity, not correctness).
    pub fn nodes_evaluated(&self) -> usize {
        self.inner.nodes_evaluated()
    }
}

/// PUCT player with int8 forward path (Issue 206 T5, promoted to default-on
/// in Issue 207). After promotion this is equivalent to `WasmPuctPlayer` —
/// kept as an explicit alias so existing JS code referencing `WasmPuctPlayerInt8`
/// continues to work. New code should prefer `WasmPuctPlayer` (the default
/// since Issue 207).
///
/// On aarch64 with dotprod, the int8 path is ~1.4× faster than f32; on WASM
/// with simd128, ~1.2× (Issue 207 measured PUCT b50 = 25.8ms, below the 30ms
/// floor). Win-rate parity confirmed at 85% (n=20) vs f32's 94% native.
#[wasm_bindgen]
pub struct WasmPuctPlayerInt8 {
    inner: puct::PuctPlayer,
}

#[wasm_bindgen]
impl WasmPuctPlayerInt8 {
    #[wasm_bindgen(constructor)]
    pub fn new(budget: usize, c_puct: f32, top_k: usize) -> Self {
        Self {
            inner: puct::PuctPlayer::with_int8(budget, c_puct, top_k),
        }
    }

    /// Run a PUCT search. See `WasmPuctPlayer::search` for the protocol.
    pub fn search(&mut self, cells: &[u8], to_play: u8, ko_point: u32, consecutive_passes: u8) -> u32 {
        let board = decode_board(cells, to_play, ko_point, consecutive_passes);
        match self.inner.select_move(&board) {
            Some(idx) => idx as u32,
            None => 255,
        }
    }

    pub fn nodes_evaluated(&self) -> usize {
        self.inner.nodes_evaluated()
    }
}

impl Default for WasmPuctPlayerInt8 {
    fn default() -> Self {
        Self::new(50, 1.5, 8)
    }
}

impl Default for WasmPuctPlayer {
    fn default() -> Self {
        Self::new(50, 1.5, 8)
    }
}

/// Decode the browser-supplied board encoding (flat `[u8; 81]` cells +
/// metadata) into this crate's `Board`. `ko_point` of 255 (or any value >= 81)
/// means "no ko".
fn decode_board(cells: &[u8], to_play: u8, ko_point: u32, consecutive_passes: u8) -> board::Board {
    let mut b = board::Board::new();
    for (i, &c) in cells.iter().take(board::AREA).enumerate() {
        b.cells[i] = match c {
            0 => board::Cell::Empty,
            1 => board::Cell::Black,
            2 => board::Cell::White,
            _ => board::Cell::Empty,
        };
    }
    b.to_play = if to_play == 0 { board::Cell::Black } else { board::Cell::White };
    b.ko_point = if (ko_point as usize) < board::AREA {
        Some(ko_point as usize)
    } else {
        None
    };
    b.consecutive_passes = consecutive_passes;
    b
}

/// Picks the legal move (or pass) with the highest policy logit — mirrors
/// the real Moka JS `selectHighestLegalMove`. `inference` is the 83-float
/// output of [`WasmMoka::infer`]; `legal` is [`WasmGame::legal_moves`]'s
/// output. Returns 81 for pass.
#[wasm_bindgen]
pub fn select_highest_legal_move(inference: &[f32], legal: &[u32]) -> u32 {
    let mut best_idx = 81u32;
    let mut best = inference[81];
    for &m in legal {
        let v = inference[m as usize];
        if v > best {
            best = v;
            best_idx = m;
        }
    }
    best_idx
}

// ── Raw C-ABI exports (Plan 565 Phase 4: wasmi comparison) ────────────────
//
// wasmi needs no JS interop at all, so these bypass wasm-bindgen's ABI
// entirely (which is designed for JS↔wasm marshalling, not native embedding)
// in favor of the same raw-linear-memory pattern this repo's own
// `wasm_pruner.rs` already uses with wasmi: caller allocates a buffer in
// wasm memory, writes floats into it, calls the compute function, reads the
// result back out. No JS runtime involved — this measures pure interpreted-
// wasm execution cost, isolated from any JS-boundary marshalling cost.

static mut WASMI_WEIGHTS: Option<moka::MokaWeights> = None;
static mut WASMI_SCRATCH: Option<moka::MokaScratch> = None;

/// Must be called once before `wasmi_infer`. Not thread-safe (fine — wasmi
/// benchmarking is single-threaded by construction, one `Store` per test).
#[unsafe(no_mangle)]
// `&raw mut` is intentional: it avoids forming a reference to the `static mut`
// (denied by Rust 2024). clippy::deref_addrof's suggested rewrite (`WASMI_WEIGHTS = ...`)
// would reintroduce that reference, so the lint is a false positive here.
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_init() {
    let _ = std::panic::catch_unwind(|| {
    unsafe {
        *(&raw mut WASMI_WEIGHTS) = Some(moka::MokaWeights::load());
        *(&raw mut WASMI_SCRATCH) = Some(moka::MokaScratch::new());
    }
});
}

/// Allocates a `len`-f32 buffer in wasm linear memory and returns its
/// pointer, so a wasmi host can write input features into it directly.
#[unsafe(no_mangle)]
pub extern "C" fn wasmi_alloc(len: usize) -> *mut f32 {
    let mut v = vec![0f32; len];
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

/// Runs one forward pass on the `POLICY_MOVES * 4` (features) input at
/// `features_ptr`, writing 83 floats (82 policy logits + value) to
/// `out_ptr`. Both pointers must come from [`wasmi_alloc`] (or otherwise be
/// valid, correctly-sized wasm-memory pointers) — this is a raw FFI
/// boundary, not a safe Rust API.
///
/// # Safety
/// `features_ptr` must point to at least `moka::INPUT_ELEMENT_COUNT` valid
/// `f32`s, and `out_ptr` to at least 83.
#[unsafe(no_mangle)]
// See `wasmi_init`: `&raw const`/`&raw mut` avoid forming references to the
// `static mut` (Rust 2024). clippy::deref_addrof is a false positive here.
#[allow(clippy::deref_addrof)]
pub unsafe extern "C" fn wasmi_infer(features_ptr: *const f32, out_ptr: *mut f32) {
    let _ = std::panic::catch_unwind(|| {
    unsafe {
        let features = std::slice::from_raw_parts(features_ptr, moka::INPUT_ELEMENT_COUNT);
        // `&raw` avoids ever forming a reference to the `static mut` itself
        // (Rust 2024 denies that) — single-threaded wasm32, one Store per
        // benchmark, so the aliasing this would otherwise risk can't happen.
        let weights = (*(&raw const WASMI_WEIGHTS)).as_ref().expect("wasmi_init not called");
        let scratch = (*(&raw mut WASMI_SCRATCH)).as_mut().expect("wasmi_init not called");
        let (policy, value) = moka::forward_with_scratch(weights, features, scratch);
        let out = std::slice::from_raw_parts_mut(out_ptr, moka::POLICY_MOVES + 1);
        out[..moka::POLICY_MOVES].copy_from_slice(&policy);
        out[moka::POLICY_MOVES] = value;
    }
});
}

// ── Issue 204: PUCT search via wasmi (mirrors `wasmi_infer`'s raw-pointer
// pattern). Lets the latency of the COMBINED build (PUCT + forward pass) be
// measured under the pure-interpreted wasm path — an upper bound on the
// real-Chrome JIT'd number, and apples-to-apples with the existing
// `wasmi_infer_latency` row in `go_arena.md`.
static mut WASMI_PUCT: Option<puct::PuctPlayer> = None;

/// Initialize the global PUCT player with `(budget, c_puct, top_k)`. Must be
/// called once before `wasmi_puct_search`. Single-threaded wasm32, one Store
/// per benchmark — not thread-safe by construction.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_puct_init(budget: usize, c_puct_bits: u32, top_k: usize) {
    let _ = std::panic::catch_unwind(|| {
    // f32 over a raw C ABI is awkward across wasm targets (some hosts widen
    // f32 args); pass the bit pattern as u32 and reconstruct. Matches how the
    // wasmi test reads the result back.
    let c_puct = f32::from_bits(c_puct_bits);
    unsafe {
        *(&raw mut WASMI_PUCT) = Some(puct::PuctPlayer::new(budget, c_puct, top_k));
    }
});
}

/// Run one PUCT search on the board encoded at `cells_ptr` (81 `u8` cells:
/// 0=empty, 1=black, 2=white), with `to_play` (0=black, 1=white),
/// `ko_point` (flat index, or 255 for none), and `consecutive_passes`.
/// Returns the chosen move: a flat board index (0..81) for a placement, or
/// 255 for pass.
///
/// # Safety
/// `cells_ptr` must point to at least 81 valid `u8`s.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub unsafe extern "C" fn wasmi_puct_search(
    cells_ptr: *const u8,
    to_play: u8,
    ko_point: u8,
    consecutive_passes: u8,
) -> u8 {
    let cells = unsafe { std::slice::from_raw_parts(cells_ptr, board::AREA) };
    let board = decode_board(cells, to_play, ko_point as u32, consecutive_passes);
    let player = unsafe { (*(&raw mut WASMI_PUCT)).as_mut() }.expect("wasmi_puct_init not called");
    match player.select_move(&board) {
        Some(idx) => idx as u8,
        None => 255,
    }
}

/// Number of value-head forward passes the most recent `wasmi_puct_search`
/// performed. Sanity knob for the bench harness.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_puct_nodes_evaluated() -> usize {
    unsafe { (*(&raw mut WASMI_PUCT)).as_mut() }.expect("wasmi_puct_init not called").nodes_evaluated()
}

// ── Issue 204 follow-up: full-game arena via wasmi (win-rate parity test) ──
//
// The latency test above answers "how fast is PUCT+WASM?". The honest gap
// left open (documented in `go_arena.md` Table C "Strength parity note") is
// "does the WASM port actually WIN at the native rate (94–98%)?". This
// section closes that gap by running complete PUCT-vs-greedy-Moka games
// entirely through the wasm binary — the strongest possible parity evidence,
// because the exact shipped wasm code is exercised end-to-end (board rules,
// feature encoding, forward pass, PUCT search, greedy argmax, scoring).
//
// wasmi is a deterministic IEEE-754 interpreter (same binary → same moves as
// Chrome's JIT — wasm spec mandates bit-identical execution modulo host
// bugs), so the win rate measured here IS the in-browser win rate. Slower
// (~46×, see Table C), but the moves chosen are identical, and win rate is an
// emergent property of move choices, not speed.
//
// Design: a single global `ArenaState` owns the board + move history + PUCT
// player + greedy (Moka weights/scratch). The host is a thin driver: reset →
// opening random moves → alternate PUCT/greedy searches (each advancing the
// board) → score. No board logic is duplicated on the host side.

struct ArenaState {
    board: board::Board,
    /// Full move history (last-2-plies feed the feature encoder). Mirrors the
    /// native `MokaPlayer::history` — both players read the SAME global history
    /// here, which is correct because both observe every ply.
    history: Vec<Option<(usize, usize)>>,
    puct: puct::PuctPlayer,
    weights: moka::MokaWeights,
    scratch: moka::MokaScratch,
    features_buf: Vec<f32>,
}

static mut WASMI_ARENA: Option<ArenaState> = None;

/// Initialize the arena: new empty board + PUCT player configured with
/// `(budget, c_puct_bits, top_k, batch_k)`. Must be called once before any other
/// `wasmi_arena_*` function. Resets any prior game state.
///
/// `batch_k`: 0 or 1 = sequential PUCT — **uses the int8 forward path by
/// default** (Issue 207 promotion; the wasmi parity floor was cleared at
/// 85% win rate, n=20). >1 = batched MCTS (virtual loss + leaf queue +
/// batched forward pass, Issue 205 — always f32 since batched int8 is
/// unimplemented).
///
/// To force the f32 path at K=1 (regression testing), use
/// [`wasmi_arena_init_f32`](fn.wasmi_arena_init_f32.html).
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_init(budget: usize, c_puct_bits: u32, top_k: usize, batch_k: usize) {
    let _ = std::panic::catch_unwind(|| {
    let c_puct = f32::from_bits(c_puct_bits);
    let state = ArenaState {
        board: board::Board::new(),
        history: Vec::new(),
        puct: puct::PuctPlayer::with_batch_k(budget, c_puct, top_k, batch_k),
        weights: moka::MokaWeights::load(),
        scratch: moka::MokaScratch::new(),
        features_buf: vec![0.0; moka::INPUT_ELEMENT_COUNT],
    };
    unsafe {
        *(&raw mut WASMI_ARENA) = Some(state);
    }
});
}

/// Initialize the arena with the **int8 forward path** enabled explicitly.
///
/// After the Issue 207 promotion, this is equivalent to
/// `wasmi_arena_init(budget, c_puct_bits, top_k, /*batch_k=*/1)` — both route
/// through the int8 forward path. Kept as an explicit alias so test code can
/// spell out "int8 path" at the call site (e.g. A/B benches against
/// `wasmi_arena_init_f32`).
///
/// All other `wasmi_arena_*` functions (`reset`, `play`, `search_puct`, etc.)
/// are shared with the f32 path — `PuctPlayer::select_move` dispatches
/// internally based on whether `weights_i8` is populated.
///
/// Used by `tests/wasmi_puct_int8_winrate.rs` (the Issue 207 promotion gate,
/// which confirmed 85% win rate at n=20 — clearing the 75% parity floor).
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_init_int8(budget: usize, c_puct_bits: u32, top_k: usize) {
    let _ = std::panic::catch_unwind(|| {
    let c_puct = f32::from_bits(c_puct_bits);
    let state = ArenaState {
        board: board::Board::new(),
        history: Vec::new(),
        puct: puct::PuctPlayer::with_int8(budget, c_puct, top_k),
        weights: moka::MokaWeights::load(),
        scratch: moka::MokaScratch::new(),
        features_buf: vec![0.0; moka::INPUT_ELEMENT_COUNT],
    };
    unsafe {
        *(&raw mut WASMI_ARENA) = Some(state);
    }
});
}

/// Initialize the arena with the **f32 forward path** (Issue 207 promotion
/// adds this as the explicit f32 escape hatch). Use for regression tests
/// against the pre-int8 baseline, or when int8 dot kernels are unavailable.
/// For normal use, prefer `wasmi_arena_init` (which defaults to int8 at K=1).
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_init_f32(budget: usize, c_puct_bits: u32, top_k: usize, batch_k: usize) {
    let _ = std::panic::catch_unwind(|| {
    let c_puct = f32::from_bits(c_puct_bits);
    let k = batch_k.max(1);
    let state = ArenaState {
        board: board::Board::new(),
        history: Vec::new(),
        puct: puct::PuctPlayer::with_f32(budget, c_puct, top_k),
        weights: moka::MokaWeights::load(),
        scratch: moka::MokaScratch::new(),
        features_buf: vec![0.0; k * moka::INPUT_ELEMENT_COUNT],
    };
    unsafe {
        *(&raw mut WASMI_ARENA) = Some(state);
    }
});
}

fn with_arena<R>(f: impl FnOnce(&mut ArenaState) -> R) -> R {
    // `&raw mut` avoids forming a reference to the `static mut` (Rust 2024
    // denies that). clippy::deref_addrof's suggested rewrite (`WASMI_ARENA`)
    // would reintroduce that reference — false positive, same as `wasmi_init`.
    #[allow(clippy::deref_addrof)]
    let state = unsafe { (*(&raw mut WASMI_ARENA)).as_mut() }.expect("wasmi_arena_init not called");
    f(state)
}

/// Reset to a fresh empty board + cleared history. PUCT player config is
/// preserved (re-init with `wasmi_arena_init` to change budget/c/top_k).
/// Called between games in a multi-game win-rate run.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_reset() {
    let _ = std::panic::catch_unwind(|| {
    with_arena(|s| {
        s.board = board::Board::new();
        s.history.clear();
    });
});
}

/// Play a stone at flat board index `idx` (0..81). Caller must have verified
/// legality (the host's `wasmi_arena_legal_count`/randomized-opening path
/// does so). Advances to_play + updates history.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_play(idx: usize) {
    let _ = std::panic::catch_unwind(|| {
    with_arena(|s| {
        s.board.play(idx);
        s.history.push(Some((idx / board::SIZE, idx % board::SIZE)));
    });
});
}

/// Pass. Advances to_play + updates history + increments consecutive-passes.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_pass() {
    let _ = std::panic::catch_unwind(|| {
    with_arena(|s| {
        s.board.pass();
        s.history.push(None);
    });
});
}

/// Write the current board's 81 cells (0=empty, 1=black, 2=white) into
/// `out_ptr`. Used by the host for diagnostics; the search functions read the
/// board internally via the global arena state, so the host does NOT need to
/// mirror board cells.
///
/// # Safety
/// `out_ptr` must point to at least 81 writable `u8`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wasmi_arena_get_cells(out_ptr: *mut u8) {
    let _ = std::panic::catch_unwind(|| {
    with_arena(|s| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, board::AREA) };
        for (i, &c) in s.board.cells.iter().enumerate() {
            out[i] = match c {
                board::Cell::Empty => 0,
                board::Cell::Black => 1,
                board::Cell::White => 2,
            };
        }
    });
});
}

/// Current player to play: 0=black, 1=white.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_to_play() -> u8 {
    with_arena(|s| match s.board.to_play {
        board::Cell::Black => 0,
        board::Cell::White => 1,
        board::Cell::Empty => 0,
    })
}

/// Number of legal non-pass moves at the current position. Used by the host's
/// randomized-opening loop to pick a random legal move.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_legal_count() -> u32 {
    with_arena(|s| s.board.legal_moves().len() as u32)
}

/// Fetch the `n`-th legal move (0-indexed, in the order `Board::legal_moves`
/// returns — ascending flat index). Returns 255 if `n >= legal_count`. The
/// host calls this after `legal_count` to implement a random legal opening ply.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_legal_move(n: u32) -> u32 {
    with_arena(|s| {
        let moves = s.board.legal_moves();
        moves.get(n as usize).map_or(255, |&i| i as u32)
    })
}

/// Run a PUCT search on the current board position, PLAY the chosen move on
/// the arena's board (advancing history + to_play), and return it: a flat
/// board index (0..81) for a placement, or 255 for pass. This mirrors the
/// native `GoPuctMokaPlayer::select_move` contract — the player both picks
/// and plays, so its internal history tracking stays consistent.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_search_puct() -> u8 {
    with_arena(|s| {
        let mv = s.puct.select_move(&s.board);
        if let Some(idx) = mv {
                s.board.play(idx);
                s.history.push(Some((idx / board::SIZE, idx % board::SIZE)));
                idx as u8
            } else {
                s.board.pass();
                s.history.push(None);
                255
            }
    })
}

/// Run ONE greedy forward pass (the real Moka greedy player: argmax over
/// policy logits including pass), PLAY the chosen move, and return it:
/// flat board index (0..81) for placement, or 255 for pass. This is the
/// exact algorithm `select_highest_legal_move` + `WasmMoka::infer` implement
/// for the browser — same weights, same forward pass, same argmax.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_search_greedy() -> u8 {
    with_arena(|s| {
        // Encode features from current board + history, forward pass.
        moka::encode_features_into(&s.board, &s.history, &mut s.features_buf);
        let (policy, _value) =
            moka::forward_with_scratch(&s.weights, &s.features_buf, &mut s.scratch);

        // Argmax over legal moves vs pass logit — mirrors MokaPlayer::select_move.
        let mut best_logit = policy[board::AREA]; // pass logit at index 81
        let mut best_move: Option<usize> = None;
        for i in s.board.legal_moves() {
            let logit = policy[i];
            if logit > best_logit {
                best_logit = logit;
                best_move = Some(i);
            }
        }

        if let Some(idx) = best_move {
                s.board.play(idx);
                s.history.push(Some((idx / board::SIZE, idx % board::SIZE)));
                idx as u8
            } else {
                s.board.pass();
                s.history.push(None);
                255
            }
    })
}

/// 1 if the game has ended (both players passed consecutively), else 0.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_is_over() -> u8 {
    with_arena(|s| u8::from(s.board.is_game_over()))
}

/// Area-score reward for `color` (0=black, 1=white): 1 if `color` is strictly
/// ahead after komi (7.5 to White), else 0. See `Board::reward`. The host
/// uses this to determine the game winner.
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_reward(color: u8) -> u8 {
    with_arena(|s| {
        let c = match color {
            0 => board::Cell::Black,
            _ => board::Cell::White,
        };
        u8::from(s.board.reward(c) > 0.5)
    })
}

/// Number of value-head forward passes the most recent `wasmi_arena_search_puct`
/// performed. Sanity knob for the bench harness (mirrors the standalone
/// `wasmi_puct_nodes_evaluated`, but reads the arena's PUCT player).
#[unsafe(no_mangle)]
#[allow(clippy::deref_addrof)]
pub extern "C" fn wasmi_arena_nodes_evaluated() -> usize {
    with_arena(|s| s.puct.nodes_evaluated())
}

// ── Issue 206: int8×int8 dot product benchmark exports ──────────────
//
// Bench 565 proved int8 SDOT is 2.5–6.3× faster per dot on native aarch64.
// These exports test whether the WASM extmul-based int8 dot kernel delivers
// a similar speedup under V8 JIT.
//
// Rust's stable `core::arch::wasm32` does NOT expose `i32x4_dot_i8x16_s`
// (the direct i8×i8→i32x4 dot product instruction). We use the extmul path:
//   1. i16x8_extmul_low_i8x16(a, b)  → 8 i16 products from low halves
//   2. i16x8_extmul_high_i8x16(a, b) → 8 i16 products from high halves
//   3. i32x4_extadd_pairwise_i16x8   → pairwise sum to i32x4
// This is 7 instructions for 16 multiplies vs f32's ~16 instructions.

/// WASM SIMD128 f32 dot product (the production kernel — separate mul+add).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn wasm_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128_load};
    unsafe {
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut acc2 = f32x4_splat(0.0);
        let mut acc3 = f32x4_splat(0.0);
        let mut i = 0usize;
        let chunks4 = len / 16;
        for _ in 0..chunks4 {
            acc0 = f32x4_add(f32x4_mul(v128_load(a.as_ptr().add(i).cast()), v128_load(b.as_ptr().add(i).cast())), acc0);
            acc1 = f32x4_add(f32x4_mul(v128_load(a.as_ptr().add(i + 4).cast()), v128_load(b.as_ptr().add(i + 4).cast())), acc1);
            acc2 = f32x4_add(f32x4_mul(v128_load(a.as_ptr().add(i + 8).cast()), v128_load(b.as_ptr().add(i + 8).cast())), acc2);
            acc3 = f32x4_add(f32x4_mul(v128_load(a.as_ptr().add(i + 12).cast()), v128_load(b.as_ptr().add(i + 12).cast())), acc3);
            i += 16;
        }
        let s01 = f32x4_add(acc0, acc1);
        let s23 = f32x4_add(acc2, acc3);
        let s = f32x4_add(s01, s23);
        let mut sum = f32x4_extract_lane::<0>(s) + f32x4_extract_lane::<1>(s) + f32x4_extract_lane::<2>(s) + f32x4_extract_lane::<3>(s);
        let remaining = (len - i) / 4;
        let mut acc = f32x4_splat(0.0);
        for _ in 0..remaining {
            acc = f32x4_add(f32x4_mul(v128_load(a.as_ptr().add(i).cast()), v128_load(b.as_ptr().add(i).cast())), acc);
            i += 4;
        }
        sum += f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc) + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc);
        while i < len {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

/// WASM SIMD128 int8 dot using extmul (stable Rust — no `i32x4_dot_i8x16_s`).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn wasm_dot_i8(a: &[i8], b: &[i8], len: usize) -> i32 {
    use core::arch::wasm32::{
        i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i32x4_add, i32x4_extadd_pairwise_i16x8,
        i32x4_extract_lane, v128_load,
    };
    unsafe {
        let mut acc0 = core::arch::wasm32::i32x4_splat(0);
        let mut acc1 = core::arch::wasm32::i32x4_splat(0);
        let mut i = 0usize;
        let chunks32 = len / 32; // 2 × 16-element blocks per iteration
        for _ in 0..chunks32 {
            let va = v128_load(a.as_ptr().add(i).cast());
            let vb = v128_load(b.as_ptr().add(i).cast());
            let lo = i16x8_extmul_low_i8x16(va, vb);
            let hi = i16x8_extmul_high_i8x16(va, vb);
            let lo32 = i32x4_extadd_pairwise_i16x8(lo);
            let hi32 = i32x4_extadd_pairwise_i16x8(hi);
            acc0 = i32x4_add(acc0, lo32);
            acc1 = i32x4_add(acc1, hi32);

            let va2 = v128_load(a.as_ptr().add(i + 16).cast());
            let vb2 = v128_load(b.as_ptr().add(i + 16).cast());
            let lo2 = i16x8_extmul_low_i8x16(va2, vb2);
            let hi2 = i16x8_extmul_high_i8x16(va2, vb2);
            let lo32_2 = i32x4_extadd_pairwise_i16x8(lo2);
            let hi32_2 = i32x4_extadd_pairwise_i16x8(hi2);
            acc0 = i32x4_add(acc0, lo32_2);
            acc1 = i32x4_add(acc1, hi32_2);
            i += 32;
        }
        // Process remaining 16-element chunks
        let chunks16 = (len - i) / 16;
        for _ in 0..chunks16 {
            let va = v128_load(a.as_ptr().add(i).cast());
            let vb = v128_load(b.as_ptr().add(i).cast());
            let lo = i16x8_extmul_low_i8x16(va, vb);
            let hi = i16x8_extmul_high_i8x16(va, vb);
            acc0 = i32x4_add(acc0, i32x4_extadd_pairwise_i16x8(lo));
            acc1 = i32x4_add(acc1, i32x4_extadd_pairwise_i16x8(hi));
            i += 16;
        }
        let s = i32x4_add(acc0, acc1);
        let mut sum = i32x4_extract_lane::<0>(s) + i32x4_extract_lane::<1>(s) + i32x4_extract_lane::<2>(s) + i32x4_extract_lane::<3>(s);
        while i < len {
            sum += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
            i += 1;
        }
        sum
    }
}

/// Benchmark: run `iters` f32 dot products of length `len` on the data at
/// `a_ptr` / `b_ptr`. Returns a bit-reinterpreted sink to prevent DCE.
/// The host times this call with `performance.now()` / `hrtime`.
///
/// # Safety
///
/// `a_ptr` and `b_ptr` must each point to at least `len` initialised,
/// properly aligned `f32`s that stay valid and unaliased for the whole call.
/// The kernel reads them with `get_unchecked` / `v128_load` and never
/// re-checks `len` against anything.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bench_dot_f32(a_ptr: *const f32, b_ptr: *const f32, len: usize, iters: usize) -> u32 {
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr, len);
        let b = std::slice::from_raw_parts(b_ptr, len);
        let mut sink = 0.0f32;
        for _ in 0..iters {
            sink = wasm_dot_f32(a, b, len);
        }
        sink.to_bits()
    }
}

/// Benchmark: run `iters` int8 dot products of length `len` on the data at
/// `a_ptr` / `b_ptr` (i8 slices reinterpreted from the raw pointers).
///
/// # Safety
///
/// `a_ptr` and `b_ptr` must each point to at least `len` initialised,
/// properly aligned `i8`s that stay valid and unaliased for the whole call.
/// Same unchecked-read contract as `bench_dot_f32`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bench_dot_i8(a_ptr: *const i8, b_ptr: *const i8, len: usize, iters: usize) -> i32 {
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr, len);
        let b = std::slice::from_raw_parts(b_ptr, len);
        let mut sink = 0i32;
        for _ in 0..iters {
            sink = wasm_dot_i8(a, b, len);
        }
        sink
    }
}
