//! `ternary_circuit` Arm B — the exact-circuit builder (Issue 1225; Research 677 §2).
//!
//! Compiles explicit integer circuit SPECS into packed-ternary programs with
//! an EXACT integer-max-margin apply contract — no `exp`, no softmax, no
//! trained weight anywhere in the kit. Three idiom families, each with its
//! source site (brianbell-x/codehygiene @ [`crate::ternary_circuit::SOURCE_PIN`]):
//!
//! 1. **Threshold template circuits** (`compile.py:_sub_filter` :176 +
//!    `_stamp_char` :169): per template byte, the seven bit planes are
//!    stamped BOTH-polarity (14 planes per char; the plane matching the
//!    template bit carries +1), and the activation law folds to an exact
//!    max-margin detector — `out > 0` iff every template bit matches AND
//!    every required condition holds AND no veto fires. Integer margin, no
//!    float in the decision.
//! 2. **Causal prefix-scan matrices** (`__init__.py:193/197`'s
//!    `W_causal{1,2}` shape): lower-triangular ±1 matrices scanned
//!    left-to-right per line — the DFA-as-prefix-matmul carrier.
//! 3. **Paley-Hadamard ±1 projections** (`_paley_uniform_proj` :262):
//!    quadratic residues mod p → Paley type-I core → Sylvester block → a
//!    deterministic ±1 projection with EXACT integer orthogonality.
//!
//! # Scope laws (pre-registered in Issue 1225 Arm B)
//!
//! - **G1-B is integer/pre-activation**: fixtures pin exact integer outputs
//!   and exact f32 bits of the (integer-valued) scale fold — there is no
//!   `exp`/sigmoid/softmax anywhere in this module, and no quality claim.
//! - **The seeded constructions are OUT OF SCOPE**: the source's
//!   `RandomState(0)` row signs + permutation are replaced by EXPLICIT spec
//!   fields ([`RowSignMode`], first-`out_cols` column pick) — deterministic
//!   and hand-computable, never seeded. The source's seed-pinned ternary
//!   Gram fits (`_ternary_sketch_proj`) are likewise not attempted.
//! - **The input cell law is the builder's own idiom-level contract**, not
//!   the source's 52-channel cell census: a byte contributes its 7 bits as
//!   two-polarity one-hot planes (`plane = 2*bit + bit_value`); modifier
//!   condition bits (word-boundary, line-lead, in-quote, …) are CALLER
//!   inputs per column — the builder stamps, the caller owns semantics.
//! - **Zero-allocation apply**: every `*_into` apply path folds over packed
//!   bytes with sequential-ascending index arithmetic and fixed-size
//!   `[u8; N]` program storage (bounded domains pinned by const caps).
//!   Compilation may allocate; application may not (canary:
//!   `ternary_circuit_build_gates`).
//! - **Fold order pinned sequential-ascending** (Issue 940): every
//!   reduction is an explicit ascending accumulation; the indexed loops ARE
//!   the pinned order.

// Every indexed loop in this module IS the pinned sequential-ascending fold
// order (Issue 940): the index arithmetic names the storage law (plane =
// 2*bit + bit_value, char block = 14 planes, row-major trit matrices).
#![allow(clippy::needless_range_loop)]

use crate::ternary_circuit::{POW3, trit_at};

/// The source's bit census per template char: 7 bits × 2 polarity planes.
pub const PLANES_PER_CHAR: usize = 14;
/// The bounded pattern domain (`MAX_PATTERN`): 16 chars per threshold spec.
pub const MAX_PATTERN: usize = 16;
/// The bounded modifier domain: 8 caller-supplied condition channels.
pub const MAX_MODIFIERS: usize = 8;
/// Packed-byte cap for a threshold program:
/// `ceil((MAX_PATTERN * PLANES_PER_CHAR + MAX_MODIFIERS) / 5)` = 47 → 48.
pub const THRESHOLD_PACKED_LEN: usize = 48;
/// The bounded causal domain: T ≤ 64 (a line-scan window).
pub const CAUSAL_MAX_T: usize = 64;
/// Packed-byte cap for a causal matrix: `ceil(CAUSAL_MAX_T² / 5)` = 820.
pub const CAUSAL_PACKED_LEN: usize = 820;
/// The supported Paley primes (all ≡ 3 mod 4, core ≤ 32 → block ≤ 64).
pub const SUPPORTED_PRIMES: [u32; 6] = [3, 7, 11, 19, 23, 31];
/// The bounded Paley domain: block rows/cols ≤ 64 (prime 31's block).
pub const PALEY_MAX_SIDE: usize = 64;
/// Packed-byte cap for a Paley projection: `ceil(64² / 5)` = 820.
pub const PALEY_PACKED_LEN: usize = 820;

/// Everything that can go wrong compiling a circuit spec.
#[derive(Debug)]
pub enum BuildError {
    /// A threshold spec's pattern length is outside `1..=MAX_PATTERN`.
    BadPatternLen { len: usize },
    /// A Paley spec's prime is outside [`SUPPORTED_PRIMES`].
    UnsupportedPrime { prime: u32 },
    /// A Paley spec's column pick exceeds the compiled row width.
    ProjectionTooWide { cols: usize, rows: usize },
    /// A causal/paley dimension exceeds its const cap (internal law).
    OverCap {
        name: &'static str,
        got: usize,
        cap: usize,
    },
    /// The Paley core failed its exact orthogonality anchor — a construction
    /// bug, never a legal build (build-time `H·Hᵀ = m·I` in i64).
    Orthogonality { prime: u32 },
}

/// The row-sign law of a Paley projection. The source draws these from
/// `RandomState(0)` — OUT OF SCOPE per the issue; a builder user pins them
/// EXPLICITLY with one of these deterministic modes instead.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSignMode {
    /// Row `r` keeps the core's sign (all rows +1).
    AllPlus = 0,
    /// Row `r` is multiplied by `(−1)^r`.
    Alternating = 1,
}

// ── the 5-trits-per-byte packer ─────────────────────────────────────────
//
// The exact inverse of [`ternary_circuit::trit_at`] (the source's
// `__init__.py:265` digit law): digit `k` of byte `b` is `(b / 3^k) % 3`,
// trit = digit − 1 (digit 0 → trit 0, 1 → +1, 2 → −1). Zero padding is
// neutral because digit 0 decodes to trit 0 — the same law the T1
// assembly asserts for the archive tail.

/// Packs `trits` (each −1/0/+1) into `out` at the given byte offset using
/// the source's base-3 digit law — the exact inverse of
/// [`ternary_circuit::trit_at`]: digit = trit mod 3 (0 → digit 0, +1 → 1,
/// −1 → 2), byte = Σ digit·3^k. Returns the packed byte count written.
/// Any trit outside {−1, 0, +1} panics — a caller bug, not a runtime event.
pub fn pack_trits(trits: &[i8], out: &mut [u8]) -> usize {
    let packed_len = trits.len().div_ceil(5);
    assert!(
        out.len() >= packed_len,
        "pack_trits: out too short ({}, need {})",
        out.len(),
        packed_len
    );
    for b in out.iter_mut().take(packed_len) {
        *b = 0;
    }
    for (i, &t) in trits.iter().enumerate() {
        debug_assert!((-1..=1).contains(&t), "pack_trits: trit {t} at {i}");
        // digit = t mod 3: −1 → 2, 0 → 0, +1 → 1 — so a zero trit encodes
        // digit 0, which is what makes zero padding decode-neutral.
        let digit = if t < 0 { 2_u8 } else { t as u8 };
        out[i / 5] += digit * POW3[i % 5];
    }
    packed_len
}

/// Decodes trit `flat_index` from the packed bytes — the shared digit law.
/// Re-exports the T1 unpacker so builder and archive read identically.
#[inline]
pub fn packed_trit(buf: &[u8], flat_index: usize) -> i8 {
    trit_at(buf, flat_index)
}

// ── idiom 1: threshold template circuits (the `_sub_filter` shape) ──────

/// A threshold template spec: a byte pattern over [`MAX_PATTERN`] columns
/// plus caller-owned modifier conditions.
///
/// The activation law (all integer until the final scale fold):
/// `dot = Σ stamps · planes`, `out = out_scale · (2·(dot − thresh) + 1)`,
/// `thresh = 7·len + requires.count_ones()` — so `out > 0` iff every
/// template bit matches AND every required condition holds AND no veto
/// fires, with integer margin `out_scale · (2k+1)` at overshoot `k`
/// (the source's `scale = 2·out_scale, bias = −(2·thresh−1)·out_scale`
/// folded to the same inequality, `compile.py:176` @ the pin).
#[derive(Clone, Copy, Debug)]
pub struct ThresholdSpec {
    /// Template bytes; only `pattern[0..len]` is compiled. Bits 0..=6 are
    /// stamped (bit 7 is outside the 7-plane law and ignored).
    pub pattern: [u8; MAX_PATTERN],
    /// The pattern length (`1..=MAX_PATTERN`).
    pub len: usize,
    /// Bitmask of condition channels that MUST hold: channel `k` stamped
    /// +1 (at the window's anchor column) and counted in `thresh`.
    pub requires: u8,
    /// Bitmask of condition channels that must NOT hold: channel `k`
    /// stamped −1 (at the anchor column) and NOT counted in `thresh` —
    /// a firing veto kills the margin exactly.
    pub vetoes: u8,
    /// The post-fold scale (the source's `out_scale`; integer-valued in
    /// practice — 1.0 keeps the whole path integer).
    pub out_scale: f32,
}

/// A compiled threshold program: fixed-size packed trits + the integer
/// threshold + the post-fold scale.
#[derive(Clone, Debug)]
pub struct ThresholdProgram {
    packed: [u8; THRESHOLD_PACKED_LEN],
    trits: usize,
    thresh: i32,
    out_scale: f32,
}

impl ThresholdSpec {
    /// Compiles the spec: stamps the bit planes, folds the threshold law,
    /// packs at the shared digit law.
    pub fn compile(&self) -> Result<ThresholdProgram, BuildError> {
        if self.len == 0 || self.len > MAX_PATTERN {
            return Err(BuildError::BadPatternLen { len: self.len });
        }
        let total = MAX_PATTERN * PLANES_PER_CHAR + MAX_MODIFIERS;
        let mut trits = [0_i8; MAX_PATTERN * PLANES_PER_CHAR + MAX_MODIFIERS];
        let mut thresh = 0_i32;
        for i in 0..self.len {
            let byte = self.pattern[i];
            for bit in 0..7_usize {
                let bit_val = ((byte >> bit) & 1) as usize;
                trits[i * PLANES_PER_CHAR + 2 * bit + bit_val] = 1;
            }
            thresh += 7;
        }
        for k in 0..MAX_MODIFIERS {
            let plane = MAX_PATTERN * PLANES_PER_CHAR + k;
            if self.requires >> k & 1 == 1 {
                trits[plane] = 1;
                thresh += 1;
            }
            if self.vetoes >> k & 1 == 1 {
                trits[plane] = -1;
            }
        }
        let mut packed = [0_u8; THRESHOLD_PACKED_LEN];
        let written = pack_trits(&trits[..total], &mut packed);
        debug_assert_eq!(written, total.div_ceil(5), "packed-length law");
        Ok(ThresholdProgram {
            packed,
            trits: total,
            thresh,
            out_scale: self.out_scale,
        })
    }
}

impl ThresholdProgram {
    /// The compiled trit count (the fixed (pattern + modifier) layout).
    pub fn trit_len(&self) -> usize {
        self.trits
    }
    /// The compiled integer threshold (`7·len + requires`).
    pub fn threshold(&self) -> i32 {
        self.thresh
    }
    /// The packed bytes (inspection surface — fixed-size Copy).
    pub fn packed_bytes(&self) -> [u8; THRESHOLD_PACKED_LEN] {
        self.packed
    }

    /// Slides the template across `line`, writing one output per column.
    ///
    /// `conditions[c]` is the caller's modifier bitmask at column `c`
    /// (absent columns read as 0 — pass `&[]` for no modifiers anywhere).
    /// Zero-alloc: the fold decodes packed trits in ascending order and
    /// derives each plane's activity from the window byte on the fly —
    /// no materialized plane grid, no heap.
    ///
    /// The output is `out_scale · (2·(dot − thresh) + 1)` — positive iff
    /// the window fires. Columns whose window extends past the line read
    /// the missing bytes as all-zero planes (a short line never fires).
    pub fn apply_into(&self, line: &[u8], conditions: &[u8], out: &mut [f32]) {
        let trits = self.trits;
        for (c, slot) in out.iter_mut().enumerate() {
            let mut dot = 0_i64;
            for i in 0..trits {
                let t = packed_trit(&self.packed, i) as i64;
                if t == 0 {
                    continue;
                }
                let active = if i < MAX_PATTERN * PLANES_PER_CHAR {
                    // Pattern plane: char block = i / 14, plane = i % 14,
                    // bit = plane / 2, polarity = plane % 2.
                    let col = c + i / PLANES_PER_CHAR;
                    match line.get(col) {
                        Some(&b) => {
                            let plane = i % PLANES_PER_CHAR;
                            (((b >> (plane / 2)) & 1) as usize == plane % 2) as i64
                        }
                        None => 0,
                    }
                } else {
                    // Modifier plane at the window's anchor column.
                    let k = i - MAX_PATTERN * PLANES_PER_CHAR;
                    match conditions.get(c) {
                        Some(&m) => ((m >> k) & 1) as i64,
                        None => 0,
                    }
                };
                dot += t * active;
            }
            let over = dot - self.thresh as i64;
            *slot = self.out_scale * (2 * over + 1) as f32;
        }
    }
}

// ── idiom 2: causal prefix-scan matrices ────────────────────────────────

/// The causal row rule of a lower-triangular ±1 prefix matrix
/// (`W_causal{1,2}`'s shape, `__init__.py:193/197` @ the pin).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalSpec {
    /// `W[t][s] = +1` for `s ≤ t` — the running prefix count.
    RunningCount = 0,
    /// `W[t][s] = (−1)^(t−s)` for `s ≤ t` — the alternating signed scan
    /// (the linear kernel family the source's sawtooth gates build on).
    ParityKernel = 1,
}

/// A compiled causal matrix: fixed-size packed trits, row-major, strictly
/// lower-triangular-inclusive (zero above the diagonal).
#[derive(Clone, Debug)]
pub struct CausalMatrix {
    packed: [u8; CAUSAL_PACKED_LEN],
    t: usize,
}

impl CausalSpec {
    /// Compiles a `t × t` matrix under the spec's row rule.
    pub fn compile(&self, t: usize) -> Result<CausalMatrix, BuildError> {
        if t == 0 || t > CAUSAL_MAX_T {
            return Err(BuildError::OverCap {
                name: "causal_t",
                got: t,
                cap: CAUSAL_MAX_T,
            });
        }
        let mut trits = [0_i8; CAUSAL_MAX_T * CAUSAL_MAX_T];
        // Row-major at stride `t` (the packed region is exactly t×t).
        for row in 0..t {
            for s in 0..=row {
                trits[row * t + s] = match self {
                    CausalSpec::RunningCount => 1,
                    CausalSpec::ParityKernel => {
                        if (row - s) % 2 == 0 {
                            1
                        } else {
                            -1
                        }
                    }
                };
            }
        }
        let mut packed = [0_u8; CAUSAL_PACKED_LEN];
        let written = pack_trits(&trits[..t * t], &mut packed);
        debug_assert_eq!(written, (t * t).div_ceil(5), "packed-length law");
        Ok(CausalMatrix { packed, t })
    }
}

impl CausalMatrix {
    /// The compiled window size `t`.
    pub fn window(&self) -> usize {
        self.t
    }
    /// Matrix trit at `(row, s)` — the gate inspection surface.
    pub fn trit(&self, row: usize, s: usize) -> i8 {
        packed_trit(&self.packed, row * self.t + s)
    }

    /// Zero-alloc scan: `y[t] = Σ_{s ≤ t} W[t][s]·x[s]`, ascending `s`.
    /// Inputs are integer gate outputs (the source's {0,1} gate plane);
    /// `|x|` must stay ≤ `i32::MAX / t` for the i64 fold to narrow safely.
    pub fn scan_into(&self, x: &[i32], y: &mut [i32]) {
        assert_eq!(x.len(), self.t, "causal scan: x length");
        assert_eq!(y.len(), self.t, "causal scan: y length");
        for row in 0..self.t {
            let mut acc = 0_i64;
            for s in 0..=row {
                let w = packed_trit(&self.packed, row * self.t + s) as i64;
                acc += w * x[s] as i64;
            }
            y[row] = acc as i32;
        }
    }
}

// ── idiom 3: Paley-Hadamard ±1 projections ──────────────────────────────

/// A Paley-Hadamard projection spec (`_paley_uniform_proj` :262 @ the pin,
/// minus the OUT-OF-SCOPE seeded sign/perm — see [`RowSignMode`]).
///
/// Construction: the prime's quadratic residues build the Paley type-I
/// core `H` (size `p+1`: row 0 and column 0 all +1, diagonal −1,
/// `H[i][j] = χ(j−i)` otherwise, χ the quadratic character), the Sylvester
/// block doubles it to `2(p+1)` when `block` is set, [`RowSignMode`] pins
/// the row signs, and the projection keeps the FIRST `out_cols` columns —
/// a deterministic, hand-computable pick (the source's seeded permutation
/// is named-blocker territory, never approximated here).
#[derive(Clone, Copy, Debug)]
pub struct PaleySpec {
    /// A prime from [`SUPPORTED_PRIMES`] (all ≡ 3 mod 4).
    pub prime: u32,
    /// Double through the Sylvester block (`H → [[H, H], [H, −H]]`).
    pub block: bool,
    /// Columns kept, counted from column 0 (`≤ compiled row width`).
    pub out_cols: usize,
    /// The deterministic row-sign law (the source's `RandomState(0)` stand-in).
    pub row_signs: RowSignMode,
}

/// A compiled Paley projection: fixed-size packed trits, row-major, all
/// entries ±1 (the kit's only zero-free idiom).
#[derive(Clone, Debug)]
pub struct PaleyProjection {
    packed: [u8; PALEY_PACKED_LEN],
    rows: usize,
    cols: usize,
}

/// The quadratic character mod `p` at the residue class of `diff`
/// (signed; reduced mod p). `diff ≡ 0` is the caller's diagonal law and is
/// never consulted. Euler's criterion in exact u64 arithmetic — p ≤ 31
/// keeps every intermediate product far inside u64; no float anywhere.
pub fn quad_char(diff: i64, p: u32) -> i8 {
    let p64 = p as i64;
    let r = diff.rem_euclid(p64) as u64;
    debug_assert!(r != 0, "quad_char at 0 — the diagonal law owns that cell");
    let mut acc = 1_u64;
    let mut base = r;
    let mut e = (p as u64 - 1) / 2;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * base % p as u64;
        }
        base = base * base % p as u64;
        e >>= 1;
    }
    if acc == 1 { 1 } else { -1 }
}

impl PaleySpec {
    /// Compiles the projection, asserting the core's EXACT integer
    /// orthogonality (`H·Hᵀ = m·I` in i64) before packing — a construction
    /// bug refuses the build instead of shipping a wrong projection.
    pub fn compile(&self) -> Result<PaleyProjection, BuildError> {
        if !SUPPORTED_PRIMES.contains(&self.prime) {
            return Err(BuildError::UnsupportedPrime { prime: self.prime });
        }
        let p = self.prime as usize;
        let m = p + 1; // core side
        let side = if self.block { 2 * m } else { m };
        if side > PALEY_MAX_SIDE {
            return Err(BuildError::OverCap {
                name: "paley_side",
                got: side,
                cap: PALEY_MAX_SIDE,
            });
        }
        if self.out_cols == 0 || self.out_cols > side {
            return Err(BuildError::ProjectionTooWide {
                cols: self.out_cols,
                rows: side,
            });
        }
        // Paley type-I core, row-major: H[i][j] = χ(i−j) for i ≠ j,
        // i,j ≥ 1 (the standard indexing; χ(−1) = −1 at p ≡ 3 mod 4 makes
        // the core antisymmetric off-diagonal).
        let mut core = [0_i8; (PALEY_MAX_SIDE / 2 + 1) * (PALEY_MAX_SIDE / 2 + 1)];
        for j in 0..m {
            core[j] = 1;
            core[j * m] = 1;
        }
        for i in 1..m {
            core[i * m + i] = -1;
            for j in (i + 1)..m {
                let ch = quad_char(i as i64 - j as i64, self.prime);
                core[i * m + j] = ch;
                core[j * m + i] = -ch;
            }
        }
        // EXACT integer anchor: core·coreᵀ == m·I (i64, no float anywhere).
        for i in 0..m {
            for j in 0..m {
                let mut dot = 0_i64;
                for k in 0..m {
                    dot += core[i * m + k] as i64 * core[j * m + k] as i64;
                }
                let want = if i == j { m as i64 } else { 0 };
                if dot != want {
                    return Err(BuildError::Orthogonality { prime: self.prime });
                }
            }
        }
        // Row signs, then the Sylvester block, then the column pick.
        let sign = |r: usize| -> i8 {
            match self.row_signs {
                RowSignMode::AllPlus => 1,
                RowSignMode::Alternating => {
                    if r.is_multiple_of(2) {
                        1
                    } else {
                        -1
                    }
                }
            }
        };
        let mut trits = [0_i8; PALEY_MAX_SIDE * PALEY_MAX_SIDE];
        for r in 0..side {
            let sr = sign(r);
            for c in 0..self.out_cols {
                let v = if !self.block {
                    core[r * m + c]
                } else {
                    let h = core[(r % m) * m + (c % m)];
                    if r >= m && c >= m { -h } else { h }
                };
                trits[r * self.out_cols + c] = sr * v;
            }
        }
        let mut packed = [0_u8; PALEY_PACKED_LEN];
        let written = pack_trits(&trits[..side * self.out_cols], &mut packed);
        debug_assert_eq!(
            written,
            (side * self.out_cols).div_ceil(5),
            "packed-length law"
        );
        Ok(PaleyProjection {
            packed,
            rows: side,
            cols: self.out_cols,
        })
    }
}

impl PaleyProjection {
    /// The compiled shape (`rows`, `cols`).
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    /// Matrix trit at `(row, col)` — the gate inspection surface.
    pub fn trit(&self, row: usize, col: usize) -> i8 {
        packed_trit(&self.packed, row * self.cols + col)
    }

    /// Zero-alloc integer matvec: `y[r] = Σ_c W[r][c]·x[c]`, ascending `c`.
    /// `|x|` must stay ≤ `i32::MAX / rows` for the i64 fold to narrow safely.
    pub fn apply_into(&self, x: &[i32], y: &mut [i32]) {
        assert_eq!(x.len(), self.cols, "paley apply: x length");
        assert_eq!(y.len(), self.rows, "paley apply: y length");
        for r in 0..self.rows {
            let mut acc = 0_i64;
            for c in 0..self.cols {
                let w = packed_trit(&self.packed, r * self.cols + c) as i64;
                acc += w * x[c] as i64;
            }
            y[r] = acc as i32;
        }
    }
}
