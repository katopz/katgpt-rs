#![cfg(feature = "ternary_circuit")]
//! Issue 1225 Arm B — the exact-circuit builder's G1-B gates.
//!
//! Pins (pre-registered in the issue — all integer/pre-activation, NO exp):
//!
//! - **G1-B threshold law**: the compiled packed bytes equal an
//!   independent in-test re-pack of the hand-derived stamp table (double
//!   derivation — the Issue-983-F1 agreement law, read back through the
//!   shared T1 digit law), and the max-margin activation
//!   `out = os·(2·(dot − thresh) + 1)` lands the hand-computed outputs on
//!   a 4-byte line for: the plain match, the `requires` condition, the
//!   `vetoes` condition, both together, and a non-unit `out_scale`
//!   (exact f32 bits — every value is a small odd integer times os).
//! - **G1-B causal scans**: `RunningCount` and `ParityKernel` rows decode
//!   exactly and scan `x = [1,0,1,1]` to the hand-computed prefix values.
//! - **G1-B Paley**: the p=3 core equals the hand-computed 4×4 matrix; the
//!   quadratic-character table at p=19 matches the hand-listed residue set
//!   {1,4,5,6,7,9,11,16,17}; the p=3 and p=19 Sylvester blocks satisfy
//!   `B·Bᵀ = n·I` EXACTLY in i64; row-sign modes and the column pick land
//!   the hand-computed projection columns.
//! - **Zero-alloc apply** (the issue's third Arm-B line): all three
//!   `*_into` apply paths allocate NOTHING in steady state — pinned by the
//!   companion canary target `ternary_circuit_build_alloc_gates` (its own
//!   process; the counting allocator is process-global).

use katgpt_core::ternary_circuit::trit_at;
use katgpt_core::ternary_circuit_build::{
    CausalSpec, MAX_MODIFIERS, MAX_PATTERN, PALEY_MAX_SIDE, PALEY_PACKED_LEN, PLANES_PER_CHAR,
    PaleySpec, RowSignMode, THRESHOLD_PACKED_LEN, ThresholdSpec, pack_trits, quad_char,
};

// ── the hand-computed fixture tables ────────────────────────────────────

/// The active bit-plane set of a byte at a char block offset — the SAME
/// law the builder compiles, derived independently here (a plain loop over
/// bits, no packed-byte decode): `plane = 2·bit + bit_value`.
fn active_planes(byte: u8, block: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(7_usize);
    for bit in 0..7_usize {
        v.push(block * PLANES_PER_CHAR + 2 * bit + ((byte >> bit) & 1) as usize);
    }
    v
}

/// A byte's bit b (0-indexed, b < 7).
fn bit_of(byte: u8, b: usize) -> usize {
    ((byte >> b) & 1) as usize
}

const A: u8 = b'a'; // 0x61 — bits b0, b4, b5 set
const B: u8 = b'b'; // 0x62 — bits b1, b5 set
const X: u8 = b'x'; // 0x78 — bits b3, b4, b5 set
const SP: u8 = b' '; // 0x20 — bit b5 set

fn spec_ab(requires: u8, vetoes: u8, os: f32) -> ThresholdSpec {
    let mut pattern = [0_u8; MAX_PATTERN];
    pattern[0] = A;
    pattern[1] = B;
    ThresholdSpec {
        pattern,
        len: 2,
        requires,
        vetoes,
        out_scale: os,
    }
}

/// The independent hand stamp table for the "ab" spec (only blocks 0-1
/// carry stamps; modifier planes and every other block decode 0).
fn expected_ab_table() -> [i8; MAX_PATTERN * PLANES_PER_CHAR + MAX_MODIFIERS] {
    let mut expected = [0_i8; MAX_PATTERN * PLANES_PER_CHAR + MAX_MODIFIERS];
    for (i, &byte) in [A, B].iter().enumerate() {
        for bit in 0..7_usize {
            expected[i * PLANES_PER_CHAR + 2 * bit + bit_of(byte, bit)] = 1;
        }
    }
    expected
}

// ── the threshold idiom ─────────────────────────────────────────────────

#[test]
fn threshold_packed_bytes_match_a_local_repack_of_the_expected_table() {
    let prog = spec_ab(0, 0, 1.0).compile().expect("compile");
    // Deterministic compilation: two compiles, one byte answer.
    let prog2 = spec_ab(0, 0, 1.0).compile().expect("compile");
    assert_eq!(prog.packed_bytes(), prog2.packed_bytes());
    assert_eq!(prog.threshold(), 14, "thresh = 7·len");
    assert_eq!(
        prog.trit_len(),
        MAX_PATTERN * PLANES_PER_CHAR + MAX_MODIFIERS
    );

    // The program's bytes are the expected table packed at the shared law.
    let expected = expected_ab_table();
    let mut packed = [0_u8; THRESHOLD_PACKED_LEN];
    let written = pack_trits(&expected, &mut packed);
    assert_eq!(written, 47, "ceil(232/5)");
    assert_eq!(prog.packed_bytes(), packed);
    // …and the shared T1 unpacker reads the same table back out.
    for (i, &t) in expected.iter().enumerate() {
        assert_eq!(trit_at(&packed, i), t, "digit law trit {i}");
    }
}

#[test]
fn threshold_max_margin_law_hand_computed_outputs() {
    // Line = b"xab " — hand-computed in the issue record:
    //   c=0: dot 9  → −9   (x,a window: 5 + 4 matched planes)
    //   c=1: dot 14 → +1   (full match, margin os)
    //   c=2: dot 10 → −7   (b,space window: 4 + 6)
    //   c=3: dot 5  → −17  (space + past-end: 5 + 0 — 'a' and ' ' share
    //        zero-bits at b1,b2,b3,b6 plus the b5=1 plane → {2,4,6,11,12})
    let prog = spec_ab(0, 0, 1.0).compile().expect("compile");
    let line = [X, A, B, SP];
    let mut out = [0.0_f32; 4];
    prog.apply_into(&line, &[], &mut out);
    assert_eq!(out, [-9.0, 1.0, -7.0, -17.0]);

    // Cross-check the dots against the independent active-plane join.
    let a0 = active_planes(A, 0);
    let b1 = active_planes(B, 1);
    let wx = active_planes(X, 0);
    let wa = active_planes(A, 0);
    let wb = active_planes(B, 0);
    let wsp = active_planes(SP, 0);
    let windows: [&[usize]; 4] = [&wx, &wa, &wb, &wsp];
    let second: [Vec<usize>; 4] = [
        active_planes(A, 1),
        active_planes(B, 1),
        active_planes(SP, 1),
        Vec::new(), // past the line
    ];
    for (c, slot) in out.iter().enumerate() {
        let dot = a0.iter().filter(|p| windows[c].contains(p)).count()
            + b1.iter().filter(|p| second[c].contains(p)).count();
        let want = (2 * (dot as i64 - 14) + 1) as f32;
        assert_eq!(*slot, want, "column {c} matches the independent join");
    }

    // Non-unit scale: exact f32 bits of small odd integers × 3.
    let prog3 = spec_ab(0, 0, 3.0).compile().expect("compile");
    prog3.apply_into(&line, &[], &mut out);
    assert_eq!(out, [-27.0, 3.0, -21.0, -51.0]);
}

#[test]
fn threshold_requires_condition_hand_computed() {
    // requires = bit 0 → thresh = 15; the condition holds only at c=1.
    let prog = spec_ab(0b01, 0, 1.0).compile().expect("compile");
    assert_eq!(prog.threshold(), 15);
    let line = [X, A, B, SP];
    let conditions = [0_u8, 0b01, 0, 0];
    let mut out = [0.0_f32; 4];
    prog.apply_into(&line, &conditions, &mut out);
    // dot: 9, 15, 10, 5 → −11, +1, −9, −19.
    assert_eq!(out, [-11.0, 1.0, -9.0, -19.0]);
    // Absent condition columns read 0 — the same call with &[] behaves as
    // "the condition never holds": c=1 drops to dot 14 → −1 …
    prog.apply_into(&line, &[], &mut out);
    assert_eq!(out, [-11.0, -1.0, -9.0, -19.0]);
}

#[test]
fn threshold_veto_condition_hand_computed() {
    // vetoes = bit 1 → −1 stamp at the anchor, thresh stays 14.
    let prog = spec_ab(0, 0b10, 1.0).compile().expect("compile");
    assert_eq!(prog.threshold(), 14);
    let line = [X, A, B, SP];
    let conditions = [0_u8, 0b10, 0, 0];
    let mut out = [0.0_f32; 4];
    prog.apply_into(&line, &conditions, &mut out);
    // The veto kills exactly the firing column: dot 14 → 13 → −1.
    assert_eq!(out, [-9.0, -1.0, -7.0, -17.0]);

    // Both modifiers: requires bit0 + vetoes bit1, thresh 15, c=1 carries
    // both condition bits → dot 14 + 1 − 1 = 14 → −1 (held, but vetoed).
    let prog2 = spec_ab(0b01, 0b10, 1.0).compile().expect("compile");
    assert_eq!(prog2.threshold(), 15);
    let conditions2 = [0_u8, 0b11, 0, 0];
    prog2.apply_into(&line, &conditions2, &mut out);
    assert_eq!(out, [-11.0, -1.0, -9.0, -19.0]);
}

#[test]
fn threshold_refuses_bad_lengths() {
    let mut spec = spec_ab(0, 0, 1.0);
    spec.len = 0;
    assert!(matches!(
        spec.compile(),
        Err(katgpt_core::ternary_circuit_build::BuildError::BadPatternLen { .. })
    ));
    spec.len = MAX_PATTERN + 1;
    assert!(matches!(
        spec.compile(),
        Err(katgpt_core::ternary_circuit_build::BuildError::BadPatternLen { .. })
    ));
}

// ── the causal idiom ────────────────────────────────────────────────────

#[test]
fn causal_rows_decode_exactly() {
    let t = 4;
    let rc = CausalSpec::RunningCount.compile(t).expect("compile");
    let pk = CausalSpec::ParityKernel.compile(t).expect("compile");
    for row in 0..t {
        for s in 0..t {
            let want_rc = if s <= row { 1 } else { 0 };
            let want_pk = if s > row {
                0
            } else if (row - s) % 2 == 0 {
                1
            } else {
                -1
            };
            assert_eq!(rc.trit(row, s), want_rc, "running count [{row}][{s}]");
            assert_eq!(pk.trit(row, s), want_pk, "parity kernel [{row}][{s}]");
        }
    }
}

#[test]
fn causal_scans_hand_computed() {
    let x = [1_i32, 0, 1, 1];
    let mut y = [0_i32; 4];
    CausalSpec::RunningCount
        .compile(4)
        .expect("compile")
        .scan_into(&x, &mut y);
    assert_eq!(y, [1, 1, 2, 3], "running prefix count");
    CausalSpec::ParityKernel
        .compile(4)
        .expect("compile")
        .scan_into(&x, &mut y);
    assert_eq!(y, [1, -1, 2, -1], "alternating signed scan");
}

#[test]
fn causal_refuses_over_cap() {
    use katgpt_core::ternary_circuit_build::{BuildError, CAUSAL_MAX_T};
    assert!(matches!(
        CausalSpec::RunningCount.compile(0),
        Err(BuildError::OverCap { .. })
    ));
    assert!(matches!(
        CausalSpec::ParityKernel.compile(CAUSAL_MAX_T + 1),
        Err(BuildError::OverCap { .. })
    ));
}

// ── the Paley idiom ─────────────────────────────────────────────────────

/// The hand-computed p=3 Paley type-I core (H[i][j] = χ(i−j), diagonal −1,
/// row/col 0 all +1): χ(1) = +1, χ(2) = −1 mod 3.
const PALEY3_CORE: [[i8; 4]; 4] = [[1, 1, 1, 1], [1, -1, -1, 1], [1, 1, -1, -1], [1, -1, 1, -1]];

#[test]
fn paley_core_p3_matches_the_hand_matrix() {
    let proj = PaleySpec {
        prime: 3,
        block: false,
        out_cols: 4,
        row_signs: RowSignMode::AllPlus,
    }
    .compile()
    .expect("compile");
    assert_eq!(proj.shape(), (4, 4));
    for (r, row) in PALEY3_CORE.iter().enumerate() {
        for (c, &want) in row.iter().enumerate() {
            assert_eq!(proj.trit(r, c), want, "core [{r}][{c}]");
        }
    }
}

#[test]
fn paley_quadratic_characters_p19_hand_listed() {
    // Squares 1²..9² mod 19 — the hand residue list; Euler's criterion at
    // every a ∈ 1..19 must agree with it.
    let residues = [1_usize, 4, 9, 16, 6, 17, 11, 7, 5];
    for a in 1..19_i64 {
        let chi = quad_char(a, 19);
        let want = if residues.contains(&(a as usize)) {
            1
        } else {
            -1
        };
        assert_eq!(chi, want, "χ({a}) mod 19");
    }
}

#[test]
fn paley_block_orthogonality_exact_i64() {
    // The compile-time anchor covers the core; this gate re-derives the
    // BLOCK's orthogonality from the COMPILED projection bytes —
    // B·Bᵀ = n·I exact in i64, for both supported small and large primes.
    for (prime, side) in [(3_u32, 8_usize), (19, 40)] {
        let proj = PaleySpec {
            prime,
            block: true,
            out_cols: side,
            row_signs: RowSignMode::AllPlus,
        }
        .compile()
        .expect("compile");
        assert_eq!(proj.shape(), (side, side));
        for i in 0..side {
            for j in 0..side {
                let mut dot = 0_i64;
                for k in 0..side {
                    dot += proj.trit(i, k) as i64 * proj.trit(j, k) as i64;
                }
                let want = if i == j { side as i64 } else { 0 };
                assert_eq!(dot, want, "block [{i}][{j}] p={prime}");
            }
        }
    }
}

#[test]
fn paley_row_signs_and_column_pick_hand_computed() {
    // The p=3 block, column pick 3. Columns < m are UN-negated on the
    // block's lower half (Sylvester negates only the lower-right quadrant),
    // so column 0 is all +1 and the alternating law makes it (−1)^r:
    // [1,−1,1,−1,1,−1,1,−1]. Projection × e₀ pins both.
    let all_plus = PaleySpec {
        prime: 3,
        block: true,
        out_cols: 3,
        row_signs: RowSignMode::AllPlus,
    }
    .compile()
    .expect("compile");
    let alt = PaleySpec {
        prime: 3,
        block: true,
        out_cols: 3,
        row_signs: RowSignMode::Alternating,
    }
    .compile()
    .expect("compile");
    let x = [1_i32, 0, 0];
    let mut y_all = [0_i32; 8];
    let mut y_alt = [0_i32; 8];
    all_plus.apply_into(&x, &mut y_all);
    alt.apply_into(&x, &mut y_alt);
    assert_eq!(y_all, [1, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(y_alt, [1, -1, 1, -1, 1, -1, 1, -1]);

    // The column pick is the FIRST out_cols columns: projection × e₁
    // equals the block's column 1 (= the core's column 1 stacked un-negated).
    let x1 = [0, 1, 0];
    all_plus.apply_into(&x1, &mut y_all);
    let mut want = [0_i32; 8];
    for r in 0..8 {
        want[r] = PALEY3_CORE[r % 4][1] as i32;
    }
    assert_eq!(y_all, want);

    // The quadrant law, seen directly in the compiled trits: with the FULL
    // width, block row 4 is [H row 0 | −H row 0] — negated only past m.
    let full = PaleySpec {
        prime: 3,
        block: true,
        out_cols: 8,
        row_signs: RowSignMode::AllPlus,
    }
    .compile()
    .expect("compile");
    assert_eq!(full.shape(), (8, 8));
    for c in 0..8 {
        let want = if c < 4 {
            PALEY3_CORE[0][c]
        } else {
            -PALEY3_CORE[0][c - 4]
        };
        assert_eq!(full.trit(4, c), want, "block row 4 col {c}");
    }
    // …and e₄ (the first lower-right column) lands [H col 0 | −H col 0].
    let x5 = [0, 0, 0, 0, 1, 0, 0, 0];
    let mut y5 = [0_i32; 8];
    full.apply_into(&x5, &mut y5);
    for r in 0..8 {
        let core_col0 = PALEY3_CORE[r % 4][0] as i32;
        assert_eq!(y5[r], if r >= 4 { -core_col0 } else { core_col0 });
    }
}

#[test]
fn paley_refuses_bad_specs_and_caps_cover_the_domain() {
    use katgpt_core::ternary_circuit_build::BuildError;
    assert!(matches!(
        PaleySpec {
            prime: 5, // ≡ 1 mod 4 — outside the supported set
            block: false,
            out_cols: 2,
            row_signs: RowSignMode::AllPlus,
        }
        .compile(),
        Err(BuildError::UnsupportedPrime { .. })
    ));
    assert!(matches!(
        PaleySpec {
            prime: 3,
            block: false,
            out_cols: 5, // wider than the 4-row core
            row_signs: RowSignMode::AllPlus,
        }
        .compile(),
        Err(BuildError::ProjectionTooWide { .. })
    ));
    // PALEY_PACKED_LEN covers the worst case: 64×64 trits → 820 bytes.
    assert_eq!(PALEY_PACKED_LEN, 820);
    assert_eq!(
        (PALEY_MAX_SIDE * PALEY_MAX_SIDE).div_ceil(5),
        PALEY_PACKED_LEN
    );
    // The threshold cap covers its worst case: 232 trits → 47 ≤ 48 bytes
    // (a compile-time law — the const block fires at build, never at run).
    const {
        assert!(THRESHOLD_PACKED_LEN * 5 >= MAX_PATTERN * PLANES_PER_CHAR + MAX_MODIFIERS);
    }
    assert_eq!(MAX_MODIFIERS, 8);
}

// ── the zero-alloc apply canary lives in its OWN target
// (ternary_circuit_build_alloc_gates) — the counting allocator is
// process-global, so a shared binary would let this file's own fixture
// Vec allocations pollute the steady window. The Arm-A gate split.
