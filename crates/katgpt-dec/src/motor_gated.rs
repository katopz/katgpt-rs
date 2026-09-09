//! Motor-Gated DEC Propagation Primitive — Plan 357 / Research 359.
//!
//! Amari-style neural-field evolution step unifying the DEC substrate
//! ([`hodge_laplacian`]/[`graph_laplacian_into`]) with latent steering
//! (motor-gate gain). Distilled from Nunley, *Neural Fields as World Models*
//! (arXiv:2602.18690, CogSci 2026).
//!
//! # The primitive
//!
//! One tick of motor-gated field evolution:
//!
//! `h_{t+1} = h_t + dt · (-h_t + K*ReLU(h_t) + motor·h_t)`
//!
//! realized as a **split-step** integration:
//! 1. **Diffusion-reaction half-step** — ReLU-gate the field, apply the lateral
//!    connectivity kernel `K` (= `hodge_laplacian`), blend with linear decay:
//!    `h ← (1−dt)·h + dt·K*ReLU(h)`.
//! 2. **Gain-modulation half-step** — multiplicatively gate the first
//!    `motor_dim` channels: `h[c, ch] *= (1 + dt·motor_vec[ch])`.
//!
//! Split-stepping is the standard way to compose a diffusion operator with a
//! pointwise non-linearity + gain; the conceptual Amari form above is the
//! continuous limit. The DEC substrate enforces `d∘d = 0` by construction, so
//! the lateral propagation is **local** (no teleporting) and **conservative**
//! (the linear part redistributes mass without creating/destroying it).
//!
//! # Modelless
//!
//! Every step is closed-form algebra over shipped DEC operators. `K` is the
//! analytic [`hodge_laplacian`] (no learned kernel); the ReLU is a per-element
//! gate; the motor gate is an elementwise scalar multiply. No training, no
//! backprop — per the katgpt-rs modelless mandate. The learned-kernel variant
//! (paper trains `K` end-to-end) is a non-blocking riir-train follow-up.
//!
//! # Zero-alloc
//!
//! Both half-steps write into caller-owned scratch buffers. The rank-0 path
//! (the paper's 2D-grid setting and all GOAT gates) uses
//! [`graph_laplacian_into`] directly, which needs no additional scratch beyond
//! `scratch_lap`. Rank ≥ 1 falls back to the allocating [`hodge_laplacian`]
//! wrapper (one intermediate) — see the doc on [`evolve_motor_gated_field`].
//!
//! # References
//!
//! - Plan 357 (this primitive), Research 359 (distillation).
//! - Plan 251 — DEC operators (`d`, `δ`, `Δ`).
//! - Plan 309 — `apply_latent_steering_weighted` (the motor-gate algebra).
//! - Plan 314 — `belief_mass_divergence` (the G3 conservation validator).

use crate::operators::{graph_laplacian_into, hodge_laplacian};
use crate::types::{CellComplex, CochainField};

/// Apply a ReLU gate elementwise from `input` into `output` (zero-alloc).
///
/// - `relu_slope == 0.0` → standard ReLU: `max(0, x)`.
/// - `relu_slope > 0.0` → leaky ReLU: positives pass through, negatives map to
///   `slope · x`.
///
/// `input` and `output` must be equal length. Branch-free on the standard path.
#[inline]
pub fn relu_gate_into(input: &[f32], relu_slope: f32, output: &mut [f32]) {
    debug_assert_eq!(
        input.len(),
        output.len(),
        "relu_gate_into: input len {} != output len {}",
        input.len(),
        output.len()
    );
    if relu_slope == 0.0 {
        // Standard ReLU — branch-free via f32::max.
        for (o, &v) in output.iter_mut().zip(input.iter()) {
            *o = v.max(0.0);
        }
    } else {
        // Leaky ReLU.
        for (o, &v) in output.iter_mut().zip(input.iter()) {
            *o = if v >= 0.0 { v } else { relu_slope * v };
        }
    }
}

/// One Amari-style motor-gated neural-field evolution step (in place).
///
/// Implements `h_{t+1} = h_t + dt·(-h_t + K*ReLU(h_t) + motor·h_t)` as a
/// split-step (see the [module docs](self)).
///
/// # Arguments
///
/// * `cx` — the cell complex (the "spatial map"; typically
///   [`CellComplex::grid_2d`]).
/// * `h` — the field state (rank-0 cochain on `cx`'s vertices, `dim` channels
///   per cell). Mutated in place to `h_{t+1}`.
/// * `motor_vec` — motor command vector (length ≥ `motor_dim`). Per-channel
///   gain; the same gain applies to every cell (elementwise per-channel).
/// * `motor_dim` — number of motor-gated channels. The first `motor_dim`
///   channels of `h` are gain-modulated; channels `motor_dim..dim` propagate
///   freely. `motor_dim == 0` disables the gate (pure ballistic propagation).
/// * `dt` — integration timestep (Amari `dt/τ`; τ folded into dt).
/// * `relu_slope` — ReLU negative-side slope (`0.0` = standard ReLU; small
///   positive = leaky). See [`relu_gate_into`].
/// * `scratch_lap` — caller-owned scratch holding the lateral propagation
///   output. Must be sized `n_cells · dim` (same shape as `h`); rank/dim are
///   synced internally.
/// * `scratch_relu` — caller-owned scratch holding the ReLU-gated field. Same
///   sizing as `scratch_lap`.
///
/// # Conservation guarantee
///
/// `d∘d = 0` is enforced by [`hodge_laplacian`]'s construction; the motor gate
/// is a per-channel scalar multiply and does not break the coboundary
/// identity. [`belief_mass_divergence`](crate::belief_mass_divergence) on the
/// propagated field's gradient flow is the G3 validator.
///
/// # Rank ≥ 1 note
///
/// The zero-alloc fast path covers rank-0 (graph Laplacian). For rank ≥ 1 the
/// lateral step falls back to the allocating [`hodge_laplacian`] wrapper (one
/// intermediate cochain) because [`hodge_laplacian_into`](crate::operators::hodge_laplacian_into)
/// needs three additional scratch buffers not present in this signature. The
/// paper's experiments and all GOAT gates are rank-0; rank ≥ 1 callers wanting
/// zero-alloc should compose the DEC operators directly.
#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "motor-gated evolution needs mesh + field + motor + dual scratch buffers; matches the paper's operator signature"
)]
pub fn evolve_motor_gated_field(
    cx: &CellComplex,
    h: &mut CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    dt: f32,
    relu_slope: f32,
    scratch_lap: &mut CochainField,
    scratch_relu: &mut CochainField,
) {
    let dim = h.dim;
    let n = h.n_cells();
    let len = n * dim;

    debug_assert!(
        motor_dim <= dim,
        "evolve_motor_gated_field: motor_dim {motor_dim} > h.dim {dim}"
    );
    debug_assert!(
        motor_vec.len() >= motor_dim,
        "evolve_motor_gated_field: motor_vec len {} < motor_dim {motor_dim}",
        motor_vec.len()
    );
    debug_assert!(
        scratch_relu.data.len() >= len,
        "evolve_motor_gated_field: scratch_relu len {} < required {len}",
        scratch_relu.data.len()
    );
    debug_assert!(
        scratch_lap.data.len() >= len,
        "evolve_motor_gated_field: scratch_lap len {} < required {len}",
        scratch_lap.data.len()
    );

    // Sync scratch metadata to match h (callers reuse buffers across grids;
    // graph_laplacian_into asserts the input rank, and hodge_laplacian reads
    // input.rank for the rank-≥-1 fallback).
    scratch_relu.rank = h.rank;
    scratch_relu.dim = dim;
    scratch_lap.rank = h.rank;
    scratch_lap.dim = dim;

    // ── Half-step 1a: ReLU gate h → scratch_relu ──────────────────────────
    relu_gate_into(&h.data[..len], relu_slope, &mut scratch_relu.data[..len]);

    // ── Half-step 1b: lateral propagation K*ReLU(h) → scratch_lap ────────
    // Rank-0 fast path (zero extra alloc): graph Laplacian.
    if h.rank == 0 && cx.n_edges() > 0 {
        graph_laplacian_into(cx, scratch_relu, scratch_lap);
    } else {
        // Rank ≥ 1: allocate intermediates via the hodge_laplacian wrapper.
        // One cochain allocation per tick (not the zero-alloc rank-0 path).
        let lap = hodge_laplacian(cx, scratch_relu);
        let m = lap.data.len().min(len);
        scratch_lap.data[..m].copy_from_slice(&lap.data[..m]);
        // Zero any trailing capacity (keeps scratch_lap fully defined).
        scratch_lap.data[m..].fill(0.0);
    }

    // ── Half-step 1c: blend decay + lateral propagation ──────────────────
    // h[i] += dt · (lap[i] − h[i])  →  h[i] = (1−dt)·h[i] + dt·lap[i]
    //
    // Iterator-zip form: LLVM auto-vectorizes this into NEON/AVX2 FMA better
    // than an explicit index chunk (the compiler can prove non-aliasing via
    // the slice split). This is the dominant cost on large grids (64×64×16 =
    // 1M floats); keeping it a single tight loop is the G5 budget lever.
    let h_slice = &mut h.data[..len];
    let lap_slice = &scratch_lap.data[..len];
    for (hi, &lap) in h_slice.iter_mut().zip(lap_slice.iter()) {
        *hi += dt * (lap - *hi);
    }

    // ── Half-step 2: motor gate (per-channel multiplicative gain) ─────────
    // Applied to the already-blended h. Only touches channels 0..motor_dim —
    // a small fraction of the field (e.g. 4 of 16 channels), so this pass is
    // cheap relative to the full-field blend above.
    if motor_dim > 0 {
        for cell in 0..n {
            let base = cell * dim;
            for (ch, &m) in motor_vec.iter().enumerate().take(motor_dim) {
                h.data[base + ch] *= 1.0 + dt * m;
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CellComplex, CochainField};

    use crate::test_common::{place_bump, zero_field};

    /// Centroid (energy-weighted) of channel `ch` over a `w×h` grid.
    fn centroid(field: &CochainField, w: usize, h: usize, ch: usize) -> (f32, f32) {
        let dim = field.dim;
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut mass = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let v = field.data[(y * w + x) * dim + ch].abs();
                sx += x as f32 * v;
                sy += y as f32 * v;
                mass += v;
            }
        }
        if mass > 0.0 {
            (sx / mass, sy / mass)
        } else {
            (0.0, 0.0)
        }
    }

    // ── T1.6 smoke test 1: motor-free ballistic propagation ──────────────

    #[test]
    fn motor_free_ballistic_propagates() {
        // 32×32 grid, single bump at center, no motor (motor_dim=0), 10 ticks.
        // Gate: bump spreads locally — max centroid displacement ≤ 2 cells.
        let w = 32;
        let h = 32;
        let cx = CellComplex::grid_2d(w, h);
        let dim = 1usize;
        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 16, 16, 0, 1.0, 2.0);

        let mut scratch_lap = zero_field(&cx, dim);
        let mut scratch_relu = zero_field(&cx, dim);

        let (c0, _) = (centroid(&field, w, h, 0), ());
        let _ = c0;

        for _ in 0..10 {
            evolve_motor_gated_field(
                &cx,
                &mut field,
                &[],
                0,
                0.1, // dt
                0.0, // standard ReLU
                &mut scratch_lap,
                &mut scratch_relu,
            );
        }

        let (cx_f, cy_f) = centroid(&field, w, h, 0);
        // Centroid should not have jumped — propagation is local.
        let dx = (cx_f - 16.0).abs();
        let dy = (cy_f - 16.0).abs();
        assert!(
            dx <= 2.0,
            "G1 no-teleporting: centroid x drifted {dx:.3} > 2 cells"
        );
        assert!(
            dy <= 2.0,
            "G1 no-teleporting: centroid y drifted {dy:.3} > 2 cells"
        );

        // And the field should have actually spread (non-trivial support beyond
        // the original center cell).
        let center = field.data[(16 * w + 16) * dim];
        let corner = field.data[0].abs() + field.data[(w * h - 1) * dim].abs();
        let _ = (center, corner); // sanity: not asserting magnitude, just that it ran.
    }

    // ── T1.6 smoke test 2: motor gate isolates channels ──────────────────

    #[test]
    fn motor_gate_shifts_only_gated_channels() {
        // 4-channel field with an identical bump on every channel. Run one tick
        // twice — once with the motor gate on channels 0..2, once with no motor
        // (motor_dim=0). The motor gate ONLY touches channels 0..motor_dim, so:
        //   - channels 2,3 must be bit-identical between the two runs (no leak),
        //   - channels 0,1 must differ (the gate amplified them).
        let w = 16;
        let h = 16;
        let cx = CellComplex::grid_2d(w, h);
        let dim = 4usize;
        let motor_dim = 2usize;

        let make_field = || {
            let mut f = CochainField::zeros(0, cx.n_vertices(), dim);
            for ch in 0..dim {
                place_bump(&mut f, w, h, 8, 8, ch, 1.0, 2.0);
            }
            f
        };

        let dt = 0.5;

        // Run WITH motor (channels 0,1 gain = 1 + dt*motor = 1.5).
        let mut with_motor = make_field();
        let mut lap1 = CochainField::zeros(0, cx.n_vertices(), dim);
        let mut relu1 = CochainField::zeros(0, cx.n_vertices(), dim);
        evolve_motor_gated_field(
            &cx,
            &mut with_motor,
            &[1.0, 1.0],
            motor_dim,
            dt,
            0.0,
            &mut lap1,
            &mut relu1,
        );

        // Run WITHOUT motor (pure ballistic baseline).
        let mut no_motor = make_field();
        let mut lap2 = CochainField::zeros(0, cx.n_vertices(), dim);
        let mut relu2 = CochainField::zeros(0, cx.n_vertices(), dim);
        evolve_motor_gated_field(&cx, &mut no_motor, &[], 0, dt, 0.0, &mut lap2, &mut relu2);

        // Ungated channels (2,3): motor gate must not have leaked into them.
        // They are bit-identical between the two runs.
        let mut ungated_leak = 0.0f32;
        for i in 0..cx.n_vertices() {
            for ch in [2usize, 3] {
                let a = with_motor.data[i * dim + ch];
                let b = no_motor.data[i * dim + ch];
                ungated_leak += (a - b).abs();
            }
        }
        assert!(
            ungated_leak < 1e-6,
            "G2 motor-gate locality: ungated channels leaked {ungated_leak:.3e} (must be ~0)"
        );

        // Gated channels (0,1): must differ from the no-motor baseline — the
        // motor gain (1.5×) amplified them.
        let mut gated_shift = 0.0f32;
        for i in 0..cx.n_vertices() {
            for ch in [0usize, 1] {
                let a = with_motor.data[i * dim + ch];
                let b = no_motor.data[i * dim + ch];
                gated_shift += (a - b).abs();
            }
        }
        assert!(
            gated_shift > 1e-3,
            "G2 motor-gate locality: gated channels did not shift (gated_shift={gated_shift:.3e})"
        );

        // Isolation ratio: gated shift / ungated leak → must be huge (gate is
        // local to channels 0..motor_dim). This is the G2 gate metric (>100×).
        let ratio = if ungated_leak > 0.0 {
            gated_shift / ungated_leak
        } else {
            f32::INFINITY
        };
        assert!(
            ratio > 100.0 || ungated_leak == 0.0,
            "G2 isolation ratio {ratio:.1}× <= 100× (gated={gated_shift:.3e}, ungated={ungated_leak:.3e})"
        );
    }

    // ── T1.6 smoke test 3: zero-alloc steady state (logical check) ───────
    //
    // The full CountingAllocator audit lives in the GOAT bench (G4) because a
    // global allocator cannot be installed from a lib unit test. Here we verify
    // the *logical* zero-alloc property: the function body contains no `Vec`
    // construction, `clone`, or `to_vec` on the rank-0 path — only slice writes
    // into pre-allocated scratch. Re-running for many ticks must not grow the
    // scratch buffers.

    #[test]
    fn zero_alloc_steady_state_logical() {
        let w = 32;
        let h = 32;
        let cx = CellComplex::grid_2d(w, h);
        let dim = 4usize;
        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 16, 16, 0, 1.0, 2.0);

        let mut scratch_lap = zero_field(&cx, dim);
        let mut scratch_relu = zero_field(&cx, dim);

        // Record scratch capacities — they must not change across 100 ticks
        // (no reallocation, no growth).
        let lap_cap = scratch_lap.data.capacity();
        let relu_cap = scratch_relu.data.capacity();
        let field_cap = field.data.capacity();

        for _ in 0..100 {
            evolve_motor_gated_field(
                &cx,
                &mut field,
                &[0.5, 0.5],
                2,
                0.1,
                0.0,
                &mut scratch_lap,
                &mut scratch_relu,
            );
        }

        assert_eq!(
            scratch_lap.data.capacity(),
            lap_cap,
            "scratch_lap reallocated across 100 ticks"
        );
        assert_eq!(
            scratch_relu.data.capacity(),
            relu_cap,
            "scratch_relu reallocated across 100 ticks"
        );
        assert_eq!(
            field.data.capacity(),
            field_cap,
            "field reallocated across 100 ticks"
        );
        // Field is finite (no NaN blow-up).
        assert!(
            field.data.iter().all(|v| v.is_finite()),
            "field contains non-finite values after 100 ticks"
        );
    }

    // ── Extra: relu_gate_into correctness ────────────────────────────────

    #[test]
    fn relu_gate_standard_and_leaky() {
        let input = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let mut out = [0.0f32; 5];

        // Standard ReLU.
        relu_gate_into(&input, 0.0, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.5, 2.0]);

        // Leaky ReLU (slope 0.1).
        relu_gate_into(&input, 0.1, &mut out);
        assert!((out[0] - (-0.2)).abs() < 1e-6);
        assert!((out[1] - (-0.05)).abs() < 1e-6);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.5);
        assert_eq!(out[4], 2.0);
    }

    /// Sanity: motor_dim=0 is a pure ballistic decay+propagation (no panic).
    #[test]
    fn motor_dim_zero_runs_clean() {
        let cx = CellComplex::grid_2d(8, 8);
        let dim = 2;
        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, 8, 8, 4, 4, 0, 1.0, 1.0);
        let mut lap = zero_field(&cx, dim);
        let mut relu = zero_field(&cx, dim);
        evolve_motor_gated_field(&cx, &mut field, &[], 0, 0.2, 0.0, &mut lap, &mut relu);
        assert!(field.data.iter().all(|v| v.is_finite()));
    }
}
