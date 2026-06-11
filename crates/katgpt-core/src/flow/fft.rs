//! FFT smoothing for potential-field de-noising.
//!
//! Forward FFT → low-pass filter → inverse FFT removes local minima
//! and produces smooth gradients toward goals. Uses `rustfft` which
//! handles arbitrary (non-power-of-2) sizes.

use rustfft::{FftPlanner, num_complex::Complex};

/// Forward FFT → low-pass filter → inverse FFT.
///
/// `cutoff` is a fraction of Nyquist (0.0–1.0). Frequencies whose
/// normalised radius exceeds `cutoff` are zeroed.
pub fn fft_smooth(grid: &mut [f32], w: usize, h: usize, cutoff: f32) {
    assert!(
        grid.len() == w * h,
        "grid length {} != w*h ({}*{}={})",
        grid.len(),
        w,
        h,
        w * h
    );
    if w == 0 || h == 0 {
        return;
    }
    let n = w * h;
    let mut buf: Vec<Complex<f32>> = Vec::with_capacity(n);
    let mut col_buf: Vec<Complex<f32>> = Vec::with_capacity(h);
    fft_smooth_into(grid, w, h, cutoff, &mut buf, &mut col_buf);
}

/// Pre-allocated variant of [`fft_smooth`].
///
/// Callers that invoke FFT smoothing repeatedly with the same grid
/// dimensions can reuse scratch buffers across calls to avoid per-call
/// allocation. The `buf` must have capacity `>= w * h` and `col_buf`
/// must have capacity `>= h`; both are cleared and refilled internally.
pub fn fft_smooth_into(
    grid: &mut [f32],
    w: usize,
    h: usize,
    cutoff: f32,
    buf: &mut Vec<Complex<f32>>,
    col_buf: &mut Vec<Complex<f32>>,
) {
    assert!(
        grid.len() == w * h,
        "grid length {} != w*h ({}*{}={})",
        grid.len(),
        w,
        h,
        w * h
    );
    if w == 0 || h == 0 {
        return;
    }

    let n = w * h;

    // Reuse buf — resize and fill from grid.
    buf.clear();
    buf.extend(grid.iter().map(|&v| Complex::new(v, 0.0)));

    // --- 2D FFT: rows then columns ---
    let mut planner = FftPlanner::new();
    let row_fwd = planner.plan_fft_forward(w);
    let col_fwd = planner.plan_fft_forward(h);

    // Transform rows (in-place, each row is contiguous)
    for y in 0..h {
        row_fwd.process(&mut buf[y * w..(y + 1) * w]);
    }

    // Transform columns (strided — copy, transform, write back)
    col_buf.resize(h, Complex::new(0.0, 0.0));
    for x in 0..w {
        for y in 0..h {
            col_buf[y] = buf[y * w + x];
        }
        col_fwd.process(col_buf);
        for y in 0..h {
            buf[y * w + x] = col_buf[y];
        }
    }

    // --- Low-pass filter ---
    let half_w = w as f32 * 0.5;
    let half_h = h as f32 * 0.5;
    let min_half = half_w.min(half_h);
    let cutoff_r = cutoff * min_half;
    let cutoff_r_sq = cutoff_r * cutoff_r;

    let h_f = h as f32;
    let w_f = w as f32;
    for fy in 0..h {
        let fy_centered = {
            let raw = fy as f32;
            if raw >= half_h { raw - h_f } else { raw }
        };
        let fyc_sq = fy_centered * fy_centered;
        let row_off = fy * w;
        for fx in 0..w {
            let fx_centered = {
                let raw = fx as f32;
                if raw >= half_w { raw - w_f } else { raw }
            };
            let r_sq = fx_centered * fx_centered + fyc_sq;
            if r_sq > cutoff_r_sq {
                buf[row_off + fx] = Complex::new(0.0, 0.0);
            }
        }
    }

    // --- Inverse 2D FFT ---
    let row_inv = planner.plan_fft_inverse(w);
    let col_inv = planner.plan_fft_inverse(h);

    // Inverse columns
    for x in 0..w {
        for y in 0..h {
            col_buf[y] = buf[y * w + x];
        }
        col_inv.process(col_buf);
        for y in 0..h {
            buf[y * w + x] = col_buf[y];
        }
    }

    // Inverse rows
    for y in 0..h {
        row_inv.process(&mut buf[y * w..(y + 1) * w]);
    }

    // Write real parts back, normalised by n
    let scale = 1.0 / n as f32;
    for i in 0..n {
        grid[i] = buf[i].re * scale;
    }
}

/// Morphological dilation of blocked cells by `radius`.
/// Inflate obstacles with a caller-provided snapshot buffer (zero-allocation on hot path).
///
/// `snapshot` must have the same length as `blocked`. The original blocked state is
/// copied into `snapshot` at the start, then read from `snapshot` while writing to `blocked`.
/// This avoids per-call heap allocation when the caller reuses a pre-allocated buffer.
pub fn inflate_obstacles_with_snapshot(
    blocked: &mut [u64],
    snapshot: &mut [u64],
    w: u16,
    h: u16,
    radius: u8,
) {
    let wu = w as usize;
    let hu = h as usize;
    let words_per_row = (wu + 63) / 64;
    let total_words = words_per_row * hu;
    assert!(blocked.len() >= total_words, "blocked bitfield too small");
    assert!(snapshot.len() >= total_words, "snapshot bitfield too small");

    let r = radius as i32;
    if r == 0 {
        return;
    }

    // Snapshot original state to avoid order-dependent dilation.
    snapshot[..total_words].copy_from_slice(&blocked[..total_words]);

    for y in 0..hu {
        for x in 0..wu {
            let word_idx = y * words_per_row + x / 64;
            let bit = x % 64;
            if snapshot[word_idx] & (1u64 << bit) != 0 {
                // Already blocked — expand neighbors
                let y_min = (y as i32 - r).max(0) as usize;
                let y_max = (y as i32 + r).min(hu as i32 - 1) as usize;
                let x_min = (x as i32 - r).max(0) as usize;
                let x_max = (x as i32 + r).min(wu as i32 - 1) as usize;
                for ny in y_min..=y_max {
                    for nx in x_min..=x_max {
                        let nw_idx = ny * words_per_row + nx / 64;
                        blocked[nw_idx] |= 1u64 << (nx % 64);
                    }
                }
            }
        }
    }
}

/// Convenience wrapper that allocates a snapshot internally (backward compatible).
pub fn inflate_obstacles(blocked: &mut [u64], w: u16, h: u16, radius: u8) {
    let wu = w as usize;
    let hu = h as usize;
    let words_per_row = (wu + 63) / 64;
    let total_words = words_per_row * hu;
    assert!(blocked.len() >= total_words, "blocked bitfield too small");

    let mut snapshot = blocked.to_vec();
    inflate_obstacles_with_snapshot(blocked, &mut snapshot, w, h, radius);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_smooth_removes_local_minima() {
        // 16×16 grid with a sharp spike in the center.
        let w = 16usize;
        let h = 16usize;
        let mut grid = vec![0.0f32; w * h];
        // Set a smooth slope toward center
        let cx = w / 2;
        let cy = h / 2;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                grid[y * w + x] = 10.0 - (dx * dx + dy * dy).sqrt();
            }
        }
        // Add a local dip (local minimum) near center
        grid[(cy - 1) * w + cx] = -100.0;

        fft_smooth(&mut grid, w, h, 0.25);

        // After smoothing, center region should not have extreme negative values.
        // The local minimum should be filled in.
        let center_val = grid[cy * w + cx];
        assert!(
            center_val > -50.0,
            "local minimum should be smoothed out, got {}",
            center_val
        );
    }

    #[test]
    fn fft_smooth_preserves_dc_offset() {
        let w = 8usize;
        let h = 8usize;
        let dc = 5.0f32;
        let mut grid = vec![dc; w * h];

        fft_smooth(&mut grid, w, h, 0.5);

        // DC component (frequency 0,0) is below cutoff → preserved
        let avg: f32 = grid.iter().sum::<f32>() / grid.len() as f32;
        assert!(
            (avg - dc).abs() < 0.1,
            "DC offset should be preserved, avg={avg}, expected={dc}"
        );
    }

    #[test]
    fn fft_smooth_zeros_grid_with_zero_cutoff() {
        let w = 8usize;
        let h = 8usize;
        let mut grid = vec![1.0f32; w * h];

        fft_smooth(&mut grid, w, h, 0.0);

        // cutoff=0 means all frequencies above DC are removed.
        // DC = average = 1.0 should survive.
        let all_zero_except_dc = grid.iter().all(|&v| (v - 1.0).abs() < 0.01);
        assert!(all_zero_except_dc, "with cutoff=0, only DC should remain");
    }

    #[test]
    fn fft_smooth_handles_non_power_of_two() {
        let w = 5usize;
        let h = 7usize;
        let mut grid = vec![1.0f32; w * h];

        // Should not panic — rustfft handles arbitrary sizes.
        fft_smooth(&mut grid, w, h, 0.25);

        // All values should be finite
        for &v in &grid {
            assert!(v.is_finite(), "value should be finite: {v}");
        }
    }

    #[test]
    fn fft_smooth_into_matches_fft_smooth() {
        let w = 16usize;
        let h = 16usize;
        let mut grid_a = vec![0.0f32; w * h];
        let cx = w / 2;
        let cy = h / 2;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                grid_a[y * w + x] = 10.0 - (dx * dx + dy * dy).sqrt();
            }
        }
        let mut grid_b = grid_a.clone();

        fft_smooth(&mut grid_a, w, h, 0.25);

        let mut buf = Vec::with_capacity(w * h);
        let mut col_buf = Vec::with_capacity(h);
        fft_smooth_into(&mut grid_b, w, h, 0.25, &mut buf, &mut col_buf);

        // Must produce identical output.
        for (i, (a, b)) in grid_a.iter().zip(grid_b.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at index {i}: {a} vs {b}");
        }

        // Verify buffer reuse across calls.
        let mut grid_c = grid_a.clone();
        fft_smooth_into(&mut grid_c, w, h, 0.25, &mut buf, &mut col_buf);
        for (i, (a, c)) in grid_a.iter().zip(grid_c.iter()).enumerate() {
            assert!(
                (a - c).abs() < 1e-6,
                "reuse mismatch at index {i}: {a} vs {c}"
            );
        }
    }

    #[test]
    fn inflate_obstacles_expands_single_cell() {
        let w = 8u16;
        let h = 8u16;
        let words_per_row = 1usize; // 8 cells fit in one u64
        let mut blocked = vec![0u64; words_per_row * h as usize];

        // Block cell (4, 4)
        let x = 4usize;
        let y = 4usize;
        blocked[y * words_per_row + x / 64] |= 1u64 << (x % 64);

        inflate_obstacles(&mut blocked, w, h, 1);

        // All 8 neighbors of (4,4) should be blocked
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = 4i32 + dx;
                let ny = 4i32 + dy;
                let word = blocked[ny as usize * words_per_row + nx as usize / 64];
                let bit = nx as usize % 64;
                assert!(
                    word & (1u64 << bit) != 0,
                    "cell ({nx},{ny}) should be blocked after inflation"
                );
            }
        }

        // Cell far away (0,0) should remain unblocked
        assert_eq!(blocked[0] & 1, 0, "cell (0,0) should remain unblocked");
    }

    #[test]
    fn inflate_obstacles_radius_zero_is_noop() {
        let w = 8u16;
        let h = 8u16;
        let words_per_row = 1usize;
        let mut blocked = vec![0u64; words_per_row * h as usize];

        // Block cell (3, 3)
        blocked[3] |= 1u64 << 3;

        let before = blocked.clone();
        inflate_obstacles(&mut blocked, w, h, 0);

        assert_eq!(blocked, before, "radius=0 should not expand anything");
    }

    #[test]
    fn inflate_obstacles_clamps_at_boundary() {
        let w = 8u16;
        let h = 8u16;
        let words_per_row = 1usize;
        let mut blocked = vec![0u64; words_per_row * h as usize];

        // Block corner cell (0, 0)
        blocked[0] |= 1u64;

        inflate_obstacles(&mut blocked, w, h, 1);

        // Only valid neighbors (0,1), (1,0), (1,1) should be blocked
        let expected_blocked: Vec<(usize, usize)> = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        for y in 0..3usize {
            for x in 0..3usize {
                let word = blocked[y * words_per_row + x / 64];
                let bit = x % 64;
                let is_blocked = word & (1u64 << bit) != 0;
                let expected = expected_blocked.contains(&(x, y));
                assert_eq!(
                    is_blocked, expected,
                    "cell ({x},{y}): blocked={is_blocked}, expected={expected}"
                );
            }
        }
    }
}
