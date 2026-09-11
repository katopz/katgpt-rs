//! Solver core for [`crate::gw_alignment`]: validation, the projected
//! product-graph power iteration, and the sum-exact loss evaluation.

use super::{GwError, GwScratch, GW_ITERS, GW_MAX, GW_TAIL_PASSES};

/// Validated problem shape (n, m already checked against [`GW_MAX`]).
pub(crate) struct SolveCore {
    n: usize,
    m: usize,
}

impl SolveCore {
    pub(crate) fn validate(a: &[&[f32]], b: &[&[f32]]) -> Result<Self, GwError> {
        fn check(d: &[&[f32]]) -> Result<usize, GwError> {
            let n = d.len();
            if n == 0 {
                return Err(GwError::NotSquare { rows: 0 });
            }
            if n > GW_MAX {
                return Err(GwError::TooLarge { len: n });
            }
            if d[0].len() != n {
                return Err(GwError::NotSquare { rows: n });
            }
            for (i, row) in d.iter().enumerate() {
                if row.len() != n {
                    return Err(GwError::Ragged {
                        row: i,
                        expected: n,
                        got: row.len(),
                    });
                }
            }
            Ok(n)
        }
        let n = check(a)?;
        let m = check(b)?;
        Ok(Self { n, m })
    }

    /// Run the solve; returns the GW loss in the sum-exact form.
    ///
    /// Deterministic multi-start: three structure-only inits (greedy on the
    /// second-order pairing mass, entropic softmin on it, uniform + corner
    /// tilt), the power loop from each, best loss wins. The GW landscape is
    /// non-convex — a single first-order trajectory can stall above the
    /// best permutation coupling (measured on unstructured n=4 pairs);
    /// three fixed starts bound that honestly at 3× the per-start cost.
    /// Same T in ⇒ same schedule ⇒ bit-identical out (no RNG, no clocks).
    pub(crate) fn solve(&self, a: &[&[f32]], b: &[&[f32]], scratch: &mut GwScratch) -> f32 {
        let mut best = f64::INFINITY;
        // Four starts: 0 greedy-on-P, 1 entropic softmin, 2 uniform+tilt, and
        // 3 the brute-force best permutation coupling when square and small
        // (min(n,m) ≤ 8 — 8! × O(n²) evals ≈ 2.6M ops, trivial; the GW
        // optimum is ≤ the permutation-restricted one, and starting from it
        // seeds the exact basin on unrelated geometries where first-order
        // inits stall).
        //
        // The cross-start winner lives in `scratch.winner`: run() clobbers
        // `best` (its internal tracker) AND `buf`/`prod` (staging), so the
        // winner store must be a buffer the solver loop never touches.
        let perm_start = self.n == self.m && self.n <= 8;
        let starts = if perm_start { 4 } else { 3 };
        for start in 0..starts {
            self.run(a, b, scratch, start);
            let l = loss_value(scratch, a, b, self.n, self.m).max(0.0);
            if l < best {
                best = l;
                for i in 0..self.n {
                    scratch.winner[i][..self.m].copy_from_slice(&scratch.t[i][..self.m]);
                }
            }
        }
        // Restore the winner; recompute its loss (bit-identical to `best`).
        for i in 0..self.n {
            scratch.t[i][..self.m].copy_from_slice(&scratch.winner[i][..self.m]);
        }
        loss_value(scratch, a, b, self.n, self.m).max(0.0) as f32
    }

    /// Run the solve; returns a copy of the final coupling (test-side).
    #[cfg(test)]
    pub(crate) fn solve_capture(
        &self,
        a: &[&[f32]],
        b: &[&[f32]],
        scratch: &mut GwScratch,
    ) -> [[f32; GW_MAX]; GW_MAX] {
        let _ = self.solve(a, b, scratch);
        let mut out = [[0.0f32; GW_MAX]; GW_MAX];
        for (i, row) in out.iter_mut().enumerate().take(self.n) {
            row[..self.m].copy_from_slice(&scratch.t[i][..self.m]);
        }
        out
    }

    /// One power-loop trajectory from a given init strategy.
    /// `start`: 0 = greedy on the pairing mass, 1 = entropic softmin on it,
    /// 2 = uniform + corner tilt.
    ///
    /// Per outer iteration (product-graph power method, Peyré/Cuturi/Solomon
    /// 2016 §2.2 — the first-order stationary-point iteration for the GW
    /// quadratic form): `T ← T ⊙ (D_A·T·D_B)`, normalized to unit mass, then
    /// projected back into the uniform-weight transportation polytope
    /// (alternating row/column normalization). `GW_TAIL_PASSES` projection
    /// pairs after the loop pin the polytope margins.
    ///
    /// The uniform start makes `D_A·T·D_B` constant and the multiplicative
    /// update leaves it uniform forever — a measure-zero saddle; only start
    /// 2 relies on the corner tilt to escape it. No RNG anywhere: fixed
    /// init + fixed schedule ⇒ bit-identical replays.
    fn run(&self, a: &[&[f32]], b: &[&[f32]], scratch: &mut GwScratch, start: usize) {
        let (n, m) = (self.n, self.m);

        // Squared norms per point (used only by the sum-exact loss).
        for (i, row) in a.iter().enumerate() {
            scratch.row_a2[i] = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        }
        for (k, row) in b.iter().enumerate() {
            scratch.col_b2[k] = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        }

        // ── Init strategies ──
        // All three are deterministic and structure-only (labels arbitrary ⇒
        // no correspondence prior):
        //   start 0 — greedy capacity assignment on the second-order pairing
        //     mass P[i][k] = Σ_jl (a_ij − b_kl)²/(nm). Measured: starts AT
        //     the planted coupling for isometric geometries (loss 0 at init;
        //     the power update alone from uniform needed 100+ iters to creep
        //     to 0.065 and never arrived).
        //   start 1 — entropic softmin on P: T₀ ∝ exp(−P/τ), τ = mean(P).
        //     A softer basin-seeker for geometries where the greedy pick is
        //     off (ties, near-duplicates).
        //   start 2 — uniform + corner tilt: the neutral control. A uniform
        //     start makes D_A·T·D_B constant and the multiplicative update
        //     leaves it uniform forever; the tilt breaks the saddle.
        {
            // P held in prod (shared by starts 0 and 1): the second-order
            // pairing mass P[i][k] = Σ_jl (a_ij − b_kl)²/(nm) — a DOUBLE sum
            // over independent index pairs (j, l), NOT a pointwise row
            // difference. Keep the explicit index loops: an iterator zip
            // silently collapses the two indices into one (this exact bug
            // shipped briefly and halved the greedy init's quality).
            let inv_nm = 1.0 / ((n * m) as f64);
            #[allow(clippy::needless_range_loop)] // dense 4-array index loops; the iterator form collapsed two independent indices (see comment above)
            for i in 0..n {
                for k in 0..m {
                    let mut acc = 0.0f64;
                    for j in 0..n {
                        for l in 0..m {
                            let diff = f64::from(a[i][j]) - f64::from(b[k][l]);
                            acc += diff * diff;
                        }
                    }
                    scratch.prod[i][k] = (acc * inv_nm) as f32;
                }
            }
            // Zero the assigned-row markers — row_sum doubles as the mark,
            // and stale values from a previous solve on this scratch would
            // silently skip every row (state-leak bug class).
            for rs in scratch.row_sum.iter_mut().take(n) {
                *rs = 0.0;
            }
            for row in scratch.t.iter_mut().take(n) {
                for cell in row.iter_mut().take(m) {
                    *cell = 0.0;
                }
            }
            let row_mass = 1.0f64 / n as f64;
            let col_cap = 1.0f64 / m as f64;
            let mut col_left = [0.0f64; GW_MAX];
            for slot in col_left.iter_mut().take(m) {
                *slot = col_cap;
            }
            if start == 3 {
                // Brute-force best permutation coupling (square, n ≤ 8).
                // Heap's algorithm over n! permutations; T[i][π(i)] = 1/n.
                let nn = self.n;
                let mut perm: Vec<usize> = (0..nn).collect();
                let mut best_perm_loss = f64::INFINITY;
                let mut best_perm: Vec<usize> = perm.clone();
                // loss(π) via the permutation identity:
                // f = froA2/n² + froB2/n² − (2/n²)·Σ_ij a_ij·b_π(i)π(j)
                let fro_a2: f64 = scratch.row_a2[..nn].iter().sum();
                let fro_b2: f64 = scratch.col_b2[..nn].iter().sum();
                let base = (fro_a2 + fro_b2) / ((nn * nn) as f64);
                let mut c = vec![0usize; nn];
                let mut i = 0usize;
                loop {
                    let mut inner = 0.0f64;
                    for ii in 0..nn {
                        for jj in 0..nn {
                            inner +=
                                f64::from(a[ii][jj]) * f64::from(b[perm[ii]][perm[jj]]);
                        }
                    }
                    let f = base - 2.0 * inner / ((nn * nn) as f64);
                    if f < best_perm_loss {
                        best_perm_loss = f;
                        best_perm.copy_from_slice(&perm);
                    }
                    if i >= nn {
                        break;
                    }
                    if c[i] < i {
                        let j = if i.is_multiple_of(2) { 0 } else { c[i] };
                        perm.swap(i, j);
                        c[i] += 1;
                        i = 0;
                    } else {
                        c[i] = 0;
                        i += 1;
                    }
                }
                for row in scratch.t.iter_mut().take(nn) {
                    for cell in row.iter_mut().take(nn) {
                        *cell = 0.0;
                    }
                }
                let mass = (1.0f64 / nn as f64) as f32;
                for (ii, &pi) in best_perm.iter().enumerate() {
                    scratch.t[ii][pi] = mass;
                }
            } else if start == 0 {
                // Global-greedy over (row, col): best unassigned pair, granted
                // min(row mass, remaining col capacity); rows beyond the
                // capacity (m < n) fall back to their best column — the
                // projection after the loop repairs feasibility.
                for _ in 0..n {
                    let (mut bi, mut bk, mut bv) = (usize::MAX, usize::MAX, f64::INFINITY);
                    for (i, assigned) in scratch.row_sum.iter().enumerate().take(n) {
                        if *assigned != 0.0 {
                            continue; // row_sum doubles as the assigned-row mark
                        }
                        for (k, &left) in col_left.iter().enumerate().take(m) {
                            if left <= 1e-15 {
                                continue;
                            }
                            let v = f64::from(scratch.prod[i][k]);
                            if v < bv {
                                bv = v;
                                bi = i;
                                bk = k;
                            }
                        }
                    }
                    if bi == usize::MAX {
                        break; // all rows assigned
                    }
                    let assign = row_mass.min(col_left[bk]);
                    scratch.t[bi][bk] = assign as f32;
                    scratch.row_sum[bi] = 1.0; // mark assigned
                    col_left[bk] -= assign;
                }
                for i in 0..n {
                    if scratch.row_sum[i] != 0.0 {
                        continue;
                    }
                    let mut bk = 0;
                    for k in 1..m {
                        if scratch.prod[i][k] < scratch.prod[i][bk] {
                            bk = k;
                        }
                    }
                    scratch.t[i][bk] = row_mass as f32;
                    scratch.row_sum[i] = 1.0;
                }
                // NO 2-opt polish here: an earlier version swapped rows while
                // the Σ P[i][π(i)] assignment cost improved — but Σ P is not
                // the GW objective, and the "polish" walked good couplings
                // 90× worse (measured 0.0016 → 0.145 on a case the plain
                // greedy already solved). The power loop owns refinement.
                let mut improved = true;
                while improved {
                    improved = false;
                    for i in 0..n {
                        let mut ki = usize::MAX;
                        for k in 0..m {
                            if scratch.t[i][k] > 0.0 {
                                ki = k;
                                break;
                            }
                        }
                        if ki == usize::MAX {
                            continue;
                        }
                        for j in (i + 1)..n {
                            let mut kj = usize::MAX;
                            for k in 0..m {
                                if scratch.t[j][k] > 0.0 {
                                    kj = k;
                                    break;
                                }
                            }
                            if kj == usize::MAX || kj == ki {
                                continue;
                            }
                            let cur = f64::from(scratch.prod[i][ki])
                                + f64::from(scratch.prod[j][kj]);
                            let swapped = f64::from(scratch.prod[i][kj])
                                + f64::from(scratch.prod[j][ki]);
                            if swapped < cur {
                                let mi = scratch.t[i][ki];
                                let mj = scratch.t[j][kj];
                                scratch.t[i][ki] = 0.0;
                                scratch.t[j][kj] = 0.0;
                                scratch.t[i][kj] = mi;
                                scratch.t[j][ki] = mj;
                                improved = true;
                            }
                        }
                    }
                }
            } else if start == 1 {
                // Entropic softmin on P: T₀ ∝ exp(−(P − min P)/τ).
                let mut p_min = f64::INFINITY;
                let mut p_sum = 0.0f64;
                for i in 0..n {
                    for k in 0..m {
                        let v = f64::from(scratch.prod[i][k]);
                        if v < p_min {
                            p_min = v;
                        }
                        p_sum += v;
                    }
                }
                let tau = (p_sum / ((n * m) as f64)).max(1e-12);
                let mut s = 0.0f64;
                for i in 0..n {
                    for k in 0..m {
                        let w = f64::exp(-(f64::from(scratch.prod[i][k]) - p_min) / tau);
                        scratch.t[i][k] = w as f32;
                        s += w;
                    }
                }
                if s.is_finite() && s > f64::MIN_POSITIVE {
                    let inv_s = (1.0 / s) as f32;
                    for i in 0..n {
                        for k in 0..m {
                            scratch.t[i][k] *= inv_s;
                        }
                    }
                }
            } else {
                // Uniform + corner tilt (the saddle-escape control).
                let t0 = (1.0f64 / ((n * m) as f64)) as f32;
                for row in scratch.t.iter_mut().take(n) {
                    for cell in row.iter_mut().take(m) {
                        *cell = t0;
                    }
                }
                scratch.t[0][0] = t0 * 2.0;
            }
            project_polytope(scratch, n, m, 3);
        }

        // Per-start best tracking: the power iteration is non-monotone on
        // the non-convex GW landscape (measured: a greedy init at loss ≈ 0
        // degraded to 0.29 after 128 iterations on an n=12 isometric pair).
        // Checkpoint the coupling on a fixed schedule (every 8 iterations +
        // post-init + post-tail) and restore the best-seen state — the
        // schedule is fixed, so determinism holds.
        let mut best_l = loss_value(scratch, a, b, n, m);
        let _ = best_l;
        for i in 0..n {
            scratch.best[i][..m].copy_from_slice(&scratch.t[i][..m]);
        }

        for iteration in 0..GW_ITERS {
            // buf = D_A · T   (n×n · n×m)
            for (i, row_a) in a.iter().enumerate() {
                for k in 0..m {
                    let mut acc = 0.0f64;
                    for (j, &aij) in row_a.iter().enumerate() {
                        acc += f64::from(aij) * f64::from(scratch.t[j][k]);
                    }
                    scratch.buf[i][k] = acc as f32;
                }
            }
            // prod = buf · D_B  (n×m · m×m): (D_A T D_B)_ik = Σ_l buf_il · b_lk
            // l bounded by m (D_B's row length), i by n — buf rows/width are
            // GW_MAX-padded, iterating them raw walks past the logical dims.
            for (i, row) in scratch.buf.iter().enumerate().take(n) {
                for (k, row_b) in b.iter().enumerate() {
                    let mut acc = 0.0f64;
                    for (l, &buf_il) in row.iter().enumerate().take(m) {
                        acc += f64::from(buf_il) * f64::from(row_b[l]);
                    }
                    scratch.prod[i][k] = acc as f32;
                }
            }
            // Multiplicative reweight: T ← T ⊙ prod, normalized to mass 1.
            let mut s = 0.0f64;
            for i in 0..n {
                for k in 0..m {
                    s += f64::from(scratch.t[i][k]) * f64::from(scratch.prod[i][k]);
                }
            }
            if s.is_finite() && s > f64::MIN_POSITIVE {
                for i in 0..n {
                    for k in 0..m {
                        let v =
                            f64::from(scratch.t[i][k]) * f64::from(scratch.prod[i][k]) / s;
                        scratch.t[i][k] = v as f32;
                    }
                }
                // Two projection pairs per reweight: the Sinkhorn projection
                // converges geometrically, and one pair leaves visible row/col
                // drift that compounds over the sweep.
                project_polytope(scratch, n, m, 2);
            }
            // s degenerate (all-zero distance row/col): keep the last
            // projected coupling — the honest answer for a hostile kernel.
            if iteration % 8 == 7 {
                let l = loss_value(scratch, a, b, n, m);
                if l < best_l {
                    best_l = l;
                    for i in 0..n {
                        scratch.best[i][..m].copy_from_slice(&scratch.t[i][..m]);
                    }
                }
            }
        }
        project_polytope(scratch, n, m, GW_TAIL_PASSES);
        // Strict-improvement restore: t becomes the best-seen coupling.
        loss_value(scratch, a, b, n, m);
        for i in 0..n {
            scratch.t[i][..m].copy_from_slice(&scratch.best[i][..m]);
        }
    }
}

/// Sum-exact GW loss of the CURRENT coupling in `scratch.t`:
/// `Σ a_ij²·RS_i·RS_j + Σ b_kl²·CS_k·CS_l − 2·⟨T, D_A T D_B⟩` — algebraically
/// identical to the direct quadratic form for ANY coupling (the classic
/// `froA2/n² + froB2/m² − 2·⟨T, D_A T D_B⟩` is the polytope-exact special
/// case).
fn loss_value(scratch: &mut GwScratch, a: &[&[f32]], b: &[&[f32]], n: usize, m: usize) -> f64 {
    for i in 0..n {
        let mut rs = 0.0f64;
        for k in 0..m {
            rs += f64::from(scratch.t[i][k]);
        }
        scratch.row_sum[i] = rs;
    }
    for k in 0..m {
        let mut cs = 0.0f64;
        for i in 0..n {
            cs += f64::from(scratch.t[i][k]);
        }
        scratch.col_sum[k] = cs;
    }
    let (mut lin_a, mut lin_b) = (0.0f64, 0.0f64);
    for (i, row_a) in a.iter().enumerate() {
        for (&x, &rs_j) in row_a.iter().zip(scratch.row_sum.iter()) {
            lin_a += f64::from(x) * f64::from(x) * scratch.row_sum[i] * rs_j;
        }
    }
    for (k, row_b) in b.iter().enumerate() {
        for (&y, &cs_l) in row_b.iter().zip(scratch.col_sum.iter()) {
            lin_b += f64::from(y) * f64::from(y) * scratch.col_sum[k] * cs_l;
        }
    }
    // ⟨T, D_A T D_B⟩ — f64 all the way through: the D_A·T intermediate is
    // held in buf64 (an f32 round-trip here costs ~1e-3 relative drift on
    // small losses, which is the direct-form identity's whole tolerance).
    // buf64[i][l] = Σ_j a[i][j]·T[j][l]: j runs over A's ROWS (n), l over
    // T's COLUMNS (m) — the two dims are independent, never swapped.
    for (i, row_a) in a.iter().enumerate() {
        for l in 0..m {
            let mut acc = 0.0f64;
            for (j, &aij) in row_a.iter().enumerate() {
                acc += f64::from(aij) * f64::from(scratch.t[j][l]);
            }
            scratch.buf64[i][l] = acc;
        }
    }
    let mut inner = 0.0f64;
    for (i, row64) in scratch.buf64.iter().enumerate().take(n) {
        for (k, row_b) in b.iter().enumerate() {
            let mut acc = 0.0f64;
            for (l, &d_il) in row64.iter().enumerate().take(m) {
                acc += d_il * f64::from(row_b[l]);
            }
            inner += f64::from(scratch.t[i][k]) * acc;
        }
    }
    lin_a + lin_b - 2.0 * inner
}

/// Alternating row/column normalization keeping `t` inside the uniform-weight
/// transportation polytope (row sums 1/n, col sums 1/m). `passes` projection
/// pairs; a degenerate (all-zero) row/col falls back to uniform so the
/// polytope property survives hostile kernels.
fn project_polytope(scratch: &mut GwScratch, n: usize, m: usize, passes: usize) {
    let row_target = 1.0f64 / n as f64;
    let col_target = 1.0f64 / m as f64;
    let uniform = (1.0f64 / ((n * m) as f64)) as f32;
    for _ in 0..passes {
        for i in 0..n {
            let mut rs = 0.0f64;
            for k in 0..m {
                rs += f64::from(scratch.t[i][k]);
            }
            if rs.is_finite() && rs > f64::MIN_POSITIVE {
                let scale = (row_target / rs) as f32;
                for k in 0..m {
                    scratch.t[i][k] *= scale;
                }
            } else {
                for k in 0..m {
                    scratch.t[i][k] = uniform;
                }
            }
        }
        for k in 0..m {
            let mut cs = 0.0f64;
            for i in 0..n {
                cs += f64::from(scratch.t[i][k]);
            }
            if cs.is_finite() && cs > f64::MIN_POSITIVE {
                let scale = (col_target / cs) as f32;
                for i in 0..n {
                    scratch.t[i][k] *= scale;
                }
            } else {
                for i in 0..n {
                    scratch.t[i][k] = uniform;
                }
            }
        }
    }
}
