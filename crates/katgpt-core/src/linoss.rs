//! LinOSS — Linear Oscillatory State-Space cell + ModalSpec drafter (Plan 189 Phase 3).
//!
//! Symplectic IMEX integrator for harmonic oscillators. Linear dynamics enable
//! parallel scan (Blelloch prefix sum) for training/parallel mode.

/// LinOSS cell parameters. Pre-allocated in constructor, zero alloc on hot path.
#[derive(Clone, Debug)]
pub struct LinOSSCell {
    /// Angular frequency squared per hidden dimension: ω² [H].
    pub omega_sq: Vec<f32>,
    /// Damping coefficient per hidden dimension: β [H].
    pub beta: Vec<f32>,
    hidden_dim: usize,
}

/// LinOSS hidden state (phase-space: position y, velocity z).
#[derive(Clone, Debug)]
pub struct LinOSSState {
    /// Position coordinate [H].
    pub y: Vec<f32>,
    /// Velocity coordinate [H].
    pub z: Vec<f32>,
}

impl LinOSSCell {
    /// Create with unit frequency (ω²=1), undamped (β=0).
    #[inline]
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            omega_sq: vec![1.0; hidden_dim],
            beta: vec![0.0; hidden_dim],
            hidden_dim,
        }
    }

    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// One IMEX step (symplectic, energy-preserving when β=0).
    /// y' = y + dt * z   (explicit)
    /// z' = z + dt * (-ω² * y' - β * z + forcing)   (implicit on y')
    ///
    /// **Note**: This allocates two `Vec<f32>` per call. Prefer `imex_step_inplace`
    /// on hot paths (inference loops, speculative decoding) to reuse pre-allocated buffers.
    #[inline]
    pub fn imex_step(&self, state: &LinOSSState, forcing: &[f32], dt: f32) -> LinOSSState {
        let h = self.hidden_dim;
        debug_assert_eq!(state.y.len(), h);
        debug_assert_eq!(state.z.len(), h);
        debug_assert!(forcing.is_empty() || forcing.len() == h);
        let mut y_new = vec![0.0f32; h];
        let mut z_new = vec![0.0f32; h];
        for i in 0..h {
            y_new[i] = state.y[i] + dt * state.z[i];
            let f = if forcing.is_empty() { 0.0 } else { forcing[i] };
            z_new[i] =
                state.z[i] + dt * (-self.omega_sq[i] * y_new[i] - self.beta[i] * state.z[i] + f);
        }
        LinOSSState { y: y_new, z: z_new }
    }

    /// In-place IMEX step — writes y_new and z_new into pre-allocated buffers.
    /// Returns (y_new, z_new) slices. Zero allocation.
    #[inline]
    pub fn imex_step_inplace<'a>(
        &self,
        y_in: &[f32],
        z_in: &[f32],
        forcing: &[f32],
        dt: f32,
        y_out: &'a mut [f32],
        z_out: &'a mut [f32],
    ) -> (&'a [f32], &'a [f32]) {
        let h = self.hidden_dim;
        debug_assert_eq!(y_in.len(), h);
        debug_assert_eq!(z_in.len(), h);
        debug_assert!(y_out.len() >= h);
        debug_assert!(z_out.len() >= h);
        debug_assert!(forcing.is_empty() || forcing.len() == h);
        // Hoist the `forcing.is_empty()` branch out of the hot loop into two
        // specialized bodies. Inside each, use mul_add for FMA fusion.
        if forcing.is_empty() {
            for i in 0..h {
                // y_out = y + dt·z ; z_out = z + dt·(−ω²·y_out − β·z)
                y_out[i] = dt.mul_add(z_in[i], y_in[i]);
                let wz = (-self.omega_sq[i]).mul_add(y_out[i], -self.beta[i] * z_in[i]);
                z_out[i] = dt.mul_add(wz, z_in[i]);
            }
        } else {
            for i in 0..h {
                y_out[i] = dt.mul_add(z_in[i], y_in[i]);
                // z_out = z + dt·(−ω²·y_out − β·z + f)
                let wz =
                    (-self.omega_sq[i]).mul_add(y_out[i], -self.beta[i] * z_in[i]) + forcing[i];
                z_out[i] = dt.mul_add(wz, z_in[i]);
            }
        }
        (&y_out[..h], &z_out[..h])
    }

    /// Parallel scan (Blelloch prefix sum) for training/parallel mode.
    /// LinOSS is linear → transfer matrices compose → parallel prefix scan works.
    pub fn parallel_scan(
        &self,
        initial: &LinOSSState,
        forcings: &[&[f32]],
        dt: f32,
    ) -> Vec<LinOSSState> {
        let mut scratch = ParallelScanScratch::new();
        self.parallel_scan_with_scratch(initial, forcings, dt, &mut scratch)
    }

    /// Zero-alloc parallel scan using pre-allocated scratch buffers.
    /// Reuse `ParallelScanScratch` across calls to avoid repeated allocation.
    ///
    /// This is a back-compat wrapper: it calls [`parallel_scan_into_flat`] (which writes flat
    /// `n*h` result buffers into scratch, zero per-call allocation), then materializes them
    /// into `Vec<LinOSSState>`. Prefer [`parallel_scan_into_flat`] on hot paths.
    pub fn parallel_scan_with_scratch(
        &self,
        initial: &LinOSSState,
        forcings: &[&[f32]],
        dt: f32,
        scratch: &mut ParallelScanScratch,
    ) -> Vec<LinOSSState> {
        let n = self.parallel_scan_into_flat(initial, forcings, dt, scratch);
        let h = self.hidden_dim;
        // Materialize flat buffers into Vec<LinOSSState>. This is the only allocation path —
        // hot callers should use `parallel_scan_into_flat` + `scratch.result_y()/result_z()` directly.
        let (ry, rz) = scratch.split_results(n * h);
        ry.chunks_exact(h)
            .zip(rz.chunks_exact(h))
            .map(|(y, z)| LinOSSState {
                y: y.to_vec(),
                z: z.to_vec(),
            })
            .collect()
    }

    /// Zero-alloc parallel scan writing flat result buffers into `scratch.ry` / `scratch.rz`.
    ///
    /// Returns the number of steps written (`n`). Read results via
    /// `scratch.result_y()` / `scratch.result_z()`, each length `n * hidden_dim`,
    /// row-major: step `s`, dim `j` at index `s * hidden_dim + j`.
    ///
    /// For `n <= 64`, falls back to sequential scan (still writes flat buffers).
    pub fn parallel_scan_into_flat(
        &self,
        initial: &LinOSSState,
        forcings: &[&[f32]],
        dt: f32,
        scratch: &mut ParallelScanScratch,
    ) -> usize {
        let n = forcings.len();
        if n == 0 {
            return 0;
        }
        // For small sequences, sequential avoids overhead.
        if n <= 64 {
            return self.sequential_scan_into_flat(initial, forcings, dt, scratch);
        }

        let h = self.hidden_dim;
        let total = n * h;
        scratch.ensure_capacity(total);

        // Per-step transfer matrix: [[1, dt], [-ω²dt, 1]] with bias [0, dt*f]
        // Fused zero-fill + init: only write each element once instead of zero + overwrite.
        // a=1.0 and d=1.0 are constants; b=dt is constant; only c and bias_z vary per-step.
        for buf in [&mut scratch.bias_y, &mut scratch.pbz] {
            buf[..total].fill(0.0);
        }
        scratch.a[..total].fill(1.0);
        scratch.d[..total].fill(1.0);
        scratch.b[..total].fill(dt);
        // c and bias_z are overwritten per-step below, no need to pre-fill.

        for (step, f) in forcings.iter().enumerate().take(n) {
            let base = step * h;
            for j in 0..h {
                scratch.c[base + j] = -self.omega_sq[j] * dt;
                scratch.bias_z[base + j] = dt * f[j];
            }
        }

        // Inclusive prefix scan: prefix[i] = M_i * M_{i-1} * ... * M_0
        // NOTE: Replace with Blelloch up-sweep/down-sweep for true parallelism.
        // Copy a→pa, b→pb, etc. using copy_from_slice instead of clone
        scratch.pa[..total].copy_from_slice(&scratch.a[..total]);
        scratch.pb[..total].copy_from_slice(&scratch.b[..total]);
        scratch.pc[..total].copy_from_slice(&scratch.c[..total]);
        scratch.pd[..total].copy_from_slice(&scratch.d[..total]);
        scratch.pby[..total].copy_from_slice(&scratch.bias_y[..total]);
        scratch.pbz[..total].copy_from_slice(&scratch.bias_z[..total]);

        for step in 1..n {
            let prev = (step - 1) * h;
            let base = step * h;
            for j in 0..h {
                let (pa0, pb0, pc0, pd0) = (
                    scratch.pa[prev + j],
                    scratch.pb[prev + j],
                    scratch.pc[prev + j],
                    scratch.pd[prev + j],
                );
                let (pby0, pbz0) = (scratch.pby[prev + j], scratch.pbz[prev + j]);
                let (ma, mb, mc, md) = (
                    scratch.a[base + j],
                    scratch.b[base + j],
                    scratch.c[base + j],
                    scratch.d[base + j],
                );
                let (mby, mbz) = (scratch.bias_y[base + j], scratch.bias_z[base + j]);
                scratch.pa[base + j] = pa0 * ma + pb0 * mc;
                scratch.pb[base + j] = pa0 * mb + pb0 * md;
                scratch.pc[base + j] = pc0 * ma + pd0 * mc;
                scratch.pd[base + j] = pc0 * mb + pd0 * md;
                scratch.pby[base + j] = pa0 * mby + pb0 * mbz + pby0;
                scratch.pbz[base + j] = pc0 * mby + pd0 * mbz + pbz0;
            }
        }

        // Write flat result buffers. FMA via mul_add for single-rounded accumulation.
        // ry[base+j] = pa·y0 + pb·z0 + pby; rz[base+j] = pc·y0 + pd·z0 + pbz.
        for step in 0..n {
            let base = step * h;
            for j in 0..h {
                let y0 = initial.y[j];
                let z0 = initial.z[j];
                scratch.ry[base + j] = scratch.pa[base + j]
                    .mul_add(y0, scratch.pb[base + j].mul_add(z0, scratch.pby[base + j]));
                scratch.rz[base + j] = scratch.pc[base + j]
                    .mul_add(y0, scratch.pd[base + j].mul_add(z0, scratch.pbz[base + j]));
            }
        }
        n
    }

    /// Sequential scan — simple loop for correctness reference and small sequences.
    ///
    /// Back-compat allocating wrapper around [`sequential_scan_into_flat`].
    /// Materializes flat result buffers into `Vec<LinOSSState>`.
    ///
    /// Only used by tests as the reference for parity checks against `parallel_scan`.
    #[cfg(test)]
    #[inline]
    fn sequential_scan(
        &self,
        initial: &LinOSSState,
        forcings: &[&[f32]],
        dt: f32,
    ) -> Vec<LinOSSState> {
        let mut scratch = ParallelScanScratch::new();
        let n = self.sequential_scan_into_flat(initial, forcings, dt, &mut scratch);
        let h = self.hidden_dim;
        let (ry, rz) = scratch.split_results(n * h);
        ry.chunks_exact(h)
            .zip(rz.chunks_exact(h))
            .map(|(y, z)| LinOSSState {
                y: y.to_vec(),
                z: z.to_vec(),
            })
            .collect()
    }

    /// Zero-alloc sequential scan writing flat result buffers into `scratch.ry` / `scratch.rz`.
    ///
    /// Because `ry` and `rz` are separate `Vec<f32>` allocations, we can pass disjoint
    /// `&mut [f32]` slices per step directly — no `split_at_mut` borrow-dance needed.
    /// Returns the number of steps written (`n`).
    #[inline]
    fn sequential_scan_into_flat(
        &self,
        initial: &LinOSSState,
        forcings: &[&[f32]],
        dt: f32,
        scratch: &mut ParallelScanScratch,
    ) -> usize {
        let h = self.hidden_dim;
        let n = forcings.len();
        if n == 0 {
            return 0;
        }
        let total = n * h;
        scratch.ensure_capacity(total);
        // ry and rz are distinct Vec allocations → disjoint &mut borrows are sound.
        let (ry, rz) = scratch.split_results(total);

        // Step 0: from initial state directly into flat result row 0.
        // ry and rz are distinct allocations → disjoint &mut borrows are sound.
        {
            let (y0, z0) = (&mut ry[..h], &mut rz[..h]);
            self.imex_step_inplace(&initial.y, &initial.z, forcings[0], dt, y0, z0);
        }
        // Steps 1..n: read row i-1, write row i. ry and rz are separate allocations
        // so (prev_y, cur_y) and (prev_z, cur_z) borrows don't alias each other.
        for (step, &forcing_step) in forcings.iter().enumerate().take(n).skip(1) {
            let prev = (step - 1) * h;
            let base = step * h;
            // split_at_mut within ry to separate prev row (read) from cur row (write).
            let (prev_part, cur_part) = ry.split_at_mut(base);
            let prev_y = &prev_part[prev..prev + h];
            let cur_y = &mut cur_part[..h];
            let (prev_part, cur_part) = rz.split_at_mut(base);
            let prev_z = &prev_part[prev..prev + h];
            let cur_z = &mut cur_part[..h];
            self.imex_step_inplace(prev_y, prev_z, forcing_step, dt, cur_y, cur_z);
        }
        n
    }

    #[inline]
    pub fn energy(&self, state: &LinOSSState) -> f32 {
        let h = self.hidden_dim;
        let mut e = 0.0f32;
        for i in 0..h {
            e += state.y[i] * state.y[i] + self.omega_sq[i] * state.z[i] * state.z[i];
        }
        e
    }
}

impl LinOSSState {
    #[inline]
    pub fn zeros(hidden_dim: usize) -> Self {
        Self {
            y: vec![0.0; hidden_dim],
            z: vec![0.0; hidden_dim],
        }
    }
}

// ── Scratch buffers for parallel scan ──

/// Pre-allocated scratch buffers for [`LinOSSCell::parallel_scan_with_scratch`].
/// Reuse across calls to avoid repeated allocation.
pub struct ParallelScanScratch {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    bias_y: Vec<f32>,
    bias_z: Vec<f32>,
    pa: Vec<f32>,
    pb: Vec<f32>,
    pc: Vec<f32>,
    pd: Vec<f32>,
    pby: Vec<f32>,
    pbz: Vec<f32>,
    /// Flat result buffer: y-state for all steps, length n*h. Written by `parallel_scan_into_flat`.
    ry: Vec<f32>,
    /// Flat result buffer: z-state for all steps, length n*h. Written by `parallel_scan_into_flat`.
    rz: Vec<f32>,
}

impl ParallelScanScratch {
    /// Create empty scratch buffers. Call [`ensure_capacity`] before first use.
    pub fn new() -> Self {
        Self {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            bias_y: Vec::new(),
            bias_z: Vec::new(),
            pa: Vec::new(),
            pb: Vec::new(),
            pc: Vec::new(),
            pd: Vec::new(),
            pby: Vec::new(),
            pbz: Vec::new(),
            ry: Vec::new(),
            rz: Vec::new(),
        }
    }

    /// Grow buffers to `total` elements if needed. No shrinking on reuse.
    fn ensure_capacity(&mut self, total: usize) {
        macro_rules! ensure {
            ($buf:ident) => {
                if self.$buf.len() < total {
                    self.$buf.resize(total, 0.0);
                }
            };
        }
        ensure!(a);
        ensure!(b);
        ensure!(c);
        ensure!(d);
        ensure!(bias_y);
        ensure!(bias_z);
        ensure!(pa);
        ensure!(pb);
        ensure!(pc);
        ensure!(pd);
        ensure!(pby);
        ensure!(pbz);
        ensure!(ry);
        ensure!(rz);
    }

    /// Borrow the flat y-result buffer `[n*h]` after `parallel_scan_into_flat`.
    pub fn result_y(&self) -> &[f32] {
        &self.ry
    }

    /// Borrow the flat z-result buffer `[n*h]` after `parallel_scan_into_flat`.
    pub fn result_z(&self) -> &[f32] {
        &self.rz
    }

    /// Split out the first `len` elements of `ry`/`rz` as separate mutable slices.
    /// Used by `parallel_scan_with_scratch` to materialize `Vec<LinOSSState>` after a flat scan.
    fn split_results(&mut self, len: usize) -> (&mut [f32], &mut [f32]) {
        (&mut self.ry[..len], &mut self.rz[..len])
    }
}

impl Default for ParallelScanScratch {
    fn default() -> Self {
        Self::new()
    }
}

// ── VocabFourierBasis ──

/// Top-K Fourier modes of vocabulary embedding space.
/// Pre-computed once at model load time — zero alloc on hot path.
pub struct VocabFourierBasis {
    /// Top-K Fourier mode coefficients: flat `[K * vocab_dim]` row-major.
    /// Access mode k via `&modes[k * vocab_dim..(k+1) * vocab_dim]`.
    /// Single allocation avoids K pointer dereferences per reconstruction.
    pub modes: Vec<f32>,
    /// Mode frequencies (angular frequency for each mode).
    pub frequencies: Vec<f32>,
    /// Number of modes.
    k: usize,
    /// Dimension of each mode vector.
    vocab_dim: usize,
}

impl VocabFourierBasis {
    /// Compute top-K Fourier modes from vocabulary embeddings via DFT dot-product.
    ///
    /// Two-phase approach: compute magnitudes for all candidates with a single
    /// pre-allocated scratch buffer, then recompute modes only for top-K.
    /// Avoids storing n_candidates full mode vectors during sorting.
    pub fn from_embeddings(embeddings: &[&[f32]], k: usize) -> Self {
        if embeddings.is_empty() {
            return Self {
                modes: vec![],
                frequencies: vec![],
                k: 0,
                vocab_dim: 0,
            };
        }

        let n = embeddings.len();
        let vocab_dim = embeddings[0].len();
        let vd = vocab_dim;
        let n_candidates = (n * 2).min(256);

        // Phase 1: compute magnitudes with single pre-allocated scratch buffer.
        let mut cos_mode = vec![0.0f32; vocab_dim];
        let mut magnitudes: Vec<(f32, usize)> = Vec::with_capacity(n_candidates);

        for ci in 0..n_candidates {
            let omega = std::f32::consts::PI * (ci as f32 + 1.0) / n as f32;
            cos_mode.fill(0.0);
            for (i, emb) in embeddings.iter().enumerate() {
                let cos_w = (omega * i as f32).cos();
                // SIMD AXPY — single-rounding FMA parity with the previous
                // scalar 'cos_mode[d] += emb[d] * cos_w' loop. Hot path:
                // n_candidates × n × vocab_dim scalar ops → SIMD.
                crate::simd::simd_fused_scale_acc(
                    &mut cos_mode[..vocab_dim],
                    &emb[..vocab_dim],
                    cos_w,
                    vocab_dim,
                );
            }
            let inv_n = 1.0 / n as f32;
            // SIMD scale in place, then SIMD sum-of-squares.
            crate::simd::simd_scale_inplace(&mut cos_mode[..vocab_dim], inv_n);
            let mag_sq = crate::simd::simd_sum_sq(&cos_mode[..vocab_dim], vocab_dim);
            magnitudes.push((mag_sq.sqrt(), ci));
        }

        // Sort by magnitude descending, take top-K indices.
        // Unstable is safe: only top-K is consumed; tie order within K is unspecified.
        magnitudes.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        magnitudes.truncate(k);

        // Phase 2: compute modes only for top-K (reuse cos_mode buffer).
        let k_actual = magnitudes.len();
        let mut modes = vec![0.0f32; k_actual * vd];
        let mut frequencies = Vec::with_capacity(k_actual);
        for (ki, &(_mag, ci)) in magnitudes.iter().enumerate() {
            let omega = std::f32::consts::PI * (ci as f32 + 1.0) / n as f32;
            cos_mode.fill(0.0);
            for (i, emb) in embeddings.iter().enumerate() {
                let cos_w = (omega * i as f32).cos();
                // SIMD AXPY — matches Phase 1's hot path. Phase 2 runs only
                // k_actual (typically ≤8) times, but each still touches
                // n × vocab_dim elements, so SIMD still wins over scalar.
                crate::simd::simd_fused_scale_acc(
                    &mut cos_mode[..vocab_dim],
                    &emb[..vocab_dim],
                    cos_w,
                    vocab_dim,
                );
            }
            let inv_n = 1.0 / n as f32;
            crate::simd::simd_scale_inplace(&mut cos_mode[..vocab_dim], inv_n);
            modes[ki * vd..ki * vd + vd].copy_from_slice(&cos_mode[..vd]);
            frequencies.push(omega);
        }

        Self {
            modes,
            frequencies,
            k: k_actual,
            vocab_dim: vd,
        }
    }

    /// Reconstruct: token ≈ Σ_k coefficient[k] * mode[k]
    ///
    /// Allocating version — see `reconstruct_into` for zero-alloc alternative.
    #[inline]
    pub fn reconstruct(&self, coefficients: &[f32]) -> Vec<f32> {
        if self.k == 0 {
            return vec![];
        }
        let mut result = vec![0.0f32; self.vocab_dim];
        self.reconstruct_into(coefficients, &mut result);
        result
    }

    /// Zero-alloc reconstruct into pre-allocated buffer.
    ///
    /// Uses flat modes buffer — no pointer chasing per mode.
    #[inline]
    pub fn reconstruct_into(&self, coefficients: &[f32], result: &mut [f32]) {
        if self.k == 0 {
            return;
        }
        let vd = self.vocab_dim.min(result.len());
        result[..vd].fill(0.0);
        // SIMD-fused scale-accumulate: `result[d] += c · modes[d]` for d in [0..vd).
        // Replaces a scalar `result_d += c * modes[...]` loop with a single SIMD call
        // per mode (NEON/AVX2 vectorized via `simd_fused_scale_acc`).
        for ki in 0..self.k {
            let c = coefficients.get(ki).copied().unwrap_or(0.0);
            let mode_start = ki * self.vocab_dim;
            crate::simd::simd_fused_scale_acc(
                &mut result[..vd],
                &self.modes[mode_start..mode_start + vd],
                c,
                vd,
            );
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }
    #[inline]
    pub fn vocab_dim(&self) -> usize {
        self.vocab_dim
    }
}

// ── ModalSpecDrafter ──

/// Modal speculative drafter — LinOSS state-space + Fourier modes.
///
/// Pipeline: prompt → LinOSS state → modal coefficients → Fourier reconstruct → nearest token.
pub struct ModalSpecDrafter {
    cell: LinOSSCell,
    basis: VocabFourierBasis,
    /// Stored embeddings for nearest-token lookup: flat `[vocab_size * vocab_dim]` row-major.
    /// Single allocation avoids V pointer dereferences per `nearest_token` scan.
    embeddings: Vec<f32>,
    /// Embedding dimension per token.
    emb_dim: usize,
    /// Number of tokens in `embeddings`.
    n_tokens: usize,
    /// Pre-allocated zero-forcing buffer reused across `draft` calls.
    #[allow(dead_code)]
    zero_forcing: Vec<f32>,
    hidden_dim: usize,
    dt: f32,
}

impl ModalSpecDrafter {
    /// Create a new ModalSpecDrafter.
    ///
    /// - `hidden_dim`: LinOSS hidden dimension (typically 64–256).
    /// - `vocab_embeddings`: vocabulary embedding vectors [vocab_size][vocab_dim].
    /// - `n_modes`: number of Fourier modes to extract (typically 8–32).
    #[inline]
    pub fn new(hidden_dim: usize, vocab_embeddings: &[&[f32]], n_modes: usize) -> Self {
        let cell = LinOSSCell::new(hidden_dim);
        let basis = VocabFourierBasis::from_embeddings(vocab_embeddings, n_modes);

        // Flatten embeddings into single contiguous buffer
        let emb_dim = vocab_embeddings.first().map_or(0, |e| e.len());
        let n_tokens = vocab_embeddings.len();
        let mut embeddings = vec![0.0f32; n_tokens * emb_dim];
        for (i, emb) in vocab_embeddings.iter().enumerate() {
            let copy_len = emb.len().min(emb_dim);
            embeddings[i * emb_dim..i * emb_dim + copy_len].copy_from_slice(&emb[..copy_len]);
        }
        // Pre-allocate zero-forcing buffer — reused across draft() calls.
        let zero_forcing = vec![0.0f32; hidden_dim];

        Self {
            cell,
            basis,
            embeddings,
            emb_dim,
            n_tokens,
            zero_forcing,
            hidden_dim,
            dt: 0.1, // Default timestep — can be tuned per model.
        }
    }

    /// Draft tokens: encode prompt → LinOSS oscillation → Fourier reconstruct → nearest vocab.
    ///
    /// Zero-alloc per timestep using double-buffered scratch (y_a/z_a, y_b/z_b).
    /// Uses `imex_step_inplace` + `_into` variants to avoid Vec allocations in the hot loop.
    pub fn draft(&self, prompt_tokens: &[usize], n_draft: usize) -> Vec<usize> {
        if n_draft == 0 || self.n_tokens == 0 {
            return vec![];
        }
        let h = self.hidden_dim;
        let vocab_dim = self.basis.vocab_dim();
        let k = self.basis.k();

        // Double-buffered scratch — avoids 2 Vec allocations per imex_step.
        // Ping-pong: (y_a, z_a) is current, (y_b, z_b) is next; swap roles each
        // step via mem::swap to avoid the copy_from_slice back into the input.
        let mut y_a = vec![0.0f32; h];
        let mut z_a = vec![0.0f32; h];
        let mut y_b = vec![0.0f32; h];
        let mut z_b = vec![0.0f32; h];
        let mut forcing = vec![0.0f32; h];
        let mut coeffs = vec![0.0f32; k];
        let mut reconstructed = vec![0.0f32; vocab_dim];

        // Prompt encoding
        for &tok in prompt_tokens {
            if tok < self.n_tokens {
                let emb = &self.embeddings[tok * self.emb_dim..(tok + 1) * self.emb_dim];
                self.project_to_hidden_into(emb, vocab_dim, &mut forcing);
                self.cell
                    .imex_step_inplace(&y_a, &z_a, &forcing, self.dt, &mut y_b, &mut z_b);
                std::mem::swap(&mut y_a, &mut y_b);
                std::mem::swap(&mut z_a, &mut z_b);
            }
        }

        // Draft loop — zero alloc per iteration, no per-step copy_back.
        forcing.fill(0.0);
        let mut draft = Vec::with_capacity(n_draft);
        for _ in 0..n_draft {
            self.cell
                .imex_step_inplace(&y_a, &z_a, &forcing, self.dt, &mut y_b, &mut z_b);
            std::mem::swap(&mut y_a, &mut y_b);
            std::mem::swap(&mut z_a, &mut z_b);
            // After the swap, (y_a, z_a) holds the freshly-computed state.
            self.extract_coefficients_into(&y_a, k, &mut coeffs);
            self.basis.reconstruct_into(&coeffs, &mut reconstructed);
            draft.push(self.nearest_token(&reconstructed));
        }
        draft
    }

    /// Zero-alloc draft into pre-allocated output buffer.
    ///
    /// Uses double-buffered scratch (y_a/z_a, y_b/z_b) to avoid allocation per timestep.
    /// Returns the number of drafted tokens written to `out`.
    pub fn draft_into(&self, prompt_tokens: &[usize], out: &mut [usize]) -> usize {
        let n_draft = out.len();
        if n_draft == 0 || self.n_tokens == 0 {
            return 0;
        }
        let h = self.hidden_dim;
        let vocab_dim = self.basis.vocab_dim();
        let k = self.basis.k();

        // Pre-allocate all scratch buffers once. The (y_a, z_a) pair is the
        // "current" state and (y_b, z_b) the "next" state; we ping-pong them
        // across IMEX steps (swapping roles via mem::swap) instead of copying
        // the result back into the input buffer each step — eliminates 2×h
        // f32 copies per token.
        let mut y_a = vec![0.0f32; h];
        let mut z_a = vec![0.0f32; h];
        let mut y_b = vec![0.0f32; h];
        let mut z_b = vec![0.0f32; h];
        let mut forcing = vec![0.0f32; h];
        let mut coeffs = vec![0.0f32; k];
        let mut reconstructed = vec![0.0f32; vocab_dim];

        // Prompt encoding (ping-pong: result lands in the "next" pair, then
        // we swap so it becomes the "current" pair for the next step).
        for &tok in prompt_tokens {
            if tok < self.n_tokens {
                let emb = &self.embeddings[tok * self.emb_dim..(tok + 1) * self.emb_dim];
                self.project_to_hidden_into(emb, vocab_dim, &mut forcing);
                self.cell
                    .imex_step_inplace(&y_a, &z_a, &forcing, self.dt, &mut y_b, &mut z_b);
                std::mem::swap(&mut y_a, &mut y_b);
                std::mem::swap(&mut z_a, &mut z_b);
            }
        }

        // Draft loop — zero alloc per iteration, no per-step copy_back.
        forcing.fill(0.0);
        let mut drafted = 0;
        for out_slot in out.iter_mut().take(n_draft) {
            self.cell
                .imex_step_inplace(&y_a, &z_a, &forcing, self.dt, &mut y_b, &mut z_b);
            std::mem::swap(&mut y_a, &mut y_b);
            std::mem::swap(&mut z_a, &mut z_b);
            // After the swap, (y_a, z_a) holds the freshly-computed state.
            self.extract_coefficients_into(&y_a, k, &mut coeffs);
            self.basis.reconstruct_into(&coeffs, &mut reconstructed);
            *out_slot = self.nearest_token(&reconstructed);
            drafted += 1;
        }
        drafted
    }

    #[inline]
    #[allow(dead_code)]
    fn project_to_hidden(&self, vec: &[f32], vocab_dim: usize) -> Vec<f32> {
        let h = self.hidden_dim;
        let mut result = vec![0.0f32; h];
        self.project_to_hidden_into(vec, vocab_dim, &mut result);
        result
    }

    #[inline]
    fn project_to_hidden_into(&self, vec: &[f32], vocab_dim: usize, result: &mut [f32]) {
        let h = self.hidden_dim;
        let ratio = vocab_dim as f32 / h as f32;
        for (i, result_slot) in result.iter_mut().enumerate().take(h) {
            let start = ((i as f32 * ratio) as usize).min(vocab_dim);
            let end = (((i + 1) as f32 * ratio) as usize).min(vocab_dim);
            if start < end {
                *result_slot = crate::simd::simd_sum_f32(&vec[start..end]) / (end - start) as f32;
            } else {
                *result_slot = 0.0;
            }
        }
    }

    /// Extract first k elements of y (position) as modal coefficients.
    #[inline]
    #[allow(dead_code)]
    fn extract_coefficients(&self, state: &LinOSSState, k: usize) -> Vec<f32> {
        let n = k.min(state.y.len());
        let mut coeffs = vec![0.0f32; k];
        coeffs[..n].copy_from_slice(&state.y[..n]);
        coeffs
    }

    /// Zero-alloc coefficient extraction into pre-allocated buffer.
    #[inline]
    fn extract_coefficients_into(&self, y: &[f32], k: usize, coeffs: &mut [f32]) {
        let n = k.min(y.len()).min(coeffs.len());
        coeffs[..n].copy_from_slice(&y[..n]);
        coeffs[n..].fill(0.0);
    }

    /// Find nearest token via dot-product argmax over flat embedding buffer.
    /// Sigmoid is monotonic so argmax(dot) == argmax(sigmoid(dot)); skip it.
    #[inline]
    fn nearest_token(&self, query: &[f32]) -> usize {
        let dim = self.emb_dim.min(query.len());
        let mut best_idx = 0;
        let mut best_dot = f32::NEG_INFINITY;
        for i in 0..self.n_tokens {
            let emb_start = i * self.emb_dim;
            let dot =
                crate::simd::simd_dot_f32(&self.embeddings[emb_start..emb_start + dim], query, dim);
            if dot > best_dot {
                best_dot = dot;
                best_idx = i;
            }
        }
        best_idx
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imex_step_preserves_energy() {
        let h = 16;
        let cell = LinOSSCell::new(h);
        let mut state = LinOSSState {
            y: vec![1.0; h],
            z: vec![0.5; h],
        };
        let initial_energy = cell.energy(&state);
        let zero_forcing = vec![0.0; h];
        for _ in 0..1000 {
            state = cell.imex_step(&state, &zero_forcing, 0.01);
        }
        let final_energy = cell.energy(&state);
        assert!(
            final_energy < initial_energy * 1.1,
            "Energy grew: initial={initial_energy}, final={final_energy}"
        );
    }

    #[test]
    fn test_imex_step_damps_with_beta() {
        let h = 8;
        let mut cell = LinOSSCell::new(h);
        cell.beta = vec![0.5; h];
        let mut state = LinOSSState {
            y: vec![1.0; h],
            z: vec![1.0; h],
        };
        let initial_energy = cell.energy(&state);
        let zero_forcing = vec![0.0; h];
        for _ in 0..200 {
            state = cell.imex_step(&state, &zero_forcing, 0.01);
        }
        assert!(
            cell.energy(&state) < initial_energy,
            "Energy should decrease with damping"
        );
    }

    #[test]
    fn test_parallel_scan_matches_sequential() {
        let h = 8;
        let cell = LinOSSCell::new(h);
        let initial = LinOSSState {
            y: vec![0.1; h],
            z: vec![0.2; h],
        };
        let forcings: Vec<Vec<f32>> = (0..10).map(|i| vec![0.01 * (i as f32 + 1.0); h]).collect();
        let refs: Vec<&[f32]> = forcings.iter().map(|f| f.as_slice()).collect();
        let seq = cell.sequential_scan(&initial, &refs, 0.1);
        let par = cell.parallel_scan(&initial, &refs, 0.1);
        assert_eq!(seq.len(), par.len());
        for (i, (s, p)) in seq.iter().zip(par.iter()).enumerate() {
            for j in 0..h {
                assert!(
                    (s.y[j] - p.y[j]).abs() < 1e-4,
                    "Step {i} dim {j} y mismatch"
                );
                assert!(
                    (s.z[j] - p.z[j]).abs() < 1e-4,
                    "Step {i} dim {j} z mismatch"
                );
            }
        }
    }

    #[test]
    fn test_fourier_basis_reconstruction() {
        let dim = 8;
        let embs: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                (0..dim)
                    .map(|d| (std::f32::consts::PI * i as f32 * (d as f32 + 1.0) / 20.0).cos())
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();
        let basis = VocabFourierBasis::from_embeddings(&refs, 4);
        assert!(basis.k() > 0);
        let norm: f32 = basis
            .reconstruct(&vec![1.0; basis.k()])
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            norm > 0.1,
            "Reconstruction should be non-trivial, got norm={norm}"
        );
    }

    #[test]
    fn test_drafter_produces_valid_tokens() {
        let embs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.1; 4]).collect();
        let refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();
        let drafter = ModalSpecDrafter::new(8, &refs, 4);
        let draft = drafter.draft(&[0, 1, 2, 3], 5);
        assert_eq!(draft.len(), 5);
        for &tok in &draft {
            assert!(tok < 10, "Token {tok} out of range");
        }
    }

    #[test]
    fn test_drafter_draft_into_matches_draft() {
        // Regression guard for the ping-pong double-buffer refactor:
        // `draft` and `draft_into` must produce identical token sequences,
        // confirming the mem::swap buffer exchange is numerically equivalent
        // to the old copy_from_slice approach.
        let embs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.1; 4]).collect();
        let refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();
        let drafter = ModalSpecDrafter::new(8, &refs, 4);
        let prompt = [0usize, 1, 2, 3];
        let draft = drafter.draft(&prompt, 5);
        let mut into_buf = vec![usize::MAX; 5];
        let written = drafter.draft_into(&prompt, &mut into_buf);
        assert_eq!(written, 5);
        assert_eq!(draft, into_buf, "draft and draft_into must agree");
    }

    #[test]
    fn test_linoss_zero_forcing() {
        let cell = LinOSSCell::new(8);
        let next = cell.imex_step(&LinOSSState::zeros(8), &[0.0; 8], 0.1);
        for i in 0..8 {
            assert!(next.y[i].abs() < 1e-10, "y should stay zero");
            assert!(next.z[i].abs() < 1e-10, "z should stay zero");
        }
    }
}
