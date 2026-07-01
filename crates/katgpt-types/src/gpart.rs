//! GPart Isometric Partition Adapter.

// ---------------------------------------------------------------------------
// GPart Isometric Partition Adapter — replaces LoRA BA with Pθ_d (Research 227, Plan 257)
// ---------------------------------------------------------------------------

/// Binary format magic for GPart adapter files.
#[cfg(feature = "gpart_adapter")]
pub const GPART_MAGIC: &[u8; 5] = b"GPART";
/// Current GPart binary format version.
#[cfg(feature = "gpart_adapter")]
pub const GPART_VERSION: u32 = 1;

/// GPart isometric partition adapter — replaces LoRA's bilinear BA with single isometric Pθ_d.
///
/// `W = W₀ + Pθ_d` where `P^T P = I_d` (isometric, no bilinear distortion).
/// Storage: d + 1 values (seed + theta). Single-pass O(N) broadcast apply.
///
/// Reference: Research 227 (arXiv 2605.14841)
#[cfg(feature = "gpart_adapter")]
#[derive(Clone, Debug)]
pub struct GpartAdapter {
    /// Partition dimension — the only hyperparameter.
    pub d: usize,
    /// Seed for deterministic pseudorandom partition generation.
    pub seed: u64,
    /// Trainable vector θ_d ∈ R^d (produced by training side, riir-ai).
    pub theta: Vec<f32>,
}

#[cfg(feature = "gpart_adapter")]
impl GpartAdapter {
    /// Apply the isometric partition adapter to base weights in-place.
    /// `W = W₀ + Pθ_d` where P is seed-generated.
    /// Single-pass O(N) broadcast. SIMD-friendly (chunked 4/8).
    pub fn apply(&self, base_weights: &mut [f32]) {
        let n = base_weights.len();
        if n == 0 || self.d == 0 {
            return;
        }

        let assignments = self.generate_assignments(n);
        let group_sizes = self.compute_group_sizes(n, &assignments);

        // Pre-compute per-group delta O(d): avoids recomputing sqrt+division for
        // each of N elements. Same approach as `apply_simd()` / `GpartPrepared`.
        let mut group_delta = vec![0.0f32; self.d];
        for g in 0..self.d {
            let scale = 1.0 / (group_sizes[g] as f32).sqrt();
            group_delta[g] = scale * self.theta[g];
        }
        // Branch-free inner loop: pure array lookup, auto-vectorizable.
        for i in 0..n {
            base_weights[i] += group_delta[assignments[i]];
        }
    }

    /// Apply with pre-allocated scratch buffers — zero alloc in hot path.
    pub fn apply_with_scratch(
        &self,
        base_weights: &mut [f32],
        assignments: &mut [usize],
        group_sizes: &mut [usize],
    ) {
        let n = base_weights.len();
        if n == 0 || self.d == 0 {
            return;
        }

        self.generate_assignments_into(n, assignments);
        self.compute_group_sizes_into(n, assignments, group_sizes);

        // Pre-compute per-group delta O(d): eliminates N recomputations of sqrt+division.
        // `d` is small (≤ 90), so a single alloc of `d` f32s amortizes across N elements.
        let mut group_delta = vec![0.0f32; self.d];
        for g in 0..self.d {
            let scale = 1.0 / (group_sizes[g] as f32).sqrt();
            group_delta[g] = scale * self.theta[g];
        }
        // Branch-free inner loop: pure array lookup, auto-vectorizable.
        for i in 0..n {
            base_weights[i] += group_delta[assignments[i]];
        }
    }

    /// Apply with pre-allocated scratch buffers AND a per-group boolean mask.
    ///
    /// Groups where `group_mask[g] == false` contribute zero to `base_weights`.
    /// This is the inference-time pruning path: low-magnitude groups are skipped,
    /// trading output fidelity for reduced effective FLOPs on the downstream matmul.
    ///
    /// `group_mask.len()` must be `>= self.d`. When `group_mask` is all-true this
    /// is identical to [`apply_with_scratch`].
    ///
    /// Gates: `gpart_pruning` feature (Issue 008). Static top-k magnitude mask;
    /// BanditPruner-driven dynamic mask is deferred to a separate plan.
    #[cfg(feature = "gpart_pruning")]
    pub fn apply_with_scratch_masked(
        &self,
        base_weights: &mut [f32],
        assignments: &mut [usize],
        group_sizes: &mut [usize],
        group_mask: &[bool],
    ) {
        let n = base_weights.len();
        if n == 0 || self.d == 0 {
            return;
        }
        debug_assert!(
            group_mask.len() >= self.d,
            "group_mask len {} < d {}",
            group_mask.len(),
            self.d,
        );

        self.generate_assignments_into(n, assignments);
        self.compute_group_sizes_into(n, assignments, group_sizes);

        // Branch-free inner loop: precompute per-group scale×θ, zeroed when masked.
        // LLVM vectorises the all-true path to match `apply_with_scratch`.
        let mut group_delta: Vec<f32> = Vec::with_capacity(self.d);
        for g in 0..self.d {
            let active = group_mask[g];
            let scale = 1.0 / (group_sizes[g] as f32).sqrt();
            // `active as f32` is 0.0 or 1.0 — multiply zeros out masked groups
            // without a per-element branch.
            group_delta.push(scale * self.theta[g] * (active as u8 as f32));
        }

        for i in 0..n {
            base_weights[i] += group_delta[assignments[i]];
        }
    }

    /// SIMD-optimised apply: pre-computes per-group delta once (O(d) divisions
    /// instead of O(N)), then applies in 8-wide chunks.
    ///
    /// The "SIMD" benefit here is from eliminating per-element division — the
    /// inner loop is branch-free (pure array lookups), which LLVM auto-vectorises
    /// well. For contiguous same-group runs, `simd_add_scalar_inplace` is used.
    pub fn apply_simd(&self, base_weights: &mut [f32]) {
        let n = base_weights.len();
        if n == 0 || self.d == 0 {
            return;
        }

        let assignments = self.generate_assignments(n);
        let group_sizes = self.compute_group_sizes(n, &assignments);

        // Pre-compute per-group delta: θ[g] / √n_g — one division per group
        let group_delta: Vec<f32> = (0..self.d)
            .map(|g| self.theta[g] / (group_sizes[g] as f32).sqrt())
            .collect();

        // Process in 8-wide chunks for auto-vectorisation
        let chunks = n / 8;
        for c in 0..chunks {
            let base = c * 8;
            for j in 0..8 {
                let i = base + j;
                base_weights[i] += group_delta[assignments[i]];
            }
        }
        // Scalar tail
        for i in (chunks * 8)..n {
            base_weights[i] += group_delta[assignments[i]];
        }
    }

    /// Build a top-k magnitude mask: keep the `k` groups with largest `|θ[g]|`.
    ///
    /// Returns a `Vec<bool>` of length `d` where `true` = active. When `k >= d`, all
    /// groups are active (no pruning). This is the static selection policy for
    /// [`apply_with_scratch_masked`] — cheap to compute, deterministic, requires no
    /// reward signal. Dynamic bandit-based masking is deferred to a separate plan.
    ///
    /// Gates: `gpart_pruning` feature (Issue 008).
    #[cfg(feature = "gpart_pruning")]
    pub fn topk_mask(&self, k: usize) -> Vec<bool> {
        if k >= self.d {
            return vec![true; self.d];
        }
        if k == 0 {
            return vec![false; self.d];
        }

        // (|θ[g]|, g) pairs — partial sort by magnitude to find the top-k threshold.
        let mut magnitudes: Vec<(f32, usize)> =
            (0..self.d).map(|g| (self.theta[g].abs(), g)).collect();
        // `select_nth_unstable_by` partitions around the k-th largest in O(d) avg.
        // Descending comparator: after it returns, indices [0..k] hold the k
        // largest magnitudes (unordered).
        magnitudes.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));

        let mut mask = vec![false; self.d];
        for &(_, g) in &magnitudes[..k] {
            mask[g] = true;
        }
        mask
    }

    /// Generate group assignments from seed.
    /// Uses fastrand for deterministic cross-platform pseudorandom permutation.
    fn generate_assignments(&self, n: usize) -> Vec<usize> {
        let mut rng = fastrand::Rng::with_seed(self.seed);
        let mut assignments = Vec::with_capacity(n);
        for _ in 0..n {
            assignments.push(rng.usize(..self.d));
        }
        assignments
    }

    /// Generate assignments into pre-allocated scratch buffer.
    fn generate_assignments_into(&self, n: usize, out: &mut [usize]) {
        let mut rng = fastrand::Rng::with_seed(self.seed);
        for i in 0..n.min(out.len()) {
            out[i] = rng.usize(..self.d);
        }
    }

    /// Compute group sizes from assignments.
    fn compute_group_sizes(&self, n: usize, assignments: &[usize]) -> Vec<usize> {
        let mut sizes = vec![0usize; self.d];
        for &g in &assignments[..n] {
            sizes[g] += 1;
        }
        sizes
    }

    /// Compute group sizes into pre-allocated scratch buffer.
    fn compute_group_sizes_into(&self, n: usize, assignments: &[usize], out: &mut [usize]) {
        for g in out.iter_mut().take(self.d) {
            *g = 0;
        }
        for &g in &assignments[..n] {
            out[g] += 1;
        }
    }

    /// BLAKE3 commitment over (seed || theta).
    pub fn commitment(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed.to_le_bytes());
        let theta_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.theta.as_ptr() as *const u8,
                self.theta.len() * std::mem::size_of::<f32>(),
            )
        };
        hasher.update(theta_bytes);
        *hasher.finalize().as_bytes()
    }

    /// Verify commitment matches.
    pub fn verify(&self, expected: &[u8; 32]) -> bool {
        self.commitment() == *expected
    }

    /// Save adapter to binary format:
    /// `[GPART(5) | version(4) | d(4) | seed(8) | blake3(32) | theta(d×4)]`
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let d_bytes = self.d as u32;
        let theta_byte_len = self.theta.len() * std::mem::size_of::<f32>();
        // magic(5) + version(4) + d(4) + seed(8) + commitment(32) + theta
        let total = 5 + 4 + 4 + 8 + 32 + theta_byte_len;
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(GPART_MAGIC);
        buf.extend_from_slice(&GPART_VERSION.to_le_bytes());
        buf.extend_from_slice(&d_bytes.to_le_bytes());
        buf.extend_from_slice(&self.seed.to_le_bytes());

        // Commitment placeholder — will overwrite after computing
        let commit_offset = buf.len();
        buf.extend_from_slice(&[0u8; 32]);

        // Theta data
        let theta_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.theta.as_ptr() as *const u8, theta_byte_len) };
        buf.extend_from_slice(theta_bytes);

        // Compute commitment over (seed || theta) and write into placeholder
        let commitment = self.commitment();
        buf[commit_offset..commit_offset + 32].copy_from_slice(&commitment);

        std::fs::write(path, &buf).map_err(|e| format!("Failed to write gpart file: {e}"))
    }

    /// Load adapter from binary format.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read gpart file: {e}"))?;

        // Minimum: magic(5) + version(4) + d(4) + seed(8) + commitment(32) = 53
        if file_data.len() < 53 {
            return Err("File too small for gpart header".into());
        }

        if &file_data[0..5] != GPART_MAGIC {
            return Err("Invalid gpart magic bytes".into());
        }

        let version = u32::from_le_bytes(
            file_data[5..9]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("Version parse: {e}"))?,
        );
        if version != GPART_VERSION {
            return Err(format!("Unsupported gpart version: {version}"));
        }

        let d = u32::from_le_bytes(
            file_data[9..13]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("d parse: {e}"))?,
        ) as usize;

        let seed = u64::from_le_bytes(
            file_data[13..21]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("seed parse: {e}"))?,
        );

        let stored_commitment = &file_data[21..53];

        let theta_bytes_start = 53;
        let theta_bytes_len = d * std::mem::size_of::<f32>();
        if theta_bytes_start + theta_bytes_len > file_data.len() {
            return Err("Truncated theta data".into());
        }

        // Load theta
        let theta: Vec<f32> = {
            #[cfg(target_endian = "little")]
            {
                let src = &file_data[theta_bytes_start..theta_bytes_start + theta_bytes_len];
                let count = d;
                let mut v = Vec::with_capacity(count);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        v.as_mut_ptr() as *mut u8,
                        theta_bytes_len,
                    );
                    v.set_len(count);
                }
                v
            }
            #[cfg(not(target_endian = "little"))]
            {
                file_data[theta_bytes_start..theta_bytes_start + theta_bytes_len]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                    .collect()
            }
        };

        // Verify commitment
        let adapter = Self { d, seed, theta };
        let computed = adapter.commitment();
        if computed != stored_commitment {
            return Err("GPart file commitment mismatch".into());
        }

        Ok(adapter)
    }

    /// Storage size in bytes (seed + theta).
    pub fn storage_bytes(&self) -> usize {
        8 + self.theta.len() * std::mem::size_of::<f32>()
    }

    /// Verify isometry: ||Pθ||² = ||θ||².
    /// For the partition matrix, each group g contributes n_g × (θ_g/√n_g)² = θ_g².
    /// Summing: Σ_g θ_g² = ||θ||². QED.
    pub fn check_isometry(&self, n: usize) -> bool {
        if n == 0 || self.d == 0 {
            return true;
        }

        let assignments = self.generate_assignments(n);
        let group_sizes = self.compute_group_sizes(n, &assignments);

        // Pre-compute per-group delta once (O(d) sqrt+div) instead of recomputing
        // for every one of N assignments. Matches the apply()/prepare() pattern.
        let mut group_delta = vec![0.0f32; self.d];
        for g in 0..self.d {
            let scale = 1.0 / (group_sizes[g] as f32).sqrt();
            group_delta[g] = scale * self.theta[g];
        }

        // Compute ||Pθ||² = Σ_i delta[assignment[i]]²
        let mut projected_norm_sq = 0.0f32;
        for &g in assignments.iter().take(n) {
            let delta = group_delta[g];
            projected_norm_sq += delta * delta;
        }

        // Compute ||θ||²
        let theta_norm_sq: f32 = self.theta.iter().map(|&v| v * v).sum();

        (projected_norm_sq - theta_norm_sq).abs() < 1e-3
    }
}

/// Pre-computed per-element deltas for zero-allocation apply.
///
/// Call [`GpartAdapter::prepare`] once for a given weight dimension `n`,
/// then [`GpartPrepared::apply`] in the hot loop — no RNG, no allocation.
#[cfg(feature = "gpart_adapter")]
#[derive(Clone, Debug)]
pub struct GpartPrepared {
    /// Per-element delta: `theta[group(i)] / sqrt(n_group)` for each weight index.
    deltas: Vec<f32>,
}

#[cfg(feature = "gpart_adapter")]
impl GpartAdapter {
    /// Pre-compute per-element deltas for a weight vector of length `n`.
    ///
    /// This is the **fast path** — call once after loading the adapter,
    /// then use [`GpartPrepared::apply`] in the hot inference loop.
    /// Cost: O(n) RNG + O(n) broadcast setup (amortised once per model load).
    pub fn prepare(&self, n: usize) -> GpartPrepared {
        if n == 0 || self.d == 0 {
            return GpartPrepared { deltas: Vec::new() };
        }
        let assignments = self.generate_assignments(n);
        let group_sizes = self.compute_group_sizes(n, &assignments);
        // Match apply() arithmetic exactly: scale = 1/sqrt(n_g), delta = scale * theta[g]
        let group_delta: Vec<f32> = (0..self.d)
            .map(|g| {
                let scale = 1.0 / (group_sizes[g] as f32).sqrt();
                scale * self.theta[g]
            })
            .collect();
        let deltas = assignments.iter().map(|&g| group_delta[g]).collect();
        GpartPrepared { deltas }
    }
}

#[cfg(feature = "gpart_adapter")]
impl GpartPrepared {
    /// Apply pre-computed deltas in-place: `w[i] += delta[i]`.
    ///
    /// Pure O(N) broadcast, zero allocation, branch-free inner loop.
    /// LLVM auto-vectorises this to SIMD.
    pub fn apply(&self, base_weights: &mut [f32]) {
        let len = base_weights.len().min(self.deltas.len());
        for (w, &delta) in base_weights.iter_mut().zip(self.deltas.iter()).take(len) {
            *w += delta;
        }
    }
}

/// A loaded GPart pair for modality-specific inference, mirroring LoraPair.
/// Reader is active during bidirectional prefill, writer during causal decode.
#[cfg(feature = "gpart_adapter")]
#[derive(Clone, Debug)]
pub struct GpartPair {
    /// GPart active during bidirectional prefill.
    pub reader: Option<GpartAdapter>,
    /// GPart active during causal decode.
    pub writer: Option<GpartAdapter>,
}

#[cfg(feature = "gpart_adapter")]
impl GpartPair {
    /// Empty pair — no GPart applied.
    pub fn none() -> Self {
        Self {
            reader: None,
            writer: None,
        }
    }

    /// Apply the reader (prefill) adapter to base weights, if present.
    pub fn apply_prefill(&self, base_weights: &mut [f32]) {
        if let Some(ref adapter) = self.reader {
            adapter.apply(base_weights);
        }
    }

    /// Apply the writer (decode) adapter to base weights, if present.
    pub fn apply_decode(&self, base_weights: &mut [f32]) {
        if let Some(ref adapter) = self.writer {
            adapter.apply(base_weights);
        }
    }
}

/// Conversion from LoRA to GPart (lossy, requires pre-computed θ_d).
///
/// **Note:** θ_d = P⁺ΔW must be computed by the riir-ai training pipeline.
/// This conversion is a placeholder until that pipeline provides θ_d.
#[cfg(feature = "gpart_adapter")]
impl TryFrom<&crate::LoraAdapter> for GpartAdapter {
    type Error = &'static str;

    fn try_from(_lora: &crate::LoraAdapter) -> Result<Self, Self::Error> {
        Err("GpartAdapter requires pre-computed θ_d from riir-ai training pipeline (P⁺ΔW)")
    }
}

// NeuronShard Pod compatibility: seed(8) + d(max=90)×4(360) = 368 bytes max.
// Exact-fit invariant — clippy::eq_op fires because 8+90*4 const-folds to 368,
// and clippy::assertions_on_constants fires because the whole thing is const-
// foldable. Both are deliberate: this is a compile-time budget guard, not a
// runtime check and not a mistake.
#[cfg(feature = "gpart_adapter")]
#[allow(clippy::eq_op, clippy::assertions_on_constants)]
const _: () = assert!(8 + 90 * 4 <= 368);
