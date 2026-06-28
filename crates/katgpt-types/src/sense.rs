//! Sense composition types.

// ---------------------------------------------------------------------------
// Shard Embedding — JL Random Orthogonal Projection (Plan 230)
// ---------------------------------------------------------------------------

/// Low-dimensional projection of NeuronShard style_weights for fast similarity search.
/// Produced by Johnson-Lindenstrauss random orthogonal projection.
/// 8 × f32 = 32 bytes — fits in cache line, suitable for SIMD cosine similarity.
///
/// Plan 230: Shard Embedding Projection — modelless linear weight-to-vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShardEmbedding(pub [f32; 8]);

impl ShardEmbedding {
    pub const ZERO: Self = Self([0.0; 8]);
    pub const DIM: usize = 8;

    /// Cosine similarity between two embeddings.
    /// Uses SIMD dot-product and SIMD sum-of-squares for the normalization.
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = crate::simd::simd_dot_f32(&self.0, &other.0, Self::DIM);
        let sq_a = crate::simd::simd_sum_sq(&self.0, Self::DIM);
        let sq_b = crate::simd::simd_sum_sq(&other.0, Self::DIM);
        let denom = sq_a * sq_b;
        if denom < 1e-16 {
            return 0.0;
        }
        let inv_norm = 1.0 / denom.sqrt();
        dot * inv_norm
    }

    /// Euclidean distance squared between two embeddings.
    /// Uses SIMD-accelerated fused-subtract-accumulate.
    #[inline]
    pub fn dist_sq(&self, other: &Self) -> f32 {
        crate::simd::simd_dist_sq(&self.0, &other.0, 8)
    }
}

impl Default for ShardEmbedding {
    fn default() -> Self {
        Self::ZERO
    }
}

// Hash for use as HashMap key (bit-level, NOT semantic hash)
impl std::hash::Hash for ShardEmbedding {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Single write of all 32 bytes — fewer virtual calls than four write_u64.
        state.write(unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, 32) });
    }
}

impl Eq for ShardEmbedding {}

// ---------------------------------------------------------------------------
// Sense Composition — KG Latent Octree (Plan 221)
// ---------------------------------------------------------------------------

/// Kind of sense module for NPC brain composition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SenseKind {
    CommonSense = 0,
    FighterSense = 1,
    GameTheorySense = 2,
    #[default]
    SpatialSense = 3,
    SocialSense = 4,
    SkillSense = 5,
    /// LinOSS Modal Threat Prediction (Plan 241). Requires `spectral_threat` feature.
    #[cfg(feature = "spectral_threat")]
    SpectralThreat = 6,
    Reserved = 7,
}

/// Ternary direction vector: +1/0/-1 encoded as two bitmasks + row scale.
/// 20 bytes each (u64, u64, f32).
/// `#[repr(C)]` required — embedded in `SenseModule` which hashes raw bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TernaryDir {
    /// Bitmask for positive (+1) entries.
    pub pos_bits: u64,
    /// Bitmask for negative (-1) entries.
    pub neg_bits: u64,
    /// Scale factor for this direction row.
    pub row_scale: f32,
}

impl TernaryDir {
    pub const SIZE: usize = 20;

    pub fn zero() -> Self {
        Self {
            pos_bits: 0,
            neg_bits: 0,
            row_scale: 0.0,
        }
    }

    /// Zero padding bytes for deterministic hashing.
    /// TernaryDir is 20 logical bytes but 24 with alignment padding.
    /// This zeroes the 4 trailing padding bytes in one write.
    #[inline(always)]
    pub fn zero_padding(&mut self) {
        // Write 4 zero bytes at once via u32 store instead of 4 individual byte writes
        unsafe {
            let ptr = self as *mut Self as *mut u8;
            (ptr.add(20) as *mut u32).write(0);
        }
    }
}

/// Fixed-size sense module: KG latent octree + ternary direction vectors.
/// ~232 bytes. BLAKE3 committed.
///
/// Field ordering: u64-aligned first, then f32, then u8 tail.
/// `commitment` must remain LAST for the hash-until-end-of-struct pattern.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct SenseModule {
    /// Octree occupancy bit-planes (up to depth 3 → 128 nodes in 2×u64).
    pub octree_bits: [u64; 4],
    /// Ternary direction vectors for projection.
    pub directions: [TernaryDir; 8],
    /// Module confidence [0, 1].
    pub confidence: f32,
    pub kind: SenseKind,
    pub version: u8,
    pub octree_depth: u8,
    pub n_directions: u8,
    pub _reserved: u8,
    /// BLAKE3 commitment over all preceding fields.
    pub commitment: [u8; 32],
}

impl SenseModule {
    /// Project HLA state onto this module's ternary directions → sigmoid scalar.
    ///
    /// KG weight bridge: output is scaled by module confidence so that
    /// high-confidence KG triples produce stronger sense activations and
    /// low-confidence triples are attenuated. Confidence 1.0 = unchanged.
    ///
    /// Optimized: branch-free bit extraction via shift+AND, flat loop
    /// (LLVM auto-vectorizes better than chunked), bounded exp sigmoid.
    #[inline(always)]
    pub fn project(&self, hla_state: &[f32; 8]) -> f32 {
        let n = self.n_directions as usize;
        let mut dot = 0.0f32;

        // Unrolled-friendly flat loop: extract ternary sign per-dim, FMA into dot.
        // bool-as-u32 then cast to f32 is zero-extend (no int-to-float conversion).
        // Zip iteration elides bounds checks on `self.directions[i]` (verified safe
        // by `n_directions ≤ 8` but the runtime bound `n` defeats LLVM's elision).
        for (i, (hla_val, dir)) in hla_state.iter().zip(self.directions.iter()).enumerate().take(n) {
            let pos = ((dir.pos_bits >> i) & 1) as u32 as f32;
            let neg = ((dir.neg_bits >> i) & 1) as u32 as f32;
            // sign ∈ {-1, 0, +1} — single FMA: dot += (sign * hla_val) * scale.
            // sign * hla_val is computed first, then FMA-fused with scale + dot.
            let sign = pos - neg;
            dot = (sign * hla_val).mul_add(dir.row_scale, dot);
        }

        // Sigmoid * confidence — uses shared crate::simd::fast_sigmoid (bounded (0,1))
        self.confidence * crate::simd::fast_sigmoid(dot)
    }

    /// Query octree occupancy at given level and index.
    /// Returns None if indices out of bounds.
    pub fn query_octree(&self, level: u8, index: u8) -> Option<bool> {
        let nodes_at_level = 1usize << (level * 2); // quadtree-like
        if index as usize >= nodes_at_level || level > self.octree_depth {
            return None;
        }
        // flat_idx = 0 for all levels in this simplified indexing;
        // the fold computes 0 * 4^level = 0, so flat_idx = index.
        let flat_idx = index as usize;
        let word = flat_idx / 64;
        let bit = flat_idx % 64;
        if word >= self.octree_bits.len() {
            return None;
        }
        Some(self.octree_bits[word] & (1 << bit) != 0)
    }

    /// Compute and store BLAKE3 commitment.
    /// Zeros TernaryDir padding bytes first for deterministic hashing.
    pub fn commit(&mut self) {
        // Zero commitment before hashing
        self.commitment = [0u8; 32];
        // Zero padding in direction vectors for deterministic hash
        for dir in &mut self.directions {
            dir.zero_padding();
        }
        let size_before_commit = std::mem::offset_of!(SenseModule, commitment);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self as *const Self as *const u8, size_before_commit)
        };
        self.commitment = *blake3::hash(bytes).as_bytes();
    }

    /// Verify BLAKE3 commitment.
    /// Uses a stack buffer to avoid cloning the entire 232-byte struct.
    /// Zeros TernaryDir padding bytes before comparing to match commit() behavior.
    pub fn verify(&self) -> bool {
        let size_before_commit = std::mem::offset_of!(SenseModule, commitment);
        let mut buf = [0u8; std::mem::size_of::<SenseModule>()];
        // Copy raw bytes to stack buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                self as *const Self as *const u8,
                buf.as_mut_ptr(),
                size_before_commit,
            );
        }
        // Zero commitment region and trailing padding in buffer
        buf[size_before_commit..].fill(0);
        // Zero TernaryDir padding in buffer — each TernaryDir is 24 bytes, padding at bytes 20..24
        let dirs_offset = std::mem::offset_of!(SenseModule, directions);
        let dir_size = std::mem::size_of::<TernaryDir>();
        for i in 0..8 {
            let dir_start = dirs_offset + i * dir_size;
            // Zero 4 padding bytes at offset 20 within each TernaryDir
            buf[dir_start + 20..dir_start + dir_size].fill(0);
        }
        let bytes = &buf[..size_before_commit];
        let expected = blake3::hash(bytes);
        self.commitment == *expected.as_bytes()
    }
}

impl Default for SenseModule {
    fn default() -> Self {
        Self {
            octree_bits: [0; 4],
            directions: [TernaryDir::zero(); 8],
            confidence: 0.0,
            kind: SenseKind::default(),
            version: 1,
            octree_depth: 3,
            n_directions: 0,
            _reserved: 0,
            commitment: [0u8; 32],
        }
    }
}

// ---------------------------------------------------------------------------
// DilationConfig — RAT+ Recurrence Bridge sparse attention (Plan 225)
// ---------------------------------------------------------------------------

/// Dilation configuration for RAT+ bridge sparse attention.
/// Controls stride D for KV cache access during decode.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DilationConfig {
    D1 = 1, // Dense (no dilation)
    D2 = 2,
    D4 = 4,
    D8 = 8,
    D16 = 16,
    D32 = 32,
    D64 = 64,
}

impl DilationConfig {
    /// Returns the dilation stride as usize.
    #[inline]
    pub fn stride(&self) -> usize {
        *self as usize
    }

    /// Select dilation from queries-per-second heuristic.
    /// Low QPS → dense, High QPS → aggressive dilation.
    pub fn from_qps(qps: f32) -> Self {
        match qps {
            ..1.0 => Self::D1,
            1.0..5.0 => Self::D4,
            5.0..20.0 => Self::D16,
            _ => Self::D64,
        }
    }
}
