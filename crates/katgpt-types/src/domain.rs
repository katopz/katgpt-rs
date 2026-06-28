//! DomainLatent embedding.

// ---------------------------------------------------------------------------
// DomainLatent — feature-gated (Plan 038)
// ---------------------------------------------------------------------------

/// Domain latent embedding for mid-layer conditioning (Plan 038).
///
/// Injected at layer `n_layer / 2` by adding to K and V projections before cache write.
/// Inspired by the Free Transformer's mid-layer latent injection, adapted for
/// supervised domain conditioning via LoRA fine-tuning.
///
/// Shape: `[kv_dim]` — one embedding per domain, matching K/V dimension for GQA.
///
/// # Binary format
///
/// ```text
/// [MAGIC: "DLAT" 4B][VERSION: 1B][KV_DIM: 4B LE][EMBEDDING: kv_dim × f32 LE][BLAKE3: 32B]
/// ```
///
/// BLAKE3 checksum covers everything before it (magic through embedding).
#[cfg(feature = "domain_latent")]
#[derive(Debug)]
pub struct DomainLatent {
    /// Domain embedding vector, shape `[kv_dim]`.
    pub embedding: Vec<f32>,
}

#[cfg(feature = "domain_latent")]
impl DomainLatent {
    const MAGIC: &[u8; 4] = b"DLAT";
    const VERSION: u8 = 1;

    /// Load domain latent from binary file.
    ///
    /// Format: `[MAGIC 4B][VERSION 1B][KV_DIM 4B LE][EMBEDDING kv_dim×f32 LE][BLAKE3 32B]`
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let data =
            std::fs::read(path).map_err(|e| format!("Failed to read domain_latent file: {e}"))?;

        // Minimum: magic(4) + version(1) + kv_dim(4) + hash(32) = 41
        if data.len() < 41 {
            return Err("File too small for domain_latent header".into());
        }

        // Validate BLAKE3 checksum — last 32 bytes cover everything before them
        let payload_end = data.len() - 32;
        let stored_checksum = &data[payload_end..];
        let computed = blake3::hash(&data[..payload_end]);
        if computed.as_bytes() != stored_checksum {
            return Err("Domain latent file checksum mismatch".into());
        }

        let mut offset = 0usize;

        // Magic
        if &data[offset..offset + 4] != Self::MAGIC {
            return Err("Invalid domain_latent magic bytes".into());
        }
        offset += 4;

        // Version
        let version = data[offset];
        if version != Self::VERSION {
            return Err(format!("Unsupported domain_latent version: {version}"));
        }
        offset += 1;

        // KV_DIM
        let kv_dim = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("kv_dim parse: {e}"))?,
        ) as usize;
        offset += 4;

        // Embedding data — bulk copy on LE targets, element-by-element otherwise
        let embed_bytes = kv_dim * std::mem::size_of::<f32>();
        if offset + embed_bytes > payload_end {
            return Err(format!(
                "Truncated embedding data: expected {embed_bytes} bytes at offset {offset}, payload ends at {payload_end}"
            ));
        }

        let embedding: Vec<f32> = {
            #[cfg(target_endian = "little")]
            {
                let mut v = Vec::with_capacity(kv_dim);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data[offset..].as_ptr(),
                        v.as_mut_ptr() as *mut u8,
                        embed_bytes,
                    );
                    v.set_len(kv_dim);
                }
                v
            }
            #[cfg(not(target_endian = "little"))]
            {
                data[offset..offset + embed_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                    .collect()
            }
        };

        if embedding.len() != kv_dim {
            return Err(format!(
                "Embedding length mismatch: got {}, expected {kv_dim}",
                embedding.len()
            ));
        }

        Ok(Self { embedding })
    }

    /// Save domain latent to binary file (for tests and training export).
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let kv_dim = self.embedding.len();
        let embed_bytes = kv_dim * std::mem::size_of::<f32>();
        let payload_len = 4 + 1 + 4 + embed_bytes;
        let mut buf = Vec::with_capacity(payload_len + 32);

        buf.extend_from_slice(Self::MAGIC);
        buf.push(Self::VERSION);
        buf.extend_from_slice(&(kv_dim as u32).to_le_bytes());
        // Bulk write embedding data — avoids per-element extend_from_slice overhead.
        // SAFETY: f32 is plain-old-data with no padding; to_ne_bytes gives [u8; 4] per f32.
        // On LE targets (all Apple Silicon, all modern x86), to_ne_bytes == to_le_bytes.
        #[cfg(target_endian = "little")]
        {
            let bytes = unsafe {
                std::slice::from_raw_parts(self.embedding.as_ptr() as *const u8, embed_bytes)
            };
            buf.extend_from_slice(bytes);
        }
        #[cfg(not(target_endian = "little"))]
        {
            for &val in &self.embedding {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        let hash = blake3::hash(&buf);
        buf.extend_from_slice(hash.as_bytes());

        std::fs::write(path, &buf)
            .map_err(|e| format!("Failed to write domain_latent file: {e}"))?;

        Ok(())
    }

    /// Create a zero-initialized domain latent of the given kv_dim.
    pub fn zeros(kv_dim: usize) -> Self {
        Self {
            embedding: vec![0.0; kv_dim],
        }
    }

    /// Create a domain latent from a raw embedding vector.
    pub fn from_vec(embedding: Vec<f32>) -> Self {
        Self { embedding }
    }
}

// ---------------------------------------------------------------------------
// Binary helper functions
// ---------------------------------------------------------------------------

pub(super) fn read_u32_le(data: &[u8], offset: &mut usize) -> Result<u32, String> {
    if *offset + 4 > data.len() {
        return Err("Unexpected end of data reading u32".into());
    }
    let val = u32::from_le_bytes(
        data[*offset..*offset + 4]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("u32 parse: {e}"))?,
    );
    *offset += 4;
    Ok(val)
}

pub(super) fn read_f32_le(data: &[u8], offset: &mut usize) -> Result<f32, String> {
    if *offset + 4 > data.len() {
        return Err("Unexpected end of data reading f32".into());
    }
    let val = f32::from_le_bytes(
        data[*offset..*offset + 4]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("f32 parse: {e}"))?,
    );
    *offset += 4;
    Ok(val)
}

pub(super) fn read_u16_le(data: &[u8], offset: &mut usize) -> Result<u16, String> {
    if *offset + 2 > data.len() {
        return Err("Unexpected end of data reading u16".into());
    }
    let val = u16::from_le_bytes(
        data[*offset..*offset + 2]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("u16 parse: {e}"))?,
    );
    *offset += 2;
    Ok(val)
}
