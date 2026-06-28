//! Contiguous weight allocation — single-buffer layout for better L2 cache locality.
//!
//! TileRT insight: packing all model parameters into one contiguous allocation
//! with alignment padding improves spatial locality for sequential weight reads.
//! For micro configs the gain is marginal (weights fit in L2 anyway), but the
//! pattern scales to larger models.
//!
//! (Plan 102: TileRT Execution Pipeline — D2)

use std::mem::size_of;

use crate::TransformerWeights;

/// Offset + length into the contiguous weight buffer for one weight matrix.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct WeightSlice {
    offset: usize,
    len: usize,
}

/// Per-layer weight slice descriptors (offsets into the contiguous buffer).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LayerSlices {
    wq: WeightSlice,
    wk: WeightSlice,
    wv: WeightSlice,
    wo: WeightSlice,
    w1: WeightSlice,
    w2: WeightSlice,
}

/// All transformer weights packed into a single contiguous allocation.
///
/// Each weight matrix is accessed via slice methods: [`layer_wq`](Self::layer_wq), etc.
/// 64-byte alignment padding between matrices ensures cache-line alignment.
#[derive(Debug, Clone)]
pub struct ContiguousWeights {
    buffer: Vec<f32>,
    layers: Vec<LayerSlices>,
    wte: WeightSlice,
    wpe: WeightSlice,
    lm_head: WeightSlice,
}

/// Cache-line alignment padding (64 bytes = 16 × f32).
const ALIGN_F32: usize = 16;

/// Align `offset` up to the next 64-byte boundary (in f32 units).
#[inline]
fn align_up(offset: usize) -> usize {
    (offset + ALIGN_F32 - 1) & !(ALIGN_F32 - 1)
}

/// Write weight data into buffer at offset, returning the slice descriptor.
#[inline]
fn write_weight(buffer: &mut [f32], offset: usize, data: &[f32]) -> WeightSlice {
    buffer[offset..offset + data.len()].copy_from_slice(data);
    WeightSlice {
        offset,
        len: data.len(),
    }
}

/// Read a slice from the buffer.
#[inline]
fn read_slice(buffer: &[f32], slice: WeightSlice) -> &[f32] {
    &buffer[slice.offset..slice.offset + slice.len]
}

impl ContiguousWeights {
    /// Pack all weights from [`TransformerWeights`] into a single contiguous buffer.
    ///
    /// Layout: `wte | pad | wpe | pad | lm_head | pad | [layer_wq | pad | … | layer_w2 | pad] | …`
    pub fn from_weights(weights: &TransformerWeights) -> Self {
        // Phase 1: Calculate total size with alignment padding
        let mut offset = 0usize;

        let wte_off = align_up(offset);
        offset = wte_off + weights.wte.len();

        let wpe_off = align_up(offset);
        offset = wpe_off + weights.wpe.len();

        let lm_head_off = align_up(offset);
        offset = lm_head_off + weights.lm_head.len();

        let mut layer_offsets: Vec<[usize; 6]> = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let mut lo = [0usize; 6];
            lo[0] = align_up(offset);
            offset = lo[0] + layer.attn_wq.len();
            lo[1] = align_up(offset);
            offset = lo[1] + layer.attn_wk.len();
            lo[2] = align_up(offset);
            offset = lo[2] + layer.attn_wv.len();
            lo[3] = align_up(offset);
            offset = lo[3] + layer.attn_wo.len();
            lo[4] = align_up(offset);
            offset = lo[4] + layer.mlp_w1.len();
            lo[5] = align_up(offset);
            offset = lo[5] + layer.mlp_w2.len();
            layer_offsets.push(lo);
        }

        // Phase 2: Allocate and fill
        let mut buffer = vec![0.0f32; offset];

        let wte = write_weight(&mut buffer, wte_off, &weights.wte);
        let wpe = write_weight(&mut buffer, wpe_off, &weights.wpe);
        let lm_head = write_weight(&mut buffer, lm_head_off, &weights.lm_head);

        let layers: Vec<LayerSlices> = weights
            .layers
            .iter()
            .zip(layer_offsets.iter())
            .map(|(layer, lo)| LayerSlices {
                wq: write_weight(&mut buffer, lo[0], &layer.attn_wq),
                wk: write_weight(&mut buffer, lo[1], &layer.attn_wk),
                wv: write_weight(&mut buffer, lo[2], &layer.attn_wv),
                wo: write_weight(&mut buffer, lo[3], &layer.attn_wo),
                w1: write_weight(&mut buffer, lo[4], &layer.mlp_w1),
                w2: write_weight(&mut buffer, lo[5], &layer.mlp_w2),
            })
            .collect();

        Self {
            buffer,
            layers,
            wte,
            wpe,
            lm_head,
        }
    }

    // ── Global weights ──────────────────────────────────────────

    /// Token embedding table (`[vocab_size, n_embd]`).
    #[inline]
    pub fn wte(&self) -> &[f32] {
        read_slice(&self.buffer, self.wte)
    }

    /// Positional embedding table (`[block_size, n_embd]`).
    #[inline]
    pub fn wpe(&self) -> &[f32] {
        read_slice(&self.buffer, self.wpe)
    }

    /// LM head weight matrix (`[vocab_size, n_embd]`).
    #[inline]
    pub fn lm_head(&self) -> &[f32] {
        read_slice(&self.buffer, self.lm_head)
    }

    // ── Per-layer weights ───────────────────────────────────────

    /// Attention Q weight for `layer`.
    #[inline]
    pub fn layer_wq(&self, layer: usize) -> &[f32] {
        read_slice(&self.buffer, self.layers[layer].wq)
    }

    /// Attention K weight for `layer`.
    #[inline]
    pub fn layer_wk(&self, layer: usize) -> &[f32] {
        read_slice(&self.buffer, self.layers[layer].wk)
    }

    /// Attention V weight for `layer`.
    #[inline]
    pub fn layer_wv(&self, layer: usize) -> &[f32] {
        read_slice(&self.buffer, self.layers[layer].wv)
    }

    /// Attention output projection weight for `layer`.
    #[inline]
    pub fn layer_wo(&self, layer: usize) -> &[f32] {
        read_slice(&self.buffer, self.layers[layer].wo)
    }

    /// MLP up-projection weight for `layer`.
    #[inline]
    pub fn layer_w1(&self, layer: usize) -> &[f32] {
        read_slice(&self.buffer, self.layers[layer].w1)
    }

    /// MLP down-projection weight for `layer`.
    #[inline]
    pub fn layer_w2(&self, layer: usize) -> &[f32] {
        read_slice(&self.buffer, self.layers[layer].w2)
    }

    // ── Metadata ────────────────────────────────────────────────

    /// Number of transformer layers.
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total buffer size in `f32` elements.
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Total buffer size in bytes.
    #[inline]
    pub fn buffer_bytes(&self) -> usize {
        self.buffer.len() * size_of::<f32>()
    }
}

/// Load a ciot-format .bits ternary weight file.
///
/// Format (little-endian):
///   magic      8 bytes  b"CIOTBIT1"
///   rows       u32
///   cols       u32
///   blocks64   u32
///   row_scale  rows × f32
///   pos_bits   rows × blocks64 × u64
///   neg_bits   rows × blocks64 × u64
#[cfg(feature = "plasma_path")]
pub fn load_ternary_bits(path: &std::path::Path) -> std::io::Result<katgpt_core::TernaryWeights> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let file_len = f.metadata()?.len() as usize;
    let mut buf = Vec::with_capacity(file_len);
    f.read_to_end(&mut buf)?;

    if buf.len() < 20 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file too small for header",
        ));
    }

    // Magic
    if &buf[0..8] != b"CIOTBIT1" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid magic",
        ));
    }

    let rows = u32::from_le_bytes(
        buf[8..12]
            .try_into()
            .expect("header length validated above"),
    ) as usize;
    let cols = u32::from_le_bytes(
        buf[12..16]
            .try_into()
            .expect("header length validated above"),
    ) as usize;
    let blocks64 = u32::from_le_bytes(
        buf[16..20]
            .try_into()
            .expect("header length validated above"),
    ) as usize;

    let expected_blocks = cols.div_ceil(64);
    if blocks64 != expected_blocks {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("blocks64 mismatch: header={blocks64}, expected={expected_blocks}"),
        ));
    }

    let scale_bytes = rows * 4;
    let pos_bytes = rows * blocks64 * 8;
    let neg_bytes = rows * blocks64 * 8;
    let expected_len = 20 + scale_bytes + pos_bytes + neg_bytes;
    if buf.len() < expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file truncated",
        ));
    }

    let mut off = 20;

    // Bulk copy: the buffer is native little-endian, so we can reinterpret directly.
    let mut row_scale = vec![0.0f32; rows];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf[off..].as_ptr(),
            row_scale.as_mut_ptr() as *mut u8,
            rows * 4,
        );
    }
    off += rows * 4;

    let pos_count = rows * blocks64;
    let mut pos_bits = vec![0u64; pos_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf[off..].as_ptr(),
            pos_bits.as_mut_ptr() as *mut u8,
            pos_count * 8,
        );
    }
    off += pos_count * 8;

    let mut neg_bits = vec![0u64; pos_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf[off..].as_ptr(),
            neg_bits.as_mut_ptr() as *mut u8,
            pos_count * 8,
        );
    }

    Ok(katgpt_core::TernaryWeights {
        rows,
        cols,
        blocks64,
        pos_bits,
        neg_bits,
        row_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::types::{Config, Rng};

    fn make_weights() -> (Config, TransformerWeights) {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (config, weights)
    }

    #[test]
    fn test_contiguous_buffer_not_empty() {
        let (config, weights) = make_weights();
        let cw = ContiguousWeights::from_weights(&weights);
        assert!(cw.buffer_len() > 0, "buffer must be non-empty");
        assert_eq!(cw.n_layers(), config.n_layer, "layer count mismatch");
    }

    #[test]
    fn test_roundtrip_wte() {
        let (_, weights) = make_weights();
        let cw = ContiguousWeights::from_weights(&weights);
        let packed = cw.wte();
        assert_eq!(packed.len(), weights.wte.len(), "wte length mismatch");
        (0..packed.len()).for_each(|i| {
            assert!(
                (packed[i] - weights.wte[i]).abs() < 1e-6,
                "wte mismatch at index {i}"
            );
        });
    }

    #[test]
    fn test_roundtrip_wpe() {
        let (_, weights) = make_weights();
        let cw = ContiguousWeights::from_weights(&weights);
        let packed = cw.wpe();
        assert_eq!(packed.len(), weights.wpe.len(), "wpe length mismatch");
        (0..packed.len()).for_each(|i| {
            assert!(
                (packed[i] - weights.wpe[i]).abs() < 1e-6,
                "wpe mismatch at index {i}"
            );
        });
    }

    #[test]
    fn test_roundtrip_lm_head() {
        let (_, weights) = make_weights();
        let cw = ContiguousWeights::from_weights(&weights);
        let packed = cw.lm_head();
        assert_eq!(
            packed.len(),
            weights.lm_head.len(),
            "lm_head length mismatch"
        );
        (0..packed.len()).for_each(|i| {
            assert!(
                (packed[i] - weights.lm_head[i]).abs() < 1e-6,
                "lm_head mismatch at index {i}"
            );
        });
    }

    #[test]
    fn test_roundtrip_layer_wq() {
        let (config, weights) = make_weights();
        let cw = ContiguousWeights::from_weights(&weights);
        for layer_idx in 0..config.n_layer {
            let orig = &weights.layers[layer_idx].attn_wq;
            let packed = cw.layer_wq(layer_idx);
            assert_eq!(
                orig.len(),
                packed.len(),
                "wq length mismatch at layer {layer_idx}"
            );
            for i in 0..orig.len() {
                assert!(
                    (orig[i] - packed[i]).abs() < 1e-6,
                    "wq mismatch at layer {layer_idx} index {i}"
                );
            }
        }
    }

    #[test]
    fn test_roundtrip_all_layer_weights() {
        let (config, weights) = make_weights();
        let cw = ContiguousWeights::from_weights(&weights);
        for layer_idx in 0..config.n_layer {
            let layer = &weights.layers[layer_idx];
            for (label, orig, packed) in [
                ("wk", layer.attn_wk.as_slice(), cw.layer_wk(layer_idx)),
                ("wv", layer.attn_wv.as_slice(), cw.layer_wv(layer_idx)),
                ("wo", layer.attn_wo.as_slice(), cw.layer_wo(layer_idx)),
                ("w1", layer.mlp_w1.as_slice(), cw.layer_w1(layer_idx)),
                ("w2", layer.mlp_w2.as_slice(), cw.layer_w2(layer_idx)),
            ] {
                assert_eq!(
                    orig.len(),
                    packed.len(),
                    "{label} length mismatch at layer {layer_idx}"
                );
                for i in 0..orig.len() {
                    assert!(
                        (orig[i] - packed[i]).abs() < 1e-6,
                        "{label} mismatch at layer {layer_idx} index {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_alignment_64byte_boundaries() {
        let (_, weights) = make_weights();
        let cw = ContiguousWeights::from_weights(&weights);
        // wte starts at offset 0 (already aligned)
        assert_eq!(cw.wte.offset, 0, "wte should start at 0");
        // Every global weight and every layer weight must be 16-f32 aligned (64 bytes)
        assert_eq!(cw.wpe.offset % ALIGN_F32, 0, "wpe not aligned");
        assert_eq!(cw.lm_head.offset % ALIGN_F32, 0, "lm_head not aligned");
        for (i, layer) in cw.layers.iter().enumerate() {
            assert_eq!(layer.wq.offset % ALIGN_F32, 0, "layer {i} wq not aligned");
            assert_eq!(layer.wk.offset % ALIGN_F32, 0, "layer {i} wk not aligned");
            assert_eq!(layer.wv.offset % ALIGN_F32, 0, "layer {i} wv not aligned");
            assert_eq!(layer.wo.offset % ALIGN_F32, 0, "layer {i} wo not aligned");
            assert_eq!(layer.w1.offset % ALIGN_F32, 0, "layer {i} w1 not aligned");
            assert_eq!(layer.w2.offset % ALIGN_F32, 0, "layer {i} w2 not aligned");
        }
    }

    #[test]
    fn test_buffer_bytes_matches() {
        let (_, weights) = make_weights();
        let cw = ContiguousWeights::from_weights(&weights);
        assert_eq!(
            cw.buffer_bytes(),
            cw.buffer_len() * size_of::<f32>(),
            "byte count mismatch"
        );
    }
}
