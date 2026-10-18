//! `.kpt` single-file weight-archive POC (Issue 841 / Research 568 fusion 2).
//!
//! .cact-style **nameless positional layer-major** single-file archive over
//! packed ternary weight layers, with the **BLAKE3/Merkle integrity .cact
//! lacks**, and **atomic weight hot-swap = file replacement** (temp write +
//! fsync + rename — a reader either sees the whole old file or the whole new
//! one, never a torn mix).
//!
//! **POC grade (owner call, 2026-09-19):** example + tests prove the concept,
//! the perf envelope, and the fail-closed security posture (bad-injection
//! arms). NOT promoted, no production consumer — the format+engine lift stays
//! parked until a NeuronShard-Merkle consumer actually wants single-file
//! mmap hot-swap (Issue 841's original deferral condition). Opt-in feature
//! `kpt_archive`; zero new deps (blake3 + bytemuck already non-optional).
//!
//! # Wire format v1 (all integers little-endian)
//!
//! ```text
//! offset 0   magic "KPT1"                     [4]
//! 4          version: u32 = 1                 [4]
//! 8          layer_count: u32                 [4]
//! 12         flags: u32 = 0                   [4]
//! 16         reserved (zeros)                 [16]
//! 32         directory: layer_count × 64 B
//!              rows u32 · cols u32 · blocks64 u32 · pad u32
//!              payload_offset u64 · payload_len u64 (16-padded, hashing incl.)
//!              layer_hash [u8;32] (BLAKE3 of the stored payload region)
//! dir_end    payloads, LAYER-MAJOR, POSITIONAL, STRICTLY SEQUENTIAL:
//!              per layer: pos_bits u64[rows*blocks64] · neg_bits u64[same]
//!                         · row_scale f32[rows] · zero pad to 16
//! end-64     merkle_root [u8;32]  (pairwise BLAKE3 over recomputed layer
//!                                   hashes; odd leaf promoted unchanged)
//! end-32     archive_id  [u8;32]  (BLAKE3 of everything before it)
//! ```
//!
//! Nameless-positional discipline is VERIFIED, not assumed: layer i's
//! `payload_offset` must equal the exact sequential position — a spliced,
//! reordered or overlapping directory is rejected before any hash is read.
//!
//! # Security posture (the bad-injection arms live in the root test)
//!
//! Every path is fail-closed: structure first (magic/version/length/
//! offsets/overflow/sequencing, all checked arithmetic), then per-layer
//! hashes, then the Merkle root, then the archive id. A tampered archive is
//! refused before its weights are ever materialized, and a REFUSED atomic
//! swap leaves the current file on disk untouched (verify-before-rename).
//! `archive_id` is the REPLAY detector: an old-but-valid archive still
//! verifies structurally — freshness is the consumer comparing ids, which is
//! why [`KptArchive::archive_id`] is the value `swap_in_place` returns.
//!
//! # Endianness / zero-copy
//!
//! File bytes are canonical LE. [`KptArchive::layer_view`] is zero-copy via
//! alignment-checked `bytemuck` casts (a real mmap is page-aligned by
//! construction; an owned `Vec<u8>` read is allocator-aligned in practice but
//! not contractually — a misaligned buffer gets [`KptError::MisalignedBuffer`]
//! and the copying path [`KptArchive::to_ternary_weights`] still works).
//! Zero-copy views additionally require a little-endian host (every target
//! this workspace ships: x86_64, aarch64, wasm32).
//!
//! .kpt single-file weight-archive POC (Issue 841 / Research 568 fusion 2):
//! .cact-style nameless positional layer-major file × BLAKE3/Merkle integrity
//! × atomic file-replacement hot-swap (write temp + fsync + rename). POC
//! grade per the 2026-09-19 owner call: example + tests prove the concept,
//! the perf envelope and the fail-closed security posture; NOT promoted, no
//! production consumer — the format+engine lift stays parked until a
//! NeuronShard-Merkle consumer wants single-file mmap hot-swap.

use core::fmt;

use katgpt_types::TernaryWeights;

/// Format magic — `"KPT1"` (the trailing digit is the MAJOR version).
pub const KPT_MAGIC: [u8; 4] = *b"KPT1";
/// Wire version this module reads and writes.
pub const KPT_VERSION: u32 = 1;
/// Header size in bytes (magic + version + counts + flags + reserved).
pub const KPT_HEADER_SIZE: u64 = 32;
/// Directory entry size in bytes.
pub const KPT_DIR_ENTRY_SIZE: u64 = 64;
/// Trailer size: merkle_root + archive_id.
pub const KPT_TRAILER_SIZE: u64 = 64;
/// Payload alignment (offsets and padded lengths).
pub const KPT_ALIGN: u64 = 16;

/// Everything that can go wrong reading or swapping a `.kpt` archive.
#[derive(Debug)]
pub enum KptError {
    Io(std::io::Error),
    /// First four bytes are not `KPT1`.
    BadMagic([u8; 4]),
    /// Version field is not [`KPT_VERSION`].
    UnsupportedVersion(u32),
    /// File smaller than header + directory + trailer, or a declared region
    /// walks past end-of-file. `need`/`have` are byte counts.
    Truncated {
        need: u64,
        have: u64,
    },
    /// A directory offset/length pair overflows u64 arithmetic (wrap attack).
    OffsetOverflow {
        layer: usize,
    },
    /// A payload region is not 16-byte aligned.
    Misaligned {
        layer: usize,
        offset: u64,
    },
    /// Layer i's `payload_offset` is not the exact sequential position —
    /// spliced, reordered, gapped or overlapping layout.
    NotSequential {
        layer: usize,
        declared: u64,
        expected: u64,
    },
    /// Recomputed payload hash ≠ directory entry.
    LayerHashMismatch {
        layer: usize,
        declared: [u8; 32],
        recomputed: [u8; 32],
    },
    /// Recomputed Merkle root ≠ trailer.
    MerkleRootMismatch {
        declared: [u8; 32],
        recomputed: [u8; 32],
    },
    /// Recomputed archive id ≠ trailer (anything upstream moved).
    ArchiveIdMismatch {
        declared: [u8; 32],
        recomputed: [u8; 32],
    },
    /// Zero-copy view requested over a buffer that is not u64/f32-aligned.
    MisalignedBuffer,
    /// `build_archive` was handed an inconsistent layer (blocks64 ≠ ceil(cols/64)).
    BadLayerSpec {
        layer: usize,
        rows: usize,
        cols: usize,
        blocks64: usize,
    },
}

impl fmt::Display for KptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::BadMagic(m) => write!(f, "bad magic {m:?} (expected {:?})", KPT_MAGIC),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported version {v} (expected {KPT_VERSION})")
            }
            Self::Truncated { need, have } => {
                write!(f, "truncated: need {need} bytes, have {have}")
            }
            Self::OffsetOverflow { layer } => {
                write!(f, "offset arithmetic overflow at layer {layer}")
            }
            Self::Misaligned { layer, offset } => {
                write!(f, "layer {layer} offset {offset} not {KPT_ALIGN}-aligned")
            }
            Self::NotSequential {
                layer,
                declared,
                expected,
            } => {
                write!(
                    f,
                    "layer {layer} offset {declared} != sequential position {expected}"
                )
            }
            Self::LayerHashMismatch { layer, .. } => {
                write!(f, "layer {layer} payload hash mismatch")
            }
            Self::MerkleRootMismatch { .. } => write!(f, "merkle root mismatch"),
            Self::ArchiveIdMismatch { .. } => write!(f, "archive id mismatch"),
            Self::MisalignedBuffer => write!(
                f,
                "buffer not aligned for zero-copy views; use to_ternary_weights"
            ),
            Self::BadLayerSpec {
                layer,
                rows,
                cols,
                blocks64,
            } => write!(
                f,
                "layer {layer} spec rows={rows} cols={cols} blocks64={blocks64} inconsistent"
            ),
        }
    }
}

impl std::error::Error for KptError {}

impl From<std::io::Error> for KptError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// One verified directory entry (values as declared AND re-verified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KptLayerEntry {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,
    pub payload_offset: u64,
    pub payload_len: u64,
    pub layer_hash: [u8; 32],
}

/// A fully VERIFIED borrowed view over `.kpt` bytes.
///
/// Construction ([`KptArchive::from_bytes`]) performs the whole fail-closed
/// ladder — structure → layer hashes → Merkle root → archive id — so a
/// `KptArchive` existing at all is the integrity statement.
#[derive(Debug)]
pub struct KptArchive<'a> {
    bytes: &'a [u8],
    layers: Vec<KptLayerEntry>,
    merkle_root: [u8; 32],
    archive_id: [u8; 32],
}

fn u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn u64_le(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

fn read_u32(b: &[u8], off: usize) -> u32 {
    u32_le(&b[off..off + 4])
}

fn read_u64(b: &[u8], off: usize) -> u64 {
    u64_le(&b[off..off + 8])
}

fn round_up_16(v: u64) -> u64 {
    // File sizes are bounded far below u64::MAX; saturate (never wrap) — the
    // saturating value is still 16-aligned and any region that large is
    // refused by the length checks anyway.
    v.saturating_add(KPT_ALIGN - 1) & !(KPT_ALIGN - 1)
}

/// Stored payload length for one layer (16-padded).
fn stored_payload_len(rows: usize, blocks64: usize) -> Option<u64> {
    let bits_len = (rows as u64)
        .checked_mul(blocks64 as u64)?
        .checked_mul(8)?
        .checked_mul(2)?; // pos + neg planes
    let scale_len = (rows as u64).checked_mul(4)?;
    Some(round_up_16(bits_len.checked_add(scale_len)?))
}

/// BLAKE3 over `a || b` (the Merkle pairing step).
fn hash_pair(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(a);
    h.update(b);
    *h.finalize().as_bytes()
}

/// Pairwise BLAKE3 Merkle root over the layer hashes (odd leaf promoted
/// unchanged; the single-leaf root is that leaf's own hash — documented,
/// deterministic).
pub fn merkle_root_over(layer_hashes: &[[u8; 32]]) -> [u8; 32] {
    let mut level: Vec<[u8; 32]> = layer_hashes.to_vec();
    if level.is_empty() {
        return *blake3::hash(b"").as_bytes();
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            if i + 1 < level.len() {
                next.push(hash_pair(&level[i], &level[i + 1]));
            } else {
                // Odd leaf promoted unchanged.
                next.push(level[i]);
            }
            i += 2;
        }
        level = next;
    }
    level[0]
}

/// Zero-copy borrowed view over one layer's planes (alignment-checked).
#[derive(Debug, Clone, Copy)]
pub struct KptLayerView<'a> {
    pub pos_bits: &'a [u64],
    pub neg_bits: &'a [u64],
    pub row_scale: &'a [f32],
}

impl<'a> KptArchive<'a> {
    /// Parse + fully verify `bytes`. Fail-closed ladder:
    /// structure → per-layer hashes → Merkle → archive id.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, KptError> {
        let have = bytes.len() as u64;
        // ── Structure ────────────────────────────────────────────────────
        if have < KPT_HEADER_SIZE + KPT_TRAILER_SIZE {
            return Err(KptError::Truncated {
                need: KPT_HEADER_SIZE + KPT_TRAILER_SIZE,
                have,
            });
        }
        let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != KPT_MAGIC {
            return Err(KptError::BadMagic(magic));
        }
        let version = read_u32(bytes, 4);
        if version != KPT_VERSION {
            return Err(KptError::UnsupportedVersion(version));
        }
        let layer_count = read_u32(bytes, 8) as usize;
        let dir_end = KPT_HEADER_SIZE
            .checked_add(
                (layer_count as u64)
                    .checked_mul(KPT_DIR_ENTRY_SIZE)
                    .ok_or(KptError::OffsetOverflow { layer: 0 })?,
            )
            .ok_or(KptError::OffsetOverflow { layer: 0 })?;
        // Payloads sit between dir_end and the trailer; every declared region
        // must land inside that span.
        let payload_end = have - KPT_TRAILER_SIZE;
        if dir_end > payload_end {
            return Err(KptError::Truncated {
                need: dir_end + KPT_TRAILER_SIZE,
                have,
            });
        }

        let mut layers: Vec<KptLayerEntry> = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            let e = KPT_HEADER_SIZE as usize + i * KPT_DIR_ENTRY_SIZE as usize;
            let rows = read_u32(bytes, e) as usize;
            let cols = read_u32(bytes, e + 4) as usize;
            let blocks64 = read_u32(bytes, e + 8) as usize;
            let payload_offset = read_u64(bytes, e + 16);
            let payload_len = read_u64(bytes, e + 24);
            let mut layer_hash = [0u8; 32];
            layer_hash.copy_from_slice(&bytes[e + 32..e + 64]);

            // blocks64 must match the dims it claims (the packing invariant).
            if blocks64 != cols.div_ceil(64) {
                return Err(KptError::BadLayerSpec {
                    layer: i,
                    rows,
                    cols,
                    blocks64,
                });
            }
            // Nameless-positional discipline: EXACT sequential layout.
            let expected_offset = match layers.last() {
                None => dir_end,
                Some(p) => round_up_16(
                    p.payload_offset
                        .checked_add(p.payload_len)
                        .ok_or(KptError::OffsetOverflow { layer: i })?,
                ),
            };
            if payload_offset != expected_offset {
                return Err(KptError::NotSequential {
                    layer: i,
                    declared: payload_offset,
                    expected: expected_offset,
                });
            }
            if !payload_offset.is_multiple_of(KPT_ALIGN) {
                return Err(KptError::Misaligned {
                    layer: i,
                    offset: payload_offset,
                });
            }
            let region_end = payload_offset
                .checked_add(payload_len)
                .ok_or(KptError::OffsetOverflow { layer: i })?;
            if region_end > payload_end {
                return Err(KptError::Truncated {
                    need: region_end + KPT_TRAILER_SIZE,
                    have,
                });
            }
            // Declared length must match the dims (padded form).
            let want_len =
                stored_payload_len(rows, blocks64).ok_or(KptError::OffsetOverflow { layer: i })?;
            if payload_len != want_len {
                return Err(KptError::BadLayerSpec {
                    layer: i,
                    rows,
                    cols,
                    blocks64,
                });
            }
            layers.push(KptLayerEntry {
                rows,
                cols,
                blocks64,
                payload_offset,
                payload_len,
                layer_hash,
            });
        }
        // The last layer must end exactly at the trailer — no dangling bytes.
        if let Some(last) = layers.last() {
            let end = round_up_16(last.payload_offset + last.payload_len);
            if end != payload_end {
                return Err(KptError::NotSequential {
                    layer: layers.len() - 1,
                    declared: end,
                    expected: payload_end,
                });
            }
        }

        // ── Per-layer hashes ─────────────────────────────────────────────
        let mut recomputed = Vec::with_capacity(layers.len());
        for (i, l) in layers.iter().enumerate() {
            let region =
                &bytes[l.payload_offset as usize..(l.payload_offset + l.payload_len) as usize];
            let h = *blake3::hash(region).as_bytes();
            if h != l.layer_hash {
                return Err(KptError::LayerHashMismatch {
                    layer: i,
                    declared: l.layer_hash,
                    recomputed: h,
                });
            }
            recomputed.push(h);
        }

        // ── Merkle root ──────────────────────────────────────────────────
        let merkle_root: [u8; 32] = bytes[(bytes.len() - 64)..(bytes.len() - 32)]
            .try_into()
            .unwrap();
        let computed_root = merkle_root_over(&recomputed);
        if computed_root != merkle_root {
            return Err(KptError::MerkleRootMismatch {
                declared: merkle_root,
                recomputed: computed_root,
            });
        }

        // ── Archive id (covers everything before it, incl. the root) ─────
        let archive_id: [u8; 32] = bytes[(bytes.len() - 32)..].try_into().unwrap();
        let computed_id = *blake3::hash(&bytes[..bytes.len() - 32]).as_bytes();
        if computed_id != archive_id {
            return Err(KptError::ArchiveIdMismatch {
                declared: archive_id,
                recomputed: computed_id,
            });
        }

        Ok(Self {
            bytes,
            layers,
            merkle_root,
            archive_id,
        })
    }

    /// Verified directory (positional order).
    pub fn layers(&self) -> &[KptLayerEntry] {
        &self.layers
    }

    /// Trailer Merkle root (verified against recomputed layer hashes).
    pub fn merkle_root(&self) -> [u8; 32] {
        self.merkle_root
    }

    /// Archive id — BLAKE3 of everything except itself. The REPLAY detector:
    /// two different archives (e.g. two model versions) always differ here,
    /// while structural validity alone cannot tell old from new.
    pub fn archive_id(&self) -> [u8; 32] {
        self.archive_id
    }

    /// Zero-copy layer view (alignment-checked; little-endian hosts only).
    /// Misaligned buffer → [`KptError::MisalignedBuffer`]; use
    /// [`Self::to_ternary_weights`] for the always-working copying path.
    pub fn layer_view(&self, i: usize) -> Result<KptLayerView<'_>, KptError> {
        #[cfg(target_endian = "big")]
        {
            return Err(KptError::MisalignedBuffer);
        }
        #[cfg(target_endian = "little")]
        {
            let l = &self.layers[i];
            let region =
                &self.bytes[l.payload_offset as usize..(l.payload_offset + l.payload_len) as usize];
            let bits_len = l.rows * l.blocks64 * 8;
            let pos: &[u64] = bytemuck::try_cast_slice(&region[..bits_len])
                .map_err(|_| KptError::MisalignedBuffer)?;
            let neg: &[u64] = bytemuck::try_cast_slice(&region[bits_len..bits_len * 2])
                .map_err(|_| KptError::MisalignedBuffer)?;
            let scale: &[f32] =
                bytemuck::try_cast_slice(&region[bits_len * 2..bits_len * 2 + l.rows * 4])
                    .map_err(|_| KptError::MisalignedBuffer)?;
            Ok(KptLayerView {
                pos_bits: pos,
                neg_bits: neg,
                row_scale: scale,
            })
        }
    }

    /// Reconstruct layer `i` into an owned [`TernaryWeights`] (copying path —
    /// endianness-canonical, alignment-free; always available).
    pub fn to_ternary_weights(&self, i: usize) -> TernaryWeights {
        let l = &self.layers[i];
        let mut tw = TernaryWeights::new(l.rows, l.cols);
        let region =
            &self.bytes[l.payload_offset as usize..(l.payload_offset + l.payload_len) as usize];
        let bits_len = l.rows * l.blocks64 * 8;
        let (pos_chunks, _) = region[..bits_len].as_chunks::<8>();
        for (dst, src) in tw.pos_bits.iter_mut().zip(pos_chunks) {
            *dst = u64::from_le_bytes(*src);
        }
        let (neg_chunks, _) = region[bits_len..bits_len * 2].as_chunks::<8>();
        for (dst, src) in tw.neg_bits.iter_mut().zip(neg_chunks) {
            *dst = u64::from_le_bytes(*src);
        }
        let (scale_chunks, _) = region[bits_len * 2..].as_chunks::<4>();
        for (dst, src) in tw.row_scale.iter_mut().zip(scale_chunks) {
            *dst = f32::from_le_bytes(*src);
        }
        tw
    }
}

/// Deterministically serialize `layers` (positional order = layer order).
pub fn build_archive(layers: &[&TernaryWeights]) -> Result<Vec<u8>, KptError> {
    for (i, w) in layers.iter().enumerate() {
        if w.blocks64 != w.cols.div_ceil(64) {
            return Err(KptError::BadLayerSpec {
                layer: i,
                rows: w.rows,
                cols: w.cols,
                blocks64: w.blocks64,
            });
        }
    }
    let dir_end = KPT_HEADER_SIZE + (layers.len() as u64) * KPT_DIR_ENTRY_SIZE;
    let mut payload: Vec<u8> = Vec::new();
    /// Writer-side directory row (positional order = layer order).
    struct DirRow {
        rows: usize,
        cols: usize,
        blocks64: usize,
        offset: u64,
        len: u64,
        hash: [u8; 32],
    }
    let mut entries: Vec<DirRow> = Vec::with_capacity(layers.len());
    for w in layers {
        let offset = round_up_16(dir_end + payload.len() as u64);
        // Pad to the 16 boundary if a previous odd-length layer left a gap.
        let pad = offset - (dir_end + payload.len() as u64);
        payload.extend(std::iter::repeat_n(0u8, pad as usize));
        for v in &w.pos_bits {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for v in &w.neg_bits {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for v in &w.row_scale {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let logical_end = payload.len() as u64; // payload-relative
        let rel_start = offset - dir_end; // payload-relative
        let stored_len = round_up_16(logical_end - rel_start);
        payload.extend(std::iter::repeat_n(
            0u8,
            (stored_len - (logical_end - rel_start)) as usize,
        ));
        // Hash exactly this layer's stored region — payload currently ends
        // at this layer's padding, so `rel_start ..` is the region.
        let layer_hash = *blake3::hash(&payload[rel_start as usize..]).as_bytes();
        entries.push(DirRow {
            rows: w.rows,
            cols: w.cols,
            blocks64: w.blocks64,
            offset,
            len: stored_len,
            hash: layer_hash,
        });
    }

    let root = merkle_root_over(&entries.iter().map(|e| e.hash).collect::<Vec<_>>());
    let mut out = Vec::with_capacity((dir_end + payload.len() as u64 + KPT_TRAILER_SIZE) as usize);
    out.extend_from_slice(&KPT_MAGIC);
    out.extend_from_slice(&KPT_VERSION.to_le_bytes());
    out.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags
    out.extend(std::iter::repeat_n(0u8, 16)); // reserved
    for e in &entries {
        out.extend_from_slice(&(e.rows as u32).to_le_bytes());
        out.extend_from_slice(&(e.cols as u32).to_le_bytes());
        out.extend_from_slice(&(e.blocks64 as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // pad
        out.extend_from_slice(&e.offset.to_le_bytes());
        out.extend_from_slice(&e.len.to_le_bytes());
        out.extend_from_slice(&e.hash);
        // Entry = 4+4+4+4+8+8+32 = exactly KPT_DIR_ENTRY_SIZE (64) — no tail pad.
    }
    debug_assert_eq!(
        out.len() as u64,
        dir_end,
        "directory region must end exactly at dir_end"
    );
    let expected_total = entries.last().map(|e| e.offset + e.len).unwrap_or(dir_end);
    debug_assert_eq!(
        dir_end + payload.len() as u64,
        expected_total,
        "payload region must end exactly at the last entry's declared end"
    );
    out.extend_from_slice(&payload);
    out.extend_from_slice(&root);
    let id = *blake3::hash(&out).as_bytes();
    out.extend_from_slice(&id);
    Ok(out)
}

/// Atomic file replace: write sibling temp + fsync + rename. A reader holds
/// either the whole old file or the whole new one — never a torn mix.
pub fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> Result<(), KptError> {
    use std::io::Write;
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "archive.kpt".into());
    let tmp = dir.join(format!(".{name}.kpt.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read `path` into an owning buffer (the load half of the hot-swap). Call
/// [`OwnedKptArchive::verify`] for the borrowed, fully-verified view — the
/// split exists because a verified view borrows the bytes it verifies.
pub struct OwnedKptArchive {
    pub bytes: Vec<u8>,
}

impl OwnedKptArchive {
    pub fn load(path: &std::path::Path) -> Result<Self, KptError> {
        Ok(Self {
            bytes: std::fs::read(path)?,
        })
    }

    /// Parse + fully verify the owned bytes (the fail-closed ladder).
    pub fn verify(&self) -> Result<KptArchive<'_>, KptError> {
        KptArchive::from_bytes(&self.bytes)
    }
}

/// The full hot-swap: build → verify the NEW bytes in memory → atomic replace
/// → re-read + re-verify from disk. Returns the new [`KptArchive::archive_id`]
/// (the replay detector). A tampered payload is refused BEFORE the rename, so
/// the file on disk is untouched on any error path.
pub fn swap_in_place(
    path: &std::path::Path,
    layers: &[&TernaryWeights],
) -> Result<[u8; 32], KptError> {
    let bytes = build_archive(layers)?;
    // Verify-before-rename: never place an archive on disk that does not
    // round-trip through the reader's own fail-closed ladder.
    let checked = KptArchive::from_bytes(&bytes)?;
    write_atomic(path, &bytes)?;
    let on_disk = OwnedKptArchive::load(path)?;
    let archive = on_disk.verify()?;
    if archive.archive_id() != checked.archive_id() {
        return Err(KptError::ArchiveIdMismatch {
            declared: checked.archive_id(),
            recomputed: archive.archive_id(),
        });
    }
    Ok(archive.archive_id())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_layers() -> Vec<TernaryWeights> {
        let a = TernaryWeights::quantize_from_f32(
            &(0..8 * 12)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
            8,
            12,
        );
        let mut b = TernaryWeights::new(4, 130); // odd cols → padded tail block
        for r in 0..4 {
            for c in 0..130 {
                b.set(r, c, ((r * 131 + c * 17) % 3) as i8 - 1);
            }
            b.row_scale[r] = 1.0 + r as f32 * 0.25;
        }
        vec![a, b]
    }

    #[test]
    fn build_is_deterministic_and_round_trips() {
        let layers = demo_layers();
        let refs: Vec<&TernaryWeights> = layers.iter().collect();
        let one = build_archive(&refs).unwrap();
        let two = build_archive(&refs).unwrap();
        assert_eq!(one, two, "build must be byte-identical across calls");

        let archive = KptArchive::from_bytes(&one).unwrap();
        assert_eq!(archive.layers().len(), 2);
        for (i, want) in layers.iter().enumerate() {
            let got = archive.to_ternary_weights(i);
            assert_eq!(got.rows, want.rows);
            assert_eq!(got.cols, want.cols);
            assert_eq!(got.pos_bits, want.pos_bits, "layer {i} pos_bits round-trip");
            assert_eq!(got.neg_bits, want.neg_bits, "layer {i} neg_bits round-trip");
            assert_eq!(
                got.row_scale, want.row_scale,
                "layer {i} row_scale round-trip"
            );
        }
    }

    #[test]
    fn single_bit_flip_is_refused() {
        let layers = demo_layers();
        let refs: Vec<&TernaryWeights> = layers.iter().collect();
        let mut bytes = build_archive(&refs).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x01;
        let err = KptArchive::from_bytes(&bytes).unwrap_err();
        // The flip lands in payload or directory or trailer — wherever it
        // lands, SOME rung of the ladder must refuse it.
        assert!(
            matches!(
                err,
                KptError::LayerHashMismatch { .. }
                    | KptError::MerkleRootMismatch { .. }
                    | KptError::ArchiveIdMismatch { .. }
                    | KptError::BadLayerSpec { .. }
                    | KptError::NotSequential { .. }
            ),
            "tampered payload must be refused by a hash rung, got {err:?}"
        );
    }

    #[test]
    fn zero_copy_view_matches_copying_path() {
        let layers = demo_layers();
        let refs: Vec<&TernaryWeights> = layers.iter().collect();
        let bytes = build_archive(&refs).unwrap();
        let archive = KptArchive::from_bytes(&bytes).unwrap();
        for i in 0..layers.len() {
            if let Ok(view) = archive.layer_view(i) {
                let copied = archive.to_ternary_weights(i);
                assert_eq!(
                    view.pos_bits,
                    copied.pos_bits.as_slice(),
                    "layer {i} zero-copy pos == copied"
                );
                assert_eq!(
                    view.neg_bits,
                    copied.neg_bits.as_slice(),
                    "layer {i} zero-copy neg == copied"
                );
                assert_eq!(
                    view.row_scale,
                    copied.row_scale.as_slice(),
                    "layer {i} zero-copy scale == copied"
                );
            }
            // Misaligned or aligned, the copying path is the contract.
        }
    }
}
