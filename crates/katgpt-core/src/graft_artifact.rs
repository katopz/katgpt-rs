//! `grafts` artifact envelope — digest-pinned sidecar serialization + the
//! monotonic mount (issue 988 T5).
//!
//! [`GraftArtifact`] is the freeze/thaw envelope for a fitted graft: a
//! versioned canonical encoding of the graft content, pinned by a BLAKE3
//! digest, stamped with a caller-supplied monotonic epoch. [`GraftSlot`]
//! is the mount seam: advance/idempotent, fork + downgrade REFUSED loud —
//! the hot-swap envelope law (`LoraPair` freeze/thaw family, the
//! `LaneSlot` monotonic-apply shape), so a graft can ride a registry
//! later without re-litigating the envelope.
//!
//! Canonical-encoding laws:
//! - **Content only.** The epoch is lineage, not content: it lives on the
//!   artifact, never inside the digest. `graft_digest` = BLAKE3 over the
//!   canonical bytes; the mount compares epochs first, digests second.
//! - **Weights are bits.** Every `f32` is encoded as its raw bit pattern
//!   (fixed-width LE) — the canonical form pins BIT-IDENTICAL weights
//!   (the Issue-940 determinism law; `-0.0` and `0.0` are different
//!   content under this pin, by intent).
//! - **Order is semantic where it is semantic.** Reads encode in order
//!   (a read's INDEX is its rule-table input id); segments encode in
//!   order (the Issue-940 sequential fold order — reordering changes the
//!   accumulation and can change a quantized value); writes encode in
//!   order (two writes on one cell accumulate biases in write order).
//!   Sorted-by-construction fields (table rules by cell, rule entries by
//!   key, write entries by value) encode sorted, and the dead subset
//!   slots encode zeroed — the encoding is a pure function of the
//!   SEMANTIC content, never of incidental memory.
//! - **Decode validates, never trusts.** Magic + version refuse loud;
//!   every element rebuilds through its validating constructor
//!   ([`CellRule::new`], [`GraftTable::from_rules`], [`GraftWrite::new`],
//!   [`Graft::new`]) so every structural invariant re-asserts on the way
//!   in; non-finite weights/biases refuse (a NaN probe is a malformed
//!   artifact, not a pinned quirk); a trailing byte refuses; and the
//!   decoded graft re-encodes to bytes that MUST equal the input — any
//!   non-canonical ordering refuses [`GraftError::NonCanonical`].
//!   ⚠ `Graft::from_canonical` alone does NOT detect body corruption (a
//!   flipped weight bit re-encodes to the same corrupted bytes) — the
//!   digest on [`GraftArtifact`] is the corruption boundary; that is
//!   what "digest-pinned" means.
//! - **Posture is not content.** `Graft::disarm()` is operational state:
//!   a minted artifact is always the armed content posture, and a decode
//!   boots armed. Disarm after mount, never through the artifact.
//!
//! Fitting inputs (examples, labels) are the consumer's lane and are NOT
//! serialized — the artifact carries the fitted/compiled graft only.
//! The apply hot path is untouched by this module (still zero-alloc,
//! fixture-pinned by `tests/graft_g4_alloc.rs`); the artifact lane
//! allocates freely — it is a sidecar, not a tick path.
//!
//! Mount-site deployment law (issue 998 T3): selectivity is the MOUNT
//! SITE's contract, never the artifact's. A [`GraftSlot`] may mount a
//! graft only inside a context the artifact was fitted and validated
//! for (the T2b cell's arithmetic-lane posture) — mounting a
//! task-fitted graft onto out-of-task traffic repeats the measured
//! G3(b) failure (armed-on-base-right: mean KL(top-40) 2.71 nats, 64.6%
//! top-1 flips at write delta 8.0) no matter how clean the artifact
//! is. The envelope verifies CONTENT (digest + monotonic epoch); it
//! cannot verify CONTEXT.
//!
//! OPT-IN behind the same `grafts` feature; compiles to nothing without
//! it.

use crate::graft::{
    CellRule, Graft, GraftError, GraftRead, GraftSwitch, GraftTable, GraftWrite, ReadSegment,
    WriteEntry, hex32,
};

/// The artifact + canonical encoding format version. Bumped ONLY by a
/// breaking encoding change; an unknown version refuses loud
/// ([`GraftError::Encoding`]) — never a best-effort parse.
pub const GRAFT_ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Canonical-form magic (`KGRFT` — katgpt graft).
const GRAFT_MAGIC: [u8; 5] = *b"KGRFT";

// ── The canonical byte codec ─────────────────────────────────────────────

struct Enc(Vec<u8>);

impl Enc {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u8v(&mut self, v: u8) {
        self.0.push(v);
    }
    /// f32 as its raw bit pattern — the canonical weight pin.
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
}

struct Dec<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], GraftError> {
        if n > self.b.len() - self.pos {
            return Err(GraftError::Encoding("truncated canonical graft"));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8v(&mut self) -> Result<u8, GraftError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, GraftError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32, GraftError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, GraftError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f32(&mut self) -> Result<f32, GraftError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn done(&self) -> Result<(), GraftError> {
        if self.pos != self.b.len() {
            return Err(GraftError::Encoding("trailing bytes after canonical graft"));
        }
        Ok(())
    }
}

/// Decode-time finite check: a non-finite probe coefficient is a
/// malformed artifact (fit paths already refuse non-finite solutions;
/// the artifact lane refuses to carry one too).
fn check_finite(v: f32, what: &'static str) -> Result<(), GraftError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(GraftError::Encoding(what))
    }
}

fn encode_segments(enc: &mut Enc, segments: &[ReadSegment]) {
    enc.u32(segments.len() as u32);
    for seg in segments {
        enc.u32(seg.layer);
        enc.u32(seg.weights.len() as u32);
        for &w in &seg.weights {
            enc.f32(w);
        }
    }
}

fn decode_segments(dec: &mut Dec) -> Result<Vec<ReadSegment>, GraftError> {
    let n = dec.u32()? as usize;
    let mut segments = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let layer = dec.u32()?;
        let wlen = dec.u32()? as usize;
        let mut weights = Vec::with_capacity(wlen.min(4096));
        for _ in 0..wlen {
            let w = dec.f32()?;
            check_finite(w, "canonical graft carries a non-finite weight")?;
            weights.push(w);
        }
        segments.push(ReadSegment { layer, weights });
    }
    Ok(segments)
}

fn decode_read(dec: &mut Dec) -> Result<GraftRead, GraftError> {
    let domain = dec.u16()?;
    let segments = decode_segments(dec)?;
    let bias = dec.f32()?;
    check_finite(bias, "canonical graft read carries a non-finite bias")?;
    GraftRead::new(segments, bias, domain)
}

fn decode_switch(dec: &mut Dec) -> Result<GraftSwitch, GraftError> {
    let segments = decode_segments(dec)?;
    let bias = dec.f32()?;
    check_finite(bias, "canonical graft switch carries a non-finite bias")?;
    let threshold = dec.f32()?;
    // Threshold validation (range, NaN refusal) rides GraftSwitch::new.
    GraftSwitch::new(segments, bias, threshold)
}

fn decode_rule(dec: &mut Dec) -> Result<CellRule, GraftError> {
    let cell = dec.u16()?;
    let subset_len = dec.u8v()?;
    let mut subset = [0u16; 3];
    for slot in &mut subset {
        *slot = dec.u16()?;
    }
    let live = subset_len as usize;
    if live > 3 {
        return Err(GraftError::Encoding("canonical rule subset_len > 3"));
    }
    let n = dec.u32()? as usize;
    let mut entries = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let key = dec.u64()?;
        let value = dec.u16()?;
        entries.push((key, value));
    }
    // Sort + duplicate refusal ride CellRule::new; the dead subset slots
    // it zero-fills make the reconstruction canonical.
    CellRule::new(cell, &subset[..live], entries)
}

fn decode_write(dec: &mut Dec) -> Result<GraftWrite, GraftError> {
    let cell = dec.u16()?;
    let n = dec.u32()? as usize;
    let mut entries = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let value = dec.u16()?;
        let nb = dec.u32()? as usize;
        let mut biases = Vec::with_capacity(nb.min(1024));
        for _ in 0..nb {
            let token = dec.u32()?;
            let bias = dec.f32()?;
            check_finite(bias, "canonical graft write carries a non-finite bias")?;
            biases.push((token, bias));
        }
        entries.push(WriteEntry { value, biases });
    }
    // Sort by value + duplicate/empty-bias refusal ride GraftWrite::new.
    GraftWrite::new(cell, entries)
}

// ── Graft: canonical form + strict inverse ───────────────────────────────

impl Graft {
    /// The canonical deterministic encoding of the graft CONTENT (no
    /// epoch — lineage lives on [`GraftArtifact`]). Versioned
    /// ([`GRAFT_ARTIFACT_FORMAT_VERSION`]), fixed-width little-endian,
    /// weights as raw bit patterns; two mints of identical content give
    /// byte-identical canonical bytes. The disarm posture is operational
    /// state and never encoded.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut enc = Enc::new();
        enc.raw(&GRAFT_MAGIC);
        enc.u32(GRAFT_ARTIFACT_FORMAT_VERSION);

        enc.u32(self.reads.len() as u32);
        for read in &self.reads {
            enc.u16(read.domain);
            encode_segments(&mut enc, &read.segments);
            enc.f32(read.bias);
        }

        enc.u32(self.switches.len() as u32);
        for sw in &self.switches {
            encode_segments(&mut enc, &sw.segments);
            enc.f32(sw.bias);
            enc.f32(sw.threshold);
        }

        enc.u32(self.table.rules.len() as u32);
        for rule in &self.table.rules {
            enc.u16(rule.cell);
            enc.u8v(rule.subset_len);
            for slot in rule.subset {
                // Dead slots encode zeroed (CellRule::new's own
                // zero-fill): the encoding is a function of semantic
                // content, not incidental memory.
                enc.u16(slot);
            }
            enc.u32(rule.entries.len() as u32);
            for &(key, value) in &rule.entries {
                enc.u64(key);
                enc.u16(value);
            }
        }

        enc.u32(self.writes.len() as u32);
        for write in &self.writes {
            enc.u16(write.cell);
            enc.u32(write.entries.len() as u32);
            for entry in &write.entries {
                enc.u16(entry.value);
                enc.u32(entry.biases.len() as u32);
                for &(token, bias) in &entry.biases {
                    enc.u32(token);
                    enc.f32(bias);
                }
            }
        }
        enc.0
    }

    /// The strict canonical inverse: refuses a wrong magic or version,
    /// any truncation or trailing byte, a non-finite payload, and every
    /// structural violation the constructors assert (unknown table ids,
    /// duplicate entries, empty surfaces). The decoded graft re-encodes
    /// to bytes that must equal the input — non-canonical orderings
    /// refuse [`GraftError::NonCanonical`].
    ///
    /// ⚠ Corruption boundary: a flipped weight bit re-encodes to the
    /// same (corrupted) bytes and passes HERE — the digest on
    /// [`GraftArtifact`] is the integrity check; always decode through
    /// [`GraftArtifact::decode`] in production.
    ///
    /// # Errors
    /// [`GraftError::Encoding`] / [`GraftError::NonCanonical`] on any
    /// malformed input; the constructors' own errors pass through.
    pub fn from_canonical(bytes: &[u8]) -> Result<Self, GraftError> {
        let mut dec = Dec::new(bytes);
        if dec.take(GRAFT_MAGIC.len())? != GRAFT_MAGIC {
            return Err(GraftError::Encoding("bad canonical graft magic"));
        }
        let version = dec.u32()?;
        if version != GRAFT_ARTIFACT_FORMAT_VERSION {
            return Err(GraftError::Encoding(
                "unsupported graft canonical format version (this build reads 1)",
            ));
        }

        let n_reads = dec.u32()? as usize;
        let mut reads = Vec::with_capacity(n_reads.min(1024));
        for _ in 0..n_reads {
            reads.push(decode_read(&mut dec)?);
        }

        let n_switches = dec.u32()? as usize;
        let mut switches = Vec::with_capacity(n_switches.min(1024));
        for _ in 0..n_switches {
            switches.push(decode_switch(&mut dec)?);
        }

        let n_rules = dec.u32()? as usize;
        let mut rules = Vec::with_capacity(n_rules.min(1024));
        for _ in 0..n_rules {
            rules.push(decode_rule(&mut dec)?);
        }
        // Rule sorting + duplicate-cell refusal ride from_rules.
        let table = GraftTable::from_rules(rules)?;

        let n_writes = dec.u32()? as usize;
        let mut writes = Vec::with_capacity(n_writes.min(1024));
        for _ in 0..n_writes {
            writes.push(decode_write(&mut dec)?);
        }
        dec.done()?;

        let graft = Graft::new(reads, table, writes, switches)?;
        // Canonicality: the reconstruction must re-encode to exactly the
        // input bytes. Catches unsorted sorted-by-construction fields
        // and any other non-canonical ordering.
        if graft.canonical_bytes().as_slice() != bytes {
            return Err(GraftError::NonCanonical);
        }
        Ok(graft)
    }
}

// ── The artifact envelope ────────────────────────────────────────────────

/// A digest-pinned, epoch-stamped graft sidecar (issue 988 T5): the
/// freeze/thaw envelope a hot-swap registry mounts without re-litigating
/// the shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraftArtifact {
    /// The encoding format version (always
    /// [`GRAFT_ARTIFACT_FORMAT_VERSION`] at mint; decode refuses any
    /// other value loud).
    pub format_version: u32,
    /// The lineage epoch — caller-supplied at mint (a registry counter
    /// or clock; this crate reads no wall clock). The mount law's
    /// ordering key.
    pub epoch: u64,
    /// BLAKE3 over the canonical content bytes. The corruption boundary:
    /// any post-mint tamper trips [`GraftError::DigestMismatch`].
    pub graft_digest: [u8; 32],
    /// The canonical content encoding ([`Graft::canonical_bytes`]).
    pub canonical: Vec<u8>,
}

impl GraftArtifact {
    /// Stamp a graft into a versioned, digest-pinned artifact. The
    /// disarm posture is NOT serialized — `graft` is encoded as the
    /// armed content posture regardless of its runtime state.
    #[must_use]
    pub fn mint(graft: &Graft, epoch: u64) -> Self {
        let canonical = graft.canonical_bytes();
        let graft_digest = blake3::hash(&canonical).into();
        Self {
            format_version: GRAFT_ARTIFACT_FORMAT_VERSION,
            epoch,
            graft_digest,
            canonical,
        }
    }

    /// The content digest, lowercase hex (logs, issue records, wire).
    #[must_use]
    pub fn graft_digest_hex(&self) -> String {
        hex32(&self.graft_digest)
    }

    /// Verify the stored digest against BLAKE3 over the canonical
    /// bytes — the corruption check.
    ///
    /// # Errors
    /// [`GraftError::DigestMismatch`] when the canonical bytes do not
    /// hash to [`Self::graft_digest`].
    pub fn verify_digest(&self) -> Result<(), GraftError> {
        let computed: [u8; 32] = blake3::hash(&self.canonical).into();
        if computed == self.graft_digest {
            Ok(())
        } else {
            Err(GraftError::DigestMismatch {
                stored: self.graft_digest,
                computed,
            })
        }
    }

    /// Full verification + reconstruction: format version, digest, then
    /// [`Graft::from_canonical`] (constructor validation + canonicality).
    ///
    /// # Errors
    /// [`GraftError::Encoding`] on a wrong format version,
    /// [`GraftError::DigestMismatch`] on tampered content, and every
    /// parse/canonicality error from [`Graft::from_canonical`].
    pub fn decode(&self) -> Result<Graft, GraftError> {
        if self.format_version != GRAFT_ARTIFACT_FORMAT_VERSION {
            return Err(GraftError::Encoding(
                "unsupported graft artifact format version (this build reads 1)",
            ));
        }
        self.verify_digest()?;
        Graft::from_canonical(&self.canonical)
    }
}

// ── The monotonic mount ──────────────────────────────────────────────────

/// What one [`GraftSlot::install`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraftInstall {
    /// A new artifact (empty slot) or a strict epoch advance took the
    /// slot; the mounted graft is the incoming one.
    Advanced,
    /// Same epoch AND same content digest as the mounted artifact —
    /// byte-equivalent content (digest equality ⇒ identical canonical
    /// bytes), already fully verified at its own install. The mount is
    /// kept; no re-decode (the digest subsumes the encoding format —
    /// same bytes ⇒ same version bytes).
    Idempotent,
}

/// One mounted graft with its lineage (epoch + content digest), kept
/// together so the digest can never drift from the graft it names.
#[derive(Debug, Clone)]
struct Mounted {
    epoch: u64,
    digest: [u8; 32],
    graft: Graft,
}

/// The monotonic graft mount (issue 988 T5): the freeze/thaw envelope
/// law at the hot-swap seam —
///
/// - **empty slot** → install ([`GraftInstall::Advanced`])
/// - **epoch advance** (incoming > mounted) → install
/// - **same epoch + same digest** → [`GraftInstall::Idempotent`]
/// - **same epoch + different digest** → [`GraftError::Fork`] — a fork
///   of the lineage, refused loud
/// - **lower epoch** → [`GraftError::Downgrade`] — refused loud
///
/// Every refusal leaves the mounted graft untouched. The digest is
/// verified BEFORE any lineage comparison (a corrupt artifact reports
/// corruption, not a lineage verdict), and a full decode validates the
/// content before the slot advances — a slot never mounts unverified
/// bytes.
#[derive(Debug, Clone, Default)]
pub struct GraftSlot {
    mounted: Option<Mounted>,
}

impl GraftSlot {
    /// An empty slot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mount `artifact` under the monotonic law above.
    ///
    /// # Errors
    /// [`GraftError::DigestMismatch`] (corrupt artifact),
    /// [`GraftError::Fork`] / [`GraftError::Downgrade`] (lineage
    /// refusals), and every decode error ([`GraftError::Encoding`] /
    /// [`GraftError::NonCanonical`] / constructor errors). The mounted
    /// graft is untouched by any error.
    pub fn install(&mut self, artifact: &GraftArtifact) -> Result<GraftInstall, GraftError> {
        // Integrity BEFORE lineage: corruption is the stronger,
        // unambiguous verdict.
        artifact.verify_digest()?;
        if let Some(m) = &self.mounted {
            if artifact.epoch == m.epoch {
                if artifact.graft_digest == m.digest {
                    return Ok(GraftInstall::Idempotent);
                }
                return Err(GraftError::Fork {
                    mounted_epoch: m.epoch,
                    incoming_digest: artifact.graft_digest,
                });
            }
            if artifact.epoch < m.epoch {
                return Err(GraftError::Downgrade {
                    mounted_epoch: m.epoch,
                    incoming_epoch: artifact.epoch,
                });
            }
        }
        let graft = artifact.decode()?;
        self.mounted = Some(Mounted {
            epoch: artifact.epoch,
            digest: artifact.graft_digest,
            graft,
        });
        Ok(GraftInstall::Advanced)
    }

    /// The mounted graft, if any.
    #[must_use]
    pub fn graft(&self) -> Option<&Graft> {
        self.mounted.as_ref().map(|m| &m.graft)
    }

    /// The mounted epoch, if any.
    #[must_use]
    pub fn epoch(&self) -> Option<u64> {
        self.mounted.as_ref().map(|m| m.epoch)
    }

    /// The mounted content digest, if any.
    #[must_use]
    pub fn digest(&self) -> Option<[u8; 32]> {
        self.mounted.as_ref().map(|m| m.digest)
    }
}

#[cfg(test)]
mod artifact_tests {
    use super::*;
    use crate::graft::{GraftExample, pack_key};

    // ── Test doubles (local; graft.rs's own tests keep theirs private) ──

    struct FixedReader {
        layers: std::collections::BTreeMap<u32, Vec<f32>>,
    }

    impl FixedReader {
        fn new(pairs: &[(u32, Vec<f32>)]) -> Self {
            Self {
                layers: pairs.iter().map(|&(l, ref v)| (l, v.clone())).collect(),
            }
        }
    }

    impl crate::graft::ActivationReader for FixedReader {
        fn read(&self, layer: u32) -> &[f32] {
            self.layers.get(&layer).map(|v| v.as_slice()).unwrap_or(&[])
        }
    }

    #[derive(Default)]
    struct VecWriter {
        biases: Vec<(u32, f32)>,
    }

    impl crate::graft::LogitWriter for VecWriter {
        fn add_bias(&mut self, token: u32, bias: f32) {
            self.biases.push((token, bias));
        }
    }

    // ── The fixed fixture graft (hand-authored constants; exercises
    //    every surface: 2 reads incl. a 2-segment one, 3 rules incl. a
    //    2-input and a reversed-subset one, 3 writes, 1 switch) ──

    fn fixture_graft() -> Graft {
        let read0 = GraftRead::new(
            vec![ReadSegment {
                layer: 3,
                weights: vec![0.5, -1.0, 2.0, 0.25],
            }],
            0.5,
            4,
        )
        .unwrap();
        let read1 = GraftRead::new(
            vec![
                ReadSegment {
                    layer: 5,
                    weights: vec![-0.5, 1.5, 0.0, -2.0],
                },
                ReadSegment {
                    layer: 7,
                    weights: vec![1.0],
                },
            ],
            -1.0,
            3,
        )
        .unwrap();
        let table = GraftTable::from_rules(vec![
            CellRule::new(0, &[0], vec![(0, 1), (1, 2), (2, 1)]).unwrap(),
            CellRule::new(
                1,
                &[0, 1],
                vec![
                    (pack_key(&[0, 0]), 0),
                    (pack_key(&[1, 1]), 2),
                    (pack_key(&[2, 0]), 1),
                ],
            )
            .unwrap(),
            CellRule::new(
                2,
                &[1, 0],
                vec![
                    (pack_key(&[0, 1]), 2),
                    (pack_key(&[0, 2]), 3),
                    (pack_key(&[2, 1]), 0),
                ],
            )
            .unwrap(),
        ])
        .unwrap();
        let writes = vec![
            GraftWrite::new(
                0,
                vec![
                    WriteEntry {
                        value: 1,
                        biases: vec![(9, 1.5), (11, -0.25)],
                    },
                    WriteEntry {
                        value: 2,
                        biases: vec![(9, -3.0)],
                    },
                ],
            )
            .unwrap(),
            GraftWrite::new(
                1,
                vec![WriteEntry {
                    value: 0,
                    biases: vec![(42, 0.125)],
                }],
            )
            .unwrap(),
            GraftWrite::new(
                2,
                vec![WriteEntry {
                    value: 3,
                    biases: vec![(7, 2.0)],
                }],
            )
            .unwrap(),
        ]
        .into_iter()
        .collect::<Vec<_>>();
        let switch = GraftSwitch::new(
            vec![ReadSegment {
                layer: 3,
                weights: vec![0.25, 0.5, -0.25, 1.0],
            }],
            0.1,
            0.4,
        )
        .unwrap();
        Graft::new(vec![read0, read1], table, writes, vec![switch]).unwrap()
    }

    fn fixture_reader() -> FixedReader {
        // read0 → 1.625 → 2; read1 → −1.0 → clamp 0; switch score
        // 1.475 → sigmoid ≈ 0.813 ≥ 0.4 → armed. All three cells decide:
        // cell0 key 2 → 1, cell1 key pack([2,0]) → 1, cell2 key
        // pack([0,2]) → 3. Fired biases: (9, 1.5), (11, −0.25), (7, 2.0).
        FixedReader::new(&[
            (3, vec![2.0, 1.0, 0.5, 0.5]),
            (5, vec![2.0, 1.0, 0.5, 0.5]),
            (7, vec![0.5]),
        ])
    }

    // ── The fixtures ──

    #[test]
    fn round_trip_preserves_semantics_and_bytes() {
        let graft = fixture_graft();
        let a = GraftArtifact::mint(&graft, 7);
        assert_eq!(a.format_version, GRAFT_ARTIFACT_FORMAT_VERSION);
        assert_eq!(a.epoch, 7);
        assert_eq!(a.canonical.len(), graft.canonical_bytes().len());

        // Deterministic mint: identical content + epoch ⇒ identical
        // artifact bytes (PartialEq covers digest + canonical).
        assert_eq!(a, GraftArtifact::mint(&graft, 7));

        let decoded = a.decode().unwrap();
        assert_eq!(decoded, graft);

        // The decoded graft re-mints to the identical artifact.
        assert_eq!(a, GraftArtifact::mint(&decoded, 7));
    }

    #[test]
    fn decoded_graft_behaves_identically() {
        let graft = fixture_graft();
        let decoded = GraftArtifact::mint(&graft, 1).decode().unwrap();

        let mut inp = [0u16; 2];
        let mut out = [0u16; 3];
        let mut w0 = VecWriter::default();
        let applied = decoded.apply(&fixture_reader(), &mut w0, &mut inp, &mut out);
        assert!(applied.fired);
        assert_eq!(applied.decided, 3);
        assert_eq!(w0.biases, vec![(9, 1.5), (11, -0.25), (7, 2.0)]);

        // Same answer as the in-memory graft (exact f32 equality — the
        // artifact round trip is bit-identical, not approximate).
        let mut w1 = VecWriter::default();
        let mut inp2 = [0u16; 2];
        let mut out2 = [0u16; 3];
        let applied1 = graft.apply(&fixture_reader(), &mut w1, &mut inp2, &mut out2);
        assert_eq!(applied1, applied);
        assert_eq!(w1.biases, w0.biases);
    }

    #[test]
    fn golden_digest_pin() {
        // The encoding-drift tripwire: ANY change to the canonical form
        // (field order, widths, magic, version) moves this digest. The
        // pin was captured at first run of this exact fixture (the T2a
        // convention); a legitimate format bump lands WITH a new pin and
        // a version bump, never by silently editing this row.
        const GOLDEN_DIGEST_HEX: &str =
            "a84025e7b6b10b9c0957b16f850f9fe36172cb853ddf1cafc28029393f10e708";
        let a = GraftArtifact::mint(&fixture_graft(), 7);
        assert_eq!(a.graft_digest_hex(), GOLDEN_DIGEST_HEX);
        // Self-describing header: magic + version ride the bytes.
        assert_eq!(&a.canonical[..5], b"KGRFT");
        assert_eq!(
            u32::from_le_bytes([
                a.canonical[5],
                a.canonical[6],
                a.canonical[7],
                a.canonical[8]
            ]),
            GRAFT_ARTIFACT_FORMAT_VERSION
        );
    }

    #[test]
    fn wrong_version_refuses_loud() {
        let mut bytes = fixture_graft().canonical_bytes();
        bytes[5..9].copy_from_slice(&2u32.to_le_bytes());
        let err = Graft::from_canonical(&bytes).unwrap_err();
        assert_eq!(
            err,
            GraftError::Encoding("unsupported graft canonical format version (this build reads 1)")
        );
        // And through the artifact envelope:
        let mut artifact = GraftArtifact::mint(&fixture_graft(), 3);
        artifact.format_version = 2;
        assert!(matches!(
            artifact.decode(),
            Err(GraftError::Encoding(
                "unsupported graft artifact format version (this build reads 1)"
            ))
        ));
    }

    #[test]
    fn bad_magic_refuses() {
        let mut bytes = fixture_graft().canonical_bytes();
        bytes[0] = b'X';
        assert_eq!(
            Graft::from_canonical(&bytes).unwrap_err(),
            GraftError::Encoding("bad canonical graft magic")
        );
    }

    #[test]
    fn tampered_body_refuses_digest() {
        let mut artifact = GraftArtifact::mint(&fixture_graft(), 5);
        let last = artifact.canonical.len() - 1;
        artifact.canonical[last] ^= 0x01;
        assert!(matches!(
            artifact.decode(),
            Err(GraftError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn non_canonical_entry_order_refuses() {
        // A hand-built canonical blob whose write entries are in
        // DESCENDING value order: parse + reconstruction succeed (the
        // constructor sorts), the re-encode differs ⇒ NonCanonical.
        let mut b = Vec::new();
        b.extend_from_slice(b"KGRFT");
        b.extend_from_slice(&1u32.to_le_bytes());
        // reads: 1 read, domain 2, 1 segment (layer 1, 1 weight = 1.0),
        // bias 0.0
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // segments
        b.extend_from_slice(&1u32.to_le_bytes()); // layer
        b.extend_from_slice(&1u32.to_le_bytes()); // weights len
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        b.extend_from_slice(&0.0f32.to_bits().to_le_bytes()); // bias
        // switches: 0
        b.extend_from_slice(&0u32.to_le_bytes());
        // rules: 1 rule, cell 0, subset [0], 1 entry (0 → 1)
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes()); // cell
        b.push(1); // subset_len
        b.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // subset slots
        b.extend_from_slice(&1u32.to_le_bytes()); // entries
        b.extend_from_slice(&0u64.to_le_bytes()); // key
        b.extend_from_slice(&1u16.to_le_bytes()); // value
        // writes: 1 write, cell 0, entries DESCENDING (2 then 1)
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes()); // cell
        b.extend_from_slice(&2u32.to_le_bytes()); // entries
        for value in [2u16, 1u16] {
            b.extend_from_slice(&value.to_le_bytes());
            b.extend_from_slice(&1u32.to_le_bytes()); // biases
            b.extend_from_slice(&5u32.to_le_bytes()); // token
            b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        }
        assert!(matches!(
            Graft::from_canonical(&b),
            Err(GraftError::NonCanonical)
        ));
    }

    #[test]
    fn truncated_and_trailing_refuse() {
        let bytes = fixture_graft().canonical_bytes();
        assert_eq!(
            Graft::from_canonical(&bytes[..bytes.len() - 1]).unwrap_err(),
            GraftError::Encoding("truncated canonical graft")
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            Graft::from_canonical(&trailing).unwrap_err(),
            GraftError::Encoding("trailing bytes after canonical graft")
        );
    }

    #[test]
    fn non_finite_payload_refuses() {
        // A NaN weight rides a CONSISTENT digest (mint-side the fit
        // lanes refuse non-finite solutions; here a hand-built artifact
        // proves the decode lane refuses too — a NaN probe is malformed,
        // not a pinned quirk).
        let mut graft = fixture_graft();
        graft.reads[0].segments[0].weights[0] = f32::NAN;
        let canonical = graft.canonical_bytes();
        let artifact = GraftArtifact {
            format_version: GRAFT_ARTIFACT_FORMAT_VERSION,
            epoch: 1,
            graft_digest: blake3::hash(&canonical).into(),
            canonical,
        };
        assert!(matches!(
            artifact.decode(),
            Err(GraftError::Encoding(
                "canonical graft carries a non-finite weight"
            ))
        ));
    }

    #[test]
    fn disarmed_posture_is_not_serialized() {
        let graft = fixture_graft();
        let mut disarmed = graft.clone();
        disarmed.disarm();
        // Identical content ⇒ identical artifact (the operational
        // posture never rides the bytes).
        assert_eq!(
            GraftArtifact::mint(&graft, 3),
            GraftArtifact::mint(&disarmed, 3)
        );
        // And a decoded graft boots ARMED — it applies.
        let decoded = GraftArtifact::mint(&disarmed, 3).decode().unwrap();
        let mut inp = [0u16; 2];
        let mut out = [0u16; 3];
        let mut w = VecWriter::default();
        assert!(
            decoded
                .apply(&fixture_reader(), &mut w, &mut inp, &mut out)
                .fired
        );
        assert_eq!(w.biases.len(), 3);
    }

    #[test]
    fn scrambled_wiring_changes_the_digest() {
        // The artifact pins WIRING, not just weights: a scrambled graft
        // mints a different digest (the T2b control's artifact is a
        // different artifact, never the armed one in disguise).
        let graft = fixture_graft();
        let d0 = GraftArtifact::mint(&graft, 1).graft_digest;
        let d1 = GraftArtifact::mint(&graft.scrambled(0x5EED), 1).graft_digest;
        assert_ne!(d0, d1);
        // The scrambled graft round-trips too.
        let decoded = GraftArtifact::mint(&graft.scrambled(0x5EED), 1)
            .decode()
            .unwrap();
        assert_eq!(decoded, graft.scrambled(0x5EED));
    }

    #[test]
    fn slot_monotonic_laws() {
        let graft = fixture_graft();
        let a1 = GraftArtifact::mint(&graft, 10);
        let a2 = GraftArtifact::mint(&graft, 11);
        // A DIFFERENT graft at the same epoch is the fork case.
        let mut other = fixture_graft();
        other.reads[0].segments[0].weights[3] = 0.75;
        let fork = GraftArtifact::mint(&other, 10);

        let mut slot = GraftSlot::new();
        assert_eq!(slot.graft(), None);
        assert_eq!(slot.epoch(), None);

        // Empty slot: any epoch installs.
        assert_eq!(slot.install(&a1), Ok(GraftInstall::Advanced));
        assert_eq!(slot.epoch(), Some(10));
        assert_eq!(slot.digest(), Some(a1.graft_digest));
        assert_eq!(slot.graft(), Some(&graft));

        // Same epoch + same digest: idempotent, mount kept.
        assert_eq!(slot.install(&a1), Ok(GraftInstall::Idempotent));
        assert_eq!(slot.epoch(), Some(10));

        // Same epoch + different digest: fork, refused loud, mount
        // untouched.
        assert_eq!(
            slot.install(&fork),
            Err(GraftError::Fork {
                mounted_epoch: 10,
                incoming_digest: fork.graft_digest,
            })
        );
        assert_eq!(slot.epoch(), Some(10));
        assert_eq!(slot.graft(), Some(&graft));

        // Lower epoch: downgrade, refused loud, mount untouched.
        let old = GraftArtifact::mint(&graft, 9);
        assert_eq!(
            slot.install(&old),
            Err(GraftError::Downgrade {
                mounted_epoch: 10,
                incoming_epoch: 9,
            })
        );
        assert_eq!(slot.graft(), Some(&graft));

        // Epoch advance (same content): installs.
        assert_eq!(slot.install(&a2), Ok(GraftInstall::Advanced));
        assert_eq!(slot.epoch(), Some(11));
        assert_eq!(slot.graft(), Some(&graft));
    }

    #[test]
    fn slot_refuses_corrupt_artifact_and_stays_mounted() {
        let graft = fixture_graft();
        let mut corrupt = GraftArtifact::mint(&graft, 99);
        corrupt.canonical[10] ^= 0x80;

        let mut slot = GraftSlot::new();
        assert_eq!(
            slot.install(&GraftArtifact::mint(&graft, 1)),
            Ok(GraftInstall::Advanced)
        );
        // The corrupt artifact's epoch is HIGHER (would advance) but the
        // digest check fires first — corruption is the verdict, not
        // lineage.
        assert!(matches!(
            slot.install(&corrupt),
            Err(GraftError::DigestMismatch { .. })
        ));
        assert_eq!(slot.epoch(), Some(1));
        assert_eq!(slot.graft(), Some(&graft));
    }

    // Silence the unused-import lint when fixture helpers evolve.
    #[allow(dead_code)]
    fn _example_helper(_: GraftExample) {}
}
