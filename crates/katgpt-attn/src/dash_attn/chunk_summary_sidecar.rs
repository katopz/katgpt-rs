//! Issue 666 T2 — the reference-query sidecar (the trained `head_cls`
//! consumption lane).
//!
//! riir-train's ZHA chunk-KL trainer (Issue 666 T1/T3, `zha_chunk_kl` there)
//! mints trained reference queries; this module is the serving-side
//! consumption half: a versioned binary artifact + a BLAKE3 sidecar file,
//! loaded through an env-selected variant following the
//! `LAYA_WEIGHTS_VARIANT` pattern (riir-infer `q8_artifact::resolve_weights_variant`
//! — the same fail-loud contract: an env typo refuses, a missing artifact
//! refuses naming the writer, a digest mismatch refuses, and NOTHING is ever
//! auto-derived).
//!
//! This is the repo's freeze/thaw mutation class (the only weight mutation
//! allowed at runtime): swapping a frozen, versioned, BLAKE3-checked
//! snapshot. No training, no runtime minting — riir-train owns the bytes.
//!
//! # Artifact format (v1, little-endian, fixed layout)
//!
//! ```text
//! offset  size                  field
//! 0       8                     magic = b"KATCLSK1"
//! 8       4                     format_version (u32 LE, = 1)
//! 12      4                     n_refs R      (u32 LE, >= 1)
//! 16      4                     n_kv_head     (u32 LE, >= 1)
//! 20      4                     head_dim      (u32 LE, >= 1)
//! 24      4 * R * H * D         payload: f32 LE, ref-major [R][H][D]
//! ```
//!
//! Strict length: no trailing bytes (a future format bumps the version and
//! refuses v1 readers, never pads). The digest file beside the artifact
//! (`<artifact>.blake3`, plain hex text) carries the BLAKE3 of the artifact
//! bytes — the loader reads both and refuses a mismatch loud.
//!
//! # Env contract — `KATGPT_HEAD_CLS_VARIANT`
//!
//! | value | meaning |
//! |---|---|
//! | unset / empty / `zero` | the incumbent zero-init query (Ok(None) — the caller keeps its own default; sidecar-absent serving is byte-identical, the free G3 anchor) |
//! | `trained` | `<dir>/derived/head_cls_trained.bin` + `.blake3` sidecar, verified then parsed |
//! | anything else | refusal naming the expected set (an env typo must fail loud, never fall back) |
//!
//! # Warm-start law (T2's monotone-improving anchor)
//!
//! A sidecar whose payload is all-zero loads to [`ChunkSummaryQuery::is_zero_init`]
//! == true → the summarizer takes the mean-pool branch → serving is
//! byte-identical to the incumbent. The T3 trainer warm-starts R1 from
//! zero-init, so the trained R1 arm can only improve on mean-pool — the
//! consumer-side guarantee is that the zero state round-trips EXACTLY.

use std::fmt;
use std::path::{Path, PathBuf};

use super::chunk_summary::ChunkSummaryQuery;

/// Artifact magic — `KATCLSK1` = **KAT**gpt **C**hunk **L**earned **S**ummary
/// **K**eys, format v1.
pub const SIDECAR_MAGIC: [u8; 8] = *b"KATCLSK1";

/// Artifact format version (the only one this reader accepts).
pub const SIDECAR_FORMAT_VERSION: u32 = 1;

/// Header size: magic (8) + version + R + H + D (4 each).
const HEADER_LEN: usize = 24;

/// The variant-selection env var (exact-match values; see the module docs).
pub const VARIANT_ENV: &str = "KATGPT_HEAD_CLS_VARIANT";

/// The explicit incumbent value (same effect as unset/empty).
const VARIANT_ZERO: &str = "zero";
/// The trained-sidecar variant.
const VARIANT_TRAINED: &str = "trained";

/// The trained artifact's path under a weights/content dir.
#[must_use]
pub fn artifact_path(dir: &Path) -> PathBuf {
    dir.join("derived").join("head_cls_trained.bin")
}

/// The digest sidecar's path (BLAKE3 hex text; the trainer's `write_sidecar`
/// writes it, this loader verifies it — a derived artifact has no upstream
/// pin, so this is its tamper-evidence).
#[must_use]
pub fn sidecar_path(dir: &Path) -> PathBuf {
    dir.join("derived").join("head_cls_trained.bin.blake3")
}

/// Sidecar lane refusals. Every variant is loud and names its repair; none
/// is ever silently swallowed by a caller-facing resolver.
#[derive(Debug)]
pub enum SidecarError {
    /// An env value outside the expected set (a typo must fail loud).
    UnknownVariant {
        got: String,
        expected: &'static [&'static str],
    },
    /// The artifact or its digest sidecar is absent — never auto-derived.
    Missing { artifact: PathBuf, sidecar: PathBuf },
    /// Filesystem failure carrying the path.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The artifact's bytes no longer match its digest sidecar.
    DigestMismatch { want: String, got: String },
    /// Leading bytes are not the [`SIDECAR_MAGIC`].
    BadMagic { got: [u8; 8] },
    /// A well-formed artifact from a different format era.
    UnsupportedVersion(u32),
    /// A header dimension is zero (an empty reference set / head / dim has
    /// no meaning here).
    ZeroDimension { field: &'static str },
    /// The payload length disagrees with the header's shape — strict, no
    /// trailing bytes, no truncation.
    LenMismatch { expected: usize, got: usize },
}

impl fmt::Display for SidecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariant { got, expected } => write!(
                f,
                "unknown {VARIANT_ENV} {got:?} — expected unset/empty or one of \
                 {expected:?} (an env typo must fail loud, never fall back)"
            ),
            Self::Missing { artifact, sidecar } => write!(
                f,
                "trained head_cls sidecar missing — need both {} and its digest \
                 file {}; derive them with \
                 katgpt_attn::dash_attn::chunk_summary_sidecar::write_sidecar \
                 (riir-train Issue 666 T3's trainer mints them) — never auto-derived",
                artifact.display(),
                sidecar.display()
            ),
            Self::Io { path, source } => write!(f, "read {}: {source}", path.display()),
            Self::DigestMismatch { want, got } => write!(
                f,
                "BLAKE3 mismatch on the trained head_cls artifact \
                 (sidecar wants {want}, artifact hashes {got}) — re-derive, never load"
            ),
            Self::BadMagic { got } => {
                write!(
                    f,
                    "not a head_cls sidecar artifact (magic {got:02x?}, want {SIDECAR_MAGIC:02x?})"
                )
            }
            Self::UnsupportedVersion(v) => write!(
                f,
                "head_cls sidecar format version {v} unsupported by this reader \
                 (it reads {SIDECAR_FORMAT_VERSION}) — regenerate the artifact"
            ),
            Self::ZeroDimension { field } => {
                write!(
                    f,
                    "head_cls sidecar header has {field} = 0 — no meaning, refusing"
                )
            }
            Self::LenMismatch { expected, got } => write!(
                f,
                "head_cls sidecar payload length {got} != header-derived {expected} \
                 (strict: no trailing bytes, no truncation)"
            ),
        }
    }
}

impl std::error::Error for SidecarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Serialize a query into the v1 artifact bytes (pure encoder — the digest
/// sidecar is written by [`write_sidecar`], which this fn feeds).
#[must_use]
pub fn serialize(query: &ChunkSummaryQuery) -> Vec<u8> {
    let n_refs = query.n_refs() as u32;
    let n_kv_head = query.n_kv_head as u32;
    let head_dim = query.head_dim as u32;
    let mut out = Vec::with_capacity(HEADER_LEN + query.head_cls.len() * 4);
    out.extend_from_slice(&SIDECAR_MAGIC);
    out.extend_from_slice(&SIDECAR_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&n_refs.to_le_bytes());
    out.extend_from_slice(&n_kv_head.to_le_bytes());
    out.extend_from_slice(&head_dim.to_le_bytes());
    for v in &query.head_cls {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Parse v1 artifact bytes into a query — PRIVATE: the only public parse
/// entry is [`load_sidecar`], which digest-verifies FIRST. An unverified
/// parse entry would be a footgun (a corrupted artifact must never load).
fn deserialize(bytes: &[u8]) -> Result<ChunkSummaryQuery, SidecarError> {
    if bytes.len() < HEADER_LEN {
        return Err(SidecarError::LenMismatch {
            expected: HEADER_LEN,
            got: bytes.len(),
        });
    }
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&bytes[..8]);
    if magic != SIDECAR_MAGIC {
        return Err(SidecarError::BadMagic { got: magic });
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("header sized above"));
    if version != SIDECAR_FORMAT_VERSION {
        return Err(SidecarError::UnsupportedVersion(version));
    }
    let n_refs = u32::from_le_bytes(bytes[12..16].try_into().expect("header sized above"));
    let n_kv_head = u32::from_le_bytes(bytes[16..20].try_into().expect("header sized above"));
    let head_dim = u32::from_le_bytes(bytes[20..24].try_into().expect("header sized above"));
    for (field, v) in [
        ("n_refs", n_refs),
        ("n_kv_head", n_kv_head),
        ("head_dim", head_dim),
    ] {
        if v == 0 {
            return Err(SidecarError::ZeroDimension { field });
        }
    }
    let (n_refs, n_kv_head, head_dim) = (n_refs as usize, n_kv_head as usize, head_dim as usize);
    let expected = HEADER_LEN + n_refs * n_kv_head * head_dim * 4;
    if bytes.len() != expected {
        return Err(SidecarError::LenMismatch {
            expected,
            got: bytes.len(),
        });
    }
    let mut head_cls = Vec::with_capacity(n_refs * n_kv_head * head_dim);
    // Strict length already verified above → the 4-byte tail is empty.
    let (words, rest) = bytes[HEADER_LEN..].as_chunks::<4>();
    debug_assert!(rest.is_empty());
    for word in words {
        head_cls.push(f32::from_le_bytes(*word));
    }
    Ok(ChunkSummaryQuery::from_ref_major(
        head_cls, n_refs, n_kv_head, head_dim,
    ))
}

/// Write the artifact + its BLAKE3 digest sidecar under `dir`
/// (`derived/head_cls_trained.bin` + `.blake3`). The artifact lands via
/// temp-file + rename (a failed write never leaves a half artifact); the
/// digest sidecar is written only after the artifact is in place. Returns
/// the digest hex.
pub fn write_sidecar(dir: &Path, query: &ChunkSummaryQuery) -> Result<String, SidecarError> {
    let artifact = artifact_path(dir);
    let digest_file = sidecar_path(dir);
    if let Some(parent) = artifact.parent() {
        std::fs::create_dir_all(parent).map_err(|e| SidecarError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let bytes = serialize(query);
    let digest = blake3::hash(&bytes).to_hex().to_string();
    let tmp = artifact.with_extension("bin.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| SidecarError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, &artifact).map_err(|e| SidecarError::Io {
        path: artifact.clone(),
        source: e,
    })?;
    std::fs::write(&digest_file, format!("{digest}\n")).map_err(|e| SidecarError::Io {
        path: digest_file,
        source: e,
    })?;
    Ok(digest)
}

/// Read + digest-verify + parse the artifact at `path` (the digest sidecar
/// is `path` + `.blake3`). The ONLY public parse entry — a corrupted or
/// tampered artifact refuses here, never in the forward.
pub fn load_sidecar(path: &Path) -> Result<ChunkSummaryQuery, SidecarError> {
    let digest_file = sidecar_digest_path_for(path);
    if !path.is_file() || !digest_file.is_file() {
        return Err(SidecarError::Missing {
            artifact: path.to_path_buf(),
            sidecar: digest_file,
        });
    }
    let want = std::fs::read_to_string(&digest_file)
        .map_err(|e| SidecarError::Io {
            path: digest_file.clone(),
            source: e,
        })?
        .trim()
        .to_string();
    let bytes = std::fs::read(path).map_err(|e| SidecarError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let got = blake3::hash(&bytes).to_hex().to_string();
    if got != want {
        return Err(SidecarError::DigestMismatch { want, got });
    }
    deserialize(&bytes)
}

/// The digest sidecar for an explicit artifact path (`<path>.blake3`).
fn sidecar_digest_path_for(artifact: &Path) -> PathBuf {
    let mut name = artifact
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".blake3");
    artifact.with_file_name(name)
}

/// Resolve the variant-selected query for `dir` from
/// [`VARIANT_ENV`] — the production entry. Returns `Ok(None)` when the
/// incumbent zero-init posture is selected (unset/empty/`zero`); `Ok(Some)`
/// carries the digest-verified trained query. Every failure mode of
/// [`SidecarError`] is a refusal, never a fallback to the incumbent.
///
/// The pure variant half lives in [`resolve_query_for_variant`] (that is
/// what tests pin); this wrapper is the 4-line env read.
///
/// # Panics (edition-2024 note)
/// Reads the env var safely (`std::env::var`); no mutation.
pub fn resolve_sidecar_query(dir: &Path) -> Result<Option<ChunkSummaryQuery>, SidecarError> {
    let requested = match std::env::var(VARIANT_ENV) {
        Ok(v) if v.is_empty() => None,
        Ok(v) => Some(v),
        Err(_) => None,
    };
    resolve_query_for_variant(dir, requested.as_deref())
}

/// The pure variant half of [`resolve_sidecar_query`] — same resolution with
/// the env value passed in (tests pin this without touching process env).
pub fn resolve_query_for_variant(
    dir: &Path,
    variant: Option<&str>,
) -> Result<Option<ChunkSummaryQuery>, SidecarError> {
    let Some(variant) = variant else {
        return Ok(None);
    };
    if variant == VARIANT_ZERO {
        return Ok(None);
    }
    if variant == VARIANT_TRAINED {
        let artifact = artifact_path(dir);
        let side = sidecar_path(dir);
        if !artifact.is_file() || !side.is_file() {
            return Err(SidecarError::Missing {
                artifact,
                sidecar: side,
            });
        }
        return load_sidecar(&artifact).map(Some);
    }
    Err(SidecarError::UnknownVariant {
        got: variant.to_string(),
        expected: &[VARIANT_ZERO, VARIANT_TRAINED],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pid-suffixed scratch dir — never a FIXED temp path (the
    /// shared-temp-path rule: one process's test dir must not collide with
    /// another process running the same test).
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zha_sidecar_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// A small non-trivial query: R=2, H=2, D=3, deterministic values, all
    /// non-zero.
    fn sample_query() -> ChunkSummaryQuery {
        let mut q = ChunkSummaryQuery::with_n_refs(2, 2, 3);
        for (i, v) in q.head_cls.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.25;
        }
        q.recompute_zero_init_cache();
        q
    }

    #[test]
    fn serialize_roundtrip_preserves_bits_and_shape() {
        let q = sample_query();
        let bytes = serialize(&q);
        assert_eq!(&bytes[..8], &SIDECAR_MAGIC);
        // Strict length: 24-byte header + 12 f32 values.
        assert_eq!(bytes.len(), 24 + 12 * 4);
        // Parse through the verified file lane (deserialize is private).
        let dir = scratch_dir("roundtrip");
        write_sidecar(&dir, &q).expect("write");
        let loaded = load_sidecar(&artifact_path(&dir)).expect("load");
        assert_eq!(loaded.n_refs(), 2);
        assert_eq!(loaded.n_kv_head, 2);
        assert_eq!(loaded.head_dim, 3);
        assert_eq!(
            loaded
                .head_cls
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            q.head_cls.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            "payload must round-trip bit-exactly (f32 LE)"
        );
        // Ref addressing survives: ref_query(r, h) reads the same slice.
        for r in 0..2 {
            for h in 0..2 {
                assert_eq!(loaded.ref_query(r, h), q.ref_query(r, h));
            }
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// NaN payloads round-trip byte-exactly (bytes are bytes — the NaN
    /// poisoning downstream is the visibility law, not a parse refusal).
    #[test]
    fn serialize_roundtrip_nan_exact() {
        let mut q = ChunkSummaryQuery::with_n_refs(1, 1, 2);
        q.head_cls.copy_from_slice(&[f32::NAN, -0.0]);
        q.recompute_zero_init_cache();
        let dir = scratch_dir("nan");
        write_sidecar(&dir, &q).expect("write");
        let loaded = load_sidecar(&artifact_path(&dir)).expect("load");
        assert_eq!(
            loaded
                .head_cls
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            q.head_cls.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// THE warm-start anchor: an all-zero payload loads to `is_zero_init()`
    /// == true → the summarizer mean-pools → serving is byte-identical to
    /// the incumbent (T2's monotone-improving consumer-side guarantee).
    #[test]
    fn zero_payload_loads_as_zero_init() {
        let q = ChunkSummaryQuery::with_n_refs(2, 3, 4);
        assert!(q.is_zero_init());
        let dir = scratch_dir("zerowarm");
        write_sidecar(&dir, &q).expect("write");
        let loaded = load_sidecar(&artifact_path(&dir)).expect("load");
        assert!(
            loaded.is_zero_init(),
            "an all-zero sidecar MUST load as zero-init (the G3 byte-identity anchor)"
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// R = 1 round-trip: the flat layout is the incumbent's — head_cls ==
    /// payload values in the original order.
    #[test]
    fn r1_sidecar_roundtrip_is_incumbent_layout() {
        let mut q = ChunkSummaryQuery::new(2, 4);
        for (i, v) in q.head_cls.iter_mut().enumerate() {
            *v = i as f32 * 0.5;
        }
        q.recompute_zero_init_cache();
        let dir = scratch_dir("r1");
        write_sidecar(&dir, &q).expect("write");
        let loaded = load_sidecar(&artifact_path(&dir)).expect("load");
        assert_eq!(loaded.n_refs(), 1);
        for h in 0..2 {
            assert_eq!(loaded.head_query(h), q.head_query(h));
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn digest_tamper_refuses() {
        let dir = scratch_dir("tamper");
        write_sidecar(&dir, &sample_query()).expect("write");
        let artifact = artifact_path(&dir);
        // Corrupt ONE payload byte (bit-flip the first value's exponent).
        let mut bytes = std::fs::read(&artifact).expect("read");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&artifact, &bytes).expect("rewrite");
        match load_sidecar(&artifact) {
            Err(SidecarError::DigestMismatch { want, got }) => {
                assert_ne!(want, got, "tampered bytes must hash differently");
            }
            other => panic!(
                "expected DigestMismatch, got {other:?}",
                other = other.as_ref().err()
            ),
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn bad_magic_and_version_refuse() {
        let dir = scratch_dir("magic");
        let artifact = artifact_path(&dir);
        let digest = sidecar_path(&dir);

        // Phase 1: corrupt the magic. Re-sidecar the mutation so the DIGEST
        // check passes and the header check is what fires.
        write_sidecar(&dir, &sample_query()).expect("write");
        let mut bytes = std::fs::read(&artifact).expect("read");
        bytes[0] = b'X';
        std::fs::write(&artifact, &bytes).expect("rewrite");
        std::fs::write(&digest, format!("{}\n", blake3::hash(&bytes).to_hex())).expect("rewrite");
        match load_sidecar(&artifact) {
            Err(SidecarError::BadMagic { got }) => assert_eq!(got[0], b'X'),
            other => panic!("expected BadMagic, got {:?}", other.as_ref().err()),
        }

        // Phase 2: FRESH artifact, corrupt only the version field.
        write_sidecar(&dir, &sample_query()).expect("rewrite fresh");
        let mut bytes = std::fs::read(&artifact).expect("read");
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        std::fs::write(&artifact, &bytes).expect("rewrite");
        std::fs::write(&digest, format!("{}\n", blake3::hash(&bytes).to_hex())).expect("rewrite");
        match load_sidecar(&artifact) {
            Err(SidecarError::UnsupportedVersion(2)) => {}
            other => panic!(
                "expected UnsupportedVersion(2), got {:?}",
                other.as_ref().err()
            ),
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn length_and_zero_dim_refuse() {
        let dir = scratch_dir("shape");
        write_sidecar(&dir, &sample_query()).expect("write");
        let artifact = artifact_path(&dir);
        let digest = sidecar_path(&dir);

        // Trailing byte → strict length refusal.
        let mut bytes = std::fs::read(&artifact).expect("read");
        bytes.push(0);
        std::fs::write(&artifact, &bytes).expect("rewrite");
        std::fs::write(&digest, format!("{}\n", blake3::hash(&bytes).to_hex())).expect("rewrite");
        match load_sidecar(&artifact) {
            Err(SidecarError::LenMismatch { expected, got }) => {
                assert_eq!(got, expected + 1);
            }
            other => panic!("expected LenMismatch, got {:?}", other.as_ref().err()),
        }

        // Zero dimension in the header.
        let mut bytes = std::fs::read(&artifact).expect("read");
        bytes.pop(); // back to the strict length
        bytes[12..16].copy_from_slice(&0u32.to_le_bytes());
        std::fs::write(&artifact, &bytes).expect("rewrite");
        std::fs::write(&digest, format!("{}\n", blake3::hash(&bytes).to_hex())).expect("rewrite");
        match load_sidecar(&artifact) {
            Err(SidecarError::ZeroDimension { field }) => assert_eq!(field, "n_refs"),
            other => panic!("expected ZeroDimension, got {:?}", other.as_ref().err()),
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn missing_sidecar_refuses_naming_the_writer() {
        let dir = scratch_dir("missing");
        match resolve_query_for_variant(&dir, Some("trained")) {
            Err(SidecarError::Missing { artifact, sidecar }) => {
                assert_eq!(artifact, artifact_path(&dir));
                assert_eq!(sidecar, sidecar_path(&dir));
                let msg = SidecarError::Missing { artifact, sidecar }.to_string();
                assert!(
                    msg.contains("write_sidecar") && msg.contains("never auto-derived"),
                    "the refusal must name the writer and the no-auto-derive law: {msg}"
                );
            }
            other => panic!("expected Missing, got {:?}", other.as_ref().err()),
        }
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn variant_resolution_table() {
        let dir = scratch_dir("variant");
        // None / empty-string handled by the env wrapper; the pure half sees
        // the remainder.
        assert!(
            resolve_query_for_variant(&dir, None)
                .expect("none")
                .is_none()
        );
        assert!(
            resolve_query_for_variant(&dir, Some("zero"))
                .expect("zero")
                .is_none()
        );
        match resolve_query_for_variant(&dir, Some("trianed")) {
            Err(SidecarError::UnknownVariant { got, expected }) => {
                assert_eq!(got, "trianed");
                assert_eq!(expected, &["zero", "trained"]);
            }
            other => panic!("expected UnknownVariant, got {:?}", other.as_ref().err()),
        }
        // Happy path: write, then resolve "trained".
        write_sidecar(&dir, &sample_query()).expect("write");
        let resolved = resolve_query_for_variant(&dir, Some("trained"))
            .expect("trained resolves")
            .expect("trained is Some");
        assert_eq!(resolved.n_refs(), 2);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// Write-then-load twice is byte-stable (the artifact is a pure function
    /// of the query; the digest sidecar tracks it).
    #[test]
    fn write_is_deterministic() {
        let dir = scratch_dir("determinism");
        let d1 = write_sidecar(&dir, &sample_query()).expect("write 1");
        let bytes1 = std::fs::read(artifact_path(&dir)).expect("read 1");
        let d2 = write_sidecar(&dir, &sample_query()).expect("write 2");
        let bytes2 = std::fs::read(artifact_path(&dir)).expect("read 2");
        assert_eq!(d1, d2);
        assert_eq!(bytes1, bytes2);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
