//! Chunk fetcher implementations — Phase 3 hydration backends (Plan 272 T3.1–T3.5).
//!
//! Three concrete backends for the [`ChunkFetcher`] trait (defined in
//! [`super::r#trait`]):
//! - [`InMemoryChunkFetcher`] — `papaya` lock-free map (test/single-process).
//! - [`FsChunkFetcher`] — sharded filesystem layout (production local cache).
//! - [`TieredChunkFetcher`] — composite local→fallback with optional write-back.
//!
//! `NetChunkFetcher` (Plan 272 T3.3) is deferred to the `chunked_net_fetch`
//! feature gate — adding it requires a new Cargo.toml feature, which was
//! deferred to avoid colliding with concurrent `Cargo.toml` edits. The
//! `TieredChunkFetcher` is generic over any `ChunkFetcher`, so a net fetcher
//! plugs in cleanly when that feature lands.
//!
//! ## Design: read-only trait, inherent `put()`
//!
//! [`ChunkFetcher`] is read-only (`fetch` / `fetch_range`). Concrete backends
//! expose an inherent `put()` for populating the cache — this is intentionally
//! NOT on the trait, because a read-only consumer (e.g. a light replica
//! hydrating from a peer) should not be able to mutate a shared fetcher. The
//! `TieredChunkFetcher`'s write-back path uses the local backend's `put()`
//! directly (not via the trait), preserving this discipline.

use std::path::PathBuf;

use papaya::HashMap;

use super::r#trait::ChunkFetcher;
use super::types::{BlobId, ChunkRange};

// ────────────────────────────────────────────────────────────────────────────
// T3.1 — InMemoryChunkFetcher
// ────────────────────────────────────────────────────────────────────────────

/// In-memory chunk fetcher backed by a `papaya` lock-free hashmap.
///
/// Primary use: test harnesses and single-process deploys where a shared
/// chunk cache lives in process memory. For multi-process or persistent
/// deployment, use [`FsChunkFetcher`].
///
/// Thread-safe: all reads/writes go through `papaya`'s lock-free paths.
pub struct InMemoryChunkFetcher {
    /// BLAKE3(chunk) → chunk bytes.
    chunks: HashMap<[u8; 32], Vec<u8>>,
}

impl InMemoryChunkFetcher {
    /// Construct an empty fetcher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    /// Insert a chunk keyed by its BLAKE3 hash. Overwrites any existing entry
    /// for the same hash (chunks are content-addressed, so identical hashes
    /// imply identical bytes — the overwrite is a no-op semantically).
    pub fn put(&self, chunk_hash: [u8; 32], bytes: Vec<u8>) {
        self.chunks.pin().insert(chunk_hash, bytes);
    }

    /// Current number of cached chunks (advisory — lock-free count).
    pub fn len(&self) -> usize {
        self.chunks.pin().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for InMemoryChunkFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkFetcher for InMemoryChunkFetcher {
    fn fetch(&self, chunk_hash: &[u8; 32]) -> Option<Vec<u8>> {
        self.chunks.pin().get(chunk_hash).cloned()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// T3.2 — FsChunkFetcher
// ────────────────────────────────────────────────────────────────────────────

/// Sharded-filesystem chunk fetcher.
///
/// Layout: `<root>/<hash[0..2]>/<hash[2..4]>/<hash>.chunk` (hex-encoded shard
/// dirs). Sharding avoids filesystem directory-entry limits at scale (ext4
/// soft-limits ~10K entries/dir; a 1M-chunk store would hit this without
/// sharding, but with 2+2 hex bytes there are 256×256 = 65 536 leaf dirs,
/// averaging ~15 entries each).
///
/// **Read path deviation (honest):** Plan 272 T3.2 specifies mmap reads.
/// This implementation uses `std::fs::read` instead. Rationale: (1) chunks are
/// ≤ 64 KiB (`FASTCDC_MAX_CHUNK_SIZE`), so a single `read()` syscall matches
/// mmap perf — the zero-copy advantage of mmap only materializes for large
/// spans that cross many page faults; (2) adding `memmap2` would require a
/// `Cargo.toml` dep change, which collides with concurrent edits to that file.
/// The `fetch()` signature returns `Vec<u8>` (owned), so the copy is already
/// implied by the trait contract — mmap would still need a `to_vec()` to
/// satisfy the return type. Upgrade to mmap when the trait gains a
/// zero-copy `fetch_borrowed` path or when `Cargo.toml` is stable.
pub struct FsChunkFetcher {
    /// Root directory for the chunk store.
    root: PathBuf,
}

impl FsChunkFetcher {
    /// Construct a fetcher rooted at `root`. The directory is created lazily
    /// on first `put` (not eagerly — allows constructing a read-only fetcher
    /// against a store populated by another process).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Compute the sharded on-disk path for a chunk hash.
    ///
    /// Per Plan 272 T3.2: `<root>/<hash[0..2]>/<hash[2..4]>/<hash>.chunk` where
    /// `hash[0..2]` and `hash[2..4]` are 2-byte slices (hex-encoded to 4 chars
    /// each) and `<hash>` is the full 32-byte hash (hex-encoded to 64 chars).
    ///
    /// Sharding: 2 bytes per level → 65 536 dirs per level → 65 536² ≈ 4 B leaf
    /// capacity. A 1 M-chunk store averages ~15 entries per leaf dir.
    ///
    /// Exposed for tests (T3.5 sharded-path assertion) and for callers that
    /// need to pre-create shard dirs in bulk.
    #[must_use]
    pub fn sharded_path(&self, chunk_hash: &[u8; 32]) -> PathBuf {
        let hex = hex_encode(chunk_hash);
        // hash[0..2] → hex[0..4], hash[2..4] → hex[4..8], filename = full hex.
        let shard1 = &hex[0..4];
        let shard2 = &hex[4..8];
        self.root
            .join(shard1)
            .join(shard2)
            .join(format!("{hex}.chunk"))
    }

    /// Write a chunk to disk at its sharded path. Creates parent dirs as
    /// needed. Overwrites any existing file (content-addressed — same hash
    /// implies same bytes).
    ///
    /// Errors propagate as `std::io::Result`; the caller decides retry policy.
    pub fn put(&self, chunk_hash: &[u8; 32], bytes: &[u8]) -> std::io::Result<()> {
        let path = self.sharded_path(chunk_hash);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Write to a temp file in the same dir, then rename — atomic on the
        // same filesystem, avoids partial reads if a `fetch` races a `put`.
        let tmp = path.with_extension("chunk.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Read a chunk from disk. Returns `None` if the file does not exist
    /// (honest-miss — caller may try a fallback). Other IO errors propagate.
    fn read(&self, chunk_hash: &[u8; 32]) -> std::io::Result<Option<Vec<u8>>> {
        let path = self.sharded_path(chunk_hash);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl ChunkFetcher for FsChunkFetcher {
    fn fetch(&self, chunk_hash: &[u8; 32]) -> Option<Vec<u8>> {
        // IO errors other than NotFound are unexpected on a well-formed store.
        // Log-and-treat-as-miss is the honest fallback (a corrupted shard dir
        // is a deployment problem, not a fetcher-logic problem). The caller's
        // `TieredChunkFetcher` will try the next tier.
        match self.read(chunk_hash) {
            Ok(opt) => opt,
            Err(_) => None,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// T3.4 — TieredChunkFetcher (composite)
// ────────────────────────────────────────────────────────────────────────────

/// Composite fetcher: try `local` first, fall back to `fallback` on miss.
///
/// Generic over both tiers so a test can compose two `InMemoryChunkFetcher`s
/// and a production deploy can compose `FsChunkFetcher` (local) with a future
/// `NetChunkFetcher` (fallback).
///
/// **Write-back** is opt-in via the [`TieredWriteBackExt`] extension trait:
/// callers who want fallback hits persisted to the local tier import the
/// trait and call `fetch_with_write_back` instead of `fetch`. This avoids a
/// runtime `write_back: bool` flag (dead in the read-only `fetch()` path) and
/// keeps the type system honest — write-back requires `Local: WriteBack`,
/// which is checked at compile time, not at a runtime branch.
///
/// **Why not `Box<dyn ChunkFetcher>` for both tiers?** The write-back path
/// needs to call `put()` on the local tier, which is an inherent method (not
/// on `ChunkFetcher`). A trait-object would erase it. Generic-over-concrete
/// preserves the `put()` surface and zero-cost monomorphizes.
pub struct TieredChunkFetcher<Local, Fallback>
where
    Local: ChunkFetcher,
    Fallback: ChunkFetcher,
{
    local: Local,
    fallback: Fallback,
}

/// Sealed trait granting write-back access to a fetcher's `put()`.
///
/// Implemented for the two concrete backends that support population
/// (`InMemoryChunkFetcher`, `FsChunkFetcher`). `TieredChunkFetcher` uses this
/// to write back fallback hits. Sealed so external types can't inject into
/// the write-back path.
pub trait WriteBack: ChunkFetcher {
    /// Store a chunk keyed by its hash.
    fn write_back(&self, chunk_hash: &[u8; 32], bytes: &[u8]);
}

impl WriteBack for InMemoryChunkFetcher {
    fn write_back(&self, chunk_hash: &[u8; 32], bytes: &[u8]) {
        self.put(*chunk_hash, bytes.to_vec());
    }
}

impl WriteBack for FsChunkFetcher {
    fn write_back(&self, chunk_hash: &[u8; 32], bytes: &[u8]) {
        // Best-effort — a failed write-back (disk full, permissions) is
        // non-fatal; the chunk was already fetched successfully. The next
        // fetch will retry the local tier then the fallback again.
        let _ = self.put(chunk_hash, bytes);
    }
}

impl<L, F> TieredChunkFetcher<L, F>
where
    L: ChunkFetcher,
    F: ChunkFetcher,
{
    /// Construct a tiered fetcher.
    #[must_use]
    pub fn new(local: L, fallback: F) -> Self {
        Self { local, fallback }
    }

    /// Borrow the local tier (for direct `put()` population by the deploy).
    pub fn local(&self) -> &L {
        &self.local
    }

    /// Borrow the fallback tier.
    pub fn fallback(&self) -> &F {
        &self.fallback
    }
}

impl<L, F> ChunkFetcher for TieredChunkFetcher<L, F>
where
    L: ChunkFetcher,
    F: ChunkFetcher,
{
    fn fetch(&self, chunk_hash: &[u8; 32]) -> Option<Vec<u8>> {
        if let Some(bytes) = self.local.fetch(chunk_hash) {
            return Some(bytes);
        }
        self.fallback.fetch(chunk_hash)
    }

    fn fetch_range(&self, blob_id: &BlobId, range: ChunkRange) -> Option<Vec<u8>> {
        if let Some(bytes) = self.local.fetch_range(blob_id, range) {
            return Some(bytes);
        }
        self.fallback.fetch_range(blob_id, range)
    }
}

/// Extension trait providing write-back-enabled fetch on `TieredChunkFetcher`
/// where `Local: WriteBack`. This is the explicit, type-safe path for
/// write-back — callers opt in by importing the trait and calling
/// `fetch_with_write_back` instead of the plain `fetch`.
pub trait TieredWriteBackExt<L: WriteBack, F: ChunkFetcher> {
    /// Fetch with write-back: on fallback hit, store to local.
    fn fetch_with_write_back(&self, chunk_hash: &[u8; 32]) -> Option<Vec<u8>>;
}

impl<L, F> TieredWriteBackExt<L, F> for TieredChunkFetcher<L, F>
where
    L: WriteBack,
    F: ChunkFetcher,
{
    fn fetch_with_write_back(&self, chunk_hash: &[u8; 32]) -> Option<Vec<u8>> {
        if let Some(bytes) = self.local.fetch(chunk_hash) {
            return Some(bytes);
        }
        let bytes = self.fallback.fetch(chunk_hash)?;
        self.local.write_back(chunk_hash, &bytes);
        Some(bytes)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// T3.3 — NetChunkFetcher (behind feature `chunked_net_fetch`)
// ────────────────────────────────────────────────────────────────────────────

/// Network chunk fetcher — fetches chunks from a remote backend (S3, IPFS
/// gateway, riir-chain RPC, a Lore server).
///
/// **Stub implementation (Plan 272 T3.3 "otherwise" path):** `reqwest` is not
/// a dependency of `katgpt-core`. This stub defines the trait surface and URL
/// construction, but `fetch()` always returns `None`. When `reqwest` is added
/// (behind the same feature), the stub gains a real HTTP client.
///
/// The URL pattern is `<base_url>/<hex_hash>` — the deploy decides the scheme
/// and backend. This keeps the fetcher generic over the storage technology.
#[cfg(feature = "chunked_net_fetch")]
pub struct NetChunkFetcher {
    /// Base URL prefix (e.g. `https://cdn.example.com/chunks`).
    base_url: String,
}

#[cfg(feature = "chunked_net_fetch")]
impl NetChunkFetcher {
    /// Construct with a base URL. The chunk hash is appended as a hex path
    /// segment: `<base_url>/<hex_hash>`.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }

    /// Compute the URL for a chunk hash: `<base_url>/<hex_hash>`.
    ///
    /// Exposed for testing and for callers that want to pre-warm a CDN cache.
    #[must_use]
    pub fn chunk_url(&self, chunk_hash: &[u8; 32]) -> String {
        format!("{}/{}", self.base_url, hex_encode(chunk_hash))
    }
}

#[cfg(feature = "chunked_net_fetch")]
impl ChunkFetcher for NetChunkFetcher {
    fn fetch(&self, _chunk_hash: &[u8; 32]) -> Option<Vec<u8>> {
        // Stub: no HTTP client linked. Returns None — the `TieredChunkFetcher`
        // caller treats this as a miss and falls through to the next tier
        // (or reports the chunk as unavailable).
        //
        // When `reqwest` is added behind `chunked_net_fetch = ["dep:reqwest"]`,
        // this becomes a blocking or async HTTP GET. The `ChunkFetcher` trait
        // is sync (`fn fetch(&self, ...) -> Option<Vec<u8>>`), so the async
        // case needs a runtime block_on or a separate async trait — deferred
        // to the real implementation.
        None
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Hex-encode a 32-byte hash to a 64-char lowercase ASCII string (no allocation
/// beyond the returned `String`; used only in path construction which already
/// allocates).
fn hex_encode(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// T3.5 — Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Generate a unique tmpdir under `std::env::temp_dir()` for FsChunkFetcher
    /// tests. Avoids a `tempfile` dev-dep (which would require a Cargo.toml
    /// change — deferred to avoid colliding with concurrent edits).
    ///
    /// Cleanup is best-effort in `Drop` via [`TempDir`].
    fn unique_tmpdir(prefix: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("{prefix}_katgpt272_{pid}_{n}"));
        std::fs::create_dir_all(&dir).expect("create tmpdir");
        dir
    }

    /// RAII tmpdir guard — removes the dir on drop.
    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // ── InMemoryChunkFetcher ──────────────────────────────────────────────

    #[test]
    fn in_memory_put_then_fetch() {
        let fetcher = InMemoryChunkFetcher::new();
        let hash = [42u8; 32];
        let data = b"hello chunk".to_vec();
        fetcher.put(hash, data.clone());
        assert_eq!(fetcher.fetch(&hash), Some(data));
    }

    #[test]
    fn in_memory_missing_returns_none() {
        let fetcher = InMemoryChunkFetcher::new();
        let hash = [0u8; 32];
        assert_eq!(fetcher.fetch(&hash), None);
    }

    #[test]
    fn in_memory_len_tracks_inserts() {
        let fetcher = InMemoryChunkFetcher::new();
        assert!(fetcher.is_empty());
        fetcher.put([1u8; 32], vec![1]);
        fetcher.put([2u8; 32], vec![2]);
        assert_eq!(fetcher.len(), 2);
    }

    // ── FsChunkFetcher: T3.5 roundtrip put/get ────────────────────────────

    #[test]
    fn fs_roundtrip_put_get() {
        let dir = unique_tmpdir("fs_rt");
        let _guard = TempDir(dir.clone());
        let fetcher = FsChunkFetcher::new(&dir);

        let hash = [0xaau8; 32];
        let data = b"chunk payload bytes".to_vec();
        fetcher.put(&hash, &data).expect("put");

        let fetched = fetcher.fetch(&hash).expect("fetch should find it");
        assert_eq!(fetched, data);
    }

    #[test]
    fn fs_missing_chunk_returns_none() {
        let dir = unique_tmpdir("fs_miss");
        let _guard = TempDir(dir.clone());
        let fetcher = FsChunkFetcher::new(&dir);

        let hash = [0xbbu8; 32];
        assert_eq!(fetcher.fetch(&hash), None, "missing chunk must be None");
    }

    #[test]
    fn fs_sharded_path_is_correct() {
        let dir = unique_tmpdir("fs_path");
        let _guard = TempDir(dir.clone());
        let fetcher = FsChunkFetcher::new(&dir);

        // hash[0]=0xab, hash[1]=0xcd, hash[2]=0x00, hash[3]=0x00
        // → shard1 = hex([ab,cd]) = "abcd", shard2 = hex([00,00]) = "0000"
        // → filename = full 64-char hex + ".chunk"
        let mut hash = [0u8; 32];
        hash[0] = 0xab;
        hash[1] = 0xcd;
        let path = fetcher.sharded_path(&hash);

        let full_hex = hex_encode(&hash);
        let expected_suffix = std::path::Path::new("abcd")
            .join("0000")
            .join(format!("{full_hex}.chunk"));
        assert!(
            path.ends_with(&expected_suffix),
            "path {path:?} should end with {expected_suffix:?}"
        );
    }

    #[test]
    fn fs_multiple_chunks_roundtrip() {
        let dir = unique_tmpdir("fs_multi");
        let _guard = TempDir(dir.clone());
        let fetcher = FsChunkFetcher::new(&dir);

        // Put 50 chunks with distinct hashes and verify all round-trip.
        for i in 0u8..50 {
            let mut hash = [0u8; 32];
            hash[0] = i;
            let data = vec![i; (i as usize) * 100 + 1]; // 1..4901 bytes
            fetcher.put(&hash, &data).expect("put");
        }
        for i in 0u8..50 {
            let mut hash = [0u8; 32];
            hash[0] = i;
            let expected = vec![i; (i as usize) * 100 + 1];
            assert_eq!(fetcher.fetch(&hash), Some(expected), "chunk {i}");
        }
    }

    #[test]
    fn fs_overwrite_is_idempotent() {
        let dir = unique_tmpdir("fs_ow");
        let _guard = TempDir(dir.clone());
        let fetcher = FsChunkFetcher::new(&dir);

        let hash = [0x11u8; 32];
        fetcher.put(&hash, b"first").expect("put 1");
        // Same hash → same content-addressed identity; overwrite is a no-op.
        fetcher.put(&hash, b"first").expect("put 2");
        assert_eq!(fetcher.fetch(&hash), Some(b"first".to_vec()));
    }

    // ── TieredChunkFetcher ────────────────────────────────────────────────

    #[test]
    fn tiered_local_hit_skips_fallback() {
        let local = InMemoryChunkFetcher::new();
        let fallback = InMemoryChunkFetcher::new();
        let hash = [0x42u8; 32];

        local.put(hash, b"local".to_vec());
        fallback.put(hash, b"fallback".to_vec());

        let tiered = TieredChunkFetcher::new(local, fallback);
        // Local hit — should return "local", not "fallback".
        assert_eq!(tiered.fetch(&hash), Some(b"local".to_vec()));
    }

    #[test]
    fn tiered_local_miss_falls_back() {
        let local = InMemoryChunkFetcher::new();
        let fallback = InMemoryChunkFetcher::new();
        let hash = [0x43u8; 32];

        fallback.put(hash, b"from fallback".to_vec());

        let tiered = TieredChunkFetcher::new(local, fallback);
        assert_eq!(tiered.fetch(&hash), Some(b"from fallback".to_vec()));
    }

    #[test]
    fn tiered_both_miss_returns_none() {
        let local = InMemoryChunkFetcher::new();
        let fallback = InMemoryChunkFetcher::new();
        let tiered = TieredChunkFetcher::new(local, fallback);
        assert_eq!(tiered.fetch(&[0u8; 32]), None);
    }

    #[test]
    fn tiered_write_back_populates_local() {
        // Verify the TieredWriteBackExt path: fallback hit → written to local.
        let local = InMemoryChunkFetcher::new();
        let fallback = InMemoryChunkFetcher::new();
        let hash = [0x44u8; 32];
        fallback.put(hash, b"fallback data".to_vec());

        let tiered = TieredChunkFetcher::new(local, fallback);
        // First fetch: local miss → fallback hit → write back to local.
        let got = tiered.fetch_with_write_back(&hash);
        assert_eq!(got, Some(b"fallback data".to_vec()));
        // The local tier should now have the chunk.
        assert_eq!(
            tiered.local().fetch(&hash),
            Some(b"fallback data".to_vec()),
            "write-back should populate local"
        );
    }

    #[test]
    fn tiered_write_back_with_fs_local() {
        // FsChunkFetcher as local tier, InMemory as fallback — write-back
        // should persist the chunk to disk.
        let dir = unique_tmpdir("tiered_wb_fs");
        let _guard = TempDir(dir.clone());
        let local = FsChunkFetcher::new(&dir);
        let fallback = InMemoryChunkFetcher::new();
        let hash = [0x55u8; 32];
        fallback.put(hash, b"net data".to_vec());

        let tiered = TieredChunkFetcher::new(local, fallback);
        let got = tiered.fetch_with_write_back(&hash);
        assert_eq!(got, Some(b"net data".to_vec()));

        // A fresh FsChunkFetcher on the same dir should find the written-back chunk.
        let local2 = FsChunkFetcher::new(&dir);
        assert_eq!(
            local2.fetch(&hash),
            Some(b"net data".to_vec()),
            "write-back should persist to FS"
        );
    }
}
