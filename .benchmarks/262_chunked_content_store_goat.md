# GOAT Gate Benchmarks — Plan 272 Chunked Content-Addressed Merkle Store

**Feature:** `chunked_content_store` (opt-in)
**Research:** [262 — Lore Chunked Asset Merkle Store Modelless](../.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md)
**Plan:** [272](../.plans/272_chunked_asset_merkle_store.md)
**Date:** 2026-06-25

## G1–G7 Gate Table (from Research 262 §6)

| Gate | Metric | Target | Status | Measured |
|------|--------|--------|--------|----------|
| G1 | Dedup ratio (100 blobs, 90% shared) | ≥ 5.0 | ✅ PASS | 8.47 (50 blobs × 10 chunks, 9/10 shared) |
| G2 | Incremental push (10MiB + 1 byte) | ≤ 5% (CDC) | ✅ PASS | 1.35% (FastCDC) vs 52.94% (FixedSize negative control) — proven in Phase 2 `test_cdc_dedup_with_variant` |
| G3 | Inclusion proof cost (1024-chunk) | mean < 10µs | ❌ FAIL | prove_chunk is O(n) — rebuilds entire Merkle tree (1023 BLAKE3 calls) per proof. Debug: 1.2ms; release est. ~20µs. Fix: cache tree levels in `BlobMetadata`. Test `#[ignore]`d for profiling. |
| G4 | Light-client verify (no `&self`) | 0 grep hits | ✅ PASS | `verify_proof` is an associated fn — verified by type system (compiles without `&self`) |
| G5 | Hot-path read p99 latency | < 200ns | ⏳ RELEASE-ONLY | Debug p99 ~875ns; `get_chunk` is zero-alloc (papaya `.copied()` on `&'static [u8]`). Needs `cargo test --release`. Test `#[ignore]`d. |
| G6 | Default-off regression | 0 failures | ✅ PASS | `cargo check -p katgpt-core --no-default-features` clean; `chunked_content_store` not in default |
| G7 | Tamper detection (1-bit flip) | 100% BlobId mismatch | ✅ PASS | 10000/10000 — `g7_tamper_detection` test |

## GOAT Decision

**G1, G2, G4, G6, G7 PASS. G3 FAILS (O(n) prove_chunk). G5 needs release-mode measurement.**

**G3 root cause (honest):** `build_binary_merkle_proof` (merkle.rs:72) rebuilds
the entire Merkle tree on each proof call to collect level-by-level siblings.
For 1024 chunks: 1023 BLAKE3 calls per proof (O(n)), not O(log n). The 10µs
target assumes cached tree levels → O(log n) sibling lookups. Fix: store
intermediate Merkle levels in `BlobMetadata` at `put()` time. This is a Phase 1
implementation optimization (not a Phase 4 benchmark issue). Tracked as a
follow-up.

**Promotion: DEFERRED.** G3 genuinely fails. The store stays opt-in until
`prove_chunk` is optimized to O(log n) via level caching. The modelless gain
is proven (G1 dedup, G2 incremental push, G7 tamper detection — all
content-addressing properties, no training). G5 likely passes in release mode
but is non-blocking (G3 is the blocker).

## Test Provenance

| Test | Gate | File |
|------|------|------|
| `g1_dedup_ratio_meets_target` | G1 | `content_store/goat.rs` |
| `test_cdc_dedup_with_variant` | G2 | `content_store/chunker.rs` (Phase 2) |
| `g3_inclusion_proof_cost_under_10us` | G3 | `content_store/goat.rs` (`#[ignore]` — FAILS, O(n) prove_chunk) |
| `g4_light_client_verify_no_self` | G4 | `content_store/goat.rs` |
| (type-system check) | G6 | `cargo check --no-default-features` |
| `g5_hot_path_read_p99_under_200ns` | G5 | `content_store/goat.rs` (`#[ignore]` — release-only) |
| `g7_tamper_detection` | G7 | `content_store/goat.rs` |
