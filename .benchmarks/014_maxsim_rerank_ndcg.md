# Benchmark 014: MaxSim Reranking — NDCG@10 vs Cosine

**Date:** 2025-05-20
**Plan:** 080 (MaxSim Late-Interaction Scoring), Task T12
**Command:** `cargo test --features maxsim --test bench_maxsim_rerank -- --nocapture`
**Machine:** macOS (Apple Silicon)
**Rust:** edition 2024, release profile

## Test Design

Synthetic REST retrieval simulation measuring retrieval quality (NDCG@10) for MaxSim late-interaction scoring vs standard Cosine mean-pooled scoring.

### Configuration

| Parameter | Value |
|-----------|-------|
| Documents | 50 |
| Query tokens (Lq) | 8 |
| Doc tokens (Ld) | 16 |
| Embedding dim | 64 |
| Trials | 100 (seeds 1000–1099) |

### Relevance Tiers

| Tier | Count | Signal Matches | Relevance (rel) |
|------|-------|---------------|-----------------|
| Highly relevant | 5 | 4 (non-overlapping dim blocks) | 3.0 |
| Partial | 15 | 2–3 matches | 1.5 |
| Irrelevant | 30 | 0 matches | 0.0 |

### Signal Construction

- **Near-orthogonal signal vectors**: non-overlapping dimension blocks ensure clean separation between relevance tiers.
- **Quantization noise**: 0.8–1.2× per-dimension scaling simulates compressed embedding artifacts.
- **Deterministic trials**: fixed seeds (1000–1099) ensure reproducibility.

## New Module

### `src/rerank.rs` (feature-gated behind `maxsim`)

| Component | Description |
|-----------|-------------|
| `RerankMethod` enum | `Cosine`, `MaxSim` |
| `RerankedDoc` struct | Document ID, score, relevance label |
| `rerank()` | Scores and sorts documents by chosen method |
| `ndcg_at()` | NDCG@k evaluation metric |
| `cosine_score()` | Mean cosine similarity across all token pairs |

### Rerank Flow

1. For each document, compute score using selected method (Cosine or MaxSim).
2. Sort documents by score descending.
3. Compute NDCG@10 against ground-truth relevance labels.

## GOAT Gate

| Gate | Metric | Threshold | Status |
|------|--------|-----------|--------|
| T12 | MaxSim NDCG@10 ≥ Cosine NDCG@10 × 1.02 | ≥2% improvement | ⏳ Pending |

### Expected Result

MaxSim should achieve significantly higher NDCG@10 than Cosine because late-interaction scoring matches token-level signals that mean-pooling dilutes:

- **Cosine (mean-pooled)**: averages all token embeddings, spreading signal across irrelevant dimensions and reducing discriminative power for partial-relevance documents.
- **MaxSim (late interaction)**: selects the best-matching token pair per query token, directly recovering signal matches even when diluted across the document.

## Cross-References

| Reference | Description |
|-----------|-------------|
| Plan 080 | MaxSim Late-Interaction Scoring |
| Research 45 | MaxSim Memory-Efficient Late-Interaction Scoring |
| Benchmark 013 | TurboQuant vs SpectralQuant MaxSim |
| Plan 009 | REST Speculative Decoding — provides retrieval infrastructure |

## Test Commands

```sh
# Run reranking NDCG benchmark
cargo test --features maxsim --test bench_maxsim_rerank -- --nocapture

# Run all maxsim tests
cargo test --features maxsim --lib --quiet

# Clippy clean
cargo clippy --features maxsim --quiet
```
