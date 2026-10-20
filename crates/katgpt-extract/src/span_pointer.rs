//! `span_pointer` — candidate-window span scoring (Plan 628 T0.2).
//!
//! PLACEHOLDER — implemented by the T0.2 lane. Contract: a per-state
//! scorer built once over the tokenized state (cold), scoring candidate
//! windows for one field query into a caller-owned top-k buffer (hot,
//! zero-alloc): BM25/ngram lexical similarity + hashed-feature embedding
//! similarity + alias-table augmentation (the field's `choices`), combined
//! and SIGMOID-normalized (never softmax), byte offsets exact into the
//! state string.
//!
//! G5 (external BLAKE3-pinned fixture vs a rule floor) binds its GOAT —
//! T0.6, not here.
