# Issue 896 — KVarN: the raw tile buffer is shared across layers, and the in-progress tile dequantizes to zeros

**Status:** OPEN — found 2026-09-25 while closing Issue 894 (HISTORY.md § Issue 894). Both defects predate 894 and are unchanged by it: the 894 oracle reproduces the old behaviour bit for bit. Nothing is fixed yet.

## Finding 1 — cross-layer corruption under decode-order stores (measured)

`KVarNKVCache` has ONE `key_buffer` / `val_buffer` (`[kv_dim × tile_size]`) for all layers (`crates/katgpt-kv/src/kvarn/kv_cache.rs`, `store_key` / `store_value`). The buffer slot depends only on `pos_in_tile`, never on `layer`. So when positions are stored in decode order (for each position, for each layer), every layer writes the same slots. When layer L's tile fills and quantizes, it reads whatever the LAST layer wrote at positions 0..tile_size−2.

Probe: 2 layers, kv_dim 16, tile 128, 8-bit, decode-order stores. Layer 0 holds values 1..13 and layer 1 holds values −50..−54.

| read | got[0..4] | want |
|---|---|---|
| L0 p3 value | −54.01, −50.02, −51.00, −51.98 | 9, 10, 11, 12 |
| L0 p3 key | −54.0, −49.93, −51.08, −52.02 | 9, 10, 11, 12 |
| L0 p127 value | 5.99, 7.02, 7.98, 8.98 | 6, 7, 8, 9 |
| L1 p3 value | −54.0, −50.0, −51.0, −52.0 | −54, −50, −51, −52 |

Layer 0 serves layer 1's data. Only layer-major store order (all positions of layer 0, then layer 1, …) is correct, and that is what every in-tree KVarN test and bench uses (`n_layers: 1`, or layer-major loops). That is why nothing caught it.

## Finding 2 — reads of the in-progress tile (measured)

A tile is quantized only when it fills (or at `max_seq_len − 1`). Until then `dequantize_*_into` reads the tile's `TileMeta::empty` metadata (or, after `reset()`, the previous sequence's stale metadata) plus zeroed packed bytes. In the probe (kv_dim 8, 10 of 128 positions stored, values 1..8), 2-bit and 4-bit value reads at p5 return **all zeros**. At 2 bits, a key read in that tile **panics** (index out of range), because the grouped scale layout is indexed against the empty, `kv_dim`-long vector. During decode the current tile is exactly this state, so the most recent ≤ tile_size − 1 positions dequantize wrong.

## Tasks

- [ ] **T1 — decide the contract.** Either (a) KVarN supports interleaved multi-layer stores, with a per-layer raw buffer (`n_layers × kv_dim × tile_size` f32, the same size as one layer's quantized payload at 32 bits), or (b) it is documented and debug-asserted as layer-major only. (a) is the correct default for a `QuantizedKVCache` backend.
- [ ] **T2 — serve the in-progress tile from the raw buffer** (exact values) instead of empty or stale metadata. Pin it with a partial-tile roundtrip test at every bit width, including 2-bit keys.
- [ ] **T3 — regression tests** covering decode-order stores at n_layers ≥ 2 and reads at every position < pos, plus `reset()` followed by a partial tile.
- [ ] **T4 — re-gate what consumed KVarN under decode order**, if anything did (grep the root `kvarn` consumers), because their quality figures may carry Finding 1.
