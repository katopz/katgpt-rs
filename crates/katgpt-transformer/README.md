# katgpt-transformer

Transformer substrate types shared between `katgpt-rs` and `riir-engine`:
`LayerWeights`, `TransformerWeights`, KV caches (`KVCache`, `MultiLayerKVCache`,
`KVSnapshot`, `KVLayerSnapshot`, `PagedKVCache`, `RavenKVCache`), context
buffers (`PrefillContext`, `WallPrefixState`, `GateStatistics`), MTP projection
loader (`MtpProjection`), and contiguous weight packing (`ContiguousWeights`).

## Why a separate crate?

`katgpt-core` ships pure substrate primitives (types, SIMD kernels, leaf
algorithms). Transformer weights and KV caches are substrate too, but their
`#[cfg(feature = "...")]` fields pull in a transformer-specific feature
namespace (`wall_attention`, `delta_routing`, `decode_specialize`,
`plasma_path`) that does not belong in core. This crate gives that namespace
its own home without bloating core.

## What stays in `katgpt-rs` root

The transformer **forward functions** (`forward`, `forward_base`,
`forward_coda`, `forward_looped`, `forward_prefill`, `forward_paged`,
`forward_raven`, `forward_quantized`, `forward_turboquant`, `generate_*`,
etc.) stay in `katgpt-rs/src/transformer.rs` because they are **composition
logic** — they call into root-only cognitive modules (`crate::hla`,
`crate::sleep`, `crate::tf_loop`, `crate::gdn2`, `crate::turboquant`,
`crate::pruners`). `ForwardContext` also stays in root: its fields reference
root-only pruner types (`CnaModulator`, `SubstrateMask`, `HydraSkipPlan`).

## Features

| Feature | Default | Gates |
|---------|---------|-------|
| `wall_attention` | on | `LayerWeights.attn_wg`, `WallPrefixState`, `GateStatistics` |
| `delta_routing` | on | `TransformerWeights.delta_routing_query` / `_norm` |
| `decode_specialize` | on | `DecodeStage` enum |
| `plasma_path` | on | `load_ternary_bits` in `contiguous.rs` |
