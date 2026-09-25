# Inference — The Speculative Decoding + Search Engine

> **What we sell here.** The core inference path: speculative drafting +
> verification, multi-hop speculation, KV-cache compression, multi-token
> prediction thresholds, and graph search. Everything that accelerates or
> extends the single-pass autoregressive decode.

## Fusion map — how the pieces compose

```
   speculative_decoding.md (DDTree + DFlash + Leviathan verify)
        │
        ├── spechop.md (continuous multi-hop on top of the draft tree)
        ├── mtp_threshold.md (when to trust a multi-token-prediction draft)
        └── progressive_mcgs.md (graph search w/ reference edges)
              │
              ▼
        kv_compression.md (the cache the above drafts read from / write to)
```

| Doc | Role |
|---|---|
| [`speculative_decoding.md`](speculative_decoding.md) | DDTree marginal-distribution trees, DFlash fast marginal prediction, Leviathan verification, D2F discrete-diffusion forcing |
| [`spechop.md`](spechop.md) | SpecHop — continuous multi-hop speculation pipeline (Plan 131, feature `spechop`) |
| [`mtp_threshold.md`](mtp_threshold.md) | MTP threshold guide — when multi-token-prediction drafts are worth accepting (Plan 055 + Plan 117) |
| [`kv_compression.md`](kv_compression.md) | KV cache compression research & alternatives (TurboQuant → SpectralQuant → OCTOPUS → KVarN), plus the fitted token-value tables (§7: mean-removed V quant, K=V+, V-cache halving, Issue 883 / Bench 895) |
| [`progressive_mcgs.md`](progressive_mcgs.md) | Progressive MCGS — Monte Carlo graph search with reference edges |
| [`spectral_pencil.md`](spectral_pencil.md) | The affine matrix pencil scalar gate `f(x)=λk(A₀+ΣxᵢAᵢ)` — shape-by-construction, exact attribution, γk≥½ seeded init (Issue 676, Research 495) |

## See also

- [`../01_orientation/architecture.md`](../01_orientation/architecture.md) — where each inference primitive plugs into the core pipeline
- [`../09_feature_catalog/opt_in_features.md`](../09_feature_catalog/opt_in_features.md) — the opt-in feature-flag reference
- [Research 442 (LOTUS)](../../.research/442_LOTUS_Looped_Parallel_CoT_Supervision_PASS.md) — **PASS** verdict validating the `lt2_looped` architecture: LOTUS's looped padded Transformer + per-position latent-block supervision is the training-recipe counterpart to our shipped weight-shared T-pass loop. Architecture ships; supervision recipe → riir-train. Cherry-pickable gains (Any-Time property, config-default trap) captured + closed (Issue 156).
