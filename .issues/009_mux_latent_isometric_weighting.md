# MUX-Latent Isometric Weighting

**Source**: Plan 257 (GPart Isometric Adapter) — Deferred Idea 3
**Priority**: Low
**Blocked**: Yes — blocked on Plan 238 (MUX-Latent)
**Depends**: Plan 238, Plan 257

## Summary
Use GPart's isometric partition matrix as the weighting mechanism in the MUX-Latent demux pipeline. Replace learned linear weights with partition-based scaling for reduced parameter count.

## Acceptance Criteria
- [-] Plan 238 MUX-Latent must be complete first (blocked on Plan 238)
- [-] Design isometric weight injection point in demux pipeline (blocked on Plan 238)
- [-] Benchmark vs. standard MUX-Latent weights (blocked on Plan 238)
- [-] GOAT gate behind `mux_isometric` feature flag (blocked on Plan 238)

## Notes
- Blocked until Plan 238 provides the MUX-Latent infrastructure
