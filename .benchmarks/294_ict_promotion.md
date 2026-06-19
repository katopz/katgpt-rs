# Plan 294 — `ict_branching` Promotion Decision (T8.4)

**Date:** 2026-06-19
**Status:** Opt-in (deferred pending G8)

## Decision

**`ict_branching` stays opt-in (`default = [...]` does NOT include it).**

Per Plan 294 §Phase 8 T8.4, default-on requires **both**:

1. **G3 (this plan, Phase 4)**: Spearman ρ(H₁, JS-uniqueness) < 0.5.
   ✅ PASS — ρ = 0.0652, 95% CI [-0.017, 0.150].
2. **G8 (riir-ai Plan 324)**: runtime fusion validated on real NPCs.
   ⏳ **OUT OF SCOPE** — Plan 324 owns the runtime fusion (CLR gating at
   branching moments, HLA updates at branching moments, KG emission,
   curiosity bursts, freeze/thaw snapshots). That validation requires real
   game workloads and is the riir-ai team's responsibility.

G3 passing means the open primitive's *information content* is correct —
JS-uniqueness is genuinely orthogonal to H₁. But "information content is
correct" is not the same as "default-on is safe". Default-on means every
downstream consumer of `katgpt-rs` pays the (small) compile-time cost and
the (1.96µs) runtime cost whether they need the ICT selector or not. We
only default-on features that have a clear broad audience.

The ICT selector's broad audience is NPC runtimes — and NPC runtime
validation lives in riir-ai Plan 324. Until that plan reports G8 PASS,
the feature stays opt-in.

## What ships regardless of promotion

| Component | Value | Audience |
|-----------|-------|----------|
| `collision_purity(π) = Σ π²` | Drop-in replacement for `shannon_h1` anywhere we use entropy as a concentration signal. ICT §A.2.5 proves it's unconditionally monotone. | Any entropy-driven gate (`llmexec_guard`, `AdaptiveTraceCompactor`, etc.) |
| `renyi_h2(π) = −log β` | H₂ entropy — the right concentration metric per ICT §A.3.3. | Same as above. |
| `js_divergence(p, q, scratch)` | Symmetric, bounded, finite-on-disjoint distributional-novelty metric. | Novelty filters (IdeaDivergence, etc.). |
| `AcceptanceForecastH2` | Bebop H₁→H₂ drop-in (G10 PASS, MAE 0.402<0.423 on long-tail). | Bebop Issue 023 (Plan 243). |
| `BranchingDetector` | The full ICT runtime selector (G3/G4/G5/G6 all PASS). | riir-ai Plan 324 (pending G8). |
| Curiosity Pulse spec | Reference doc only — implementation in riir-ai Plan 274. | riir-ai Plan 274 / Plan 187. |

## How callers opt in

```toml
# In a downstream Cargo.toml:
[dependencies]
katgpt-rs = { version = "...", features = ["ict_branching"] }
```

Or for katgpt-core direct consumers:

```toml
[dependencies]
katgpt-core = { version = "...", features = ["ict_branching"] }
```

The feature is purely additive — no other feature depends on it, and
`cargo build --no-default-features` succeeds without it (verified by G6).

## Re-promotion criteria

`ict_branching` will be promoted to default-on when **all** of:

1. riir-ai Plan 324 reports G8 PASS (runtime fusion validated on real NPCs).
2. The `k_percent` sweep recommended by [Issue 033](../.issues/033_ict_g2_inflection_37_percent_npc_domain.md)
   is confirmed on real game data (paper's 10% may not survive the NPC
   domain transfer — Issue 033 measured 37.5% on synthetic data).
3. The Bebop H₁→H₂ upgrade (G10) has been re-calibrated on production
   acceptance-length data (synthetic G10 proves direction; magnitude needs
   real data).

Until then, opt-in is the conservative choice — anyone who needs the
primitive can enable it with one feature flag.

## References

- Plan 294 §Phase 8 T8.4
- [G3 benchmark doc](294_ict_g3.md) — the make-or-break PASS
- [Issue 033](../.issues/033_ict_g2_inflection_37_percent_npc_domain.md) — k_percent sweep follow-up
- riir-ai Plan 324 — runtime fusion (G7–G9, G8 specifically)
