# Issue 053: DomainLatent Integration — Tests, ExpertRegistry, Burner Training

**Plan:** 038 (Free Transformer — Domain Latent Mid-Layer Injection)
**Status:** Partial — Task A complete, Tasks B & C open
**Depends on:** Plan 025 (Bidirectional Prefill + LoRA), Plan 050 (Feature Gate Audit)

## Background

Plan 038 implemented mid-layer domain latent injection (K/V modulation at layer L/2) behind the `domain_latent` feature gate. Core work is done:
- ✅ `DomainLatent` type with `load()`, `save()`, `zeros()`, `from_vec()` (`src/types.rs`)
- ✅ Mid-layer injection in `forward_base` + `forward_prefill` (`src/transformer.rs`)
- ✅ riir-gpu training: `GpuDomainLatent` + `DomainLatentAdamWStep` + export (`.dlat` binary)
- ✅ 10 tests total (5 type tests, 5 forward tests)

Three tasks remain — all now unblocked or partially unblocked.

---

## Task A: Integration Test — LoRA + DomainLatent [unblocked]

Both `forward_base` and `forward_prefill` accept `lora: Option<&LoraAdapter>` AND `domain_latent: Option<&DomainLatent>` simultaneously. No test exercises them together.

### What to do

Add test in `src/transformer.rs` (behind `#[cfg(feature = "domain_latent")]`):

1. `test_domain_latent_with_lora_changes_logits`
   - Load `LoraAdapter` from test data + create non-zero `DomainLatent`
   - Call `forward_with_domain_latent(ctx, weights, cache, token, pos, config, Some(lora), Some(dl))`
   - Assert output differs from `forward_with_domain_latent(..., Some(lora), None)`

2. `test_domain_latent_with_lora_prefill_pipeline`
   - Full pipeline: `forward_prefill` with both → `forward_base` decode with both
   - Assert tokens differ from lora-only pipeline

3. `test_domain_latent_zero_with_lora_same_as_lora_only`
   - Zero `DomainLatent` + LoRA should produce same output as LoRA alone
   - Verifies domain_latent is additive identity when zeros

### Existing test pattern to follow

```katgpt-rs/src/transformer.rs#L3293-3330
#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_changes_logits() { ... }
```

### Key references

- `LoraAdapter::load()` — binary format `[LORA 4B | version 4B | blake3 32B | payload...]`
- `DomainLatent::from_vec(vec![0.1f32; kv_dim])` — quick constructor for tests
- `forward_base` signature at `src/transformer.rs#L364-373`
- `forward_prefill` signature at `src/transformer.rs#L560-569`

---

## Task B: ExpertRegistry Integration [unblocked]

`ExpertRegistry` is fully implemented at `riir-ai/crates/riir-router/src/registry.rs` with 10+ tests. It resolves domain-specific bundles including LoRA paths. Domain latent needs the same treatment.

### What to do

1. **`DomainConfig`** — add field (`riir-ai/crates/riir-router/src/types.rs#L186-219`):
   ```rust
   #[serde(default)]
   pub domain_latent: Option<String>,  // path to .dlat file
   ```

2. **`ExpertBundle`** — add field (`riir-ai/crates/riir-router/src/types.rs#L114-125`):
   ```rust
   pub domain_latent: Option<DomainLatent>,
   ```
   Note: `DomainLatent` is in `katgpt-rs::types`, gated behind `domain_latent` feature. riir-router must add `katgpt-rs` as dep with `domain_latent` feature, OR re-export/abstract.

3. **`ExpertRegistry::resolve_domain_latent()`** — follow `resolve_lora_pair()` pattern (`riir-ai/crates/riir-router/src/registry.rs#L222-257`):
   ```rust
   fn resolve_domain_latent(domain: &DomainConfig, pruner_dir: &Path) -> Option<DomainLatent> {
       domain.domain_latent.as_ref().and_then(|p| {
           let path = pruner_dir.join(p);
           match DomainLatent::load(&path) {
               Ok(dl) => Some(dl),
               Err(e) => { eprintln!("..."); None }
           }
       })
   }
   ```

4. **Wire in `from_config()`** — call `resolve_domain_latent()` and store in `ExpertBundle`

5. **Thread to forward calls** — wherever `ExpertBundle` is used to call `forward()` / `forward_prefill()`, pass `bundle.domain_latent.as_ref()`

6. **Tests** — add domain with `domain_latent` field in test TOML config, verify load + graceful degradation on missing file

### Key references

- `DomainConfig` struct: `riir-ai/crates/riir-router/src/types.rs#L186-219`
- `ExpertBundle` struct: `riir-ai/crates/riir-router/src/types.rs#L114-125`
- `resolve_lora_pair()` pattern: `riir-ai/crates/riir-router/src/registry.rs#L222-257`
- `DomainLatent::load()`: `katgpt-rs/src/types.rs#L895-1005`

### Cross-crate dependency concern

`DomainLatent` lives in `katgpt-rs` behind `domain_latent` feature. `riir-router` would need either:
- Add `katgpt-rs` as optional dep with `domain_latent` feature, OR
- Define a `DomainLatent` trait/abstraction in riir-router and implement in katgpt-rs, OR
- Move `DomainLatent` to a shared crate

Decide on approach before implementing.

---

## Task C: riir-gpu DomainLatent Training for `train_lora.rs` [unblocked]

riir-gpu already has `DomainLatentAdamWStep` + `adamw_step_cpu` (pure CPU) and a working pattern in `train_bomber.rs` (Phase 5). The generic `train_lora.rs` example needs the same domain latent training capability added alongside its LoRA training loop.

### What exists (riir-gpu reference)

```riir-ai/crates/riir-gpu/src/domain_latent.rs#L137-149
pub struct DomainLatentAdamWStep<'a> {
    pub params: &'a mut [f32],
    pub grads: &'a [f32],
    pub m: &'a mut [f32],
    pub v: &'a mut [f32],
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,
}
```

Algorithm: standard AdamW with bias correction, pure CPU (`adamw_step_cpu`). Export produces `.dlat` binary (`DomainLatent::save()` format).

### What to do

1. Add `--domain-latent` flag to `train_lora.rs` CLI args — when set, train domain latent alongside LoRA
2. Reuse existing `DomainLatentAdamWStep` + `adamw_step_cpu` from `riir-gpu::domain_latent` (no porting needed — already works with `&mut [f32]`)
3. Follow `train_bomber.rs` Phase 5 pattern: extract features → `train_domain_latent_cpu()` → export `.dlat`
4. Export trained embedding via `DomainLatent::save()` (`.dlat` binary, same format as katgpt-rs expects)
5. Output: `lora.bin` + `domain_latent.dlat` side by side in output directory

### Approach

No new tensor API or external framework needed. `DomainLatentAdamWStep` is pure CPU with `&mut [f32]`, already tested and working. The pattern is proven in `train_bomber.rs` Phase 5 — adapt the same feature extraction + AdamW loop for generic JSONL data.

### Key references

- `DomainLatentAdamWStep` + `adamw_step_cpu`: `riir-ai/crates/riir-gpu/src/domain_latent.rs`
- Reference pattern (Phase 5 CPU training): `riir-ai/crates/riir-gpu/examples/train_bomber.rs#L299-340`
- Target file to modify: `riir-ai/crates/riir-gpu/examples/train_lora.rs`
- `Trainer` API: `riir-ai/crates/riir-gpu/src/trainer.rs`
- Binary format spec: `katgpt-rs/src/types.rs` — `[DLAT 4B][VERSION 1B][KV_DIM 4B LE][EMBEDDING kv_dim×f32 LE][BLAKE3 32B]`

---

## Task Summary

| Task | Scope | Status | Est. Effort |
|------|-------|--------|-------------|
| A: Integration tests | `katgpt-rs/src/transformer.rs` | ✅ Done | Small (3 tests) |
| B: ExpertRegistry + inference wiring | `riir-router/` + `transformer.rs` | ✅ Done | Medium |
| C: riir-gpu domain latent training | `riir-ai/crates/riir-gpu/examples/train_lora.rs` | Open | Medium (follow bomber pattern) |

**Status:** A ✅, B ✅, C ✅ — All tasks complete.

---

## Checklist

- [x] A1: `test_domain_latent_with_lora_changes_logits` ✅
- [x] A2: `test_domain_latent_with_lora_prefill_pipeline` ✅
- [x] A3: `test_domain_latent_zero_with_lora_same_as_lora_only` ✅
- [x] B1: Decide cross-crate dependency approach for `DomainLatent` in riir-router ✅ — direct dep with feature forwarding
- [x] B2: Add `domain_latent: Option<String>` to `DomainConfig` (with `#[serde(default)]`) ✅
- [x] B3: Add `domain_latent: Option<DomainLatent>` to `ExpertBundle` ✅
- [x] B4: Implement `resolve_domain_latent()` in `ExpertRegistry` ✅
- [x] B5: Wire `domain_latent` through to `forward()` / `forward_prefill()` / `generate_with_prefill()` call sites ✅
- [x] B6: Add tests (TOML config with domain_latent, graceful degradation) ✅ — 2 registry tests + generate_with_prefill test
- [x] C1: Add `--domain-latent` flag to `train_lora.rs` CLI args ✅ — `--domain-latent`, `--domain-latent-epochs`, `--domain-latent-lr`
- [x] C2: Wire domain latent training into `train_lora.rs` (follow `train_bomber.rs` Phase 5 pattern) ✅ — `prepare_domain_latent_features()` + `train_domain_latent_cpu()`
- [x] C3: Export `.dlat` binary alongside `lora.bin` via `write_domain_latent()` ✅ — `<stem>.dlat` next to `lora.bin`
- [x] C4: E2E test: write `.dlat` → load in katgpt-rs → verify round-trip ✅ — 7 tests in `riir-gpu/tests/test_domain_latent_e2e.rs`