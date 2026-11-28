# Issue 1253 — katgpt-attn: `mla_backward_token_with_cache_grads` — export the per-position KV-cache-entry gradients

**Claimed:** zed-300be342@4090, 1795872397, expires 2026-12-04T00:00:00+07:00

Status: OPEN — consumer-gated training-lane export (riir-train Issue 618 T0.b is the named first consumer). Additive out-param variant over `mla_backward_token` (the `kimi_k3_backward_sequence` → `_with_input_grad` precedent verbatim); zero behavior change when the slots are absent.

## Why (the consumer's blocking prerequisite)

riir-train Issue 618 (LPT shared-KV arm) T0 requires the cross-iteration KV-cache gradient: `ouro_looped_forward` rewinds + re-appends MLA cache entries per iteration (rewind_to(pos) contract, katgpt-attn `mla.rs` L331-348), so the looped backward must route `dL/d(cache entry)` back into every writer iteration's parameter chain. Today `mla_backward_token` computes exactly these quantities internally and folds them away:

- `d_c_kv_normed_all[j]` (mla_backward.rs L480, accumulated at L526-532 via W_UV^T·d_v and L584-590 via W_UK^T·d_k) — the gradient w.r.t. each cached latent `c_kv_normed_j`, for ALL `j ∈ 0..seq` (including `j == pos`, whose saved copy equals the cache entry by the debug-asserted identity at L167-186).
- `d_k_r_all[j·d_r..]` (L477, accumulated at L596-599) — the gradient w.r.t. each cached (post-RoPE) rope key, consumed at L668-679 by inverse-RoPE into `w_kr`/`all_dh`.

Both are consumed pre-export (rmsnorm_backward → `w_dkv` + `all_dh[j]`; inverse RoPE → `w_kr` + `all_dh[j]`), so the recompute alternative in the consumer would duplicate the whole MLA backward — a parallel implementation, refused per the substrate-first law. The export is ~10 lines of slot accumulation beside the existing loops.

## Spec

`pub fn mla_backward_token_with_cache_grads(..., d_cache_latent: Option<&mut [f32]>, d_cache_rope: Option<&mut [f32]>)` — same body as `mla_backward_token`; when armed, before the rmsnorm/inverse-RoPE consumption:

- `d_cache_latent[j*d_c..(j+1)*d_c] += d_c_kv_normed_all[j]` for all `j ∈ 0..seq` (the pre-rmsnorm value = dL/d(cache latent entry)).
- `d_cache_rope[j*d_r..(j+1)*d_r] += d_k_r_all[j*d_r..]` (the pre-inverse-RoPE value = dL/d(cache rope entry)).

`mla_backward_token` delegates with `None`/`None` — bit-identical incumbent. Slice-length asserts name `max_seq` sizing; a writer-side census (2026-11-29): the training surface's ONLY cache-append seam is `mla_decode_rope_key_and_cache` L585 (shared decode midfix, reconstruct + absorbed); `mla_chunk_prologue_stage` L1008 is prefill-only and out of the looped-training surface, so the decode-path export covers the consumer fully.

## Acceptance

1. katgpt-attn unit gate: a toy MLA arm (the riir-train toy dims — kv 8 / q 8 / 2 heads / nope) reads nonzero, correctly-shaped slot gradients; the `None` posture is bit-identical to the incumbent (same grads, same all_dh).
2. Consumer-side FD gate (riir-train plan 721 T0.e): central differences on the toy arm — perturb `cache.latent_kv[j]` / `cache.rope_key[j]` directly, assert the exported gradient matches within the rel/abs band, and assert it is nonzero for j < pos (the detach-collapse counter-proof).
3. Existing batteries untouched-green: mla spec-match pair + the riir-train `issue614_mla_recompute_parity` (5/5) + kimi_k3_ouro 49/49.

## Boundary

katgpt-rs owns the backward math (this issue); riir-train owns the ouro compose (writer-mapping: entry `j < pos` routes to iteration 0's chain, entry `pos` routes to ALL iterations — each writer's value is `rmsnorm(W_DKV·h_pos^{(t)})`, so `dL/d(entry pos)` adds an input-gradient row to every iteration's `h_pos`). Design of record: riir-train `.plans/721_issue618_t0_cross_iteration_backward.md`.

## Refs

- riir-train Issue 618 + `.plans/721_issue618_t0_cross_iteration_backward.md` (the consumer + design)
- katgpt-attn `mla_backward.rs` L371-700 (`mla_backward_token`), L167-186 (the saved≡cache identity)
- katgpt-rs `src/kimi_k3/backward.rs` L847+ (the `_with_input_grad` naming precedent)
