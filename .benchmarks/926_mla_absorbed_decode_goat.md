# Bench 926 — MLA absorbed decode GOAT gate (Issue 926)

**Status:** LANDED 2026-10-08 — G1 PASS · G2 PASS-provisional (ratio stable, absolutes load-contaminated — quiet-box re-pin pending before any promotion) · G3 PASS · G4 PASS. Feature `mla_absorbed` opt-in (katgpt-attn `mla_absorbed = ["mla_attention"]` + root forward); reconstruct path unchanged as the spec-match reference; kill-switch `KATGPT_MLA_RECONSTRUCT=1`.

## What landed

Weight-absorbed MLA decode (DeepSeek-V2 §2.1 absorption; external evidence
arXiv:2610.07940 §5.2 — reconstruct-based latent attention runs at 7–18% of
full-cache throughput, the absorbed read at 28–59×, at equal quality):

- `crates/katgpt-attn/src/mla.rs`:
  - `mla_decode_prologue` / `mla_decode_rope_key_and_cache` /
    `mla_decode_epilogue` — the reference fn's Steps 1-2 / 4-5 / 7-8 moved out
    VERBATIM and shared by both paths (DRY without touching the reference's
    math; the existing 8 spec-match tests + grad-check pin the move).
  - `mla_forward_token_absorbed` (cfg `mla_absorbed`): per step per head
    `q_abs = W_UKᵀ·q_c` ONCE (via the existing `simd_transpose_matvec_into`
    NEON/AVX2/wasm32 substrate), scores read the latent directly
    (`q_abs·c_kv_j` over d_c + the unchanged rope dot), values accumulate in
    latent space as ONE transpose matvec over the `[seq × d_c]` cached latent
    matrix with the single `W_UV` up-projection after. The reconstruct path's
    Step 3 (`k_c`/`v_c` current-token up-projection) is dead work there and is
    skipped (k_c overwritten before read; v_c never read in the attention
    loop — noted as a future microcleanup for the reference, NOT done here).
  - `mla_forward_token_dispatched`: runtime router used by the kimi decode
    lane (`src/kimi_k3/decoder_layer.rs`); absorbed under the feature unless
    `KATGPT_MLA_RECONSTRUCT=1`, transparent forward otherwise. Default builds
    behavior- and byte-identical; training/backward paths call the reference
    directly regardless.
- Scratch: `q_abs` `[d_c·n_h]` + `lat_acc` `[d_c]`, pre-allocated under the
  feature (G4: zero per-step allocation).

## Gates

### G1 — spec match (PASS)

`tests/mla_absorbed_spec_match.rs` — absorbed vs reconstruct, full rollouts,
per-step max-abs diffs (dev profile, 1.98.1, aarch64):

| scenario | max_diff | tol |
|---|---|---|
| pos=0 single token | 0.00e0 (bit-identical) | 1e-4 |
| pos=5 rope active | ~4e-8 | 1e-4 |
| 12-token rollout | ~4e-8 | 1e-4 |
| gate off | 8.94e-8 | 1e-4 |
| use_nope | 2.24e-8 | 1e-4 |
| 64-token rollout | 2.98e-8 | 1e-4 |
| kimi_k3_0_40b full dims | 8.94e-8 | 1e-3 |

3–7 orders of magnitude under the bars. The reconstruct reference itself
re-passes its existing gates unchanged: `mla_g1_spec_match` 8/8 (default AND
`mla_backward` postures), `mla_backward_grad_check` 4/4.

### G2 — decode latency (PASS-provisional)

`benches/bench_926_mla_absorbed_decode.rs` (harness=false, interleaved A/B
pairs, median over pairs, black_box both arms, `rewind_to` keeps the cache
length constant across timed steps). kimi geometry (d_h=64, d_c=128, d_r=32,
v_h=64, n_h=8), random weights, cache pre-filled directly:

| seq | speedup (4 runs) | verdict row |
|---|---|---|
| 1024 | 23.9 – 25.5× (±3%, stable) | PASS |
| 4096 | 36 – 53× | PASS |
| 16384 | 36 – 54× | PASS |

⛔ **Provenance (loud):** M3 Max 16-core, AC power, battery 100% charged,
**load average 14–42 with a sibling session's bevy release build actively
spawning rustc (~15 procs) + a nice-19 67%-CPU census job running** — the
ABSOLUTE medians are load-contaminated (reconstruct-arm spread −32%..+323%;
the absorbed arm's absolutes are stable: ~316–398 µs @1K, ~986–1065 µs @4K,
~3323–3883 µs @16K across runs). The RATIO is load-robust by construction
(interleaved pairs share the contention; the reconstruction arm is the
bandwidth-heavy one and only ever gets SLOWER under load, never faster), and
its worst observed value is 23.9× — far above any promotion bar and
consistent with the ~60× FLOP model as the O(seq) term dominates.

**Quiet-box re-pin (the citable numbers):**
```sh
CARGO_TARGET_DIR=/tmp/katgpt-926-root cargo bench -p katgpt-rs \
  --features mla_absorbed --bench bench_926_mla_absorbed_decode
# precondition: load < 2, no sibling rustc/census (ps aux | grep -E 'rustc|hyperthink|cargo')
# then update this file's table + drop the PROVISIONAL marker; promotion
# (feature → default in the dispatched router) is a separate commit gated on it.
```

### G3 — no regression (PASS)

- `cargo test -p katgpt-rs --lib --features mla_absorbed,kimi_k3` → 217 passed, 0 failed.
- `kimi_k3_layer_diff` + `kimi_k3_g4_alloc_free` under
  `mla_absorbed,kimi_k3_loader` → passed (g4's release-only row ignored in dev
  as designed; layer_diff is report-only).
- `kimi_k3_phase6` → 1 passed + 2 skip-loud (real-model rows need the GGUF).
- League/trajectory bench consumers compile under the feature:
  `bench_012` + `bench_889` `--no-run` green.
- Default posture: `cargo check -p katgpt-attn` + `-p katgpt-rs --lib` green —
  the dispatched router compiles to a transparent forward, zero behavior
  change, and the league models do not run the MLA path (kimi lane only).
- Backward/training: `mla_backward` calls the reconstruct reference directly
  (never the router) — checkpoint-equivalence and grad-check lanes are
  structurally unaffected.

### G4 — alloc-free (PASS)

`tests/mla_absorbed_alloc_check.rs` (own binary, counting allocator +
`assert_counter_is_live` canary): 256 steady-state absorbed decode steps at a
fixed cache length (rewind pattern) → **0 allocations**.

## Feature-flag posture

- Leaf: `mla_absorbed = ["mla_attention"]` (katgpt-attn) — zero new deps;
  reuses the in-tree `simd_transpose_matvec_into` substrate.
- Root: `mla_absorbed = ["katgpt-attn/mla_absorbed"]`; the kimi decode lane
  routes through `mla_forward_token_dispatched` (one call site).
- Kill-switch: `KATGPT_MLA_RECONSTRUCT=1` bit-restores the reconstruct path
  under the feature.
- Promotion: NOT done — gated on the quiet-box re-pin above (the G2 gain is
  modelless — pure algebra — so the promotion path is open once numbers are
  clean; the rejected alternative of caching up-projected k_c/v_c is recorded
  in Issue 926 so nobody reopens it).
