# Issue 016: newton_schulz `ns_inv_sqrt_psd_into` Borrow Violation

**Date:** 2026-06-14
**Severity:** 🔴 Build-blocking (default features)
**Discovered by:** CHIAR Plan 269 integration attempt
**Plan origin:** Plan 270 WIP (`src/newton_schulz.rs` modifications)

## Problem

`cargo build` fails with default features:

```
error[E0502]: cannot borrow `scratch.p_sq` as mutable because it is also borrowed as immutable
   --> src/newton_schulz.rs:466:41
    |
462 |             &scratch.p_sq[..rr]
    |              ------------ immutable borrow occurs here
...
466 |         matmul_symmetric(p_cur, r, &mut scratch.p_sq[..rr]);
    |                                        ^^^^^^^^^^^ mutable borrow occurs here
```

The ping-pong pattern in `ns_inv_sqrt_psd_into` (Plan 270 WIP) reads `p_cur` from either `scratch.p_norm` or `scratch.p_sq`. On iterations where `p_cur` aliases `scratch.p_sq`, the subsequent `matmul_symmetric` write violates Rust's aliasing rules.

## Repro

```sh
cargo build --lib
# or
cargo build --lib --features newton_schulz
```

## Workaround

Build with `--no-default-features --features <your-feature>` (e.g., `chiaroscuro`). This excludes `newton_schulz` from the build.

## Fix

The fix is straightforward — snapshot `p_cur` to an owned `Vec<f32>` when it would alias the destination buffer, OR restructure the ping-pong to avoid same-buffer read-then-write.

Suggested fix at `src/newton_schulz.rs:459-466`:

```rust
let p_cur_owned: Vec<f32> = if p_cur_is_norm {
    scratch.p_norm[..rr].to_vec()
} else {
    scratch.p_sq[..rr].to_vec()
};
let p_cur = &p_cur_owned[..];
matmul_symmetric(p_cur, r, &mut scratch.p_sq[..rr]);
```

This allocates a snapshot on alternating iterations (cheap relative to the matmul). A more efficient fix would use a third buffer in the scratch struct.

## Impact

- Blocks `cargo build` with default features
- Blocks `cargo test --lib` with default features
- Affects any feature that depends on `newton_schulz` (e.g., `newton_schulz`, `parallax_attn`)
- **Does NOT affect** feature-isolated work like `chiaroscuro` (verified)

## Related

- Plan 270 (LoRA-Muon distillation) introduced `ns_inv_sqrt_psd_into`
- Plan 269 (CHIAR) discovered the bug while attempting InferenceRouter integration
