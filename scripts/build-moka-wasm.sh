#!/usr/bin/env bash
# scripts/build-moka-wasm.sh — build the katgpt-moka-wasm binary with SIMD enabled.
#
# WHY THIS SCRIPT EXISTS (the lesson):
#   `wasm32-unknown-unknown` defaults to NO SIMD. The crate's hot path
#   (`simd_dot_f32`) is gated on `target_feature = "simd128"` — without the
#   RUSTFLAGS below, the build silently runs the SCALAR FALLBACK, which is
#   ~16× SLOWER (500 ms/move vs 30 ms/move at PUCT budget=50). This was
#   diagnosed the hard way in Issue 205 (time spent debugging "correct" code
#   before realizing the toolchain had dropped the flag). This script encodes
#   the requirement so the regression cannot recur silently.
#
#   ⛔ THIS SCRIPT IS A BUILD, NOT A GATE (Issue 737). Until 2026-09-07 it was
#   the ONLY thing in this repo that ever compiled for wasm32, so the browser
#   crate's lint surface was checked exactly as often as someone happened to
#   deploy — which turned out to be 15 findings' worth of "not often enough",
#   11 of them `unsafe_op_in_unsafe_fn` on edition 2024. The gate is
#   `scripts/full_gate.sh` layer 2b (both simd128 arms, derived package list).
#   Do not treat a green run of this script as a lint claim.
#
#   The matching `wasm-opt --enable-simd` is required at the optimize step:
#   without it, wasm-opt strips the SIMD instructions it doesn't know about.
#
# Pipeline:
#   1. RUSTFLAGS='-C target-feature=+simd128' cargo build --release --target wasm32-unknown-unknown
#   2. wasm-bindgen --target <nodejs|web> → JS glue + raw _bg.wasm
#   3. wasm-opt -Oz --enable-simd → optimized .opt.wasm (SIMD preserved)
#
# Output: crates/katgpt-moka-wasm/target-wasm/<nodejs|web>/
#
# Usage:
#   ./scripts/build-moka-wasm.sh             # nodejs target (for bench harness)
#   ./scripts/build-moka-wasm.sh --web       # web target (for browser deployment)
#   ./scripts/build-moka-wasm.sh --skip-opt  # skip wasm-opt (faster, larger binary)
#
# After building (nodejs target), measure latency:
#   node crates/katgpt-moka-wasm/bench/bench_puct.js
set -euo pipefail

BINDGEN_TARGET="nodejs"
SKIP_OPT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --web)        BINDGEN_TARGET="web"; shift ;;
    --nodejs)     BINDGEN_TARGET="nodejs"; shift ;;
    --skip-opt)   SKIP_OPT=true; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Resolve project root + crate paths from this script's location (works from
# any CWD — mirrors scripts/release.sh's pattern).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && git rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/..")"
CRATE_DIR="$PROJECT_ROOT/crates/katgpt-moka-wasm"
OUT_DIR="$CRATE_DIR/target-wasm/$BINDGEN_TARGET"

# ── Preflight: verify the wasm toolchain is installed ──────────────────
command -v wasm-bindgen >/dev/null 2>&1 || {
  echo "error: wasm-bindgen not found. Install: cargo install wasm-bindgen-cli" >&2
  exit 1
}
if [[ "$SKIP_OPT" == "false" ]]; then
  command -v wasm-opt >/dev/null 2>&1 || {
    echo "error: wasm-opt not found. Install: brew install binaryen" >&2
    exit 1
  }
fi
rustup target list --installed 2>/dev/null | grep -q 'wasm32-unknown-unknown' || {
  echo "error: wasm32-unknown-unknown target not installed." >&2
  echo "  run: rustup target add wasm32-unknown-unknown" >&2
  exit 1
}

echo "→ building katgpt-moka-wasm (SIMD ON, target=$BINDGEN_TARGET)..."
echo "  RUSTFLAGS='-C target-feature=+simd128'  (REQUIRED — without it, ~16× slower scalar fallback)"

# ── Step 1: cargo build with SIMD flag ─────────────────────────────────
RUSTFLAGS='-C target-feature=+simd128' \
  cargo build --release --target wasm32-unknown-unknown -p katgpt-moka-wasm

WASM_SRC="target/wasm32-unknown-unknown/release/katgpt_moka_wasm.wasm"
[[ -f "$WASM_SRC" ]] || { echo "error: build output not found at $WASM_SRC" >&2; exit 1; }

# ── Step 2: wasm-bindgen ───────────────────────────────────────────────
echo "→ wasm-bindgen --target $BINDGEN_TARGET..."
mkdir -p "$OUT_DIR"
wasm-bindgen --target "$BINDGEN_TARGET" --out-dir "$OUT_DIR" --out-name katgpt_moka_wasm "$WASM_SRC"

# __raw_wasm export: the bench harness reads the C-ABI exports directly via
# `moka.__raw_wasm`. wasm-bindgen only exposes this when the wasm has a
# `#[wasm_bindgen]` wrapper that references it. The crate's `WasmGame` type
# pulls it in, so the JS glue exposes `__raw_wasm` for the nodejs target.
# (If a future refactor removes WasmGame, the bench harness needs a different
#  access pattern — but the C-ABI exports themselves are stable.)

# Expose the raw wasm instance exports as `__raw_wasm` for the nodejs target.
# The bench harness (bench/bench_puct.js) calls the C-ABI exports
# (wasmi_arena_init/play/search_puct) directly via `moka.__raw_wasm` —
# wasm-bindgen wraps the high-level API (WasmGame etc.) but the raw arena
# exports are only reachable through the instance. The web target skips
# this (browser deployment wraps everything through wasm-bindgen).
if [[ "$BINDGEN_TARGET" == "nodejs" ]]; then
  echo 'module.exports.__raw_wasm = wasm;' >> "$OUT_DIR/katgpt_moka_wasm.js"
fi

# ── Step 3: wasm-opt with --enable-simd (REQUIRED to preserve SIMD) ─────
if [[ "$SKIP_OPT" == "false" ]]; then
  echo "→ wasm-opt -Oz --enable-simd  (REQUIRED — without --enable-simd, SIMD ops are stripped)..."
  RAW_WASM="$OUT_DIR/katgpt_moka_wasm_bg.wasm"
  OPT_WASM="$OUT_DIR/katgpt_moka_wasm_bg.opt.wasm"
  wasm-opt -Oz --enable-simd "$RAW_WASM" -o "$OPT_WASM"
  # Overwrite the raw wasm with the optimized one so the JS glue picks it up
  # automatically (the JS glue loads `katgpt_moka_wasm_bg.wasm` by name).
  mv "$OPT_WASM" "$RAW_WASM"
  echo "  optimized: $(du -h "$RAW_WASM" | cut -f1)"
fi

# ── Done ───────────────────────────────────────────────────────────────
echo ""
echo "✓ build complete → $OUT_DIR/"
ls -1 "$OUT_DIR/" | sed 's/^/    /'
echo ""

if [[ "$BINDGEN_TARGET" == "nodejs" ]]; then
  echo "To measure PUCT latency (V8 JIT, same engine as Chrome):"
  echo "  node $CRATE_DIR/bench/bench_puct.js"
  echo ""
  echo "Expected at PUCT budget=50, K=1 (sequential): ~30 ms/move."
  echo "If you see ~500 ms/move, the SIMD flag was lost — rebuild with this script."
fi
