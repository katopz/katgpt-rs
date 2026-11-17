---
name: disk-clean
description: Audit and reclaim disk space across the multi-repo workspace, home caches, and SDXC1TB. Scans cargo target dirs, incremental caches, GGUF model dirs, Unity Library caches, rustup and cargo registries. Classifies findings into regenerable-safe, load-bearing, and irreplaceable — the regenerable-safe class is deleted autonomously (before/after measured and reported), the load-bearing and irreplaceable classes are presented as a list for user approval and deleted only per approved item. Use when disk is full or low, before large builds or GPU training runs, or when asked what can be deleted or cleaned.
---

# Disk Clean

Reclaim disk space on the workstation. The contract is **two-tier**: SAFE-class regenerable caches are deleted AUTONOMOUSLY (measure before/after, verify, report — approval was removed by owner decree 2026-11-17 after amo's 11 GB ENOSPC trough proved waiting on approval for regenerable caches breaks the box); LOAD-BEARING and IRREPLACEABLE classes stay **list first, delete only what the user approves per item**. Never delete anything classified LOAD-BEARING or IRREPLACEABLE without explicit per-item confirmation.

## When to use

- Disk full / low-space (builds failing with ENOSPC, macOS low-disk warnings)
- User asks "what can we delete", "clean disk", "free space", "disk full"
- Proactively before large builds, GPU training runs, or model downloads

## Phase 1 — Assess

All read-only. Record the `df` numbers first — they are the before/after proof.

```bash
df -h /                                             # internal disk state
df -h /Volumes/SDXC1TB 2>/dev/null                  # SD card (may be unmounted)
du -sh /Users/katopz/git/*/target 2>/dev/null | sort -rh            # build caches (~1 TB when dirty)
du -sh /Users/katopz/git/*/target/debug/incremental 2>/dev/null | sort -rh  # the safest slice
du -sh /Users/katopz/git/*/data 2>/dev/null | sort -rh            # model/GGUF dirs (only repos that have one answer)
du -sh ~/.cargo/registry ~/.cargo/git ~/.rustup 2>/dev/null
du -sh /Volumes/SDXC1TB/* /Volumes/SDXC1TB/git/* 2>/dev/null | sort -rh | head -20
```

Notes:

- `du` on large trees takes minutes — generous timeouts (600000 ms+). On the
  exFAT SD card, much longer.
- A line that does not appear = the dir is absent (e.g. riir-ai has no
  incremental dir). Do not re-run to "fix" a missing line.
- `du` on `~` broadly is too slow — scan the known hot paths above instead.
- macOS reports 3 partitions on `/` (system/volume/data) — read `Avail` from
  `df -h /` only.

## Phase 2 — Classify

Every finding goes in exactly one bucket. Bucket decides presentation order
and wording, never silently decides deletion.

### SAFE — regenerable caches (the AUTONOMOUS-DELETE class — no approval gate)

| Item | Typical size when dirty | Regenerates how |
|---|---|---|
| `*/target/debug/incremental` | 100–350G total | next `cargo build` (slower first build only) |
| `*/target/debug` (whole dir) | 100–300G per repo | full cold debug rebuild (10–30 min/repo on M3) |
| `*/target/wasm32-unknown-unknown` | 5–20G | next wasm build |
| Unity `Library/` | 30–50G | Unity re-imports on project open |
| `~/.cargo/registry` + `~/.cargo/git` | 5–10G | re-downloads on next build |
| `~/.rustup` stale toolchains | 5–20G | `rustup toolchain list` then uninstall old |
| `*.bak`, `*.42-header.bak` in data dirs | small | gone (backup files) |

The single best zero-risk pick is usually **all incremental dirs + the biggest
repo's debug dir**. Measured precedent (2026-09-13): that freed ~480G internal
(41Gi → 520Gi free).

### LOAD-BEARING — do not break these

- **`riir-refine/target/release/cargo-refine`** — `~/.cargo/bin/cargo-refine`
  symlinks here. NEVER `rm -rf` that repo's whole `target/`. Delete
  `target/debug` only. After touching that repo's target at all, verify
  `~/.cargo/bin/cargo-refine` still resolves and runs.
- **`target/release` dirs in every repo** — keep; they hold warm release
  builds, benches, and installed binaries.
- **Running processes** — before deleting, run
  `ps aux | grep -E "cargo|rustc" | grep -v grep`. A
  `cargo run --release -p X` process means X's release dir is IN USE (its
  debug/incremental dirs are still safe to delete). Deleting a target dir
  under a live cargo breaks that build.
- **`/Volumes/SDXC1TB/git/zed`** — the ONLY copy of the user's patched Zed
  source (they build Zed from it). Its `target/` (359G measured) is deletable
  cache; the source never is. Warn about the ~1h+ cold rebuild.
- **`/Volumes/SDXC1TB/tmp`** — the sanctioned `CARGO_TARGET_DIR` overflow
  (global AGENTS.md rule). A sibling agent may have an active build there —
  check for running cargo before touching.

### IRREPLACEABLE — never delete without explicit per-item user confirmation

- **GGUF models** in `riir-train/data/` — multi-GB downloads, slow to replace.
  Priority keep list (AGENTS.md): `Ternary-Bonsai-27B-Q2_0.gguf` (THE priority
  model — never suggest deleting), `gemma-2-2b-it-f16.gguf`,
  `MiniCPM5-1B-F16.gguf`, `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (perf-rematch
  league). Everything else is a candidate — but the USER decides each one.
- **SD card data — mostly only-copies**: `seal-m/` (~109G SealM PC-rip +
  original + mobile — the asset source), `seal/` (~66G legacy Seal raw +
  extracted — the editor's import source), `seal-game-editor/content` (~43G
  snapshot), `SealM on CROSS*.zip` (~9G).
- **Unity `Assets/`** — source art. Only `Library/` is cache.
- **`katgpt-rs/data/kimi-k3-0.40b`** — small (1.5G) and "dumb/test-tier" per
  owner, but still a model — user confirms.
- `.Trashes` on SD is not shell-accessible — tell the user to empty via
  Finder if it matters.

## Phase 3 — Present the list, execute the SAFE class

Present a markdown table (path, size, bucket, one-line note) sorted by size, then name the "max space, zero risk" pick explicitly with its expected freed total.

- **SAFE rows execute autonomously** — no approval wait, no `ask_user` for them. Record the before/after `df` pair and the per-path deletions in the run report (the autonomous act is trusted BECAUSE it is measured and reported). The running-process re-check (Phase 4 step 1) still gates every deletion.
- **LOAD-BEARING and IRREPLACEABLE rows** go to the user with the `ask_user` tool and concrete options (e.g. "clean both", "X only", "nothing") — delete only per approved item.
- List-only requests end here (present, do not execute even the SAFE class when the user explicitly asked for a list).

## Phase 4 — Execute (SAFE rows autonomously; gated rows only after explicit approval)

1. Re-check `ps aux | grep -E "cargo|rustc" | grep -v grep` immediately
   before deleting — sibling agents start builds concurrently.
2. Delete with explicit FULL paths — no shell globbing across repos, no
   `$VAR` interpolation. One `rm -rf` per item or group; for gated rows the
   item list must equal the approved list exactly (no additions).
3. Timeouts: APFS metadata-heavy deletions take minutes
   (1200000 ms); exFAT/SD with many small files can take much longer
   (1800000 ms).
4. Parallel `rm -rf` calls on disjoint trees are safe and faster.

## Phase 5 — Verify + report

```bash
df -h / ; df -h /Volumes/SDXC1TB
du -sh <each touched repo>/target          # spot-check after sizes
~/.cargo/bin/cargo-refine --version 2>/dev/null || ls -la ~/.cargo/bin/cargo-refine  # if riir-refine target touched
```

Report a before/after table (disk free before → after; per-path before →
after). State the cold-rebuild cost honestly: which repos now build cold and
roughly how long the first rebuild takes. Nothing here touches git-tracked
files — there is nothing to commit (target/ and caches are gitignored).

## Offload alternative

When the internal disk is full but the data must be kept (models, datasets),
MOVE to `/Volumes/SDXC1TB/tmp` (the sanctioned overflow) instead of deleting —
check SD free space first. Also relevant when a sibling cargo build holds the
local `target/` lock: `CARGO_TARGET_DIR=/Volumes/SDXC1TB/tmp/...` per the
global AGENTS.md rule.
