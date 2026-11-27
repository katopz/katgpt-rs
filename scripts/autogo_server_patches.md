# AutoGo server patches — what the arena lane adds to the research clone

The AutoGo arena server (`scripts/autogo_server.sh` +
`scripts/autogo_server.Dockerfile`) runs the UPSTREAM clone at
`.raw/autogo` (research-local, gitignored; upstream:
`https://github.com/ericjang/autogo`, read at depth-1 2026-11-27). The
clone needs FOUR local patches before the headless server image works —
they live only in the clone (this repo never carries third-party code),
so re-clone = re-patch. This file is the exact list (plan-065 lineage;
written by the riir-train Plan 518 T3 lane, 2026-11-27).

## 1. `src/alpha_go/agents/gtp_agent.py` — NEW FILE

Registers the `gnugo1` agent the plan-065 docs and predictions table
always assumed ("Our Greedy vs their gnugo1 ~30%") but this upstream
revision lacks (registry ships only `random` + torch-needing NN agents).
GNU Go level 1 via a dedicated `GTPEngine`:

- `start_game` → `GTPEngine.new(size, level=1)`.
- `notify_move(row, col)` replays EVERY tracker move into its engine
  (pass arrives as `(None, None)` from the human path and `(-1, -1)`
  from the AI echo — normalize both).
- `select_move` = `engine.genmove()` then `engine.undo()` — genmove
  PLAYS on its own engine; the undo makes `notify_move`'s replay the
  single source of truth.
- `end_game` closes the process.
- Known edge (accepted, rare): a level-1 genmove RESIGN followed by a
  failing GTP undo drifts this engine's internal state; the tracker
  stays authoritative so game integrity holds.

## 2. `src/alpha_go/agents/__init__.py` — torch/grpc imports optional

`nn_agent` (torch) and `nn_mcts` (torch+grpc) imports wrapped in
`try/except ImportError`; the `nn_agent` registration import guarded on
`NNAgent is not None`; `gtp_agent` imported (registers `gnugo1`).

## 3. `src/alpha_go/__init__.py` — model import optional

The package `__init__` imports `alpha_go.model` (torch + mup)
unconditionally; wrapped in `try/except ImportError` so the slim server
image (no torch) can import the package at all.

## 4. `src/alpha_go/play.py` — `_cleanup_game` reaper

Finished games were never reaped (`games`/`game_agents` grow
unboundedly; each game holds a gnugo PROCESS). Measured 2026-11-27:
the container OOM-killed (exit 137) under a 400-game arena. Added
`_cleanup_game(game_id)` (close tracker engine, `agent.end_game()`,
drop all registry rows) called from `make_move` at BOTH terminal
returns ("already over" and "Game over!"). `pass_move` delegates to
`make_move` so it inherits the reap.

## The server image

`scripts/autogo_server.Dockerfile` (committed): `python:3.10-slim-bookworm`
+ `gnugo cmake build-essential`, a venv with only
`fastapi uvicorn[standard] numpy pydantic python-multipart`, builds
`alpha_go_cpp` via the repo's `scripts/build_cpp.sh`
(`UV_PROJECT_ENVIRONMENT=/app/.venv` so the script picks the venv
python), `PYTHONPATH=/app/src`, CMD `alpha_go.play` on :8000. The
repo's own `.devcontainer/Dockerfile` is a DEV image (bind-mounted
workspace + `postCreateCommand` install) and CANNOT serve from a plain
`docker run` — the headless lane needs this file.

Restart one-liner after a stop: `bash scripts/autogo_server.sh start`
(builds if the image is missing, ~4 min cold).
