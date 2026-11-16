#!/usr/bin/env python3
"""Issue 1225 T1b: emit the Arm A forward fixtures from the pinned Python source.

Runs the REAL upstream forward (brianbell-x/codehygiene, PIN
423c7e1f02ef409bbaea209268bfcde36e0f1cb0) over five embedded snippets and
writes one full-precision JSON pin per snippet plus an index + pins file.
These are the oracle the Rust forward port (Arm A) replays under the
pre-registered G1-A band (issue 1225): sigmoid heads + line probs at
|dp| <= 1e-4 ABSOLUTE; the ordinal + style softmaxes banded as
reproduction-only side outputs; the byte grid bit-exact upstream of the
first float operation.

Usage:
    python3 emit_forward_fixtures.py <source-dir> <weights.npz> <out-dir>

    <source-dir>  the cloned repo root containing the codehygiene package
                  (e.g. .raw/codehygiene); inserted at sys.path[0].
    <weights.npz> the committed T1a fixture weights (the script refuses a
                  blake3 mismatch against the recorded T1a pin).
    <out-dir>     where the pins land (one JSON per snippet + index.json +
                  pins.json). Created if missing; existing files overwritten.

Pins written per snippet (forward_<name>.json):
    provenance   source pin + content blake3 of the package __init__.py +
                 numpy/python versions + platform + emitted timestamp.
    grid         the (lines x 128) u8 byte grid as hex (the stage-(a)
                 bit-exactness surface; tabs expand to 2 columns, non-ASCII
                 codepoints clamp to 127, lines clip at 128 columns).
    heads        noul (2 sigmoid heads; the source names them score /
                 rl_hygiene), the 6 ordinal subscores, the 5 style probs,
                 and the argmax archetype with its top-two margin. Every
                 float carries its decimal value AND the raw f32
                 little-endian bits.
    line_probs   the full (lines x 8) squiggle matrix at full precision,
                 plus per-row argmax kind + row max.
    squiggles    the score()-threshold 0.55 rows (line, cols, kind, prob).

Determinism guard: every snippet is encoded + forwarded TWICE in-process and
the two runs must agree bit-exactly (numpy is deterministic; a disagreement
refuses the emission before anything is written). All outputs must be
finite.

Regeneration provenance on this box: numpy 2.4.6 / CPython 3.11.17 /
macOS aarch64. The committed pins are the authority; a regeneration on a
different numpy must re-run the committed fixtures through the Rust gate
before replacing them (the gate is the arbiter, not the re-run).

Source attribution: MIT declared at pyproject.toml:10 only; no LICENSE file
in the source tree - attribution recorded in issue 1225. The snippets below
are original minimal reproductions written for this fixture lane.
"""

import hashlib
import json
import platform
import struct
import sys
import time
from pathlib import Path

import blake3
import numpy as np

EXPECTED_SOURCE_PIN = "423c7e1f02ef409bbaea209268bfcde36e0f1cb0"
EXPECTED_NPZ_BLAKE3 = "118cb02b922195867c873b16052ebbf91790a1fb9b73ba14d78a0785752e65be"
EXPECTED_INIT_BLAKE3 = "b0dd344a6a4726d83a8c4804354f0948151d86346e78dfab412bb261a2d2922c"
EXPECTED_VERSION = "0.4.0"

N_KINDS = 8
DEFAULT_THRESHOLD = 0.55

# The five snippets. The three language samples are escape-free raw strings
# (real tabs in the Go sample - expandtabs(2) turns each into two columns).
# edge_trap is built from normal strings so the escapes below are codepoints:
#   \t     tab -> expandtabs(2)
#   \u00e9 one non-ASCII codepoint (clamps to 127 in the grid)
#   \U0001F600 an astral codepoint (also clamps to 127; its UTF-8 byte
#          length differs from its codepoint count - pins len() vs bytes)
#   \x0c   form feed: str.splitlines() treats it as a line boundary (the
#          Rust splitter must match, or num_lines diverges loudly)
# plus a 150-column line clipped at 128 and interleaved empty lines.
SNIPPETS = [
    (
        "rust_sample",
        "rust",
        r'''use std::collections::HashMap;

// TODO: rename this before the audit
pub struct Slot {
    pub id: u64,
    pub label: String,
}

fn pick(mut slots: Vec<Slot>, key: &str) -> Option<Slot> {
    let mut idx = 0;
    for s in slots.iter() {
        if s.label == key {
            break;
        }
        idx += 1;
    }
    if idx >= slots.len() {
        return None;
    }
    let mut best = None;
    let mut score = -1;
    let mut i = 0;
    while i < slots.len() {
        if slots[i].id > 0 {
            if slots[i].id % 2 == 0 {
                if score < 0 {
                    score = 1;
                    best = Some(slots[i].clone());
                }
            }
        }
        i += 1;
    }
    match best {
        Some(s) => Some(s),
        None => {
            let _ = slots.pop();
            None
        }
    }
}

fn dup(a: &Slot) -> String { format!("{}-{}", a.id, a.label) }
fn dup2(a: &Slot) -> String { format!("{}-{}", a.id, a.label) }
''',
    ),
    (
        "python_sample",
        "python",
        r'''import json


class HandlerFactory:
    """Builds handlers."""

    def make(self, kind):
        handler = None
        try:
            handler = HANDLERS[kind]
        except KeyError:
            pass
        return handler or default_handler


HANDLERS = {"a": lambda x: x, "b": lambda x: [y for y in sorted(x, reverse=True) if y is not None and str(y).strip() != "" and y not in SEEN]}

def default_handler(x):
    rows = json.loads(x)
    out = []
    # iterate rows and append each row to the out list
    for row in rows:
        out.append(row)
    return out
''',
    ),
    (
        "go_sample",
        "go",
        r'''package main

import (
	"errors"
	"fmt"
)

func run(items []string, mode int) (string, error) {
	out := ""
	for _, it := range items {
		if err := check(it); err != nil {
			if mode == 1 {
				err = nil
			} else if mode == 2 {
				err = nil
			} else {
				return "", err
			}
		}
		out += fmt.Sprintf("%s,", it)
	}
	switch mode {
	case 1:
		return out, nil
	case 2:
		return out, nil
	case 3:
		return out, errors.New("mode 3")
	}
	return out, nil
}

func check(s string) error {
	if len(s) == 0 {
		return errors.New("empty")
	}
	return nil
}
''',
    ),
    (
        "edge_trap",
        "go",
        "func\twide(line int) int {\n"
        "\t// tabs expand to two columns; this comment rides a tab stop\n"
        "\tpad\t:=\tline\t*\t4\n"
        "\tna\u00e9ve := pad - 1 + 0x7f\n"
        "\tmark := \"r\u00e9sum\u00e9 \U0001F600\" // accents + astral clamp to 127\n"
        "\treturn na\u00e9ve + wide(longline)\n"
        "}\n"
        "\n"
        "// the next line is 131 columns and gets clipped at 128 by the grid\n"
        "const longline = 1111111111 + 2222222222 + 3333333333 + 4444444444 + 5555555555 + 6666666666 + 7777777777 + 8888888888 + 9999999999\n"
        "\n"
        "\x0c\n"
        "func after_formfeed() int { return 1 }\n",
    ),
    (
        "edge_empty",
        "text",
        "",
    ),
]


def blake3_hex(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def f32_bits(value) -> str:
    return struct.pack("<f", float(value)).hex()


def pin_float(value) -> dict:
    return {"value": float(value), "bits_le_hex": f32_bits(value)}


def provenance_block(init_digest: str) -> dict:
    return {
        "source_pin": EXPECTED_SOURCE_PIN,
        "init_py_blake3": init_digest,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "emitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def encode_and_forward(mod, weights, code: str):
    grid, lines = mod.encode_byte_grid(code)
    line_probs, noul, subscores, style_probs = mod._forward(weights, grid)
    return grid, lines, line_probs, noul, subscores, style_probs


def squiggle_rows(mod, lines, line_probs) -> list:
    rows = []
    for idx, raw_line in enumerate(lines):
        best = int(np.argmax(line_probs[idx]))
        prob = float(line_probs[idx, best])
        if prob < DEFAULT_THRESHOLD:
            continue
        stripped = raw_line.lstrip(" ")
        col_start = len(raw_line) - len(stripped) + 1 if stripped else 1
        rows.append(
            {
                "line": idx + 1,
                "col_start": col_start,
                "col_end": max(col_start, len(raw_line)),
                "kind": mod.SQUIGGLE_KINDS[best],
                "probability": prob,
            }
        )
    return rows


def snippet_doc(mod, weights, name, lang, code, init_digest) -> dict:
    grid, lines, line_probs, noul, subscores, style_probs = encode_and_forward(
        mod, weights, code
    )
    # Determinism guard: re-encode + re-forward; the two runs must agree
    # bit-exactly before anything is pinned.
    grid2, lines2, line_probs2, noul2, subscores2, style_probs2 = encode_and_forward(
        mod, weights, code
    )
    assert np.array_equal(grid, grid2), f"{name}: grid not deterministic"
    assert list(lines) == list(lines2), f"{name}: line split not deterministic"
    for a, b in (
        (line_probs, line_probs2),
        (noul, noul2),
        (subscores, subscores2),
        (style_probs, style_probs2),
    ):
        assert np.array_equal(a, b), f"{name}: forward not deterministic"

    assert grid.dtype == np.uint8 and grid.shape == (len(lines), 128), (
        f"{name}: unexpected grid {grid.dtype} {grid.shape}"
    )
    assert line_probs.shape == (len(lines), N_KINDS), (
        f"{name}: line_probs shape {line_probs.shape} vs lines {len(lines)}"
    )

    heads = {
        "noul": [pin_float(v) for v in noul],
        "subscores": {
            score_name: pin_float(v) for score_name, v in zip(mod.SCORE_NAMES, subscores)
        },
        "style_probs": {
            arch: pin_float(v) for arch, v in zip(mod.ARCHETYPE_NAMES, style_probs)
        },
    }
    order = np.argsort(style_probs)[::-1]
    heads["archetype_argmax"] = {
        "name": mod.ARCHETYPE_NAMES[int(order[0])],
        "top1": pin_float(style_probs[order[0]]),
        "top2": pin_float(style_probs[order[1]]),
        "margin": float(style_probs[order[0]] - style_probs[order[1]]),
    }

    for arr in (line_probs, noul, subscores, style_probs):
        assert np.isfinite(arr).all(), f"{name}: non-finite forward output"

    return {
        "provenance": provenance_block(init_digest),
        "snippet": {
            "name": name,
            "lang": lang,
            "bytes": len(code.encode("utf-8")),
            "num_lines": len(lines),
            "sha256_source": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        },
        "grid": {
            "num_lines": int(grid.shape[0]),
            "cols": int(grid.shape[1]),
            "hex": grid.tobytes().hex(),
        },
        "heads": heads,
        "line_probs": {
            "shape": [int(d) for d in line_probs.shape],
            "values": [[float(v) for v in row] for row in line_probs],
            "argmax_kind": [mod.SQUIGGLE_KINDS[int(np.argmax(row))] for row in line_probs],
            "row_max": [float(np.max(row)) for row in line_probs],
        },
        "squiggles_at_threshold": {
            "threshold": DEFAULT_THRESHOLD,
            "rows": squiggle_rows(mod, lines, line_probs),
        },
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: emit_forward_fixtures.py <source-dir> <weights.npz> <out-dir>"
        )
    source_dir, npz_path, out_dir = (Path(p) for p in sys.argv[1:4])

    init_path = source_dir / "codehygiene" / "__init__.py"
    if not init_path.is_file():
        raise SystemExit(f"source package not found at {init_path}")
    init_digest = blake3_hex(init_path.read_bytes())
    if init_digest != EXPECTED_INIT_BLAKE3:
        raise SystemExit(
            f"codehygiene/__init__.py blake3 {init_digest} != pinned "
            f"{EXPECTED_INIT_BLAKE3} - the source tree is not at the recorded pin"
        )

    npz_digest = blake3_hex(npz_path.read_bytes())
    if npz_digest != EXPECTED_NPZ_BLAKE3:
        raise SystemExit(
            f"weights.npz blake3 {npz_digest} != T1a pin {EXPECTED_NPZ_BLAKE3}"
        )

    sys.path.insert(0, str(source_dir))
    import codehygiene as mod

    resolved = Path(mod.__file__).resolve()
    if source_dir.resolve() not in resolved.parents:
        raise SystemExit(f"imported codehygiene from {resolved}, not under {source_dir}")
    if mod.__version__ != EXPECTED_VERSION:
        raise SystemExit(
            f"codehygiene __version__ {mod.__version__} != {EXPECTED_VERSION}"
        )

    weights = mod.load_weights(npz_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for name, lang, code in SNIPPETS:
        doc = snippet_doc(mod, weights, name, lang, code, init_digest)
        payload = json.dumps(doc, indent=1)
        target = out_dir / f"forward_{name}.json"
        target.write_text(payload + "\n", encoding="utf-8")
        summary = doc["heads"]
        index_rows.append(
            {
                "name": name,
                "lang": lang,
                "num_lines": doc["snippet"]["num_lines"],
                "file": target.name,
                "blake3": blake3_hex(target.read_bytes()),
            }
        )
        print(
            f"{name}: lines={doc['snippet']['num_lines']} "
            f"noul={summary['noul'][0]['value']:.6f},{summary['noul'][1]['value']:.6f} "
            f"archetype={summary['archetype_argmax']['name']} "
            f"squiggles={len(doc['squiggles_at_threshold']['rows'])}"
        )

    index = {
        "provenance": provenance_block(init_digest),
        "weights_npz_blake3": npz_digest,
        "threshold": DEFAULT_THRESHOLD,
        "squiggle_kinds": list(mod.SQUIGGLE_KINDS),
        "score_names": list(mod.SCORE_NAMES),
        "archetype_names": list(mod.ARCHETYPE_NAMES),
        "snippets": index_rows,
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    pins = {
        "source_pin": EXPECTED_SOURCE_PIN,
        "init_py_blake3": init_digest,
        "weights_npz_blake3": npz_digest,
        **{row["file"]: row["blake3"] for row in index_rows},
        "index.json": blake3_hex(index_path.read_bytes()),
    }
    (out_dir / "pins.json").write_text(
        json.dumps(pins, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(index_rows)} snippet pins + index + pins to {out_dir}")


if __name__ == "__main__":
    main()
