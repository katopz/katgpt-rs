#!/usr/bin/env python3
"""GLiNER-class open-NER lane for the T0.7 span fixture (Plan 628, katgpt-extract).

The TRAINED-LANE reference baseline for tests/g5_span_baselines.rs — never
the GOAT bar (the GOAT gate is span_pointer vs the RULE floor). OPT-IN by
design: the harness skips the lane loud unless GLINER_SPAN_PRED names this
script's output, and this script needs its own GLiNER install. No CI lane
runs it; a human or agent arms it on a box that has the venv.

Setup (one-time, CPU is fine; ~200M-param model):

    cd <repo-root>/crates/katgpt-extract
    uv venv .raw/gliner-env
    uv pip install --python .raw/gliner-env/bin/python gliner

Run:

    .raw/gliner-env/bin/python tools/gliner_span_lane.py \
        tests/fixtures/external/mit_movie_trivia_test.jsonl /tmp/gliner_pred.jsonl

Then arm the harness:

    GLINER_SPAN_PRED=/tmp/gliner_pred.jsonl \
        cargo test -p katgpt-extract --test g5_span_baselines --features span_pointer -- --nocapture

Env knobs: GLINER_MODEL (default urchade/gliner_multi-v2.1), GLINER_THRESHOLD
(default 0.5), GLINER_BATCH (default 8). Defaults are recorded in the bench
notes so a re-run can price sensitivity against the same record.

Output: one JSON line per fixture row
    {"id": "M-0001", "spans": [["Actor", 41, 56], ...]}
(byte offsets into the row text, end-exclusive — the fixture is pure ASCII
so char and byte offsets coincide; the script asserts that).
"""

import json
import os
import sys

# Upstream IOB type -> natural-language label (mechanical, one-to-one).
LABELS = {
    "Actor": "actor",
    "Award": "award",
    "Character_Name": "fictional character name",
    "Director": "movie director",
    "Genre": "movie genre",
    "Opinion": "opinion",
    "Origin": "place of origin",
    "Plot": "plot element",
    "Quote": "quotation",
    "Relationship": "family relationship",
    "Soundtrack": "soundtrack song",
    "Year": "year",
}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: gliner_span_lane.py <fixture.jsonl> <out.jsonl>", file=sys.stderr)
        return 2
    fixture_path, out_path = sys.argv[1], sys.argv[2]

    model_name = os.environ.get("GLINER_MODEL", "urchade/gliner_multi-v2.1")
    threshold = float(os.environ.get("GLINER_THRESHOLD", "0.5"))
    batch_size = int(os.environ.get("GLINER_BATCH", "8"))

    rows = []
    with open(fixture_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    # The lane's offset contract: char offsets == byte offsets (pure ASCII).
    for row in rows:
        text = row["text"]
        if len(text) != len(text.encode("utf-8")):
            print(
                "FATAL: row %s is not pure ASCII; the lane's byte-offset contract breaks"
                % row["id"],
                file=sys.stderr,
            )
            return 1

    try:
        from gliner import GLiNER
    except ImportError:
        print(
            "FATAL: gliner is not installed in this interpreter — build the venv per the module docstring",
            file=sys.stderr,
        )
        return 1

    label_list = [LABELS[t] for t in sorted(LABELS)]
    type_of_label = {LABELS[t]: t for t in LABELS}

    model = GLiNER.from_pretrained(model_name)
    predicted = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            entities = model.predict_entities(
                row["text"], label_list, threshold=threshold, batch_size=batch_size
            )
            spans = []
            for ent in entities:
                ty = type_of_label.get(ent.get("label"))
                if ty is None:
                    continue
                spans.append([ty, int(ent["start"]), int(ent["end"])])
            predicted += len(spans)
            out.write(
                json.dumps({"id": row["id"], "spans": spans}, separators=(",", ":")) + "\n"
            )
    print("wrote %s: %d rows, %d spans" % (out_path, len(rows), predicted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
