"""Shared tomllib import surface for the scripts gate family (Issue 1245 T1).

`import tomllib` is Python 3.11+; this workspace's workstation python3 is
3.9.6 (the 3.9 syntax-floor gate exists because of it — Issue 965) and the
macOS CI runner images have drifted below 3.11 as well, so a bare module
import crashes a gate before any verdict. The gate family imports through
this shim instead:

    from toml_compat import tomllib

Resolution order: stdlib tomllib (3.11+) -> the `tomli` backport -> a loud
ModuleNotFoundError naming both remedies. The failure raises
ModuleNotFoundError (an ImportError subclass) deliberately: existing
`except ModuleNotFoundError` / `except ImportError` guards around family
imports (docs_drift_sweep's lazy auditor import, suite_membership_audit's
stub install) keep working, and the guard's message now carries the remedy.

There is deliberately NO stub/silent path here: a TOML-parsing gate that
cannot parse must refuse, never guess. (suite_membership_audit installs its
own tomllib stub BEFORE importing the family so its regex-based scan can run
without a parser — the stub satisfies this shim's stdlib import and is never
reached through it; that posture is unchanged.)

Remedies, in the workspace's own order of preference:
  1. run the gate through a newer interpreter — the workspace Python
     toolchain is uv, and docs_gate.sh probes `uv python find '>=3.11'`:
         uv run --python 3.12 python3 scripts/<gate>.py ...
  2. CI pins the interpreter with actions/setup-python (python-version 3.11)
     ahead of every step that invokes a family gate with bare python3.
  3. a 3.9/3.10 box can install the backport: python3 -m pip install tomli
"""

import sys

try:
    import tomllib  # py3.11+: stdlib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # the 3.9/3.10 backport (pure python, MIT)
    except ModuleNotFoundError as _exc:
        raise ModuleNotFoundError(
            "tomllib needs Python >= 3.11 (stdlib) or the `tomli` backport; "
            "neither is importable on this interpreter (python %s). Run the "
            "gate through a newer interpreter — the workspace toolchain is "
            "uv: `uv run --python 3.12 python3 scripts/<gate>.py ...` (the "
            "same probe docs_gate.sh makes) — or install the backport: "
            "python3 -m pip install tomli" % sys.version.split()[0]
        ) from _exc

__all__ = ["tomllib"]
