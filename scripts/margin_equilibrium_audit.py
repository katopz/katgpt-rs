#!/usr/bin/env python3
"""Margin-equilibrium config audit (Issue 959 T3, Research 632 section 2.3).

Given an optimizer epsilon and an accumulator float format, print the
predicted post-memorization logit margin m* = ln(1/eps), the format's
precision cliff c_d = p*ln2 (arXiv:2610.31642 Softmax Collapse cliff),
and ONE verdict line:

  HAZARD  m* > c_d  -- the standard max-subtraction form saturates the
                        target slot at exactly -0.0 BEFORE equilibrium
                        (the paper's "Slingshot" regime prediction);
  CLEAR   m* <= c_d -- equilibrium stays inside the resolvable band.

The verdict names the REGIME from config alone; it does not measure a
live model. The Rust half of the same law ships as
`katgpt_core::softmax_fix::margin_equilibrium` (the golden asserts live
in that module's tests); this CLI is the offline audit form for config
sweeps.

Usage:
    python3 scripts/margin_equilibrium_audit.py [--eps 1e-8] [--format fp32]
    python3 scripts/margin_equilibrium_audit.py --sweep          # the paper's table
    python3 scripts/margin_equilibrium_audit.py --self-test      # golden asserts

Exit: 0 verdicts printed (or self-test PASS); 1 self-test FAIL; 2 bad args.
"""

from __future__ import annotations

import argparse
import math
import sys

import console_safe  # noqa: E402

console_safe.apply()

# Significand bits p (explicit + implicit leading 1) per accumulator
# format. Mirrors katgpt_core::softmax_fix::significand_bits — the two
# tables must not drift (the self-test pins the shared goldens).
FORMAT_BITS = {
    "fp8e4m3": 4,
    "bf16": 8,
    "fp16": 11,
    "tf32": 11,
    "fp32": 24,
    "fp64": 53,
}

LN_2 = math.log(2.0)

# Paper goldens (nats): fp8e4m3 ~2.8, bf16 5.5, fp16/tf32 ~7.6, fp32 16.6,
# fp64 36.7 (±0.1 nat budget — the same envelope the Rust tests pin).
PAPER_GOLDENS = {
    "fp8e4m3": 2.8,
    "bf16": 5.5,
    "fp16": 7.6,
    "tf32": 7.6,
    "fp32": 16.6,
    "fp64": 36.7,
}

# The paper's Slingshot golden: default AdamW eps puts m* just past the
# fp32 cliff (m* = 18.42 vs 16.64) -> HAZARD; eps = 1e-7 clears.
SLINGSHOT_GOLDEN = ("fp32", 1e-8, 18.420680743952367, 16.635532333438687, "HAZARD")
CLEAR_GOLDEN = ("fp32", 1e-7, 16.11809565095833, 16.635532333438687, "CLEAR")


def cliff_nats(fmt: str) -> float:
    return FORMAT_BITS[fmt] * LN_2


def verdict(eps: float, fmt: str) -> tuple[float, float, str]:
    m_star = -math.log(eps)
    cliff = cliff_nats(fmt)
    return m_star, cliff, "HAZARD" if m_star > cliff else "CLEAR"


def print_row(eps: float, fmt: str) -> None:
    m_star, cliff, v = verdict(eps, fmt)
    headroom = cliff - m_star
    print(
        "eps={:<8g} format={:<8s} m*={:.4f} nats  cliff={:.4f} nats  "
        "headroom={:+.4f}  verdict: {}".format(eps, fmt, m_star, cliff, headroom, v)
    )


def self_test() -> int:
    failures = 0
    for fmt, paper in PAPER_GOLDENS.items():
        c = cliff_nats(fmt)
        if abs(c - paper) >= 0.1:
            print(
                "self-test: FAIL {} cliff {:.4f} vs paper {} beyond +/-0.1 nat".format(
                    fmt, c, paper
                )
            )
            failures += 1
    for fmt, eps, m_gold, cliff_gold, v_gold in (SLINGSHOT_GOLDEN, CLEAR_GOLDEN):
        m, cliff, v = verdict(eps, fmt)
        if abs(m - m_gold) > 1e-9 or abs(cliff - cliff_gold) > 1e-9 or v != v_gold:
            print(
                "self-test: FAIL golden eps={} {}: got ({:.6f}, {:.6f}, {}), "
                "want ({:.6f}, {:.6f}, {})".format(
                    eps, fmt, m, cliff, v, m_gold, cliff_gold, v_gold
                )
            )
            failures += 1
    if failures:
        print("self-test: FAIL ({} assertion(s))".format(failures))
        return 1
    print("self-test: PASS — cliff goldens vs paper; Slingshot + clear goldens exact")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eps", type=float, default=None, help="optimizer epsilon")
    parser.add_argument(
        "--format",
        choices=sorted(FORMAT_BITS),
        default=None,
        help="accumulator float format (the cliff binds HERE, not in the storage format)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="audit the paper's table: both default-ish epsilons x every format",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run the golden asserts and exit"
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.sweep:
        for fmt in sorted(FORMAT_BITS):
            for eps in (1e-8, 1e-7):
                print_row(eps, fmt)
        return 0

    if args.eps is None or args.format is None:
        parser.error("--eps and --format are required without --sweep/--self-test")
        return 2
    if args.eps <= 0.0:
        print("error: --eps must be positive", file=sys.stderr)
        return 2
    print_row(args.eps, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
