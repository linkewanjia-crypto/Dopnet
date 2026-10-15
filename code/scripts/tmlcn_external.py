#!/usr/bin/env python3
"""Independent external evaluation adapters; help imports only the stdlib.

Use the primary tmlcn.py facade for a plan or an explicitly requested run.
The verifier checks a newly produced Forti artifact against the public view.
It does not reuse historical validation or delivery receipts.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

MODULES = {
    "fortitran": "fortitran_evaluate",
    "wimae": "wimae_evaluate",
    "verify-forti": "fortitran_verify",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODULES)
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values == ["--help"] or values == ["-h"]:
        parser.print_help()
        return 0
    args = parser.parse_args(values[:1])
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "code/training/portable", "code/training/baselines/fortitran_budget_matched",
        "code/training/baselines/wimae_author", "code/training/runtime/verification_support",
    ):
        sys.path.insert(0, str(root / relative))
    module = importlib.import_module("dopnet_external_public_reference." + MODULES[args.mode])
    previous = sys.argv
    sys.argv = [str(Path(module.__file__))] + values[1:]
    try:
        module.main()
    except Exception as error:
        if args.mode == "wimae" and isinstance(error, module.DeadlineReached):
            print("WIMAE_EVALUATION_INCOMPLETE_TIME_LIMIT " + str(error), file=sys.stderr, flush=True)
            return 75
        raise
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
