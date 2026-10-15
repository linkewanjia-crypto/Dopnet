"""Explicit new-run CLI. Scientific implementations live in this namespace."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .policy import PolicyError, PolicyStopped


def _common(parser):
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True,
                        help="Explicit relocated shared verification/support tree.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-new-run", action="store_true", required=True,
                        help="Acknowledge a separately authorized new run.")
    parser.add_argument("--authorization-text", required=True,
                        help="Actual operator authorization; no default authorization is supplied.")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--deadline-utc", help="Optional new-run UTC deadline; no legacy deadline is used.")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare a new candidate manifest/configuration.")
    _common(prepare)
    prepare.add_argument("--configs-dir", type=Path, required=True,
                         help="Explicit pure scientific seed-config directory; no installed location is guessed.")
    prepare.add_argument("--validation-cache", type=Path, required=True)
    prepare.add_argument("--reference-results", type=Path, required=True,
                         help="Explicit original experiment/evaluation scientific reference directory.")
    train = commands.add_parser("train", help="Run the independent candidate training implementation.")
    _common(train)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--checkpoint-reader", type=Path, required=True,
                       help="Explicit restricted metadata reader; controller verifies its identity.")
    train.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Run the independent candidate evaluation implementation.")
    _common(evaluate)
    evaluate.add_argument("--training-output", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True,
                          help="Explicit new prepared manifest; no historical output is guessed.")
    evaluate.add_argument("--checkpoint-reader", type=Path, required=True,
                          help="Explicit restricted metadata reader; evaluation verifies its identity.")
    evaluate.add_argument("--reference-results", type=Path, required=True)
    evaluate.add_argument("--parent-archive", type=Path, required=True)
    evaluate.add_argument("--inference-microbatch", type=int, choices=(1, 2, 4, 8), default=1)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.authorization_text.strip():
        raise PolicyError("authorization_text must be an actual nonempty operator statement")
    # Lazy imports remain inside the new namespace. No legacy entry is imported.
    if args.command == "prepare":
        from .controller import prepare
        value = prepare(args)
    elif args.command == "train":
        from .controller import train
        value = train(args)
    else:
        from .evaluate import evaluate
        value = evaluate(args)
    return value if type(value) is int else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PolicyStopped as exc:
        print("NEW_RUN_STOPPED: " + str(exc), file=sys.stderr)
        raise SystemExit(75)
    except PolicyError as exc:
        print("NEW_RUN_POLICY_REJECTED: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
