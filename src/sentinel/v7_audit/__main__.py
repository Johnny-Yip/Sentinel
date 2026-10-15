"""Run read-only inspection and write new V7 evidence files exclusively."""

import argparse
import json
from pathlib import Path

from sentinel.v7_audit.audit import run_audit


def write_new_json(path, value):
    """Never overwrite existing evidence, including when paths are symlinks."""
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "baseline"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--features", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="New file; existing paths are refused")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.mode == "audit":
            paths = args.features or sorted((root / "artifacts/v6-phase8-revalidation/features").glob("*.csv"))
            result = run_audit(paths, root)
        else:
            from sentinel.v7_audit.baseline import baseline_manifest
            result = baseline_manifest(root)
        write_new_json(args.output, result)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")
    return 2 if result.get("input_status") == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
