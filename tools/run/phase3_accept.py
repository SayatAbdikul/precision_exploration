"""Validate native evidence and attempt conservative accumulator acceptance."""
import argparse
from pathlib import Path
from tools.phase3.common import read, reference, ROOT
from tools.phase3.acceptance import accept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    args = parser.parse_args()
    result = accept(read(args.prepared), reference(args.pilot))
    print(result)
    if result["status"] != "accepted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
