"""Run an installed ECC CLI without editable-package rebuilds.

The external ECC venv has compiled DreamPlace artifacts already installed, but
its scikit-build editable finder tries to rebuild in a read-only external
checkout on every import. This wrapper disables only that automatic rebuild;
it does not modify the installed package or its compiled artifacts. Run with
the ECC venv's Python, and keep the ECC project/output under this worktree.
"""

from __future__ import annotations

import sys


def main() -> int:
    for finder in sys.meta_path:
        if type(finder).__name__ == "ScikitBuildRedirectingFinder":
            finder.rebuild_flag = False
    from chipcompiler.cli.main import main as ecc_main
    sys.argv[0] = "ecc"
    return ecc_main()


if __name__ == "__main__":
    raise SystemExit(main())
