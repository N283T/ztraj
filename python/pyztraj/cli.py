"""CLI entry point that delegates to the bundled ztraj binary."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_binary() -> str:
    """Locate the bundled ztraj binary."""
    package_dir = Path(__file__).parent
    name = "ztraj.exe" if sys.platform == "win32" else "ztraj"
    binary = package_dir / name
    if not binary.exists():
        msg = (
            f"ztraj binary not found at {binary}. "
            "This may indicate a broken installation. "
            "Try reinstalling: pip install --force-reinstall pyztraj"
        )
        raise FileNotFoundError(msg)
    return str(binary)


def main() -> None:
    """Run the bundled ztraj CLI binary."""
    binary = _find_binary()
    try:
        os.execvp(binary, [binary, *sys.argv[1:]])
    except OSError as err:
        print(
            f"Error: failed to execute ztraj binary at {binary}: {err}\n"
            "The binary may be corrupted or built for a different platform.\n"
            "Try reinstalling: pip install --force-reinstall pyztraj",
            file=sys.stderr,
        )
        sys.exit(1)
