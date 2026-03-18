"""Custom build hook to compile Zig shared library during pip install."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ZigBuildHook(BuildHookInterface):
    """Build hook that compiles the Zig shared library before packaging."""

    PLUGIN_NAME = "zig-build"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Build the Zig shared library and copy it to the package directory."""
        build_data["infer_tag"] = True

        root_dir = Path(self.root).parent  # Go up from python/ to project root
        python_dir = Path(self.root)
        package_dir = python_dir / "pyztraj"

        if sys.platform == "darwin":
            lib_name = "libztraj.dylib"
        elif sys.platform == "win32":
            lib_name = "ztraj.dll"
        else:
            lib_name = "libztraj.so"

        if sys.platform == "win32":
            lib_src = root_dir / "zig-out" / "bin" / lib_name
        else:
            lib_src = root_dir / "zig-out" / "lib" / lib_name

        lib_dst = package_dir / lib_name

        if self._needs_build(lib_src, lib_dst):
            self._build_zig(root_dir)

        self._copy_artifact(lib_src, lib_dst, "shared library")

        build_data["force_include"][str(lib_dst)] = f"pyztraj/{lib_name}"

    def _needs_build(self, src: Path, dst: Path) -> bool:
        """Check if a build is needed based on file existence and timestamps."""
        if not src.exists():
            return True
        if not dst.exists():
            return False
        return src.stat().st_mtime > dst.stat().st_mtime

    def _copy_artifact(self, src: Path, dst: Path, label: str) -> None:
        """Copy a build artifact from zig-out to the package directory."""
        if not src.exists():
            msg = f"Zig build artifact not found: {src} ({label})"
            raise FileNotFoundError(msg)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                msg = f"Failed to copy {label} from {src} to {dst}"
                raise RuntimeError(msg) from e

    def _find_zig(self) -> list[str]:
        """Find the Zig compiler command."""
        if shutil.which("zig"):
            return ["zig"]
        try:
            subprocess.run(
                [sys.executable, "-m", "ziglang", "version"],
                capture_output=True,
                check=True,
                timeout=10,
            )
            return [sys.executable, "-m", "ziglang"]
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def _build_zig(self, root_dir: Path) -> None:
        """Run zig build command."""
        self.app.display_info("Building Zig shared library...")

        zig_cmd = self._find_zig()
        if not zig_cmd:
            msg = (
                "Zig compiler not found. Install Zig 0.15.2+ from "
                "https://ziglang.org/download/ or run: pip install ziglang"
            )
            raise RuntimeError(msg)

        try:
            subprocess.run(
                [*zig_cmd, "build", "-Doptimize=ReleaseFast"],
                cwd=root_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.app.display_success("Zig shared library built successfully")
        except subprocess.CalledProcessError as e:
            self.app.display_error(f"Zig build failed:\n{e.stderr}")
            raise RuntimeError("Zig build failed") from e
