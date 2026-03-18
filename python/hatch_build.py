"""Custom build hook to compile Zig library and CLI binary during pip install."""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ZigBuildHook(BuildHookInterface):
    """Build hook that compiles the Zig shared library and CLI binary before packaging."""

    PLUGIN_NAME = "zig-build"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Build the Zig library and CLI binary, then copy them to the package directory."""
        build_data["infer_tag"] = True

        root_dir = Path(self.root).parent  # Go up from python/ to project root
        python_dir = Path(self.root)
        package_dir = python_dir / "pyztraj"

        # Platform-specific names
        if sys.platform == "darwin":
            lib_name = "libztraj.dylib"
        elif sys.platform == "win32":
            lib_name = "ztraj.dll"
        else:
            lib_name = "libztraj.so"

        exe_name = "ztraj.exe" if sys.platform == "win32" else "ztraj"

        # Source paths (zig-out/)
        if sys.platform == "win32":
            lib_src = root_dir / "zig-out" / "bin" / lib_name
        else:
            lib_src = root_dir / "zig-out" / "lib" / lib_name
        exe_src = root_dir / "zig-out" / "bin" / exe_name

        # Destination paths (pyztraj/)
        lib_dst = package_dir / lib_name
        exe_dst = package_dir / exe_name

        # Build if needed
        needs_build = self._needs_build(lib_src, lib_dst) or self._needs_build(exe_src, exe_dst)
        if needs_build:
            self._build_zig(root_dir)

        # Copy library
        self._copy_artifact(lib_src, lib_dst, "shared library")

        # Copy CLI binary
        self._copy_artifact(exe_src, exe_dst, "CLI binary")
        if sys.platform != "win32":
            exe_dst.chmod(exe_dst.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # Include both artifacts in the wheel
        build_data["force_include"][str(lib_dst)] = f"pyztraj/{lib_name}"
        build_data["force_include"][str(exe_dst)] = f"pyztraj/{exe_name}"

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
        self.app.display_info("Building Zig library and CLI binary...")

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
            self.app.display_success("Zig library and CLI binary built successfully")
        except subprocess.CalledProcessError as e:
            msg = f"Zig build failed:\n{e.stderr}"
            self.app.display_error(msg)
            raise RuntimeError(msg) from e
        except subprocess.TimeoutExpired as e:
            msg = (
                "Zig build timed out after 600 seconds. "
                "Try building manually: cd <root> && zig build -Doptimize=ReleaseFast"
            )
            raise RuntimeError(msg) from e
