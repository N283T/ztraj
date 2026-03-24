"""Shared helpers for pyztraj modules.

All coordinates are in Angstroms. Angles are in radians.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyztraj._ffi import get_ffi

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ZtrajError(Exception):
    """Error raised by ztraj C library."""


_ERROR_MESSAGES = {
    -1: "Invalid input parameters",
    -2: "Out of memory",
    -3: "File I/O error",
    -4: "Parse error",
    -5: "End of file",
}


def _check(rc: int, operation: str = "") -> None:
    """Check return code from C API and raise on error."""
    if rc != 0:
        msg = _ERROR_MESSAGES.get(rc, f"Unknown error (code {rc})")
        if operation:
            msg = f"{operation}: {msg}"
        raise ZtrajError(msg)


def _to_soa(coords: NDArray[np.float32]) -> tuple[NDArray, NDArray, NDArray]:
    """Convert (n_atoms, 3) AOS coords to SOA (x, y, z) arrays."""
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        msg = f"coords must be (n_atoms, 3), got {coords.shape}"
        raise ValueError(msg)
    return coords[:, 0].copy(), coords[:, 1].copy(), coords[:, 2].copy()


def _as_u32(arr: NDArray) -> NDArray[np.uint32]:
    """Ensure array is contiguous uint32."""
    return np.ascontiguousarray(arr, dtype=np.uint32)


def _ptr_f32(arr: NDArray[np.float32]):
    """Get CFFI pointer to float32 array data."""
    return get_ffi().cast("float*", arr.ctypes.data)


def _ptr_f64(arr: NDArray[np.float64]):
    """Get CFFI pointer to float64 array data."""
    return get_ffi().cast("double*", arr.ctypes.data)


def _ptr_u32(arr: NDArray[np.uint32]):
    """Get CFFI pointer to uint32 array data."""
    return get_ffi().cast("uint32_t*", arr.ctypes.data)


def _ptr_i32(arr: NDArray[np.int32]):
    """Get CFFI pointer to int32 array data."""
    return get_ffi().cast("int32_t*", arr.ctypes.data)


def _load_topology_handle(structure, lib, ffi, operation: str = ""):
    """Re-load topology from a Structure's source file, returning an opaque handle.

    Uses the correct loader function based on the original file format.
    Caller must free the handle with lib.ztraj_free_structure().
    """
    path_bytes = str(structure._path).encode("utf-8")
    handle_ptr = ffi.new("void**")
    load_fn = getattr(lib, structure._loader)
    _check(load_fn(path_bytes, handle_ptr), operation)
    return handle_ptr[0]


def _pack_frames(
    frames: list[NDArray[np.float32]],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], int, int]:
    """Pack list of (n_atoms, 3) frames into flat SOA arrays.

    Returns:
        (all_x, all_y, all_z, n_frames, n_atoms)
    """
    n_frames = len(frames)
    n_atoms = frames[0].shape[0]

    all_x = np.empty(n_frames * n_atoms, dtype=np.float32)
    all_y = np.empty(n_frames * n_atoms, dtype=np.float32)
    all_z = np.empty(n_frames * n_atoms, dtype=np.float32)

    for i, frame in enumerate(frames):
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        if frame.shape != (n_atoms, 3):
            msg = f"Frame {i} shape {frame.shape} != expected ({n_atoms}, 3)"
            raise ValueError(msg)
        offset = i * n_atoms
        all_x[offset : offset + n_atoms] = frame[:, 0]
        all_y[offset : offset + n_atoms] = frame[:, 1]
        all_z[offset : offset + n_atoms] = frame[:, 2]

    return all_x, all_y, all_z, n_frames, n_atoms
