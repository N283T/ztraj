"""High-level Python API for ztraj trajectory analysis.

All coordinates are in Angstroms. Angles are in radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pyztraj._ffi import get_ffi, get_lib

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


def _as_f32(arr: NDArray) -> NDArray[np.float32]:
    """Ensure array is contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


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


# =============================================================================
# Version
# =============================================================================


def get_version() -> str:
    """Get the ztraj library version string."""
    return get_ffi().string(get_lib().ztraj_version()).decode("utf-8")


# =============================================================================
# I/O
# =============================================================================


@dataclass
class Structure:
    """A loaded molecular structure with coordinates and topology."""

    coords: NDArray[np.float32]  # (n_atoms, 3)
    masses: NDArray[np.float64]  # (n_atoms,)
    atom_names: list[str]  # length n_atoms
    residue_names: list[str]  # length n_atoms (per-atom)
    resids: NDArray[np.int32]  # (n_atoms,) per-atom residue IDs
    n_atoms: int


def load_pdb(path: str | Path) -> Structure:
    """Load a PDB file and return a Structure with coordinates and topology.

    Args:
        path: Path to the PDB file.

    Returns:
        Structure with coords (Angstroms), masses, atom/residue names.
    """
    ffi = get_ffi()
    lib = get_lib()

    path_bytes = str(path).encode("utf-8")
    handle_ptr = ffi.new("void**")

    _check(lib.ztraj_load_pdb(path_bytes, handle_ptr), f"load_pdb({path})")
    handle = handle_ptr[0]
    if handle == ffi.NULL:
        raise ZtrajError(f"load_pdb({path}): returned success but handle is null")

    try:
        n_atoms = lib.ztraj_get_n_atoms(handle)

        # Coordinates
        x = np.empty(n_atoms, dtype=np.float32)
        y = np.empty(n_atoms, dtype=np.float32)
        z = np.empty(n_atoms, dtype=np.float32)
        _check(lib.ztraj_get_coords(handle, _ptr_f32(x), _ptr_f32(y), _ptr_f32(z)))
        coords = np.column_stack([x, y, z])

        # Masses
        masses = np.empty(n_atoms, dtype=np.float64)
        _check(lib.ztraj_get_masses(handle, _ptr_f64(masses)))

        # Atom names (4 bytes each)
        name_buf = ffi.new(f"char[{n_atoms * 4}]")
        _check(lib.ztraj_get_atom_names(handle, name_buf))
        atom_names = [
            ffi.string(name_buf + i * 4, 4).decode("utf-8").strip() for i in range(n_atoms)
        ]

        # Residue names (5 bytes each, per-atom)
        res_buf = ffi.new(f"char[{n_atoms * 5}]")
        _check(lib.ztraj_get_residue_names(handle, res_buf))
        residue_names = [
            ffi.string(res_buf + i * 5, 5).decode("utf-8").strip() for i in range(n_atoms)
        ]

        # Residue IDs
        resids = np.empty(n_atoms, dtype=np.int32)
        _check(lib.ztraj_get_resids(handle, _ptr_i32(resids)))

    finally:
        lib.ztraj_free_structure(handle)

    return Structure(
        coords=coords,
        masses=masses,
        atom_names=atom_names,
        residue_names=residue_names,
        resids=resids,
        n_atoms=n_atoms,
    )


class XtcReader:
    """Streaming XTC trajectory reader (context manager).

    Usage::

        with pyztraj.open_xtc("traj.xtc", n_atoms) as reader:
            for frame in reader:
                # frame.coords is (n_atoms, 3) float32
                print(frame.time, frame.coords.shape)
    """

    @dataclass
    class Frame:
        """A single trajectory frame."""

        coords: NDArray[np.float32]  # (n_atoms, 3)
        time: float  # picoseconds
        step: int

    def __init__(self, path: str | Path, n_atoms: int) -> None:
        self._ffi = get_ffi()
        self._lib = get_lib()
        self._path = str(path).encode("utf-8")
        self._expected_n_atoms = n_atoms
        self._handle = None
        self._n_atoms = 0

    def __enter__(self) -> XtcReader:
        n_atoms_out = self._ffi.new("size_t*")
        handle_ptr = self._ffi.new("void**")
        _check(self._lib.ztraj_open_xtc(self._path, n_atoms_out, handle_ptr), "open_xtc")
        self._handle = handle_ptr[0]
        self._n_atoms = n_atoms_out[0]

        if self._n_atoms != self._expected_n_atoms:
            self._lib.ztraj_close_xtc(self._handle)
            self._handle = None
            msg = (
                f"XTC has {self._n_atoms} atoms but expected {self._expected_n_atoms} "
                f"(from topology)"
            )
            raise ValueError(msg)

        return self

    def __exit__(self, *args) -> None:
        if self._handle is not None:
            self._lib.ztraj_close_xtc(self._handle)
            self._handle = None

    def __iter__(self):
        return self

    def __next__(self) -> XtcReader.Frame:
        if self._handle is None:
            raise StopIteration

        x = np.empty(self._n_atoms, dtype=np.float32)
        y = np.empty(self._n_atoms, dtype=np.float32)
        z = np.empty(self._n_atoms, dtype=np.float32)
        time_out = self._ffi.new("float*")
        step_out = self._ffi.new("int32_t*")

        rc = self._lib.ztraj_read_xtc_frame(
            self._handle, _ptr_f32(x), _ptr_f32(y), _ptr_f32(z), time_out, step_out
        )

        if rc == self._lib.ZTRAJ_ERROR_EOF:
            raise StopIteration

        _check(rc, "read_xtc_frame")

        coords = np.column_stack([x, y, z])
        return XtcReader.Frame(coords=coords, time=float(time_out[0]), step=int(step_out[0]))


def open_xtc(path: str | Path, n_atoms: int) -> XtcReader:
    """Open an XTC trajectory file for streaming frame-by-frame reading.

    Args:
        path: Path to the XTC file.
        n_atoms: Expected number of atoms (from topology).

    Returns:
        XtcReader context manager yielding Frame objects.
    """
    return XtcReader(path, n_atoms)


# =============================================================================
# Geometry: single-frame functions
# =============================================================================


def compute_distances(
    coords: NDArray[np.float32],
    pairs: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Compute pairwise Euclidean distances.

    Args:
        coords: (n_atoms, 3) coordinates in Angstroms.
        pairs: (n_pairs, 2) atom index pairs (0-based).

    Returns:
        (n_pairs,) distances in Angstroms.
    """
    x, y, z = _to_soa(coords)
    pairs = _as_u32(pairs.reshape(-1))
    n_pairs = len(pairs) // 2
    result = np.empty(n_pairs, dtype=np.float32)
    _check(
        get_lib().ztraj_distances(
            _ptr_f32(x),
            _ptr_f32(y),
            _ptr_f32(z),
            len(x),
            _ptr_u32(pairs),
            n_pairs,
            _ptr_f32(result),
        )
    )
    return result


def compute_angles(
    coords: NDArray[np.float32],
    triplets: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Compute bond angles.

    Args:
        coords: (n_atoms, 3) coordinates in Angstroms.
        triplets: (n_triplets, 3) atom index triplets. Angle measured at middle atom.

    Returns:
        (n_triplets,) angles in radians.
    """
    x, y, z = _to_soa(coords)
    triplets = _as_u32(triplets.reshape(-1))
    n_triplets = len(triplets) // 3
    result = np.empty(n_triplets, dtype=np.float32)
    _check(
        get_lib().ztraj_angles(
            _ptr_f32(x),
            _ptr_f32(y),
            _ptr_f32(z),
            len(x),
            _ptr_u32(triplets),
            n_triplets,
            _ptr_f32(result),
        )
    )
    return result


def compute_dihedrals(
    coords: NDArray[np.float32],
    quartets: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Compute dihedral (torsion) angles.

    Args:
        coords: (n_atoms, 3) coordinates in Angstroms.
        quartets: (n_quartets, 4) atom index quartets.

    Returns:
        (n_quartets,) dihedral angles in radians, range [-pi, pi].
    """
    x, y, z = _to_soa(coords)
    quartets = _as_u32(quartets.reshape(-1))
    n_quartets = len(quartets) // 4
    result = np.empty(n_quartets, dtype=np.float32)
    _check(
        get_lib().ztraj_dihedrals(
            _ptr_f32(x),
            _ptr_f32(y),
            _ptr_f32(z),
            len(x),
            _ptr_u32(quartets),
            n_quartets,
            _ptr_f32(result),
        )
    )
    return result


def compute_rmsd(
    coords: NDArray[np.float32],
    ref_coords: NDArray[np.float32],
    atom_indices: NDArray[np.uint32] | None = None,
) -> float:
    """Compute RMSD between two structures using the QCP algorithm.

    The algorithm internally centers and optimally superposes the structures.

    Args:
        coords: (n_atoms, 3) target coordinates.
        ref_coords: (n_atoms, 3) reference coordinates.
        atom_indices: Optional subset of atom indices to use.

    Returns:
        RMSD in Angstroms.
    """
    ffi = get_ffi()
    if coords.shape != ref_coords.shape:
        msg = f"coords shape {coords.shape} != ref_coords shape {ref_coords.shape}"
        raise ValueError(msg)
    x, y, z = _to_soa(coords)
    ref_x, ref_y, ref_z = _to_soa(ref_coords)
    n_atoms = len(x)

    if atom_indices is not None:
        atom_indices = _as_u32(atom_indices)
        idx_ptr = _ptr_u32(atom_indices)
        n_indices = len(atom_indices)
    else:
        idx_ptr = ffi.cast("uint32_t*", 0)
        n_indices = 0

    result = ffi.new("double*")
    _check(
        get_lib().ztraj_rmsd(
            _ptr_f32(ref_x),
            _ptr_f32(ref_y),
            _ptr_f32(ref_z),
            _ptr_f32(x),
            _ptr_f32(y),
            _ptr_f32(z),
            idx_ptr,
            n_indices,
            n_atoms,
            result,
        )
    )
    return float(result[0])


def compute_rg(
    coords: NDArray[np.float32],
    masses: NDArray[np.float64],
    atom_indices: NDArray[np.uint32] | None = None,
) -> float:
    """Compute mass-weighted radius of gyration.

    Args:
        coords: (n_atoms, 3) coordinates.
        masses: (n_atoms,) atomic masses.
        atom_indices: Optional subset of atom indices.

    Returns:
        Radius of gyration in Angstroms.
    """
    ffi = get_ffi()
    x, y, z = _to_soa(coords)
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    n_atoms = len(x)

    if atom_indices is not None:
        atom_indices = _as_u32(atom_indices)
        idx_ptr = _ptr_u32(atom_indices)
        n_indices = len(atom_indices)
    else:
        idx_ptr = ffi.cast("uint32_t*", 0)
        n_indices = 0

    result = ffi.new("double*")
    _check(
        get_lib().ztraj_rg(
            _ptr_f32(x),
            _ptr_f32(y),
            _ptr_f32(z),
            _ptr_f64(masses),
            idx_ptr,
            n_indices,
            n_atoms,
            result,
        )
    )
    return float(result[0])


def compute_center_of_mass(
    coords: NDArray[np.float32],
    masses: NDArray[np.float64],
    atom_indices: NDArray[np.uint32] | None = None,
) -> NDArray[np.float64]:
    """Compute mass-weighted center of mass.

    Args:
        coords: (n_atoms, 3) coordinates.
        masses: (n_atoms,) atomic masses.
        atom_indices: Optional subset of atom indices.

    Returns:
        (3,) center of mass coordinates in Angstroms.
    """
    ffi = get_ffi()
    x, y, z = _to_soa(coords)
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    n_atoms = len(x)

    if atom_indices is not None:
        atom_indices = _as_u32(atom_indices)
        idx_ptr = _ptr_u32(atom_indices)
        n_indices = len(atom_indices)
    else:
        idx_ptr = ffi.cast("uint32_t*", 0)
        n_indices = 0

    cx = ffi.new("double*")
    cy = ffi.new("double*")
    cz = ffi.new("double*")
    _check(
        get_lib().ztraj_center_of_mass(
            _ptr_f32(x),
            _ptr_f32(y),
            _ptr_f32(z),
            _ptr_f64(masses),
            idx_ptr,
            n_indices,
            n_atoms,
            cx,
            cy,
            cz,
        )
    )
    return np.array([cx[0], cy[0], cz[0]], dtype=np.float64)


def compute_center_of_geometry(
    coords: NDArray[np.float32],
    atom_indices: NDArray[np.uint32] | None = None,
) -> NDArray[np.float64]:
    """Compute unweighted center of geometry.

    Args:
        coords: (n_atoms, 3) coordinates.
        atom_indices: Optional subset of atom indices.

    Returns:
        (3,) center of geometry coordinates in Angstroms.
    """
    ffi = get_ffi()
    x, y, z = _to_soa(coords)
    n_atoms = len(x)

    if atom_indices is not None:
        atom_indices = _as_u32(atom_indices)
        idx_ptr = _ptr_u32(atom_indices)
        n_indices = len(atom_indices)
    else:
        idx_ptr = ffi.cast("uint32_t*", 0)
        n_indices = 0

    cx = ffi.new("double*")
    cy = ffi.new("double*")
    cz = ffi.new("double*")
    _check(
        get_lib().ztraj_center_of_geometry(
            _ptr_f32(x),
            _ptr_f32(y),
            _ptr_f32(z),
            idx_ptr,
            n_indices,
            n_atoms,
            cx,
            cy,
            cz,
        )
    )
    return np.array([cx[0], cy[0], cz[0]], dtype=np.float64)


def compute_inertia(
    coords: NDArray[np.float32],
    masses: NDArray[np.float64],
    atom_indices: NDArray[np.uint32] | None = None,
) -> NDArray[np.float64]:
    """Compute 3x3 inertia tensor around center of mass.

    Args:
        coords: (n_atoms, 3) coordinates.
        masses: (n_atoms,) atomic masses.
        atom_indices: Optional subset of atom indices.

    Returns:
        (3, 3) inertia tensor.
    """
    ffi = get_ffi()
    x, y, z = _to_soa(coords)
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    n_atoms = len(x)

    if atom_indices is not None:
        atom_indices = _as_u32(atom_indices)
        idx_ptr = _ptr_u32(atom_indices)
        n_indices = len(atom_indices)
    else:
        idx_ptr = ffi.cast("uint32_t*", 0)
        n_indices = 0

    result = np.empty(9, dtype=np.float64)
    _check(
        get_lib().ztraj_inertia(
            _ptr_f32(x),
            _ptr_f32(y),
            _ptr_f32(z),
            _ptr_f64(masses),
            idx_ptr,
            n_indices,
            n_atoms,
            _ptr_f64(result),
        )
    )
    return result.reshape(3, 3)


def compute_principal_moments(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute principal moments of inertia from a 3x3 tensor.

    Args:
        tensor: (3, 3) inertia tensor.

    Returns:
        (3,) principal moments in ascending order.
    """
    tensor = np.ascontiguousarray(tensor.reshape(9), dtype=np.float64)
    result = np.empty(3, dtype=np.float64)
    _check(get_lib().ztraj_principal_moments(_ptr_f64(tensor), _ptr_f64(result)))
    return result


def compute_rmsf(
    frames: list[NDArray[np.float32]],
    atom_indices: NDArray[np.uint32] | None = None,
) -> NDArray[np.float64]:
    """Compute per-atom RMSF over multiple frames.

    RMSF_i = sqrt(mean_t(|r_i(t) - <r_i>|^2))

    Args:
        frames: List of (n_atoms, 3) coordinate arrays.
        atom_indices: Optional subset of atom indices.

    Returns:
        (n_selected,) RMSF values in Angstroms. Length is len(atom_indices)
        if provided, otherwise n_atoms.
    """
    if len(frames) == 0:
        msg = "frames list must not be empty"
        raise ValueError(msg)

    ffi = get_ffi()
    lib = get_lib()

    n_frames = len(frames)
    n_atoms = frames[0].shape[0]

    # Build flat contiguous arrays: all_x[frame * n_atoms + atom]
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

    if atom_indices is not None:
        atom_indices = _as_u32(atom_indices)
        idx_ptr = _ptr_u32(atom_indices)
        n_indices = len(atom_indices)
        n_out = n_indices
    else:
        idx_ptr = ffi.cast("uint32_t*", 0)
        n_indices = 0
        n_out = n_atoms

    result = np.empty(n_out, dtype=np.float64)
    _check(
        lib.ztraj_rmsf(
            _ptr_f32(all_x),
            _ptr_f32(all_y),
            _ptr_f32(all_z),
            n_frames,
            n_atoms,
            idx_ptr,
            n_indices,
            _ptr_f64(result),
        ),
        "compute_rmsf",
    )
    return result
