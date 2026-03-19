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
    _pdb_path: str = ""  # internal: path for re-loading topology in analysis functions


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
        _pdb_path=str(path),
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


# =============================================================================
# Analysis functions
# =============================================================================


def compute_rdf(
    sel1_coords: NDArray[np.float32],
    sel2_coords: NDArray[np.float32],
    box_volume: float,
    r_min: float = 0.0,
    r_max: float = 10.0,
    n_bins: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute radial distribution function g(r) between two atom selections.

    Args:
        sel1_coords: (n_sel1, 3) coordinates for selection 1.
        sel2_coords: (n_sel2, 3) coordinates for selection 2.
        box_volume: Simulation box volume in cubic Angstroms.
        r_min: Minimum distance (Angstroms). Default 0.0.
        r_max: Maximum distance (Angstroms). Default 10.0.
        n_bins: Number of histogram bins. Default 100.

    Returns:
        Tuple of (r, g_r) arrays, each of length n_bins.
        r contains bin center positions, g_r contains g(r) values.
    """
    s1_x, s1_y, s1_z = _to_soa(sel1_coords)
    s2_x, s2_y, s2_z = _to_soa(sel2_coords)

    r_out = np.empty(n_bins, dtype=np.float64)
    g_r_out = np.empty(n_bins, dtype=np.float64)

    _check(
        get_lib().ztraj_rdf(
            _ptr_f32(s1_x),
            _ptr_f32(s1_y),
            _ptr_f32(s1_z),
            len(s1_x),
            _ptr_f32(s2_x),
            _ptr_f32(s2_y),
            _ptr_f32(s2_z),
            len(s2_x),
            box_volume,
            r_min,
            r_max,
            n_bins,
            _ptr_f64(r_out),
            _ptr_f64(g_r_out),
        ),
        "compute_rdf",
    )
    return r_out, g_r_out


@dataclass
class HBond:
    """A detected hydrogen bond."""

    donor: int  # atom index of donor heavy atom
    hydrogen: int  # atom index of hydrogen
    acceptor: int  # atom index of acceptor
    distance: float  # H...A distance in Angstroms
    angle: float  # D-H...A angle in degrees


def detect_hbonds(
    structure: Structure,
    coords: NDArray[np.float32],
    dist_cutoff: float = 2.5,
    angle_cutoff: float = 120.0,
    *,
    _handle: object = None,
) -> list[HBond]:
    """Detect hydrogen bonds using Baker-Hubbard criteria.

    Requires a structure loaded via load_pdb() to access bond topology.

    Args:
        structure: Structure from load_pdb().
        coords: (n_atoms, 3) coordinates for the frame to analyze.
        dist_cutoff: Maximum H...A distance in Angstroms. Default 2.5.
        angle_cutoff: Minimum D-H...A angle in degrees. Default 120.0.

    Returns:
        List of HBond objects.
    """
    ffi = get_ffi()
    lib = get_lib()
    x, y, z = _to_soa(coords)
    n_atoms = len(x)

    if n_atoms != structure.n_atoms:
        msg = f"coords has {n_atoms} atoms but structure has {structure.n_atoms}"
        raise ValueError(msg)

    # Load structure handle for topology access
    path_bytes = str(structure._pdb_path).encode("utf-8")
    handle_ptr = ffi.new("void**")
    _check(lib.ztraj_load_pdb(path_bytes, handle_ptr), "detect_hbonds/load_pdb")
    handle = handle_ptr[0]

    try:
        # First pass: detect with large capacity
        capacity = n_atoms * 4  # generous estimate
        hbonds_buf = ffi.new(f"CHBond[{capacity}]")
        n_found = ffi.new("size_t*")

        _check(
            lib.ztraj_detect_hbonds(
                handle,
                _ptr_f32(x),
                _ptr_f32(y),
                _ptr_f32(z),
                n_atoms,
                dist_cutoff,
                angle_cutoff,
                hbonds_buf,
                capacity,
                n_found,
            ),
            "detect_hbonds",
        )

        actual = n_found[0]

        # If buffer was too small, retry with exact size
        if actual > capacity:
            hbonds_buf = ffi.new(f"CHBond[{actual}]")
            _check(
                lib.ztraj_detect_hbonds(
                    handle,
                    _ptr_f32(x),
                    _ptr_f32(y),
                    _ptr_f32(z),
                    n_atoms,
                    dist_cutoff,
                    angle_cutoff,
                    hbonds_buf,
                    actual,
                    n_found,
                ),
                "detect_hbonds",
            )

        return [
            HBond(
                donor=int(hbonds_buf[i].donor),
                hydrogen=int(hbonds_buf[i].hydrogen),
                acceptor=int(hbonds_buf[i].acceptor),
                distance=float(hbonds_buf[i].distance),
                angle=float(hbonds_buf[i].angle),
            )
            for i in range(min(int(n_found[0]), actual))
        ]
    finally:
        lib.ztraj_free_structure(handle)


@dataclass
class Contact:
    """A detected residue-residue contact."""

    residue_i: int  # 0-based residue index
    residue_j: int  # 0-based residue index
    distance: float  # representative distance in Angstroms


def compute_contacts(
    structure: Structure,
    coords: NDArray[np.float32],
    cutoff: float = 4.5,
    scheme: str = "closest_heavy",
) -> list[Contact]:
    """Compute residue-residue contacts.

    Args:
        structure: Structure from load_pdb().
        coords: (n_atoms, 3) coordinates for the frame to analyze.
        cutoff: Distance cutoff in Angstroms. Default 4.5.
        scheme: Distance measurement scheme. One of "closest", "ca",
            "closest_heavy". Default "closest_heavy".

    Returns:
        List of Contact objects.
    """
    ffi = get_ffi()
    lib = get_lib()
    x, y, z = _to_soa(coords)
    n_atoms = len(x)

    if n_atoms != structure.n_atoms:
        msg = f"coords has {n_atoms} atoms but structure has {structure.n_atoms}"
        raise ValueError(msg)

    scheme_map = {"closest": 0, "ca": 1, "closest_heavy": 2}
    scheme_int = scheme_map.get(scheme)
    if scheme_int is None:
        msg = f"Unknown scheme '{scheme}'. Use one of: {list(scheme_map.keys())}"
        raise ValueError(msg)

    # Load structure handle for topology access
    path_bytes = str(structure._pdb_path).encode("utf-8")
    handle_ptr = ffi.new("void**")
    _check(lib.ztraj_load_pdb(path_bytes, handle_ptr), "compute_contacts/load_pdb")
    handle = handle_ptr[0]

    try:
        # Generous initial capacity
        n_residues = len(set(structure.resids.tolist()))
        capacity = n_residues * n_residues // 2
        contacts_buf = ffi.new(f"CContact[{max(capacity, 1)}]")
        n_found = ffi.new("size_t*")

        _check(
            lib.ztraj_compute_contacts(
                handle,
                _ptr_f32(x),
                _ptr_f32(y),
                _ptr_f32(z),
                n_atoms,
                scheme_int,
                cutoff,
                contacts_buf,
                capacity,
                n_found,
            ),
            "compute_contacts",
        )

        actual = int(n_found[0])

        if actual > capacity:
            contacts_buf = ffi.new(f"CContact[{actual}]")
            _check(
                lib.ztraj_compute_contacts(
                    handle,
                    _ptr_f32(x),
                    _ptr_f32(y),
                    _ptr_f32(z),
                    n_atoms,
                    scheme_int,
                    cutoff,
                    contacts_buf,
                    actual,
                    n_found,
                ),
                "compute_contacts",
            )
            actual = int(n_found[0])

        return [
            Contact(
                residue_i=int(contacts_buf[i].residue_i),
                residue_j=int(contacts_buf[i].residue_j),
                distance=float(contacts_buf[i].distance),
            )
            for i in range(min(actual, capacity if actual > capacity else actual))
        ]
    finally:
        lib.ztraj_free_structure(handle)


@dataclass
class SasaResult:
    """SASA calculation result."""

    total_area: float  # total SASA in Å²
    atom_areas: NDArray[np.float64]  # per-atom SASA (n_atoms,) in Å²


def compute_sasa(
    structure: Structure,
    coords: NDArray[np.float32],
    n_points: int = 100,
    probe_radius: float = 1.4,
    n_threads: int = 0,
) -> SasaResult:
    """Compute Solvent Accessible Surface Area using Shrake-Rupley algorithm.

    Uses element-based van der Waals radii from the structure topology.

    Args:
        structure: Structure from load_pdb().
        coords: (n_atoms, 3) coordinates in Angstroms.
        n_points: Number of test points per atom sphere. Default 100.
        probe_radius: Probe radius in Angstroms. Default 1.4 (water).
        n_threads: Number of threads (0 = auto-detect). Default 0.

    Returns:
        SasaResult with total_area and per-atom atom_areas.
    """
    if n_points < 1:
        msg = f"n_points must be >= 1, got {n_points}"
        raise ValueError(msg)
    if probe_radius <= 0.0:
        msg = f"probe_radius must be > 0, got {probe_radius}"
        raise ValueError(msg)
    if n_threads < 0:
        msg = f"n_threads must be >= 0, got {n_threads}"
        raise ValueError(msg)

    ffi = get_ffi()
    lib = get_lib()
    x, y, z = _to_soa(coords)
    n_atoms = len(x)

    if n_atoms != structure.n_atoms:
        msg = f"coords has {n_atoms} atoms but structure has {structure.n_atoms}"
        raise ValueError(msg)

    path_bytes = str(structure._pdb_path).encode("utf-8")
    handle_ptr = ffi.new("void**")
    _check(lib.ztraj_load_pdb(path_bytes, handle_ptr), "compute_sasa/load_pdb")
    handle = handle_ptr[0]

    try:
        atom_areas = np.empty(n_atoms, dtype=np.float64)
        total_area = ffi.new("double*")

        _check(
            lib.ztraj_compute_sasa(
                handle,
                _ptr_f32(x),
                _ptr_f32(y),
                _ptr_f32(z),
                n_atoms,
                n_points,
                probe_radius,
                n_threads,
                _ptr_f64(atom_areas),
                total_area,
            ),
            "compute_sasa",
        )

        return SasaResult(
            total_area=float(total_area[0]),
            atom_areas=atom_areas,
        )
    finally:
        lib.ztraj_free_structure(handle)


# =============================================================================
# Combined analysis
# =============================================================================


def analyze_all(
    structure: Structure,
    frames: list[NDArray[np.float32]],
    ref_frame: int = 0,
) -> dict:
    """Run all analyses on a trajectory and return combined results.

    Computes RMSD, RMSF, Rg, SASA, center of mass, hbonds count, and contacts
    count for each frame in a single call.

    Args:
        structure: Structure from load_pdb().
        frames: List of (n_atoms, 3) coordinate arrays.
        ref_frame: Reference frame index for RMSD. Default 0.

    Returns:
        Dict with keys: n_frames, n_atoms, rmsd, rmsf, rg, sasa,
        center_of_mass, n_hbonds, n_contacts.
    """
    if len(frames) == 0:
        msg = "frames list must not be empty"
        raise ValueError(msg)

    n_frames = len(frames)
    ref_coords = frames[ref_frame]

    rmsd_vals = np.empty(n_frames, dtype=np.float64)
    rg_vals = np.empty(n_frames, dtype=np.float64)
    sasa_vals = np.empty(n_frames, dtype=np.float64)
    com_vals = np.empty((n_frames, 3), dtype=np.float64)
    hbonds_vals = np.empty(n_frames, dtype=np.int32)
    contacts_vals = np.empty(n_frames, dtype=np.int32)

    for i, frame_coords in enumerate(frames):
        rmsd_vals[i] = compute_rmsd(frame_coords, ref_coords)
        rg_vals[i] = compute_rg(frame_coords, structure.masses)
        com_vals[i] = compute_center_of_mass(frame_coords, structure.masses)

        sasa_result = compute_sasa(structure, frame_coords)
        sasa_vals[i] = sasa_result.total_area

        hbonds = detect_hbonds(structure, frame_coords)
        hbonds_vals[i] = len(hbonds)

        contacts = compute_contacts(structure, frame_coords)
        contacts_vals[i] = len(contacts)

    rmsf_vals = compute_rmsf(frames)

    return {
        "n_frames": n_frames,
        "n_atoms": structure.n_atoms,
        "rmsd": rmsd_vals,
        "rmsf": rmsf_vals,
        "rg": rg_vals,
        "sasa": sasa_vals,
        "center_of_mass": com_vals,
        "n_hbonds": hbonds_vals,
        "n_contacts": contacts_vals,
    }
