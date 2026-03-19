"""Geometry analysis functions: distances, angles, dihedrals, RMSD, RMSF, Rg, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyztraj._ffi import get_ffi, get_lib
from pyztraj._helpers import _as_u32, _check, _ptr_f32, _ptr_f64, _ptr_u32, _to_soa

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
            _ptr_f32(x), _ptr_f32(y), _ptr_f32(z),
            len(x), _ptr_u32(pairs), n_pairs, _ptr_f32(result),
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
            _ptr_f32(x), _ptr_f32(y), _ptr_f32(z),
            len(x), _ptr_u32(triplets), n_triplets, _ptr_f32(result),
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
            _ptr_f32(x), _ptr_f32(y), _ptr_f32(z),
            len(x), _ptr_u32(quartets), n_quartets, _ptr_f32(result),
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
            _ptr_f32(ref_x), _ptr_f32(ref_y), _ptr_f32(ref_z),
            _ptr_f32(x), _ptr_f32(y), _ptr_f32(z),
            idx_ptr, n_indices, n_atoms, result,
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
            _ptr_f32(x), _ptr_f32(y), _ptr_f32(z),
            _ptr_f64(masses), idx_ptr, n_indices, n_atoms, result,
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
            _ptr_f32(x), _ptr_f32(y), _ptr_f32(z),
            _ptr_f64(masses), idx_ptr, n_indices, n_atoms, cx, cy, cz,
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
            _ptr_f32(x), _ptr_f32(y), _ptr_f32(z),
            idx_ptr, n_indices, n_atoms, cx, cy, cz,
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
            _ptr_f32(x), _ptr_f32(y), _ptr_f32(z),
            _ptr_f64(masses), idx_ptr, n_indices, n_atoms, _ptr_f64(result),
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
        (n_selected,) RMSF values in Angstroms.
    """
    if len(frames) == 0:
        msg = "frames list must not be empty"
        raise ValueError(msg)

    ffi = get_ffi()
    lib = get_lib()

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
            _ptr_f32(all_x), _ptr_f32(all_y), _ptr_f32(all_z),
            n_frames, n_atoms, idx_ptr, n_indices, _ptr_f64(result),
        ),
        "compute_rmsf",
    )
    return result
