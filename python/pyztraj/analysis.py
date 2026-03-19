"""Analysis functions: RDF, hydrogen bonds, contacts, SASA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyztraj._ffi import get_ffi, get_lib
from pyztraj._helpers import _check, _ptr_f32, _ptr_f64, _to_soa
from pyztraj.io import Structure

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    donor: int
    hydrogen: int
    acceptor: int
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

    path_bytes = str(structure._pdb_path).encode("utf-8")
    handle_ptr = ffi.new("void**")
    _check(lib.ztraj_load_pdb(path_bytes, handle_ptr), "detect_hbonds/load_pdb")
    handle = handle_ptr[0]

    try:
        capacity = n_atoms * 4
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

    residue_i: int
    residue_j: int
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
        scheme: One of "closest", "ca", "closest_heavy". Default "closest_heavy".

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

    path_bytes = str(structure._pdb_path).encode("utf-8")
    handle_ptr = ffi.new("void**")
    _check(lib.ztraj_load_pdb(path_bytes, handle_ptr), "compute_contacts/load_pdb")
    handle = handle_ptr[0]

    try:
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

    total_area: float  # total SASA in square Angstroms
    atom_areas: NDArray[np.float64]  # per-atom SASA (n_atoms,)


def compute_sasa(
    structure: Structure,
    coords: NDArray[np.float32],
    n_points: int = 100,
    probe_radius: float = 1.4,
    n_threads: int = 0,
) -> SasaResult:
    """Compute Solvent Accessible Surface Area using Shrake-Rupley algorithm.

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

        return SasaResult(total_area=float(total_area[0]), atom_areas=atom_areas)
    finally:
        lib.ztraj_free_structure(handle)
