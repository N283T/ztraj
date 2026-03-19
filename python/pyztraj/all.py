"""Combined trajectory analysis: run all metrics in one call."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyztraj.analysis import compute_contacts, compute_sasa, detect_hbonds
from pyztraj.geometry import compute_center_of_mass, compute_rg, compute_rmsd, compute_rmsf
from pyztraj.io import Structure

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    if ref_frame < 0 or ref_frame >= len(frames):
        msg = f"ref_frame {ref_frame} out of range for {len(frames)} frames"
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
