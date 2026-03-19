#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mdtraj",
#     "numpy",
# ]
# ///
"""Generate reference values from mdtraj for ztraj validation.

Uses 3tvj_I (MD ATLAS) as test system:
- 3tvj_I.pdb: topology (protein only, no solvent)
- 3tvj_I_R1.xtc: 1000 frames, 100ns, 100ps/frame

Output: validation/reference/ directory with JSON files.
"""

import json
import sys
from pathlib import Path

import mdtraj as md
import numpy as np

DATA_DIR = Path(__file__).parent / "test_data"
REF_DIR = Path(__file__).parent / "reference"
REF_DIR.mkdir(exist_ok=True)

PDB = str(DATA_DIR / "3tvj_I.pdb")
XTC = str(DATA_DIR / "3tvj_I_R1.xtc")


def save_json(name: str, data: dict) -> None:
    path = REF_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  -> {path}")


def gen_rmsd() -> None:
    """RMSD of each frame against frame 0, backbone atoms, in angstroms."""
    traj = md.load(XTC, top=PDB)
    backbone = traj.topology.select("backbone")
    rmsd = md.rmsd(traj, traj, frame=0, atom_indices=backbone)
    # mdtraj returns nm, convert to angstroms
    rmsd_ang = (rmsd * 10.0).tolist()
    save_json(
        "rmsd_backbone",
        {
            "description": "RMSD vs frame 0, backbone atoms, angstroms",
            "system": "3tvj_I_R1",
            "n_frames": len(rmsd_ang),
            "n_atoms_backbone": len(backbone),
            "atom_indices": backbone.tolist(),
            "values": rmsd_ang,
        },
    )


def gen_rmsf() -> None:
    """Per-atom RMSF for CA atoms, in angstroms."""
    traj = md.load(XTC, top=PDB)
    ca = traj.topology.select("name CA")
    rmsf = md.rmsf(traj, traj, atom_indices=ca)
    rmsf_ang = (rmsf * 10.0).tolist()
    save_json(
        "rmsf_ca",
        {
            "description": "RMSF of CA atoms over trajectory, angstroms",
            "system": "3tvj_I_R1",
            "n_frames": traj.n_frames,
            "n_atoms_ca": len(ca),
            "atom_indices": ca.tolist(),
            "values": rmsf_ang,
        },
    )


def gen_rg() -> None:
    """Radius of gyration per frame, in angstroms."""
    traj = md.load(XTC, top=PDB)
    rg = md.compute_rg(traj)
    rg_ang = (rg * 10.0).tolist()
    save_json(
        "rg",
        {
            "description": "Radius of gyration per frame, angstroms",
            "system": "3tvj_I_R1",
            "n_frames": len(rg_ang),
            "values": rg_ang,
        },
    )


def gen_distances() -> None:
    """Pairwise distances for a few atom pairs, in angstroms."""
    traj = md.load(XTC, top=PDB)
    # Pick some representative pairs (0-based indices)
    pairs = [[0, 10], [5, 50], [10, 100], [0, 200]]
    dists = md.compute_distances(traj, pairs, periodic=False)
    dists_ang = (dists * 10.0).tolist()
    save_json(
        "distances",
        {
            "description": "Pairwise distances for selected atom pairs, angstroms",
            "system": "3tvj_I_R1",
            "n_frames": traj.n_frames,
            "pairs": pairs,
            "values": dists_ang,
        },
    )


def gen_angles() -> None:
    """Bond angles for a few atom triplets, in radians."""
    traj = md.load(XTC, top=PDB)
    triplets = [[0, 1, 2], [10, 11, 12], [50, 51, 52]]
    angles = md.compute_angles(traj, triplets)
    save_json(
        "angles",
        {
            "description": "Bond angles for selected triplets, radians",
            "system": "3tvj_I_R1",
            "n_frames": traj.n_frames,
            "triplets": triplets,
            "values": angles.tolist(),
        },
    )


def gen_dihedrals() -> None:
    """Dihedral angles for a few atom quartets, in radians."""
    traj = md.load(XTC, top=PDB)
    quartets = [[0, 1, 2, 3], [10, 11, 12, 13], [50, 51, 52, 53]]
    dihedrals = md.compute_dihedrals(traj, quartets)
    save_json(
        "dihedrals",
        {
            "description": "Dihedral angles for selected quartets, radians",
            "system": "3tvj_I_R1",
            "n_frames": traj.n_frames,
            "quartets": quartets,
            "values": dihedrals.tolist(),
        },
    )


def gen_center_of_mass() -> None:
    """Center of mass per frame, in angstroms."""
    traj = md.load(XTC, top=PDB)
    com = md.compute_center_of_mass(traj)
    com_ang = (com * 10.0).tolist()
    save_json(
        "center_of_mass",
        {
            "description": "Center of mass per frame, angstroms [x, y, z]",
            "system": "3tvj_I_R1",
            "n_frames": traj.n_frames,
            "values": com_ang,
        },
    )


def gen_dssp() -> None:
    """Generate DSSP reference: per-residue SS for frame 0 (PDB only)."""
    traj = md.load(PDB)
    dssp = md.compute_dssp(traj, simplified=False)  # full 8-state
    # dssp returns array of shape (n_frames, n_residues) with single-char strings
    ss_string = "".join(dssp[0])

    save_json(
        "dssp",
        {
            "description": "DSSP secondary structure assignment (frame 0, 8-state)",
            "system": "3tvj_I",
            "n_residues": traj.n_residues,
            "ss": ss_string,
        },
    )


def gen_sasa() -> None:
    """Generate SASA reference: per-atom SASA for frame 0."""
    traj = md.load(PDB)
    # mdtraj.shrake_rupley returns nm², convert to Å²
    sasa = md.shrake_rupley(traj, n_sphere_points=960, mode="atom")
    sasa_ang2 = (sasa[0] * 100.0).tolist()  # nm² → Å²

    save_json(
        "sasa",
        {
            "description": "SASA per atom, frame 0, Shrake-Rupley (960 points), angstrom^2",
            "system": "3tvj_I",
            "n_atoms": traj.n_atoms,
            "total": float(np.sum(sasa[0]) * 100.0),
            "atom_areas": sasa_ang2,
        },
    )


def main() -> None:
    print("Generating mdtraj reference values for 3tvj_I_R1...")
    print(f"  PDB: {PDB}")
    print(f"  XTC: {XTC}")
    print()

    generators = [
        ("RMSD (backbone)", gen_rmsd),
        ("RMSF (CA)", gen_rmsf),
        ("Rg", gen_rg),
        ("Distances", gen_distances),
        ("Angles", gen_angles),
        ("Dihedrals", gen_dihedrals),
        ("Center of mass", gen_center_of_mass),
        ("DSSP", gen_dssp),
        ("SASA", gen_sasa),
    ]

    failed = []
    for name, func in generators:
        print(f"Generating {name}...")
        try:
            func()
        except Exception as e:
            print(f"  ERROR: {name}: {e}", file=sys.stderr)
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print("Done. All references generated successfully.")


if __name__ == "__main__":
    main()
