"""pyztraj — fast molecular dynamics trajectory analysis powered by Zig."""

from pyztraj._ffi import get_ffi, get_lib
from pyztraj._helpers import ZtrajError
from pyztraj.combined import analyze_all
from pyztraj.analysis import (
    Contact,
    HBond,
    SasaResult,
    compute_contacts,
    compute_rdf,
    compute_sasa,
    detect_hbonds,
)
from pyztraj.geometry import (
    compute_angles,
    compute_center_of_geometry,
    compute_center_of_mass,
    compute_dihedrals,
    compute_distances,
    compute_inertia,
    compute_principal_moments,
    compute_rg,
    compute_rmsd,
    compute_rmsf,
)
from pyztraj.io import Structure, XtcReader, load_pdb, open_xtc


def get_version() -> str:
    """Get the ztraj library version string."""
    return get_ffi().string(get_lib().ztraj_version()).decode("utf-8")


__all__ = [
    "Contact",
    "HBond",
    "SasaResult",
    "Structure",
    "XtcReader",
    "ZtrajError",
    "analyze_all",
    "compute_angles",
    "compute_center_of_geometry",
    "compute_center_of_mass",
    "compute_contacts",
    "compute_dihedrals",
    "compute_distances",
    "compute_inertia",
    "compute_principal_moments",
    "compute_rdf",
    "compute_rg",
    "compute_rmsd",
    "compute_rmsf",
    "compute_sasa",
    "detect_hbonds",
    "get_version",
    "load_pdb",
    "open_xtc",
]
