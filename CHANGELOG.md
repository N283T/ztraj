# Changelog

## [0.2.0] - 2026-03-19

### Added
- SASA computation via zsasa dependency (Shrake-Rupley algorithm, element-based VdW radii)
- `all` command: combined analysis (RMSD, RMSF, Rg, SASA, COM, hbonds, contacts) in one pass
- Python `analyze_all()` convenience function for batch trajectory analysis
- `compute_sasa()` Python API with per-atom and total SASA

### Changed
- Split monolithic `core.py` (1080 lines) into focused modules:
  `_helpers.py`, `io.py`, `geometry.py`, `analysis.py`, `combined.py`
- External Python API unchanged (all re-exported from `__init__.py`)

### Fixed
- PyPI publish action: use tag instead of SHA for Docker-based action

## [0.1.0] - 2026-03-18

### Added
- Python bindings package `pyztraj` with CFFI-based NumPy API
  - I/O: `load_pdb`, `open_xtc` (streaming XTC reader)
  - Geometry: `compute_distances`, `compute_angles`, `compute_dihedrals`,
    `compute_rmsd`, `compute_rmsf`, `compute_rg`,
    `compute_center_of_mass`, `compute_center_of_geometry`,
    `compute_inertia`, `compute_principal_moments`
  - Analysis: `detect_hbonds`, `compute_contacts`, `compute_rdf`
- C API layer (`src/c_api.zig`) with shared library build target
- Hatchling build hook for automatic Zig compilation during `pip install`
- Bundled CLI binary in PyPI wheel (`pip install pyztraj` installs `ztraj` command)
- GitHub Actions CI (format check, build, Python tests, validation suite)
- GitHub Actions publish workflow (cibuildwheel, PyPI, GitHub Releases)
- Validation suite against mdtraj reference values (9/9 metrics)
- Dihedral angle sign convention fix (IUPAC/Prelog-Klyne)

### Fixed
- Dihedral sign convention: changed cross product order to match IUPAC
- Error handling in validation scripts (exit codes, tracebacks)
