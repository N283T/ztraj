# Changelog

## [Unreleased]

### Added
- GRO (GROMACS) format support for topology and coordinate reading
  - Dynamic coordinate field width detection (handles variable precision)
  - Orthogonal and triclinic box vector support
  - Automatic nm-to-Angstrom unit conversion
- `summary` CLI command: show structure/trajectory overview (atoms, residues, chains, box, elements)
- `convert` CLI command: convert between PDB and GRO formats
- PDB and GRO file writers for format conversion

## [0.3.0] - 2026-03-21

### Added
- Native DSSP secondary structure assignment (replaced vendored zdssp)
- Protein-specific dihedrals: phi, psi, omega, chi1-chi4 CLI commands
- Structural superposition (QCP-based optimal alignment with `Frame.initView`)
- PBC support: wrap coordinates, minimum image distance, make molecules whole
- TRR trajectory format support (GROMACS)
- Native contacts Q value (hard-cut and soft-cut)
- MSD (Mean Square Displacement) for diffusion analysis
- PCA (Principal Component Analysis) of coordinate fluctuations
- Validation suite expanded to 14 tests (pytest-based, vs mdtraj references)
  - Added phi/psi, superposition, PBC distance, DSSP, SASA validations
- `Frame.initView()` for safe non-owning views over const coordinate slices

### Changed
- DSSP: native implementation using ztraj types (atom indices into Frame)
  instead of vendored zdssp copy (removed 9058 lines)
- Validation suite moved from PEP 723 script to pytest
- Python `_pack_frames` helper extracted to eliminate 3× duplication
- CI validation job uses pytest instead of standalone script

### Fixed
- Superposition S matrix convention (was transposed vs rmsd.zig)
- PBC box validation (reject zero/negative diagonals)
- PCA dimension overflow guard (reject dim > 30,000)
- Native contacts bounds check on atom indices
- `@constCast` eliminated from C API via `Frame.initView()`

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
