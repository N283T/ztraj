# Changelog

## [0.6.1] - 2026-04-13

### Added
- #90 AMBER PRMTOP topology parser (.prmtop, .parm7, .top)
  - Atom names, residues, bonds, partial charges, force-field masses
  - ATOMIC_NUMBER element assignment (amber12+) with fallback heuristic
  - Chamber-style topology detection with clear error
- #91 AMBER NetCDF trajectory reader (.nc, .ncdf)
  - NetCDF-3 classic and 64-bit offset format (no external dependency)
  - Streaming NcReader API matching XtcReader/DcdReader/TrrReader pattern
  - Coordinates, time, cell_lengths/cell_angles with triclinic box conversion
- #93 Python bindings for AMBER formats
  - `pyztraj.load_prmtop()`: topology loading with masses and charges
  - `pyztraj.open_nc()` / `NcReader`: streaming NetCDF trajectory reader
  - C API: `ztraj_load_prmtop`, `ztraj_open_nc`, `ztraj_read_nc_frame`, `ztraj_close_nc`
- #96 AMBER NetCDF trajectory writer (.nc)
  - NcWriter with open/writeFrame/close API, orthogonal and triclinic box support
  - CLI `ztraj convert` supports .nc output (NC ↔ XTC/TRR/DCD)
  - Python `NcWriter` class and C API (`ztraj_open_nc_writer`, etc.)

### Changed
- Topology gains optional `charges` and `explicit_masses` fields (null when absent)
- `Topology.masses()` prefers explicit force-field masses over element-derived values
- `Topology.validate()` checks charges/explicit_masses length consistency

### Fixed
- `ztraj_get_n_atoms` C API uses topology atom count (was frame count, returned 0 for topology-only formats)
- Analysis C API functions validate n_atoms against topology (was frame, broke prmtop workflow)

## [0.6.0] - 2026-03-29

### Added
- #78 Interactive marimo notebook for trajectory analysis
- #88 CLAUDE.md with project conventions and build instructions

### Fixed
- #79 Infer D-H bonds when topology lacks bond records
- #85 Address review findings across CLI, FFI, packaging, and CI
  - Extract duplicate validation helpers to shared module
  - Replace `std.process.exit(1)` with proper error returns
  - Add `n_atoms` validation to C API dihedral/DSSP exports
  - Fix contacts reallocation path
  - Add Windows CI support
  - Tighten publish workflow tag pattern
- #87 Handle DSSP compute errors individually instead of catch-all

## [0.5.1] - 2026-03-26

### Added
- #75 Progress reporting for CLI frame loading and per-frame analysis (`std.Progress`)

## [0.5.0] - 2026-03-26

### Added
- #68 Multi-threading for distances, RMSF, contacts with `--threads` CLI flag
- #69 Multi-threading for RDF, MSD, hydrogen bonds
- #70 Multi-threading for PCA covariance matrix
- #71 SASA bitmask algorithm option (`shrake_rupley_bitmask`)
- #73 Expose SASA bitmask algorithm in C API and Python (`algorithm` parameter)

### Performance
- #67 SIMD vectorize geometry modules (center, rg, distances, rmsd, superpose)
- #68 SIMD inner loop for RMSF computation
- #69 SIMD inner loops for RDF distance computation and MSD displacement
- #70 SIMD outer product vectorization for PCA covariance matrix
- #70 DSSP near-pair finding reduced from O(n_res^2) to O(n_res) via cell-list spatial indexing
- #72 Spatial cell list for hydrogen bond acceptor lookup (50-100x for large proteins)
- #72 SIMD batch distance testing in neighbor list construction

### Changed
- #66 Update GitHub Actions to Node.js 24

## [0.4.2] - 2026-03-25

### Fixed
- #64 fix: use basename for program name in CLI help

### Changed
- #63 docs: clarify PyPI vs source installation

## [0.4.1] - 2026-03-24

### Added
- Python bindings for all I/O formats: `load_gro`, `load_mmcif`, `open_trr`, `open_dcd`
- Python file writers: `write_pdb`, `write_gro`, `XtcWriter`, `TrrWriter` context managers
- Python protein dihedral functions: `compute_phi`, `compute_psi`, `compute_omega`, `compute_chi`
- Python DSSP secondary structure: `compute_dssp`
- Python atom selection: `Structure.select()` method (backbone, protein, water, by name)
- C API: structure loaders (`ztraj_load_gro`, `ztraj_load_mmcif`)
- C API: trajectory readers (`ztraj_open_trr/dcd` + read/close)
- C API: file writers (`ztraj_write_pdb/gro`, `ztraj_open_xtc/trr_writer` + write/close)
- C API: protein dihedrals (`ztraj_compute_phi/psi/omega/chi`)
- C API: DSSP (`ztraj_compute_dssp`)
- C API: atom selection (`ztraj_select_keyword`, `ztraj_select_name`)

### Fixed
- Analysis functions now work with GRO/mmCIF-loaded structures (format-aware topology reloading)
- Writer close errors properly surfaced in Python (not silently discarded)
- ResourceWarning for unclosed trajectory writers

### Changed
- pyproject.toml version bumped to 0.4.1
- Refactored `_load_structure` helper and `_TrajectoryReader`/`_TrajectoryWriter` base classes

## [0.4.0] - 2026-03-23

### Added
- GRO (GROMACS) format support for topology and coordinate reading
  - Dynamic coordinate field width detection (handles variable precision)
  - Orthogonal and triclinic box vector support
  - Automatic nm-to-Angstrom unit conversion
- `summary` CLI command: show structure/trajectory overview (atoms, residues, chains, box, elements)
- `convert` CLI command: convert between structure and trajectory formats
  - PDB and GRO file writers for structure format conversion
  - XTC and TRR writers for trajectory format conversion (via zxdrfile v0.2.0)
  - Auto-detects output format from file extension

### Fixed
- GRO element inference for ions and metals (Fe, Zn, Na, Cl, Mg, etc.)
  - Uses residue name context to distinguish from protein atom names
  - Supports GROMACS, CHARMM (SOD/CLA/POT), and AMBER naming conventions

### Changed
- Upgraded zxdrfile dependency from v0.1.1 to v0.2.0

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
