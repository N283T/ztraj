# ztraj

[![CI](https://github.com/N283T/ztraj/actions/workflows/ci.yml/badge.svg)](https://github.com/N283T/ztraj/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyztraj)](https://pypi.org/project/pyztraj/)
[![Zig](https://img.shields.io/badge/Zig-0.15.2-f7a41d?logo=zig)](https://ziglang.org/)
[![Python](https://img.shields.io/pypi/pyversions/pyztraj)](https://pypi.org/project/pyztraj/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

High-performance molecular dynamics trajectory analysis library and CLI, written in Zig with Python bindings.

## Features

- **Fast**: SIMD-accelerated computation with multi-threaded execution
- **Comprehensive**: RMSD, RMSF, Rg, distances, angles, dihedrals, inertia tensor, hydrogen bonds, contacts, RDF
- **Dual interface**: CLI tool (`ztraj`) and Python library (`pyztraj`)
- **Streaming**: Memory-efficient frame-by-frame XTC/TRR trajectory reading
- **Validated**: Tested against mdtraj reference values and MD ATLAS

## CLI

```bash
# Install from PyPI (no Zig required)
uv tool install pyztraj

# Or build from source (requires Zig 0.15.2+)
zig build -Doptimize=ReleaseFast
```

```bash
# RMSD against frame 0 for backbone atoms
ztraj rmsd traj.xtc --top structure.pdb --select backbone --ref 0

# Radius of gyration
ztraj rg traj.xtc --top structure.pdb --select protein

# Pairwise distances
ztraj distances traj.xtc --top structure.pdb --pairs "0-10,1-20"

# RDF between selections
ztraj rdf traj.xtc --top structure.pdb --sel1 "O" --sel2 "H" --rmax 8.0

# GRO topology support
ztraj rmsd traj.xtc --top structure.gro --select backbone --ref 0

# Format conversion (structure)
ztraj convert input.gro --output output.pdb
ztraj convert input.pdb --output output.gro

# Trajectory conversion
ztraj convert traj.xtc --top structure.pdb --output traj.trr
ztraj convert traj.trr --top structure.gro --output traj.xtc

# Hydrogen bonds
ztraj hbonds structure.pdb

# Output formats: json (default), csv, tsv
ztraj rg traj.xtc --top structure.pdb --format csv --output rg.csv
```

### Available commands

`rmsd`, `rmsf`, `distances`, `angles`, `dihedrals`, `rg`, `center`, `inertia`, `hbonds`, `contacts`, `rdf`, `sasa`, `all`, `dssp`, `phi`, `psi`, `omega`, `chi`, `summary`, `convert`

## Python bindings (pyztraj)

### Installation

```bash
pip install pyztraj
# or
uv pip install pyztraj
```

Pre-built wheels are available for Linux (x86_64/aarch64), macOS (x86_64/ARM64), and Windows (amd64) with Python 3.11–3.13.

To build from source instead, see [Building from source](#building-from-source).

### Usage

```python
import numpy as np
import pyztraj

# Load structure
struct = pyztraj.load_pdb("structure.pdb")

# Compute distances
pairs = np.array([[0, 10], [1, 20]], dtype=np.uint32)
distances = pyztraj.compute_distances(struct.coords, pairs)

# Stream trajectory
with pyztraj.open_xtc("traj.xtc", struct.n_atoms) as reader:
    for frame in reader:
        rg = pyztraj.compute_rg(frame.coords, struct.masses)
        rmsd = pyztraj.compute_rmsd(frame.coords, struct.coords)

# Hydrogen bonds
hbonds = pyztraj.detect_hbonds(struct, struct.coords)

# Residue contacts
contacts = pyztraj.compute_contacts(struct, struct.coords, cutoff=8.0)

# RDF
r, g_r = pyztraj.compute_rdf(sel1_coords, sel2_coords, box_volume=1000.0)
```

### API

| Category | Functions |
|----------|-----------|
| I/O | `load_pdb`, `load_gro`, `load_mmcif`, `open_xtc`, `open_trr`, `open_dcd`, `write_pdb`, `write_gro` |
| Geometry | `compute_distances`, `compute_angles`, `compute_dihedrals`, `compute_phi`, `compute_psi`, `compute_omega`, `compute_chi` |
| Structure | `compute_rmsd`, `compute_rmsf`, `compute_rg`, `compute_msd` |
| Properties | `compute_center_of_mass`, `compute_center_of_geometry`, `compute_inertia`, `compute_principal_moments`, `minimum_image_distance`, `wrap_coords`, `make_molecules_whole` |
| Analysis | `detect_hbonds`, `compute_contacts`, `compute_rdf`, `compute_sasa`, `compute_dssp`, `compute_native_contacts_q`, `compute_pca`, `analyze_all` |

## Supported formats

| Format | Read | Notes |
|--------|------|-------|
| PDB | Yes | Topology + coordinates |
| mmCIF | Yes | Topology + coordinates |
| GRO | Yes | Topology + coordinates (GROMACS) |
| XTC | Yes | Streaming trajectory (GROMACS) |
| TRR | Yes | Trajectory (GROMACS) |
| DCD | Yes | Trajectory (CHARMM/NAMD) |

## Building from source

Requires [Zig 0.15.2+](https://ziglang.org/download/).

```bash
# Build CLI + shared library
zig build -Doptimize=ReleaseFast

# Run tests
zig build test

# Build Python bindings from source
cd python
pip install .

# Run validation suite (requires numpy, mdtraj reference data)
uv run --with numpy --script validation/validate.py
```

## License

MIT
