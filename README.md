# ztraj

High-performance molecular dynamics trajectory analysis library and CLI, written in Zig with Python bindings.

## Features

- **Fast**: SIMD-accelerated computation with multi-threaded execution
- **Comprehensive**: RMSD, RMSF, Rg, distances, angles, dihedrals, inertia tensor, hydrogen bonds, contacts, RDF
- **Dual interface**: CLI tool (`ztraj`) and Python library (`pyztraj`)
- **Streaming**: Memory-efficient frame-by-frame XTC trajectory reading
- **Validated**: Tested against mdtraj reference values and MD ATLAS

## CLI

```bash
zig build -Doptimize=ReleaseFast

# RMSD against frame 0 for backbone atoms
ztraj rmsd traj.xtc --top structure.pdb --select backbone --ref 0

# Radius of gyration
ztraj rg traj.xtc --top structure.pdb --select protein

# Pairwise distances
ztraj distances traj.xtc --top structure.pdb --pairs "0-10,1-20"

# RDF between selections
ztraj rdf traj.xtc --top structure.pdb --sel1 "O" --sel2 "H" --rmax 8.0

# Hydrogen bonds
ztraj hbonds structure.pdb

# Output formats: json (default), csv, tsv
ztraj rg traj.xtc --top structure.pdb --format csv --output rg.csv
```

### Available commands

`rmsd`, `rmsf`, `distances`, `angles`, `dihedrals`, `rg`, `center`, `inertia`, `hbonds`, `contacts`, `rdf`

## Python bindings (pyztraj)

### Installation

```bash
# From source (requires Zig 0.15.2+)
cd python
pip install .
```

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
| I/O | `load_pdb`, `open_xtc` |
| Geometry | `compute_distances`, `compute_angles`, `compute_dihedrals` |
| Structure | `compute_rmsd`, `compute_rmsf`, `compute_rg` |
| Properties | `compute_center_of_mass`, `compute_center_of_geometry`, `compute_inertia`, `compute_principal_moments` |
| Analysis | `detect_hbonds`, `compute_contacts`, `compute_rdf` |

## Supported formats

| Format | Read | Notes |
|--------|------|-------|
| PDB | Yes | Topology + coordinates |
| mmCIF | Yes | Topology + coordinates |
| XTC | Yes | Streaming trajectory (GROMACS) |
| DCD | Yes | Trajectory (CHARMM/NAMD) |

## Building from source

Requires [Zig 0.15.2+](https://ziglang.org/download/).

```bash
# Build CLI + shared library
zig build -Doptimize=ReleaseFast

# Run tests
zig build test

# Run validation suite (requires numpy, mdtraj reference data)
uv run --with numpy --script validation/validate.py
```

## License

MIT
