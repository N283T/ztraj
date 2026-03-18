# pyztraj

Fast molecular dynamics trajectory analysis powered by Zig.

## Installation

```bash
pip install pyztraj
```

Requires Zig 0.15.2+ for building from source.

## Usage

```python
import pyztraj

# Load a PDB structure
struct = pyztraj.load_pdb("structure.pdb")

# Compute pairwise distances
distances = pyztraj.compute_distances(struct.coords, pairs)

# Stream XTC trajectory frames
with pyztraj.open_xtc("trajectory.xtc", struct.n_atoms) as reader:
    for frame in reader:
        rg = pyztraj.compute_rg(frame.coords, struct.masses)
```
