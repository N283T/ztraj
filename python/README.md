# pyztraj

Fast molecular dynamics trajectory analysis powered by Zig.

## Installation

### From PyPI (recommended)

```bash
pip install pyztraj
# or
uv pip install pyztraj
```

Pre-built wheels are available for Linux (x86_64/aarch64), macOS (x86_64/ARM64), and Windows (amd64) with Python 3.11–3.13.

### From source

Requires [Zig 0.15.2+](https://ziglang.org/download/).

```bash
git clone https://github.com/N283T/ztraj.git
cd ztraj/python
pip install .
```

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
