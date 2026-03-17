# ztraj Design Spec

## Overview

ztraj is a high-performance molecular dynamics trajectory analysis library and CLI tool written in Zig. It aims to provide the core analysis capabilities of mdtraj and MDAnalysis with significantly better performance through SIMD vectorization, multi-threading, and explicit allocation control.

## Goals

- **Speed**: Outperform Python-based MD analysis tools through SIMD, multi-threading, and zero-copy I/O
- **Correctness**: All results validated against mdtraj reference implementations
- **Minimal allocations**: User-controllable memory management, arena allocators where possible
- **Simple API**: Free functions operating on plain structs. No OOP abstractions

## Non-Goals (Phase 1)

- Python bindings (Phase 2)
- DSSP secondary structure (separate project: dssp-zig)
- Standalone trajectory alignment / superposition (but RMSD includes internal alignment via QCP)
- PBC wrap/unwrap operations
- GUI or interactive visualization
- Full atom selection DSL (Phase 1 uses index-based and keyword selections only)

## Architecture

### Data Model

Flat structs with free functions. No inheritance, no vtables, no AtomGroup-style views.

```zig
pub const Range = struct { start: u32, len: u32 };

/// Fixed-size string (no heap allocation). Reuses zsasa's FixedString pattern.
pub fn FixedString(comptime N: u8) type {
    return struct {
        data: [N]u8 = [_]u8{0} ** N,
        len: u8 = 0,
        // eql(), eqlSlice(), slice() methods
    };
}

pub const Topology = struct {
    atoms: []Atom,
    residues: []Residue,
    chains: []Chain,
    bonds: []Bond,       // covalent bonds (needed for H-bond donor/acceptor inference)
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Topology) void { ... }
};

pub const Bond = struct {
    atom_i: u32,
    atom_j: u32,
};

pub const Atom = struct {
    name: FixedString(4),       // e.g. "CA", "N", "OG1"
    element: Element,
    residue_index: u32,
};

pub const Residue = struct {
    name: FixedString(5),       // 5 chars for mmCIF compatibility
    chain_index: u32,
    atom_range: Range,
};

/// SOA layout for SIMD-friendly coordinate access.
/// Primary representation for all computation.
pub const Frame = struct {
    x: []f32,                   // natoms
    y: []f32,                   // natoms
    z: []f32,                   // natoms
    box_vectors: ?[3][3]f32,
    time: f32,
    step: i32,
};

/// Full trajectory (for small systems / complete loading)
pub const Trajectory = struct {
    topology: Topology,
    frames: []Frame,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Trajectory) void { ... }
};

/// Streaming iterator for large trajectories (frame-at-a-time)
pub const FrameIterator = struct {
    // Opaque handle to underlying reader (XTC, DCD, etc.)
    // Reuses a single Frame buffer to minimize allocations
    pub fn next(self: *FrameIterator) ?*const Frame { ... }
    pub fn reset(self: *FrameIterator) void { ... }
    pub fn deinit(self: *FrameIterator) void { ... }
};
```

Key decisions:
- **SOA layout** for coordinates (`x[]`, `y[]`, `z[]`) — SIMD-native, no transposition needed. zsasa uses this pattern throughout
- **`f32` for coordinates** (matches XTC/DCD precision, half the memory of f64). Accumulation operations (RMSD, center-of-mass) use `f64` intermediates to avoid catastrophic cancellation
- **`FixedString(N)`** for atom/residue names — carries length, avoids heap allocation per atom. `FixedString(5)` for residue names (mmCIF compatibility, learned from zsasa)
- **`Range`** (start, len) for residue-to-atom mapping (serialization-friendly)
- **`FrameIterator`** for streaming large trajectories without loading all frames into memory
- **Ownership**: every struct with an allocator field owns its data and has `deinit()`. Callers never free individual fields
- Optional box vectors (not all simulations use PBC)

### Unit Convention

Internal unit: **angstroms** (consistent with mdtraj and PDB convention).
- XTC files store in nanometers → multiply by 10.0 at read time
- DCD files store in angstroms → no conversion
- PDB/mmCIF files store in angstroms → no conversion

Conversion happens at the I/O boundary only.

### Error Handling

Each module defines its own error set (e.g., `PdbError`, `DcdError`, `RmsdError`). Errors propagate via Zig error unions. The CLI catches errors at the top level and prints human-readable messages to stderr.

### Atom Selection (Phase 1 — Minimal)

Phase 1 supports three selection modes for CLI:
- **Index-based**: `--atoms 0,1,5-10` (0-based atom indices)
- **Name-based**: `--select "name CA"`, `--select "element O"`
- **Keywords**: `--select backbone`, `--select protein`, `--select water`

A full selection DSL (MDAnalysis-style `"resid 1:10 and name CA"`) is deferred to Phase 2.

### Module Structure

```
src/
├── root.zig              # Public API re-exports
├── main.zig              # CLI entry point
├── types.zig             # Core data structures (Topology, Frame, Trajectory)
├── element.zig           # Periodic table / element data
│
├── io/                   # File format readers
│   ├── pdb.zig           # PDB parser
│   ├── mmcif.zig         # mmCIF/PDBx parser
│   ├── xtc.zig           # XTC reader (via zxdrfile)
│   └── dcd.zig           # DCD reader
│
├── geometry/             # Structural analysis
│   ├── distances.zig     # Pairwise distances
│   ├── angles.zig        # Bond angles
│   ├── dihedrals.zig     # Dihedral angles
│   ├── rmsd.zig          # RMSD (Kabsch / QCP algorithm)
│   ├── rmsf.zig          # RMSF (per-atom fluctuation)
│   ├── rg.zig            # Radius of gyration
│   ├── center.zig        # Center of mass / geometry
│   └── inertia.zig       # Inertia tensor, principal moments
│
├── analysis/             # Interaction analysis
│   ├── hbonds.zig        # Hydrogen bond detection
│   ├── contacts.zig      # Residue-residue contacts
│   └── rdf.zig           # Radial distribution function
│
├── select.zig            # Atom selection (index, name, keyword)
├── simd.zig              # SIMD utilities (from zsasa patterns)
├── thread_pool.zig       # Thread pool (from zsasa patterns)
├── neighbor_list.zig     # Spatial neighbor search
└── mmap_reader.zig       # Memory-mapped file I/O
```

### Dependencies

- **zxdrfile**: XTC/XDR file reading (existing, maintained by user)
- **No other external dependencies**

zsasa code (parsers, simd, thread_pool, neighbor_list, mmap_reader) is copied/adapted, not imported as a dependency. This keeps ztraj self-contained and allows format-specific optimizations.

### Performance Strategy

From day one:

1. **SIMD** — Vectorized distance/angle/dihedral calculations. Based on zsasa's `simd.zig` patterns
2. **Multi-threading** — Per-frame parallelism for trajectory-wide analysis. Based on zsasa's `thread_pool.zig`
3. **Memory-mapped I/O** — Large trajectory files read via mmap where possible
4. **Arena allocators** — Per-analysis temporary allocations via arena, freed in bulk
5. **SOA layout** — Coordinates stored as separate x/y/z arrays (the canonical Frame representation), directly consumable by SIMD without transposition

### CLI Design

```bash
# RMSD against reference frame
ztraj rmsd trajectory.xtc --top structure.pdb --ref 0

# Pairwise distances for specific atom pairs
ztraj distances trajectory.xtc --top structure.pdb --pairs "1-10,2-20"

# Radius of gyration over trajectory
ztraj rg trajectory.xtc --top structure.pdb --select backbone

# RDF between two atom selections
ztraj rdf trajectory.xtc --top structure.pdb --sel1 "O" --sel2 "H"

# Hydrogen bonds
ztraj hbonds trajectory.xtc --top structure.pdb

# Output formats: JSON (default), CSV, TSV
ztraj rmsd traj.xtc --top top.pdb --format csv > rmsd.csv
```

Pattern: `ztraj <command> <trajectory> --top <topology> [options]`

Output to stdout by default (composable with pipes), file output via `--output`.

## Phase 1 Scope

### File I/O
- PDB reader (topology + single frame)
- mmCIF reader (topology + single frame)
- XTC reader (multi-frame trajectory via zxdrfile)
- DCD reader (multi-frame trajectory)

### Geometry Analysis
| Function | Description | Reference (mdtraj) |
|----------|-------------|-------------------|
| `rmsd` | RMSD between frames (QCP algorithm with internal optimal alignment) | `md.rmsd()` |
| `rmsf` | Per-atom RMSF over trajectory | `md.rmsf()` |
| `distances` | Pairwise atom distances | `md.compute_distances()` |
| `angles` | 3-atom angle calculation | `md.compute_angles()` |
| `dihedrals` | 4-atom dihedral angle | `md.compute_dihedrals()` |
| `rg` | Radius of gyration | `md.compute_rg()` |
| `center_of_mass` | Center of mass | `md.compute_center_of_mass()` |
| `inertia_tensor` | Moment of inertia tensor | `md.compute_inertia_tensor()` |

### Interaction Analysis
| Function | Description | Reference |
|----------|-------------|-----------|
| `hbonds` | Hydrogen bond detection (distance + angle) | `md.baker_hubbard()` |
| `contacts` | Residue-residue contact distances | `md.compute_contacts()` |
| `rdf` | Radial distribution function g(r) | `md.compute_rdf()` |

### Validation

Each analysis function is validated against mdtraj:

1. Prepare test trajectories (small systems, ~100 atoms, ~10 frames)
2. Compute reference values with mdtraj (Python scripts in `validation/`)
3. Compare ztraj output within per-function tolerances:
   - Simple distances/angles: `1e-5` (single f32 operation)
   - Accumulated values (RMSD, Rg, center-of-mass): `1e-4` (f64 intermediates keep this tight)
   - RDF histogram bins: `1e-3` (binning introduces discretization)
4. Store reference values as test fixtures

### Provenance of Copied Code

Files adapted from zsasa (source: `/Users/nagaet/freesasa-zig/src/`):
- `simd.zig` — SIMD vector utilities
- `thread_pool.zig` — Work-stealing thread pool
- `neighbor_list.zig` — Cell-list spatial search
- `mmap_reader.zig` — Memory-mapped file reader
- `element.zig` — Periodic table data
- `io/pdb.zig` — Adapted from `pdb_parser.zig` (extended for full topology)
- `io/mmcif.zig` — Adapted from `mmcif_parser.zig` (extended for full topology)
- `io/dcd.zig` — Adapted from `dcd.zig`

Divergence from zsasa is expected and intentional.

### Build System

- `build.zig` with Nix flake (same pattern as zsasa)
- `zig build test` for unit tests
- `zig build -Doptimize=ReleaseFast` for CLI binary

## Phase 2 (Future)

- Python bindings (C API + ctypes, NumPy array interop)
- DSSP integration (from dssp-zig project)
- Trajectory alignment / fitting
- PBC wrap/unwrap
- Additional formats (TRR, GRO, NetCDF)
- Protein-specific dihedrals (phi/psi/chi)
- PyPI wheel distribution
