# ztraj

High-performance molecular dynamics trajectory analysis library + CLI in Zig, with Python bindings (pyztraj).

## Architecture

- **Zig core** (`src/`): Library + CLI, flat structs + free functions (Zig-idiomatic, no OOP)
- **Python bindings** (`python/`): cffi-based Python package (pyztraj) wrapping the Zig shared library
- **Validation** (`validation/`): Cross-validation scripts against mdtraj reference values

### Key Design Decisions

- SOA coordinate layout (x[], y[], z[]) for SIMD vectorization
- f32 coordinates, f64 intermediate precision for accumulation
- Internal units: angstroms (XTC nm→Å conversion at I/O boundary)
- FrameIterator for streaming large trajectories
- zxdrfile as only external Zig dependency

## Build & Test

```bash
# Build CLI + shared library
zig build -Doptimize=ReleaseFast

# Run Zig tests
zig build test

# Build Python bindings
cd python && pip install .

# Run Python tests
cd python && pytest tests -x -q

# Lint Python
cd python && ruff format . && ruff check --fix . && ty check

# Run validation suite
uv run --with numpy --script validation/validate.py
```

## Project Structure

```
src/
  main.zig          # CLI entry point
  root.zig          # Library root (public API)
  c_api.zig         # C ABI exports for Python bindings
  types.zig         # Core types (Topology, Frame, Atom, etc.)
  element.zig       # Periodic table data
  select.zig        # Atom selection language
  simd.zig          # SIMD primitives
  thread_pool.zig   # Multi-threaded execution
  output.zig        # Output formatting (JSON, CSV, TSV)
  io/               # Format parsers (PDB, mmCIF, GRO, XTC, TRR, DCD)
  geometry/         # Geometry analysis (RMSD, RMSF, Rg, distances, angles, etc.)
  analysis/         # Interaction analysis (H-bonds, contacts, RDF, DSSP, SASA, etc.)
  cli/              # CLI argument parsing and command runners
python/
  pyztraj/          # Python package source
  tests/            # Python tests
validation/         # Cross-validation against mdtraj
test_data/          # Test PDB/XTC/DCD files
```

## Conventions

- All code, comments, commits, and docs in English
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Never commit directly to main; always use feature branches + PR
