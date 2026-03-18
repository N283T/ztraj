# Python Bindings for ztraj

## Overview

Add Python bindings to ztraj as a standalone package `pyztraj`. Follows the proven
zsasa pattern: Zig C API → shared library → CFFI → Python. The package is
self-contained with its own I/O (no MDTraj/MDAnalysis dependency required).

## Architecture

```
src/c_api.zig          → callconv(.c) exports for all functions + I/O
build.zig              → shared library target (libztraj.so/.dylib/.dll)
python/
  pyproject.toml       → Hatchling + custom build hook
  hatch_build.py       → Runs zig build during pip install
  pyztraj/
    __init__.py        → Public API
    _ffi.py            → CFFI library loading + cdef declarations
    core.py            → High-level Python API (NumPy in/out)
  tests/
    test_core.py       → Unit tests
    conftest.py        → Shared fixtures
```

## Design Decisions

1. **Standalone package** — pyztraj loads PDB/XTC/DCD itself via C API. No MDTraj
   or MDAnalysis required. Users who want integration can convert coordinates.
2. **Caller-owned buffers** — All C API functions take input and write to
   pre-allocated output. No Zig-allocated memory crosses FFI boundary, except for
   variable-length results (hbonds/contacts) which use a two-pass pattern.
3. **AOS↔SOA at boundary** — Python API accepts (n_atoms, 3) arrays (standard
   convention). Conversion to SOA (x[], y[], z[]) happens in Python before FFI call.
4. **Units: Angstroms throughout** — Same as ztraj internal convention.

## Phase 1: C API + Shared Library

### 1.1 Create `src/c_api.zig`

Error codes and version:
```zig
pub const ZTRAJ_OK: c_int = 0;
pub const ZTRAJ_ERROR_INVALID_INPUT: c_int = -1;
pub const ZTRAJ_ERROR_OUT_OF_MEMORY: c_int = -2;
pub const ZTRAJ_ERROR_FILE_IO: c_int = -3;
pub const ZTRAJ_ERROR_PARSE: c_int = -4;

export fn ztraj_version() callconv(.c) [*:0]const u8;
```

Geometry functions (single-frame, caller-owned buffers):
```zig
export fn ztraj_distances(x, y, z: [*]const f32, n_atoms: usize,
    pairs: [*]const u32, n_pairs: usize, result: [*]f32) callconv(.c) c_int;
export fn ztraj_angles(x, y, z: [*]const f32, n_atoms: usize,
    triplets: [*]const u32, n_triplets: usize, result: [*]f32) callconv(.c) c_int;
export fn ztraj_dihedrals(x, y, z: [*]const f32, n_atoms: usize,
    quartets: [*]const u32, n_quartets: usize, result: [*]f32) callconv(.c) c_int;
export fn ztraj_rmsd(ref_x, ref_y, ref_z, x, y, z: [*]const f32,
    atom_indices: ?[*]const u32, n_indices: usize, n_atoms: usize,
    result: *f64) callconv(.c) c_int;
export fn ztraj_rg(x, y, z: [*]const f32, masses: [*]const f64,
    atom_indices: ?[*]const u32, n_indices: usize, n_atoms: usize,
    result: *f64) callconv(.c) c_int;
export fn ztraj_center_of_mass(x, y, z: [*]const f32, masses: [*]const f64,
    atom_indices: ?[*]const u32, n_indices: usize, n_atoms: usize,
    cx: *f64, cy: *f64, cz: *f64) callconv(.c) c_int;
export fn ztraj_inertia(x, y, z: [*]const f32, masses: [*]const f64,
    atom_indices: ?[*]const u32, n_indices: usize, n_atoms: usize,
    result: [*]f64) callconv(.c) c_int;  // 9 elements, row-major
```

I/O functions (opaque handle pattern):
```zig
// PDB loading — allocates topology + frame, returns opaque handle
export fn ztraj_load_pdb(path: [*:0]const u8, handle: *?*anyopaque) callconv(.c) c_int;
// Accessors for loaded structure
export fn ztraj_get_n_atoms(handle: *anyopaque) callconv(.c) usize;
export fn ztraj_get_coords(handle: *anyopaque, x: [*]f32, y: [*]f32, z: [*]f32) callconv(.c) c_int;
export fn ztraj_get_masses(handle: *anyopaque, masses: [*]f64) callconv(.c) c_int;
export fn ztraj_get_atom_names(handle: *anyopaque, names: [*][4]u8) callconv(.c) c_int;
export fn ztraj_get_residue_names(handle: *anyopaque, names: [*][5]u8) callconv(.c) c_int;
export fn ztraj_free_structure(handle: *anyopaque) callconv(.c) void;

// XTC streaming — opaque reader handle
export fn ztraj_open_xtc(path: [*:0]const u8, n_atoms: usize,
    handle: *?*anyopaque) callconv(.c) c_int;
export fn ztraj_read_xtc_frame(handle: *anyopaque,
    x: [*]f32, y: [*]f32, z: [*]f32) callconv(.c) c_int;  // returns ZTRAJ_OK or EOF
export fn ztraj_close_xtc(handle: *anyopaque) callconv(.c) void;
```

### 1.2 Add shared library target to `build.zig`

Add `b.addSharedLibrary()` with c_api.zig as root, link libc.

### 1.3 Add C API tests

Test blocks in c_api.zig calling exported functions to verify wrapping.

- [ ] **DONE** - Phase 1 complete

## Phase 2: Python Package Scaffold

### 2.1 Create `python/pyproject.toml`
- Package name: `pyztraj`
- Dependencies: `numpy>=1.24`, `cffi>=1.15`
- Build: Hatchling + custom hook
- Dev: pytest, pytest-cov, ruff, ty

### 2.2 Create `python/hatch_build.py`
- Adapt zsasa's hook: change names, remove CLI binary copy

### 2.3 Create `python/pyztraj/_ffi.py`
- CFFI cdef declarations matching C API
- Library loading (platform-specific .so/.dylib/.dll)

### 2.4 Create `python/pyztraj/__init__.py`
- Re-export public API from core

- [ ] **DONE** - Phase 2 complete

## Phase 3: Core Python API + Tests

### 3.1 Create `python/pyztraj/core.py`

High-level API returning NumPy arrays:

```python
# I/O
def load_pdb(path: str) -> Structure
def open_xtc(path: str, n_atoms: int) -> XtcReader (context manager)

# Single-frame geometry (accept (n_atoms, 3) f32 arrays)
def compute_distances(coords: NDArray, pairs: NDArray) -> NDArray[f32]
def compute_angles(coords: NDArray, triplets: NDArray) -> NDArray[f32]
def compute_dihedrals(coords: NDArray, quartets: NDArray) -> NDArray[f32]
def compute_rmsd(coords: NDArray, ref_coords: NDArray, atom_indices=None) -> float
def compute_rg(coords: NDArray, masses: NDArray, atom_indices=None) -> float
def compute_center_of_mass(coords: NDArray, masses: NDArray, atom_indices=None) -> NDArray[f64]
def compute_inertia(coords: NDArray, masses: NDArray, atom_indices=None) -> NDArray[f64]  # (3,3)

# Multi-frame convenience (loops over trajectory)
def trajectory_rmsd(xtc_path, pdb_path, select=None, ref_frame=0) -> NDArray
def trajectory_rg(xtc_path, pdb_path, select=None) -> NDArray
```

`Structure` dataclass:
```python
@dataclass
class Structure:
    coords: NDArray[np.float32]    # (n_atoms, 3)
    masses: NDArray[np.float64]    # (n_atoms,)
    atom_names: list[str]          # length n_atoms
    residue_names: list[str]       # length n_atoms (per-atom residue name)
    n_atoms: int
```

### 3.2 Create `python/tests/test_core.py`

Test against known geometries + validation test data (3tvj_I.pdb / XTC).

### 3.3 Create `python/tests/conftest.py`

Shared fixtures: simple coords, PDB path, etc.

- [ ] **DONE** - Phase 3 complete

## Phase 4: Analysis Functions (hbonds, contacts, rdf)

### 4.1 Add to C API

hbonds — two-pass pattern:
```zig
export fn ztraj_hbonds_count(...) callconv(.c) c_int;  // returns count
export fn ztraj_hbonds_detect(..., buf: [*]ZtrajHBond, capacity: usize,
    n_found: *usize) callconv(.c) c_int;
```

contacts — same two-pass pattern.
rdf — caller allocates r[] and g_r[] of n_bins size.

### 4.2 Add Python wrappers + tests

- [ ] **DONE** - Phase 4 complete

## Phase 5: RMSF (Multi-Frame C API)

### 5.1 Add to C API

```zig
// Flat contiguous layout: all_x[frame * n_atoms + atom]
export fn ztraj_rmsf(all_x, all_y, all_z: [*]const f32,
    n_frames: usize, n_atoms: usize,
    atom_indices: ?[*]const u32, n_indices: usize,
    result: [*]f64) callconv(.c) c_int;
```

### 5.2 Add Python wrapper + test

- [ ] **DONE** - Phase 5 complete

## Phase 6: CI / Distribution

### 6.1 GitHub Actions cibuildwheel workflow
### 6.2 Update README + CHANGELOG

- [ ] **DONE** - Phase 6 complete

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Opaque handle for I/O across FFI | Medium | Follow zsasa pattern, use c_allocator |
| hbonds/contacts variable-length output | Medium | Two-pass count→allocate→fill |
| RMSF multi-frame data layout | Medium | Flat contiguous arrays, frame-major |
| SOA/AOS conversion overhead | Low | NumPy column slicing is fast |
| Cross-platform library loading | Low | Proven zsasa pattern |

## Estimated Complexity: Medium-High

Phases 1-3 are straightforward (proven pattern). Phase 4-5 have FFI complexity.

---
- [ ] **DONE** - Plan complete
