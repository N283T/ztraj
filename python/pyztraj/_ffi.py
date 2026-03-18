"""CFFI bindings for libztraj shared library."""

from __future__ import annotations

import sys
from pathlib import Path

import cffi

_CDEF = """
    // Error codes
    #define ZTRAJ_OK 0
    #define ZTRAJ_ERROR_INVALID_INPUT -1
    #define ZTRAJ_ERROR_OUT_OF_MEMORY -2
    #define ZTRAJ_ERROR_FILE_IO -3
    #define ZTRAJ_ERROR_PARSE -4
    #define ZTRAJ_ERROR_EOF -5

    // Version
    const char* ztraj_version(void);

    // Geometry
    int ztraj_distances(const float* x, const float* y, const float* z,
                        size_t n_atoms, const uint32_t* pairs, size_t n_pairs,
                        float* result);

    int ztraj_angles(const float* x, const float* y, const float* z,
                     size_t n_atoms, const uint32_t* triplets, size_t n_triplets,
                     float* result);

    int ztraj_dihedrals(const float* x, const float* y, const float* z,
                        size_t n_atoms, const uint32_t* quartets, size_t n_quartets,
                        float* result);

    int ztraj_rmsd(const float* ref_x, const float* ref_y, const float* ref_z,
                   const float* x, const float* y, const float* z,
                   const uint32_t* atom_indices, size_t n_indices,
                   size_t n_atoms, double* result);

    int ztraj_rg(const float* x, const float* y, const float* z,
                 const double* masses, const uint32_t* atom_indices,
                 size_t n_indices, size_t n_atoms, double* result);

    int ztraj_center_of_mass(const float* x, const float* y, const float* z,
                             const double* masses, const uint32_t* atom_indices,
                             size_t n_indices, size_t n_atoms,
                             double* cx, double* cy, double* cz);

    int ztraj_center_of_geometry(const float* x, const float* y, const float* z,
                                 const uint32_t* atom_indices, size_t n_indices,
                                 size_t n_atoms,
                                 double* cx, double* cy, double* cz);

    int ztraj_inertia(const float* x, const float* y, const float* z,
                      const double* masses, const uint32_t* atom_indices,
                      size_t n_indices, size_t n_atoms, double* result);

    int ztraj_principal_moments(const double* tensor, double* result);

    int ztraj_rmsf(const float* all_x, const float* all_y, const float* all_z,
                   size_t n_frames, size_t n_atoms,
                   const uint32_t* atom_indices, size_t n_indices,
                   double* result);

    // I/O: PDB
    int ztraj_load_pdb(const char* path, void** handle_out);
    size_t ztraj_get_n_atoms(void* handle);
    int ztraj_get_coords(void* handle, float* x, float* y, float* z);
    int ztraj_get_masses(void* handle, double* masses);
    int ztraj_get_atom_names(void* handle, char* names);
    int ztraj_get_residue_names(void* handle, char* names);
    int ztraj_get_resids(void* handle, int32_t* resids);
    void ztraj_free_structure(void* handle);

    // I/O: XTC
    int ztraj_open_xtc(const char* path, size_t* n_atoms_out, void** handle_out);
    int ztraj_read_xtc_frame(void* handle, float* x, float* y, float* z,
                             float* time, int32_t* step);
    void ztraj_close_xtc(void* handle);
"""

_ffi = cffi.FFI()
_ffi.cdef(_CDEF)

_lib = None


def _load_library():
    """Load the libztraj shared library."""
    global _lib  # noqa: PLW0603
    if _lib is not None:
        return _lib

    pkg_dir = Path(__file__).parent

    if sys.platform == "darwin":
        lib_name = "libztraj.dylib"
    elif sys.platform == "win32":
        lib_name = "ztraj.dll"
    else:
        lib_name = "libztraj.so"

    lib_path = pkg_dir / lib_name
    if not lib_path.exists():
        msg = (
            f"Shared library not found at {lib_path}. "
            "Install with 'pip install .' from the python/ directory, or build "
            "manually with 'zig build -Doptimize=ReleaseFast' and copy the "
            "library to the pyztraj package directory."
        )
        raise FileNotFoundError(msg)

    try:
        _lib = _ffi.dlopen(str(lib_path))
    except OSError as e:
        msg = (
            f"Failed to load shared library at {lib_path}: {e}. "
            "This may indicate an architecture mismatch or ABI incompatibility. "
            "Try rebuilding: 'zig build -Doptimize=ReleaseFast'"
        )
        raise OSError(msg) from e
    return _lib


def get_lib():
    """Get the loaded C library instance."""
    return _load_library()


def get_ffi() -> cffi.FFI:
    """Get the FFI instance."""
    return _ffi
