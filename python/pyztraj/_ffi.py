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
                   double* result, uint32_t n_threads);

    // I/O: Structure loaders (PDB, GRO, mmCIF)
    int ztraj_load_pdb(const char* path, void** handle_out);
    int ztraj_load_gro(const char* path, void** handle_out);
    int ztraj_load_mmcif(const char* path, void** handle_out);
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

    // I/O: TRR
    int ztraj_open_trr(const char* path, size_t* n_atoms_out, void** handle_out);
    int ztraj_read_trr_frame(void* handle, float* x, float* y, float* z,
                             float* time, int32_t* step);
    void ztraj_close_trr(void* handle);

    // I/O: DCD
    int ztraj_open_dcd(const char* path, size_t* n_atoms_out, void** handle_out);
    int ztraj_read_dcd_frame(void* handle, float* x, float* y, float* z,
                             float* time, int32_t* step);
    void ztraj_close_dcd(void* handle);

    // I/O: AMBER PRMTOP
    int ztraj_load_prmtop(const char* path, void** handle_out);

    // I/O: AMBER NetCDF
    int ztraj_open_nc(const char* path, size_t* n_atoms_out, void** handle_out);
    int ztraj_read_nc_frame(void* handle, float* x, float* y, float* z,
                            float* time, int32_t* step);
    void ztraj_close_nc(void* handle);

    // Analysis: RDF
    int ztraj_rdf(const float* sel1_x, const float* sel1_y, const float* sel1_z,
                  size_t n_sel1,
                  const float* sel2_x, const float* sel2_y, const float* sel2_z,
                  size_t n_sel2,
                  double box_volume, float r_min, float r_max, uint32_t n_bins,
                  double* r_out, double* g_r_out, uint32_t n_threads);

    // Analysis: Hydrogen Bonds
    typedef struct { uint32_t donor; uint32_t hydrogen; uint32_t acceptor;
                     float distance; float angle; } CHBond;
    int ztraj_detect_hbonds(void* structure_handle,
                            const float* x, const float* y, const float* z,
                            size_t n_atoms, float dist_cutoff, float angle_cutoff,
                            CHBond* hbonds_out, size_t capacity, size_t* n_found,
                            uint32_t n_threads);

    // Analysis: Contacts
    #define ZTRAJ_SCHEME_CLOSEST 0
    #define ZTRAJ_SCHEME_CA 1
    #define ZTRAJ_SCHEME_CLOSEST_HEAVY 2
    typedef struct { uint32_t residue_i; uint32_t residue_j;
                     float distance; } CContact;
    int ztraj_compute_contacts(void* structure_handle,
                               const float* x, const float* y, const float* z,
                               size_t n_atoms, int scheme, float cutoff,
                               CContact* contacts_out, size_t capacity,
                               size_t* n_found, uint32_t n_threads);

    // MSD
    int ztraj_msd(const float* all_x, const float* all_y, const float* all_z,
                  size_t n_frames, size_t n_atoms,
                  const uint32_t* atom_indices, size_t n_indices,
                  double* result, uint32_t n_threads);

    // PCA
    int ztraj_pca_covariance(const float* all_x, const float* all_y, const float* all_z,
                             size_t n_frames, size_t n_atoms,
                             const uint32_t* atom_indices, size_t n_indices,
                             double* cov_out, uint32_t n_threads);

    // PBC
    int ztraj_wrap_coords(float* x, float* y, float* z,
                          size_t n_atoms, const float* box);
    int ztraj_minimum_image_distance(float x1, float y1, float z1,
                                     float x2, float y2, float z2,
                                     const float* box, float* result);
    int ztraj_make_molecules_whole(void* structure_handle,
                                   float* x, float* y, float* z,
                                   size_t n_atoms, const float* box);

    // Analysis: Native contacts
    int ztraj_native_contacts_q(const float* ref_x, const float* ref_y, const float* ref_z,
                                const float* x, const float* y, const float* z,
                                size_t n_atoms,
                                const uint32_t* indices_a, size_t n_a,
                                const uint32_t* indices_b, size_t n_b,
                                float cutoff, double* result);

    // Analysis: SASA
    int ztraj_compute_sasa(void* structure_handle,
                           const float* x, const float* y, const float* z,
                           size_t n_atoms, uint32_t n_points, double probe_radius,
                           size_t n_threads, uint32_t algorithm,
                           double* atom_areas, double* total_area);

    // Protein dihedrals
    int ztraj_compute_phi(void* handle, const float* x, const float* y, const float* z,
                          size_t n_atoms, float* result, size_t* n_residues_out);
    int ztraj_compute_psi(void* handle, const float* x, const float* y, const float* z,
                          size_t n_atoms, float* result, size_t* n_residues_out);
    int ztraj_compute_omega(void* handle, const float* x, const float* y, const float* z,
                            size_t n_atoms, float* result, size_t* n_residues_out);
    int ztraj_compute_chi(void* handle, const float* x, const float* y, const float* z,
                          size_t n_atoms, uint8_t chi_level, float* result, size_t* n_residues_out);

    // DSSP
    int ztraj_compute_dssp(void* handle, const float* x, const float* y, const float* z,
                           size_t n_atoms, uint8_t* result, size_t* n_residues_out,
                           uint32_t n_threads);

    // File writers: structure
    int ztraj_write_pdb(void* handle, const float* x, const float* y, const float* z,
                        size_t n_atoms, const char* path);
    int ztraj_write_gro(void* handle, const float* x, const float* y, const float* z,
                        size_t n_atoms, const char* path);

    // File writers: XTC trajectory
    int ztraj_open_xtc_writer(const char* path, size_t n_atoms, void** handle_out);
    int ztraj_write_xtc_frame(void* handle, const float* x, const float* y, const float* z,
                              size_t n_atoms, float time, int32_t step);
    int ztraj_close_xtc_writer(void* handle);

    // File writers: TRR trajectory
    int ztraj_open_trr_writer(const char* path, size_t n_atoms, void** handle_out);
    int ztraj_write_trr_frame(void* handle, const float* x, const float* y, const float* z,
                              size_t n_atoms, float time, int32_t step);
    int ztraj_close_trr_writer(void* handle);

    // File writers: AMBER NetCDF trajectory
    int ztraj_open_nc_writer(const char* path, uint32_t n_atoms, _Bool has_cell,
                             void** handle_out);
    int ztraj_write_nc_frame(void* handle, const float* x, const float* y, const float* z,
                             size_t n_atoms, float time, int32_t step);
    int ztraj_close_nc_writer(void* handle);

    // Atom selection
    int ztraj_select_keyword(void* handle, int keyword, uint32_t** indices_out, size_t* count_out);
    int ztraj_select_name(void* handle, const char* name,
                          uint32_t** indices_out, size_t* count_out);
    void ztraj_free_selection(uint32_t* indices, size_t count);
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
