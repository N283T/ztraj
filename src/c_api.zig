//! C ABI interface for ztraj library.
//!
//! Provides C-compatible functions callable from Python, C, etc. via FFI.
//! All functions use caller-owned buffers — no Zig-allocated memory crosses
//! the FFI boundary except for opaque handles (which must be freed by the
//! corresponding close/free function).

const std = @import("std");
const types = @import("types.zig");
const distances_mod = @import("geometry/distances.zig");
const angles_mod = @import("geometry/angles.zig");
const dihedrals_mod = @import("geometry/dihedrals.zig");
const rmsd_mod = @import("geometry/rmsd.zig");
const rg_mod = @import("geometry/rg.zig");
const center_mod = @import("geometry/center.zig");
const inertia_mod = @import("geometry/inertia.zig");
const rmsf_mod = @import("geometry/rmsf.zig");
const pdb_mod = @import("io/pdb.zig");
const gro_mod = @import("io/gro.zig");
const mmcif_mod = @import("io/mmcif.zig");
const xtc_mod = @import("io/xtc.zig");
const trr_mod = @import("io/trr.zig");
const dcd_mod = @import("io/dcd.zig");
const hbonds_mod = @import("analysis/hbonds.zig");
const contacts_mod = @import("analysis/contacts.zig");
const rdf_mod = @import("analysis/rdf.zig");
const sasa_mod = @import("analysis/sasa.zig");
const native_contacts_mod = @import("analysis/native_contacts.zig");
const msd_mod = @import("analysis/msd.zig");
const pca_mod = @import("analysis/pca.zig");
const pbc_mod = @import("geometry/pbc.zig");
const protein_dihedrals_mod = @import("geometry/protein_dihedrals.zig");
const dssp_mod = @import("analysis/dssp/dssp.zig");
const dssp_types = @import("analysis/dssp/types.zig");
const select_mod = @import("select.zig");

// =============================================================================
// Error Codes
// =============================================================================

/// No error.
pub const ZTRAJ_OK: c_int = 0;
/// Invalid input parameters (null pointers, zero counts, etc.).
pub const ZTRAJ_ERROR_INVALID_INPUT: c_int = -1;
/// Memory allocation failed.
pub const ZTRAJ_ERROR_OUT_OF_MEMORY: c_int = -2;
/// File I/O error (file not found, read error).
pub const ZTRAJ_ERROR_FILE_IO: c_int = -3;
/// Parse error (malformed file).
pub const ZTRAJ_ERROR_PARSE: c_int = -4;
/// End of file (for streaming readers).
pub const ZTRAJ_ERROR_EOF: c_int = -5;

// =============================================================================
// Version
// =============================================================================

const VERSION: [*:0]const u8 = "0.4.1";

/// Get the library version string.
export fn ztraj_version() callconv(.c) [*:0]const u8 {
    return VERSION;
}

// =============================================================================
// Thread-safe allocator for C API
// =============================================================================

const c_allocator = std.heap.c_allocator;

/// Maximum PDB file size (100 MiB) to prevent excessive memory allocation.
const MAX_PDB_FILE_SIZE = 100 * 1024 * 1024;

// =============================================================================
// Geometry: Distances
// =============================================================================

/// Compute pairwise Euclidean distances.
///
/// `pairs` is a flat array of u32 with length `n_pairs * 2`, interpreted as
/// consecutive (i, j) pairs. `result` must have length `n_pairs`.
export fn ztraj_distances(
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    pairs: [*]const u32,
    n_pairs: usize,
    result: [*]f32,
) callconv(.c) c_int {
    if (n_pairs == 0) return ZTRAJ_OK;
    if (n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;

    // Bounds-check indices
    const flat = pairs[0 .. n_pairs * 2];
    for (flat) |idx| {
        if (idx >= n_atoms) return ZTRAJ_ERROR_INVALID_INPUT;
    }

    const pairs_slice: [*]const [2]u32 = @ptrCast(@alignCast(pairs));
    distances_mod.compute(
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        pairs_slice[0..n_pairs],
        result[0..n_pairs],
    );

    return ZTRAJ_OK;
}

// =============================================================================
// Geometry: Angles
// =============================================================================

/// Compute bond angles (in radians) for atom triplets.
///
/// `triplets` is a flat array of u32 with length `n_triplets * 3`.
/// `result` must have length `n_triplets`.
export fn ztraj_angles(
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    triplets: [*]const u32,
    n_triplets: usize,
    result: [*]f32,
) callconv(.c) c_int {
    if (n_triplets == 0) return ZTRAJ_OK;
    if (n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;

    const flat = triplets[0 .. n_triplets * 3];
    for (flat) |idx| {
        if (idx >= n_atoms) return ZTRAJ_ERROR_INVALID_INPUT;
    }

    const triplets_slice: [*]const [3]u32 = @ptrCast(@alignCast(triplets));
    angles_mod.compute(
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        triplets_slice[0..n_triplets],
        result[0..n_triplets],
    );

    return ZTRAJ_OK;
}

// =============================================================================
// Geometry: Dihedrals
// =============================================================================

/// Compute dihedral angles (in radians, range [-pi, pi]) for atom quartets.
///
/// `quartets` is a flat array of u32 with length `n_quartets * 4`.
/// `result` must have length `n_quartets`.
export fn ztraj_dihedrals(
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    quartets: [*]const u32,
    n_quartets: usize,
    result: [*]f32,
) callconv(.c) c_int {
    if (n_quartets == 0) return ZTRAJ_OK;
    if (n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;

    const flat = quartets[0 .. n_quartets * 4];
    for (flat) |idx| {
        if (idx >= n_atoms) return ZTRAJ_ERROR_INVALID_INPUT;
    }

    const quartets_slice: [*]const [4]u32 = @ptrCast(@alignCast(quartets));
    dihedrals_mod.compute(
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        quartets_slice[0..n_quartets],
        result[0..n_quartets],
    );

    return ZTRAJ_OK;
}

// =============================================================================
// Geometry: RMSD
// =============================================================================

/// Compute RMSD between two structures using the QCP algorithm.
///
/// If `atom_indices` is non-null and `n_indices > 0`, only those atoms are used.
/// Otherwise all `n_atoms` atoms are used.
/// Result is written to `result`.
export fn ztraj_rmsd(
    ref_x: [*]const f32,
    ref_y: [*]const f32,
    ref_z: [*]const f32,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    atom_indices: ?[*]const u32,
    n_indices: usize,
    n_atoms: usize,
    result: *f64,
) callconv(.c) c_int {
    if (n_atoms == 0) {
        result.* = 0.0;
        return ZTRAJ_OK;
    }

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    result.* = rmsd_mod.compute(
        ref_x[0..n_atoms],
        ref_y[0..n_atoms],
        ref_z[0..n_atoms],
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        indices,
    );

    return ZTRAJ_OK;
}

// =============================================================================
// Geometry: Radius of Gyration
// =============================================================================

/// Compute mass-weighted radius of gyration.
///
/// If `atom_indices` is non-null and `n_indices > 0`, only those atoms are used.
export fn ztraj_rg(
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    masses: [*]const f64,
    atom_indices: ?[*]const u32,
    n_indices: usize,
    n_atoms: usize,
    result: *f64,
) callconv(.c) c_int {
    if (n_atoms == 0) {
        result.* = 0.0;
        return ZTRAJ_OK;
    }

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    result.* = rg_mod.compute(
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        masses[0..n_atoms],
        indices,
    );

    return ZTRAJ_OK;
}

// =============================================================================
// Geometry: Center of Mass
// =============================================================================

/// Compute mass-weighted center of mass. Result written to cx, cy, cz.
export fn ztraj_center_of_mass(
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    masses: [*]const f64,
    atom_indices: ?[*]const u32,
    n_indices: usize,
    n_atoms: usize,
    cx: *f64,
    cy: *f64,
    cz: *f64,
) callconv(.c) c_int {
    if (n_atoms == 0) {
        cx.* = 0.0;
        cy.* = 0.0;
        cz.* = 0.0;
        return ZTRAJ_OK;
    }

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    const com = center_mod.ofMass(
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        masses[0..n_atoms],
        indices,
    );
    cx.* = com[0];
    cy.* = com[1];
    cz.* = com[2];

    return ZTRAJ_OK;
}

// =============================================================================
// Geometry: Center of Geometry
// =============================================================================

/// Compute unweighted center of geometry. Result written to cx, cy, cz.
export fn ztraj_center_of_geometry(
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    atom_indices: ?[*]const u32,
    n_indices: usize,
    n_atoms: usize,
    cx: *f64,
    cy: *f64,
    cz: *f64,
) callconv(.c) c_int {
    if (n_atoms == 0) {
        cx.* = 0.0;
        cy.* = 0.0;
        cz.* = 0.0;
        return ZTRAJ_OK;
    }

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    const cog = center_mod.ofGeometry(
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        indices,
    );
    cx.* = cog[0];
    cy.* = cog[1];
    cz.* = cog[2];

    return ZTRAJ_OK;
}

// =============================================================================
// Geometry: Inertia Tensor
// =============================================================================

/// Compute 3x3 inertia tensor. `result` must point to 9 f64 values (row-major).
export fn ztraj_inertia(
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    masses: [*]const f64,
    atom_indices: ?[*]const u32,
    n_indices: usize,
    n_atoms: usize,
    result: [*]f64,
) callconv(.c) c_int {
    if (n_atoms == 0) {
        for (0..9) |i| result[i] = 0.0;
        return ZTRAJ_OK;
    }

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    const tensor = inertia_mod.compute(
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        masses[0..n_atoms],
        indices,
    );

    // Flatten [3][3]f64 to row-major [9]f64
    for (0..3) |row| {
        for (0..3) |col| {
            result[row * 3 + col] = tensor[row][col];
        }
    }

    return ZTRAJ_OK;
}

/// Compute principal moments from a 3x3 inertia tensor.
/// `tensor` must point to 9 f64 values (row-major).
/// `result` must point to 3 f64 values (ascending order).
export fn ztraj_principal_moments(
    tensor: [*]const f64,
    result: [*]f64,
) callconv(.c) c_int {
    // Unflatten row-major [9]f64 to [3][3]f64
    var t: [3][3]f64 = undefined;
    for (0..3) |row| {
        for (0..3) |col| {
            t[row][col] = tensor[row * 3 + col];
        }
    }

    const moments = inertia_mod.principalMoments(t);
    result[0] = moments[0];
    result[1] = moments[1];
    result[2] = moments[2];

    return ZTRAJ_OK;
}

// =============================================================================
// Geometry: RMSF (multi-frame)
// =============================================================================

/// Compute per-atom RMSF over multiple frames.
///
/// Coordinate data is flat contiguous: `all_x[frame * n_atoms + atom]`.
/// If `atom_indices` is non-null and `n_indices > 0`, only those atoms are
/// processed and `result` must have `n_indices` elements.
/// Otherwise `result` must have `n_atoms` elements.
export fn ztraj_rmsf(
    all_x: [*]const f32,
    all_y: [*]const f32,
    all_z: [*]const f32,
    n_frames: usize,
    n_atoms: usize,
    atom_indices: ?[*]const u32,
    n_indices: usize,
    result: [*]f64,
    n_threads: u32,
) callconv(.c) c_int {
    if (n_frames == 0 or n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    // Build non-owning Frame views from flat contiguous data.
    const frames = c_allocator.alloc(types.Frame, n_frames) catch return ZTRAJ_ERROR_OUT_OF_MEMORY;
    defer c_allocator.free(frames);

    for (0..n_frames) |f| {
        const offset = f * n_atoms;
        frames[f] = types.Frame.initView(
            all_x[offset .. offset + n_atoms],
            all_y[offset .. offset + n_atoms],
            all_z[offset .. offset + n_atoms],
        );
    }

    const thread_count: usize = if (n_threads == 0) (std.Thread.getCpuCount() catch 1) else @intCast(n_threads);
    const rmsf_result = rmsf_mod.computeParallel(c_allocator, frames, indices, thread_count) catch |err| {
        return switch (err) {
            error.NoFrames => ZTRAJ_ERROR_INVALID_INPUT,
            error.OutOfMemory, error.SystemResources, error.ThreadQuotaExceeded, error.LockedMemoryLimitExceeded, error.Unexpected => ZTRAJ_ERROR_OUT_OF_MEMORY,
        };
    };
    defer c_allocator.free(rmsf_result);

    const n_out = if (indices) |idx| idx.len else n_atoms;
    for (0..n_out) |i| {
        result[i] = rmsf_result[i];
    }

    return ZTRAJ_OK;
}

// =============================================================================
// I/O: PDB Loading (opaque handle)
// =============================================================================

/// Magic number for handle type validation.
const STRUCTURE_MAGIC: u64 = 0xDEAD_BEEF_5DB0_0001;
const XTC_MAGIC: u64 = 0xDEAD_BEEF_58C0_0001;

/// Opaque structure handle returned by ztraj_load_pdb.
const StructureHandle = struct {
    magic: u64 = STRUCTURE_MAGIC,
    parse_result: types.ParseResult,
    masses: []f64,
    allocator: std.mem.Allocator,

    fn deinit(self: *StructureHandle) void {
        self.magic = 0; // Invalidate to detect double-free
        self.allocator.free(self.masses);
        self.parse_result.deinit();
        self.allocator.destroy(self);
    }
};

/// Validate and cast a structure handle. Returns null if invalid.
fn castStructureHandle(handle: ?*anyopaque) ?*StructureHandle {
    const h: *StructureHandle = @ptrCast(@alignCast(handle orelse return null));
    if (h.magic != STRUCTURE_MAGIC) return null;
    return h;
}

/// Load a PDB file and return an opaque handle.
///
/// On success, `handle_out` is set to a non-null opaque pointer.
/// The handle must be freed with ztraj_free_structure().
export fn ztraj_load_pdb(
    path: [*:0]const u8,
    handle_out: *?*anyopaque,
) callconv(.c) c_int {
    handle_out.* = null;

    // Read file
    const path_slice = std.mem.sliceTo(path, 0);
    const data = std.fs.cwd().readFileAlloc(c_allocator, path_slice, MAX_PDB_FILE_SIZE) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    defer c_allocator.free(data);

    // Parse
    var parse_result = pdb_mod.parse(c_allocator, data) catch {
        return ZTRAJ_ERROR_PARSE;
    };
    errdefer parse_result.deinit();

    // Compute masses
    const masses = parse_result.topology.masses(c_allocator) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    errdefer c_allocator.free(masses);

    // Allocate handle
    const handle = c_allocator.create(StructureHandle) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    handle.* = .{
        .parse_result = parse_result,
        .masses = masses,
        .allocator = c_allocator,
    };

    handle_out.* = @ptrCast(handle);
    return ZTRAJ_OK;
}

/// Get number of atoms from a loaded structure. Returns 0 on invalid handle.
export fn ztraj_get_n_atoms(handle: ?*anyopaque) callconv(.c) usize {
    const h = castStructureHandle(handle) orelse return 0;
    return h.parse_result.frame.nAtoms();
}

/// Copy coordinates from a loaded structure into caller-owned buffers.
/// Buffers must have at least n_atoms elements each.
export fn ztraj_get_coords(
    handle: ?*anyopaque,
    x: [*]f32,
    y: [*]f32,
    z: [*]f32,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const frame = h.parse_result.frame;
    const n = frame.nAtoms();

    @memcpy(x[0..n], frame.x);
    @memcpy(y[0..n], frame.y);
    @memcpy(z[0..n], frame.z);

    return ZTRAJ_OK;
}

/// Copy masses from a loaded structure into a caller-owned buffer.
/// Buffer must have at least n_atoms elements.
export fn ztraj_get_masses(
    handle: ?*anyopaque,
    masses: [*]f64,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const n = h.masses.len;
    @memcpy(masses[0..n], h.masses);
    return ZTRAJ_OK;
}

/// Copy atom names from a loaded structure.
/// `names` must point to n_atoms * 4 bytes (each name is 4 bytes, space-padded).
export fn ztraj_get_atom_names(
    handle: ?*anyopaque,
    names: [*]u8,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const atoms = h.parse_result.topology.atoms;
    for (atoms, 0..) |atom, i| {
        const offset = i * 4;
        const name = atom.name.slice();
        // Copy name and pad with spaces
        for (0..4) |j| {
            names[offset + j] = if (j < name.len) name[j] else ' ';
        }
    }
    return ZTRAJ_OK;
}

/// Copy residue names from a loaded structure (per-atom).
/// `names` must point to n_atoms * 5 bytes (each name is 5 bytes, space-padded).
export fn ztraj_get_residue_names(
    handle: ?*anyopaque,
    names: [*]u8,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    for (topo.atoms, 0..) |atom, i| {
        const offset = i * 5;
        const res = topo.residues[atom.residue_index];
        const name = res.name.slice();
        for (0..5) |j| {
            names[offset + j] = if (j < name.len) name[j] else ' ';
        }
    }
    return ZTRAJ_OK;
}

/// Copy residue IDs (sequence numbers) from a loaded structure (per-atom).
/// `resids` must have at least n_atoms elements.
export fn ztraj_get_resids(
    handle: ?*anyopaque,
    resids: [*]i32,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    for (topo.atoms, 0..) |atom, i| {
        resids[i] = topo.residues[atom.residue_index].resid;
    }
    return ZTRAJ_OK;
}

/// Free a structure handle returned by ztraj_load_pdb. Safe to call with null.
export fn ztraj_free_structure(handle: ?*anyopaque) callconv(.c) void {
    const h = castStructureHandle(handle) orelse return;
    h.deinit();
}

/// Load a GRO file and return an opaque structure handle.
///
/// On success, `handle_out` is set to a non-null opaque pointer.
/// The handle must be freed with ztraj_free_structure().
export fn ztraj_load_gro(
    path: [*:0]const u8,
    handle_out: *?*anyopaque,
) callconv(.c) c_int {
    handle_out.* = null;

    const path_slice = std.mem.sliceTo(path, 0);
    const data = std.fs.cwd().readFileAlloc(c_allocator, path_slice, MAX_PDB_FILE_SIZE) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    defer c_allocator.free(data);

    var parse_result = gro_mod.parse(c_allocator, data) catch {
        return ZTRAJ_ERROR_PARSE;
    };
    errdefer parse_result.deinit();

    const masses = parse_result.topology.masses(c_allocator) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    errdefer c_allocator.free(masses);

    const handle = c_allocator.create(StructureHandle) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    handle.* = .{
        .parse_result = parse_result,
        .masses = masses,
        .allocator = c_allocator,
    };

    handle_out.* = @ptrCast(handle);
    return ZTRAJ_OK;
}

/// Load an mmCIF file and return an opaque structure handle.
///
/// On success, `handle_out` is set to a non-null opaque pointer.
/// The handle must be freed with ztraj_free_structure().
export fn ztraj_load_mmcif(
    path: [*:0]const u8,
    handle_out: *?*anyopaque,
) callconv(.c) c_int {
    handle_out.* = null;

    const path_slice = std.mem.sliceTo(path, 0);
    const data = std.fs.cwd().readFileAlloc(c_allocator, path_slice, MAX_PDB_FILE_SIZE) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    defer c_allocator.free(data);

    var parse_result = mmcif_mod.parse(c_allocator, data) catch {
        return ZTRAJ_ERROR_PARSE;
    };
    errdefer parse_result.deinit();

    const masses = parse_result.topology.masses(c_allocator) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    errdefer c_allocator.free(masses);

    const handle = c_allocator.create(StructureHandle) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    handle.* = .{
        .parse_result = parse_result,
        .masses = masses,
        .allocator = c_allocator,
    };

    handle_out.* = @ptrCast(handle);
    return ZTRAJ_OK;
}

// =============================================================================
// I/O: XTC Streaming Reader (opaque handle)
// =============================================================================

/// Opaque XTC reader handle.
const XtcHandle = struct {
    magic: u64 = XTC_MAGIC,
    reader: xtc_mod.XtcReader,
    allocator: std.mem.Allocator,

    fn deinit(self: *XtcHandle) void {
        self.magic = 0;
        self.reader.deinit();
        self.allocator.destroy(self);
    }
};

/// Validate and cast an XTC handle. Returns null if invalid.
fn castXtcHandle(handle: ?*anyopaque) ?*XtcHandle {
    const h: *XtcHandle = @ptrCast(@alignCast(handle orelse return null));
    if (h.magic != XTC_MAGIC) return null;
    return h;
}

/// Open an XTC file for streaming frame-by-frame reading.
///
/// `n_atoms_out` receives the number of atoms per frame.
/// The handle must be closed with ztraj_close_xtc().
export fn ztraj_open_xtc(
    path: [*:0]const u8,
    n_atoms_out: *usize,
    handle_out: *?*anyopaque,
) callconv(.c) c_int {
    handle_out.* = null;

    const path_slice = std.mem.sliceTo(path, 0);
    var reader = xtc_mod.XtcReader.open(c_allocator, path_slice) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    errdefer reader.deinit();

    n_atoms_out.* = reader.frame.nAtoms();

    const handle = c_allocator.create(XtcHandle) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    handle.* = .{
        .reader = reader,
        .allocator = c_allocator,
    };

    handle_out.* = @ptrCast(handle);
    return ZTRAJ_OK;
}

/// Read the next XTC frame into caller-owned buffers.
///
/// Returns ZTRAJ_OK on success, ZTRAJ_ERROR_EOF at end of file.
/// Coordinates are in angstroms (converted from nm at read time).
/// `x`, `y`, `z` must have at least n_atoms elements.
export fn ztraj_read_xtc_frame(
    handle: ?*anyopaque,
    x: [*]f32,
    y: [*]f32,
    z: [*]f32,
    time: *f32,
    step: *i32,
) callconv(.c) c_int {
    const h = castXtcHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;

    const frame_ptr = h.reader.next() catch |err| {
        return switch (err) {
            xtc_mod.XtcReadError.InvalidMagic, xtc_mod.XtcReadError.DecompressionError => ZTRAJ_ERROR_PARSE,
            else => ZTRAJ_ERROR_FILE_IO,
        };
    };

    if (frame_ptr) |frame| {
        const n = frame.nAtoms();
        @memcpy(x[0..n], frame.x);
        @memcpy(y[0..n], frame.y);
        @memcpy(z[0..n], frame.z);
        time.* = frame.time;
        step.* = frame.step;
        return ZTRAJ_OK;
    } else {
        return ZTRAJ_ERROR_EOF;
    }
}

/// Close an XTC reader handle. Safe to call with null.
export fn ztraj_close_xtc(handle: ?*anyopaque) callconv(.c) void {
    const h = castXtcHandle(handle) orelse return;
    h.deinit();
}

// =============================================================================
// I/O: TRR Streaming Reader (opaque handle)
// =============================================================================

const TRR_MAGIC: u64 = 0xD1CE_8765_4321_CAFE;

/// Opaque TRR reader handle.
const TrrHandle = struct {
    magic: u64 = TRR_MAGIC,
    reader: trr_mod.TrrReader,
    allocator: std.mem.Allocator,

    fn deinit(self: *TrrHandle) void {
        self.magic = 0;
        self.reader.deinit();
        self.allocator.destroy(self);
    }
};

/// Validate and cast a TRR handle. Returns null if invalid.
fn castTrrHandle(handle: ?*anyopaque) ?*TrrHandle {
    const h: *TrrHandle = @ptrCast(@alignCast(handle orelse return null));
    if (h.magic != TRR_MAGIC) return null;
    return h;
}

/// Open a TRR file for streaming frame-by-frame reading.
///
/// `n_atoms_out` receives the number of atoms per frame.
/// The handle must be closed with ztraj_close_trr().
export fn ztraj_open_trr(
    path: [*:0]const u8,
    n_atoms_out: *usize,
    handle_out: *?*anyopaque,
) callconv(.c) c_int {
    handle_out.* = null;

    const path_slice = std.mem.sliceTo(path, 0);
    var reader = trr_mod.TrrReader.open(c_allocator, path_slice) catch |err| {
        return switch (err) {
            trr_mod.TrrReadError.FileNotFound => ZTRAJ_ERROR_FILE_IO,
            trr_mod.TrrReadError.InvalidMagic, trr_mod.TrrReadError.InvalidHeader => ZTRAJ_ERROR_PARSE,
            trr_mod.TrrReadError.OutOfMemory => ZTRAJ_ERROR_OUT_OF_MEMORY,
            else => ZTRAJ_ERROR_FILE_IO,
        };
    };
    errdefer reader.deinit();

    n_atoms_out.* = reader.frame.nAtoms();

    const handle = c_allocator.create(TrrHandle) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    handle.* = .{
        .reader = reader,
        .allocator = c_allocator,
    };

    handle_out.* = @ptrCast(handle);
    return ZTRAJ_OK;
}

/// Read the next TRR frame into caller-owned buffers.
///
/// Returns ZTRAJ_OK on success, ZTRAJ_ERROR_EOF at end of file.
/// Coordinates are in angstroms (converted from nm at read time).
/// `x`, `y`, `z` must have at least n_atoms elements.
export fn ztraj_read_trr_frame(
    handle: ?*anyopaque,
    x: [*]f32,
    y: [*]f32,
    z: [*]f32,
    time: *f32,
    step: *i32,
) callconv(.c) c_int {
    const h = castTrrHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;

    const frame_ptr = h.reader.next() catch |err| {
        return switch (err) {
            trr_mod.TrrReadError.InvalidMagic, trr_mod.TrrReadError.InvalidHeader => ZTRAJ_ERROR_PARSE,
            trr_mod.TrrReadError.OutOfMemory => ZTRAJ_ERROR_OUT_OF_MEMORY,
            else => ZTRAJ_ERROR_FILE_IO,
        };
    };

    if (frame_ptr) |frame| {
        const n = frame.nAtoms();
        @memcpy(x[0..n], frame.x);
        @memcpy(y[0..n], frame.y);
        @memcpy(z[0..n], frame.z);
        time.* = frame.time;
        step.* = frame.step;
        return ZTRAJ_OK;
    } else {
        return ZTRAJ_ERROR_EOF;
    }
}

/// Close a TRR reader handle. Safe to call with null.
export fn ztraj_close_trr(handle: ?*anyopaque) callconv(.c) void {
    const h = castTrrHandle(handle) orelse return;
    h.deinit();
}

// =============================================================================
// I/O: DCD Streaming Reader (opaque handle)
// =============================================================================

const DCD_MAGIC: u64 = 0xDCDC_DCDC_DCDC_DCDC;

/// Opaque DCD reader handle.
const DcdHandle = struct {
    magic: u64 = DCD_MAGIC,
    reader: dcd_mod.DcdReader,
    allocator: std.mem.Allocator,

    fn deinit(self: *DcdHandle) void {
        self.magic = 0;
        self.reader.deinit();
        self.allocator.destroy(self);
    }
};

/// Validate and cast a DCD handle. Returns null if invalid.
fn castDcdHandle(handle: ?*anyopaque) ?*DcdHandle {
    const h: *DcdHandle = @ptrCast(@alignCast(handle orelse return null));
    if (h.magic != DCD_MAGIC) return null;
    return h;
}

/// Open a DCD file for streaming frame-by-frame reading.
///
/// `n_atoms_out` receives the number of atoms per frame.
/// The handle must be closed with ztraj_close_dcd().
export fn ztraj_open_dcd(
    path: [*:0]const u8,
    n_atoms_out: *usize,
    handle_out: *?*anyopaque,
) callconv(.c) c_int {
    handle_out.* = null;

    const path_slice = std.mem.sliceTo(path, 0);
    var reader = dcd_mod.DcdReader.open(c_allocator, path_slice) catch |err| {
        return switch (err) {
            dcd_mod.DcdError.FileNotFound => ZTRAJ_ERROR_FILE_IO,
            dcd_mod.DcdError.InvalidMagic, dcd_mod.DcdError.BadFormat => ZTRAJ_ERROR_PARSE,
            dcd_mod.DcdError.OutOfMemory => ZTRAJ_ERROR_OUT_OF_MEMORY,
            dcd_mod.DcdError.FixedAtomsNotSupported => ZTRAJ_ERROR_PARSE,
            else => ZTRAJ_ERROR_FILE_IO,
        };
    };
    errdefer reader.deinit();

    n_atoms_out.* = reader.nAtoms();

    const handle = c_allocator.create(DcdHandle) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    handle.* = .{
        .reader = reader,
        .allocator = c_allocator,
    };

    handle_out.* = @ptrCast(handle);
    return ZTRAJ_OK;
}

/// Read the next DCD frame into caller-owned buffers.
///
/// Returns ZTRAJ_OK on success, ZTRAJ_ERROR_EOF at end of file.
/// Coordinates are in angstroms (DCD native units, no conversion needed).
/// `x`, `y`, `z` must have at least n_atoms elements.
export fn ztraj_read_dcd_frame(
    handle: ?*anyopaque,
    x: [*]f32,
    y: [*]f32,
    z: [*]f32,
    time: *f32,
    step: *i32,
) callconv(.c) c_int {
    const h = castDcdHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;

    const frame_ptr = h.reader.next() catch |err| {
        return switch (err) {
            dcd_mod.DcdError.InvalidMagic, dcd_mod.DcdError.BadFormat => ZTRAJ_ERROR_PARSE,
            dcd_mod.DcdError.OutOfMemory => ZTRAJ_ERROR_OUT_OF_MEMORY,
            else => ZTRAJ_ERROR_FILE_IO,
        };
    };

    if (frame_ptr) |frame| {
        const n = frame.nAtoms();
        @memcpy(x[0..n], frame.x);
        @memcpy(y[0..n], frame.y);
        @memcpy(z[0..n], frame.z);
        time.* = frame.time;
        step.* = frame.step;
        return ZTRAJ_OK;
    } else {
        return ZTRAJ_ERROR_EOF;
    }
}

/// Close a DCD reader handle. Safe to call with null.
export fn ztraj_close_dcd(handle: ?*anyopaque) callconv(.c) void {
    const h = castDcdHandle(handle) orelse return;
    h.deinit();
}

// =============================================================================
// Analysis: RDF
// =============================================================================

/// Compute radial distribution function g(r) between two atom selections.
///
/// `sel1_x/y/z` are SOA coordinates for selection 1 (length `n_sel1`).
/// `sel2_x/y/z` are SOA coordinates for selection 2 (length `n_sel2`).
/// `box_volume` is the simulation box volume in cubic Angstroms.
/// `r_out` and `g_r_out` must each have `n_bins` elements.
export fn ztraj_rdf(
    sel1_x: [*]const f32,
    sel1_y: [*]const f32,
    sel1_z: [*]const f32,
    n_sel1: usize,
    sel2_x: [*]const f32,
    sel2_y: [*]const f32,
    sel2_z: [*]const f32,
    n_sel2: usize,
    box_volume: f64,
    r_min: f32,
    r_max: f32,
    n_bins: u32,
    r_out: [*]f64,
    g_r_out: [*]f64,
    n_threads: u32,
) callconv(.c) c_int {
    if (n_sel1 == 0 or n_sel2 == 0 or n_bins == 0) return ZTRAJ_ERROR_INVALID_INPUT;
    if (r_max <= r_min) return ZTRAJ_ERROR_INVALID_INPUT;
    if (box_volume <= 0.0) return ZTRAJ_ERROR_INVALID_INPUT;

    const config = rdf_mod.Config{
        .r_min = r_min,
        .r_max = r_max,
        .n_bins = n_bins,
    };

    const thread_count: usize = if (n_threads == 0) (std.Thread.getCpuCount() catch 1) else @intCast(n_threads);

    var rdf_result = rdf_mod.computeParallel(
        c_allocator,
        sel1_x[0..n_sel1],
        sel1_y[0..n_sel1],
        sel1_z[0..n_sel1],
        sel2_x[0..n_sel2],
        sel2_y[0..n_sel2],
        sel2_z[0..n_sel2],
        box_volume,
        config,
        thread_count,
    ) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    defer rdf_result.deinit();

    @memcpy(r_out[0..n_bins], rdf_result.r);
    @memcpy(g_r_out[0..n_bins], rdf_result.g_r);

    return ZTRAJ_OK;
}

// =============================================================================
// Analysis: Hydrogen Bonds
// =============================================================================

/// C-compatible hydrogen bond record.
const CHBond = extern struct {
    donor: u32,
    hydrogen: u32,
    acceptor: u32,
    distance: f32,
    angle: f32,
};

/// Detect hydrogen bonds using Baker-Hubbard criteria.
///
/// Uses topology from `structure_handle` (must be a valid StructureHandle) to
/// find D-H bonds, and `x/y/z` for the current frame coordinates.
/// Results are written to `hbonds_out` (max `capacity` elements).
/// `n_found` receives the actual number of hydrogen bonds detected.
/// If `n_found > capacity`, only the first `capacity` are written.
export fn ztraj_detect_hbonds(
    structure_handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    dist_cutoff: f32,
    angle_cutoff: f32,
    hbonds_out: [*]CHBond,
    capacity: usize,
    n_found: *usize,
    n_threads: u32,
) callconv(.c) c_int {
    const h = castStructureHandle(structure_handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    if (n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;
    if (n_atoms != h.parse_result.frame.nAtoms()) return ZTRAJ_ERROR_INVALID_INPUT;

    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    const config = hbonds_mod.Config{
        .dist_cutoff = dist_cutoff,
        .angle_cutoff = angle_cutoff,
    };

    const thread_count: usize = if (n_threads == 0) (std.Thread.getCpuCount() catch 1) else @intCast(n_threads);

    const hbonds = hbonds_mod.detectParallel(c_allocator, h.parse_result.topology, frame, config, thread_count) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    defer c_allocator.free(hbonds);

    n_found.* = hbonds.len;

    const n_copy = @min(hbonds.len, capacity);
    for (0..n_copy) |i| {
        hbonds_out[i] = .{
            .donor = hbonds[i].donor,
            .hydrogen = hbonds[i].hydrogen,
            .acceptor = hbonds[i].acceptor,
            .distance = hbonds[i].distance,
            .angle = hbonds[i].angle,
        };
    }

    return ZTRAJ_OK;
}

// =============================================================================
// Analysis: Residue-Residue Contacts
// =============================================================================

/// C-compatible contact record.
const CContact = extern struct {
    residue_i: u32,
    residue_j: u32,
    distance: f32,
};

/// Contact distance scheme constants.
pub const ZTRAJ_SCHEME_CLOSEST: c_int = 0;
pub const ZTRAJ_SCHEME_CA: c_int = 1;
pub const ZTRAJ_SCHEME_CLOSEST_HEAVY: c_int = 2;

/// Compute residue-residue contacts.
///
/// Uses topology from `structure_handle` and `x/y/z` for coordinates.
/// `scheme`: 0=closest, 1=ca, 2=closest_heavy.
/// Results written to `contacts_out` (max `capacity` elements).
/// `n_found` receives the actual number of contacts detected.
export fn ztraj_compute_contacts(
    structure_handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    scheme: c_int,
    cutoff: f32,
    contacts_out: [*]CContact,
    capacity: usize,
    n_found: *usize,
    n_threads: u32,
) callconv(.c) c_int {
    const h = castStructureHandle(structure_handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    if (n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;
    if (n_atoms != h.parse_result.frame.nAtoms()) return ZTRAJ_ERROR_INVALID_INPUT;
    if (cutoff <= 0.0) return ZTRAJ_ERROR_INVALID_INPUT;

    const zig_scheme: contacts_mod.Scheme = switch (scheme) {
        ZTRAJ_SCHEME_CLOSEST => .closest,
        ZTRAJ_SCHEME_CA => .ca,
        ZTRAJ_SCHEME_CLOSEST_HEAVY => .closest_heavy,
        else => return ZTRAJ_ERROR_INVALID_INPUT,
    };

    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    const thread_count: usize = if (n_threads == 0) (std.Thread.getCpuCount() catch 1) else @intCast(n_threads);
    const contacts = contacts_mod.computeParallel(c_allocator, h.parse_result.topology, frame, zig_scheme, cutoff, thread_count) catch |err| {
        return switch (err) {
            error.OutOfMemory, error.SystemResources, error.ThreadQuotaExceeded, error.LockedMemoryLimitExceeded, error.Unexpected => ZTRAJ_ERROR_OUT_OF_MEMORY,
        };
    };
    defer c_allocator.free(contacts);

    n_found.* = contacts.len;

    const n_copy = @min(contacts.len, capacity);
    for (0..n_copy) |i| {
        contacts_out[i] = .{
            .residue_i = contacts[i].residue_i,
            .residue_j = contacts[i].residue_j,
            .distance = contacts[i].distance,
        };
    }

    return ZTRAJ_OK;
}

// =============================================================================
// Analysis: SASA (Solvent Accessible Surface Area)
// =============================================================================

/// Compute SASA using the Shrake-Rupley algorithm.
///
/// Uses topology from `structure_handle` for element-based VdW radii.
/// `x/y/z` are coordinates for the frame to analyze.
/// `atom_areas` must have at least `n_atoms` elements.
/// `total_area` receives the total SASA.
export fn ztraj_compute_sasa(
    structure_handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    n_points: u32,
    probe_radius: f64,
    n_threads: usize,
    atom_areas: [*]f64,
    total_area: *f64,
) callconv(.c) c_int {
    const h = castStructureHandle(structure_handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    if (n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;
    if (n_atoms != h.parse_result.frame.nAtoms()) return ZTRAJ_ERROR_INVALID_INPUT;
    if (n_points == 0) return ZTRAJ_ERROR_INVALID_INPUT;
    if (probe_radius <= 0.0) return ZTRAJ_ERROR_INVALID_INPUT;

    const config = sasa_mod.SasaConfig{
        .n_points = n_points,
        .probe_radius = probe_radius,
        .n_threads = n_threads,
    };

    var result = sasa_mod.compute(
        c_allocator,
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        h.parse_result.topology,
        null,
        config,
    ) catch |err| {
        return switch (err) {
            error.NoAtoms, error.IndexOutOfBounds => ZTRAJ_ERROR_INVALID_INPUT,
            error.OutOfMemory => ZTRAJ_ERROR_OUT_OF_MEMORY,
            else => ZTRAJ_ERROR_OUT_OF_MEMORY,
        };
    };
    defer result.deinit();

    total_area.* = result.total_area;
    @memcpy(atom_areas[0..n_atoms], result.atom_areas);

    return ZTRAJ_OK;
}

// =============================================================================
// PBC (Periodic Boundary Conditions)
// =============================================================================

/// Compute native contacts Q value (hard-cut).
export fn ztraj_native_contacts_q(
    ref_x: [*]const f32,
    ref_y: [*]const f32,
    ref_z: [*]const f32,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    indices_a: [*]const u32,
    n_a: usize,
    indices_b: [*]const u32,
    n_b: usize,
    cutoff: f32,
    result: *f64,
) callconv(.c) c_int {
    if (n_atoms == 0 or n_a == 0 or n_b == 0) {
        result.* = 0.0;
        return ZTRAJ_OK;
    }
    if (cutoff < 0) return ZTRAJ_ERROR_INVALID_INPUT;

    // Bounds check indices
    for (indices_a[0..n_a]) |idx| {
        if (idx >= n_atoms) return ZTRAJ_ERROR_INVALID_INPUT;
    }
    for (indices_b[0..n_b]) |idx| {
        if (idx >= n_atoms) return ZTRAJ_ERROR_INVALID_INPUT;
    }

    result.* = native_contacts_mod.computeQ(
        ref_x[0..n_atoms],
        ref_y[0..n_atoms],
        ref_z[0..n_atoms],
        x[0..n_atoms],
        y[0..n_atoms],
        z[0..n_atoms],
        indices_a[0..n_a],
        indices_b[0..n_b],
        cutoff,
    );
    return ZTRAJ_OK;
}

/// Compute MSD as a function of lag time.
export fn ztraj_msd(
    all_x: [*]const f32,
    all_y: [*]const f32,
    all_z: [*]const f32,
    n_frames: usize,
    n_atoms: usize,
    atom_indices: ?[*]const u32,
    n_indices: usize,
    result: [*]f64,
    n_threads: u32,
) callconv(.c) c_int {
    if (n_frames == 0 or n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    // SAFETY: msd only reads frame coordinates
    const frames = c_allocator.alloc(types.Frame, n_frames) catch return ZTRAJ_ERROR_OUT_OF_MEMORY;
    defer c_allocator.free(frames);

    for (0..n_frames) |f| {
        const offset = f * n_atoms;
        frames[f] = types.Frame.initView(
            all_x[offset .. offset + n_atoms],
            all_y[offset .. offset + n_atoms],
            all_z[offset .. offset + n_atoms],
        );
    }

    const thread_count: usize = if (n_threads == 0) (std.Thread.getCpuCount() catch 1) else @intCast(n_threads);

    const msd_result = msd_mod.computeParallel(c_allocator, frames, indices, thread_count) catch |err| {
        return switch (err) {
            error.NoFrames, error.IndexOutOfBounds => ZTRAJ_ERROR_INVALID_INPUT,
            error.OutOfMemory, error.LockedMemoryLimitExceeded => ZTRAJ_ERROR_OUT_OF_MEMORY,
            error.SystemResources, error.Unexpected, error.ThreadQuotaExceeded => ZTRAJ_ERROR_OUT_OF_MEMORY,
        };
    };
    defer c_allocator.free(msd_result);

    @memcpy(result[0..n_frames], msd_result);
    return ZTRAJ_OK;
}

/// Compute PCA covariance matrix of coordinate fluctuations.
export fn ztraj_pca_covariance(
    all_x: [*]const f32,
    all_y: [*]const f32,
    all_z: [*]const f32,
    n_frames: usize,
    n_atoms: usize,
    atom_indices: ?[*]const u32,
    n_indices: usize,
    cov_out: [*]f64,
    n_threads: u32,
) callconv(.c) c_int {
    if (n_frames < 2 or n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    // SAFETY: pca only reads frame coordinates
    const frames = c_allocator.alloc(types.Frame, n_frames) catch return ZTRAJ_ERROR_OUT_OF_MEMORY;
    defer c_allocator.free(frames);

    for (0..n_frames) |f| {
        const offset = f * n_atoms;
        frames[f] = types.Frame.initView(
            all_x[offset .. offset + n_atoms],
            all_y[offset .. offset + n_atoms],
            all_z[offset .. offset + n_atoms],
        );
    }

    const threads: usize = if (n_threads == 0) (std.Thread.getCpuCount() catch 1) else @as(usize, n_threads);
    const cov = pca_mod.computeCovarianceMatrixParallel(c_allocator, frames, indices, threads) catch |err| {
        return switch (err) {
            error.TooFewFrames, error.IndexOutOfBounds, error.DimensionTooLarge => ZTRAJ_ERROR_INVALID_INPUT,
            error.OutOfMemory, error.LockedMemoryLimitExceeded, error.SystemResources, error.Unexpected, error.ThreadQuotaExceeded => ZTRAJ_ERROR_OUT_OF_MEMORY,
        };
    };
    defer c_allocator.free(cov);

    const n_sel = if (indices) |idx| idx.len else n_atoms;
    const dim = n_sel * 3;
    @memcpy(cov_out[0 .. dim * dim], cov);

    return ZTRAJ_OK;
}

fn parseBox(box_ptr: [*]const f32) [3][3]f32 {
    var b: [3][3]f32 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            b[i][j] = box_ptr[i * 3 + j];
        }
    }
    return b;
}

fn mapPbcError(err: pbc_mod.PbcError) c_int {
    return switch (err) {
        pbc_mod.PbcError.InvalidBox, pbc_mod.PbcError.InvalidBondIndex => ZTRAJ_ERROR_INVALID_INPUT,
        pbc_mod.PbcError.OutOfMemory => ZTRAJ_ERROR_OUT_OF_MEMORY,
    };
}

/// Wrap coordinates into primary simulation box (in-place).
/// `box` must point to 9 f32 values (row-major 3x3). Diagonals must be positive.
export fn ztraj_wrap_coords(
    x: [*]f32,
    y: [*]f32,
    z: [*]f32,
    n_atoms: usize,
    box: [*]const f32,
) callconv(.c) c_int {
    if (n_atoms == 0) return ZTRAJ_OK;
    const b = parseBox(box);
    pbc_mod.wrapCoords(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms], b) catch |err| {
        return mapPbcError(err);
    };
    return ZTRAJ_OK;
}

/// Compute minimum image distance between two atoms under PBC.
export fn ztraj_minimum_image_distance(
    x1: f32,
    y1: f32,
    z1: f32,
    x2: f32,
    y2: f32,
    z2: f32,
    box: [*]const f32,
    result: *f32,
) callconv(.c) c_int {
    const b = parseBox(box);
    result.* = pbc_mod.minimumImageDistance(x1, y1, z1, x2, y2, z2, b) catch |err| {
        return mapPbcError(err);
    };
    return ZTRAJ_OK;
}

/// Make molecules whole across box boundaries (in-place).
/// Requires topology with bond information. Box diagonals must be positive.
export fn ztraj_make_molecules_whole(
    structure_handle: ?*anyopaque,
    x: [*]f32,
    y: [*]f32,
    z: [*]f32,
    n_atoms: usize,
    box: [*]const f32,
) callconv(.c) c_int {
    const h = castStructureHandle(structure_handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    if (n_atoms == 0) return ZTRAJ_OK;
    if (n_atoms != h.parse_result.frame.nAtoms()) return ZTRAJ_ERROR_INVALID_INPUT;

    const b = parseBox(box);
    pbc_mod.makeMoleculesWhole(c_allocator, x[0..n_atoms], y[0..n_atoms], z[0..n_atoms], h.parse_result.topology, b) catch |err| {
        return mapPbcError(err);
    };
    return ZTRAJ_OK;
}

// =============================================================================
// Analysis: Protein Dihedrals
// =============================================================================

/// Compute backbone phi dihedral angles for all residues.
/// `result` must have n_residues elements. Undefined angles are NaN.
/// `n_residues_out` receives the number of residues.
export fn ztraj_compute_phi(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    result: [*]f32,
    n_residues_out: *usize,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    n_residues_out.* = topo.residues.len;

    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    const angles = protein_dihedrals_mod.computePhi(c_allocator, topo, frame) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    defer c_allocator.free(angles);

    for (angles, 0..) |angle, i| {
        result[i] = angle orelse std.math.nan(f32);
    }
    return ZTRAJ_OK;
}

/// Compute backbone psi dihedral angles for all residues.
/// `result` must have n_residues elements. Undefined angles are NaN.
/// `n_residues_out` receives the number of residues.
export fn ztraj_compute_psi(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    result: [*]f32,
    n_residues_out: *usize,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    n_residues_out.* = topo.residues.len;

    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    const angles = protein_dihedrals_mod.computePsi(c_allocator, topo, frame) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    defer c_allocator.free(angles);

    for (angles, 0..) |angle, i| {
        result[i] = angle orelse std.math.nan(f32);
    }
    return ZTRAJ_OK;
}

/// Compute backbone omega dihedral angles for all residues.
/// `result` must have n_residues elements. Undefined angles are NaN.
/// `n_residues_out` receives the number of residues.
export fn ztraj_compute_omega(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    result: [*]f32,
    n_residues_out: *usize,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    n_residues_out.* = topo.residues.len;

    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    const angles = protein_dihedrals_mod.computeOmega(c_allocator, topo, frame) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    defer c_allocator.free(angles);

    for (angles, 0..) |angle, i| {
        result[i] = angle orelse std.math.nan(f32);
    }
    return ZTRAJ_OK;
}

/// Compute side-chain chi dihedral angles for all residues at the given level (1-4).
/// `result` must have n_residues elements. Undefined angles (wrong level or residue
/// type) are NaN. `n_residues_out` receives the number of residues.
export fn ztraj_compute_chi(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    chi_level: u8,
    result: [*]f32,
    n_residues_out: *usize,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    n_residues_out.* = topo.residues.len;

    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    const angles = protein_dihedrals_mod.computeChi(c_allocator, topo, frame, chi_level) catch |err| {
        return switch (err) {
            error.InvalidChiLevel => ZTRAJ_ERROR_INVALID_INPUT,
            error.OutOfMemory => ZTRAJ_ERROR_OUT_OF_MEMORY,
        };
    };
    defer c_allocator.free(angles);

    for (angles, 0..) |angle, i| {
        result[i] = angle orelse std.math.nan(f32);
    }
    return ZTRAJ_OK;
}

// =============================================================================
// Analysis: DSSP Secondary Structure
// =============================================================================

/// Compute DSSP secondary structure assignment.
/// `result` must have n_residues elements. Each byte is a DSSP code:
/// 'H'=alpha-helix, 'E'=strand, 'G'=3-10 helix, 'I'=pi-helix,
/// 'T'=turn, 'S'=bend, 'B'=beta-bridge, 'P'=PP-II, ' '=loop.
/// Residues not assignable (incomplete backbone) get ' '.
export fn ztraj_compute_dssp(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    result: [*]u8,
    n_residues_out: *usize,
    n_threads: u32,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    n_residues_out.* = topo.residues.len;

    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    var dssp_result = dssp_mod.compute(c_allocator, topo, frame, .{ .n_threads = @as(usize, n_threads) }) catch {
        // Fill with spaces (loop) for residues that couldn't be assigned
        @memset(result[0..topo.residues.len], ' ');
        return ZTRAJ_OK;
    };
    defer dssp_result.deinit();

    // Initialize all to loop (' ')
    @memset(result[0..topo.residues.len], ' ');

    // Fill in DSSP assignments for complete residues
    for (dssp_result.residues) |res| {
        if (res.residue_index < topo.residues.len) {
            result[res.residue_index] = res.secondary_structure.toChar();
        }
    }
    return ZTRAJ_OK;
}

// =============================================================================
// File Writers: PDB, GRO
// =============================================================================

/// Write coordinates to a PDB file.
///
/// Uses topology from the structure handle and coordinates from caller buffers.
/// `n_atoms` must match the number of atoms in the topology.
export fn ztraj_write_pdb(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    path: [*:0]const u8,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    if (n_atoms != topo.atoms.len) return ZTRAJ_ERROR_INVALID_INPUT;
    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    const path_slice = std.mem.sliceTo(path, 0);
    const file = std.fs.cwd().createFile(path_slice, .{}) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    defer file.close();

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(c_allocator);
    pdb_mod.write(buf.writer(c_allocator), topo, frame) catch |err| {
        return if (err == error.OutOfMemory) ZTRAJ_ERROR_OUT_OF_MEMORY else ZTRAJ_ERROR_FILE_IO;
    };
    file.writeAll(buf.items) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };

    return ZTRAJ_OK;
}

/// Write coordinates to a GRO file.
///
/// Uses topology from the structure handle and coordinates from caller buffers.
/// `n_atoms` must match the number of atoms in the topology.
export fn ztraj_write_gro(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    path: [*:0]const u8,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    if (n_atoms != topo.atoms.len) return ZTRAJ_ERROR_INVALID_INPUT;
    const frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);

    const path_slice = std.mem.sliceTo(path, 0);
    const file = std.fs.cwd().createFile(path_slice, .{}) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    defer file.close();

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(c_allocator);
    gro_mod.write(buf.writer(c_allocator), topo, frame) catch |err| {
        return if (err == error.OutOfMemory) ZTRAJ_ERROR_OUT_OF_MEMORY else ZTRAJ_ERROR_FILE_IO;
    };
    file.writeAll(buf.items) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };

    return ZTRAJ_OK;
}

// =============================================================================
// Trajectory Writer: XTC
// =============================================================================

const XTC_WRITER_MAGIC: u64 = 0xCAFE_BABE_58C0_0002;

const XtcWriterHandle = struct {
    magic: u64 = XTC_WRITER_MAGIC,
    writer: xtc_mod.XtcWriter,
    allocator: std.mem.Allocator,

    fn deinit(self: *XtcWriterHandle) void {
        self.magic = 0;
        self.writer.deinit();
        self.allocator.destroy(self);
    }
};

fn castXtcWriterHandle(handle: ?*anyopaque) ?*XtcWriterHandle {
    const h: *XtcWriterHandle = @ptrCast(@alignCast(handle orelse return null));
    if (h.magic != XTC_WRITER_MAGIC) return null;
    return h;
}

/// Open an XTC trajectory file for writing.
///
/// `n_atoms` must match the number of atoms that will be written per frame.
/// On success `handle_out` is set to a non-null opaque pointer.
/// The handle must be closed with `ztraj_close_xtc_writer`.
export fn ztraj_open_xtc_writer(
    path: [*:0]const u8,
    n_atoms: usize,
    handle_out: *?*anyopaque,
) callconv(.c) c_int {
    handle_out.* = null;
    const path_slice = std.mem.sliceTo(path, 0);

    var writer = xtc_mod.XtcWriter.open(c_allocator, path_slice, n_atoms) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    errdefer writer.deinit();

    const handle = c_allocator.create(XtcWriterHandle) catch {
        writer.deinit();
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    handle.* = .{ .writer = writer, .allocator = c_allocator };
    handle_out.* = @ptrCast(handle);
    return ZTRAJ_OK;
}

/// Write one frame to an open XTC writer.
///
/// `n_atoms` must equal the value passed to `ztraj_open_xtc_writer`.
export fn ztraj_write_xtc_frame(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    time: f32,
    step: i32,
) callconv(.c) c_int {
    const h = castXtcWriterHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    var frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);
    frame.time = time;
    frame.step = step;
    h.writer.writeFrame(frame) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    return ZTRAJ_OK;
}

/// Flush and close an XTC writer, freeing all resources.
///
/// Always call this when done writing; do not use the handle afterwards.
export fn ztraj_close_xtc_writer(handle: ?*anyopaque) callconv(.c) c_int {
    const h = castXtcWriterHandle(handle) orelse return ZTRAJ_OK;
    h.writer.close() catch {
        h.deinit();
        return ZTRAJ_ERROR_FILE_IO;
    };
    h.magic = 0;
    h.allocator.destroy(h);
    return ZTRAJ_OK;
}

// =============================================================================
// Trajectory Writer: TRR
// =============================================================================

const TRR_WRITER_MAGIC: u64 = 0xCAFE_BABE_58C0_0003;

const TrrWriterHandle = struct {
    magic: u64 = TRR_WRITER_MAGIC,
    writer: trr_mod.TrrWriter,
    allocator: std.mem.Allocator,

    fn deinit(self: *TrrWriterHandle) void {
        self.magic = 0;
        self.writer.deinit();
        self.allocator.destroy(self);
    }
};

fn castTrrWriterHandle(handle: ?*anyopaque) ?*TrrWriterHandle {
    const h: *TrrWriterHandle = @ptrCast(@alignCast(handle orelse return null));
    if (h.magic != TRR_WRITER_MAGIC) return null;
    return h;
}

/// Open a TRR trajectory file for writing.
///
/// `n_atoms` must match the number of atoms that will be written per frame.
/// On success `handle_out` is set to a non-null opaque pointer.
/// The handle must be closed with `ztraj_close_trr_writer`.
export fn ztraj_open_trr_writer(
    path: [*:0]const u8,
    n_atoms: usize,
    handle_out: *?*anyopaque,
) callconv(.c) c_int {
    handle_out.* = null;
    const path_slice = std.mem.sliceTo(path, 0);

    var writer = trr_mod.TrrWriter.open(c_allocator, path_slice, n_atoms) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    errdefer writer.deinit();

    const handle = c_allocator.create(TrrWriterHandle) catch {
        writer.deinit();
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };
    handle.* = .{ .writer = writer, .allocator = c_allocator };
    handle_out.* = @ptrCast(handle);
    return ZTRAJ_OK;
}

/// Write one frame to an open TRR writer.
///
/// `n_atoms` must equal the value passed to `ztraj_open_trr_writer`.
export fn ztraj_write_trr_frame(
    handle: ?*anyopaque,
    x: [*]const f32,
    y: [*]const f32,
    z: [*]const f32,
    n_atoms: usize,
    time: f32,
    step: i32,
) callconv(.c) c_int {
    const h = castTrrWriterHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    var frame = types.Frame.initView(x[0..n_atoms], y[0..n_atoms], z[0..n_atoms]);
    frame.time = time;
    frame.step = step;
    h.writer.writeFrame(frame) catch {
        return ZTRAJ_ERROR_FILE_IO;
    };
    return ZTRAJ_OK;
}

/// Flush and close a TRR writer, freeing all resources.
///
/// Always call this when done writing; do not use the handle afterwards.
export fn ztraj_close_trr_writer(handle: ?*anyopaque) callconv(.c) c_int {
    const h = castTrrWriterHandle(handle) orelse return ZTRAJ_OK;
    h.writer.close() catch {
        h.deinit();
        return ZTRAJ_ERROR_FILE_IO;
    };
    h.magic = 0;
    h.allocator.destroy(h);
    return ZTRAJ_OK;
}

// =============================================================================
// Atom Selection
// =============================================================================

/// Select atoms by keyword: 0=backbone, 1=protein, 2=water.
/// Allocates and returns an array of atom indices.
/// Caller must free with ztraj_free_selection().
export fn ztraj_select_keyword(
    handle: ?*anyopaque,
    keyword: c_int,
    indices_out: *[*]u32,
    count_out: *usize,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;

    const kw: select_mod.Keyword = switch (keyword) {
        0 => .backbone,
        1 => .protein,
        2 => .water,
        else => return ZTRAJ_ERROR_INVALID_INPUT,
    };

    const result = select_mod.byKeyword(c_allocator, topo, kw) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };

    indices_out.* = result.ptr;
    count_out.* = result.len;
    return ZTRAJ_OK;
}

/// Select atoms by name (exact match, e.g. "CA", "N").
/// Caller must free with ztraj_free_selection().
export fn ztraj_select_name(
    handle: ?*anyopaque,
    name: [*:0]const u8,
    indices_out: *[*]u32,
    count_out: *usize,
) callconv(.c) c_int {
    const h = castStructureHandle(handle) orelse return ZTRAJ_ERROR_INVALID_INPUT;
    const topo = h.parse_result.topology;
    const name_slice = std.mem.sliceTo(name, 0);

    const result = select_mod.byName(c_allocator, topo, name_slice) catch {
        return ZTRAJ_ERROR_OUT_OF_MEMORY;
    };

    indices_out.* = result.ptr;
    count_out.* = result.len;
    return ZTRAJ_OK;
}

/// Free a selection result returned by ztraj_select_*. Safe to call with count=0.
export fn ztraj_free_selection(
    indices: ?[*]u32,
    count: usize,
) callconv(.c) void {
    const ptr = indices orelse return;
    if (count == 0) return;
    c_allocator.free(ptr[0..count]);
}

// =============================================================================
// Tests
// =============================================================================

test "c_api: version returns non-empty string" {
    const ver = ztraj_version();
    const ver_slice = std.mem.sliceTo(ver, 0);
    try std.testing.expect(ver_slice.len > 0);
}

test "c_api: distances" {
    // atoms at (0,0,0) and (3,4,0) -> distance = 5.0
    const x = [_]f32{ 0.0, 3.0 };
    const y = [_]f32{ 0.0, 4.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const pairs = [_]u32{ 0, 1 };
    var result: [1]f32 = undefined;

    const rc = ztraj_distances(&x, &y, &z, 2, &pairs, 1, &result);
    try std.testing.expectEqual(ZTRAJ_OK, rc);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 1e-5);
}

test "c_api: angles" {
    // 90 degree angle: (1,0,0)-(0,0,0)-(0,1,0)
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 1.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    const triplets = [_]u32{ 0, 1, 2 };
    var result: [1]f32 = undefined;

    const rc = ztraj_angles(&x, &y, &z, 3, &triplets, 1, &result);
    try std.testing.expectEqual(ZTRAJ_OK, rc);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, result[0], 1e-5);
}

test "c_api: dihedrals" {
    // 90 degree dihedral: i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,0,1)
    const x = [_]f32{ 0.0, 0.0, 1.0, 1.0 };
    const y = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 1.0 };
    const quartets = [_]u32{ 0, 1, 2, 3 };
    var result: [1]f32 = undefined;

    const rc = ztraj_dihedrals(&x, &y, &z, 4, &quartets, 1, &result);
    try std.testing.expectEqual(ZTRAJ_OK, rc);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, @abs(result[0]), 1e-5);
}

test "c_api: rmsd identical structures" {
    const x = [_]f32{ 0.0, 1.0, 2.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    var result: f64 = undefined;

    const rc = ztraj_rmsd(&x, &y, &z, &x, &y, &z, null, 0, 3, &result);
    try std.testing.expectEqual(ZTRAJ_OK, rc);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result, 1e-10);
}

test "c_api: center_of_mass" {
    // Two atoms at (0,0,0) mass=1 and (2,0,0) mass=1 -> COM=(1,0,0)
    const x = [_]f32{ 0.0, 2.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const masses = [_]f64{ 1.0, 1.0 };
    var cx: f64 = undefined;
    var cy: f64 = undefined;
    var cz: f64 = undefined;

    const rc = ztraj_center_of_mass(&x, &y, &z, &masses, null, 0, 2, &cx, &cy, &cz);
    try std.testing.expectEqual(ZTRAJ_OK, rc);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), cx, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), cy, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), cz, 1e-10);
}

test "c_api: rg" {
    // Two atoms at (-1,0,0) and (1,0,0) with equal mass -> Rg = 1.0
    const x = [_]f32{ -1.0, 1.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const masses = [_]f64{ 1.0, 1.0 };
    var result: f64 = undefined;

    const rc = ztraj_rg(&x, &y, &z, &masses, null, 0, 2, &result);
    try std.testing.expectEqual(ZTRAJ_OK, rc);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result, 1e-10);
}

test "c_api: inertia" {
    // Single atom at (1,0,0) mass=1 -> Ixx=0, Iyy=1, Izz=1
    // COM at (1,0,0), so relative position is (0,0,0) -> all zeros
    // Two atoms: (0,0,0) m=1 and (2,0,0) m=1 -> COM=(1,0,0)
    // Relative: (-1,0,0) and (1,0,0)
    // Ixx = sum(m*(y²+z²)) = 0, Iyy = sum(m*(x²+z²)) = 2, Izz = sum(m*(x²+y²)) = 2
    const x = [_]f32{ 0.0, 2.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const masses = [_]f64{ 1.0, 1.0 };
    var result: [9]f64 = undefined;

    const rc = ztraj_inertia(&x, &y, &z, &masses, null, 0, 2, &result);
    try std.testing.expectEqual(ZTRAJ_OK, rc);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result[0], 1e-10); // Ixx
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[4], 1e-10); // Iyy
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[8], 1e-10); // Izz
}

test "c_api: principal_moments" {
    // Diagonal tensor: Ixx=1, Iyy=2, Izz=3
    var tensor = [_]f64{ 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0 };
    var result: [3]f64 = undefined;

    const rc = ztraj_principal_moments(&tensor, &result);
    try std.testing.expectEqual(ZTRAJ_OK, rc);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result[2], 1e-10);
}
