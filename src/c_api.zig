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
const xtc_mod = @import("io/xtc.zig");

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

const VERSION: [*:0]const u8 = "0.1.0";

/// Get the library version string.
export fn ztraj_version() callconv(.c) [*:0]const u8 {
    return VERSION;
}

// =============================================================================
// Thread-safe allocator for C API
// =============================================================================

const c_allocator = std.heap.c_allocator;

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
) callconv(.c) c_int {
    if (n_frames == 0 or n_atoms == 0) return ZTRAJ_ERROR_INVALID_INPUT;

    const indices: ?[]const u32 = if (atom_indices) |ptr|
        (if (n_indices > 0) ptr[0..n_indices] else null)
    else
        null;

    // Build Frame slice from flat contiguous data
    const frames = c_allocator.alloc(types.Frame, n_frames) catch return ZTRAJ_ERROR_OUT_OF_MEMORY;
    defer c_allocator.free(frames);

    for (0..n_frames) |f| {
        const offset = f * n_atoms;
        frames[f] = .{
            .x = @constCast(all_x[offset .. offset + n_atoms]),
            .y = @constCast(all_y[offset .. offset + n_atoms]),
            .z = @constCast(all_z[offset .. offset + n_atoms]),
            .box_vectors = null,
            .time = 0.0,
            .step = 0,
            .allocator = c_allocator,
        };
    }

    const rmsf_result = rmsf_mod.compute(c_allocator, frames, indices) catch |err| {
        return switch (err) {
            error.NoFrames => ZTRAJ_ERROR_INVALID_INPUT,
            error.OutOfMemory => ZTRAJ_ERROR_OUT_OF_MEMORY,
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

/// Opaque structure handle returned by ztraj_load_pdb.
const StructureHandle = struct {
    parse_result: types.ParseResult,
    masses: []f64,
    allocator: std.mem.Allocator,

    fn deinit(self: *StructureHandle) void {
        self.allocator.free(self.masses);
        self.parse_result.deinit();
        self.allocator.destroy(self);
    }
};

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
    const data = std.fs.cwd().readFileAlloc(c_allocator, path_slice, 100 * 1024 * 1024) catch {
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

/// Get number of atoms from a loaded structure.
export fn ztraj_get_n_atoms(handle: *anyopaque) callconv(.c) usize {
    const h: *StructureHandle = @ptrCast(@alignCast(handle));
    return h.parse_result.frame.nAtoms();
}

/// Copy coordinates from a loaded structure into caller-owned buffers.
/// Buffers must have at least n_atoms elements each.
export fn ztraj_get_coords(
    handle: *anyopaque,
    x: [*]f32,
    y: [*]f32,
    z: [*]f32,
) callconv(.c) c_int {
    const h: *StructureHandle = @ptrCast(@alignCast(handle));
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
    handle: *anyopaque,
    masses: [*]f64,
) callconv(.c) c_int {
    const h: *StructureHandle = @ptrCast(@alignCast(handle));
    const n = h.masses.len;
    @memcpy(masses[0..n], h.masses);
    return ZTRAJ_OK;
}

/// Copy atom names from a loaded structure.
/// `names` must point to n_atoms * 4 bytes (each name is 4 bytes, space-padded).
export fn ztraj_get_atom_names(
    handle: *anyopaque,
    names: [*]u8,
) callconv(.c) c_int {
    const h: *StructureHandle = @ptrCast(@alignCast(handle));
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
    handle: *anyopaque,
    names: [*]u8,
) callconv(.c) c_int {
    const h: *StructureHandle = @ptrCast(@alignCast(handle));
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
    handle: *anyopaque,
    resids: [*]i32,
) callconv(.c) c_int {
    const h: *StructureHandle = @ptrCast(@alignCast(handle));
    const topo = h.parse_result.topology;
    for (topo.atoms, 0..) |atom, i| {
        resids[i] = topo.residues[atom.residue_index].resid;
    }
    return ZTRAJ_OK;
}

/// Free a structure handle returned by ztraj_load_pdb.
export fn ztraj_free_structure(handle: *anyopaque) callconv(.c) void {
    const h: *StructureHandle = @ptrCast(@alignCast(handle));
    h.deinit();
}

// =============================================================================
// I/O: XTC Streaming Reader (opaque handle)
// =============================================================================

/// Opaque XTC reader handle.
const XtcHandle = struct {
    reader: xtc_mod.XtcReader,
    allocator: std.mem.Allocator,

    fn deinit(self: *XtcHandle) void {
        self.reader.deinit();
        self.allocator.destroy(self);
    }
};

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
    handle: *anyopaque,
    x: [*]f32,
    y: [*]f32,
    z: [*]f32,
    time: *f32,
    step: *i32,
) callconv(.c) c_int {
    const h: *XtcHandle = @ptrCast(@alignCast(handle));

    const frame_ptr = h.reader.next() catch {
        return ZTRAJ_ERROR_FILE_IO;
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

/// Close an XTC reader handle.
export fn ztraj_close_xtc(handle: *anyopaque) callconv(.c) void {
    const h: *XtcHandle = @ptrCast(@alignCast(handle));
    h.deinit();
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
