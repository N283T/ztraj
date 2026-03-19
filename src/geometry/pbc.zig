//! Periodic Boundary Conditions: wrap, minimum image, make molecules whole.
//!
//! Supports orthorhombic and triclinic boxes via lower-triangular matrix.
//! Box vectors are stored as [3][3]f32 rows: box[0]=a, box[1]=b, box[2]=c.
//!
//! Note: For triclinic boxes, uses the sequential-round heuristic (c -> b -> a).
//! This is exact for boxes with tilt factors < 0.5*diagonal but may not find
//! the true minimum image for highly skewed cells.

const std = @import("std");
const types = @import("../types.zig");

pub const PbcError = error{
    /// Box has zero or negative diagonal elements.
    InvalidBox,
    /// Bond index exceeds number of atoms.
    InvalidBondIndex,
    OutOfMemory,
};

// ============================================================================
// Box validation
// ============================================================================

/// Validate that box diagonals are positive. Returns error for invalid boxes.
fn validateBox(box: [3][3]f32) PbcError!void {
    if (box[0][0] <= 0 or box[1][1] <= 0 or box[2][2] <= 0) {
        return PbcError.InvalidBox;
    }
}

/// Check if box is orthorhombic (off-diagonal elements are zero).
pub fn isOrthorhombic(box: [3][3]f32) bool {
    const eps: f32 = 1e-6;
    return @abs(box[0][1]) < eps and @abs(box[0][2]) < eps and
        @abs(box[1][0]) < eps and @abs(box[1][2]) < eps and
        @abs(box[2][0]) < eps and @abs(box[2][1]) < eps;
}

// ============================================================================
// Wrap coordinates into primary cell
// ============================================================================

/// Wrap all atom coordinates into the primary simulation box (in-place).
pub fn wrapCoords(x: []f32, y: []f32, z: []f32, box: [3][3]f32) PbcError!void {
    std.debug.assert(x.len == y.len and y.len == z.len);
    try validateBox(box);

    if (isOrthorhombic(box)) {
        wrapOrthorhombic(x, y, z, box);
    } else {
        wrapTriclinic(x, y, z, box);
    }
}

fn wrapOrthorhombic(x: []f32, y: []f32, z: []f32, box: [3][3]f32) void {
    const lx = box[0][0];
    const ly = box[1][1];
    const lz = box[2][2];
    for (0..x.len) |i| {
        x[i] -= @floor(x[i] / lx) * lx;
        y[i] -= @floor(y[i] / ly) * ly;
        z[i] -= @floor(z[i] / lz) * lz;
    }
}

fn wrapTriclinic(x: []f32, y: []f32, z: []f32, box: [3][3]f32) void {
    const inv_cz = 1.0 / box[2][2];
    const inv_by = 1.0 / box[1][1];
    const inv_ax = 1.0 / box[0][0];

    for (0..x.len) |i| {
        const sc = @floor(z[i] * inv_cz);
        x[i] -= sc * box[2][0];
        y[i] -= sc * box[2][1];
        z[i] -= sc * box[2][2];

        const sb = @floor(y[i] * inv_by);
        x[i] -= sb * box[1][0];
        y[i] -= sb * box[1][1];

        const sa = @floor(x[i] * inv_ax);
        x[i] -= sa * box[0][0];
    }
}

// ============================================================================
// Minimum image distance
// ============================================================================

/// Compute the minimum image distance between two atoms under PBC.
pub fn minimumImageDistance(
    x1: f32,
    y1: f32,
    z1: f32,
    x2: f32,
    y2: f32,
    z2: f32,
    box: [3][3]f32,
) PbcError!f32 {
    try validateBox(box);
    var dx = x2 - x1;
    var dy = y2 - y1;
    var dz = z2 - z1;
    minimumImageVec(&dx, &dy, &dz, box);
    return @sqrt(dx * dx + dy * dy + dz * dz);
}

/// Apply minimum image convention to a displacement vector (in-place).
fn minimumImageVec(dx: *f32, dy: *f32, dz: *f32, box: [3][3]f32) void {
    if (isOrthorhombic(box)) {
        minimumImageOrtho(dx, dy, dz, box);
    } else {
        minimumImageTriclinic(dx, dy, dz, box);
    }
}

fn minimumImageOrtho(dx: *f32, dy: *f32, dz: *f32, box: [3][3]f32) void {
    dx.* -= @round(dx.* / box[0][0]) * box[0][0];
    dy.* -= @round(dy.* / box[1][1]) * box[1][1];
    dz.* -= @round(dz.* / box[2][2]) * box[2][2];
}

fn minimumImageTriclinic(dx: *f32, dy: *f32, dz: *f32, box: [3][3]f32) void {
    const inv_cz = 1.0 / box[2][2];
    const inv_by = 1.0 / box[1][1];
    const inv_ax = 1.0 / box[0][0];

    const sc = @round(dz.* * inv_cz);
    dx.* -= sc * box[2][0];
    dy.* -= sc * box[2][1];
    dz.* -= sc * box[2][2];

    const sb = @round(dy.* * inv_by);
    dx.* -= sb * box[1][0];
    dy.* -= sb * box[1][1];

    const sa = @round(dx.* * inv_ax);
    dx.* -= sa * box[0][0];
}

// ============================================================================
// Make molecules whole (unwrap)
// ============================================================================

/// Make molecules whole by unwrapping atoms split across box boundaries.
///
/// Uses bond information from topology. BFS from each unvisited atom;
/// each child atom is moved to the nearest image of its parent.
/// Atoms without bonds are left unchanged.
///
/// Modifies coordinates in-place.
pub fn makeMoleculesWhole(
    allocator: std.mem.Allocator,
    x: []f32,
    y: []f32,
    z: []f32,
    topology: types.Topology,
    box: [3][3]f32,
) PbcError!void {
    std.debug.assert(x.len == y.len and y.len == z.len);
    try validateBox(box);

    const n_atoms: u32 = @intCast(x.len);
    if (n_atoms < 2 or topology.bonds.len == 0) return;

    // Validate bond indices
    for (topology.bonds) |bond| {
        if (bond.atom_i >= n_atoms or bond.atom_j >= n_atoms) {
            return PbcError.InvalidBondIndex;
        }
    }

    // Build adjacency list (CSR format)
    const adj_starts = allocator.alloc(u32, n_atoms + 1) catch return PbcError.OutOfMemory;
    defer allocator.free(adj_starts);
    const adj_targets = allocator.alloc(u32, topology.bonds.len * 2) catch return PbcError.OutOfMemory;
    defer allocator.free(adj_targets);

    @memset(adj_starts, 0);
    for (topology.bonds) |bond| {
        adj_starts[bond.atom_i] += 1;
        adj_starts[bond.atom_j] += 1;
    }

    {
        var sum: u32 = 0;
        for (0..n_atoms) |i| {
            const deg = adj_starts[i];
            adj_starts[i] = sum;
            sum += deg;
        }
        adj_starts[n_atoms] = sum;
    }

    const write_pos = allocator.alloc(u32, n_atoms) catch return PbcError.OutOfMemory;
    defer allocator.free(write_pos);
    @memcpy(write_pos, adj_starts[0..n_atoms]);

    for (topology.bonds) |bond| {
        adj_targets[write_pos[bond.atom_i]] = bond.atom_j;
        write_pos[bond.atom_i] += 1;
        adj_targets[write_pos[bond.atom_j]] = bond.atom_i;
        write_pos[bond.atom_j] += 1;
    }

    // BFS unwrap
    const visited = allocator.alloc(bool, n_atoms) catch return PbcError.OutOfMemory;
    defer allocator.free(visited);
    @memset(visited, false);

    const queue = allocator.alloc(u32, n_atoms) catch return PbcError.OutOfMemory;
    defer allocator.free(queue);

    for (0..n_atoms) |start_usize| {
        const start: u32 = @intCast(start_usize);
        if (visited[start]) continue;

        visited[start] = true;
        var head: usize = 0;
        var tail: usize = 1;
        queue[0] = start;

        while (head < tail) {
            const parent = queue[head];
            head += 1;

            const adj_begin = adj_starts[parent];
            const adj_end = adj_starts[parent + 1];
            for (adj_begin..adj_end) |ai| {
                const child = adj_targets[ai];
                if (visited[child]) continue;
                visited[child] = true;

                var dx = x[child] - x[parent];
                var dy = y[child] - y[parent];
                var dz = z[child] - z[parent];
                minimumImageVec(&dx, &dy, &dz, box);
                x[child] = x[parent] + dx;
                y[child] = y[parent] + dy;
                z[child] = z[parent] + dz;

                queue[tail] = child;
                tail += 1;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "isOrthorhombic" {
    const ortho = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    try std.testing.expect(isOrthorhombic(ortho));

    const triclinic = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 1.0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    try std.testing.expect(!isOrthorhombic(triclinic));
}

test "wrapCoords: orthorhombic" {
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    var x = [_]f32{ -1.0, 11.0, 5.0, 25.0 };
    var y = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    try wrapCoords(&x, &y, &z, box);

    try std.testing.expectApproxEqAbs(@as(f32, 9.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), x[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), x[3], 1e-5);
}

test "wrapCoords: zero box returns error" {
    const box = [3][3]f32{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    var x = [_]f32{1.0};
    var y = [_]f32{0.0};
    var z = [_]f32{0.0};
    try std.testing.expectError(PbcError.InvalidBox, wrapCoords(&x, &y, &z, box));
}

test "minimumImageDistance: orthorhombic across boundary" {
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    const dist = try minimumImageDistance(1.0, 0, 0, 9.0, 0, 0, box);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dist, 1e-5);
}

test "minimumImageDistance: zero box returns error" {
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 10.0 } };
    try std.testing.expectError(PbcError.InvalidBox, minimumImageDistance(0, 0, 0, 1, 0, 0, box));
}

test "minimumImageDistance: triclinic" {
    // Tilted box: a=(10,0,0), b=(2,10,0), c=(0,0,10)
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 2.0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    // Two atoms along x, well within box
    const dist = try minimumImageDistance(1.0, 5.0, 5.0, 1.0, 5.0, 5.0, box);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dist, 1e-6);
}

test "makeMoleculesWhole: dimer across boundary" {
    const allocator = std.testing.allocator;
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };

    var x = [_]f32{ 1.0, 9.0 };
    var y = [_]f32{ 0.0, 0.0 };
    var z = [_]f32{ 0.0, 0.0 };

    const topo_sizes = types.TopologySizes{ .n_atoms = 2, .n_residues = 1, .n_chains = 1, .n_bonds = 1 };
    var topo = try types.Topology.init(allocator, topo_sizes);
    defer topo.deinit();

    topo.atoms[0] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 0 };
    topo.atoms[1] = .{ .name = types.FixedString(4).fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 };
    topo.residues[0] = .{ .name = types.FixedString(5).fromSlice("HOH"), .chain_index = 0, .atom_range = .{ .start = 0, .len = 2 }, .resid = 1 };
    topo.chains[0] = .{ .name = types.FixedString(4).fromSlice("A"), .residue_range = .{ .start = 0, .len = 1 } };

    try makeMoleculesWhole(allocator, &x, &y, &z, topo, box);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), x[1], 1e-5);
}

test "makeMoleculesWhole: no bonds → no change" {
    const allocator = std.testing.allocator;
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    var x = [_]f32{ 1.0, 9.0 };
    var y = [_]f32{ 0.0, 0.0 };
    var z = [_]f32{ 0.0, 0.0 };

    const topo_sizes = types.TopologySizes{ .n_atoms = 2, .n_residues = 1, .n_chains = 1, .n_bonds = 0 };
    var topo = try types.Topology.init(allocator, topo_sizes);
    defer topo.deinit();

    topo.atoms[0] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 0 };
    topo.atoms[1] = .{ .name = types.FixedString(4).fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.residues[0] = .{ .name = types.FixedString(5).fromSlice("HOH"), .chain_index = 0, .atom_range = .{ .start = 0, .len = 2 }, .resid = 1 };
    topo.chains[0] = .{ .name = types.FixedString(4).fromSlice("A"), .residue_range = .{ .start = 0, .len = 1 } };

    try makeMoleculesWhole(allocator, &x, &y, &z, topo, box);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), x[1], 1e-5);
}

test "makeMoleculesWhole: zero box returns error" {
    const allocator = std.testing.allocator;
    const box = [3][3]f32{ .{ 0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    var x = [_]f32{ 1.0, 9.0 };
    var y = [_]f32{ 0.0, 0.0 };
    var z = [_]f32{ 0.0, 0.0 };

    const topo_sizes = types.TopologySizes{ .n_atoms = 2, .n_residues = 1, .n_chains = 1, .n_bonds = 1 };
    var topo = try types.Topology.init(allocator, topo_sizes);
    defer topo.deinit();
    topo.atoms[0] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 0 };
    topo.atoms[1] = .{ .name = types.FixedString(4).fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 };
    topo.residues[0] = .{ .name = types.FixedString(5).fromSlice("HOH"), .chain_index = 0, .atom_range = .{ .start = 0, .len = 2 }, .resid = 1 };
    topo.chains[0] = .{ .name = types.FixedString(4).fromSlice("A"), .residue_range = .{ .start = 0, .len = 1 } };

    try std.testing.expectError(PbcError.InvalidBox, makeMoleculesWhole(allocator, &x, &y, &z, topo, box));
}
