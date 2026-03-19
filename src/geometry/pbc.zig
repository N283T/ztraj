//! Periodic Boundary Conditions: wrap, minimum image, make molecules whole.
//!
//! Supports orthorhombic and triclinic boxes via lower-triangular matrix.
//! Box vectors are stored as [3][3]f32 rows: box[0]=a, box[1]=b, box[2]=c.

const std = @import("std");
const types = @import("../types.zig");

// ============================================================================
// Box helpers
// ============================================================================

/// Check if box is orthorhombic (off-diagonal elements are zero).
pub fn isOrthorhombic(box: [3][3]f32) bool {
    return box[0][1] == 0 and box[0][2] == 0 and
        box[1][0] == 0 and box[1][2] == 0 and
        box[2][0] == 0 and box[2][1] == 0;
}

// ============================================================================
// Wrap coordinates into primary cell
// ============================================================================

/// Wrap all atom coordinates into the primary simulation box (in-place).
///
/// For orthorhombic boxes: x_i' = x_i - floor(x_i / L_i) * L_i
/// For triclinic boxes: sequential wrap along c → b → a axes.
pub fn wrapCoords(
    x: []f32,
    y: []f32,
    z: []f32,
    box: [3][3]f32,
) void {
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
    // Lower-triangular: a=(box[0][0],0,0), b=(box[1][0],box[1][1],0), c=(box[2][0],box[2][1],box[2][2])
    const inv_cz = 1.0 / box[2][2];
    const inv_by = 1.0 / box[1][1];
    const inv_ax = 1.0 / box[0][0];

    for (0..x.len) |i| {
        // Wrap along c-axis
        var sc = @floor(z[i] * inv_cz);
        x[i] -= sc * box[2][0];
        y[i] -= sc * box[2][1];
        z[i] -= sc * box[2][2];

        // Wrap along b-axis
        var sb = @floor(y[i] * inv_by);
        x[i] -= sb * box[1][0];
        y[i] -= sb * box[1][1];

        // Wrap along a-axis
        var sa = @floor(x[i] * inv_ax);
        x[i] -= sa * box[0][0];

        // Suppress unused variable warnings
        _ = &sc;
        _ = &sb;
        _ = &sa;
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
) f32 {
    var dx = x2 - x1;
    var dy = y2 - y1;
    var dz = z2 - z1;

    if (isOrthorhombic(box)) {
        minimumImageOrtho(&dx, &dy, &dz, box);
    } else {
        minimumImageTriclinic(&dx, &dy, &dz, box);
    }

    return @sqrt(dx * dx + dy * dy + dz * dz);
}

/// Apply minimum image convention to a displacement vector (in-place).
pub fn minimumImage(dx: *f32, dy: *f32, dz: *f32, box: [3][3]f32) void {
    if (isOrthorhombic(box)) {
        minimumImageOrtho(dx, dy, dz, box);
    } else {
        minimumImageTriclinic(dx, dy, dz, box);
    }
}

fn minimumImageOrtho(dx: *f32, dy: *f32, dz: *f32, box: [3][3]f32) void {
    const lx = box[0][0];
    const ly = box[1][1];
    const lz = box[2][2];
    dx.* -= @round(dx.* / lx) * lx;
    dy.* -= @round(dy.* / ly) * ly;
    dz.* -= @round(dz.* / lz) * lz;
}

fn minimumImageTriclinic(dx: *f32, dy: *f32, dz: *f32, box: [3][3]f32) void {
    // Sequential round along c → b → a (mdtraj approach)
    const inv_cz = 1.0 / box[2][2];
    const inv_by = 1.0 / box[1][1];
    const inv_ax = 1.0 / box[0][0];

    // c-axis
    const sc = @round(dz.* * inv_cz);
    dx.* -= sc * box[2][0];
    dy.* -= sc * box[2][1];
    dz.* -= sc * box[2][2];

    // b-axis
    const sb = @round(dy.* * inv_by);
    dx.* -= sb * box[1][0];
    dy.* -= sb * box[1][1];

    // a-axis
    const sa = @round(dx.* * inv_ax);
    dx.* -= sa * box[0][0];
}

// ============================================================================
// Make molecules whole (unwrap)
// ============================================================================

/// Make molecules whole by unwrapping atoms that are split across box boundaries.
///
/// Uses bond information from the topology to determine connectivity.
/// BFS from each unvisited atom produces a sorted bond list where the parent
/// atom is always already placed. Then each child atom is moved to the nearest
/// image of its parent.
///
/// Modifies coordinates in-place.
pub fn makeMoleculesWhole(
    allocator: std.mem.Allocator,
    x: []f32,
    y: []f32,
    z: []f32,
    topology: types.Topology,
    box: [3][3]f32,
) !void {
    const n_atoms = x.len;
    if (n_atoms < 2 or topology.bonds.len == 0) return;

    // Build adjacency list from bonds
    const adj_starts = try allocator.alloc(u32, n_atoms + 1);
    defer allocator.free(adj_starts);
    const adj_targets = try allocator.alloc(u32, topology.bonds.len * 2);
    defer allocator.free(adj_targets);

    // Count degree for each atom
    @memset(adj_starts, 0);
    for (topology.bonds) |bond| {
        adj_starts[bond.atom_i] += 1;
        adj_starts[bond.atom_j] += 1;
    }

    // Prefix sum to get start indices
    {
        var sum: u32 = 0;
        for (0..n_atoms) |i| {
            const deg = adj_starts[i];
            adj_starts[i] = sum;
            sum += deg;
        }
        adj_starts[n_atoms] = sum;
    }

    // Fill adjacency targets (CSR format)
    const write_pos = try allocator.alloc(u32, n_atoms);
    defer allocator.free(write_pos);
    @memcpy(write_pos, adj_starts[0..n_atoms]);

    for (topology.bonds) |bond| {
        const i = bond.atom_i;
        const j = bond.atom_j;
        adj_targets[write_pos[i]] = j;
        write_pos[i] += 1;
        adj_targets[write_pos[j]] = i;
        write_pos[j] += 1;
    }

    // BFS: visit each atom exactly once, unwrapping child to nearest image of parent
    const visited = try allocator.alloc(bool, n_atoms);
    defer allocator.free(visited);
    @memset(visited, false);

    const queue = try allocator.alloc(u32, n_atoms);
    defer allocator.free(queue);

    for (0..n_atoms) |start_usize| {
        const start: u32 = @intCast(start_usize);
        if (visited[start]) continue;

        // BFS from this atom
        visited[start] = true;
        var head: usize = 0;
        var tail: usize = 1;
        queue[0] = start;

        while (head < tail) {
            const parent = queue[head];
            head += 1;

            // Visit all neighbors
            const adj_begin = adj_starts[parent];
            const adj_end = adj_starts[parent + 1];
            for (adj_begin..adj_end) |ai| {
                const child = adj_targets[ai];
                if (visited[child]) continue;
                visited[child] = true;

                // Move child to nearest image of parent
                var dx = x[child] - x[parent];
                var dy = y[child] - y[parent];
                var dz = z[child] - z[parent];
                minimumImage(&dx, &dy, &dz, box);
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

    wrapCoords(&x, &y, &z, box);

    try std.testing.expectApproxEqAbs(@as(f32, 9.0), x[0], 1e-5); // -1 → 9
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[1], 1e-5); // 11 → 1
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), x[2], 1e-5); // 5 → 5
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), x[3], 1e-5); // 25 → 5
}

test "minimumImageDistance: orthorhombic" {
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    // Atoms at x=1 and x=9 → raw dist=8, min image dist=2 (across boundary)
    const dist = minimumImageDistance(1.0, 0, 0, 9.0, 0, 0, box);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dist, 1e-5);
}

test "minimumImageDistance: same position" {
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };
    const dist = minimumImageDistance(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, box);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dist, 1e-7);
}

test "makeMoleculesWhole: simple dimer across boundary" {
    const allocator = std.testing.allocator;
    const box = [3][3]f32{ .{ 10.0, 0, 0 }, .{ 0, 10.0, 0 }, .{ 0, 0, 10.0 } };

    // Two bonded atoms: atom 0 at x=1, atom 1 at x=9 (should unwrap to x=-1)
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

    // Atom 1 should be unwrapped to nearest image of atom 0 (x=1)
    // 9 → -1 (distance 2 across boundary), not 9 (distance 8 direct)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5); // unchanged
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), x[1], 1e-5); // unwrapped
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

    // No bonds → coordinates unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), x[1], 1e-5);
}
