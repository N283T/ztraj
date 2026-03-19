const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");

const Vec3f32 = types.Vec3f32;

// ---------------------------------------------------------------------------
// Cell for spatial hash grid
// ---------------------------------------------------------------------------

const Cell = struct {
    atoms: std.ArrayListAligned(u32, null),

    fn init() Cell {
        return .{ .atoms = .empty };
    }

    fn deinit(self: *Cell, allocator: Allocator) void {
        self.atoms.deinit(allocator);
    }

    fn append(self: *Cell, allocator: Allocator, value: u32) !void {
        try self.atoms.append(allocator, value);
    }
};

// ---------------------------------------------------------------------------
// Spatial hash grid (cell list)
// ---------------------------------------------------------------------------

/// 3D spatial hash grid for efficient neighbor lookups on f32 coordinates.
///
/// Atoms are binned into cubic cells. Neighbor queries only need to check
/// the 27 surrounding cells (3×3×3 neighbourhood), reducing the search
/// from O(N²) to O(N + nearby pairs).
pub const CellList = struct {
    cells: []Cell,
    nx: usize,
    ny: usize,
    nz: usize,
    cell_size: f32,
    x_min: f32,
    y_min: f32,
    z_min: f32,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        positions: []const Vec3f32,
        cell_size: f32,
    ) !CellList {
        if (positions.len == 0) return error.NoAtoms;
        if (cell_size <= 0.0) return error.InvalidCellSize;

        // Bounding box
        var x_min = positions[0].x;
        var x_max = positions[0].x;
        var y_min = positions[0].y;
        var y_max = positions[0].y;
        var z_min = positions[0].z;
        var z_max = positions[0].z;

        for (positions) |pos| {
            x_min = @min(x_min, pos.x);
            x_max = @max(x_max, pos.x);
            y_min = @min(y_min, pos.y);
            y_max = @max(y_max, pos.y);
            z_min = @min(z_min, pos.z);
            z_max = @max(z_max, pos.z);
        }

        // Padding
        x_min -= cell_size;
        y_min -= cell_size;
        z_min -= cell_size;
        x_max += cell_size;
        y_max += cell_size;
        z_max += cell_size;

        const nx = @max(1, @as(usize, @intFromFloat(@ceil((x_max - x_min) / cell_size))));
        const ny = @max(1, @as(usize, @intFromFloat(@ceil((y_max - y_min) / cell_size))));
        const nz = @max(1, @as(usize, @intFromFloat(@ceil((z_max - z_min) / cell_size))));

        const n_cells = nx * ny * nz;

        const cells = try allocator.alloc(Cell, n_cells);
        errdefer allocator.free(cells);

        for (cells) |*cell| {
            cell.* = Cell.init();
        }

        var cell_list = CellList{
            .cells = cells,
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .cell_size = cell_size,
            .x_min = x_min,
            .y_min = y_min,
            .z_min = z_min,
            .allocator = allocator,
        };

        for (positions, 0..) |pos, i| {
            const cell_idx = cell_list.getCellIndex(pos);
            try cell_list.cells[cell_idx].append(allocator, @intCast(i));
        }

        return cell_list;
    }

    pub fn deinit(self: *CellList) void {
        for (self.cells) |*cell| {
            cell.deinit(self.allocator);
        }
        self.allocator.free(self.cells);
    }

    fn getCellIndex(self: CellList, pos: Vec3f32) usize {
        // Compute cell coordinates with clamping to valid range
        // Clamp before integer conversion to avoid overflow with extreme coordinates
        const max_x: f32 = @floatFromInt(self.nx - 1);
        const max_y: f32 = @floatFromInt(self.ny - 1);
        const max_z: f32 = @floatFromInt(self.nz - 1);

        const raw_x = (pos.x - self.x_min) / self.cell_size;
        const raw_y = (pos.y - self.y_min) / self.cell_size;
        const raw_z = (pos.z - self.z_min) / self.cell_size;

        const cix = @as(usize, @intFromFloat(@max(0.0, @min(raw_x, max_x))));
        const ciy = @as(usize, @intFromFloat(@max(0.0, @min(raw_y, max_y))));
        const ciz = @as(usize, @intFromFloat(@max(0.0, @min(raw_z, max_z))));

        return ciz * self.nx * self.ny + ciy * self.nx + cix;
    }

    fn getCellCoords(self: CellList, idx: usize) struct { ix: usize, iy: usize, iz: usize } {
        const iz = idx / (self.nx * self.ny);
        const remainder = idx % (self.nx * self.ny);
        const iy = remainder / self.nx;
        const ix = remainder % self.nx;
        return .{ .ix = ix, .iy = iy, .iz = iz };
    }

    fn getCellIndexFromCoords(self: CellList, ix: i64, iy: i64, iz: i64) ?usize {
        if (ix < 0 or iy < 0 or iz < 0) return null;
        const uix = @as(usize, @intCast(ix));
        const uiy = @as(usize, @intCast(iy));
        const uiz = @as(usize, @intCast(iz));
        if (uix >= self.nx or uiy >= self.ny or uiz >= self.nz) return null;
        return uiz * self.nx * self.ny + uiy * self.nx + uix;
    }
};

// ---------------------------------------------------------------------------
// Near-pair finding with spatial hashing
// ---------------------------------------------------------------------------

/// Find all pairs of residue indices whose CA atoms are within a given cutoff.
///
/// Uses a spatial hash grid (cell list) for O(N + pairs) performance
/// instead of brute-force O(N²).
///
/// Returns the same [][2]u32 format as the original hbond.findNearPairs.
pub fn findNearPairsGrid(
    ca_positions: []const Vec3f32,
    complete: []const bool,
    cutoff: f32,
    allocator: Allocator,
) ![][2]u32 {
    if (ca_positions.len < 2) {
        return allocator.alloc([2]u32, 0);
    }

    var cell_list = try CellList.init(allocator, ca_positions, cutoff);
    defer cell_list.deinit();

    var pairs: std.ArrayListAligned([2]u32, null) = .empty;
    errdefer pairs.deinit(allocator);

    const cutoff_sq = cutoff * cutoff;
    const n_cells = cell_list.nx * cell_list.ny * cell_list.nz;

    for (0..n_cells) |cell_idx| {
        const coords = cell_list.getCellCoords(cell_idx);
        const ix = @as(i64, @intCast(coords.ix));
        const iy = @as(i64, @intCast(coords.iy));
        const iz = @as(i64, @intCast(coords.iz));

        // Check all 27 neighbouring cells (including self)
        var dz: i64 = -1;
        while (dz <= 1) : (dz += 1) {
            var dy: i64 = -1;
            while (dy <= 1) : (dy += 1) {
                var dx: i64 = -1;
                while (dx <= 1) : (dx += 1) {
                    const neighbor_idx = cell_list.getCellIndexFromCoords(ix + dx, iy + dy, iz + dz);
                    if (neighbor_idx) |nidx| {
                        // Only process each cell pair once
                        if (nidx < cell_idx) continue;

                        try addPairsFromCells(
                            allocator,
                            &cell_list.cells[cell_idx],
                            &cell_list.cells[nidx],
                            ca_positions,
                            complete,
                            cutoff_sq,
                            &pairs,
                            cell_idx == nidx,
                        );
                    }
                }
            }
        }
    }

    return pairs.toOwnedSlice(allocator);
}

fn addPairsFromCells(
    allocator: Allocator,
    cell1: *const Cell,
    cell2: *const Cell,
    positions: []const Vec3f32,
    complete: []const bool,
    cutoff_sq: f32,
    pairs: *std.ArrayListAligned([2]u32, null),
    same_cell: bool,
) !void {
    for (cell1.atoms.items) |i| {
        if (!complete[i]) continue;

        for (cell2.atoms.items) |j| {
            if (i == j) continue;
            if (same_cell and j <= i) continue;
            if (!complete[j]) continue;

            const dist_sq = positions[i].distanceSq(positions[j]);
            if (dist_sq < cutoff_sq) {
                // Store in canonical order (smaller index first)
                const a = @min(i, j);
                const b = @max(i, j);
                try pairs.append(allocator, .{ a, b });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Residue-level neighbor list for accessibility
// ---------------------------------------------------------------------------

/// Build a per-residue neighbor list: for each residue i, collect indices
/// of all other residues whose CA is within the given cutoff.
///
/// This is used by the accessibility calculation to avoid O(N²) scanning
/// per atom.
pub fn buildResidueNeighborList(
    ca_positions: []const Vec3f32,
    complete: []const bool,
    cutoff: f32,
    allocator: Allocator,
) ![][]u32 {
    const n = ca_positions.len;

    var lists = try allocator.alloc(std.ArrayListAligned(u32, null), n);
    errdefer {
        for (lists) |*list| {
            list.deinit(allocator);
        }
        allocator.free(lists);
    }
    for (lists) |*list| {
        list.* = .empty;
    }

    if (n < 2) {
        // Convert to owned slices
        const result = try allocator.alloc([]u32, n);
        for (result, 0..) |*r, idx| {
            r.* = try lists[idx].toOwnedSlice(allocator);
        }
        allocator.free(lists);
        return result;
    }

    var cell_list = try CellList.init(allocator, ca_positions, cutoff);
    defer cell_list.deinit();

    const cutoff_sq = cutoff * cutoff;
    const n_cells = cell_list.nx * cell_list.ny * cell_list.nz;

    for (0..n_cells) |cell_idx| {
        const coords = cell_list.getCellCoords(cell_idx);
        const cix = @as(i64, @intCast(coords.ix));
        const ciy = @as(i64, @intCast(coords.iy));
        const ciz = @as(i64, @intCast(coords.iz));

        var dz: i64 = -1;
        while (dz <= 1) : (dz += 1) {
            var dy: i64 = -1;
            while (dy <= 1) : (dy += 1) {
                var dx: i64 = -1;
                while (dx <= 1) : (dx += 1) {
                    const neighbor_idx = cell_list.getCellIndexFromCoords(cix + dx, ciy + dy, ciz + dz);
                    if (neighbor_idx) |nidx| {
                        if (nidx < cell_idx) continue;

                        try addNeighborPairsSymmetric(
                            allocator,
                            &cell_list.cells[cell_idx],
                            &cell_list.cells[nidx],
                            ca_positions,
                            complete,
                            cutoff_sq,
                            lists,
                            cell_idx == nidx,
                        );
                    }
                }
            }
        }
    }

    // Convert ArrayLists to owned slices
    const result = try allocator.alloc([]u32, n);
    for (result, 0..) |*r, idx| {
        r.* = try lists[idx].toOwnedSlice(allocator);
    }
    allocator.free(lists);

    return result;
}

fn addNeighborPairsSymmetric(
    allocator: Allocator,
    cell1: *const Cell,
    cell2: *const Cell,
    positions: []const Vec3f32,
    complete: []const bool,
    cutoff_sq: f32,
    lists: []std.ArrayListAligned(u32, null),
    same_cell: bool,
) !void {
    for (cell1.atoms.items) |i| {
        if (!complete[i]) continue;

        for (cell2.atoms.items) |j| {
            if (i == j) continue;
            if (same_cell and j <= i) continue;
            if (!complete[j]) continue;

            const dist_sq = positions[i].distanceSq(positions[j]);
            if (dist_sq < cutoff_sq) {
                // Symmetric: add both directions
                try lists[i].append(allocator, j);
                try lists[j].append(allocator, i);
            }
        }
    }
}

/// Free a neighbor list returned by buildResidueNeighborList.
pub fn freeResidueNeighborList(neighbor_list: [][]u32, allocator: Allocator) void {
    for (neighbor_list) |list| {
        allocator.free(list);
    }
    allocator.free(neighbor_list);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "CellList - single atom" {
    const allocator = std.testing.allocator;
    const positions = &[_]Vec3f32{
        Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
    };

    var cell_list = try CellList.init(allocator, positions, 9.0);
    defer cell_list.deinit();

    try std.testing.expect(cell_list.nx >= 1);
    try std.testing.expect(cell_list.ny >= 1);
    try std.testing.expect(cell_list.nz >= 1);
}

test "CellList - atoms in different cells" {
    const allocator = std.testing.allocator;
    const positions = &[_]Vec3f32{
        Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 20.0, .y = 0.0, .z = 0.0 },
    };

    var cell_list = try CellList.init(allocator, positions, 9.0);
    defer cell_list.deinit();

    try std.testing.expect(cell_list.nx > 1);
}

test "findNearPairsGrid - close residues" {
    const allocator = std.testing.allocator;
    const positions = &[_]Vec3f32{
        Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 5.0, .y = 0.0, .z = 0.0 }, // 5 Å < 9 Å
        Vec3f32{ .x = 20.0, .y = 0.0, .z = 0.0 }, // 20 Å > 9 Å from first
    };
    const complete = &[_]bool{ true, true, true };

    const pairs = try findNearPairsGrid(positions, complete, 9.0, allocator);
    defer allocator.free(pairs);

    // (0,1) should be a near pair; (0,2) and (1,2) should not
    try std.testing.expectEqual(@as(usize, 1), pairs.len);
    try std.testing.expectEqual(@as(u32, 0), pairs[0][0]);
    try std.testing.expectEqual(@as(u32, 1), pairs[0][1]);
}

test "findNearPairsGrid - matches brute force" {
    const allocator = std.testing.allocator;

    // Create a set of positions and verify grid gives same pairs as brute force
    const positions = &[_]Vec3f32{
        Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 5.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 8.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 20.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 22.0, .y = 3.0, .z = 0.0 },
    };
    const complete = &[_]bool{ true, true, true, true, true };

    const grid_pairs = try findNearPairsGrid(positions, complete, 9.0, allocator);
    defer allocator.free(grid_pairs);

    // Brute force count
    const cutoff_sq: f32 = 9.0 * 9.0;
    var brute_count: usize = 0;
    for (0..positions.len) |i| {
        for (i + 1..positions.len) |j| {
            if (positions[i].distanceSq(positions[j]) < cutoff_sq) {
                brute_count += 1;
            }
        }
    }

    try std.testing.expectEqual(brute_count, grid_pairs.len);
}

test "findNearPairsGrid - skips incomplete residues" {
    const allocator = std.testing.allocator;
    const positions = &[_]Vec3f32{
        Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 5.0, .y = 0.0, .z = 0.0 },
    };
    const complete = &[_]bool{ true, false }; // second residue incomplete

    const pairs = try findNearPairsGrid(positions, complete, 9.0, allocator);
    defer allocator.free(pairs);

    try std.testing.expectEqual(@as(usize, 0), pairs.len);
}

test "buildResidueNeighborList - basic" {
    const allocator = std.testing.allocator;
    const positions = &[_]Vec3f32{
        Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 5.0, .y = 0.0, .z = 0.0 }, // within 12 Å of 0
        Vec3f32{ .x = 50.0, .y = 0.0, .z = 0.0 }, // far away
    };
    const complete = &[_]bool{ true, true, true };

    const neighbors = try buildResidueNeighborList(positions, complete, 12.0, allocator);
    defer freeResidueNeighborList(neighbors, allocator);

    try std.testing.expectEqual(@as(usize, 3), neighbors.len);
    // 0 and 1 should be neighbors of each other
    try std.testing.expectEqual(@as(usize, 1), neighbors[0].len);
    try std.testing.expectEqual(@as(usize, 1), neighbors[1].len);
    // 2 should have no neighbors
    try std.testing.expectEqual(@as(usize, 0), neighbors[2].len);
}

test "buildResidueNeighborList - symmetry" {
    const allocator = std.testing.allocator;
    const positions = &[_]Vec3f32{
        Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 3.0, .y = 0.0, .z = 0.0 },
        Vec3f32{ .x = 6.0, .y = 0.0, .z = 0.0 },
    };
    const complete = &[_]bool{ true, true, true };

    const neighbors = try buildResidueNeighborList(positions, complete, 12.0, allocator);
    defer freeResidueNeighborList(neighbors, allocator);

    // Check symmetry: if j in neighbors[i] then i in neighbors[j]
    for (0..3) |i| {
        for (neighbors[i]) |j| {
            var found = false;
            for (neighbors[j]) |k| {
                if (k == @as(u32, @intCast(i))) {
                    found = true;
                    break;
                }
            }
            try std.testing.expect(found);
        }
    }
}
