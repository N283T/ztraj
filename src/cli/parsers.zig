//! Atom selection resolver and index spec parsers.

const std = @import("std");
const ztraj = @import("ztraj");

const types = ztraj.types;
const select = ztraj.select;

// ============================================================================
// Atom selection helper
// ============================================================================

fn printStderr(msg: []const u8) void {
    std.fs.File.stderr().writeAll(msg) catch {};
}

fn requireNonEmptySelection(allocator: std.mem.Allocator, selection: []u32, sel_str: []const u8) ![]u32 {
    if (selection.len > 0) return selection;
    allocator.free(selection);
    var buf: [256]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "error: selection '{s}' matched no atoms in the topology\n", .{sel_str}) catch
        "error: selection matched no atoms in the topology\n";
    printStderr(msg);
    return error.EmptySelection;
}

/// Resolve the --select string to a slice of atom indices, or null (all atoms).
/// Caller owns the returned slice.
pub fn resolveSelection(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    sel_str: ?[]const u8,
) !?[]u32 {
    const s = sel_str orelse return null;

    if (std.mem.eql(u8, s, "backbone")) {
        return try requireNonEmptySelection(allocator, try select.byKeyword(allocator, topology, .backbone), s);
    } else if (std.mem.eql(u8, s, "protein")) {
        return try requireNonEmptySelection(allocator, try select.byKeyword(allocator, topology, .protein), s);
    } else if (std.mem.eql(u8, s, "water")) {
        return try requireNonEmptySelection(allocator, try select.byKeyword(allocator, topology, .water), s);
    } else if (std.mem.startsWith(u8, s, "name ")) {
        return try requireNonEmptySelection(allocator, try select.byName(allocator, topology, s[5..]), s);
    } else if (std.mem.startsWith(u8, s, "index ")) {
        return try requireNonEmptySelection(allocator, try select.byIndex(allocator, s[6..]), s);
    } else {
        // Treat as atom name shortcut ("CA", "N", ...).
        return try requireNonEmptySelection(allocator, try select.byName(allocator, topology, s), s);
    }
}

// ============================================================================
// Index spec parsers
// ============================================================================

/// Parse "i-j,k-l" into [][2]u32.
pub fn parsePairs(allocator: std.mem.Allocator, spec: []const u8) ![][2]u32 {
    var list = std.ArrayList([2]u32){};
    errdefer list.deinit(allocator);

    var tok = std.mem.tokenizeScalar(u8, spec, ',');
    while (tok.next()) |token| {
        const t = std.mem.trim(u8, token, " \t");
        const dash = std.mem.indexOfScalar(u8, t, '-') orelse return error.InvalidSpec;
        const a = std.fmt.parseInt(u32, t[0..dash], 10) catch return error.InvalidSpec;
        const b = std.fmt.parseInt(u32, t[dash + 1 ..], 10) catch return error.InvalidSpec;
        try list.append(allocator, .{ a, b });
    }
    return list.toOwnedSlice(allocator);
}

/// Parse "i-j-k,l-m-n" into [][3]u32.
pub fn parseTriplets(allocator: std.mem.Allocator, spec: []const u8) ![][3]u32 {
    var list = std.ArrayList([3]u32){};
    errdefer list.deinit(allocator);

    var tok = std.mem.tokenizeScalar(u8, spec, ',');
    while (tok.next()) |token| {
        const t = std.mem.trim(u8, token, " \t");
        var parts: [3]u32 = undefined;
        var idx: usize = 0;
        var sub = std.mem.tokenizeScalar(u8, t, '-');
        while (sub.next()) |p| {
            if (idx >= 3) return error.InvalidSpec;
            parts[idx] = std.fmt.parseInt(u32, p, 10) catch return error.InvalidSpec;
            idx += 1;
        }
        if (idx != 3) return error.InvalidSpec;
        try list.append(allocator, parts);
    }
    return list.toOwnedSlice(allocator);
}

/// Parse "i-j-k-l,m-n-o-p" into [][4]u32.
pub fn parseQuartets(allocator: std.mem.Allocator, spec: []const u8) ![][4]u32 {
    var list = std.ArrayList([4]u32){};
    errdefer list.deinit(allocator);

    var tok = std.mem.tokenizeScalar(u8, spec, ',');
    while (tok.next()) |token| {
        const t = std.mem.trim(u8, token, " \t");
        var parts: [4]u32 = undefined;
        var idx: usize = 0;
        var sub = std.mem.tokenizeScalar(u8, t, '-');
        while (sub.next()) |p| {
            if (idx >= 4) return error.InvalidSpec;
            parts[idx] = std.fmt.parseInt(u32, p, 10) catch return error.InvalidSpec;
            idx += 1;
        }
        if (idx != 4) return error.InvalidSpec;
        try list.append(allocator, parts);
    }
    return list.toOwnedSlice(allocator);
}

/// Validate that all atom indices in a tuple array are within bounds.
pub fn validateIndices(comptime N: usize, tuples: []const [N]u32, n_atoms: u32) error{IndexOutOfRange}!void {
    for (tuples, 0..) |tuple, ti| {
        for (tuple) |idx| {
            if (idx >= n_atoms) {
                var buf: [128]u8 = undefined;
                const msg = std.fmt.bufPrint(&buf, "error: atom index {d} in tuple {d} is out of range (topology has {d} atoms)\n", .{ idx, ti, n_atoms }) catch
                    "error: atom index out of range\n";
                printStderr(msg);
                return error.IndexOutOfRange;
            }
        }
    }
}

test "resolveSelection rejects empty matches" {
    const allocator = std.testing.allocator;
    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 2,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.atoms[0] = .{ .name = types.FixedString(4).fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = types.FixedString(4).fromSlice("CA"), .element = .C, .residue_index = 0 };
    topo.residues[0] = .{
        .name = types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 2 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };

    try std.testing.expectError(error.EmptySelection, resolveSelection(allocator, topo, "ZZ"));
    try std.testing.expectError(error.EmptySelection, resolveSelection(allocator, topo, "index "));
}
