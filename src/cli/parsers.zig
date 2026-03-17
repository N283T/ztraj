//! Atom selection resolver and index spec parsers.

const std = @import("std");
const ztraj = @import("ztraj");

const types = ztraj.types;
const select = ztraj.select;

// ============================================================================
// Atom selection helper
// ============================================================================

/// Resolve the --select string to a slice of atom indices, or null (all atoms).
/// Caller owns the returned slice.
pub fn resolveSelection(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    sel_str: ?[]const u8,
) !?[]u32 {
    const s = sel_str orelse return null;

    if (std.mem.eql(u8, s, "backbone")) {
        return try select.byKeyword(allocator, topology, .backbone);
    } else if (std.mem.eql(u8, s, "protein")) {
        return try select.byKeyword(allocator, topology, .protein);
    } else if (std.mem.eql(u8, s, "water")) {
        return try select.byKeyword(allocator, topology, .water);
    } else if (std.mem.startsWith(u8, s, "name ")) {
        return try select.byName(allocator, topology, s[5..]);
    } else if (std.mem.startsWith(u8, s, "index ")) {
        return try select.byIndex(allocator, s[6..]);
    } else {
        // Treat as atom name shortcut ("CA", "N", ...).
        return try select.byName(allocator, topology, s);
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
pub fn validateIndices(comptime N: usize, tuples: []const [N]u32, n_atoms: u32) void {
    for (tuples, 0..) |tuple, ti| {
        for (tuple) |idx| {
            if (idx >= n_atoms) {
                std.debug.print(
                    "error: atom index {d} in tuple {d} is out of range (topology has {d} atoms)\n",
                    .{ idx, ti, n_atoms },
                );
                std.process.exit(1);
            }
        }
    }
}
