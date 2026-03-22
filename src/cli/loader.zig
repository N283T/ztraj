//! File extension detection and frame/topology loading helpers.

const std = @import("std");
const ztraj = @import("ztraj");

const types = ztraj.types;
const io = ztraj.io;

// ============================================================================
// File extension helpers
// ============================================================================

pub fn endsWithCI(s: []const u8, suffix: []const u8) bool {
    if (s.len < suffix.len) return false;
    const tail = s[s.len - suffix.len ..];
    for (tail, suffix) |a, b| {
        if (std.ascii.toLower(a) != std.ascii.toLower(b)) return false;
    }
    return true;
}

pub fn isPdb(path: []const u8) bool {
    return endsWithCI(path, ".pdb");
}

pub fn isCif(path: []const u8) bool {
    return endsWithCI(path, ".cif") or endsWithCI(path, ".mmcif");
}

pub fn isXtc(path: []const u8) bool {
    return endsWithCI(path, ".xtc");
}

pub fn isDcd(path: []const u8) bool {
    return endsWithCI(path, ".dcd");
}

pub fn isTrr(path: []const u8) bool {
    return endsWithCI(path, ".trr");
}

pub fn isGro(path: []const u8) bool {
    return endsWithCI(path, ".gro");
}

// ============================================================================
// Topology loading
// ============================================================================

/// Load topology + first frame from a PDB, CIF, or GRO file.
/// Returns a ParseResult; caller must call .deinit().
pub fn loadTopology(allocator: std.mem.Allocator, path: []const u8) !types.ParseResult {
    const data = try std.fs.cwd().readFileAlloc(allocator, path, 512 * 1024 * 1024);
    defer allocator.free(data);

    if (isPdb(path)) {
        return io.pdb.parse(allocator, data);
    } else if (isCif(path)) {
        return io.mmcif.parse(allocator, data);
    } else if (isGro(path)) {
        return io.gro.parse(allocator, data);
    } else {
        std.debug.print(
            "error: unsupported topology format for '{s}' (supported: .pdb, .cif, .mmcif, .gro)\n",
            .{path},
        );
        std.process.exit(1);
    }
}

// ============================================================================
// Frame collection
// ============================================================================

/// Load every frame from a trajectory or single-structure file.
/// Returns allocated []Frame (caller frees each frame then the slice).
pub fn loadAllFrames(
    allocator: std.mem.Allocator,
    traj_path: []const u8,
    n_atoms: usize,
) ![]types.Frame {
    var frames = std.ArrayList(types.Frame){};
    errdefer {
        for (frames.items) |*f| f.deinit();
        frames.deinit(allocator);
    }

    if (isXtc(traj_path)) {
        var reader = try io.xtc.XtcReader.open(allocator, traj_path);
        defer reader.deinit();
        while (try reader.next()) |frame_ptr| {
            var copy = try types.Frame.init(allocator, n_atoms);
            @memcpy(copy.x, frame_ptr.x);
            @memcpy(copy.y, frame_ptr.y);
            @memcpy(copy.z, frame_ptr.z);
            copy.time = frame_ptr.time;
            copy.step = frame_ptr.step;
            copy.box_vectors = frame_ptr.box_vectors;
            try frames.append(allocator, copy);
        }
    } else if (isDcd(traj_path)) {
        var reader = try io.dcd.DcdReader.open(allocator, traj_path);
        defer reader.deinit();
        while (try reader.next()) |frame_ptr| {
            var copy = try types.Frame.init(allocator, n_atoms);
            @memcpy(copy.x, frame_ptr.x);
            @memcpy(copy.y, frame_ptr.y);
            @memcpy(copy.z, frame_ptr.z);
            copy.time = frame_ptr.time;
            copy.step = frame_ptr.step;
            copy.box_vectors = frame_ptr.box_vectors;
            try frames.append(allocator, copy);
        }
    } else if (isTrr(traj_path)) {
        var reader = try io.trr.TrrReader.open(allocator, traj_path);
        defer reader.deinit();
        while (try reader.next()) |frame_ptr| {
            var copy = try types.Frame.init(allocator, n_atoms);
            @memcpy(copy.x, frame_ptr.x);
            @memcpy(copy.y, frame_ptr.y);
            @memcpy(copy.z, frame_ptr.z);
            copy.time = frame_ptr.time;
            copy.step = frame_ptr.step;
            copy.box_vectors = frame_ptr.box_vectors;
            try frames.append(allocator, copy);
        }
    } else {
        // Single-frame structure file — parse only for its coordinates.
        const data = try std.fs.cwd().readFileAlloc(allocator, traj_path, 512 * 1024 * 1024);
        defer allocator.free(data);
        var pr: types.ParseResult = if (isPdb(traj_path))
            try io.pdb.parse(allocator, data)
        else if (isCif(traj_path))
            try io.mmcif.parse(allocator, data)
        else if (isGro(traj_path))
            try io.gro.parse(allocator, data)
        else {
            std.debug.print(
                "error: unsupported trajectory/structure format for '{s}' (supported: .xtc, .trr, .dcd, .pdb, .cif, .mmcif, .gro)\n",
                .{traj_path},
            );
            std.process.exit(1);
        };
        pr.topology.deinit();
        try frames.append(allocator, pr.frame);
    }

    return frames.toOwnedSlice(allocator);
}
