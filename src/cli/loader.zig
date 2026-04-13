//! File extension detection and frame/topology loading helpers.

const std = @import("std");
const ztraj = @import("ztraj");

const types = ztraj.types;
const io = ztraj.io;

fn printStderr(msg: []const u8) void {
    std.fs.File.stderr().writeAll(msg) catch {};
}

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

pub fn isPrmtop(path: []const u8) bool {
    return endsWithCI(path, ".prmtop") or endsWithCI(path, ".parm7") or endsWithCI(path, ".top");
}

pub fn isNc(path: []const u8) bool {
    return endsWithCI(path, ".nc") or endsWithCI(path, ".ncdf");
}

// ============================================================================
// Topology loading
// ============================================================================

/// Load topology + first frame from a structure or topology file.
/// Supported formats: PDB, CIF/mmCIF, GRO, PRMTOP/PARM7.
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
    } else if (isPrmtop(path)) {
        // PRMTOP is topology-only — wrap in ParseResult with a zero-atom frame
        const topo = try io.prmtop.parseTopology(allocator, data);
        errdefer {
            var t = topo;
            t.deinit();
        }
        var frame = try types.Frame.init(allocator, 0);
        errdefer frame.deinit();
        return types.ParseResult{ .topology = topo, .frame = frame };
    } else {
        printStderr("error: unsupported topology format (supported: .pdb, .cif, .mmcif, .gro, .prmtop, .parm7, .top)\n");
        return error.UnsupportedFormat;
    }
}

// ============================================================================
// Frame collection
// ============================================================================

/// Summary information for a trajectory or single-structure file.
pub const TrajectoryInfo = struct {
    n_frames: usize,
    first_time: ?f32,
    last_time: ?f32,
};

fn ensureAtomCount(path: []const u8, expected: usize, actual: usize) !void {
    if (expected == actual) return;

    var buf: [256]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "error: '{s}' has {d} atoms but topology expects {d}\n", .{ path, actual, expected }) catch
        "error: atom count mismatch between trajectory and topology\n";
    printStderr(msg);
    return error.InvalidAtomCount;
}

/// Count frames and capture the first/last timestamps without materializing
/// every frame in memory.
pub fn loadTrajectoryInfo(
    allocator: std.mem.Allocator,
    traj_path: []const u8,
    expected_n_atoms: ?usize,
) !TrajectoryInfo {
    if (isXtc(traj_path)) {
        var reader = try io.xtc.XtcReader.open(allocator, traj_path);
        defer reader.deinit();
        if (expected_n_atoms) |n_atoms| try ensureAtomCount(traj_path, n_atoms, reader.nAtoms());

        var n_frames: usize = 0;
        var first_time: ?f32 = null;
        var last_time: ?f32 = null;
        while (try reader.next()) |frame_ptr| {
            if (first_time == null) first_time = frame_ptr.time;
            last_time = frame_ptr.time;
            n_frames += 1;
        }
        return .{ .n_frames = n_frames, .first_time = first_time, .last_time = last_time };
    } else if (isDcd(traj_path)) {
        var reader = try io.dcd.DcdReader.open(allocator, traj_path);
        defer reader.deinit();
        if (expected_n_atoms) |n_atoms| try ensureAtomCount(traj_path, n_atoms, reader.nAtoms());

        var n_frames: usize = 0;
        var first_time: ?f32 = null;
        var last_time: ?f32 = null;
        while (try reader.next()) |frame_ptr| {
            if (first_time == null) first_time = frame_ptr.time;
            last_time = frame_ptr.time;
            n_frames += 1;
        }
        return .{ .n_frames = n_frames, .first_time = first_time, .last_time = last_time };
    } else if (isTrr(traj_path)) {
        var reader = try io.trr.TrrReader.open(allocator, traj_path);
        defer reader.deinit();
        if (expected_n_atoms) |n_atoms| try ensureAtomCount(traj_path, n_atoms, reader.nAtoms());

        var n_frames: usize = 0;
        var first_time: ?f32 = null;
        var last_time: ?f32 = null;
        while (try reader.next()) |frame_ptr| {
            if (first_time == null) first_time = frame_ptr.time;
            last_time = frame_ptr.time;
            n_frames += 1;
        }
        return .{ .n_frames = n_frames, .first_time = first_time, .last_time = last_time };
    } else if (isNc(traj_path)) {
        var reader = try io.nc.NcReader.open(allocator, traj_path);
        defer reader.deinit();
        if (expected_n_atoms) |n_atoms| try ensureAtomCount(traj_path, n_atoms, reader.nAtoms());

        var n_frames: usize = 0;
        var first_time: ?f32 = null;
        var last_time: ?f32 = null;
        while (try reader.next()) |frame_ptr| {
            if (first_time == null) first_time = frame_ptr.time;
            last_time = frame_ptr.time;
            n_frames += 1;
        }
        return .{ .n_frames = n_frames, .first_time = first_time, .last_time = last_time };
    } else {
        const data = try std.fs.cwd().readFileAlloc(allocator, traj_path, 512 * 1024 * 1024);
        defer allocator.free(data);

        var pr: types.ParseResult = if (isPdb(traj_path))
            try io.pdb.parse(allocator, data)
        else if (isCif(traj_path))
            try io.mmcif.parse(allocator, data)
        else if (isGro(traj_path))
            try io.gro.parse(allocator, data)
        else {
            return unsupportedTrajectoryFormat(traj_path);
        };
        defer pr.deinit();
        if (expected_n_atoms) |n_atoms| try ensureAtomCount(traj_path, n_atoms, pr.frame.nAtoms());

        return .{
            .n_frames = 1,
            .first_time = pr.frame.time,
            .last_time = pr.frame.time,
        };
    }
}

fn unsupportedTrajectoryFormat(_: []const u8) error{UnsupportedFormat} {
    printStderr("error: unsupported trajectory/structure format (supported: .xtc, .trr, .dcd, .nc, .pdb, .cif, .mmcif, .gro)\n");
    return error.UnsupportedFormat;
}

/// Load every frame from a trajectory or single-structure file.
/// Returns allocated []Frame (caller frees each frame then the slice).
/// An optional progress node is updated per frame loaded.
pub fn loadAllFrames(
    allocator: std.mem.Allocator,
    traj_path: []const u8,
    n_atoms: usize,
    progress_node: std.Progress.Node,
) ![]types.Frame {
    var frames = std.ArrayList(types.Frame){};
    errdefer {
        for (frames.items) |*f| f.deinit();
        frames.deinit(allocator);
    }

    if (isXtc(traj_path)) {
        var reader = try io.xtc.XtcReader.open(allocator, traj_path);
        defer reader.deinit();
        try ensureAtomCount(traj_path, n_atoms, reader.nAtoms());
        while (try reader.next()) |frame_ptr| {
            var copy = try types.Frame.init(allocator, n_atoms);
            @memcpy(copy.x, frame_ptr.x);
            @memcpy(copy.y, frame_ptr.y);
            @memcpy(copy.z, frame_ptr.z);
            copy.time = frame_ptr.time;
            copy.step = frame_ptr.step;
            copy.box_vectors = frame_ptr.box_vectors;
            try frames.append(allocator, copy);
            progress_node.completeOne();
        }
    } else if (isDcd(traj_path)) {
        var reader = try io.dcd.DcdReader.open(allocator, traj_path);
        defer reader.deinit();
        try ensureAtomCount(traj_path, n_atoms, reader.nAtoms());
        while (try reader.next()) |frame_ptr| {
            var copy = try types.Frame.init(allocator, n_atoms);
            @memcpy(copy.x, frame_ptr.x);
            @memcpy(copy.y, frame_ptr.y);
            @memcpy(copy.z, frame_ptr.z);
            copy.time = frame_ptr.time;
            copy.step = frame_ptr.step;
            copy.box_vectors = frame_ptr.box_vectors;
            try frames.append(allocator, copy);
            progress_node.completeOne();
        }
    } else if (isTrr(traj_path)) {
        var reader = try io.trr.TrrReader.open(allocator, traj_path);
        defer reader.deinit();
        try ensureAtomCount(traj_path, n_atoms, reader.nAtoms());
        while (try reader.next()) |frame_ptr| {
            var copy = try types.Frame.init(allocator, n_atoms);
            @memcpy(copy.x, frame_ptr.x);
            @memcpy(copy.y, frame_ptr.y);
            @memcpy(copy.z, frame_ptr.z);
            copy.time = frame_ptr.time;
            copy.step = frame_ptr.step;
            copy.box_vectors = frame_ptr.box_vectors;
            try frames.append(allocator, copy);
            progress_node.completeOne();
        }
    } else if (isNc(traj_path)) {
        var reader = try io.nc.NcReader.open(allocator, traj_path);
        defer reader.deinit();
        try ensureAtomCount(traj_path, n_atoms, reader.nAtoms());
        while (try reader.next()) |frame_ptr| {
            var copy = try types.Frame.init(allocator, n_atoms);
            @memcpy(copy.x, frame_ptr.x);
            @memcpy(copy.y, frame_ptr.y);
            @memcpy(copy.z, frame_ptr.z);
            copy.time = frame_ptr.time;
            copy.step = frame_ptr.step;
            copy.box_vectors = frame_ptr.box_vectors;
            try frames.append(allocator, copy);
            progress_node.completeOne();
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
            return unsupportedTrajectoryFormat(traj_path);
        };
        errdefer pr.frame.deinit();
        defer pr.topology.deinit();

        try ensureAtomCount(traj_path, n_atoms, pr.frame.nAtoms());
        try frames.append(allocator, pr.frame);
        progress_node.completeOne();
    }

    return frames.toOwnedSlice(allocator);
}

/// Convenience wrapper: creates a "Loading frames" progress node, loads all
/// frames, and ends the node automatically via defer.
pub fn loadAllFramesWithProgress(
    allocator: std.mem.Allocator,
    traj_path: []const u8,
    n_atoms: usize,
    progress_root: std.Progress.Node,
) ![]types.Frame {
    const load_node = progress_root.start("Loading frames", 0);
    defer load_node.end();
    return loadAllFrames(allocator, traj_path, n_atoms, load_node);
}

test "loadTrajectoryInfo counts frames without materializing them" {
    const allocator = std.testing.allocator;
    const info = try loadTrajectoryInfo(allocator, "validation/test_data/3tvj_I_R1.xtc", null);
    try std.testing.expect(info.n_frames > 0);
    try std.testing.expect(info.first_time != null);
    try std.testing.expect(info.last_time != null);
    try std.testing.expect(info.last_time.? >= info.first_time.?);
}

test "loadTrajectoryInfo handles single-structure files" {
    const allocator = std.testing.allocator;
    const info = try loadTrajectoryInfo(allocator, "test_data/1l2y.pdb", null);
    try std.testing.expectEqual(@as(usize, 1), info.n_frames);
    try std.testing.expect(info.first_time != null);
    try std.testing.expect(info.last_time != null);
}

test "loadTrajectoryInfo rejects mismatched trajectory atom count" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.InvalidAtomCount,
        loadTrajectoryInfo(allocator, "validation/test_data/3tvj_I_R1.xtc", 1),
    );
}

test "loadAllFrames rejects mismatched trajectory atom count" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.InvalidAtomCount,
        loadAllFrames(allocator, "validation/test_data/3tvj_I_R1.xtc", 1, std.Progress.Node.none),
    );
}

test "loadAllFrames rejects mismatched structure atom count" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.InvalidAtomCount,
        loadAllFrames(allocator, "test_data/1l2y.pdb", 1, std.Progress.Node.none),
    );
}

test "loadTopology rejects unsupported format" {
    const allocator = std.testing.allocator;
    // Rename check: use an existing file but with unsupported extension
    // The file will be read successfully but format detection should fail.
    try std.testing.expectError(
        error.UnsupportedFormat,
        loadTopology(allocator, "build.zig"),
    );
}
