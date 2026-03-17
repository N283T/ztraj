//! ztraj CLI entry point.
//!
//! Pattern: ztraj <command> <trajectory> --top <topology> [options]
//! Output to stdout by default; file output via --output.

const std = @import("std");
const build_options = @import("build_options");
const ztraj = @import("ztraj");

const types = ztraj.types;
const io = ztraj.io;
const geometry = ztraj.geometry;
const analysis = ztraj.analysis;
const select = ztraj.select;
const output = ztraj.output;

// ============================================================================
// Subcommand enum
// ============================================================================

const Subcommand = enum {
    rmsd,
    rmsf,
    distances,
    angles,
    dihedrals,
    rg,
    center,
    inertia,
    hbonds,
    contacts,
    rdf,
};

// ============================================================================
// Parsed CLI arguments
// ============================================================================

const Args = struct {
    subcommand: Subcommand,
    /// Trajectory or single-structure file path.
    traj_path: []const u8,
    /// Topology file path (PDB or mmCIF).
    top_path: ?[]const u8 = null,
    /// Output file path (null = stdout).
    output_path: ?[]const u8 = null,
    /// Output format.
    format: output.Format = .json,
    /// Atom selection string.
    select_str: ?[]const u8 = null,
    /// Reference frame index for RMSD.
    ref_frame: usize = 0,
    /// Pair specification for distances ("i-j,k-l").
    pairs_spec: ?[]const u8 = null,
    /// Triplet specification for angles ("i-j-k").
    triplets_spec: ?[]const u8 = null,
    /// Quartet specification for dihedrals ("i-j-k-l").
    quartets_spec: ?[]const u8 = null,
    /// First selection for RDF.
    sel1: ?[]const u8 = null,
    /// Second selection for RDF.
    sel2: ?[]const u8 = null,
    /// RDF: maximum distance in Angstroms.
    rmax: f32 = 10.0,
    /// RDF: number of bins.
    rdf_bins: u32 = 100,
    /// H-bond distance cutoff (Angstroms).
    hbond_dist_cutoff: f32 = 2.5,
    /// H-bond angle cutoff (degrees).
    hbond_angle_cutoff: f32 = 120.0,
    /// Contacts: scheme name.
    contacts_scheme: []const u8 = "closest_heavy",
    /// Contacts: distance cutoff (Angstroms).
    contacts_cutoff: f32 = 5.0,
};

// ============================================================================
// Usage / help
// ============================================================================

fn printUsage(prog: []const u8) void {
    std.debug.print(
        \\Usage: {s} <command> <traj> [options]
        \\
        \\Commands:
        \\  rmsd        Compute RMSD per frame vs reference (QCP algorithm)
        \\  rmsf        Compute per-atom RMSF over trajectory
        \\  distances   Pairwise atom distances per frame
        \\  angles      Bond angles per frame
        \\  dihedrals   Dihedral (torsion) angles per frame
        \\  rg          Radius of gyration per frame
        \\  center      Center of mass per frame [x, y, z]
        \\  inertia     Principal moments of inertia per frame
        \\  hbonds      Hydrogen bond detection per frame
        \\  contacts    Residue-residue contacts per frame
        \\  rdf         Radial distribution function g(r)
        \\
        \\Common options:
        \\  --top <file>           Topology file (PDB or mmCIF); required unless traj is PDB/CIF
        \\  --select <expr>        Atom selection: backbone|protein|water|name <name>|index <spec>
        \\  --format <fmt>         Output format: json (default), csv, tsv
        \\  --output <file>        Write output to file (default: stdout)
        \\
        \\Per-command options:
        \\  rmsd      --ref <n>                   Reference frame index (default: 0)
        \\  distances --pairs <i-j,k-l,...>        Atom index pairs (0-based)
        \\  angles    --triplets <i-j-k,...>       Atom index triplets (0-based)
        \\  dihedrals --quartets <i-j-k-l,...>     Atom index quartets (0-based)
        \\  rdf       --sel1 <sel> --sel2 <sel> --rmax <f> --bins <n>
        \\  hbonds    --dist-cutoff <f>  --angle-cutoff <f>
        \\  contacts  --scheme <closest|ca|closest_heavy>  --cutoff <f>
        \\
        \\Examples:
        \\  {s} rmsd traj.xtc --top top.pdb --ref 0 --select backbone
        \\  {s} rg traj.xtc --top top.pdb --select protein
        \\  {s} distances traj.xtc --top top.pdb --pairs "0-10,1-20"
        \\  {s} rdf traj.xtc --top top.pdb --sel1 "O" --sel2 "H" --rmax 8.0 --bins 200
        \\  {s} hbonds top.pdb
        \\
        \\  -V, --version    Print version and exit
        \\  -h, --help       Print this help and exit
        \\
    , .{ prog, prog, prog, prog, prog, prog });
}

// ============================================================================
// Argument parsing
// ============================================================================

const ParseArgsError = error{
    MissingSubcommand,
    MissingTrajectory,
    UnknownSubcommand,
    UnknownFlag,
    MissingValue,
    InvalidFormat,
    InvalidNumber,
};

fn parseArgs(raw: []const []const u8) ParseArgsError!Args {
    if (raw.len < 2) return ParseArgsError.MissingSubcommand;

    const sub_str = raw[1];
    const subcmd: Subcommand = blk: {
        if (std.mem.eql(u8, sub_str, "rmsd")) break :blk .rmsd;
        if (std.mem.eql(u8, sub_str, "rmsf")) break :blk .rmsf;
        if (std.mem.eql(u8, sub_str, "distances")) break :blk .distances;
        if (std.mem.eql(u8, sub_str, "angles")) break :blk .angles;
        if (std.mem.eql(u8, sub_str, "dihedrals")) break :blk .dihedrals;
        if (std.mem.eql(u8, sub_str, "rg")) break :blk .rg;
        if (std.mem.eql(u8, sub_str, "center")) break :blk .center;
        if (std.mem.eql(u8, sub_str, "inertia")) break :blk .inertia;
        if (std.mem.eql(u8, sub_str, "hbonds")) break :blk .hbonds;
        if (std.mem.eql(u8, sub_str, "contacts")) break :blk .contacts;
        if (std.mem.eql(u8, sub_str, "rdf")) break :blk .rdf;
        return ParseArgsError.UnknownSubcommand;
    };

    var traj_path: ?[]const u8 = null;
    var args = Args{
        .subcommand = subcmd,
        .traj_path = undefined,
    };

    var i: usize = 2;
    while (i < raw.len) : (i += 1) {
        const arg = raw[i];

        if (std.mem.eql(u8, arg, "--top")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.top_path = raw[i];
        } else if (std.mem.eql(u8, arg, "--output")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.output_path = raw[i];
        } else if (std.mem.eql(u8, arg, "--format")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            const fmt = raw[i];
            if (std.mem.eql(u8, fmt, "json")) {
                args.format = .json;
            } else if (std.mem.eql(u8, fmt, "csv")) {
                args.format = .csv;
            } else if (std.mem.eql(u8, fmt, "tsv")) {
                args.format = .tsv;
            } else {
                return ParseArgsError.InvalidFormat;
            }
        } else if (std.mem.eql(u8, arg, "--select") or std.mem.eql(u8, arg, "--atoms")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.select_str = raw[i];
        } else if (std.mem.eql(u8, arg, "--ref")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.ref_frame = std.fmt.parseInt(usize, raw[i], 10) catch return ParseArgsError.InvalidNumber;
        } else if (std.mem.eql(u8, arg, "--pairs")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.pairs_spec = raw[i];
        } else if (std.mem.eql(u8, arg, "--triplets")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.triplets_spec = raw[i];
        } else if (std.mem.eql(u8, arg, "--quartets")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.quartets_spec = raw[i];
        } else if (std.mem.eql(u8, arg, "--sel1")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.sel1 = raw[i];
        } else if (std.mem.eql(u8, arg, "--sel2")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.sel2 = raw[i];
        } else if (std.mem.eql(u8, arg, "--rmax")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.rmax = std.fmt.parseFloat(f32, raw[i]) catch return ParseArgsError.InvalidNumber;
        } else if (std.mem.eql(u8, arg, "--bins")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.rdf_bins = std.fmt.parseInt(u32, raw[i], 10) catch return ParseArgsError.InvalidNumber;
        } else if (std.mem.eql(u8, arg, "--dist-cutoff")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.hbond_dist_cutoff = std.fmt.parseFloat(f32, raw[i]) catch return ParseArgsError.InvalidNumber;
        } else if (std.mem.eql(u8, arg, "--angle-cutoff")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.hbond_angle_cutoff = std.fmt.parseFloat(f32, raw[i]) catch return ParseArgsError.InvalidNumber;
        } else if (std.mem.eql(u8, arg, "--scheme")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.contacts_scheme = raw[i];
        } else if (std.mem.eql(u8, arg, "--cutoff")) {
            i += 1;
            if (i >= raw.len) return ParseArgsError.MissingValue;
            args.contacts_cutoff = std.fmt.parseFloat(f32, raw[i]) catch return ParseArgsError.InvalidNumber;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return ParseArgsError.UnknownFlag;
        } else {
            // Positional argument: trajectory path.
            if (traj_path == null) {
                traj_path = arg;
            }
        }
    }

    args.traj_path = traj_path orelse return ParseArgsError.MissingTrajectory;
    return args;
}

// ============================================================================
// File extension helpers
// ============================================================================

fn endsWithCI(s: []const u8, suffix: []const u8) bool {
    if (s.len < suffix.len) return false;
    const tail = s[s.len - suffix.len ..];
    for (tail, suffix) |a, b| {
        if (std.ascii.toLower(a) != std.ascii.toLower(b)) return false;
    }
    return true;
}

fn isPdb(path: []const u8) bool {
    return endsWithCI(path, ".pdb");
}

fn isCif(path: []const u8) bool {
    return endsWithCI(path, ".cif") or endsWithCI(path, ".mmcif");
}

fn isXtc(path: []const u8) bool {
    return endsWithCI(path, ".xtc");
}

fn isDcd(path: []const u8) bool {
    return endsWithCI(path, ".dcd");
}

// ============================================================================
// Topology loading
// ============================================================================

/// Load topology + first frame from a PDB or CIF file.
/// Returns a ParseResult; caller must call .deinit().
fn loadTopology(allocator: std.mem.Allocator, path: []const u8) !types.ParseResult {
    const data = try std.fs.cwd().readFileAlloc(allocator, path, 512 * 1024 * 1024);
    defer allocator.free(data);

    if (isPdb(path)) {
        return io.pdb.parse(allocator, data);
    } else if (isCif(path)) {
        return io.mmcif.parse(allocator, data);
    } else {
        // Default: try PDB.
        return io.pdb.parse(allocator, data);
    }
}

// ============================================================================
// Atom selection helper
// ============================================================================

/// Resolve the --select string to a slice of atom indices, or null (all atoms).
/// Caller owns the returned slice.
fn resolveSelection(
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
fn parsePairs(allocator: std.mem.Allocator, spec: []const u8) ![][2]u32 {
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
fn parseTriplets(allocator: std.mem.Allocator, spec: []const u8) ![][3]u32 {
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
fn parseQuartets(allocator: std.mem.Allocator, spec: []const u8) ![][4]u32 {
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

// ============================================================================
// Output: build result in an ArrayList and flush to file/stdout
// ============================================================================

/// Write the buffered output to either a file or stdout.
fn flushOutput(buf: []const u8, output_path: ?[]const u8) !void {
    if (output_path) |p| {
        const file = try std.fs.cwd().createFile(p, .{});
        defer file.close();
        try file.writeAll(buf);
    } else {
        const stdout = std.fs.File.stdout();
        try stdout.writeAll(buf);
    }
}

/// Write a single f64 array in the chosen format (one column).
fn writeScalarSeriesBuf(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    fmt: output.Format,
    key: []const u8,
    values: []const f64,
) !void {
    const w = buf.writer(allocator);
    switch (fmt) {
        .json => {
            try w.writeAll("{\n  ");
            try output.writeJsonArray(w, key, values);
            try w.writeAll("\n}\n");
        },
        .csv => {
            const headers = [_][]const u8{key};
            var rows = try allocator.alloc([]const f64, values.len);
            defer allocator.free(rows);
            for (values, 0..) |*v, i| rows[i] = v[0..1];
            try output.writeDelimited(w, &headers, rows, ',');
        },
        .tsv => {
            const headers = [_][]const u8{key};
            var rows = try allocator.alloc([]const f64, values.len);
            defer allocator.free(rows);
            for (values, 0..) |*v, i| rows[i] = v[0..1];
            try output.writeDelimited(w, &headers, rows, '\t');
        },
    }
}

// ============================================================================
// Frame collection
// ============================================================================

/// Load every frame from a trajectory or single-structure file.
/// Returns allocated []Frame (caller frees each frame then the slice).
fn loadAllFrames(
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
        while (reader.next()) |frame_ptr| {
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
        while (reader.next()) |frame_ptr| {
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
        else
            try io.pdb.parse(allocator, data);
        pr.topology.deinit();
        try frames.append(allocator, pr.frame);
    }

    return frames.toOwnedSlice(allocator);
}

// ============================================================================
// Subcommand: rmsd
// ============================================================================

fn runRmsd(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    if (frames.len == 0) {
        std.debug.print("error: no frames loaded\n", .{});
        std.process.exit(1);
    }

    if (args.ref_frame >= frames.len) {
        std.debug.print(
            "error: reference frame index {d} out of range (trajectory has {d} frames)\n",
            .{ args.ref_frame, frames.len },
        );
        std.process.exit(1);
    }

    const ref = frames[args.ref_frame];

    const rmsd_vals = try allocator.alloc(f64, frames.len);
    defer allocator.free(rmsd_vals);
    for (frames, 0..) |frame, fi| {
        rmsd_vals[fi] = geometry.rmsd.compute(
            ref.x, ref.y, ref.z,
            frame.x, frame.y, frame.z,
            atom_indices,
        );
    }

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    try writeScalarSeriesBuf(allocator, &buf, args.format, "rmsd", rmsd_vals);
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: rmsf
// ============================================================================

fn runRmsf(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const rmsf_vals = try geometry.rmsf.compute(allocator, frames, atom_indices);
    defer allocator.free(rmsf_vals);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    try writeScalarSeriesBuf(allocator, &buf, args.format, "rmsf", rmsf_vals);
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: distances
// ============================================================================

fn runDistances(allocator: std.mem.Allocator, args: Args) !void {
    const spec = args.pairs_spec orelse {
        std.debug.print("error: --pairs required for distances subcommand\n", .{});
        std.process.exit(1);
    };
    const pairs = try parsePairs(allocator, spec);
    defer allocator.free(pairs);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const n_pairs = pairs.len;
    const dist_buf = try allocator.alloc(f32, n_pairs);
    defer allocator.free(dist_buf);

    var all_rows = try allocator.alloc([]const f64, frames.len);
    defer allocator.free(all_rows);
    const row_storage = try allocator.alloc(f64, frames.len * n_pairs);
    defer allocator.free(row_storage);

    for (frames, 0..) |frame, fi| {
        geometry.distances.compute(frame.x, frame.y, frame.z, pairs, dist_buf);
        const row = row_storage[fi * n_pairs .. (fi + 1) * n_pairs];
        for (dist_buf, row) |d, *r| r.* = @floatCast(d);
        all_rows[fi] = row;
    }

    var headers = try allocator.alloc([]u8, n_pairs);
    defer {
        for (headers) |h| allocator.free(h);
        allocator.free(headers);
    }
    for (0..n_pairs) |pi| {
        headers[pi] = try std.fmt.allocPrint(allocator, "pair_{d}", .{pi});
    }
    const const_headers: []const []const u8 = @ptrCast(headers);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    switch (args.format) {
        .json => {
            try w.writeAll("{\n");
            for (0..n_pairs) |pi| {
                const col = try allocator.alloc(f64, frames.len);
                defer allocator.free(col);
                for (frames, 0..) |_, fi| col[fi] = all_rows[fi][pi];
                try w.writeAll("  ");
                try output.writeJsonArray(w, headers[pi], col);
                if (pi + 1 < n_pairs) try w.writeByte(',');
                try w.writeByte('\n');
            }
            try w.writeAll("}\n");
        },
        .csv => try output.writeDelimited(w, const_headers, all_rows, ','),
        .tsv => try output.writeDelimited(w, const_headers, all_rows, '\t'),
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: angles
// ============================================================================

fn runAngles(allocator: std.mem.Allocator, args: Args) !void {
    const spec = args.triplets_spec orelse {
        std.debug.print("error: --triplets required for angles subcommand\n", .{});
        std.process.exit(1);
    };
    const triplets = try parseTriplets(allocator, spec);
    defer allocator.free(triplets);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const n_tri = triplets.len;
    const angle_buf = try allocator.alloc(f32, n_tri);
    defer allocator.free(angle_buf);

    var all_rows = try allocator.alloc([]const f64, frames.len);
    defer allocator.free(all_rows);
    const row_storage = try allocator.alloc(f64, frames.len * n_tri);
    defer allocator.free(row_storage);

    for (frames, 0..) |frame, fi| {
        geometry.angles.compute(frame.x, frame.y, frame.z, triplets, angle_buf);
        const row = row_storage[fi * n_tri .. (fi + 1) * n_tri];
        for (angle_buf, row) |a, *r| r.* = @floatCast(a);
        all_rows[fi] = row;
    }

    var headers = try allocator.alloc([]u8, n_tri);
    defer {
        for (headers) |h| allocator.free(h);
        allocator.free(headers);
    }
    for (0..n_tri) |ti| {
        headers[ti] = try std.fmt.allocPrint(allocator, "angle_{d}", .{ti});
    }
    const const_headers: []const []const u8 = @ptrCast(headers);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);
    switch (args.format) {
        .json => {
            try w.writeAll("{\n");
            for (0..n_tri) |ti| {
                const col = try allocator.alloc(f64, frames.len);
                defer allocator.free(col);
                for (frames, 0..) |_, fi| col[fi] = all_rows[fi][ti];
                try w.writeAll("  ");
                try output.writeJsonArray(w, headers[ti], col);
                if (ti + 1 < n_tri) try w.writeByte(',');
                try w.writeByte('\n');
            }
            try w.writeAll("}\n");
        },
        .csv => try output.writeDelimited(w, const_headers, all_rows, ','),
        .tsv => try output.writeDelimited(w, const_headers, all_rows, '\t'),
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: dihedrals
// ============================================================================

fn runDihedrals(allocator: std.mem.Allocator, args: Args) !void {
    const spec = args.quartets_spec orelse {
        std.debug.print("error: --quartets required for dihedrals subcommand\n", .{});
        std.process.exit(1);
    };
    const quartets = try parseQuartets(allocator, spec);
    defer allocator.free(quartets);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const n_q = quartets.len;
    const dih_buf = try allocator.alloc(f32, n_q);
    defer allocator.free(dih_buf);

    var all_rows = try allocator.alloc([]const f64, frames.len);
    defer allocator.free(all_rows);
    const row_storage = try allocator.alloc(f64, frames.len * n_q);
    defer allocator.free(row_storage);

    for (frames, 0..) |frame, fi| {
        geometry.dihedrals.compute(frame.x, frame.y, frame.z, quartets, dih_buf);
        const row = row_storage[fi * n_q .. (fi + 1) * n_q];
        for (dih_buf, row) |d, *r| r.* = @floatCast(d);
        all_rows[fi] = row;
    }

    var headers = try allocator.alloc([]u8, n_q);
    defer {
        for (headers) |h| allocator.free(h);
        allocator.free(headers);
    }
    for (0..n_q) |qi| {
        headers[qi] = try std.fmt.allocPrint(allocator, "dihedral_{d}", .{qi});
    }
    const const_headers: []const []const u8 = @ptrCast(headers);

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);
    switch (args.format) {
        .json => {
            try w.writeAll("{\n");
            for (0..n_q) |qi| {
                const col = try allocator.alloc(f64, frames.len);
                defer allocator.free(col);
                for (frames, 0..) |_, fi| col[fi] = all_rows[fi][qi];
                try w.writeAll("  ");
                try output.writeJsonArray(w, headers[qi], col);
                if (qi + 1 < n_q) try w.writeByte(',');
                try w.writeByte('\n');
            }
            try w.writeAll("}\n");
        },
        .csv => try output.writeDelimited(w, const_headers, all_rows, ','),
        .tsv => try output.writeDelimited(w, const_headers, all_rows, '\t'),
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: rg
// ============================================================================

fn runRg(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const rg_vals = try allocator.alloc(f64, frames.len);
    defer allocator.free(rg_vals);
    for (frames, 0..) |frame, fi| {
        rg_vals[fi] = geometry.rg.compute(frame.x, frame.y, frame.z, masses, atom_indices);
    }

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    try writeScalarSeriesBuf(allocator, &buf, args.format, "rg", rg_vals);
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: center
// ============================================================================

fn runCenter(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    var all_rows = try allocator.alloc([]const f64, frames.len);
    defer allocator.free(all_rows);
    const row_storage = try allocator.alloc(f64, frames.len * 3);
    defer allocator.free(row_storage);

    for (frames, 0..) |frame, fi| {
        const com = geometry.center.ofMass(frame.x, frame.y, frame.z, masses, atom_indices);
        const row = row_storage[fi * 3 .. (fi + 1) * 3];
        row[0] = com[0];
        row[1] = com[1];
        row[2] = com[2];
        all_rows[fi] = row;
    }

    const headers = [_][]const u8{ "cx", "cy", "cz" };
    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);
    switch (args.format) {
        .json => {
            const cx_vals = try allocator.alloc(f64, frames.len);
            defer allocator.free(cx_vals);
            const cy_vals = try allocator.alloc(f64, frames.len);
            defer allocator.free(cy_vals);
            const cz_vals = try allocator.alloc(f64, frames.len);
            defer allocator.free(cz_vals);
            for (frames, 0..) |_, fi| {
                cx_vals[fi] = all_rows[fi][0];
                cy_vals[fi] = all_rows[fi][1];
                cz_vals[fi] = all_rows[fi][2];
            }
            const keys = [_][]const u8{ "cx", "cy", "cz" };
            const vals = [_][]const f64{ cx_vals, cy_vals, cz_vals };
            try output.writeJsonObject(w, &keys, &vals);
            try w.writeByte('\n');
        },
        .csv => try output.writeDelimited(w, &headers, all_rows, ','),
        .tsv => try output.writeDelimited(w, &headers, all_rows, '\t'),
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: inertia
// ============================================================================

fn runInertia(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    var all_rows = try allocator.alloc([]const f64, frames.len);
    defer allocator.free(all_rows);
    const row_storage = try allocator.alloc(f64, frames.len * 3);
    defer allocator.free(row_storage);

    for (frames, 0..) |frame, fi| {
        const tensor = geometry.inertia.compute(frame.x, frame.y, frame.z, masses, atom_indices);
        const moments = geometry.inertia.principalMoments(tensor);
        const row = row_storage[fi * 3 .. (fi + 1) * 3];
        row[0] = moments[0];
        row[1] = moments[1];
        row[2] = moments[2];
        all_rows[fi] = row;
    }

    const headers = [_][]const u8{ "I1", "I2", "I3" };
    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);
    switch (args.format) {
        .json => {
            const mom1 = try allocator.alloc(f64, frames.len);
            defer allocator.free(mom1);
            const mom2 = try allocator.alloc(f64, frames.len);
            defer allocator.free(mom2);
            const mom3 = try allocator.alloc(f64, frames.len);
            defer allocator.free(mom3);
            for (frames, 0..) |_, fi| {
                mom1[fi] = all_rows[fi][0];
                mom2[fi] = all_rows[fi][1];
                mom3[fi] = all_rows[fi][2];
            }
            const keys = [_][]const u8{ "I1", "I2", "I3" };
            const vals = [_][]const f64{ mom1, mom2, mom3 };
            try output.writeJsonObject(w, &keys, &vals);
            try w.writeByte('\n');
        },
        .csv => try output.writeDelimited(w, &headers, all_rows, ','),
        .tsv => try output.writeDelimited(w, &headers, all_rows, '\t'),
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: hbonds
// ============================================================================

fn runHbonds(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const cfg = analysis.hbonds.Config{
        .dist_cutoff = args.hbond_dist_cutoff,
        .angle_cutoff = args.hbond_angle_cutoff,
    };

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    switch (args.format) {
        .json => {
            try w.writeAll("[\n");
            for (frames, 0..) |frame, fi| {
                const bonds = try analysis.hbonds.detect(allocator, parsed.topology, frame, cfg);
                defer allocator.free(bonds);
                if (fi > 0) try w.writeAll(",\n");
                try w.print("  {{\"frame\": {d}, \"hbonds\": [", .{fi});
                for (bonds, 0..) |hb, bi| {
                    if (bi > 0) try w.writeAll(", ");
                    try w.print(
                        "{{\"donor\": {d}, \"hydrogen\": {d}, \"acceptor\": {d}, \"distance\": {d:.4}, \"angle\": {d:.4}}}",
                        .{ hb.donor, hb.hydrogen, hb.acceptor, hb.distance, hb.angle },
                    );
                }
                try w.writeAll("]}");
            }
            try w.writeAll("\n]\n");
        },
        .csv, .tsv => {
            const delim: u8 = if (args.format == .csv) ',' else '\t';
            try w.print("frame{c}donor{c}hydrogen{c}acceptor{c}distance{c}angle\n", .{ delim, delim, delim, delim, delim });
            for (frames, 0..) |frame, fi| {
                const bonds = try analysis.hbonds.detect(allocator, parsed.topology, frame, cfg);
                defer allocator.free(bonds);
                for (bonds) |hb| {
                    try w.print("{d}{c}{d}{c}{d}{c}{d}{c}{d:.4}{c}{d:.4}\n", .{
                        fi, delim, hb.donor, delim, hb.hydrogen, delim, hb.acceptor,
                        delim, hb.distance, delim, hb.angle,
                    });
                }
            }
        },
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: contacts
// ============================================================================

fn runContacts(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const scheme: analysis.contacts.Scheme = blk: {
        if (std.mem.eql(u8, args.contacts_scheme, "ca")) break :blk .ca;
        if (std.mem.eql(u8, args.contacts_scheme, "closest")) break :blk .closest;
        break :blk .closest_heavy;
    };

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    switch (args.format) {
        .json => {
            try w.writeAll("[\n");
            for (frames, 0..) |frame, fi| {
                const ctcts = try analysis.contacts.compute(
                    allocator, parsed.topology, frame, scheme, args.contacts_cutoff,
                );
                defer allocator.free(ctcts);
                if (fi > 0) try w.writeAll(",\n");
                try w.print("  {{\"frame\": {d}, \"contacts\": [", .{fi});
                for (ctcts, 0..) |ct, ci| {
                    if (ci > 0) try w.writeAll(", ");
                    try w.print(
                        "{{\"residue_i\": {d}, \"residue_j\": {d}, \"distance\": {d:.4}}}",
                        .{ ct.residue_i, ct.residue_j, ct.distance },
                    );
                }
                try w.writeAll("]}");
            }
            try w.writeAll("\n]\n");
        },
        .csv, .tsv => {
            const delim: u8 = if (args.format == .csv) ',' else '\t';
            try w.print("frame{c}residue_i{c}residue_j{c}distance\n", .{ delim, delim, delim });
            for (frames, 0..) |frame, fi| {
                const ctcts = try analysis.contacts.compute(
                    allocator, parsed.topology, frame, scheme, args.contacts_cutoff,
                );
                defer allocator.free(ctcts);
                for (ctcts) |ct| {
                    try w.print("{d}{c}{d}{c}{d}{c}{d:.4}\n", .{
                        fi, delim, ct.residue_i, delim, ct.residue_j, delim, ct.distance,
                    });
                }
            }
        },
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: rdf
// ============================================================================

fn runRdf(allocator: std.mem.Allocator, args: Args) !void {
    if (args.sel1 == null or args.sel2 == null) {
        std.debug.print("error: --sel1 and --sel2 required for rdf subcommand\n", .{});
        std.process.exit(1);
    }

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loadTopology(allocator, top_path);
    defer parsed.deinit();

    const idx1 = try resolveSelection(allocator, parsed.topology, args.sel1);
    defer if (idx1) |ai| allocator.free(ai);
    const idx2 = try resolveSelection(allocator, parsed.topology, args.sel2);
    defer if (idx2) |ai| allocator.free(ai);

    const frames = try loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    if (frames.len == 0) {
        std.debug.print("error: no frames loaded\n", .{});
        std.process.exit(1);
    }

    const cfg = analysis.rdf.Config{
        .r_max = args.rmax,
        .n_bins = args.rdf_bins,
    };

    // Accumulate g(r) over all frames then average.
    var accumulated_r: ?[]f64 = null;
    defer if (accumulated_r) |ar| allocator.free(ar);
    const accumulated_gr = try allocator.alloc(f64, args.rdf_bins);
    defer allocator.free(accumulated_gr);
    @memset(accumulated_gr, 0.0);

    for (frames) |frame| {
        const n1: usize = if (idx1) |ai| ai.len else frame.nAtoms();
        const n2: usize = if (idx2) |ai| ai.len else frame.nAtoms();

        const s1x = try allocator.alloc(f32, n1);
        defer allocator.free(s1x);
        const s1y = try allocator.alloc(f32, n1);
        defer allocator.free(s1y);
        const s1z = try allocator.alloc(f32, n1);
        defer allocator.free(s1z);
        const s2x = try allocator.alloc(f32, n2);
        defer allocator.free(s2x);
        const s2y = try allocator.alloc(f32, n2);
        defer allocator.free(s2y);
        const s2z = try allocator.alloc(f32, n2);
        defer allocator.free(s2z);

        if (idx1) |ai| {
            for (ai, 0..) |atom_idx, k| {
                s1x[k] = frame.x[atom_idx];
                s1y[k] = frame.y[atom_idx];
                s1z[k] = frame.z[atom_idx];
            }
        } else {
            @memcpy(s1x, frame.x);
            @memcpy(s1y, frame.y);
            @memcpy(s1z, frame.z);
        }

        if (idx2) |ai| {
            for (ai, 0..) |atom_idx, k| {
                s2x[k] = frame.x[atom_idx];
                s2y[k] = frame.y[atom_idx];
                s2z[k] = frame.z[atom_idx];
            }
        } else {
            @memcpy(s2x, frame.x);
            @memcpy(s2y, frame.y);
            @memcpy(s2z, frame.z);
        }

        // Estimate box volume from box vectors, or fall back to 1000 Å³.
        const box_vol: f64 = if (frame.box_vectors) |bv| blk: {
            const ax: f64 = bv[0][0];
            const ay: f64 = bv[0][1];
            const az: f64 = bv[0][2];
            const bx: f64 = bv[1][0];
            const by: f64 = bv[1][1];
            const bz: f64 = bv[1][2];
            const cx: f64 = bv[2][0];
            const cy: f64 = bv[2][1];
            const cz: f64 = bv[2][2];
            const cross_x = by * cz - bz * cy;
            const cross_y = bz * cx - bx * cz;
            const cross_z = bx * cy - by * cx;
            const vol = @abs(ax * cross_x + ay * cross_y + az * cross_z);
            break :blk if (vol > 0.0) vol else 1000.0;
        } else 1000.0;

        var result = try analysis.rdf.compute(
            allocator, s1x, s1y, s1z, s2x, s2y, s2z, box_vol, cfg,
        );
        defer result.deinit();

        if (accumulated_r == null) {
            accumulated_r = try allocator.dupe(f64, result.r);
        }
        for (result.g_r, accumulated_gr) |g, *acc| acc.* += g;
    }

    const n_frames_f: f64 = @floatFromInt(frames.len);
    for (accumulated_gr) |*g| g.* /= n_frames_f;

    const r_vals = accumulated_r orelse {
        std.debug.print("error: RDF accumulation failed\n", .{});
        std.process.exit(1);
    };

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    const headers = [_][]const u8{ "r", "g_r" };
    var all_rows = try allocator.alloc([]const f64, r_vals.len);
    defer allocator.free(all_rows);
    const row_storage = try allocator.alloc(f64, r_vals.len * 2);
    defer allocator.free(row_storage);
    for (r_vals, 0..) |r, ri| {
        row_storage[ri * 2] = r;
        row_storage[ri * 2 + 1] = accumulated_gr[ri];
        all_rows[ri] = row_storage[ri * 2 .. ri * 2 + 2];
    }

    switch (args.format) {
        .json => {
            const keys = [_][]const u8{ "r", "g_r" };
            const vals = [_][]const f64{ r_vals, accumulated_gr };
            try output.writeJsonObject(w, &keys, &vals);
            try w.writeByte('\n');
        },
        .csv => try output.writeDelimited(w, &headers, all_rows, ','),
        .tsv => try output.writeDelimited(w, &headers, all_rows, '\t'),
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// main
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const raw_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, raw_args);

    if (raw_args.len < 2) {
        printUsage(raw_args[0]);
        std.process.exit(1);
    }

    const first = raw_args[1];

    if (std.mem.eql(u8, first, "--version") or std.mem.eql(u8, first, "-V")) {
        const stdout = std.fs.File.stdout();
        var buf: [64]u8 = undefined;
        const line = try std.fmt.bufPrint(&buf, "ztraj {s}\n", .{build_options.version});
        try stdout.writeAll(line);
        return;
    }
    if (std.mem.eql(u8, first, "--help") or std.mem.eql(u8, first, "-h")) {
        printUsage(raw_args[0]);
        return;
    }

    const args = parseArgs(raw_args) catch |err| {
        switch (err) {
            ParseArgsError.MissingSubcommand => std.debug.print("error: missing subcommand\n", .{}),
            ParseArgsError.MissingTrajectory => std.debug.print("error: missing trajectory/structure file argument\n", .{}),
            ParseArgsError.UnknownSubcommand => std.debug.print("error: unknown subcommand '{s}'\n", .{first}),
            ParseArgsError.UnknownFlag => std.debug.print("error: unknown flag in arguments\n", .{}),
            ParseArgsError.MissingValue => std.debug.print("error: flag requires a value\n", .{}),
            ParseArgsError.InvalidFormat => std.debug.print("error: invalid --format (use json, csv, or tsv)\n", .{}),
            ParseArgsError.InvalidNumber => std.debug.print("error: invalid numeric argument\n", .{}),
        }
        printUsage(raw_args[0]);
        std.process.exit(1);
    };

    const result = switch (args.subcommand) {
        .rmsd => runRmsd(allocator, args),
        .rmsf => runRmsf(allocator, args),
        .distances => runDistances(allocator, args),
        .angles => runAngles(allocator, args),
        .dihedrals => runDihedrals(allocator, args),
        .rg => runRg(allocator, args),
        .center => runCenter(allocator, args),
        .inertia => runInertia(allocator, args),
        .hbonds => runHbonds(allocator, args),
        .contacts => runContacts(allocator, args),
        .rdf => runRdf(allocator, args),
    };

    result catch |err| {
        std.debug.print("error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}
