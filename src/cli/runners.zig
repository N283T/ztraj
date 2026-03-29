//! Subcommand runner functions and output helpers.

const std = @import("std");
const ztraj = @import("ztraj");

const types = ztraj.types;
const geometry = ztraj.geometry;
const analysis = ztraj.analysis;
const output = ztraj.output;

const args_mod = @import("args.zig");
const Args = args_mod.Args;
const loader = @import("loader.zig");
const parsers = @import("parsers.zig");

// ============================================================================
// Output: build result in an ArrayList and flush to file/stdout
// ============================================================================

/// Write the buffered output to either a file or stdout.
pub fn flushOutput(buf: []const u8, output_path: ?[]const u8) !void {
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
pub fn writeScalarSeriesBuf(
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
// Subcommand: rmsd
// ============================================================================

pub fn runRmsd(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "rmsd" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
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
    const analysis_node = progress_root.start("Computing RMSD", frames.len);
    for (frames, 0..) |frame, fi| {
        rmsd_vals[fi] = geometry.rmsd.compute(
            ref.x,
            ref.y,
            ref.z,
            frame.x,
            frame.y,
            frame.z,
            atom_indices,
        );
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    try writeScalarSeriesBuf(allocator, &buf, args.format, "rmsd", rmsd_vals);
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: rmsf
// ============================================================================

pub fn runRmsf(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "rmsf" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const n_threads = if (args.n_threads == 0) (std.Thread.getCpuCount() catch 1) else args.n_threads;
    const analysis_node = progress_root.start("Computing RMSF", 0);
    const rmsf_vals = try geometry.rmsf.computeParallel(allocator, frames, atom_indices, n_threads);
    analysis_node.end();
    defer allocator.free(rmsf_vals);
    progress_root.end();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    try writeScalarSeriesBuf(allocator, &buf, args.format, "rmsf", rmsf_vals);
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: distances
// ============================================================================

pub fn runDistances(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "distances" });

    const spec = args.pairs_spec orelse {
        std.debug.print("error: --pairs required for distances subcommand\n", .{});
        std.process.exit(1);
    };
    const pairs = try parsers.parsePairs(allocator, spec);
    defer allocator.free(pairs);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    try parsers.validateIndices(2, pairs, @intCast(parsed.topology.atoms.len));

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
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

    const analysis_node = progress_root.start("Computing distances", frames.len);
    for (frames, 0..) |frame, fi| {
        geometry.distances.compute(frame.x, frame.y, frame.z, pairs, dist_buf);
        const row = row_storage[fi * n_pairs .. (fi + 1) * n_pairs];
        for (dist_buf, row) |d, *r| r.* = @floatCast(d);
        all_rows[fi] = row;
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

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

pub fn runAngles(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "angles" });

    const spec = args.triplets_spec orelse {
        std.debug.print("error: --triplets required for angles subcommand\n", .{});
        std.process.exit(1);
    };
    const triplets = try parsers.parseTriplets(allocator, spec);
    defer allocator.free(triplets);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    try parsers.validateIndices(3, triplets, @intCast(parsed.topology.atoms.len));

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
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

    const analysis_node = progress_root.start("Computing angles", frames.len);
    for (frames, 0..) |frame, fi| {
        geometry.angles.compute(frame.x, frame.y, frame.z, triplets, angle_buf);
        const row = row_storage[fi * n_tri .. (fi + 1) * n_tri];
        for (angle_buf, row) |a, *r| r.* = @floatCast(a);
        all_rows[fi] = row;
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

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

pub fn runDihedrals(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "dihedrals" });

    const spec = args.quartets_spec orelse {
        std.debug.print("error: --quartets required for dihedrals subcommand\n", .{});
        std.process.exit(1);
    };
    const quartets = try parsers.parseQuartets(allocator, spec);
    defer allocator.free(quartets);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    try parsers.validateIndices(4, quartets, @intCast(parsed.topology.atoms.len));

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
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

    const analysis_node = progress_root.start("Computing dihedrals", frames.len);
    for (frames, 0..) |frame, fi| {
        geometry.dihedrals.compute(frame.x, frame.y, frame.z, quartets, dih_buf);
        const row = row_storage[fi * n_q .. (fi + 1) * n_q];
        for (dih_buf, row) |d, *r| r.* = @floatCast(d);
        all_rows[fi] = row;
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

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

pub fn runRg(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "rg" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const rg_vals = try allocator.alloc(f64, frames.len);
    defer allocator.free(rg_vals);
    const analysis_node = progress_root.start("Computing Rg", frames.len);
    for (frames, 0..) |frame, fi| {
        rg_vals[fi] = geometry.rg.compute(frame.x, frame.y, frame.z, masses, atom_indices);
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    try writeScalarSeriesBuf(allocator, &buf, args.format, "rg", rg_vals);
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: center
// ============================================================================

pub fn runCenter(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "center" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    var all_rows = try allocator.alloc([]const f64, frames.len);
    defer allocator.free(all_rows);
    const row_storage = try allocator.alloc(f64, frames.len * 3);
    defer allocator.free(row_storage);

    const analysis_node = progress_root.start("Computing center", frames.len);
    for (frames, 0..) |frame, fi| {
        const com = geometry.center.ofMass(frame.x, frame.y, frame.z, masses, atom_indices);
        const row = row_storage[fi * 3 .. (fi + 1) * 3];
        row[0] = com[0];
        row[1] = com[1];
        row[2] = com[2];
        all_rows[fi] = row;
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

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

pub fn runInertia(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "inertia" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    var all_rows = try allocator.alloc([]const f64, frames.len);
    defer allocator.free(all_rows);
    const row_storage = try allocator.alloc(f64, frames.len * 3);
    defer allocator.free(row_storage);

    const analysis_node = progress_root.start("Computing inertia", frames.len);
    for (frames, 0..) |frame, fi| {
        const tensor = geometry.inertia.compute(frame.x, frame.y, frame.z, masses, atom_indices);
        const moments = geometry.inertia.principalMoments(tensor);
        const row = row_storage[fi * 3 .. (fi + 1) * 3];
        row[0] = moments[0];
        row[1] = moments[1];
        row[2] = moments[2];
        all_rows[fi] = row;
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

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

pub fn runHbonds(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "hbonds" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const cfg = analysis.hbonds.Config{
        .dist_cutoff = args.hbond_dist_cutoff,
        .angle_cutoff = args.hbond_angle_cutoff,
    };

    const n_threads = if (args.n_threads == 0) (std.Thread.getCpuCount() catch 1) else args.n_threads;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    const analysis_node = progress_root.start("Detecting H-bonds", frames.len);
    switch (args.format) {
        .json => {
            try w.writeAll("[\n");
            for (frames, 0..) |frame, fi| {
                const bonds = try analysis.hbonds.detectParallel(allocator, parsed.topology, frame, cfg, n_threads);
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
                analysis_node.completeOne();
            }
            try w.writeAll("\n]\n");
        },
        .csv, .tsv => {
            const delim: u8 = if (args.format == .csv) ',' else '\t';
            try w.print("frame{c}donor{c}hydrogen{c}acceptor{c}distance{c}angle\n", .{ delim, delim, delim, delim, delim });
            for (frames, 0..) |frame, fi| {
                const bonds = try analysis.hbonds.detectParallel(allocator, parsed.topology, frame, cfg, n_threads);
                defer allocator.free(bonds);
                for (bonds) |hb| {
                    try w.print("{d}{c}{d}{c}{d}{c}{d}{c}{d:.4}{c}{d:.4}\n", .{
                        fi,    delim,       hb.donor, delim,    hb.hydrogen, delim, hb.acceptor,
                        delim, hb.distance, delim,    hb.angle,
                    });
                }
                analysis_node.completeOne();
            }
        },
    }
    analysis_node.end();
    progress_root.end();
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: contacts
// ============================================================================

pub fn runContacts(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "contacts" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const scheme: analysis.contacts.Scheme = blk: {
        if (std.mem.eql(u8, args.contacts_scheme, "ca")) break :blk .ca;
        if (std.mem.eql(u8, args.contacts_scheme, "closest")) break :blk .closest;
        if (std.mem.eql(u8, args.contacts_scheme, "closest_heavy")) break :blk .closest_heavy;
        std.debug.print(
            "error: unknown contacts scheme '{s}' (valid values: ca, closest, closest_heavy)\n",
            .{args.contacts_scheme},
        );
        std.process.exit(1);
    };

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    const analysis_node = progress_root.start("Computing contacts", frames.len);
    switch (args.format) {
        .json => {
            const n_threads = if (args.n_threads == 0) (std.Thread.getCpuCount() catch 1) else args.n_threads;
            try w.writeAll("[\n");
            for (frames, 0..) |frame, fi| {
                const ctcts = try analysis.contacts.computeParallel(
                    allocator,
                    parsed.topology,
                    frame,
                    scheme,
                    args.contacts_cutoff,
                    n_threads,
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
                analysis_node.completeOne();
            }
            try w.writeAll("\n]\n");
        },
        .csv, .tsv => {
            const n_threads = if (args.n_threads == 0) (std.Thread.getCpuCount() catch 1) else args.n_threads;
            const delim: u8 = if (args.format == .csv) ',' else '\t';
            try w.print("frame{c}residue_i{c}residue_j{c}distance\n", .{ delim, delim, delim });
            for (frames, 0..) |frame, fi| {
                const ctcts = try analysis.contacts.computeParallel(
                    allocator,
                    parsed.topology,
                    frame,
                    scheme,
                    args.contacts_cutoff,
                    n_threads,
                );
                defer allocator.free(ctcts);
                for (ctcts) |ct| {
                    try w.print("{d}{c}{d}{c}{d}{c}{d:.4}\n", .{
                        fi, delim, ct.residue_i, delim, ct.residue_j, delim, ct.distance,
                    });
                }
                analysis_node.completeOne();
            }
        },
    }
    analysis_node.end();
    progress_root.end();
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: rdf
// ============================================================================

pub fn runRdf(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "rdf" });

    if (args.sel1 == null or args.sel2 == null) {
        std.debug.print("error: --sel1 and --sel2 required for rdf subcommand\n", .{});
        std.process.exit(1);
    }

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const idx1 = try parsers.resolveSelection(allocator, parsed.topology, args.sel1);
    defer if (idx1) |ai| allocator.free(ai);
    const idx2 = try parsers.resolveSelection(allocator, parsed.topology, args.sel2);
    defer if (idx2) |ai| allocator.free(ai);

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
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

    const n_threads = if (args.n_threads == 0) (std.Thread.getCpuCount() catch 1) else args.n_threads;

    const analysis_node = progress_root.start("Computing RDF", frames.len);
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

        // RDF requires periodic box information to compute number density.
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
            if (vol <= 0.0) {
                std.debug.print(
                    "error: rdf requires a valid periodic box but frame box volume is zero\n",
                    .{},
                );
                std.process.exit(1);
            }
            break :blk vol;
        } else {
            std.debug.print(
                "error: rdf requires periodic box information but no box vectors are present in the trajectory\n",
                .{},
            );
            std.process.exit(1);
        };

        var result = try analysis.rdf.computeParallel(
            allocator,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
            box_vol,
            cfg,
            n_threads,
        );
        defer result.deinit();

        if (accumulated_r == null) {
            accumulated_r = try allocator.dupe(f64, result.r);
        }
        for (result.g_r, accumulated_gr) |g, *acc| acc.* += g;
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

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
// Subcommand: sasa
// ============================================================================

pub fn runSasa(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "sasa" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    // Per-frame total SASA
    const sasa_vals = try allocator.alloc(f64, frames.len);
    defer allocator.free(sasa_vals);

    const analysis_node = progress_root.start("Computing SASA", frames.len);
    for (frames, 0..) |frame, fi| {
        var result = try analysis.sasa.compute(
            allocator,
            frame.x,
            frame.y,
            frame.z,
            parsed.topology,
            atom_indices,
            .{ .n_points = 100, .probe_radius = 1.4, .n_threads = 0 },
        );
        defer result.deinit();
        sasa_vals[fi] = result.total_area;
        analysis_node.completeOne();
    }
    analysis_node.end();
    progress_root.end();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    try writeScalarSeriesBuf(allocator, &buf, args.format, "sasa", sasa_vals);
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: all (combined analysis)
// ============================================================================

pub fn runAll(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "all" });

    // The all command only supports JSON output (multi-metric data
    // doesn't map to flat CSV/TSV)
    if (args.format != .json) {
        const stderr = std.fs.File.stderr();
        try stderr.writeAll("error: 'all' command only supports JSON output (--format json)\n");
        return error.UnsupportedFormat;
    }

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }

    const n_frames = frames.len;
    if (n_frames == 0) return error.NoFrames;

    // Pre-allocate result arrays
    const rmsd_vals = try allocator.alloc(f64, n_frames);
    defer allocator.free(rmsd_vals);
    const rg_vals = try allocator.alloc(f64, n_frames);
    defer allocator.free(rg_vals);
    const sasa_vals = try allocator.alloc(f64, n_frames);
    defer allocator.free(sasa_vals);
    const com_x = try allocator.alloc(f64, n_frames);
    defer allocator.free(com_x);
    const com_y = try allocator.alloc(f64, n_frames);
    defer allocator.free(com_y);
    const com_z = try allocator.alloc(f64, n_frames);
    defer allocator.free(com_z);
    const n_hbonds = try allocator.alloc(f64, n_frames);
    defer allocator.free(n_hbonds);
    const n_contacts = try allocator.alloc(f64, n_frames);
    defer allocator.free(n_contacts);

    // Reference frame for RMSD
    const ref = frames[0];

    // Per-frame analysis
    const n_threads = if (args.n_threads == 0) (std.Thread.getCpuCount() catch 1) else args.n_threads;
    const hbonds_cfg = analysis.hbonds.Config{};
    const contacts_scheme: analysis.contacts.Scheme = .closest_heavy;
    const contacts_cutoff: f32 = 4.5;

    const analysis_node = progress_root.start("Analyzing frames", n_frames);
    for (frames, 0..) |frame, fi| {
        // RMSD
        rmsd_vals[fi] = geometry.rmsd.compute(
            ref.x,
            ref.y,
            ref.z,
            frame.x,
            frame.y,
            frame.z,
            atom_indices,
        );

        // Rg
        rg_vals[fi] = geometry.rg.compute(frame.x, frame.y, frame.z, masses, atom_indices);

        // Center of mass
        const com = geometry.center.ofMass(frame.x, frame.y, frame.z, masses, atom_indices);
        com_x[fi] = com[0];
        com_y[fi] = com[1];
        com_z[fi] = com[2];

        // SASA
        var sasa_result = try analysis.sasa.compute(
            allocator,
            frame.x,
            frame.y,
            frame.z,
            parsed.topology,
            atom_indices,
            .{ .n_points = 100, .probe_radius = 1.4, .n_threads = 0 },
        );
        defer sasa_result.deinit();
        sasa_vals[fi] = sasa_result.total_area;

        // Hbonds count
        const hbonds = try analysis.hbonds.detectParallel(allocator, parsed.topology, frame, hbonds_cfg, n_threads);
        defer allocator.free(hbonds);
        n_hbonds[fi] = @floatFromInt(hbonds.len);

        // Contacts count
        const contacts = try analysis.contacts.computeParallel(allocator, parsed.topology, frame, contacts_scheme, contacts_cutoff, n_threads);
        defer allocator.free(contacts);
        n_contacts[fi] = @floatFromInt(contacts.len);
        analysis_node.completeOne();
    }
    analysis_node.end();

    // RMSF (across all frames)
    const rmsf_node = progress_root.start("Computing RMSF", 0);
    const rmsf_vals = try geometry.rmsf.computeParallel(allocator, frames, atom_indices, n_threads);
    rmsf_node.end();
    defer allocator.free(rmsf_vals);
    progress_root.end();

    // Write JSON output
    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    const n_atoms = if (atom_indices) |ai| ai.len else parsed.topology.atoms.len;

    try w.print("{{\n  \"n_frames\": {d},\n  \"n_atoms\": {d},\n", .{ n_frames, n_atoms });

    // Scalar series
    const series_keys = [_][]const u8{ "rmsd", "rg", "sasa", "n_hbonds", "n_contacts" };
    const series_vals = [_][]const f64{ rmsd_vals, rg_vals, sasa_vals, n_hbonds, n_contacts };
    for (series_keys, series_vals) |key, vals| {
        try w.writeAll("  ");
        try output.writeJsonArray(w, key, vals);
        try w.writeAll(",\n");
    }

    // RMSF
    try w.writeAll("  ");
    try output.writeJsonArray(w, "rmsf", rmsf_vals);
    try w.writeAll(",\n");

    // Center of mass as array of [x, y, z]
    try w.writeAll("  \"center_of_mass\": [");
    for (0..n_frames) |fi| {
        if (fi > 0) try w.writeAll(", ");
        try w.print("[{d:.6}, {d:.6}, {d:.6}]", .{ com_x[fi], com_y[fi], com_z[fi] });
    }
    try w.writeAll("]\n}\n");

    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: dssp
// ============================================================================

pub fn runDssp(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "dssp" });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }
    if (frames.len == 0) return error.NoFrames;

    const dssp_mod = ztraj.dssp;
    const n_threads = if (args.n_threads == 0) (std.Thread.getCpuCount() catch 1) else args.n_threads;
    const config = dssp_mod.DsspConfigT{ .n_threads = n_threads };

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    if (frames.len == 1) {
        // Single frame: output per-residue SS
        var result = try dssp_mod.compute(allocator, parsed.topology, frames[0], config);
        defer result.deinit();

        switch (args.format) {
            .json => {
                try w.writeAll("[\n");
                for (result.residues, 0..) |res, i| {
                    if (i > 0) try w.writeAll(",\n");
                    const topo_res = parsed.topology.residues[res.residue_index];
                    try w.print(
                        "  {{\"index\": {d}, \"resid\": {d}, \"resname\": \"{s}\", \"ss\": \"{c}\"}}",
                        .{
                            i,
                            topo_res.resid,
                            topo_res.name.slice(),
                            res.secondary_structure.toChar(),
                        },
                    );
                }
                try w.writeAll("\n]\n");
            },
            .csv, .tsv => {
                const delim: u8 = if (args.format == .csv) ',' else '\t';
                try w.print("index{c}resid{c}resname{c}ss\n", .{ delim, delim, delim });
                for (result.residues, 0..) |res, i| {
                    const topo_res = parsed.topology.residues[res.residue_index];
                    try w.print("{d}{c}{d}{c}{s}{c}{c}\n", .{
                        i,                                delim,                 topo_res.resid,
                        delim,                            topo_res.name.slice(), delim,
                        res.secondary_structure.toChar(),
                    });
                }
            },
        }
    } else {
        // Multi-frame: output per-frame SS string
        const analysis_node = progress_root.start("Computing DSSP", frames.len);
        switch (args.format) {
            .json => {
                try w.writeAll("{\n  \"dssp\": [\n");
                for (frames, 0..) |frame, fi| {
                    var result = try dssp_mod.compute(allocator, parsed.topology, frame, config);
                    defer result.deinit();

                    if (fi > 0) try w.writeAll(",\n");
                    try w.writeAll("    \"");
                    for (result.residues) |res| {
                        try w.writeByte(res.secondary_structure.toChar());
                    }
                    try w.writeByte('"');
                    analysis_node.completeOne();
                }
                try w.writeAll("\n  ]\n}\n");
            },
            .csv, .tsv => {
                const delim: u8 = if (args.format == .csv) ',' else '\t';
                try w.print("frame{c}dssp\n", .{delim});
                for (frames, 0..) |frame, fi| {
                    var result = try dssp_mod.compute(allocator, parsed.topology, frame, config);
                    defer result.deinit();

                    try w.print("{d}{c}", .{ fi, delim });
                    for (result.residues) |res| {
                        try w.writeByte(res.secondary_structure.toChar());
                    }
                    try w.writeByte('\n');
                    analysis_node.completeOne();
                }
            },
        }
        analysis_node.end();
    }
    progress_root.end();
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommands: phi, psi, omega, chi (protein-specific dihedrals)
// ============================================================================

const prot_dih = geometry.protein_dihedrals;

fn runProteinDihedral(
    allocator: std.mem.Allocator,
    args: Args,
    comptime key: []const u8,
    computeFn: anytype,
) !void {
    const progress_root = std.Progress.start(.{ .root_name = key });

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }
    if (frames.len == 0) return error.NoFrames;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    // Compute for each frame
    const analysis_node = progress_root.start("Computing " ++ key, frames.len);
    defer {
        analysis_node.end();
        progress_root.end();
    }
    switch (args.format) {
        .json => {
            try w.writeAll("{\n  \"" ++ key ++ "\": [\n");
            for (frames, 0..) |frame, fi| {
                const vals = try computeFn(allocator, parsed.topology, frame);
                defer allocator.free(vals);
                if (fi > 0) try w.writeAll(",\n");
                try w.writeAll("    [");
                for (vals, 0..) |v, i| {
                    if (i > 0) try w.writeAll(", ");
                    if (v) |angle| {
                        try w.print("{d:.6}", .{angle});
                    } else {
                        try w.writeAll("null");
                    }
                }
                try w.writeByte(']');
                analysis_node.completeOne();
            }
            try w.writeAll("\n  ]\n}\n");
        },
        .csv, .tsv => {
            const delim: u8 = if (args.format == .csv) ',' else '\t';
            // Header: frame, res_0, res_1, ...
            try w.writeAll("frame");
            for (0..parsed.topology.residues.len) |ri| {
                try w.print("{c}res_{d}", .{ delim, ri });
            }
            try w.writeByte('\n');
            for (frames, 0..) |frame, fi| {
                const vals = try computeFn(allocator, parsed.topology, frame);
                defer allocator.free(vals);
                try w.print("{d}", .{fi});
                for (vals) |v| {
                    try w.writeByte(delim);
                    if (v) |angle| {
                        try w.print("{d:.6}", .{angle});
                    }
                }
                try w.writeByte('\n');
                analysis_node.completeOne();
            }
        },
    }
    try flushOutput(buf.items, args.output_path);
}

pub fn runPhi(allocator: std.mem.Allocator, args: Args) !void {
    return runProteinDihedral(allocator, args, "phi", prot_dih.computePhi);
}

pub fn runPsi(allocator: std.mem.Allocator, args: Args) !void {
    return runProteinDihedral(allocator, args, "psi", prot_dih.computePsi);
}

pub fn runOmega(allocator: std.mem.Allocator, args: Args) !void {
    return runProteinDihedral(allocator, args, "omega", prot_dih.computeOmega);
}

pub fn runChi(allocator: std.mem.Allocator, args: Args) !void {
    const progress_root = std.Progress.start(.{ .root_name = "chi" });

    // chi level could be a CLI arg, default to 1 for now
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loader.loadAllFramesWithProgress(allocator, args.traj_path, parsed.topology.atoms.len, progress_root);
    defer {
        for (frames) |*f| @constCast(f).deinit();
        allocator.free(frames);
    }
    if (frames.len == 0) return error.NoFrames;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    // Output chi1-chi4 for each frame
    const analysis_node = progress_root.start("Computing chi", frames.len);
    defer {
        analysis_node.end();
        progress_root.end();
    }
    switch (args.format) {
        .json => {
            try w.writeAll("{\n");
            const chi_names = [_][]const u8{ "chi1", "chi2", "chi3", "chi4" };
            for (chi_names, 0..) |chi_name, ci| {
                const level: u8 = @intCast(ci + 1);
                if (ci > 0) try w.writeAll(",\n");
                try w.print("  \"{s}\": [\n", .{chi_name});
                for (frames, 0..) |frame, fi| {
                    const vals = try prot_dih.computeChi(allocator, parsed.topology, frame, level);
                    defer allocator.free(vals);
                    if (fi > 0) try w.writeAll(",\n");
                    try w.writeAll("    [");
                    for (vals, 0..) |v, i| {
                        if (i > 0) try w.writeAll(", ");
                        if (v) |angle| {
                            try w.print("{d:.6}", .{angle});
                        } else {
                            try w.writeAll("null");
                        }
                    }
                    try w.writeByte(']');
                    if (ci == 0) analysis_node.completeOne();
                }
                try w.writeAll("\n  ]");
            }
            try w.writeAll("\n}\n");
        },
        .csv, .tsv => {
            // For CSV, just output chi1
            const delim: u8 = if (args.format == .csv) ',' else '\t';
            try w.writeAll("frame");
            for (0..parsed.topology.residues.len) |ri| {
                try w.print("{c}res_{d}", .{ delim, ri });
            }
            try w.writeByte('\n');
            for (frames, 0..) |frame, fi| {
                const vals = try prot_dih.computeChi(allocator, parsed.topology, frame, 1);
                defer allocator.free(vals);
                try w.print("{d}", .{fi});
                for (vals) |v| {
                    try w.writeByte(delim);
                    if (v) |angle| {
                        try w.print("{d:.6}", .{angle});
                    }
                }
                try w.writeByte('\n');
                analysis_node.completeOne();
            }
        },
    }
    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: summary
// ============================================================================

pub fn runSummary(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const topo = &parsed.topology;
    const frame = &parsed.frame;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);

    // -- File + basic counts ------------------------------------------------
    try w.print("File:     {s}\n", .{top_path});
    try w.print("Atoms:    {d}  Residues: {d}  Chains: {d}  Bonds: {d}\n", .{
        topo.atoms.len, topo.residues.len, topo.chains.len, topo.bonds.len,
    });

    // -- Box vectors --------------------------------------------------------
    if (frame.box_vectors) |box| {
        const is_ortho = @abs(box[0][1]) < 0.001 and @abs(box[0][2]) < 0.001 and
            @abs(box[1][0]) < 0.001 and @abs(box[1][2]) < 0.001 and
            @abs(box[2][0]) < 0.001 and @abs(box[2][1]) < 0.001;
        if (is_ortho) {
            try w.print("Box:      {d:.3} x {d:.3} x {d:.3} A\n", .{ box[0][0], box[1][1], box[2][2] });
        } else {
            try w.print("Box:      [{d:.3}, {d:.3}, {d:.3}]\n", .{ box[0][0], box[0][1], box[0][2] });
            try w.print("          [{d:.3}, {d:.3}, {d:.3}]\n", .{ box[1][0], box[1][1], box[1][2] });
            try w.print("          [{d:.3}, {d:.3}, {d:.3}]\n", .{ box[2][0], box[2][1], box[2][2] });
        }
    } else {
        try w.print("Box:      (none)\n", .{});
    }

    // -- Time ---------------------------------------------------------------
    if (frame.time != 0.0) {
        try w.print("Time:     {d:.3} ps\n", .{frame.time});
    }

    // -- Chains + residues --------------------------------------------------
    if (topo.chains.len > 0) {
        try w.print("\nChains:\n", .{});
        for (topo.chains, 0..) |chain, ci| {
            const rr = chain.residue_range;
            if (rr.len == 0) continue;
            const first_res = topo.residues[rr.start];
            const last_res = topo.residues[rr.start + rr.len - 1];
            var chain_atoms: u32 = 0;
            for (topo.residues[rr.start .. rr.start + rr.len]) |res| {
                chain_atoms += res.atom_range.len;
            }
            const chain_name = chain.name.slice();
            const name_str = if (chain_name.len > 0) chain_name else "(unnamed)";
            try w.print("  [{d}] {s} : {d} residues ({s} {d} .. {s} {d}), {d} atoms\n", .{
                ci,
                name_str,
                rr.len,
                first_res.name.slice(),
                first_res.resid,
                last_res.name.slice(),
                last_res.resid,
                chain_atoms,
            });
        }
    }

    // -- Element composition ------------------------------------------------
    const Element = ztraj.element.Element;
    const num_elements = @typeInfo(Element).@"enum".fields.len;
    var elem_counts = [_]u32{0} ** num_elements;
    for (topo.atoms) |atom| {
        elem_counts[@intFromEnum(atom.element)] += 1;
    }
    try w.print("\nElements: ", .{});
    var first_elem = true;
    const common_elems = [_]Element{ .H, .C, .N, .O, .S, .P, .Fe, .Zn, .Ca, .Mg, .Na, .Cl, .K };
    for (common_elems) |e| {
        const idx = @intFromEnum(e);
        if (elem_counts[idx] > 0) {
            if (!first_elem) try w.print("  ", .{});
            try w.print("{s}:{d}", .{ @tagName(e), elem_counts[idx] });
            elem_counts[idx] = 0;
            first_elem = false;
        }
    }
    for (elem_counts, 0..) |count, idx| {
        if (count > 0) {
            const e: Element = @enumFromInt(idx);
            if (!first_elem) try w.print("  ", .{});
            try w.print("{s}:{d}", .{ @tagName(e), count });
            first_elem = false;
        }
    }
    if (first_elem) try w.print("(none)", .{});
    try w.print("\n", .{});

    // -- Trajectory info (if separate trajectory file) ----------------------
    const has_traj = args.top_path != null;
    if (has_traj) {
        const traj_info = try loader.loadTrajectoryInfo(allocator, args.traj_path, topo.atoms.len);
        try w.print("\nTrajectory: {s}\n", .{args.traj_path});
        try w.print("Frames:     {d}\n", .{traj_info.n_frames});
        if (traj_info.first_time != null and traj_info.last_time != null) {
            const first_time = traj_info.first_time.?;
            const last_time = traj_info.last_time.?;
            if (first_time != 0.0 or last_time != 0.0) {
                try w.print("Time:       {d:.3} .. {d:.3} ps\n", .{ first_time, last_time });
            }
        }
    }

    try flushOutput(buf.items, args.output_path);
}

// ============================================================================
// Subcommand: convert
// ============================================================================

pub fn runConvert(allocator: std.mem.Allocator, args: Args) !void {
    const output_path = args.output_path orelse {
        std.debug.print("error: convert requires --output <file>\n", .{});
        std.process.exit(1);
    };

    const is_traj_output = loader.isXtc(output_path) or loader.isTrr(output_path);
    const is_traj_input = loader.isXtc(args.traj_path) or
        loader.isTrr(args.traj_path) or loader.isDcd(args.traj_path);

    // Trajectory conversion: read all frames, write to trajectory format
    if (is_traj_output or is_traj_input) {
        // Need topology for atom count
        const top_path = args.top_path orelse args.traj_path;
        var parsed = try loader.loadTopology(allocator, top_path);
        defer parsed.deinit();
        const n_atoms = parsed.topology.atoms.len;

        // Load all frames from input
        const frames = try loader.loadAllFrames(allocator, args.traj_path, n_atoms, std.Progress.Node.none);
        defer {
            for (frames) |*f| @constCast(f).deinit();
            allocator.free(frames);
        }

        if (frames.len == 0) {
            std.debug.print("error: no frames found in '{s}'\n", .{args.traj_path});
            std.process.exit(1);
        }

        if (loader.isXtc(output_path)) {
            var writer = try ztraj.io.xtc.XtcWriter.open(allocator, output_path, n_atoms);
            defer writer.deinit();
            for (frames) |frame| try writer.writeFrame(frame);
            try writer.close();
        } else if (loader.isTrr(output_path)) {
            var writer = try ztraj.io.trr.TrrWriter.open(allocator, output_path, n_atoms);
            defer writer.deinit();
            for (frames) |frame| try writer.writeFrame(frame);
            try writer.close();
        } else if (loader.isPdb(output_path)) {
            // Trajectory → single-structure: write first frame only
            var buf = std.ArrayList(u8){};
            defer buf.deinit(allocator);
            if (frames.len > 0) {
                try ztraj.io.pdb.write(buf.writer(allocator), parsed.topology, frames[0]);
            }
            try flushOutput(buf.items, output_path);
        } else if (loader.isGro(output_path)) {
            var buf = std.ArrayList(u8){};
            defer buf.deinit(allocator);
            if (frames.len > 0) {
                try ztraj.io.gro.write(buf.writer(allocator), parsed.topology, frames[0]);
            }
            try flushOutput(buf.items, output_path);
        } else {
            std.debug.print(
                "error: unsupported output format for '{s}' (supported: .pdb, .gro, .xtc, .trr)\n",
                .{output_path},
            );
            std.process.exit(1);
        }

        std.debug.print("Converted {s} -> {s} ({d} atoms, {d} frames)\n", .{
            args.traj_path, output_path, n_atoms, frames.len,
        });
    } else {
        // Structure-only conversion (PDB/GRO/mmCIF → PDB/GRO)
        var parsed = try loader.loadTopology(allocator, args.traj_path);
        defer parsed.deinit();

        var buf = std.ArrayList(u8){};
        defer buf.deinit(allocator);
        const w = buf.writer(allocator);

        if (loader.isPdb(output_path)) {
            try ztraj.io.pdb.write(w, parsed.topology, parsed.frame);
        } else if (loader.isGro(output_path)) {
            try ztraj.io.gro.write(w, parsed.topology, parsed.frame);
        } else {
            std.debug.print(
                "error: unsupported output format for '{s}' (supported: .pdb, .gro, .xtc, .trr)\n",
                .{output_path},
            );
            std.process.exit(1);
        }

        try flushOutput(buf.items, output_path);

        std.debug.print("Converted {s} -> {s} ({d} atoms, {d} residues)\n", .{
            args.traj_path, output_path, parsed.topology.atoms.len, parsed.topology.residues.len,
        });
    }
}
