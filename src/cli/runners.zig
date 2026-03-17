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
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runRmsf(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runDistances(allocator: std.mem.Allocator, args: Args) !void {
    const spec = args.pairs_spec orelse {
        std.debug.print("error: --pairs required for distances subcommand\n", .{});
        std.process.exit(1);
    };
    const pairs = try parsers.parsePairs(allocator, spec);
    defer allocator.free(pairs);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    parsers.validateIndices(2, pairs, @intCast(parsed.topology.atoms.len));

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runAngles(allocator: std.mem.Allocator, args: Args) !void {
    const spec = args.triplets_spec orelse {
        std.debug.print("error: --triplets required for angles subcommand\n", .{});
        std.process.exit(1);
    };
    const triplets = try parsers.parseTriplets(allocator, spec);
    defer allocator.free(triplets);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    parsers.validateIndices(3, triplets, @intCast(parsed.topology.atoms.len));

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runDihedrals(allocator: std.mem.Allocator, args: Args) !void {
    const spec = args.quartets_spec orelse {
        std.debug.print("error: --quartets required for dihedrals subcommand\n", .{});
        std.process.exit(1);
    };
    const quartets = try parsers.parseQuartets(allocator, spec);
    defer allocator.free(quartets);

    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    parsers.validateIndices(4, quartets, @intCast(parsed.topology.atoms.len));

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runRg(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runCenter(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runInertia(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const atom_indices = try parsers.resolveSelection(allocator, parsed.topology, args.select_str);
    defer if (atom_indices) |ai| allocator.free(ai);

    const masses = try parsed.topology.masses(allocator);
    defer allocator.free(masses);

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runHbonds(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runContacts(allocator: std.mem.Allocator, args: Args) !void {
    const top_path = args.top_path orelse args.traj_path;
    var parsed = try loader.loadTopology(allocator, top_path);
    defer parsed.deinit();

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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

pub fn runRdf(allocator: std.mem.Allocator, args: Args) !void {
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

    const frames = try loader.loadAllFrames(allocator, args.traj_path, parsed.topology.atoms.len);
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
