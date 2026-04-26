//! ztraj CLI entry point.
//!
//! Pattern: ztraj <command> <trajectory> --top <topology> [options]
//! Output to stdout by default; file output via --output.

const std = @import("std");
const build_options = @import("build_options");

const cli_args = @import("cli/args.zig");
const runners = @import("cli/runners.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args_z = try init.minimal.args.toSlice(init.arena.allocator());
    const raw_args: []const []const u8 = @ptrCast(args_z);

    if (raw_args.len < 2) {
        cli_args.printUsage(raw_args[0]);
        std.process.exit(1);
    }

    const first = raw_args[1];

    if (std.mem.eql(u8, first, "--version") or std.mem.eql(u8, first, "-V")) {
        const stdout = std.Io.File.stdout();
        var buf: [64]u8 = undefined;
        const line = try std.fmt.bufPrint(&buf, "ztraj {s}\n", .{build_options.version});
        try stdout.writeStreamingAll(io, line);
        return;
    }
    if (std.mem.eql(u8, first, "--help") or std.mem.eql(u8, first, "-h")) {
        cli_args.printUsage(raw_args[0]);
        return;
    }

    const args = cli_args.parseArgs(raw_args) catch |err| {
        switch (err) {
            cli_args.ParseArgsError.MissingSubcommand => std.debug.print("error: missing subcommand\n", .{}),
            cli_args.ParseArgsError.MissingTrajectory => std.debug.print("error: missing trajectory/structure file argument\n", .{}),
            cli_args.ParseArgsError.UnknownSubcommand => std.debug.print("error: unknown subcommand '{s}'\n", .{first}),
            cli_args.ParseArgsError.UnknownFlag => std.debug.print("error: unknown flag in arguments\n", .{}),
            cli_args.ParseArgsError.MissingValue => std.debug.print("error: flag requires a value\n", .{}),
            cli_args.ParseArgsError.InvalidFormat => std.debug.print("error: invalid --format (use json, csv, or tsv)\n", .{}),
            cli_args.ParseArgsError.InvalidNumber => std.debug.print("error: invalid numeric argument\n", .{}),
        }
        cli_args.printUsage(raw_args[0]);
        std.process.exit(1);
    };

    const result = switch (args.subcommand) {
        .rmsd => runners.runRmsd(io, allocator, args),
        .rmsf => runners.runRmsf(io, allocator, args),
        .distances => runners.runDistances(io, allocator, args),
        .angles => runners.runAngles(io, allocator, args),
        .dihedrals => runners.runDihedrals(io, allocator, args),
        .rg => runners.runRg(io, allocator, args),
        .center => runners.runCenter(io, allocator, args),
        .inertia => runners.runInertia(io, allocator, args),
        .hbonds => runners.runHbonds(io, allocator, args),
        .contacts => runners.runContacts(io, allocator, args),
        .rdf => runners.runRdf(io, allocator, args),
        .sasa => runners.runSasa(io, allocator, args),
        .all => runners.runAll(io, allocator, args),
        .dssp => runners.runDssp(io, allocator, args),
        .phi => runners.runPhi(io, allocator, args),
        .psi => runners.runPsi(io, allocator, args),
        .omega => runners.runOmega(io, allocator, args),
        .chi => runners.runChi(io, allocator, args),
        .summary => runners.runSummary(io, allocator, args),
        .convert => runners.runConvert(io, allocator, args),
    };

    result catch |err| {
        const top_path = args.top_path orelse "(none)";
        std.debug.print(
            "error: {s} failed for trajectory '{s}' (topology: '{s}'): {s}\n",
            .{ @tagName(args.subcommand), args.traj_path, top_path, @errorName(err) },
        );
        std.process.exit(1);
    };
}
