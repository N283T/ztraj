//! ztraj CLI entry point.
//!
//! Pattern: ztraj <command> <trajectory> --top <topology> [options]
//! Output to stdout by default; file output via --output.

const std = @import("std");
const build_options = @import("build_options");

const cli_args = @import("cli/args.zig");
const runners = @import("cli/runners.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const raw_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, raw_args);

    if (raw_args.len < 2) {
        cli_args.printUsage(raw_args[0]);
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
        .rmsd => runners.runRmsd(allocator, args),
        .rmsf => runners.runRmsf(allocator, args),
        .distances => runners.runDistances(allocator, args),
        .angles => runners.runAngles(allocator, args),
        .dihedrals => runners.runDihedrals(allocator, args),
        .rg => runners.runRg(allocator, args),
        .center => runners.runCenter(allocator, args),
        .inertia => runners.runInertia(allocator, args),
        .hbonds => runners.runHbonds(allocator, args),
        .contacts => runners.runContacts(allocator, args),
        .rdf => runners.runRdf(allocator, args),
        .sasa => runners.runSasa(allocator, args),
        .all => runners.runAll(allocator, args),
        .dssp => runners.runDssp(allocator, args),
        .phi => runners.runPhi(allocator, args),
        .psi => runners.runPsi(allocator, args),
        .omega => runners.runOmega(allocator, args),
        .chi => runners.runChi(allocator, args),
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
