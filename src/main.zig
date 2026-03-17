//! ztraj CLI entry point.
//!
//! Pattern: ztraj <command> <trajectory> --top <topology> [options]
//! Output to stdout by default; file output via --output.

const std = @import("std");
const build_options = @import("build_options");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    if (args.len > 1) {
        const arg = args[1];

        if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-V")) {
            try stdout.print("ztraj {s}\n", .{build_options.version});
            return;
        }

        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage(args[0]);
            return;
        }

        if (std.mem.eql(u8, arg, "rmsd")) {
            const stderr = std.io.getStdErr().writer();
            try stderr.print("error: 'rmsd' subcommand not yet implemented\n", .{});
            std.process.exit(1);
        }

        if (std.mem.eql(u8, arg, "distances")) {
            const stderr = std.io.getStdErr().writer();
            try stderr.print("error: 'distances' subcommand not yet implemented\n", .{});
            std.process.exit(1);
        }

        if (std.mem.eql(u8, arg, "rg")) {
            const stderr = std.io.getStdErr().writer();
            try stderr.print("error: 'rg' subcommand not yet implemented\n", .{});
            std.process.exit(1);
        }

        if (std.mem.eql(u8, arg, "rdf")) {
            const stderr = std.io.getStdErr().writer();
            try stderr.print("error: 'rdf' subcommand not yet implemented\n", .{});
            std.process.exit(1);
        }

        if (std.mem.eql(u8, arg, "hbonds")) {
            const stderr = std.io.getStdErr().writer();
            try stderr.print("error: 'hbonds' subcommand not yet implemented\n", .{});
            std.process.exit(1);
        }

        const stderr = std.io.getStdErr().writer();
        try stderr.print("error: unknown subcommand '{s}'\n\n", .{arg});
        printUsage(args[0]);
        std.process.exit(1);
    }

    printUsage(args[0]);
}

fn printUsage(program_name: []const u8) void {
    const stderr = std.io.getStdErr().writer();
    stderr.print(
        \\Usage: {s} <command> [options]
        \\
        \\Commands:
        \\  rmsd        Compute RMSD between frames (QCP algorithm)
        \\  distances   Pairwise atom distances over trajectory
        \\  rg          Radius of gyration over trajectory
        \\  rdf         Radial distribution function g(r)
        \\  hbonds      Hydrogen bond detection
        \\
        \\Options:
        \\  --top <file>      Topology file (PDB or mmCIF)
        \\  --ref <frame>     Reference frame index for RMSD (default: 0)
        \\  --select <expr>   Atom selection (e.g. "backbone", "name CA")
        \\  --format <fmt>    Output format: json (default), csv, tsv
        \\  --output <file>   Write output to file (default: stdout)
        \\  -V, --version     Print version and exit
        \\  -h, --help        Print this help and exit
        \\
        \\Examples:
        \\  {s} rmsd trajectory.xtc --top structure.pdb --ref 0
        \\  {s} rg trajectory.xtc --top structure.pdb --select backbone
        \\  {s} distances trajectory.xtc --top structure.pdb --pairs "1-10,2-20"
        \\  {s} rdf trajectory.xtc --top structure.pdb --sel1 "O" --sel2 "H"
        \\  {s} hbonds trajectory.xtc --top structure.pdb --format csv
        \\
    , .{
        program_name,
        program_name,
        program_name,
        program_name,
        program_name,
        program_name,
    }) catch {};
}
