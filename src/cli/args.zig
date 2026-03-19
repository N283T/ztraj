//! CLI argument types and parser.

const std = @import("std");
const ztraj = @import("ztraj");
const output = ztraj.output;

// ============================================================================
// Subcommand enum
// ============================================================================

pub const Subcommand = enum {
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
    sasa,
    all,
    dssp,
    phi,
    psi,
    omega,
    chi,
};

// ============================================================================
// Parsed CLI arguments
// ============================================================================

pub const Args = struct {
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

pub fn printUsage(prog: []const u8) void {
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
        \\  sasa        Solvent accessible surface area (Shrake-Rupley)
        \\  all         Combined analysis (RMSD, RMSF, Rg, SASA, COM, hbonds, contacts)
        \\  dssp        Secondary structure assignment (DSSP algorithm)
        \\  phi         Backbone phi dihedral angles per residue
        \\  psi         Backbone psi dihedral angles per residue
        \\  omega       Backbone omega dihedral angles per residue
        \\  chi         Side-chain chi dihedral angles per residue
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

pub const ParseArgsError = error{
    MissingSubcommand,
    MissingTrajectory,
    UnknownSubcommand,
    UnknownFlag,
    MissingValue,
    InvalidFormat,
    InvalidNumber,
};

pub fn parseArgs(raw: []const []const u8) ParseArgsError!Args {
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
        if (std.mem.eql(u8, sub_str, "sasa")) break :blk .sasa;
        if (std.mem.eql(u8, sub_str, "all")) break :blk .all;
        if (std.mem.eql(u8, sub_str, "dssp")) break :blk .dssp;
        if (std.mem.eql(u8, sub_str, "phi")) break :blk .phi;
        if (std.mem.eql(u8, sub_str, "psi")) break :blk .psi;
        if (std.mem.eql(u8, sub_str, "omega")) break :blk .omega;
        if (std.mem.eql(u8, sub_str, "chi")) break :blk .chi;
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
