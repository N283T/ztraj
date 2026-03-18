//! PDB file parser.
//!
//! Parses ATOM/HETATM records from PDB format files and returns a ParseResult
//! containing a Topology (atoms, residues, chains, bonds) and a Frame
//! (SOA coordinate arrays).
//!
//! ## PDB Record Format (Fixed Width, 1-indexed columns)
//!
//! ATOM/HETATM records:
//! - 1-6:   Record name (ATOM  / HETATM)
//! - 7-11:  Atom serial number
//! - 13-16: Atom name
//! - 17:    Alternate location indicator
//! - 18-20: Residue name
//! - 22:    Chain identifier
//! - 23-26: Residue sequence number
//! - 27:    Insertion code
//! - 31-38: X coordinate (Angstroms)
//! - 39-46: Y coordinate (Angstroms)
//! - 47-54: Z coordinate (Angstroms)
//! - 55-60: Occupancy
//! - 61-66: Temperature factor
//! - 77-78: Element symbol
//!
//! CONECT records:
//! - 7-11:  Atom serial 1
//! - 12-16: Atom serial 2
//! - 17-21: Atom serial 3 (optional)
//! - 22-26: Atom serial 4 (optional)
//! - 27-31: Atom serial 5 (optional)

const std = @import("std");
const types = @import("../types.zig");
const elem = @import("../element.zig");

/// Error types for PDB parsing
pub const ParseError = error{
    /// No ATOM/HETATM records found in the file
    NoAtomsFound,
};

/// Intermediate flat atom record collected during parsing
const RawAtom = struct {
    name: types.FixedString(4),
    res_name: types.FixedString(5),
    chain_id: types.FixedString(4),
    resid: i32,
    ins_code: u8,
    x: f32,
    y: f32,
    z: f32,
    element: elem.Element,
    serial: u32,
    is_hetatm: bool,
};

/// Parse a PDB coordinate field (8-character fixed-width).
/// Returns null when the field is empty or unparseable.
fn parseCoord(field: []const u8) ?f32 {
    var start: usize = 0;
    while (start < field.len and field[start] == ' ') : (start += 1) {}
    if (start == field.len) return null;

    var negative = false;
    if (field[start] == '-') {
        negative = true;
        start += 1;
    } else if (field[start] == '+') {
        start += 1;
    }

    var int_part: i64 = 0;
    var has_digits = false;
    while (start < field.len and field[start] >= '0' and field[start] <= '9') : (start += 1) {
        has_digits = true;
        int_part = int_part * 10 + @as(i64, field[start] - '0');
    }

    var frac: f64 = 0;
    if (start < field.len and field[start] == '.') {
        start += 1;
        var mult: f64 = 0.1;
        while (start < field.len and field[start] >= '0' and field[start] <= '9') : (start += 1) {
            has_digits = true;
            frac += @as(f64, @floatFromInt(field[start] - '0')) * mult;
            mult *= 0.1;
        }
    }

    if (!has_digits) return null;

    const result = @as(f64, @floatFromInt(int_part)) + frac;
    return @floatCast(if (negative) -result else result);
}

/// Infer element from PDB atom name field (columns 13-16, 0-indexed 12-15).
/// Standard convention: element is right-justified in columns 13-14.
fn inferElement(atom_name: []const u8) elem.Element {
    if (atom_name.len < 2) return .X;
    const first = atom_name[0];
    const second = atom_name[1];
    if (first == ' ' or (first >= '0' and first <= '9')) {
        return elem.fromSymbol(&[_]u8{second});
    }
    return elem.fromSymbol(&[_]u8{first});
}

/// Parse a single ATOM or HETATM line into a RawAtom.
/// Returns null if the line is too short or coordinates are missing.
fn parseAtomLine(line: []const u8, is_hetatm: bool) ?RawAtom {
    if (line.len < 54) return null;

    // Coordinates (0-indexed: 30-38, 38-46, 46-54)
    const x = parseCoord(line[30..38]) orelse return null;
    const y = parseCoord(line[38..46]) orelse return null;
    const z = parseCoord(line[46..54]) orelse return null;

    // Atom name (cols 13-16, 0-indexed 12-16)
    const name_raw = if (line.len >= 16) line[12..16] else "    ";
    const name_trimmed = std.mem.trim(u8, name_raw, " ");

    // Alternate location indicator (col 17, 0-indexed 16)
    const alt_loc: u8 = if (line.len > 16) line[16] else ' ';

    // Residue name (cols 18-20, 0-indexed 17-20)
    const res_raw = if (line.len >= 20) line[17..20] else "   ";
    const res_trimmed = std.mem.trim(u8, res_raw, " ");

    // Chain ID (col 22, 0-indexed 21)
    const chain_char: u8 = if (line.len > 21) line[21] else ' ';
    const chain_str: []const u8 = if (chain_char != ' ') &[_]u8{chain_char} else "";

    // Residue sequence number (cols 23-26, 0-indexed 22-26)
    const resnum_str = if (line.len >= 26) std.mem.trim(u8, line[22..26], " ") else "";
    const resid = std.fmt.parseInt(i32, resnum_str, 10) catch 0;

    // Insertion code (col 27, 0-indexed 26)
    const ins_code: u8 = if (line.len > 26) line[26] else ' ';

    // Element symbol (cols 77-78, 0-indexed 76-78)
    const element: elem.Element = if (line.len >= 78) blk: {
        const sym = std.mem.trim(u8, line[76..78], " ");
        break :blk if (sym.len > 0) elem.fromSymbol(sym) else inferElement(name_raw);
    } else inferElement(name_raw);

    // Atom serial (cols 7-11, 0-indexed 6-11)
    const serial_str = if (line.len >= 11) std.mem.trim(u8, line[6..11], " ") else "0";
    const serial = std.fmt.parseInt(u32, serial_str, 10) catch 0;

    // Skip alternate locations — keep only ' ' or first non-space alt loc.
    // We track the first seen alt_loc in the caller; here we just expose it.
    _ = alt_loc; // handled in the calling loop

    return RawAtom{
        .name = types.FixedString(4).fromSlice(name_trimmed),
        .res_name = types.FixedString(5).fromSlice(res_trimmed),
        .chain_id = types.FixedString(4).fromSlice(chain_str),
        .resid = resid,
        .ins_code = ins_code,
        .x = x,
        .y = y,
        .z = z,
        .element = element,
        .serial = serial,
        .is_hetatm = is_hetatm,
    };
}

/// Parse a PDB-format string and return a ParseResult.
///
/// - Parses ATOM and HETATM records (both are included).
/// - Stops after the first MODEL's ENDMDL (multi-model files: only first model).
/// - Parses CONECT records for bonds.
/// - Builds Topology (atoms, residues, chains, bonds) and Frame (x/y/z SOA).
pub fn parse(allocator: std.mem.Allocator, data: []const u8) !types.ParseResult {
    // -------------------------------------------------------------------------
    // Pass 1: collect raw atoms and CONECT entries
    // -------------------------------------------------------------------------
    var raw_atoms = std.ArrayListUnmanaged(RawAtom){};
    defer raw_atoms.deinit(allocator);

    // CONECT: list of (serial_i, serial_j) pairs
    const ConectPair = struct { i: u32, j: u32 };
    var conect_pairs = std.ArrayListUnmanaged(ConectPair){};
    defer conect_pairs.deinit(allocator);

    var first_alt_loc: u8 = 0; // 0 = not yet set
    var in_first_model = true;
    var seen_model = false;

    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "MODEL ") or std.mem.startsWith(u8, line, "MODEL\t")) {
            if (seen_model) {
                // Second model — stop collecting
                break;
            }
            seen_model = true;
            in_first_model = true;
            continue;
        }
        if (std.mem.startsWith(u8, line, "ENDMDL")) {
            if (seen_model) {
                in_first_model = false;
                break;
            }
            continue;
        }
        if (!in_first_model) continue;

        const is_atom = std.mem.startsWith(u8, line, "ATOM  ");
        const is_hetatm = std.mem.startsWith(u8, line, "HETATM");

        if (is_atom or is_hetatm) {
            // Alternate location check
            const alt_loc: u8 = if (line.len > 16) line[16] else ' ';
            if (alt_loc != ' ') {
                if (first_alt_loc == 0) {
                    first_alt_loc = alt_loc;
                } else if (alt_loc != first_alt_loc) {
                    continue; // skip non-first alt locs
                }
            }

            if (parseAtomLine(line, is_hetatm)) |raw| {
                try raw_atoms.append(allocator, raw);
            }
            continue;
        }

        if (std.mem.startsWith(u8, line, "CONECT")) {
            // CONECT record: atom serials in columns 7-11, 12-16, 17-21, 22-26, 27-31
            if (line.len < 11) continue;
            const s1_str = std.mem.trim(u8, line[6..@min(11, line.len)], " ");
            const s1 = std.fmt.parseInt(u32, s1_str, 10) catch continue;
            // Up to 4 bonded partners
            const partner_ranges = [_][2]usize{
                .{ 11, 16 },
                .{ 16, 21 },
                .{ 21, 26 },
                .{ 26, 31 },
            };
            for (partner_ranges) |rng| {
                if (line.len <= rng[0]) break;
                const end = @min(rng[1], line.len);
                const s2_str = std.mem.trim(u8, line[rng[0]..end], " ");
                if (s2_str.len == 0) continue;
                const s2 = std.fmt.parseInt(u32, s2_str, 10) catch continue;
                if (s2 == 0) continue;
                // Only add each pair once (s1 < s2)
                if (s1 < s2) {
                    try conect_pairs.append(allocator, .{ .i = s1, .j = s2 });
                }
            }
        }
    }

    if (raw_atoms.items.len == 0) return ParseError.NoAtomsFound;

    // -------------------------------------------------------------------------
    // Pass 2: build residue and chain boundaries
    // -------------------------------------------------------------------------

    // Identify unique (chain_id, resid, ins_code) triples in order.
    const ResKey = struct {
        chain_id: types.FixedString(4),
        resid: i32,
        ins_code: u8,
    };

    // Count residues and chains by scanning raw_atoms
    var n_residues: u32 = 0;
    var n_chains: u32 = 0;
    var prev_res_key: ?ResKey = null;
    var prev_chain: ?types.FixedString(4) = null;

    for (raw_atoms.items) |ra| {
        const key = ResKey{ .chain_id = ra.chain_id, .resid = ra.resid, .ins_code = ra.ins_code };
        const chain_changed = if (prev_chain) |pc| !pc.eql(&ra.chain_id) else true;
        const res_changed = if (prev_res_key) |pk|
            pk.resid != key.resid or pk.ins_code != key.ins_code or !pk.chain_id.eql(&key.chain_id)
        else
            true;

        if (chain_changed) {
            n_chains += 1;
            prev_chain = ra.chain_id;
        }
        if (res_changed) {
            n_residues += 1;
            prev_res_key = key;
        }
    }

    const n_atoms: u32 = @intCast(raw_atoms.items.len);

    // Build serial -> atom_index map for CONECT resolution
    var serial_to_idx = std.AutoHashMapUnmanaged(u32, u32){};
    defer serial_to_idx.deinit(allocator);
    try serial_to_idx.ensureTotalCapacity(allocator, n_atoms);
    for (raw_atoms.items, 0..) |ra, i| {
        serial_to_idx.putAssumeCapacity(ra.serial, @intCast(i));
    }

    // Deduplicate CONECT bonds
    const BondKey = struct { i: u32, j: u32 };
    var bond_set = std.AutoHashMapUnmanaged(BondKey, void){};
    defer bond_set.deinit(allocator);
    for (conect_pairs.items) |cp| {
        const idx_i = serial_to_idx.get(cp.i) orelse continue;
        const idx_j = serial_to_idx.get(cp.j) orelse continue;
        const bk = if (idx_i <= idx_j)
            BondKey{ .i = idx_i, .j = idx_j }
        else
            BondKey{ .i = idx_j, .j = idx_i };
        try bond_set.put(allocator, bk, {});
    }
    const n_bonds: u32 = @intCast(bond_set.count());

    // -------------------------------------------------------------------------
    // Allocate Topology and Frame
    // -------------------------------------------------------------------------
    var topology = try types.Topology.init(allocator, .{
        .n_atoms = n_atoms,
        .n_residues = n_residues,
        .n_chains = n_chains,
        .n_bonds = n_bonds,
    });
    errdefer topology.deinit();

    var frame = try types.Frame.init(allocator, n_atoms);
    errdefer frame.deinit();

    // -------------------------------------------------------------------------
    // Pass 3: fill Topology and Frame
    // -------------------------------------------------------------------------
    var atom_idx: u32 = 0;
    var res_idx: u32 = 0;
    var chain_idx: u32 = 0;

    prev_res_key = null;
    prev_chain = null;

    // Track starts for range building
    var chain_res_start: u32 = 0;
    var res_atom_start: u32 = 0;

    for (raw_atoms.items) |ra| {
        const key = ResKey{ .chain_id = ra.chain_id, .resid = ra.resid, .ins_code = ra.ins_code };
        const chain_changed = if (prev_chain) |pc| !pc.eql(&ra.chain_id) else true;
        const res_changed = if (prev_res_key) |pk|
            pk.resid != key.resid or pk.ins_code != key.ins_code or !pk.chain_id.eql(&key.chain_id)
        else
            true;

        if (chain_changed) {
            // Close previous chain
            if (chain_idx > 0) {
                topology.chains[chain_idx - 1].residue_range = .{
                    .start = chain_res_start,
                    .len = res_idx - chain_res_start,
                };
            }
            // Close previous residue if needed (already handled by res_changed below)
            chain_res_start = res_idx;
            topology.chains[chain_idx] = types.Chain{
                .name = ra.chain_id,
                .residue_range = .{ .start = res_idx, .len = 0 }, // finalized later
            };
            chain_idx += 1;
            prev_chain = ra.chain_id;
        }

        if (res_changed) {
            // Close previous residue
            if (res_idx > 0 and atom_idx > res_atom_start) {
                topology.residues[res_idx - 1].atom_range = .{
                    .start = res_atom_start,
                    .len = atom_idx - res_atom_start,
                };
            }
            res_atom_start = atom_idx;
            // chain_idx was already incremented above; current chain is chain_idx-1
            topology.residues[res_idx] = types.Residue{
                .name = ra.res_name,
                .chain_index = chain_idx - 1,
                .atom_range = .{ .start = atom_idx, .len = 0 }, // finalized later
                .resid = ra.resid,
            };
            res_idx += 1;
            prev_res_key = key;
        }

        // Fill atom
        topology.atoms[atom_idx] = types.Atom{
            .name = ra.name,
            .element = ra.element,
            .residue_index = res_idx - 1,
        };

        // Fill frame coordinates
        frame.x[atom_idx] = ra.x;
        frame.y[atom_idx] = ra.y;
        frame.z[atom_idx] = ra.z;

        atom_idx += 1;
    }

    // Close last residue and last chain
    if (res_idx > 0) {
        topology.residues[res_idx - 1].atom_range = .{
            .start = res_atom_start,
            .len = atom_idx - res_atom_start,
        };
    }
    if (chain_idx > 0) {
        topology.chains[chain_idx - 1].residue_range = .{
            .start = chain_res_start,
            .len = res_idx - chain_res_start,
        };
    }

    // Fill bonds
    var bond_iter = bond_set.keyIterator();
    var bond_idx: u32 = 0;
    while (bond_iter.next()) |bk| {
        topology.bonds[bond_idx] = types.Bond{ .atom_i = bk.i, .atom_j = bk.j };
        bond_idx += 1;
    }

    try topology.validate();
    return types.ParseResult{ .topology = topology, .frame = frame };
}

// ============================================================================
// Tests
// ============================================================================

test "parse inline PDB" {
    const pdb_content =
        \\ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 11.68           N
        \\ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  9.13           C
        \\ATOM      3  C   ALA A   1      10.480   5.927  -4.153  1.00  7.65           C
        \\ATOM      4  N   GLY A   2       9.207   5.600  -4.678  1.00  6.01           N
        \\ATOM      5  CA  GLY A   2       8.100   5.400  -3.720  1.00  5.50           C
        \\HETATM  100  O   HOH B   1       5.000   5.000   5.000  1.00 20.00           O
        \\END
    ;

    var result = try parse(std.testing.allocator, pdb_content);
    defer result.deinit();

    const topo = &result.topology;
    const frame = &result.frame;

    // Atoms: 6 total (ATOM + HETATM)
    try std.testing.expectEqual(@as(usize, 6), topo.atoms.len);
    try std.testing.expectEqual(@as(usize, 6), frame.nAtoms());

    // Residues: ALA A1, GLY A2, HOH B1
    try std.testing.expectEqual(@as(usize, 3), topo.residues.len);
    try std.testing.expect(topo.residues[0].name.eqlSlice("ALA"));
    try std.testing.expect(topo.residues[1].name.eqlSlice("GLY"));
    try std.testing.expect(topo.residues[2].name.eqlSlice("HOH"));

    // Chains: A, B
    try std.testing.expectEqual(@as(usize, 2), topo.chains.len);
    try std.testing.expect(topo.chains[0].name.eqlSlice("A"));
    try std.testing.expect(topo.chains[1].name.eqlSlice("B"));

    // Coordinates of first atom
    try std.testing.expectApproxEqAbs(@as(f32, 11.104), frame.x[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.134), frame.y[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -6.504), frame.z[0], 0.001);

    // Element of atom 0 (N)
    try std.testing.expectEqual(elem.Element.N, topo.atoms[0].element);
    // Element of atom 1 (C)
    try std.testing.expectEqual(elem.Element.C, topo.atoms[1].element);
}

test "parse single chain residue ranges" {
    const pdb_content =
        \\ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
        \\ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
        \\ATOM      3  N   GLY A   2       9.207   5.600  -4.678  1.00  0.00           N
        \\END
    ;

    var result = try parse(std.testing.allocator, pdb_content);
    defer result.deinit();

    const topo = &result.topology;
    try std.testing.expectEqual(@as(usize, 3), topo.atoms.len);
    try std.testing.expectEqual(@as(usize, 2), topo.residues.len);
    try std.testing.expectEqual(@as(usize, 1), topo.chains.len);

    // ALA: atoms 0-1
    try std.testing.expectEqual(@as(u32, 0), topo.residues[0].atom_range.start);
    try std.testing.expectEqual(@as(u32, 2), topo.residues[0].atom_range.len);
    // GLY: atom 2
    try std.testing.expectEqual(@as(u32, 2), topo.residues[1].atom_range.start);
    try std.testing.expectEqual(@as(u32, 1), topo.residues[1].atom_range.len);
    // Chain A: residues 0-1
    try std.testing.expectEqual(@as(u32, 0), topo.chains[0].residue_range.start);
    try std.testing.expectEqual(@as(u32, 2), topo.chains[0].residue_range.len);
}

test "parse CONECT bonds" {
    const pdb_content =
        \\ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
        \\ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
        \\ATOM      3  C   ALA A   1      10.480   5.927  -4.153  1.00  0.00           C
        \\CONECT    1    2
        \\CONECT    2    3
        \\END
    ;

    var result = try parse(std.testing.allocator, pdb_content);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.topology.bonds.len);
}

test "parse no atoms returns error" {
    const pdb_content =
        \\HEADER    EMPTY
        \\END
    ;

    const result = parse(std.testing.allocator, pdb_content);
    try std.testing.expectError(ParseError.NoAtomsFound, result);
}

test "parse first model only" {
    const pdb_content =
        \\MODEL        1
        \\ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
        \\ENDMDL
        \\MODEL        2
        \\ATOM      2  CA  ALA A   1      99.999  99.999  99.999  1.00  0.00           C
        \\ENDMDL
        \\END
    ;

    var result = try parse(std.testing.allocator, pdb_content);
    defer result.deinit();

    // Only the first model's atom should be present
    try std.testing.expectEqual(@as(usize, 1), result.topology.atoms.len);
    try std.testing.expectApproxEqAbs(@as(f32, 11.104), result.frame.x[0], 0.001);
}

test "parse 1l2y.pdb" {
    const pdb_data = @embedFile("../../test_data/1l2y.pdb");

    var result = try parse(std.testing.allocator, pdb_data);
    defer result.deinit();

    const topo = &result.topology;
    const frame = &result.frame;

    // 1L2Y model 1 has 20 residues and many atoms
    try std.testing.expect(topo.atoms.len > 0);
    try std.testing.expect(topo.residues.len > 0);
    try std.testing.expect(topo.chains.len > 0);
    try std.testing.expectEqual(topo.atoms.len, frame.nAtoms());

    // Coordinates should not all be zero
    var all_zero = true;
    for (frame.x) |v| {
        if (v != 0.0) {
            all_zero = false;
            break;
        }
    }
    try std.testing.expect(!all_zero);

    // First residue should be ASN (1L2Y starts with ASN)
    try std.testing.expect(topo.residues[0].name.eqlSlice("ASN"));

    // Chain A should be present
    try std.testing.expect(topo.chains[0].name.eqlSlice("A"));

    // Elements should be correctly parsed (not all unknown)
    var has_known_element = false;
    for (topo.atoms) |a| {
        if (a.element != .X) {
            has_known_element = true;
            break;
        }
    }
    try std.testing.expect(has_known_element);

    // 1L2Y model 1: 20 residues
    try std.testing.expectEqual(@as(usize, 20), topo.residues.len);
}
