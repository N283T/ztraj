//! PDB Parser for DSSP.
//!
//! This module provides a PDB format parser that extracts atom coordinates
//! and groups them into residues for DSSP secondary structure calculation.
//!
//! ## Supported Records
//!
//! - ATOM: Standard amino acid atoms
//! - HETATM: Heteroatoms (optional)
//! - SSBOND: Disulfide bond records
//! - MODEL/ENDMDL: Multi-model support
//!
//! ## PDB Format Reference
//!
//! ATOM record columns (1-indexed):
//! - 1-6: Record name ("ATOM  " or "HETATM")
//! - 7-11: Atom serial number
//! - 13-16: Atom name
//! - 17: Alternate location indicator
//! - 18-20: Residue name
//! - 22: Chain identifier
//! - 23-26: Residue sequence number
//! - 27: Insertion code
//! - 31-38: X coordinate
//! - 39-46: Y coordinate
//! - 47-54: Z coordinate
//! - 77-78: Element symbol (optional)
//!
//! SSBOND record columns:
//! - 1-6: "SSBOND"
//! - 16: Chain ID 1
//! - 18-21: Sequence number 1
//! - 30: Chain ID 2
//! - 32-35: Sequence number 2

const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const residue_mod = @import("residue.zig");
const json_parser = @import("json_parser.zig");

const Vec3f32 = types.Vec3f32;
const ResidueType = types.ResidueType;
const Residue = residue_mod.Residue;
const SideChainAtom = residue_mod.SideChainAtom;
const SSBond = json_parser.SSBond;
const ParseResult = json_parser.ParseResult;

/// Maximum file size for reading (256 MB)
const MAX_FILE_SIZE = 256 * 1024 * 1024;

/// Error types for PDB parsing
pub const ParseError = error{
    /// No ATOM records found
    NoAtomRecords,
    /// Invalid coordinate value
    InvalidCoordinate,
    /// File read error
    FileReadError,
    /// Memory allocation failed
    OutOfMemory,
};

/// Temporary atom data during parsing
const TempAtom = struct {
    chain_id: u8,
    seq_id: i32,
    ins_code: u8,
    comp_id: [3]u8,
    atom_name: [4]u8,
    x: f32,
    y: f32,
    z: f32,
    is_backbone: bool,
};

/// PDB Parser
pub const PdbParser = struct {
    allocator: Allocator,
    /// Filter to include only ATOM records (exclude HETATM)
    atom_only: bool = true,
    /// Filter to include only first alternate location
    first_alt_loc_only: bool = true,
    /// Model number to extract (null = first model, 0 = all models)
    model_num: ?u32 = 1,

    pub fn init(allocator: Allocator) PdbParser {
        return .{ .allocator = allocator };
    }

    /// Parse PDB from a string
    pub fn parse(self: *PdbParser, source: []const u8) ParseError!ParseResult {
        var atoms = std.ArrayListUnmanaged(TempAtom){};
        defer atoms.deinit(self.allocator);

        var ss_bonds = std.ArrayListUnmanaged(SSBond){};
        defer ss_bonds.deinit(self.allocator);

        var first_alt_loc: ?u8 = null;
        var current_model: u32 = 1;
        var in_target_model = true;

        // Parse line by line
        var lines = std.mem.splitScalar(u8, source, '\n');
        while (lines.next()) |line| {
            // Skip short lines
            if (line.len < 6) continue;

            const record_type = line[0..6];

            // Handle MODEL records
            if (std.mem.eql(u8, record_type, "MODEL ")) {
                if (line.len >= 14) {
                    current_model = std.fmt.parseInt(u32, std.mem.trim(u8, line[10..14], " "), 10) catch 1;
                }
                if (self.model_num) |target| {
                    in_target_model = (current_model == target);
                } else {
                    in_target_model = true;
                }
                continue;
            }

            if (std.mem.eql(u8, record_type, "ENDMDL")) {
                // If we've processed the target model, stop
                if (self.model_num) |target| {
                    if (current_model == target) break;
                }
                continue;
            }

            // Parse ATOM/HETATM records
            if (std.mem.eql(u8, record_type, "ATOM  ") or
                (!self.atom_only and std.mem.eql(u8, record_type, "HETATM")))
            {
                if (!in_target_model) continue;

                if (self.parseAtomLine(line, &first_alt_loc)) |atom| {
                    atoms.append(self.allocator, atom) catch return ParseError.OutOfMemory;
                }
                continue;
            }

            // Parse SSBOND records
            if (std.mem.eql(u8, record_type, "SSBOND")) {
                if (self.parseSSBondLine(line)) |bond| {
                    ss_bonds.append(self.allocator, bond) catch return ParseError.OutOfMemory;
                }
                continue;
            }
        }

        if (atoms.items.len == 0) {
            return ParseError.NoAtomRecords;
        }

        // Group atoms into residues
        return try self.buildResidues(atoms.items, ss_bonds.items);
    }

    /// Parse PDB from a file
    pub fn parseFile(self: *PdbParser, path: []const u8) ParseError!ParseResult {
        const file = std.fs.cwd().openFile(path, .{}) catch {
            return ParseError.FileReadError;
        };
        defer file.close();

        const source = file.readToEndAlloc(self.allocator, MAX_FILE_SIZE) catch |err| {
            return if (err == error.OutOfMemory) ParseError.OutOfMemory else ParseError.FileReadError;
        };
        defer self.allocator.free(source);

        return self.parse(source);
    }

    /// Parse a single ATOM/HETATM line
    fn parseAtomLine(self: *PdbParser, line: []const u8, first_alt_loc: *?u8) ?TempAtom {
        if (line.len < 54) return null;

        // Check alternate location
        if (self.first_alt_loc_only and line.len > 16) {
            const alt_loc = line[16];
            if (alt_loc != ' ') {
                if (first_alt_loc.*) |first| {
                    if (alt_loc != first) return null;
                } else {
                    first_alt_loc.* = alt_loc;
                }
            }
        }

        // Extract atom name (columns 13-16, 0-indexed: 12-16)
        var atom_name: [4]u8 = .{ ' ', ' ', ' ', ' ' };
        if (line.len >= 16) {
            @memcpy(&atom_name, line[12..16]);
        }

        // Trim and check atom name
        const trimmed_name = std.mem.trim(u8, &atom_name, " ");

        // Skip hydrogen atoms - check element symbol (columns 77-78) if available
        if (line.len >= 78) {
            const element = std.mem.trim(u8, line[76..78], " ");
            if (element.len > 0 and (element[0] == 'H' or element[0] == 'D')) {
                return null;
            }
        } else if (trimmed_name.len > 0 and (trimmed_name[0] == 'H' or trimmed_name[0] == 'D')) {
            // Fallback: in protein ATOM records, names starting with H/D are hydrogen/deuterium
            // (HG for mercury would be in HETATM with element symbol "HG")
            return null;
        }

        // Check for backbone atoms
        const is_backbone = std.mem.eql(u8, trimmed_name, "N") or
            std.mem.eql(u8, trimmed_name, "CA") or
            std.mem.eql(u8, trimmed_name, "C") or
            std.mem.eql(u8, trimmed_name, "O");

        // Extract residue name (columns 18-20, 0-indexed: 17-20)
        var comp_id: [3]u8 = .{ ' ', ' ', ' ' };
        if (line.len >= 20) {
            @memcpy(&comp_id, line[17..20]);
        }

        // Extract chain ID (column 22, 0-indexed: 21)
        const chain_id: u8 = if (line.len > 21) line[21] else ' ';

        // Extract sequence number (columns 23-26, 0-indexed: 22-26)
        var seq_id: i32 = 0;
        if (line.len >= 26) {
            const seq_str = std.mem.trim(u8, line[22..26], " ");
            seq_id = std.fmt.parseInt(i32, seq_str, 10) catch 0;
        }

        // Extract insertion code (column 27, 0-indexed: 26)
        const ins_code: u8 = if (line.len > 26) line[26] else ' ';

        // Extract coordinates (columns 31-38, 39-46, 47-54, 0-indexed: 30-38, 38-46, 46-54)
        const x = parseCoord(line[30..38]) catch return null;
        const y = parseCoord(line[38..46]) catch return null;
        const z = parseCoord(line[46..54]) catch return null;

        return TempAtom{
            .chain_id = chain_id,
            .seq_id = seq_id,
            .ins_code = ins_code,
            .comp_id = comp_id,
            .atom_name = atom_name,
            .x = @floatCast(x),
            .y = @floatCast(y),
            .z = @floatCast(z),
            .is_backbone = is_backbone,
        };
    }

    /// Parse a single SSBOND line
    fn parseSSBondLine(self: *PdbParser, line: []const u8) ?SSBond {
        _ = self;
        if (line.len < 35) return null;

        var bond = SSBond{};

        // Chain 1 (column 16, 0-indexed: 15)
        bond.chain1[0] = line[15];
        bond.chain1_len = 1;

        // Sequence number 1 (columns 18-21, 0-indexed: 17-21)
        const seq1_str = std.mem.trim(u8, line[17..21], " ");
        bond.seq1 = std.fmt.parseInt(i32, seq1_str, 10) catch return null;

        // Chain 2 (column 30, 0-indexed: 29)
        bond.chain2[0] = line[29];
        bond.chain2_len = 1;

        // Sequence number 2 (columns 32-35, 0-indexed: 31-35)
        const seq2_str = std.mem.trim(u8, line[31..35], " ");
        bond.seq2 = std.fmt.parseInt(i32, seq2_str, 10) catch return null;

        return bond;
    }

    /// Build Residue structs from collected atoms
    fn buildResidues(self: *PdbParser, atoms: []const TempAtom, ss_bond_list: []const SSBond) ParseError!ParseResult {
        // Group atoms by (chain_id, seq_id, ins_code)
        var residue_starts = std.ArrayListUnmanaged(usize){};
        defer residue_starts.deinit(self.allocator);

        var prev_chain: u8 = 0;
        var prev_seq: i32 = std.math.minInt(i32);
        var prev_ins: u8 = 0;

        for (atoms, 0..) |atom, i| {
            if (atom.chain_id != prev_chain or atom.seq_id != prev_seq or atom.ins_code != prev_ins) {
                residue_starts.append(self.allocator, i) catch return ParseError.OutOfMemory;
                prev_chain = atom.chain_id;
                prev_seq = atom.seq_id;
                prev_ins = atom.ins_code;
            }
        }

        const num_residues = residue_starts.items.len;
        if (num_residues == 0) {
            return ParseError.NoAtomRecords;
        }

        // Count side chain atoms
        var total_side_chain: usize = 0;
        for (atoms) |atom| {
            if (!atom.is_backbone) total_side_chain += 1;
        }

        // Allocate residues and side chain storage
        const residues = self.allocator.alloc(Residue, num_residues) catch return ParseError.OutOfMemory;
        errdefer self.allocator.free(residues);

        const side_chain_storage = self.allocator.alloc(SideChainAtom, total_side_chain) catch return ParseError.OutOfMemory;
        errdefer self.allocator.free(side_chain_storage);

        var sc_offset: usize = 0;

        for (residue_starts.items, 0..) |start, res_idx| {
            const end = if (res_idx + 1 < residue_starts.items.len)
                residue_starts.items[res_idx + 1]
            else
                atoms.len;

            var res = Residue{};
            const first_atom = atoms[start];

            // Chain ID
            res.chain_id[0] = first_atom.chain_id;
            res.chain_id_len = 1;

            // Sequence ID
            res.seq_id = first_atom.seq_id;

            // Compound ID
            const comp_trimmed = std.mem.trim(u8, &first_atom.comp_id, " ");
            const cpd_len = @min(comp_trimmed.len, 4);
            @memcpy(res.compound_id[0..cpd_len], comp_trimmed[0..cpd_len]);
            res.compound_id_len = @intCast(cpd_len);

            // Residue type
            res.residue_type = ResidueType.fromCompoundId(comp_trimmed);

            // Sequential number - use safe cast for defensive programming
            res.number = std.math.cast(u32, res_idx) orelse return ParseError.OutOfMemory;

            // Process atoms for this residue
            var atom_count: u8 = 0;
            const sc_start = sc_offset;

            for (atoms[start..end]) |atom| {
                const pos = Vec3f32{ .x = atom.x, .y = atom.y, .z = atom.z };
                const name = std.mem.trim(u8, &atom.atom_name, " ");

                if (std.mem.eql(u8, name, "N")) {
                    res.n = pos;
                    atom_count += 1;
                } else if (std.mem.eql(u8, name, "CA")) {
                    res.ca = pos;
                    atom_count += 1;
                } else if (std.mem.eql(u8, name, "C")) {
                    res.c = pos;
                    atom_count += 1;
                } else if (std.mem.eql(u8, name, "O")) {
                    res.o = pos;
                    atom_count += 1;
                } else {
                    // Side chain atom
                    var sc_atom = SideChainAtom{};
                    const name_len = @min(name.len, 4);
                    @memcpy(sc_atom.name[0..name_len], name[0..name_len]);
                    sc_atom.name_len = @intCast(name_len);
                    sc_atom.pos = pos;
                    side_chain_storage[sc_offset] = sc_atom;
                    sc_offset += 1;
                }
            }

            res.complete = (atom_count == 4);
            res.side_chain = side_chain_storage[sc_start..sc_offset];
            residues[res_idx] = res;
        }

        // Copy SS bonds
        const ss_bonds = self.allocator.alloc(SSBond, ss_bond_list.len) catch return ParseError.OutOfMemory;
        errdefer self.allocator.free(ss_bonds);
        @memcpy(ss_bonds, ss_bond_list);

        return ParseResult{
            .residues = residues,
            .ss_bonds = ss_bonds,
            .side_chain_storage = side_chain_storage,
            .allocator = self.allocator,
        };
    }
};

/// Parse a PDB coordinate field (8 characters, right-justified)
fn parseCoord(field: []const u8) !f64 {
    const trimmed = std.mem.trim(u8, field, " ");
    if (trimmed.len == 0) return ParseError.InvalidCoordinate;
    return std.fmt.parseFloat(f64, trimmed) catch {
        return ParseError.InvalidCoordinate;
    };
}

// ============================================================================
// Tests
// ============================================================================

test "parse simple PDB" {
    const source =
        \\ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00 10.00           N
        \\ATOM      2  CA  ALA A   1      11.000  21.000  31.000  1.00 10.00           C
        \\ATOM      3  C   ALA A   1      12.000  22.000  32.000  1.00 10.00           C
        \\ATOM      4  O   ALA A   1      13.000  23.000  33.000  1.00 10.00           O
        \\ATOM      5  CB  ALA A   1      11.500  20.500  30.500  1.00 10.00           C
        \\END
    ;

    var parser = PdbParser.init(std.testing.allocator);
    var result = try parser.parse(source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.residues.len);
    try std.testing.expectEqualStrings("A", result.residues[0].getChainId());
    try std.testing.expectEqual(@as(i32, 1), result.residues[0].seq_id);
    try std.testing.expectEqualStrings("ALA", result.residues[0].getCompoundId());
    try std.testing.expect(result.residues[0].complete);

    // Check backbone coordinates
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result.residues[0].n.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), result.residues[0].ca.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), result.residues[0].c.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), result.residues[0].o.x, 0.001);

    // Check side chain
    try std.testing.expectEqual(@as(usize, 1), result.residues[0].side_chain.len);
    try std.testing.expectEqualStrings("CB", result.residues[0].side_chain[0].getName());
}

test "parse multiple residues" {
    const source =
        \\ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00 10.00           N
        \\ATOM      2  CA  ALA A   1      11.000  21.000  31.000  1.00 10.00           C
        \\ATOM      3  C   ALA A   1      12.000  22.000  32.000  1.00 10.00           C
        \\ATOM      4  O   ALA A   1      13.000  23.000  33.000  1.00 10.00           O
        \\ATOM      5  N   GLY A   2      14.000  24.000  34.000  1.00 10.00           N
        \\ATOM      6  CA  GLY A   2      15.000  25.000  35.000  1.00 10.00           C
        \\ATOM      7  C   GLY A   2      16.000  26.000  36.000  1.00 10.00           C
        \\ATOM      8  O   GLY A   2      17.000  27.000  37.000  1.00 10.00           O
        \\END
    ;

    var parser = PdbParser.init(std.testing.allocator);
    var result = try parser.parse(source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.residues.len);
    try std.testing.expectEqualStrings("ALA", result.residues[0].getCompoundId());
    try std.testing.expectEqualStrings("GLY", result.residues[1].getCompoundId());
    try std.testing.expectEqual(@as(i32, 1), result.residues[0].seq_id);
    try std.testing.expectEqual(@as(i32, 2), result.residues[1].seq_id);
}

test "parse with SS bonds" {
    const source =
        \\SSBOND   1 CYS A    6    CYS A  127                          1555   1555  2.03
        \\ATOM      1  N   CYS A   6      10.000  20.000  30.000  1.00 10.00           N
        \\ATOM      2  CA  CYS A   6      11.000  21.000  31.000  1.00 10.00           C
        \\ATOM      3  C   CYS A   6      12.000  22.000  32.000  1.00 10.00           C
        \\ATOM      4  O   CYS A   6      13.000  23.000  33.000  1.00 10.00           O
        \\END
    ;

    var parser = PdbParser.init(std.testing.allocator);
    var result = try parser.parse(source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.ss_bonds.len);
    try std.testing.expectEqual(@as(i32, 6), result.ss_bonds[0].seq1);
    try std.testing.expectEqual(@as(i32, 127), result.ss_bonds[0].seq2);
}

test "skip hydrogen atoms" {
    const source =
        \\ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00 10.00           N
        \\ATOM      2  H   ALA A   1      10.500  20.500  30.500  1.00 10.00           H
        \\ATOM      3  CA  ALA A   1      11.000  21.000  31.000  1.00 10.00           C
        \\ATOM      4  HA  ALA A   1      11.500  21.500  31.500  1.00 10.00           H
        \\ATOM      5  C   ALA A   1      12.000  22.000  32.000  1.00 10.00           C
        \\ATOM      6  O   ALA A   1      13.000  23.000  33.000  1.00 10.00           O
        \\END
    ;

    var parser = PdbParser.init(std.testing.allocator);
    var result = try parser.parse(source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.residues.len);
    try std.testing.expect(result.residues[0].complete);
    // No side chain atoms (all hydrogens skipped)
    try std.testing.expectEqual(@as(usize, 0), result.residues[0].side_chain.len);
}

test "alternate locations" {
    const source =
        \\ATOM      1  N  AALA A   1      10.000  20.000  30.000  0.50 10.00           N
        \\ATOM      2  N  BALA A   1      10.100  20.100  30.100  0.50 10.00           N
        \\ATOM      3  CA AALA A   1      11.000  21.000  31.000  0.50 10.00           C
        \\ATOM      4  CA BALA A   1      11.100  21.100  31.100  0.50 10.00           C
        \\ATOM      5  C  AALA A   1      12.000  22.000  32.000  0.50 10.00           C
        \\ATOM      6  C  BALA A   1      12.100  22.100  32.100  0.50 10.00           C
        \\ATOM      7  O  AALA A   1      13.000  23.000  33.000  0.50 10.00           O
        \\ATOM      8  O  BALA A   1      13.100  23.100  33.100  0.50 10.00           O
        \\END
    ;

    var parser = PdbParser.init(std.testing.allocator);
    parser.first_alt_loc_only = true;
    var result = try parser.parse(source);
    defer result.deinit();

    // Should only have atoms from alternate location A
    try std.testing.expectEqual(@as(usize, 1), result.residues.len);
    try std.testing.expect(result.residues[0].complete);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result.residues[0].n.x, 0.001);
}

test "multi-model PDB" {
    const source =
        \\MODEL        1
        \\ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00 10.00           N
        \\ATOM      2  CA  ALA A   1      11.000  21.000  31.000  1.00 10.00           C
        \\ATOM      3  C   ALA A   1      12.000  22.000  32.000  1.00 10.00           C
        \\ATOM      4  O   ALA A   1      13.000  23.000  33.000  1.00 10.00           O
        \\ENDMDL
        \\MODEL        2
        \\ATOM      1  N   ALA A   1      20.000  30.000  40.000  1.00 10.00           N
        \\ATOM      2  CA  ALA A   1      21.000  31.000  41.000  1.00 10.00           C
        \\ATOM      3  C   ALA A   1      22.000  32.000  42.000  1.00 10.00           C
        \\ATOM      4  O   ALA A   1      23.000  33.000  43.000  1.00 10.00           O
        \\ENDMDL
        \\END
    ;

    var parser = PdbParser.init(std.testing.allocator);
    parser.model_num = 1; // Only parse model 1
    var result = try parser.parse(source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.residues.len);
    // Should have coordinates from model 1
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result.residues[0].n.x, 0.001);
}

test "no ATOM records" {
    const source =
        \\HEADER    TEST
        \\TITLE     EMPTY STRUCTURE
        \\END
    ;

    var parser = PdbParser.init(std.testing.allocator);
    const result = parser.parse(source);
    try std.testing.expectError(ParseError.NoAtomRecords, result);
}
