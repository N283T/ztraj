//! mmCIF file parser.
//!
//! Parses the _atom_site category from mmCIF/PDBx format files and returns a
//! ParseResult containing a Topology (atoms, residues, chains) and a Frame
//! (SOA coordinate arrays in Angstroms).
//!
//! ## Extracted _atom_site fields
//!
//! - label_atom_id   : atom name (e.g. CA, N)
//! - label_comp_id   : residue name (e.g. ALA)
//! - label_asym_id   : chain ID (strand ID)
//! - label_seq_id    : residue sequence number
//! - Cartn_x         : x coordinate (Angstroms)
//! - Cartn_y         : y coordinate (Angstroms)
//! - Cartn_z         : z coordinate (Angstroms)
//! - type_symbol     : element symbol

const std = @import("std");
const types = @import("../types.zig");
const elem = @import("../element.zig");
const cif = @import("cif_tokenizer.zig");

/// Error types for mmCIF parsing
pub const ParseError = error{
    /// No _atom_site loop found in the file
    NoAtomSiteLoop,
    /// Required coordinate fields missing (Cartn_x/y/z)
    MissingCoordinateField,
    /// A coordinate value is not a valid number
    InvalidCoordinate,
    /// Too many rows had unparseable coordinates (>10% of total rows)
    TooManyUnparseableCoordinates,
};

/// Indices of relevant _atom_site columns within the loop
const Columns = struct {
    cartn_x: ?usize = null,
    cartn_y: ?usize = null,
    cartn_z: ?usize = null,
    type_symbol: ?usize = null,
    label_atom_id: ?usize = null,
    auth_atom_id: ?usize = null,
    label_comp_id: ?usize = null,
    auth_comp_id: ?usize = null,
    label_asym_id: ?usize = null,
    auth_asym_id: ?usize = null,
    label_seq_id: ?usize = null,
    auth_seq_id: ?usize = null,
    group_pdb: ?usize = null,
    label_alt_id: ?usize = null,
    pdbx_pdb_model_num: ?usize = null,

    fn hasCoords(self: Columns) bool {
        return self.cartn_x != null and self.cartn_y != null and self.cartn_z != null;
    }

    fn atomNameCol(self: Columns) ?usize {
        return self.label_atom_id orelse self.auth_atom_id;
    }

    fn resNameCol(self: Columns) ?usize {
        return self.label_comp_id orelse self.auth_comp_id;
    }

    fn chainCol(self: Columns) ?usize {
        return self.label_asym_id orelse self.auth_asym_id;
    }

    fn seqIdCol(self: Columns) ?usize {
        return self.label_seq_id orelse self.auth_seq_id;
    }
};

/// Map a lowercase _atom_site field name to its column index within Columns.
fn assignColumn(cols: *Columns, field: []const u8, idx: usize) void {
    if (eqlCI(field, "Cartn_x")) {
        cols.cartn_x = idx;
    } else if (eqlCI(field, "Cartn_y")) {
        cols.cartn_y = idx;
    } else if (eqlCI(field, "Cartn_z")) {
        cols.cartn_z = idx;
    } else if (eqlCI(field, "type_symbol")) {
        cols.type_symbol = idx;
    } else if (eqlCI(field, "label_atom_id")) {
        cols.label_atom_id = idx;
    } else if (eqlCI(field, "auth_atom_id")) {
        cols.auth_atom_id = idx;
    } else if (eqlCI(field, "label_comp_id")) {
        cols.label_comp_id = idx;
    } else if (eqlCI(field, "auth_comp_id")) {
        cols.auth_comp_id = idx;
    } else if (eqlCI(field, "label_asym_id")) {
        cols.label_asym_id = idx;
    } else if (eqlCI(field, "auth_asym_id")) {
        cols.auth_asym_id = idx;
    } else if (eqlCI(field, "label_seq_id")) {
        cols.label_seq_id = idx;
    } else if (eqlCI(field, "auth_seq_id")) {
        cols.auth_seq_id = idx;
    } else if (eqlCI(field, "group_PDB")) {
        cols.group_pdb = idx;
    } else if (eqlCI(field, "label_alt_id")) {
        cols.label_alt_id = idx;
    } else if (eqlCI(field, "pdbx_PDB_model_num")) {
        cols.pdbx_pdb_model_num = idx;
    }
}

/// Case-insensitive string equality
fn eqlCI(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        if (std.ascii.toLower(ca) != std.ascii.toLower(cb)) return false;
    }
    return true;
}

/// Case-insensitive prefix match
fn startsWithCI(haystack: []const u8, prefix: []const u8) bool {
    if (haystack.len < prefix.len) return false;
    for (haystack[0..prefix.len], prefix) |h, p| {
        if (std.ascii.toLower(h) != std.ascii.toLower(p)) return false;
    }
    return true;
}

/// Parse a float, stripping CIF parenthetical uncertainty notation.
fn parseFloat(s: []const u8) !f32 {
    if (cif.isNull(s)) return ParseError.InvalidCoordinate;
    var end = s.len;
    for (s, 0..) |c, i| {
        if (c == '(') {
            end = i;
            break;
        }
    }
    const v = std.fmt.parseFloat(f64, s[0..end]) catch return ParseError.InvalidCoordinate;
    return @floatCast(v);
}

/// Flat intermediate atom record from one mmCIF _atom_site row
const RawAtom = struct {
    name: types.FixedString(4),
    res_name: types.FixedString(5),
    chain_id: types.FixedString(4),
    resid: i32,
    element: elem.Element,
    x: f32,
    y: f32,
    z: f32,
};

/// Parse an mmCIF-format string and return a ParseResult.
///
/// - Finds the first _atom_site loop and reads all rows.
/// - Both ATOM and HETATM group_PDB rows are included.
/// - Only the first model (pdbx_PDB_model_num == 1) is read when the field
///   is present; all atoms are taken when the field is absent.
/// - Builds Topology and Frame from the collected rows.
pub fn parse(allocator: std.mem.Allocator, data: []const u8) !types.ParseResult {
    var tokenizer = cif.Tokenizer.init(data);

    // ------------------------------------------------------------------
    // Phase 1: locate the _atom_site loop and collect column indices
    // ------------------------------------------------------------------
    var cols = Columns{};
    var num_cols: usize = 0;
    var found_loop = false;
    var in_atom_loop = false;

    outer: while (true) {
        const saved_pos = tokenizer.pos;
        const saved_line = tokenizer.line;
        const saved_col_pos = tokenizer.col;

        const tok = tokenizer.next();
        switch (tok) {
            .eof => break :outer,
            .loop => {
                in_atom_loop = false;
                cols = Columns{};
                num_cols = 0;
            },
            .tag => |tag| {
                if (startsWithCI(tag, "_atom_site.")) {
                    in_atom_loop = true;
                    const field = tag["_atom_site.".len..];
                    assignColumn(&cols, field, num_cols);
                    num_cols += 1;
                } else if (in_atom_loop) {
                    // A tag from a different category signals end of column list.
                    // Restore the tokenizer so the value-parsing phase sees it.
                    tokenizer.pos = saved_pos;
                    tokenizer.line = saved_line;
                    tokenizer.col = saved_col_pos;
                    if (cols.hasCoords()) {
                        found_loop = true;
                    }
                    break :outer;
                }
            },
            .value => {
                if (in_atom_loop) {
                    // First value — restore and let the row parser start fresh
                    tokenizer.pos = saved_pos;
                    tokenizer.line = saved_line;
                    tokenizer.col = saved_col_pos;
                    if (cols.hasCoords()) {
                        found_loop = true;
                    }
                    break :outer;
                }
            },
            .data_block => {
                if (in_atom_loop and cols.hasCoords()) {
                    tokenizer.pos = saved_pos;
                    tokenizer.line = saved_line;
                    tokenizer.col = saved_col_pos;
                    found_loop = true;
                    break :outer;
                }
                in_atom_loop = false;
                cols = Columns{};
                num_cols = 0;
            },
            else => {},
        }
    }

    if (!found_loop) {
        if (in_atom_loop and cols.hasCoords()) {
            // EOF reached while still in atom loop with valid cols — treat as found
            found_loop = true;
        }
    }

    if (!found_loop) {
        if (!cols.hasCoords() and in_atom_loop) return ParseError.MissingCoordinateField;
        return ParseError.NoAtomSiteLoop;
    }
    if (!cols.hasCoords()) return ParseError.MissingCoordinateField;

    // ------------------------------------------------------------------
    // Phase 2: read rows and collect raw atoms
    // ------------------------------------------------------------------
    var raw_atoms = std.ArrayList(RawAtom).empty;
    defer raw_atoms.deinit(allocator);

    var row = try allocator.alloc([]const u8, num_cols);
    defer allocator.free(row);

    var col_idx: usize = 0;
    var first_alt_loc: u8 = 0;
    var first_model: ?i32 = null;
    var rows_total: usize = 0;
    var rows_coord_failed: usize = 0;

    row_loop: while (true) {
        const tok = tokenizer.next();
        switch (tok) {
            .value => |val| {
                row[col_idx] = val;
                col_idx += 1;

                if (col_idx < num_cols) continue;

                // Complete row — reset column counter
                col_idx = 0;

                // Model filter: keep only the first model encountered
                if (cols.pdbx_pdb_model_num) |mc| {
                    const model_str = row[mc];
                    if (!cif.isNull(model_str)) {
                        const model_num = std.fmt.parseInt(i32, model_str, 10) catch 1;
                        if (first_model == null) {
                            first_model = model_num;
                        } else if (model_num != first_model.?) {
                            continue :row_loop;
                        }
                    }
                }

                // Alternate location filter
                if (cols.label_alt_id) |ac| {
                    const alt = row[ac];
                    if (!cif.isNull(alt) and alt.len > 0) {
                        const ac_char = alt[0];
                        if (first_alt_loc == 0) {
                            first_alt_loc = ac_char;
                        } else if (ac_char != first_alt_loc) {
                            continue :row_loop;
                        }
                    }
                }

                rows_total += 1;

                // Parse coordinates — track failures instead of silently skipping.
                const x = parseFloat(row[cols.cartn_x.?]) catch {
                    rows_coord_failed += 1;
                    continue :row_loop;
                };
                const y = parseFloat(row[cols.cartn_y.?]) catch {
                    rows_coord_failed += 1;
                    continue :row_loop;
                };
                const z = parseFloat(row[cols.cartn_z.?]) catch {
                    rows_coord_failed += 1;
                    continue :row_loop;
                };

                // Element
                const element: elem.Element = if (cols.type_symbol) |tc|
                    (if (!cif.isNull(row[tc])) elem.fromSymbol(row[tc]) else .X)
                else
                    .X;

                // Atom name
                const atom_name: types.FixedString(4) = if (cols.atomNameCol()) |nc|
                    (if (!cif.isNull(row[nc]))
                        types.FixedString(4).fromSlice(row[nc])
                    else
                        types.FixedString(4).fromSlice(""))
                else
                    types.FixedString(4).fromSlice("");

                // Residue name
                const res_name: types.FixedString(5) = if (cols.resNameCol()) |rc|
                    (if (!cif.isNull(row[rc]))
                        types.FixedString(5).fromSlice(row[rc])
                    else
                        types.FixedString(5).fromSlice("UNK"))
                else
                    types.FixedString(5).fromSlice("UNK");

                // Chain ID
                const chain_id: types.FixedString(4) = if (cols.chainCol()) |cc|
                    (if (!cif.isNull(row[cc]))
                        types.FixedString(4).fromSlice(row[cc])
                    else
                        types.FixedString(4).fromSlice(""))
                else
                    types.FixedString(4).fromSlice("");

                // Residue sequence number
                const resid: i32 = if (cols.seqIdCol()) |sc|
                    (if (!cif.isNull(row[sc]))
                        std.fmt.parseInt(i32, row[sc], 10) catch 0
                    else
                        0)
                else
                    0;

                try raw_atoms.append(allocator, RawAtom{
                    .name = atom_name,
                    .res_name = res_name,
                    .chain_id = chain_id,
                    .resid = resid,
                    .element = element,
                    .x = x,
                    .y = y,
                    .z = z,
                });
            },
            .eof, .loop, .data_block, .tag => break :row_loop,
            else => {},
        }
    }

    if (raw_atoms.items.len == 0) return ParseError.NoAtomSiteLoop;

    // Reject files where more than 10% of candidate rows had unparseable coordinates.
    if (rows_coord_failed > 0 and rows_total > 0) {
        // Use integer arithmetic: failed * 10 > total means >10%.
        if (rows_coord_failed * 10 > rows_total) {
            return ParseError.TooManyUnparseableCoordinates;
        }
    }

    // ------------------------------------------------------------------
    // Phase 3: count unique residues and chains
    // ------------------------------------------------------------------
    const ResKey = struct {
        chain_id: types.FixedString(4),
        resid: i32,
    };

    var n_residues: u32 = 0;
    var n_chains: u32 = 0;
    var prev_res: ?ResKey = null;
    var prev_chain: ?types.FixedString(4) = null;

    for (raw_atoms.items) |ra| {
        const chain_changed = if (prev_chain) |pc| !pc.eql(&ra.chain_id) else true;
        const res_key = ResKey{ .chain_id = ra.chain_id, .resid = ra.resid };
        const res_changed = if (prev_res) |pr|
            pr.resid != res_key.resid or !pr.chain_id.eql(&res_key.chain_id)
        else
            true;

        if (chain_changed) {
            n_chains += 1;
            prev_chain = ra.chain_id;
        }
        if (res_changed) {
            n_residues += 1;
            prev_res = res_key;
        }
    }

    const n_atoms: u32 = @intCast(raw_atoms.items.len);

    // ------------------------------------------------------------------
    // Phase 4: allocate and fill Topology + Frame
    // ------------------------------------------------------------------
    var topology = try types.Topology.init(allocator, .{
        .n_atoms = n_atoms,
        .n_residues = n_residues,
        .n_chains = n_chains,
        .n_bonds = 0,
    });
    errdefer topology.deinit();

    var frame = try types.Frame.init(allocator, n_atoms);
    errdefer frame.deinit();

    var atom_idx: u32 = 0;
    var res_idx: u32 = 0;
    var chain_idx: u32 = 0;

    prev_res = null;
    prev_chain = null;
    var chain_res_start: u32 = 0;
    var res_atom_start: u32 = 0;

    for (raw_atoms.items) |ra| {
        const chain_changed = if (prev_chain) |pc| !pc.eql(&ra.chain_id) else true;
        const res_key = ResKey{ .chain_id = ra.chain_id, .resid = ra.resid };
        const res_changed = if (prev_res) |pr|
            pr.resid != res_key.resid or !pr.chain_id.eql(&res_key.chain_id)
        else
            true;

        if (chain_changed) {
            if (chain_idx > 0) {
                topology.chains[chain_idx - 1].residue_range = .{
                    .start = chain_res_start,
                    .len = res_idx - chain_res_start,
                };
            }
            chain_res_start = res_idx;
            topology.chains[chain_idx] = types.Chain{
                .name = ra.chain_id,
                .residue_range = .{ .start = res_idx, .len = 0 },
            };
            chain_idx += 1;
            prev_chain = ra.chain_id;
        }

        if (res_changed) {
            if (res_idx > 0 and atom_idx > res_atom_start) {
                topology.residues[res_idx - 1].atom_range = .{
                    .start = res_atom_start,
                    .len = atom_idx - res_atom_start,
                };
            }
            res_atom_start = atom_idx;
            topology.residues[res_idx] = types.Residue{
                .name = ra.res_name,
                .chain_index = chain_idx - 1,
                .atom_range = .{ .start = atom_idx, .len = 0 },
                .resid = ra.resid,
            };
            res_idx += 1;
            prev_res = res_key;
        }

        topology.atoms[atom_idx] = types.Atom{
            .name = ra.name,
            .element = ra.element,
            .residue_index = res_idx - 1,
        };
        frame.x[atom_idx] = ra.x;
        frame.y[atom_idx] = ra.y;
        frame.z[atom_idx] = ra.z;
        atom_idx += 1;
    }

    // Close last residue and chain
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

    try topology.validate();
    return types.ParseResult{ .topology = topology, .frame = frame };
}

// ============================================================================
// Tests
// ============================================================================

test "parse simple mmCIF" {
    const source =
        \\data_TEST
        \\#
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.label_atom_id
        \\_atom_site.label_comp_id
        \\_atom_site.label_asym_id
        \\_atom_site.label_seq_id
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\_atom_site.Cartn_z
        \\1 N N   ASN A 1 10.000 20.000 30.000
        \\2 C CA  ASN A 1 11.000 21.000 31.000
        \\3 O O   ASN A 1 12.000 22.000 32.000
        \\4 N N   LEU A 2 13.000 23.000 33.000
        \\#
    ;

    var result = try parse(std.testing.allocator, source);
    defer result.deinit();

    const topo = &result.topology;
    const frame = &result.frame;

    try std.testing.expectEqual(@as(usize, 4), topo.atoms.len);
    try std.testing.expectEqual(@as(usize, 4), frame.nAtoms());

    // Residues: ASN (resid 1), LEU (resid 2)
    try std.testing.expectEqual(@as(usize, 2), topo.residues.len);
    try std.testing.expect(topo.residues[0].name.eqlSlice("ASN"));
    try std.testing.expect(topo.residues[1].name.eqlSlice("LEU"));

    // Chain A
    try std.testing.expectEqual(@as(usize, 1), topo.chains.len);
    try std.testing.expect(topo.chains[0].name.eqlSlice("A"));

    // Coordinates
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), frame.x[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), frame.y[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), frame.z[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), frame.x[1], 0.001);

    // Elements
    try std.testing.expectEqual(elem.Element.N, topo.atoms[0].element);
    try std.testing.expectEqual(elem.Element.C, topo.atoms[1].element);
    try std.testing.expectEqual(elem.Element.O, topo.atoms[2].element);

    // Atom names
    try std.testing.expect(topo.atoms[0].name.eqlSlice("N"));
    try std.testing.expect(topo.atoms[1].name.eqlSlice("CA"));
}

test "parse mmCIF residue and chain ranges" {
    const source =
        \\data_TEST
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.label_atom_id
        \\_atom_site.label_comp_id
        \\_atom_site.label_asym_id
        \\_atom_site.label_seq_id
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\_atom_site.Cartn_z
        \\1 N N  ALA A 1 1.0 2.0 3.0
        \\2 C CA ALA A 1 4.0 5.0 6.0
        \\3 N N  GLY A 2 7.0 8.0 9.0
        \\4 O O  HOH B 1 0.0 0.0 0.0
        \\#
    ;

    var result = try parse(std.testing.allocator, source);
    defer result.deinit();

    const topo = &result.topology;

    // 3 residues: ALA(A1), GLY(A2), HOH(B1)
    try std.testing.expectEqual(@as(usize, 3), topo.residues.len);
    // 2 chains: A, B
    try std.testing.expectEqual(@as(usize, 2), topo.chains.len);

    // ALA: atoms 0-1
    try std.testing.expectEqual(@as(u32, 0), topo.residues[0].atom_range.start);
    try std.testing.expectEqual(@as(u32, 2), topo.residues[0].atom_range.len);
    // GLY: atom 2
    try std.testing.expectEqual(@as(u32, 2), topo.residues[1].atom_range.start);
    try std.testing.expectEqual(@as(u32, 1), topo.residues[1].atom_range.len);
    // HOH: atom 3
    try std.testing.expectEqual(@as(u32, 3), topo.residues[2].atom_range.start);
    try std.testing.expectEqual(@as(u32, 1), topo.residues[2].atom_range.len);

    // Chain A: residues 0-1
    try std.testing.expectEqual(@as(u32, 0), topo.chains[0].residue_range.start);
    try std.testing.expectEqual(@as(u32, 2), topo.chains[0].residue_range.len);
    // Chain B: residue 2
    try std.testing.expectEqual(@as(u32, 2), topo.chains[1].residue_range.start);
    try std.testing.expectEqual(@as(u32, 1), topo.chains[1].residue_range.len);
}

test "parse mmCIF missing required fields" {
    const source =
        \\data_TEST
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\1 C 10.0 20.0
        \\#
    ;

    const result = parse(std.testing.allocator, source);
    try std.testing.expectError(ParseError.MissingCoordinateField, result);
}

test "parse mmCIF no atom_site loop" {
    const source =
        \\data_TEST
        \\loop_
        \\_cell.length_a
        \\_cell.length_b
        \\10.0 20.0
        \\#
    ;

    const result = parse(std.testing.allocator, source);
    try std.testing.expectError(ParseError.NoAtomSiteLoop, result);
}

test "parse mmCIF parenthetical uncertainty" {
    const source =
        \\data_TEST
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.label_atom_id
        \\_atom_site.label_comp_id
        \\_atom_site.label_asym_id
        \\_atom_site.label_seq_id
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\_atom_site.Cartn_z
        \\1 C CA ALA A 1 10.123(45) 20.456(67) 30.789(89)
        \\#
    ;

    var result = try parse(std.testing.allocator, source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.topology.atoms.len);
    try std.testing.expectApproxEqAbs(@as(f32, 10.123), result.frame.x[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20.456), result.frame.y[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 30.789), result.frame.z[0], 0.001);
}

test "parse mmCIF first model only" {
    const source =
        \\data_TEST
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.label_atom_id
        \\_atom_site.label_comp_id
        \\_atom_site.label_asym_id
        \\_atom_site.label_seq_id
        \\_atom_site.pdbx_PDB_model_num
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\_atom_site.Cartn_z
        \\1 N N ALA A 1 1 10.0 20.0 30.0
        \\2 C CA ALA A 1 1 11.0 21.0 31.0
        \\3 N N ALA A 1 2 99.0 99.0 99.0
        \\#
    ;

    var result = try parse(std.testing.allocator, source);
    defer result.deinit();

    // Only model 1 atoms
    try std.testing.expectEqual(@as(usize, 2), result.topology.atoms.len);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result.frame.x[0], 0.001);
}
