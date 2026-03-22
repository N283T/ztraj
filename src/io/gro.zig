//! GRO (GROMACS) file parser.
//!
//! Parses GROMACS coordinate files (.gro) and returns a ParseResult
//! containing a Topology (atoms, residues, chains, bonds) and a Frame
//! (SOA coordinate arrays).
//!
//! ## GRO Format (Fixed Width)
//!
//! Line 1:  Title (free text, may contain "t= X.XXX" for time in ps)
//! Line 2:  Number of atoms (integer)
//! Atom lines:
//!   Cols  0-4:   Residue number (right-aligned int, wraps at 99999)
//!   Cols  5-9:   Residue name (left-aligned, max 5 chars)
//!   Cols 10-14:  Atom name (right-aligned, max 5 chars)
//!   Cols 15-19:  Atom serial (right-aligned int)
//!   Col  20+:    X Y Z coordinates in nm (variable width, typically 8.3f)
//! Last line: Box vectors (3 or 9 floats in nm, space-separated)
//!
//! All coordinates and box vectors are in nm; ztraj uses Angstroms (×10).

const std = @import("std");
const types = @import("../types.zig");
const elem = @import("../element.zig");

/// Error types for GRO parsing
pub const ParseError = error{
    /// No atoms were found in the file
    NoAtomsFound,
    /// The atom count line could not be parsed
    InvalidAtomCount,
    /// The file structure is malformed
    InvalidFormat,
};

/// Intermediate flat atom record collected during parsing.
const RawAtom = struct {
    name: types.FixedString(4),
    res_name: types.FixedString(5),
    resid: i32,
    x: f32,
    y: f32,
    z: f32,
    element: elem.Element,
};

/// Detect the coordinate field width from the first atom line.
/// Finds the first '.' at or after index 20, then the second '.' after it.
/// Returns the distance between them (field width), or null if not found.
fn detectCoordWidth(line: []const u8) ?usize {
    if (line.len <= 20) return null;

    // Find first '.' at or after column 20
    var first_dot: usize = 20;
    while (first_dot < line.len and line[first_dot] != '.') : (first_dot += 1) {}
    if (first_dot >= line.len) return null;

    // Find second '.' after the first one
    var second_dot: usize = first_dot + 1;
    while (second_dot < line.len and line[second_dot] != '.') : (second_dot += 1) {}
    if (second_dot >= line.len) return null;

    return second_dot - first_dot;
}

/// Parse a coordinate field string to f32.
/// Strips leading whitespace and parses sign, integer part, and decimal part.
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

/// Infer element from GRO atom name (GRO has no element column).
/// Uses PDB convention: element is right-justified in the name field.
fn inferElement(atom_name: []const u8) elem.Element {
    if (atom_name.len < 2) return .X;
    const first = atom_name[0];
    const second = atom_name[1];
    if (first == ' ' or (first >= '0' and first <= '9')) {
        return elem.fromSymbol(&[_]u8{second});
    }
    return elem.fromSymbol(&[_]u8{first});
}

/// Extract simulation time from a GRO title line.
/// Scans for "t=" and parses the float that follows.
/// Returns 0.0 if not found or unparseable.
fn parseTime(title: []const u8) f32 {
    var i: usize = 0;
    while (i + 1 < title.len) : (i += 1) {
        if (title[i] == 't' and title[i + 1] == '=' and
            (i == 0 or title[i - 1] == ' '))
        {
            // Skip "t=" and optional spaces
            var j = i + 2;
            while (j < title.len and title[j] == ' ') : (j += 1) {}
            // Find end of number (space or end of string)
            var end = j;
            while (end < title.len and title[end] != ' ' and title[end] != '\r') : (end += 1) {}
            if (j < end) {
                return parseCoord(title[j..end]) orelse 0.0;
            }
        }
    }
    return 0.0;
}

/// Parse the GRO box vector line (last line of the file).
/// 3 values → orthogonal box (diagonal matrix).
/// 9 values → triclinic box (full 3×3 matrix).
/// All values are multiplied by 10 (nm → Angstroms).
/// Returns null on parse failure.
fn parseBoxVectors(line: []const u8) ?[3][3]f32 {
    var values: [9]f32 = [_]f32{0} ** 9;
    var count: usize = 0;

    var it = std.mem.tokenizeScalar(u8, line, ' ');
    while (it.next()) |token| {
        if (count >= 9) break;
        const v = parseCoord(token) orelse return null;
        values[count] = v;
        count += 1;
    }

    if (count != 3 and count != 9) return null;

    // Multiply all by 10 (nm → Å)
    for (&values) |*v| {
        v.* *= 10.0;
    }

    if (count == 3) {
        // Orthogonal: diagonal matrix
        return [3][3]f32{
            .{ values[0], 0.0, 0.0 },
            .{ 0.0, values[1], 0.0 },
            .{ 0.0, 0.0, values[2] },
        };
    } else {
        // Triclinic: GRO order is v1x v2y v3z v1y v1z v2x v2z v3x v3y
        //                         [0]  [1] [2] [3] [4] [5] [6] [7] [8]
        // ztraj convention: rows are box vectors (row-major, same as XTC/DCD).
        // Row 0 (a = v1): [v1x, v1y, v1z] = [idx0, idx3, idx4]
        // Row 1 (b = v2): [v2x, v2y, v2z] = [idx5, idx1, idx6]
        // Row 2 (c = v3): [v3x, v3y, v3z] = [idx7, idx8, idx2]
        return [3][3]f32{
            .{ values[0], values[3], values[4] },
            .{ values[5], values[1], values[6] },
            .{ values[7], values[8], values[2] },
        };
    }
}

/// Parse a GRO-format string and return a ParseResult.
///
/// - Parses all atom lines from the GRO file.
/// - Assigns all atoms to a single chain (GRO has no chain ID).
/// - Converts coordinates from nm to Angstroms (×10).
/// - Parses box vectors from the last line.
/// - Extracts simulation time from the title if present ("t= X.XXX").
pub fn parse(allocator: std.mem.Allocator, data: []const u8) !types.ParseResult {
    var lines = std.mem.splitScalar(u8, data, '\n');

    // -------------------------------------------------------------------------
    // Line 1: Title
    // -------------------------------------------------------------------------
    const title_line = lines.next() orelse return ParseError.InvalidFormat;
    const title = std.mem.trimRight(u8, title_line, "\r");
    const frame_time = parseTime(title);

    // -------------------------------------------------------------------------
    // Line 2: Atom count
    // -------------------------------------------------------------------------
    const count_line_raw = lines.next() orelse return ParseError.InvalidFormat;
    const count_line = std.mem.trim(u8, count_line_raw, " \r");
    const n_atoms_expected = std.fmt.parseInt(u32, count_line, 10) catch {
        return ParseError.InvalidAtomCount;
    };

    if (n_atoms_expected == 0) return ParseError.NoAtomsFound;

    // -------------------------------------------------------------------------
    // Pass 1: collect raw atoms
    // -------------------------------------------------------------------------
    var raw_atoms = std.ArrayListUnmanaged(RawAtom){};
    defer raw_atoms.deinit(allocator);

    try raw_atoms.ensureTotalCapacity(allocator, n_atoms_expected);

    var coord_width: ?usize = null;
    var box_line: []const u8 = "";

    var atom_count: u32 = 0;
    while (lines.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");

        if (atom_count >= n_atoms_expected) {
            // This is the box vector line (or trailing content)
            if (line.len > 0) {
                box_line = line;
            }
            break;
        }

        // Detect coordinate width from the first atom line
        if (coord_width == null) {
            coord_width = detectCoordWidth(line);
        }

        const cw = coord_width orelse return ParseError.InvalidFormat;

        // Need at least 20 + 3*cw characters for x, y, z
        if (line.len < 20 + 3 * cw) return ParseError.InvalidFormat;

        // Residue number: cols 0-4
        const resnum_str = std.mem.trim(u8, line[0..@min(5, line.len)], " ");
        const resid = std.fmt.parseInt(i32, resnum_str, 10) catch 0;

        // Residue name: cols 5-9
        const res_raw = if (line.len >= 10) line[5..10] else line[5..];
        const res_name = std.mem.trim(u8, res_raw, " ");

        // Atom name: cols 10-14
        const atom_raw = if (line.len >= 15) line[10..15] else line[10..];
        const atom_name = std.mem.trim(u8, atom_raw, " ");

        // Coordinates: starting at col 20, each field is cw wide
        const x_field = line[20 .. 20 + cw];
        const y_field = line[20 + cw .. 20 + 2 * cw];
        const z_field = line[20 + 2 * cw .. 20 + 3 * cw];

        const x_nm = parseCoord(x_field) orelse return ParseError.InvalidFormat;
        const y_nm = parseCoord(y_field) orelse return ParseError.InvalidFormat;
        const z_nm = parseCoord(z_field) orelse return ParseError.InvalidFormat;

        const element = inferElement(atom_name);

        raw_atoms.appendAssumeCapacity(RawAtom{
            .name = types.FixedString(4).fromSlice(atom_name),
            .res_name = types.FixedString(5).fromSlice(res_name),
            .resid = resid,
            .x = x_nm * 10.0,
            .y = y_nm * 10.0,
            .z = z_nm * 10.0,
            .element = element,
        });

        atom_count += 1;
    }

    if (raw_atoms.items.len == 0) return ParseError.NoAtomsFound;

    // -------------------------------------------------------------------------
    // Pass 2: count residue boundaries
    // -------------------------------------------------------------------------
    var n_residues: u32 = 0;
    var prev_resid: ?i32 = null;
    var prev_res_name: ?types.FixedString(5) = null;

    for (raw_atoms.items) |ra| {
        const res_changed = if (prev_resid) |pid|
            pid != ra.resid or !prev_res_name.?.eql(&ra.res_name)
        else
            true;

        if (res_changed) {
            n_residues += 1;
            prev_resid = ra.resid;
            prev_res_name = ra.res_name;
        }
    }

    // GRO has no chain IDs — always a single chain
    const n_chains: u32 = 1;
    const n_total_atoms: u32 = @intCast(raw_atoms.items.len);

    // -------------------------------------------------------------------------
    // Allocate Topology and Frame
    // -------------------------------------------------------------------------
    var topology = try types.Topology.init(allocator, .{
        .n_atoms = n_total_atoms,
        .n_residues = n_residues,
        .n_chains = n_chains,
        .n_bonds = 0,
    });
    errdefer topology.deinit();

    var frame = try types.Frame.init(allocator, n_total_atoms);
    errdefer frame.deinit();

    frame.time = frame_time;

    // Parse box vectors if we have them
    if (box_line.len > 0) {
        frame.box_vectors = parseBoxVectors(box_line);
    }

    // -------------------------------------------------------------------------
    // Pass 3: fill Topology and Frame
    // -------------------------------------------------------------------------
    var atom_idx: u32 = 0;
    var res_idx: u32 = 0;

    prev_resid = null;
    prev_res_name = null;

    var res_atom_start: u32 = 0;

    // Set the single chain
    topology.chains[0] = types.Chain{
        .name = types.FixedString(4).fromSlice(""),
        .residue_range = .{ .start = 0, .len = n_residues },
    };

    for (raw_atoms.items) |ra| {
        const res_changed = if (prev_resid) |pid|
            pid != ra.resid or !prev_res_name.?.eql(&ra.res_name)
        else
            true;

        if (res_changed) {
            // Close previous residue
            if (res_idx > 0) {
                topology.residues[res_idx - 1].atom_range = .{
                    .start = res_atom_start,
                    .len = atom_idx - res_atom_start,
                };
            }
            res_atom_start = atom_idx;
            topology.residues[res_idx] = types.Residue{
                .name = ra.res_name,
                .chain_index = 0,
                .atom_range = .{ .start = atom_idx, .len = 0 }, // finalized later
                .resid = ra.resid,
            };
            res_idx += 1;
            prev_resid = ra.resid;
            prev_res_name = ra.res_name;
        }

        // Fill atom
        topology.atoms[atom_idx] = types.Atom{
            .name = ra.name,
            .element = ra.element,
            .residue_index = res_idx - 1,
        };

        // Fill frame coordinates (already converted nm → Å in pass 1)
        frame.x[atom_idx] = ra.x;
        frame.y[atom_idx] = ra.y;
        frame.z[atom_idx] = ra.z;

        atom_idx += 1;
    }

    // Close the last residue
    if (res_idx > 0) {
        topology.residues[res_idx - 1].atom_range = .{
            .start = res_atom_start,
            .len = atom_idx - res_atom_start,
        };
    }

    try topology.validate();
    return types.ParseResult{ .topology = topology, .frame = frame };
}

// ============================================================================
// Tests
// ============================================================================

test "parse minimal GRO" {
    const data =
        \\Minimal test
        \\    3
        \\    1ALA      N    1   0.100   0.200   0.300
        \\    1ALA     CA    2   0.400   0.500   0.600
        \\    1ALA      C    3   0.700   0.800   0.900
        \\   1.000   1.000   1.000
    ;
    const allocator = std.testing.allocator;
    var result = try parse(allocator, data);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.topology.atoms.len);
    try std.testing.expectEqual(@as(usize, 1), result.topology.residues.len);
    try std.testing.expectEqual(@as(usize, 1), result.topology.chains.len);
    try std.testing.expectEqual(@as(usize, 0), result.topology.bonds.len);

    // Coordinates converted from nm to Angstroms (×10)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.frame.x[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result.frame.y[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result.frame.z[0], 0.01);

    // Atom names
    try std.testing.expectEqualStrings("N", result.topology.atoms[0].name.slice());
    try std.testing.expectEqualStrings("CA", result.topology.atoms[1].name.slice());
    try std.testing.expectEqualStrings("C", result.topology.atoms[2].name.slice());

    // Residue
    try std.testing.expectEqualStrings("ALA", result.topology.residues[0].name.slice());
    try std.testing.expectEqual(@as(i32, 1), result.topology.residues[0].resid);
}

test "parse GRO with multiple residues" {
    const data =
        \\Two residues
        \\    4
        \\    1ALA      N    1   0.100   0.200   0.300
        \\    1ALA     CA    2   0.400   0.500   0.600
        \\    2GLY      N    3   0.700   0.800   0.900
        \\    2GLY     CA    4   1.000   1.100   1.200
        \\   5.000   5.000   5.000
    ;
    const allocator = std.testing.allocator;
    var result = try parse(allocator, data);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 4), result.topology.atoms.len);
    try std.testing.expectEqual(@as(usize, 2), result.topology.residues.len);
    try std.testing.expectEqual(@as(usize, 1), result.topology.chains.len);

    try std.testing.expectEqualStrings("ALA", result.topology.residues[0].name.slice());
    try std.testing.expectEqualStrings("GLY", result.topology.residues[1].name.slice());

    // Residue atom ranges
    try std.testing.expectEqual(@as(u32, 0), result.topology.residues[0].atom_range.start);
    try std.testing.expectEqual(@as(u32, 2), result.topology.residues[0].atom_range.len);
    try std.testing.expectEqual(@as(u32, 2), result.topology.residues[1].atom_range.start);
    try std.testing.expectEqual(@as(u32, 2), result.topology.residues[1].atom_range.len);
}

test "parse GRO time from title" {
    const data =
        \\MD of protein, t= 100.000 step= 50000
        \\    1
        \\    1ALA      N    1   0.100   0.200   0.300
        \\   5.000   5.000   5.000
    ;
    const allocator = std.testing.allocator;
    var result = try parse(allocator, data);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 100.0), result.frame.time, 0.1);
}

test "parse GRO box vectors orthogonal" {
    const data =
        \\Box test
        \\    1
        \\    1ALA      N    1   0.100   0.200   0.300
        \\   2.000   3.000   4.000
    ;
    const allocator = std.testing.allocator;
    var result = try parse(allocator, data);
    defer result.deinit();

    try std.testing.expect(result.frame.box_vectors != null);
    const box = result.frame.box_vectors.?;
    // nm → Å: 2.000 × 10 = 20.0, etc.
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), box[0][0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), box[1][1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), box[2][2], 0.01);
    // Off-diagonal should be zero
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), box[0][1], 0.001);
}

test "parse GRO no atoms returns error" {
    const data =
        \\Empty
        \\    0
    ;
    const result = parse(std.testing.allocator, data);
    try std.testing.expectError(ParseError.NoAtomsFound, result);
}

test "parse GRO triclinic box vectors" {
    const data =
        \\Triclinic test
        \\    1
        \\    1ALA      N    1   0.100   0.200   0.300
        \\   1.00000   2.00000   3.00000   0.10000   0.20000   0.30000   0.40000   0.50000   0.60000
    ;
    const allocator = std.testing.allocator;
    var result = try parse(allocator, data);
    defer result.deinit();

    try std.testing.expect(result.frame.box_vectors != null);
    const box = result.frame.box_vectors.?;
    // GRO order: v1x v2y v3z v1y v1z v2x v2z v3x v3y
    //            [0]  [1] [2] [3] [4] [5] [6] [7] [8]
    // ztraj row-major: rows = box vectors
    // Row 0 (a=v1): [v1x, v1y, v1z] = [idx0, idx3, idx4] = [1.0, 0.1, 0.2] * 10
    // Row 1 (b=v2): [v2x, v2y, v2z] = [idx5, idx1, idx6] = [0.3, 2.0, 0.4] * 10
    // Row 2 (c=v3): [v3x, v3y, v3z] = [idx7, idx8, idx2] = [0.5, 0.6, 3.0] * 10
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), box[0][0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), box[0][1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), box[0][2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), box[1][0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), box[1][1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), box[1][2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), box[2][0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), box[2][1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), box[2][2], 0.01);
}

test "parse empty GRO returns error" {
    const result = parse(std.testing.allocator, "");
    try std.testing.expectError(error.InvalidFormat, result);
}

test "parse GRO with negative coordinates" {
    const data =
        \\Negative coords
        \\    1
        \\    1ALA      N    1  -0.100  -0.200  -0.300
        \\   1.000   1.000   1.000
    ;
    const allocator = std.testing.allocator;
    var result = try parse(allocator, data);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result.frame.x[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), result.frame.y[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), result.frame.z[0], 0.01);
}

test "parse GRO element inference" {
    const data =
        \\Element test
        \\    4
        \\    1ALA      N    1   0.100   0.200   0.300
        \\    1ALA     CA    2   0.400   0.500   0.600
        \\    1ALA      O    3   0.700   0.800   0.900
        \\    1ALA      H    4   1.000   1.100   1.200
        \\   1.000   1.000   1.000
    ;
    const allocator = std.testing.allocator;
    var result = try parse(allocator, data);
    defer result.deinit();

    const elem_mod = @import("../element.zig");
    try std.testing.expectEqual(elem_mod.Element.N, result.topology.atoms[0].element);
    try std.testing.expectEqual(elem_mod.Element.C, result.topology.atoms[1].element);
    try std.testing.expectEqual(elem_mod.Element.O, result.topology.atoms[2].element);
    try std.testing.expectEqual(elem_mod.Element.H, result.topology.atoms[3].element);
}

test "parse GRO without time in title" {
    const data =
        \\Simple title no time
        \\    1
        \\    1ALA      N    1   0.100   0.200   0.300
        \\   1.000   1.000   1.000
    ;
    const allocator = std.testing.allocator;
    var result = try parse(allocator, data);
    defer result.deinit();

    try std.testing.expectEqual(@as(f32, 0.0), result.frame.time);
}
