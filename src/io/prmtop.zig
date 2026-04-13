//! AMBER PRMTOP topology file parser.
//!
//! Parses AMBER parm7 format files (.prmtop, .parm7, .top) and returns a
//! Topology containing atoms, residues, chains, bonds, charges, and masses.
//!
//! ## PRMTOP Format
//!
//! The file consists of `%FLAG` sections, each with a `%FORMAT` directive
//! specifying FORTRAN-style fixed-width fields. Sections used:
//!
//! - POINTERS:             System counts (NATOM, NRES, NBONH, MBONA, ...)
//! - ATOM_NAME:            Atom names (20a4)
//! - ATOMIC_NUMBER:        Atomic numbers (10I8) — optional (amber12+)
//! - CHARGE:               Partial charges (5E16.8) — optional
//! - MASS:                 Atomic masses in daltons (5E16.8) — optional
//! - RESIDUE_LABEL:        Residue names (20a4)
//! - RESIDUE_POINTER:      First atom of each residue (10I8, 1-indexed)
//! - BONDS_INC_HYDROGEN:   Bonds with H (10I8, coordinate indices)
//! - BONDS_WITHOUT_HYDROGEN: Bonds without H (10I8, coordinate indices)
//!
//! Bond atom indices in prmtop are stored as coordinate array offsets
//! (actual_atom_index * 3). To recover atom indices, divide by 3.
//!
//! Reference: https://ambermd.org/FileFormats.php

const std = @import("std");
const types = @import("../types.zig");
const elem = @import("../element.zig");

/// Error types for PRMTOP parsing
pub const ParseError = error{
    /// Not a valid PRMTOP file (missing %VERSION header)
    InvalidFormat,
    /// Chamber-style topology (CHARMM-converted) is not supported
    ChamberNotSupported,
    /// Required %FLAG section is missing
    MissingSections,
    /// A FORTRAN format specification could not be parsed
    InvalidFormatSpec,
    /// A numeric field could not be parsed
    InvalidFieldValue,
    /// Data is inconsistent (e.g. residue pointer out of range)
    InconsistentData,
};

/// POINTERS indices (per AMBER specification)
const PTR_NATOM: usize = 0;
const PTR_NRES: usize = 11;
const PTR_NBONH: usize = 2;
const PTR_MBONA: usize = 3;

/// AMBER internal charge unit conversion factor.
/// AMBER stores charges pre-multiplied by 18.2223 (= sqrt(332.0636 kcal*A/mol*e^2))
/// so that Coulomb energy E = q_i * q_j / r can be computed without the constant.
const amber_charge_factor: f64 = 18.2223;

/// Parsed FORTRAN format: only the field width is retained.
/// e.g. "20a4" -> width=4; "10I8" -> width=8; "5E16.8" -> width=16
const FortranFormat = struct {
    width: u32,
};

/// Parse a FORTRAN format string like "(20a4)" or "(10I8)" or "(5E16.8)".
/// We only need the field width for fixed-width slicing.
fn parseFortranFormat(fmt_line: []const u8) ?FortranFormat {
    // Find content between parentheses
    const open = std.mem.indexOfScalar(u8, fmt_line, '(') orelse return null;
    const close = std.mem.indexOfScalar(u8, fmt_line, ')') orelse return null;
    if (close <= open + 1) return null;
    const inner = fmt_line[open + 1 .. close];

    // Parse: skip leading count digits, then type char, then width digits
    var pos: usize = 0;
    // Skip count (e.g. "20" in "20a4")
    while (pos < inner.len and inner[pos] >= '0' and inner[pos] <= '9') : (pos += 1) {}
    // Skip type character (a, I, E, etc.)
    if (pos >= inner.len) return null;
    pos += 1;
    // Parse width
    const width_start = pos;
    while (pos < inner.len and inner[pos] >= '0' and inner[pos] <= '9') : (pos += 1) {}
    if (pos == width_start) return null;
    const width = std.fmt.parseInt(u32, inner[width_start..pos], 10) catch return null;

    return FortranFormat{ .width = width };
}

/// Infer element from AMBER atom name when ATOMIC_NUMBER is unavailable.
/// Follows mdtraj's heuristic: check for common multi-char element prefixes
/// (CL, NA, MG, ZN), then fall back to first character.
fn inferElement(atom_name: []const u8) elem.Element {
    if (atom_name.len == 0) return .X;

    var upper: [4]u8 = .{ 0, 0, 0, 0 };
    const copy_len = @min(atom_name.len, 4);
    for (0..copy_len) |i| {
        upper[i] = std.ascii.toUpper(atom_name[i]);
    }

    // Check multi-character elements that don't clash with common atom names.
    // Only the most unambiguous ions/metals are matched here (following mdtraj).
    // Others (Fe, Cu, Mn, Co, Ni, Br, Ca) are typically resolved via ATOMIC_NUMBER.
    if (copy_len >= 2) {
        if (upper[0] == 'C' and upper[1] == 'L') return .Cl;
        if (upper[0] == 'N' and upper[1] == 'A') return .Na;
        if (upper[0] == 'M' and upper[1] == 'G') return .Mg;
        if (upper[0] == 'Z' and upper[1] == 'N') return .Zn;
    }

    // Fall back to first character
    return elem.fromSymbol(&[_]u8{upper[0]});
}

/// Parse a PRMTOP file and return topology data.
/// prmtop is a topology-only format — no coordinate data is stored.
/// Returns Topology directly; caller must call .deinit().
pub fn parseTopology(allocator: std.mem.Allocator, data: []const u8) !types.Topology {
    // -------------------------------------------------------------------------
    // Validate header
    // -------------------------------------------------------------------------
    var lines = std.mem.splitScalar(u8, data, '\n');
    const first_line = lines.next() orelse return ParseError.InvalidFormat;
    const first_trimmed = std.mem.trimRight(u8, first_line, "\r ");
    if (!std.mem.startsWith(u8, first_trimmed, "%VERSION")) {
        return ParseError.InvalidFormat;
    }

    // -------------------------------------------------------------------------
    // Collect raw sections: for each %FLAG, store the data lines
    // -------------------------------------------------------------------------
    const Section = struct {
        format: FortranFormat,
        data_start: usize,
        data_end: usize,
    };

    var sections = std.StringHashMap(Section).init(allocator);
    defer sections.deinit();

    // Re-iterate to collect sections
    var line_iter = std.mem.splitScalar(u8, data, '\n');
    var current_flag: ?[]const u8 = null;
    var current_format: ?FortranFormat = null;
    var data_start: ?usize = null;
    var byte_offset: usize = 0;

    while (line_iter.next()) |line| {
        const trimmed = std.mem.trimRight(u8, line, "\r");
        const line_end = byte_offset + line.len + 1; // +1 for \n

        if (std.mem.startsWith(u8, trimmed, "%FLAG")) {
            // Close previous section
            if (current_flag) |flag| {
                if (current_format) |fmt| {
                    if (data_start) |ds| {
                        try sections.put(flag, Section{
                            .format = fmt,
                            .data_start = ds,
                            .data_end = byte_offset,
                        });
                    }
                }
            }
            // Extract flag name
            const rest = std.mem.trimLeft(u8, trimmed[5..], " ");
            const flag_end = std.mem.indexOfAny(u8, rest, " \t") orelse rest.len;
            current_flag = rest[0..flag_end];
            current_format = null;
            data_start = null;

            // Detect Chamber-style topology
            if (std.mem.eql(u8, current_flag.?, "CTITLE")) {
                return ParseError.ChamberNotSupported;
            }
        } else if (std.mem.startsWith(u8, trimmed, "%FORMAT")) {
            current_format = parseFortranFormat(trimmed) orelse
                return ParseError.InvalidFormatSpec;
            data_start = line_end;
        } else if (std.mem.startsWith(u8, trimmed, "%VERSION") or
            std.mem.startsWith(u8, trimmed, "%COMMENT"))
        {
            // Skip
        }

        byte_offset = line_end;
    }
    // Close last section
    if (current_flag) |flag| {
        if (current_format) |fmt| {
            if (data_start) |ds| {
                try sections.put(flag, Section{
                    .format = fmt,
                    .data_start = ds,
                    .data_end = byte_offset,
                });
            }
        }
    }

    // -------------------------------------------------------------------------
    // Parse POINTERS
    // -------------------------------------------------------------------------
    const pointers_sec = sections.get("POINTERS") orelse return ParseError.MissingSections;
    var pointers: [31]u32 = undefined;
    {
        const sec_data = data[pointers_sec.data_start..@min(pointers_sec.data_end, data.len)];
        var ptr_lines = std.mem.splitScalar(u8, sec_data, '\n');
        var ptr_idx: usize = 0;
        while (ptr_lines.next()) |pline| {
            const ptrimmed = std.mem.trimRight(u8, pline, "\r ");
            if (ptrimmed.len == 0) continue;
            if (std.mem.startsWith(u8, ptrimmed, "%")) break;
            const width = pointers_sec.format.width;
            var col: usize = 0;
            while (col + width <= ptrimmed.len and ptr_idx < 31) : (col += width) {
                const field = std.mem.trim(u8, ptrimmed[col .. col + width], " ");
                if (field.len > 0) {
                    pointers[ptr_idx] = std.fmt.parseInt(u32, field, 10) catch
                        return ParseError.InvalidFieldValue;
                    ptr_idx += 1;
                }
            }
        }
        if (ptr_idx < 12) return ParseError.MissingSections;
    }

    const n_atoms = pointers[PTR_NATOM];
    const n_res = pointers[PTR_NRES];
    const n_bonds_h = pointers[PTR_NBONH];
    const n_bonds_no_h = pointers[PTR_MBONA];
    const n_bonds = n_bonds_h + n_bonds_no_h;

    // -------------------------------------------------------------------------
    // Parse fixed-width section helper
    // -------------------------------------------------------------------------
    const SectionData = struct {
        sec_data: []const u8,
        width: u32,

        const Self = @This();

        fn items(self: Self, allocator_: std.mem.Allocator, expected: u32) ![][]const u8 {
            var result = std.ArrayListUnmanaged([]const u8){};
            errdefer result.deinit(allocator_);

            if (expected > 0) {
                try result.ensureTotalCapacity(allocator_, expected);
            }

            var sec_lines = std.mem.splitScalar(u8, self.sec_data, '\n');
            while (sec_lines.next()) |sline| {
                const strimmed = std.mem.trimRight(u8, sline, "\r");
                if (strimmed.len == 0) continue;
                if (std.mem.startsWith(u8, strimmed, "%")) break;
                var col: usize = 0;
                while (col + self.width <= strimmed.len) : (col += self.width) {
                    const field = strimmed[col .. col + self.width];
                    try result.append(allocator_, field);
                }
                // Handle partial last field
                if (col < strimmed.len) {
                    const remaining = strimmed[col..];
                    if (std.mem.trim(u8, remaining, " ").len > 0) {
                        try result.append(allocator_, remaining);
                    }
                }
            }
            return result.toOwnedSlice(allocator_);
        }
    };

    // -------------------------------------------------------------------------
    // Parse ATOM_NAME
    // -------------------------------------------------------------------------
    const atom_name_sec = sections.get("ATOM_NAME") orelse return ParseError.MissingSections;
    const atom_name_reader = SectionData{
        .sec_data = data[atom_name_sec.data_start..@min(atom_name_sec.data_end, data.len)],
        .width = atom_name_sec.format.width,
    };
    const atom_names = try atom_name_reader.items(allocator, n_atoms);
    defer allocator.free(atom_names);

    if (atom_names.len < n_atoms) return ParseError.InconsistentData;

    // -------------------------------------------------------------------------
    // Parse ATOMIC_NUMBER (optional, amber12+)
    // -------------------------------------------------------------------------
    var atomic_numbers: ?[][]const u8 = null;
    defer if (atomic_numbers) |an| allocator.free(an);

    if (sections.get("ATOMIC_NUMBER")) |an_sec| {
        const an_reader = SectionData{
            .sec_data = data[an_sec.data_start..@min(an_sec.data_end, data.len)],
            .width = an_sec.format.width,
        };
        atomic_numbers = try an_reader.items(allocator, n_atoms);
    }

    // -------------------------------------------------------------------------
    // Parse CHARGE (optional — AMBER charges / 18.2223 = electron charge units)
    // -------------------------------------------------------------------------
    var charge_fields: ?[][]const u8 = null;
    defer if (charge_fields) |cf| allocator.free(cf);

    if (sections.get("CHARGE")) |charge_sec| {
        const charge_reader = SectionData{
            .sec_data = data[charge_sec.data_start..@min(charge_sec.data_end, data.len)],
            .width = charge_sec.format.width,
        };
        charge_fields = try charge_reader.items(allocator, n_atoms);
    }

    // -------------------------------------------------------------------------
    // Parse MASS (optional — daltons)
    // -------------------------------------------------------------------------
    var mass_fields: ?[][]const u8 = null;
    defer if (mass_fields) |mf| allocator.free(mf);

    if (sections.get("MASS")) |mass_sec| {
        const mass_reader = SectionData{
            .sec_data = data[mass_sec.data_start..@min(mass_sec.data_end, data.len)],
            .width = mass_sec.format.width,
        };
        mass_fields = try mass_reader.items(allocator, n_atoms);
    }

    // -------------------------------------------------------------------------
    // Parse RESIDUE_LABEL
    // -------------------------------------------------------------------------
    const res_label_sec = sections.get("RESIDUE_LABEL") orelse return ParseError.MissingSections;
    const res_label_reader = SectionData{
        .sec_data = data[res_label_sec.data_start..@min(res_label_sec.data_end, data.len)],
        .width = res_label_sec.format.width,
    };
    const res_labels = try res_label_reader.items(allocator, n_res);
    defer allocator.free(res_labels);

    if (res_labels.len < n_res) return ParseError.InconsistentData;

    // -------------------------------------------------------------------------
    // Parse RESIDUE_POINTER
    // -------------------------------------------------------------------------
    const res_ptr_sec = sections.get("RESIDUE_POINTER") orelse return ParseError.MissingSections;
    const res_ptr_reader = SectionData{
        .sec_data = data[res_ptr_sec.data_start..@min(res_ptr_sec.data_end, data.len)],
        .width = res_ptr_sec.format.width,
    };
    const res_ptrs_raw = try res_ptr_reader.items(allocator, n_res);
    defer allocator.free(res_ptrs_raw);

    if (res_ptrs_raw.len < n_res) return ParseError.InconsistentData;

    // Convert to 0-based first-atom indices
    const res_first_atom = try allocator.alloc(u32, n_res + 1);
    defer allocator.free(res_first_atom);

    for (0..n_res) |i| {
        const field = std.mem.trim(u8, res_ptrs_raw[i], " ");
        const val = std.fmt.parseInt(u32, field, 10) catch return ParseError.InvalidFieldValue;
        if (val == 0) return ParseError.InconsistentData;
        res_first_atom[i] = val - 1; // 1-indexed -> 0-indexed
    }
    res_first_atom[n_res] = n_atoms;

    // -------------------------------------------------------------------------
    // Parse bond sections
    // -------------------------------------------------------------------------
    const BondRaw = struct { atom_i: u32, atom_j: u32 };
    var bonds_list = std.ArrayListUnmanaged(BondRaw){};
    defer bonds_list.deinit(allocator);
    try bonds_list.ensureTotalCapacity(allocator, n_bonds);

    inline for (.{ "BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN" }) |flag| {
        if (sections.get(flag)) |bond_sec| {
            const bond_reader = SectionData{
                .sec_data = data[bond_sec.data_start..@min(bond_sec.data_end, data.len)],
                .width = bond_sec.format.width,
            };
            const bond_fields = try bond_reader.items(allocator, 0);
            defer allocator.free(bond_fields);

            // Each bond record is 3 fields: atom_i*3, atom_j*3, type_index
            var fi: usize = 0;
            while (fi + 2 < bond_fields.len) : (fi += 3) {
                const ai_str = std.mem.trim(u8, bond_fields[fi], " ");
                const aj_str = std.mem.trim(u8, bond_fields[fi + 1], " ");
                const ai_raw = std.fmt.parseInt(i32, ai_str, 10) catch
                    return ParseError.InvalidFieldValue;
                const aj_raw = std.fmt.parseInt(i32, aj_str, 10) catch
                    return ParseError.InvalidFieldValue;
                if (ai_raw < 0 or aj_raw < 0) continue;
                try bonds_list.append(allocator, BondRaw{
                    .atom_i = @intCast(@divTrunc(@as(u32, @intCast(ai_raw)), 3)),
                    .atom_j = @intCast(@divTrunc(@as(u32, @intCast(aj_raw)), 3)),
                });
            }
        }
    }

    // -------------------------------------------------------------------------
    // Build Topology
    // -------------------------------------------------------------------------
    const n_chains: u32 = 1; // PRMTOP has no chain information
    var topology = try types.Topology.init(allocator, .{
        .n_atoms = n_atoms,
        .n_residues = n_res,
        .n_chains = n_chains,
        .n_bonds = @intCast(bonds_list.items.len),
    });
    errdefer topology.deinit();

    // Fill chain (single chain for the whole system)
    topology.chains[0] = types.Chain{
        .name = types.FixedString(4).fromSlice(""),
        .residue_range = .{ .start = 0, .len = n_res },
    };

    // Fill residues
    for (0..n_res) |ri| {
        const res_name = std.mem.trim(u8, res_labels[ri], " ");
        topology.residues[ri] = types.Residue{
            .name = types.FixedString(5).fromSlice(res_name),
            .chain_index = 0,
            .atom_range = .{
                .start = res_first_atom[ri],
                .len = res_first_atom[ri + 1] - res_first_atom[ri],
            },
            .resid = @intCast(ri + 1),
        };
    }

    // Fill atoms
    for (0..n_atoms) |ai| {
        const atom_name = std.mem.trim(u8, atom_names[ai], " ");

        // Determine element
        const element_val: elem.Element = if (atomic_numbers) |an| blk: {
            if (ai < an.len) {
                const num_str = std.mem.trim(u8, an[ai], " ");
                const atomic_num = std.fmt.parseInt(u8, num_str, 10) catch break :blk inferElement(atom_name);
                if (atomic_num > 0) break :blk elem.fromAtomicNumber(atomic_num);
            }
            break :blk inferElement(atom_name);
        } else inferElement(atom_name);

        // Find residue index for this atom
        var res_idx: u32 = 0;
        var found_res = false;
        for (0..n_res) |ri| {
            if (ai >= res_first_atom[ri] and ai < res_first_atom[ri + 1]) {
                res_idx = @intCast(ri);
                found_res = true;
                break;
            }
        }
        if (!found_res) return ParseError.InconsistentData;

        topology.atoms[ai] = types.Atom{
            .name = types.FixedString(4).fromSlice(atom_name),
            .element = element_val,
            .residue_index = res_idx,
        };
    }

    // Fill bonds
    for (bonds_list.items, 0..) |bond, bi| {
        topology.bonds[bi] = types.Bond{
            .atom_i = bond.atom_i,
            .atom_j = bond.atom_j,
        };
    }

    // Fill charges (convert from AMBER internal units to electron charges)
    if (charge_fields) |cf| {
        if (cf.len < n_atoms) return ParseError.InconsistentData;
        const charges = try allocator.alloc(f32, n_atoms);
        errdefer allocator.free(charges);
        for (0..n_atoms) |ai| {
            const field = std.mem.trim(u8, cf[ai], " ");
            const amber_charge = std.fmt.parseFloat(f64, field) catch
                return ParseError.InvalidFieldValue;
            charges[ai] = @floatCast(amber_charge / amber_charge_factor);
        }
        topology.charges = charges;
    }

    // Fill explicit masses (daltons, directly usable)
    if (mass_fields) |mf| {
        if (mf.len < n_atoms) return ParseError.InconsistentData;
        const em = try allocator.alloc(f32, n_atoms);
        errdefer allocator.free(em);
        for (0..n_atoms) |ai| {
            const field = std.mem.trim(u8, mf[ai], " ");
            em[ai] = @floatCast(std.fmt.parseFloat(f64, field) catch
                return ParseError.InvalidFieldValue);
        }
        topology.explicit_masses = em;
    }

    try topology.validate();
    return topology;
}

// ============================================================================
// Tests
// ============================================================================

test "parseFortranFormat basic formats" {
    const f1 = parseFortranFormat("%FORMAT(20a4)").?;
    try std.testing.expectEqual(@as(u32, 4), f1.width);

    const f2 = parseFortranFormat("%FORMAT(10I8)").?;
    try std.testing.expectEqual(@as(u32, 8), f2.width);

    const f3 = parseFortranFormat("%FORMAT(5E16.8)").?;
    try std.testing.expectEqual(@as(u32, 16), f3.width);

    try std.testing.expect(parseFortranFormat("no parens") == null);
    try std.testing.expect(parseFortranFormat("()") == null);
}

test "inferElement common atoms" {
    try std.testing.expectEqual(elem.Element.H, inferElement("H"));
    try std.testing.expectEqual(elem.Element.C, inferElement("CA")); // alpha carbon, not calcium
    try std.testing.expectEqual(elem.Element.N, inferElement("N"));
    try std.testing.expectEqual(elem.Element.O, inferElement("O"));
    try std.testing.expectEqual(elem.Element.Cl, inferElement("CL"));
    try std.testing.expectEqual(elem.Element.Na, inferElement("NA"));
    try std.testing.expectEqual(elem.Element.Mg, inferElement("MG"));
    try std.testing.expectEqual(elem.Element.Zn, inferElement("ZN"));
    // Fe, Cu, etc. fall back to first character (F, C) without ATOMIC_NUMBER
    try std.testing.expectEqual(elem.Element.F, inferElement("FE"));
}

test "inferElement atom names" {
    // Typical AMBER atom names
    try std.testing.expectEqual(elem.Element.H, inferElement("HH31"));
    try std.testing.expectEqual(elem.Element.C, inferElement("CH3"));
    try std.testing.expectEqual(elem.Element.C, inferElement("CB"));
    try std.testing.expectEqual(elem.Element.N, inferElement("NH"));
    try std.testing.expectEqual(elem.Element.O, inferElement("OD1"));
    try std.testing.expectEqual(elem.Element.H, inferElement("HA"));
    try std.testing.expectEqual(elem.Element.H, inferElement("HB1"));
    try std.testing.expectEqual(elem.Element.H, inferElement("H1"));
}

fn readTestFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(allocator, path, 16 * 1024 * 1024);
}

test "parse alanine dipeptide implicit prmtop" {
    const allocator = std.testing.allocator;
    const data = try readTestFile(allocator, "test_data/alanine-dipeptide-implicit.prmtop");
    defer allocator.free(data);

    var topo = try parseTopology(allocator, data);
    defer topo.deinit();

    // alanine dipeptide: 22 atoms, 3 residues (ACE, ALA, NME), 21 bonds
    try std.testing.expectEqual(@as(usize, 22), topo.atoms.len);
    try std.testing.expectEqual(@as(usize, 3), topo.residues.len);
    try std.testing.expectEqual(@as(usize, 1), topo.chains.len);
    try std.testing.expectEqual(@as(usize, 21), topo.bonds.len);

    // Check residue names
    try std.testing.expect(topo.residues[0].name.eqlSlice("ACE"));
    try std.testing.expect(topo.residues[1].name.eqlSlice("ALA"));
    try std.testing.expect(topo.residues[2].name.eqlSlice("NME"));

    // Check residue atom ranges
    try std.testing.expectEqual(@as(u32, 0), topo.residues[0].atom_range.start);
    try std.testing.expectEqual(@as(u32, 6), topo.residues[0].atom_range.len);
    try std.testing.expectEqual(@as(u32, 6), topo.residues[1].atom_range.start);
    try std.testing.expectEqual(@as(u32, 10), topo.residues[1].atom_range.len);
    try std.testing.expectEqual(@as(u32, 16), topo.residues[2].atom_range.start);
    try std.testing.expectEqual(@as(u32, 6), topo.residues[2].atom_range.len);

    // Check some atom names (ff14SB naming)
    try std.testing.expect(topo.atoms[0].name.eqlSlice("H1"));
    try std.testing.expect(topo.atoms[1].name.eqlSlice("CH3"));
    try std.testing.expect(topo.atoms[4].name.eqlSlice("C"));
    try std.testing.expect(topo.atoms[6].name.eqlSlice("N"));

    // Check element assignment (ATOMIC_NUMBER present in ff14SB prmtop)
    try std.testing.expectEqual(elem.Element.H, topo.atoms[0].element); // H1
    try std.testing.expectEqual(elem.Element.C, topo.atoms[1].element); // CH3
    try std.testing.expectEqual(elem.Element.C, topo.atoms[4].element); // C
    try std.testing.expectEqual(elem.Element.O, topo.atoms[5].element); // O
    try std.testing.expectEqual(elem.Element.N, topo.atoms[6].element); // N
    try std.testing.expectEqual(elem.Element.H, topo.atoms[7].element); // H

    // Check charges are populated (AMBER charge / 18.2223)
    try std.testing.expect(topo.charges != null);
    const charges = topo.charges.?;
    try std.testing.expectEqual(@as(usize, 22), charges.len);
    // HH31: AMBER charge 2.04636429E+00 / 18.2223 ≈ 0.1123
    try std.testing.expectApproxEqAbs(@as(f32, 0.1123), charges[0], 0.001);
    // O: AMBER charge -1.03484442E+01 / 18.2223 ≈ -0.5679
    try std.testing.expectApproxEqAbs(@as(f32, -0.5679), charges[5], 0.001);

    // Check explicit masses are populated
    try std.testing.expect(topo.explicit_masses != null);
    const em = topo.explicit_masses.?;
    try std.testing.expectEqual(@as(usize, 22), em.len);
    // HH31: mass 1.008
    try std.testing.expectApproxEqAbs(@as(f32, 1.008), em[0], 0.001);
    // CH3 (carbon): mass 12.01
    try std.testing.expectApproxEqAbs(@as(f32, 12.01), em[1], 0.001);
    // O: mass 16.0
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), em[5], 0.01);

    // Verify masses() prefers explicit_masses
    const mass_arr = try topo.masses(allocator);
    defer allocator.free(mass_arr);
    try std.testing.expectApproxEqAbs(@as(f64, 1.008), mass_arr[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 12.01), mass_arr[1], 0.001);
}

test "parse pentapeptide prmtop with ATOMIC_NUMBER" {
    const allocator = std.testing.allocator;
    const data = try readTestFile(allocator, "test_data/cpptraj_traj.prmtop");
    defer allocator.free(data);

    var topo = try parseTopology(allocator, data);
    defer topo.deinit();

    // ASP-ALA-TRP-GLU-ILE pentapeptide: 80 atoms, 5 residues
    try std.testing.expectEqual(@as(usize, 80), topo.atoms.len);
    try std.testing.expectEqual(@as(usize, 5), topo.residues.len);

    // Verify ATOMIC_NUMBER-based element assignment
    try std.testing.expectEqual(elem.Element.N, topo.atoms[0].element); // N (atomic number 7)
    try std.testing.expectEqual(elem.Element.H, topo.atoms[1].element); // H (atomic number 1)
    try std.testing.expectEqual(elem.Element.C, topo.atoms[2].element); // CA (atomic number 6)

    // Verify exact bond count (NBONH=36 + MBONA=45 = 81)
    try std.testing.expectEqual(@as(usize, 81), topo.bonds.len);

    // Verify charges and masses are populated
    try std.testing.expect(topo.charges != null);
    try std.testing.expectEqual(@as(usize, 80), topo.charges.?.len);
    try std.testing.expect(topo.explicit_masses != null);
    try std.testing.expectEqual(@as(usize, 80), topo.explicit_masses.?.len);
}

test "invalid prmtop rejected" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(ParseError.InvalidFormat, parseTopology(allocator, "not a prmtop file"));
    try std.testing.expectError(ParseError.InvalidFormat, parseTopology(allocator, ""));
}

test "chamber prmtop rejected" {
    const allocator = std.testing.allocator;
    const chamber_data = "%VERSION  VERSION_STAMP = V0001.000\n%FLAG CTITLE\n%FORMAT(20a4)\nsome title\n";
    try std.testing.expectError(ParseError.ChamberNotSupported, parseTopology(allocator, chamber_data));
}
