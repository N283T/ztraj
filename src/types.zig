//! Core data structures for molecular dynamics trajectory analysis.
//!
//! All structs with an allocator field own their data and must be deinitialized
//! via their deinit() method. Callers never free individual fields directly.

const std = @import("std");
const element_mod = @import("element.zig");
pub const Element = element_mod.Element;

// ============================================================================
// FixedString — generic comptime fixed-size string (no heap allocation)
// ============================================================================

/// A fixed-capacity string stored inline, carrying its own length.
/// Avoids heap allocation per atom/residue name.
/// N must be > 0. Strings longer than N are silently truncated.
pub fn FixedString(comptime N: u8) type {
    comptime {
        if (N == 0) @compileError("FixedString capacity must be > 0");
    }

    return struct {
        data: [N]u8 = [_]u8{0} ** N,
        len: u8 = 0,

        const Self = @This();

        /// Create from a slice. Truncates if s.len > N.
        pub fn fromSlice(s: []const u8) Self {
            var result = Self{};
            const copy_len: u8 = @intCast(@min(s.len, N));
            @memcpy(result.data[0..copy_len], s[0..copy_len]);
            result.len = copy_len;
            return result;
        }

        /// Return the active bytes as a slice.
        pub fn slice(self: *const Self) []const u8 {
            return self.data[0..self.len];
        }

        /// Equality with another FixedString of the same capacity.
        pub fn eql(self: *const Self, other: *const Self) bool {
            return self.len == other.len and std.mem.eql(u8, self.slice(), other.slice());
        }

        /// Equality with an arbitrary byte slice.
        pub fn eqlSlice(self: *const Self, s: []const u8) bool {
            return self.len == s.len and std.mem.eql(u8, self.slice(), s);
        }

        /// Support std.fmt formatting so FixedString prints its contents.
        /// The {f} specifier calls format(writer) in Zig 0.15+.
        pub fn format(self: Self, writer: anytype) !void {
            try writer.writeAll(self.data[0..self.len]);
        }
    };
}

// ============================================================================
// Range — contiguous index span
// ============================================================================

/// A half-open interval [start, start+len) used to map parent-to-child
/// relationships (e.g. chain -> residues, residue -> atoms) without
/// storing pointers.
pub const Range = struct {
    start: u32,
    len: u32,

    /// One-past-the-end index (exclusive upper bound).
    pub fn end(self: Range) u32 {
        return self.start + self.len;
    }
};

// ============================================================================
// Atom, Residue, Chain, Bond
// ============================================================================

/// A single atom within the topology.
pub const Atom = struct {
    /// Standard atom name (e.g. "CA", "N", "OG1"). 4-char PDB convention.
    name: FixedString(4),
    /// Chemical element.
    element: Element,
    /// 0-based index into Topology.residues.
    residue_index: u32,
};

/// A residue (amino acid, nucleotide, ligand, solvent molecule, etc.).
pub const Residue = struct {
    /// Residue name (e.g. "ALA", "GLY"). 5 chars covers mmCIF extended IDs.
    name: FixedString(5),
    /// 0-based index into Topology.chains.
    chain_index: u32,
    /// Half-open range into Topology.atoms.
    atom_range: Range,
    /// Sequence number from the source file (may be negative for insertion codes).
    resid: i32,
};

/// A polymer chain or molecule.
pub const Chain = struct {
    /// Chain identifier (e.g. "A", "B"). Stored as FixedString(4) for
    /// mmCIF multi-char strand IDs.
    name: FixedString(4),
    /// Half-open range into Topology.residues.
    residue_range: Range,
};

/// A covalent bond between two atoms (needed for H-bond donor/acceptor inference).
pub const Bond = struct {
    /// 0-based index of the first atom.
    atom_i: u32,
    /// 0-based index of the second atom.
    atom_j: u32,
};

// ============================================================================
// Topology
// ============================================================================

/// Sizes passed to Topology.init to pre-allocate all slices at once.
pub const TopologySizes = struct {
    n_atoms: usize,
    n_residues: usize,
    n_chains: usize,
    n_bonds: usize,
};

/// Flat topology: atoms, residues, chains, and bonds owned by this struct.
/// Created via init(), freed via deinit().
pub const Topology = struct {
    atoms: []Atom,
    residues: []Residue,
    chains: []Chain,
    bonds: []Bond,
    allocator: std.mem.Allocator,

    /// Allocate all slices with the given sizes.
    /// Memory is uninitialized — callers must fill every element before use.
    pub fn init(allocator: std.mem.Allocator, sizes: TopologySizes) !Topology {
        const atoms = try allocator.alloc(Atom, sizes.n_atoms);
        errdefer allocator.free(atoms);

        const residues = try allocator.alloc(Residue, sizes.n_residues);
        errdefer allocator.free(residues);

        const chains = try allocator.alloc(Chain, sizes.n_chains);
        errdefer allocator.free(chains);

        const bonds = try allocator.alloc(Bond, sizes.n_bonds);
        errdefer allocator.free(bonds);

        return Topology{
            .atoms = atoms,
            .residues = residues,
            .chains = chains,
            .bonds = bonds,
            .allocator = allocator,
        };
    }

    /// Free all slices owned by this topology.
    pub fn deinit(self: *Topology) void {
        self.allocator.free(self.atoms);
        self.allocator.free(self.residues);
        self.allocator.free(self.chains);
        self.allocator.free(self.bonds);
    }

    /// Validate cross-reference consistency.
    /// Call once after populating the topology to catch parser bugs early.
    pub fn validate(self: Topology) !void {
        for (self.atoms) |atom| {
            if (atom.residue_index >= self.residues.len)
                return error.InvalidResidueIndex;
        }
        for (self.residues) |res| {
            if (res.chain_index >= self.chains.len)
                return error.InvalidChainIndex;
            if (res.atom_range.start + res.atom_range.len > self.atoms.len)
                return error.InvalidAtomRange;
        }
        for (self.chains) |chain| {
            if (chain.residue_range.start + chain.residue_range.len > self.residues.len)
                return error.InvalidResidueRange;
        }
        for (self.bonds) |bond| {
            if (bond.atom_i >= self.atoms.len or bond.atom_j >= self.atoms.len)
                return error.InvalidBondIndex;
        }
    }

    /// Build a flat array of atomic masses (daltons) in atom-index order.
    /// Caller owns the returned slice and must free it with the same allocator.
    pub fn masses(self: Topology, allocator: std.mem.Allocator) ![]f64 {
        const result = try allocator.alloc(f64, self.atoms.len);
        for (self.atoms, 0..) |atom, i| {
            result[i] = atom.element.mass();
        }
        return result;
    }
};

// ============================================================================
// Frame — single trajectory snapshot (SOA layout for SIMD)
// ============================================================================

/// A single simulation snapshot.
/// Coordinates are stored in Structure-of-Arrays layout (separate x/y/z slices)
/// for SIMD-native access without transposition.
/// Units: Angstroms.
pub const Frame = struct {
    /// x coordinates, one per atom (Angstroms).
    x: []f32,
    /// y coordinates, one per atom (Angstroms).
    y: []f32,
    /// z coordinates, one per atom (Angstroms).
    z: []f32,
    /// Periodic boundary condition box vectors (rows are a, b, c vectors).
    /// null for non-periodic simulations.
    box_vectors: ?[3][3]f32,
    /// Simulation time in picoseconds. 0.0 if not stored in the file.
    time: f32,
    /// Integer step counter. 0 if not stored in the file.
    step: i32,
    allocator: std.mem.Allocator,
    /// Whether this Frame owns its coordinate memory.
    owns: bool = true,

    /// Allocate x/y/z arrays for n_atoms atoms.
    /// Arrays are zeroed to avoid spurious values if partially written.
    pub fn init(allocator: std.mem.Allocator, n_atoms: usize) !Frame {
        const x = try allocator.alloc(f32, n_atoms);
        errdefer allocator.free(x);

        const y = try allocator.alloc(f32, n_atoms);
        errdefer allocator.free(y);

        const z = try allocator.alloc(f32, n_atoms);
        errdefer allocator.free(z);

        @memset(x, 0.0);
        @memset(y, 0.0);
        @memset(z, 0.0);

        return Frame{
            .x = x,
            .y = y,
            .z = z,
            .box_vectors = null,
            .time = 0.0,
            .step = 0,
            .allocator = allocator,
            .owns = true,
        };
    }

    /// Create a non-owning Frame view over const coordinate slices.
    /// The resulting Frame does NOT own its coordinate memory —
    /// calling deinit() on it is a no-op.
    /// Use this for C API functions that receive raw const pointers.
    pub fn initView(
        x: []const f32,
        y: []const f32,
        z: []const f32,
    ) Frame {
        std.debug.assert(x.len == y.len and y.len == z.len);
        return .{
            // SAFETY: The coordinate slices are treated as read-only by all
            // analysis functions. @constCast is needed because Frame.x is []f32
            // (required by I/O writers that fill frame buffers). The 'owns'
            // flag ensures deinit() does not attempt to free this memory.
            .x = @constCast(x),
            .y = @constCast(y),
            .z = @constCast(z),
            .box_vectors = null,
            .time = 0.0,
            .step = 0,
            .allocator = undefined,
            .owns = false,
        };
    }

    /// Free all coordinate arrays (only if this Frame owns them).
    pub fn deinit(self: *Frame) void {
        if (!self.owns) return;
        self.allocator.free(self.x);
        self.allocator.free(self.y);
        self.allocator.free(self.z);
    }

    /// Number of atoms in this frame.
    pub fn nAtoms(self: Frame) usize {
        std.debug.assert(self.x.len == self.y.len and self.y.len == self.z.len);
        return self.x.len;
    }
};

// ============================================================================
// Trajectory — complete in-memory trajectory
// ============================================================================

/// A full trajectory loaded into memory.
/// Use for small systems or when random access to all frames is required.
/// For large trajectories, use a FrameIterator (future).
pub const Trajectory = struct {
    topology: Topology,
    frames: []Frame,
    allocator: std.mem.Allocator,

    /// Free all frames and the topology.
    pub fn deinit(self: *Trajectory) void {
        for (self.frames) |*frame| {
            frame.deinit();
        }
        self.allocator.free(self.frames);
        self.topology.deinit();
    }

    /// Number of frames in the trajectory.
    pub fn nFrames(self: Trajectory) usize {
        return self.frames.len;
    }

    /// Number of atoms (from topology).
    pub fn nAtoms(self: Trajectory) usize {
        return self.topology.atoms.len;
    }
};

// ============================================================================
// ParseResult — combined topology + single frame from a structure file
// ============================================================================

/// Returned by single-structure file parsers (PDB, mmCIF).
/// Owns both topology and frame; deinit() frees both.
pub const ParseResult = struct {
    topology: Topology,
    frame: Frame,

    /// Free both the topology and the frame.
    pub fn deinit(self: *ParseResult) void {
        self.topology.deinit();
        self.frame.deinit();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "FixedString(4) fromSlice and slice" {
    const FS4 = FixedString(4);
    const s = FS4.fromSlice("CA");
    try std.testing.expectEqualStrings("CA", s.slice());
    try std.testing.expectEqual(@as(u8, 2), s.len);
}

test "FixedString(4) exact capacity" {
    const FS4 = FixedString(4);
    const s = FS4.fromSlice("OG1A");
    try std.testing.expectEqualStrings("OG1A", s.slice());
    try std.testing.expectEqual(@as(u8, 4), s.len);
}

test "FixedString(4) truncation" {
    const FS4 = FixedString(4);
    // 5 chars -> truncated to 4
    const s = FS4.fromSlice("ABCDE");
    try std.testing.expectEqualStrings("ABCD", s.slice());
    try std.testing.expectEqual(@as(u8, 4), s.len);
}

test "FixedString(4) empty" {
    const FS4 = FixedString(4);
    const s = FS4.fromSlice("");
    try std.testing.expectEqualStrings("", s.slice());
    try std.testing.expectEqual(@as(u8, 0), s.len);
}

test "FixedString(4) eql" {
    const FS4 = FixedString(4);
    const a = FS4.fromSlice("CA");
    const b = FS4.fromSlice("CA");
    const c = FS4.fromSlice("CB");
    try std.testing.expect(a.eql(&b));
    try std.testing.expect(!a.eql(&c));
}

test "FixedString(4) eqlSlice" {
    const FS4 = FixedString(4);
    const s = FS4.fromSlice("N");
    try std.testing.expect(s.eqlSlice("N"));
    try std.testing.expect(!s.eqlSlice("NA"));
    try std.testing.expect(!s.eqlSlice(""));
}

test "FixedString(5) residue name" {
    const FS5 = FixedString(5);
    const ala = FS5.fromSlice("ALA");
    try std.testing.expectEqualStrings("ALA", ala.slice());
    try std.testing.expectEqual(@as(u8, 3), ala.len);

    // 5-char mmCIF extended ID
    const ext = FS5.fromSlice("ACNMP");
    try std.testing.expectEqualStrings("ACNMP", ext.slice());
    try std.testing.expectEqual(@as(u8, 5), ext.len);
}

test "FixedString(5) truncation at 5" {
    const FS5 = FixedString(5);
    const s = FS5.fromSlice("ABCDEF");
    try std.testing.expectEqualStrings("ABCDE", s.slice());
}

test "FixedString format" {
    const FS4 = FixedString(4);
    const s = FS4.fromSlice("CA");
    const rendered = try std.fmt.allocPrint(std.testing.allocator, "{f}", .{s});
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("CA", rendered);
}

test "FixedString default is empty" {
    const FS4 = FixedString(4);
    const s = FS4{};
    try std.testing.expectEqual(@as(u8, 0), s.len);
    try std.testing.expectEqualStrings("", s.slice());
}

test "Range end" {
    const r = Range{ .start = 10, .len = 5 };
    try std.testing.expectEqual(@as(u32, 15), r.end());
}

test "Range zero length" {
    const r = Range{ .start = 3, .len = 0 };
    try std.testing.expectEqual(@as(u32, 3), r.end());
}

test "Topology init and deinit" {
    const allocator = std.testing.allocator;
    var topo = try Topology.init(allocator, .{
        .n_atoms = 10,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 9,
    });
    defer topo.deinit();

    try std.testing.expectEqual(@as(usize, 10), topo.atoms.len);
    try std.testing.expectEqual(@as(usize, 2), topo.residues.len);
    try std.testing.expectEqual(@as(usize, 1), topo.chains.len);
    try std.testing.expectEqual(@as(usize, 9), topo.bonds.len);
}

test "Topology init empty" {
    const allocator = std.testing.allocator;
    var topo = try Topology.init(allocator, .{
        .n_atoms = 0,
        .n_residues = 0,
        .n_chains = 0,
        .n_bonds = 0,
    });
    defer topo.deinit();

    try std.testing.expectEqual(@as(usize, 0), topo.atoms.len);
}

test "Topology masses" {
    const allocator = std.testing.allocator;
    var topo = try Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.atoms[0] = .{
        .name = FixedString(4).fromSlice("N"),
        .element = .N,
        .residue_index = 0,
    };
    topo.atoms[1] = .{
        .name = FixedString(4).fromSlice("CA"),
        .element = .C,
        .residue_index = 0,
    };
    topo.atoms[2] = .{
        .name = FixedString(4).fromSlice("O"),
        .element = .O,
        .residue_index = 0,
    };

    const m = try topo.masses(allocator);
    defer allocator.free(m);

    try std.testing.expectEqual(@as(usize, 3), m.len);
    try std.testing.expectApproxEqAbs(@as(f64, 14.007), m[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 12.011), m[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 15.999), m[2], 0.001);
}

test "Topology masses unknown element is zero" {
    const allocator = std.testing.allocator;
    var topo = try Topology.init(allocator, .{
        .n_atoms = 1,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.atoms[0] = .{
        .name = FixedString(4).fromSlice("X"),
        .element = .X,
        .residue_index = 0,
    };

    const m = try topo.masses(allocator);
    defer allocator.free(m);

    try std.testing.expectEqual(@as(f64, 0.0), m[0]);
}

test "Frame init and deinit" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 100);
    defer frame.deinit();

    try std.testing.expectEqual(@as(usize, 100), frame.x.len);
    try std.testing.expectEqual(@as(usize, 100), frame.y.len);
    try std.testing.expectEqual(@as(usize, 100), frame.z.len);
    try std.testing.expectEqual(@as(usize, 100), frame.nAtoms());
    try std.testing.expectEqual(@as(?[3][3]f32, null), frame.box_vectors);
    try std.testing.expectEqual(@as(f32, 0.0), frame.time);
    try std.testing.expectEqual(@as(i32, 0), frame.step);
}

test "Frame init zeroed coordinates" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 5);
    defer frame.deinit();

    for (frame.x) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    for (frame.y) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    for (frame.z) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
}

test "Frame coordinate assignment" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 3);
    defer frame.deinit();

    frame.x[0] = 1.0;
    frame.y[0] = 2.0;
    frame.z[0] = 3.0;
    frame.time = 0.5;
    frame.step = 10;

    try std.testing.expectEqual(@as(f32, 1.0), frame.x[0]);
    try std.testing.expectEqual(@as(f32, 2.0), frame.y[0]);
    try std.testing.expectEqual(@as(f32, 3.0), frame.z[0]);
    try std.testing.expectEqual(@as(f32, 0.5), frame.time);
    try std.testing.expectEqual(@as(i32, 10), frame.step);
}

test "Frame box vectors" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 1);
    defer frame.deinit();

    const box: [3][3]f32 = .{
        .{ 50.0, 0.0, 0.0 },
        .{ 0.0, 50.0, 0.0 },
        .{ 0.0, 0.0, 50.0 },
    };
    frame.box_vectors = box;

    try std.testing.expect(frame.box_vectors != null);
    try std.testing.expectEqual(@as(f32, 50.0), frame.box_vectors.?[0][0]);
}

test "Frame init zero atoms" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 0);
    defer frame.deinit();

    try std.testing.expectEqual(@as(usize, 0), frame.nAtoms());
}

test "Trajectory nFrames and nAtoms" {
    const allocator = std.testing.allocator;

    const topo = try Topology.init(allocator, .{
        .n_atoms = 5,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 0,
    });

    const frames = try allocator.alloc(Frame, 3);
    for (frames) |*f| {
        f.* = try Frame.init(allocator, 5);
    }

    var traj = Trajectory{
        .topology = topo,
        .frames = frames,
        .allocator = allocator,
    };
    defer traj.deinit();

    try std.testing.expectEqual(@as(usize, 3), traj.nFrames());
    try std.testing.expectEqual(@as(usize, 5), traj.nAtoms());
}

test "ParseResult deinit frees both" {
    const allocator = std.testing.allocator;

    const topo = try Topology.init(allocator, .{
        .n_atoms = 4,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 3,
    });

    var frame = try Frame.init(allocator, 4);

    // Set up minimal valid topology so masses works
    for (topo.atoms, 0..) |*atom, i| {
        atom.* = .{
            .name = FixedString(4).fromSlice("C"),
            .element = .C,
            .residue_index = 0,
        };
        _ = i;
    }
    frame.time = 1.0;

    var result = ParseResult{
        .topology = topo,
        .frame = frame,
    };
    result.deinit(); // must not leak (testing allocator checks this)
}

test "Bond fields" {
    const b = Bond{ .atom_i = 0, .atom_j = 1 };
    try std.testing.expectEqual(@as(u32, 0), b.atom_i);
    try std.testing.expectEqual(@as(u32, 1), b.atom_j);
}

test "Atom fields" {
    const atom = Atom{
        .name = FixedString(4).fromSlice("CA"),
        .element = .C,
        .residue_index = 3,
    };
    try std.testing.expect(atom.name.eqlSlice("CA"));
    try std.testing.expectEqual(Element.C, atom.element);
    try std.testing.expectEqual(@as(u32, 3), atom.residue_index);
}

test "Residue fields" {
    const res = Residue{
        .name = FixedString(5).fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 10, .len = 7 },
        .resid = 42,
    };
    try std.testing.expect(res.name.eqlSlice("GLY"));
    try std.testing.expectEqual(@as(i32, 42), res.resid);
    try std.testing.expectEqual(@as(u32, 17), res.atom_range.end());
}

test "Chain fields" {
    const chain = Chain{
        .name = FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 100 },
    };
    try std.testing.expect(chain.name.eqlSlice("A"));
    try std.testing.expectEqual(@as(u32, 100), chain.residue_range.end());
}

test "Frame initView creates non-owning view" {
    const x = [_]f32{ 1.0, 2.0 };
    const y = [_]f32{ 3.0, 4.0 };
    const z = [_]f32{ 5.0, 6.0 };
    var frame = Frame.initView(&x, &y, &z);
    try std.testing.expect(!frame.owns);
    try std.testing.expectEqual(@as(usize, 2), frame.nAtoms());
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), frame.x[0], 1e-7);
    // deinit on view is a no-op — should not crash
    frame.deinit();
}
