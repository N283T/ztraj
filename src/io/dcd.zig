//! DCD trajectory reader for ztraj.
//!
//! DCD is a binary trajectory format used by NAMD and CHARMM. Coordinates are
//! stored in angstroms (no unit conversion needed). The format uses
//! Fortran-style record markers: each data block is bracketed by int32
//! byte-count values.
//!
//! Endianness is auto-detected from the first int32 (expected value: 84).
//! Both little-endian and big-endian files are supported.
//!
//! Adapted from freesasa-zig DCD reader, modified to yield ztraj Frame
//! values in SOA layout (separate x/y/z slices).

const std = @import("std");
const types = @import("../types.zig");

pub const DcdError = error{
    FileNotFound,
    InvalidMagic,
    EndOfFile,
    ReadError,
    BadFormat,
    OutOfMemory,
    FixedAtomsNotSupported,
    /// The file ends in the middle of a frame (cut-off or partially written).
    TruncatedFrame,
};

/// One AKMA time unit in picoseconds (CHARMM/NAMD convention).
pub const AKMA_TO_PS: f64 = 0.04888821;

/// CHARMM extension flags parsed from the DCD header.
const DCD_IS_CHARMM: i32 = 0x01;
const DCD_HAS_4DIMS: i32 = 0x02;
const DCD_HAS_EXTRA_BLOCK: i32 = 0x04;

/// Internal header information read once at open time.
const DcdHeader = struct {
    natoms: i32,
    nsets: i32,
    istart: i32,
    nsavc: i32,
    delta: f32,
    charmm: i32,
    reverse_endian: bool,
};

/// Streaming DCD reader that yields one Frame at a time.
///
/// Usage:
///
///   var reader = try DcdReader.open(io, allocator, "trajectory.dcd");
///   defer reader.deinit();
///
///   while (try reader.next()) |frame| {
///       _ = frame.x[0]; // coordinates already in angstroms
///   }
///
/// The reader reuses a single Frame buffer. The returned pointer is valid
/// until the next call to next() or deinit().
pub const DcdReader = struct {
    io: std.Io,
    file: std.Io.File,
    reader: std.Io.File.Reader,
    read_buffer: []u8,
    header: DcdHeader,
    /// Reused coordinate buffer (temporary, interleaved AOS) for reading.
    coord_buf: []f32,
    /// Reused SOA frame returned to callers.
    frame: types.Frame,
    frames_read: u32,
    /// Unit-cell angles (alpha, beta, gamma) in degrees of the frame most
    /// recently returned by next(). null when the file stores no unit cell.
    box_angles: ?[3]f32 = null,
    /// Unit-cell edge lengths (a, b, c) in angstroms of the frame most
    /// recently returned by next(). null when the file stores no unit cell.
    box_lengths: ?[3]f32 = null,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Open a DCD file for reading. Reads the header and allocates buffers.
    pub fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => DcdError.FileNotFound,
                else => DcdError.ReadError,
            };
        };
        errdefer file.close(io);

        const read_buffer = allocator.alloc(u8, 64 * 1024) catch return DcdError.OutOfMemory;
        errdefer allocator.free(read_buffer);

        var reader = file.reader(io, read_buffer);

        // Bootstrap reader to parse header (we need endianness first).
        var hdr = DcdHeader{
            .natoms = 0,
            .nsets = 0,
            .istart = 0,
            .nsavc = 0,
            .delta = 0,
            .charmm = 0,
            .reverse_endian = false,
        };

        try readHeader(&reader, &hdr);

        const n_atoms: usize = @intCast(hdr.natoms);

        const coord_buf = allocator.alloc(f32, n_atoms) catch return DcdError.OutOfMemory;
        errdefer allocator.free(coord_buf);

        var frame = types.Frame.init(allocator, n_atoms) catch return DcdError.OutOfMemory;
        errdefer frame.deinit();

        return Self{
            .io = io,
            .file = file,
            .reader = reader,
            .read_buffer = read_buffer,
            .header = hdr,
            .coord_buf = coord_buf,
            .frame = frame,
            .frames_read = 0,
            .allocator = allocator,
        };
    }

    /// Free all resources and close the file.
    pub fn deinit(self: *Self) void {
        self.frame.deinit();
        self.allocator.free(self.coord_buf);
        self.allocator.free(self.read_buffer);
        self.file.close(self.io);
    }

    /// Number of atoms in the trajectory.
    pub fn nAtoms(self: *const Self) u32 {
        return @intCast(self.header.natoms);
    }

    /// Total number of frames reported in the header.
    /// Note: some writers set this to 0 even for non-empty files.
    pub fn nFrames(self: *const Self) u32 {
        return @intCast(@max(0, self.header.nsets));
    }

    /// Number of frames returned by next() so far. When nFrames() is non-zero
    /// and next() returns null with fewer frames read, the file holds fewer
    /// frames than its header announces (some writers leave the header stale,
    /// so this is not treated as an error).
    pub fn framesRead(self: *const Self) u32 {
        return self.frames_read;
    }

    /// Read the next frame.
    ///
    /// Returns a pointer to the internal SOA frame on success.
    /// Returns null at a clean end of file (the file ends between frames).
    /// Returns DcdError.TruncatedFrame if the file ends inside a frame.
    /// Returns an error for any other read or format failure.
    /// The returned pointer is valid until the next call to next() or deinit().
    pub fn next(self: *Self) !?*const types.Frame {
        // End of file exactly at a frame boundary is the only clean end.
        _ = self.reader.interface.peekByte() catch |err| switch (err) {
            error.EndOfStream => return null,
            else => return DcdError.ReadError,
        };
        self.readFrameInto() catch |err| {
            if (err == DcdError.EndOfFile) return DcdError.TruncatedFrame;
            return err;
        };
        return &self.frame;
    }

    /// Internal implementation of frame reading.
    fn readFrameInto(self: *Self) !void {
        const natoms: usize = @intCast(self.header.natoms);

        // Compute step and time from frame metadata.
        const fc: i64 = @intCast(self.frames_read);
        const step_i64: i64 = @as(i64, self.header.istart) + fc * @as(i64, self.header.nsavc);
        const step: i32 = @intCast(@min(step_i64, std.math.maxInt(i32)));
        // DELTA is in AKMA units per step; convert (istart + i * nsavc) steps to ps.
        const time: f32 = @floatCast(@as(f64, @floatFromInt(step_i64)) *
            @as(f64, self.header.delta) * AKMA_TO_PS);

        // 1. Optionally read CHARMM unit cell block.
        var box_vectors: ?[3][3]f32 = null;
        var box_lengths: ?[3]f32 = null;
        var box_angles: ?[3]f32 = null;
        if ((self.header.charmm & DCD_IS_CHARMM) != 0 and
            (self.header.charmm & DCD_HAS_EXTRA_BLOCK) != 0)
        {
            const uc_marker = self.readInt32() catch |err| {
                return switch (err) {
                    DcdError.EndOfFile => DcdError.EndOfFile,
                    else => err,
                };
            };
            if (uc_marker == 48) {
                // CHARMM stores 6 f64 values: a, gamma, b, beta, alpha, c
                var uc: [6]f64 = undefined;
                for (0..6) |i| {
                    uc[i] = try self.readFloat64();
                }
                // Slots are A, gamma, B, beta, alpha, C. Newer CHARMM stores the
                // cosines of the angles; older files and NAMD store degrees.
                const ang = unitCellAngles(uc[4], uc[3], uc[1]);
                const lengths = [3]f32{ @floatCast(uc[0]), @floatCast(uc[2]), @floatCast(uc[5]) };
                box_lengths = lengths;
                box_angles = ang;
                box_vectors = triclinicVectors(lengths, ang);
            } else {
                // Unknown block — skip it.
                self.reader.seekBy(@intCast(uc_marker)) catch return DcdError.ReadError;
            }
            _ = try self.readRawInt32();
        }

        // 2. Read X, Y, Z coordinate blocks into the frame SOA arrays.
        try self.readCoordBlockInto(self.frame.x, natoms);
        try self.readCoordBlockInto(self.frame.y, natoms);
        try self.readCoordBlockInto(self.frame.z, natoms);

        // 3. Skip optional 4th-dimension block.
        if ((self.header.charmm & DCD_IS_CHARMM) != 0 and
            (self.header.charmm & DCD_HAS_4DIMS) != 0)
        {
            const dim4_marker = try self.readInt32();
            self.reader.seekBy(@intCast(dim4_marker)) catch return DcdError.ReadError;
            _ = try self.readRawInt32();
        }

        self.frame.step = step;
        self.frame.time = time;
        self.frame.box_vectors = box_vectors;
        self.box_lengths = box_lengths;
        self.box_angles = box_angles;
        self.frames_read += 1;
    }

    /// Read a single coordinate block (X, Y, or Z) directly into a SOA slice.
    ///
    /// DCD stores each component as float32[natoms] flanked by Fortran markers.
    fn readCoordBlockInto(self: *Self, dest: []f32, natoms: usize) !void {
        const expected_size: i32 = @intCast(natoms * 4);
        const lead_marker = self.readInt32() catch |err| {
            return switch (err) {
                DcdError.EndOfFile => DcdError.EndOfFile,
                else => err,
            };
        };
        if (lead_marker != expected_size) return DcdError.BadFormat;

        if (self.header.reverse_endian) {
            // Read and byte-swap each float individually.
            var raw: [4]u8 = undefined;
            for (0..natoms) |i| {
                self.reader.interface.readSliceAll(&raw) catch |err| switch (err) {
                    error.EndOfStream => return DcdError.EndOfFile,
                    else => return DcdError.ReadError,
                };
                const swapped = @byteSwap(std.mem.readInt(u32, &raw, .little));
                dest[i] = @bitCast(swapped);
            }
        } else {
            // Read all floats in a single call via the reusable coord_buf.
            const byte_count = natoms * 4;
            const tmp = self.coord_buf[0..natoms];
            const tmp_bytes: [*]u8 = @ptrCast(tmp.ptr);
            self.reader.interface.readSliceAll(tmp_bytes[0..byte_count]) catch |err| switch (err) {
                error.EndOfStream => return DcdError.EndOfFile,
                else => return DcdError.ReadError,
            };
            @memcpy(dest, tmp);
        }

        const trail_marker = try self.readInt32();
        if (trail_marker != expected_size) return DcdError.BadFormat;
    }

    // ========================================================================
    // Low-level I/O helpers
    // ========================================================================

    /// Read a raw int32 as little-endian regardless of endianness flag.
    /// Used for the very first int (before we know endianness).
    fn readRawInt32(self: *Self) !i32 {
        var buf: [4]u8 = undefined;
        self.reader.interface.readSliceAll(&buf) catch |err| switch (err) {
            error.EndOfStream => return DcdError.EndOfFile,
            else => return DcdError.ReadError,
        };
        return @bitCast(std.mem.readInt(u32, &buf, .little));
    }

    /// Read int32 with endian handling.
    fn readInt32(self: *Self) !i32 {
        var buf: [4]u8 = undefined;
        self.reader.interface.readSliceAll(&buf) catch |err| switch (err) {
            error.EndOfStream => return DcdError.EndOfFile,
            else => return DcdError.ReadError,
        };
        if (self.header.reverse_endian) {
            return @bitCast(std.mem.readInt(u32, &buf, .big));
        }
        return @bitCast(std.mem.readInt(u32, &buf, .little));
    }

    /// Read float64 with endian handling.
    fn readFloat64(self: *Self) !f64 {
        var buf: [8]u8 = undefined;
        self.reader.interface.readSliceAll(&buf) catch |err| switch (err) {
            error.EndOfStream => return DcdError.EndOfFile,
            else => return DcdError.ReadError,
        };
        if (self.header.reverse_endian) {
            return @bitCast(std.mem.readInt(u64, &buf, .big));
        }
        return @bitCast(std.mem.readInt(u64, &buf, .little));
    }
};

/// Convert the three stored angle values (alpha, beta, gamma) to degrees.
/// Values with magnitude <= 1 are cosines (newer CHARMM); others are degrees.
fn unitCellAngles(alpha: f64, beta: f64, gamma: f64) [3]f32 {
    const are_cosines = @abs(alpha) <= 1.0 and @abs(beta) <= 1.0 and @abs(gamma) <= 1.0;
    if (are_cosines) {
        const to_deg = 180.0 / std.math.pi;
        return .{
            @floatCast(std.math.acos(alpha) * to_deg),
            @floatCast(std.math.acos(beta) * to_deg),
            @floatCast(std.math.acos(gamma) * to_deg),
        };
    }
    return .{ @floatCast(alpha), @floatCast(beta), @floatCast(gamma) };
}

/// Box vectors (rows a, b, c) from edge lengths and angles in degrees.
/// Orthogonal cells give a diagonal matrix.
fn triclinicVectors(l: [3]f32, ang: [3]f32) [3][3]f32 {
    const to_rad = std.math.pi / 180.0;
    const ca: f64 = @cos(@as(f64, ang[0]) * to_rad);
    const cb: f64 = @cos(@as(f64, ang[1]) * to_rad);
    const cg: f64 = @cos(@as(f64, ang[2]) * to_rad);
    const sg: f64 = @sin(@as(f64, ang[2]) * to_rad);
    // Exact zeros for right angles (cos(90 deg) is not exactly 0 in floating point).
    const ca_ = if (ang[0] == 90.0) 0.0 else ca;
    const cb_ = if (ang[1] == 90.0) 0.0 else cb;
    const cg_ = if (ang[2] == 90.0) 0.0 else cg;
    const sg_ = if (ang[2] == 90.0) 1.0 else sg;
    const a: f64 = l[0];
    const b: f64 = l[1];
    const c: f64 = l[2];
    const cx = c * cb_;
    const cy = if (sg_ != 0) c * (ca_ - cb_ * cg_) / sg_ else 0.0;
    const cz2 = c * c - cx * cx - cy * cy;
    const cz = if (cz2 > 0) @sqrt(cz2) else 0.0;
    return .{
        .{ @floatCast(a), 0.0, 0.0 },
        .{ @floatCast(b * cg_), @floatCast(b * sg_), 0.0 },
        .{ @floatCast(cx), @floatCast(cy), @floatCast(cz) },
    };
}

// ============================================================================
// Header parser (free function, operates on a std.Io.File.Reader)
// ============================================================================

fn readHeader(reader: *std.Io.File.Reader, hdr: *DcdHeader) !void {
    // Helper closures — inline so we can use the reader and hdr from the outer scope.
    const readRawInt = struct {
        fn call(r: *std.Io.File.Reader) !i32 {
            var buf: [4]u8 = undefined;
            r.interface.readSliceAll(&buf) catch |err| switch (err) {
                error.EndOfStream => return DcdError.EndOfFile,
                else => return DcdError.ReadError,
            };
            return @bitCast(std.mem.readInt(u32, &buf, .little));
        }
    }.call;

    const intFromBuf = struct {
        fn call(buf: *const [4]u8, swap: bool) i32 {
            if (swap) return @bitCast(std.mem.readInt(u32, buf, .big));
            return @bitCast(std.mem.readInt(u32, buf, .little));
        }
    }.call;

    const floatFromBuf = struct {
        fn call(buf: *const [4]u8, swap: bool) f32 {
            if (swap) return @bitCast(std.mem.readInt(u32, buf, .big));
            return @bitCast(std.mem.readInt(u32, buf, .little));
        }
    }.call;

    const doubleFromBuf = struct {
        fn call(buf: *const [8]u8, swap: bool) f64 {
            if (swap) return @bitCast(std.mem.readInt(u64, buf, .big));
            return @bitCast(std.mem.readInt(u64, buf, .little));
        }
    }.call;

    // 1. Read first int32 — should be 84.
    var first_int = try readRawInt(reader);
    if (first_int != 84) {
        first_int = @bitCast(@byteSwap(@as(u32, @bitCast(first_int))));
        if (first_int == 84) {
            hdr.reverse_endian = true;
        } else {
            return DcdError.InvalidMagic;
        }
    }
    const swap = hdr.reverse_endian;

    // 2. Read 84-byte header block.
    var hdrbuf: [84]u8 = undefined;
    reader.interface.readSliceAll(&hdrbuf) catch |err| switch (err) {
        error.EndOfStream => return DcdError.BadFormat,
        else => return DcdError.ReadError,
    };

    // Check "CORD" magic.
    if (hdrbuf[0] != 'C' or hdrbuf[1] != 'O' or hdrbuf[2] != 'R' or hdrbuf[3] != 'D') {
        return DcdError.InvalidMagic;
    }

    // CHARMM detection: last int32 in header (offset 80) nonzero means CHARMM.
    const charmm_ver = intFromBuf(hdrbuf[80..84], swap);
    if (charmm_ver != 0) {
        hdr.charmm = DCD_IS_CHARMM;
        if (intFromBuf(hdrbuf[44..48], swap) != 0) {
            hdr.charmm |= DCD_HAS_EXTRA_BLOCK;
        }
        if (intFromBuf(hdrbuf[48..52], swap) == 1) {
            hdr.charmm |= DCD_HAS_4DIMS;
        }
    }

    hdr.nsets = intFromBuf(hdrbuf[4..8], swap);
    hdr.istart = intFromBuf(hdrbuf[8..12], swap);
    hdr.nsavc = intFromBuf(hdrbuf[12..16], swap);

    const namnf = intFromBuf(hdrbuf[36..40], swap);

    // DELTA: float32 for CHARMM, float64 for X-PLOR.
    if ((hdr.charmm & DCD_IS_CHARMM) != 0) {
        hdr.delta = floatFromBuf(hdrbuf[40..44], swap);
    } else {
        hdr.delta = @floatCast(doubleFromBuf(hdrbuf[40..48], swap));
    }

    // Trailing marker of block 1.
    const readIntSwap = struct {
        fn call(r: *std.Io.File.Reader, s: bool) !i32 {
            var buf: [4]u8 = undefined;
            r.interface.readSliceAll(&buf) catch |err| switch (err) {
                error.EndOfStream => return DcdError.EndOfFile,
                else => return DcdError.ReadError,
            };
            if (s) return @bitCast(std.mem.readInt(u32, &buf, .big));
            return @bitCast(std.mem.readInt(u32, &buf, .little));
        }
    }.call;

    const trail1 = try readIntSwap(reader, swap);
    if (trail1 != 84) return DcdError.BadFormat;

    // 3. Title block.
    const title_block_size = try readIntSwap(reader, swap);
    if (title_block_size < 4 or @mod(title_block_size - 4, 80) != 0) return DcdError.BadFormat;

    const ntitle = try readIntSwap(reader, swap);
    if (ntitle < 0) return DcdError.BadFormat;

    const title_bytes: usize = @intCast(ntitle * 80);
    if (title_bytes > 0) {
        reader.seekBy(@intCast(title_bytes)) catch return DcdError.ReadError;
    }
    _ = try readRawInt(reader); // trailing marker

    // 4. Natoms block.
    const natom_marker = try readIntSwap(reader, swap);
    if (natom_marker != 4) return DcdError.BadFormat;

    hdr.natoms = try readIntSwap(reader, swap);
    if (hdr.natoms <= 0) return DcdError.BadFormat;

    const natom_trail = try readIntSwap(reader, swap);
    if (natom_trail != 4) return DcdError.BadFormat;

    // Fixed atoms are unsupported.
    if (namnf != 0) return DcdError.FixedAtomsNotSupported;
}

// ============================================================================
// Tests
// ============================================================================

fn testIo() std.Io {
    const t = struct {
        var threaded: std.Io.Threaded = .init_single_threaded;
    };
    return t.threaded.io();
}

test "DcdReader open returns FileNotFound for missing path" {
    const allocator = std.testing.allocator;
    const result = DcdReader.open(testIo(), allocator, "nonexistent.dcd");
    try std.testing.expectError(DcdError.FileNotFound, result);
}

test "DcdReader open returns FileNotFound for nonexistent path" {
    const allocator = std.testing.allocator;
    const err = DcdReader.open(testIo(), allocator, "/no/such/file/trajectory.dcd");
    try std.testing.expectError(DcdError.FileNotFound, err);
}

test "DcdReader reads existing DCD file" {
    const allocator = std.testing.allocator;

    var reader = DcdReader.open(testIo(), allocator, "test_data/1l2y.dcd") catch |err| {
        if (err == DcdError.FileNotFound) return; // Skip if test data not present.
        return err;
    };
    defer reader.deinit();

    // 1l2y has 304 atoms.
    try std.testing.expectEqual(@as(u32, 304), reader.nAtoms());

    // Read first frame.
    const frame = (try reader.next()) orelse return error.TestUnexpectedNull;

    try std.testing.expectEqual(@as(usize, 304), frame.x.len);
    try std.testing.expectEqual(@as(usize, 304), frame.y.len);
    try std.testing.expectEqual(@as(usize, 304), frame.z.len);

    // DCD stores angstroms. XTC atom[0] = [-0.8901, 0.4127, -0.0555] nm
    // → [-8.901, 4.127, -0.555] angstroms.
    const tol: f32 = 0.05;
    try std.testing.expectApproxEqAbs(@as(f32, -8.901), frame.x[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 4.127), frame.y[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, -0.555), frame.z[0], tol);
}

test "DcdReader reads all frames from DCD file" {
    const allocator = std.testing.allocator;

    var reader = DcdReader.open(testIo(), allocator, "test_data/1l2y.dcd") catch |err| {
        if (err == DcdError.FileNotFound) return;
        return err;
    };
    defer reader.deinit();

    var count: u32 = 0;
    while (try reader.next()) |_| {
        count += 1;
    }

    // 1l2y.dcd has 38 frames.
    try std.testing.expectEqual(@as(u32, 38), count);
}

// ============================================================================
// Synthetic-file tests (no DCD fixture ships with the repo)
// ============================================================================

const TestDcd = struct {
    nframes: u32 = 3,
    natoms: u32 = 2,
    istart: i32 = 100,
    nsavc: i32 = 10,
    delta: f32 = 0.5,
    /// Six unit-cell values per frame (A, gamma, B, beta, alpha, C); null = no unit cell.
    cell: ?[6]f64 = null,
};

fn putI32(list: *std.ArrayList(u8), a: std.mem.Allocator, v: i32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(i32, &b, v, .little);
    try list.appendSlice(a, &b);
}

fn buildTestDcd(a: std.mem.Allocator, p: TestDcd) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var hdr = [_]u8{0} ** 84;
    @memcpy(hdr[0..4], "CORD");
    std.mem.writeInt(i32, hdr[4..8], @intCast(p.nframes), .little);
    std.mem.writeInt(i32, hdr[8..12], p.istart, .little);
    std.mem.writeInt(i32, hdr[12..16], p.nsavc, .little);
    std.mem.writeInt(u32, hdr[40..44], @bitCast(p.delta), .little);
    std.mem.writeInt(i32, hdr[44..48], if (p.cell != null) 1 else 0, .little);
    std.mem.writeInt(i32, hdr[80..84], 24, .little); // CHARMM version
    try putI32(&out, a, 84);
    try out.appendSlice(a, &hdr);
    try putI32(&out, a, 84);
    try putI32(&out, a, 84); // title block: ntitle + 1 line
    try putI32(&out, a, 1);
    try out.appendNTimes(a, ' ', 80);
    try putI32(&out, a, 84);
    try putI32(&out, a, 4);
    try putI32(&out, a, @intCast(p.natoms));
    try putI32(&out, a, 4);
    for (0..p.nframes) |f| {
        if (p.cell) |c| {
            try putI32(&out, a, 48);
            for (c) |v| {
                var b: [8]u8 = undefined;
                std.mem.writeInt(u64, &b, @bitCast(v), .little);
                try out.appendSlice(a, &b);
            }
            try putI32(&out, a, 48);
        }
        for (0..3) |axis| {
            try putI32(&out, a, @intCast(p.natoms * 4));
            for (0..p.natoms) |i| {
                const v: f32 = @floatFromInt(f * 100 + axis * 10 + i);
                var b: [4]u8 = undefined;
                std.mem.writeInt(u32, &b, @bitCast(v), .little);
                try out.appendSlice(a, &b);
            }
            try putI32(&out, a, @intCast(p.natoms * 4));
        }
    }
    return out.toOwnedSlice(a);
}

fn openTestDcd(tmp: *std.testing.TmpDir, bytes: []const u8) !DcdReader {
    const io = testIo();
    try tmp.dir.writeFile(io, .{ .sub_path = "t.dcd", .data = bytes });
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/t.dcd", .{tmp.sub_path});
    return DcdReader.open(io, std.testing.allocator, path);
}

test "DcdReader clean end between frames returns null" {
    const a = std.testing.allocator;
    const bytes = try buildTestDcd(a, .{});
    defer a.free(bytes);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try openTestDcd(&tmp, bytes);
    defer r.deinit();
    var n: u32 = 0;
    while (try r.next()) |_| n += 1;
    try std.testing.expectEqual(@as(u32, 3), n);
    try std.testing.expectEqual(@as(u32, 3), r.framesRead());
}

test "DcdReader reports TruncatedFrame for a file cut inside the last frame" {
    const a = std.testing.allocator;
    const bytes = try buildTestDcd(a, .{});
    defer a.free(bytes);
    // Frame size: 3 * (4 + 8 + 4) = 48 bytes. Cut at several offsets inside the last frame.
    for ([_]usize{ 1, 3, 4, 10, 30, 47 }) |cut| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try openTestDcd(&tmp, bytes[0 .. bytes.len - cut]);
        defer r.deinit();
        try std.testing.expect((try r.next()) != null);
        try std.testing.expect((try r.next()) != null);
        try std.testing.expectError(DcdError.TruncatedFrame, r.next());
    }
}

test "DcdReader time is (istart + i*nsavc) * delta in ps" {
    const a = std.testing.allocator;
    const bytes = try buildTestDcd(a, .{ .istart = 100, .nsavc = 10, .delta = 0.5 });
    defer a.free(bytes);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try openTestDcd(&tmp, bytes);
    defer r.deinit();
    // step 100 -> 100 * 0.5 * 0.04888821 = 2.4444105 ps
    const f0 = (try r.next()).?;
    try std.testing.expectEqual(@as(i32, 100), f0.step);
    try std.testing.expectApproxEqAbs(@as(f32, 2.4444105), f0.time, 1e-5);
    const f1 = (try r.next()).?;
    try std.testing.expectEqual(@as(i32, 110), f1.step);
    try std.testing.expectApproxEqAbs(@as(f32, 2.6888516), f1.time, 1e-5);
    const f2 = (try r.next()).?;
    try std.testing.expectEqual(@as(i32, 120), f2.step);
    try std.testing.expectApproxEqAbs(@as(f32, 2.9332926), f2.time, 1e-5);
}

test "DcdReader exposes unit-cell angles in degrees" {
    const a = std.testing.allocator;
    const bytes = try buildTestDcd(a, .{ .cell = .{ 30.0, 70.0, 40.0, 80.0, 60.0, 50.0 } });
    defer a.free(bytes);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try openTestDcd(&tmp, bytes);
    defer r.deinit();
    const f = (try r.next()).?;
    try std.testing.expectEqual([3]f32{ 30, 40, 50 }, r.box_lengths.?);
    try std.testing.expectEqual([3]f32{ 60, 80, 70 }, r.box_angles.?);
    const bv = f.box_vectors.?;
    // Row lengths equal the cell edges; angle between a and b is gamma.
    const lb = @sqrt(bv[1][0] * bv[1][0] + bv[1][1] * bv[1][1] + bv[1][2] * bv[1][2]);
    const lc = @sqrt(bv[2][0] * bv[2][0] + bv[2][1] * bv[2][1] + bv[2][2] * bv[2][2]);
    try std.testing.expectApproxEqAbs(@as(f32, 40), lb, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 50), lc, 1e-3);
    const cos_g = (bv[0][0] * bv[1][0]) / (30.0 * 40.0);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 70) * std.math.pi / 180.0), cos_g, 1e-5);
}

test "DcdReader converts cosine-encoded unit-cell angles to degrees" {
    const a = std.testing.allocator;
    // cos(60) = 0.5, cos(90) = 0, cos(120) = -0.5 (alpha=60, beta=90, gamma=120)
    const bytes = try buildTestDcd(a, .{ .cell = .{ 10.0, -0.5, 10.0, 0.0, 0.5, 10.0 } });
    defer a.free(bytes);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try openTestDcd(&tmp, bytes);
    defer r.deinit();
    _ = (try r.next()).?;
    const ang = r.box_angles.?;
    try std.testing.expectApproxEqAbs(@as(f32, 60), ang[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 90), ang[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 120), ang[2], 1e-3);
}

test "DcdReader orthogonal cell keeps a diagonal box" {
    const a = std.testing.allocator;
    const bytes = try buildTestDcd(a, .{ .cell = .{ 30.0, 90.0, 40.0, 90.0, 90.0, 50.0 } });
    defer a.free(bytes);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try openTestDcd(&tmp, bytes);
    defer r.deinit();
    const f = (try r.next()).?;
    try std.testing.expectEqual([3][3]f32{ .{ 30, 0, 0 }, .{ 0, 40, 0 }, .{ 0, 0, 50 } }, f.box_vectors.?);
    try std.testing.expectEqual([3]f32{ 90, 90, 90 }, r.box_angles.?);
}

test "DcdReader without unit cell has no box angles" {
    const a = std.testing.allocator;
    const bytes = try buildTestDcd(a, .{});
    defer a.free(bytes);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try openTestDcd(&tmp, bytes);
    defer r.deinit();
    _ = (try r.next()).?;
    try std.testing.expect(r.box_angles == null);
    try std.testing.expect(r.box_lengths == null);
}
