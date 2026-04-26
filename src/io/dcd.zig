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
};

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

    /// Read the next frame.
    ///
    /// Returns a pointer to the internal SOA frame on success.
    /// Returns null at genuine end of file.
    /// Returns an error for any actual read or format failure.
    /// The returned pointer is valid until the next call to next() or deinit().
    pub fn next(self: *Self) !?*const types.Frame {
        self.readFrameInto() catch |err| {
            if (err == DcdError.EndOfFile) return null;
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
        const time: f32 = @as(f32, @floatFromInt(self.frames_read)) * self.header.delta;

        // 1. Optionally read CHARMM unit cell block.
        var box_vectors: ?[3][3]f32 = null;
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
                // Build an orthogonal box from a, b, c lengths (indices 0, 2, 5).
                const a: f32 = @floatCast(uc[0]);
                const b: f32 = @floatCast(uc[2]);
                const c: f32 = @floatCast(uc[5]);
                box_vectors = .{
                    .{ a, 0.0, 0.0 },
                    .{ 0.0, b, 0.0 },
                    .{ 0.0, 0.0, c },
                };
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
