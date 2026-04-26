//! XTC trajectory reader for ztraj.
//!
//! Wraps the zxdrfile XtcReader to yield Frame values in ztraj's SOA layout.
//! Coordinates are converted from nanometers (XTC native) to angstroms (*10.0)
//! at read time. The conversion happens at the I/O boundary; all internal
//! representations use angstroms.

const std = @import("std");
const types = @import("../types.zig");
const zxdrfile = @import("zxdrfile");

const XtcReaderInner = zxdrfile.XtcReader;
const XtcWriterInner = zxdrfile.XtcWriter;
const XtcError = zxdrfile.XtcError;

pub const XtcReadError = error{
    FileNotFound,
    InvalidMagic,
    EndOfFile,
    ReadError,
    DecompressionError,
    OutOfMemory,
};

/// Streaming XTC reader that yields one Frame at a time.
///
/// Usage:
///
///   var reader = try XtcReader.open(allocator, "trajectory.xtc");
///   defer reader.deinit();
///
///   while (try reader.next()) |frame| {
///       // frame is valid until the next call to next() or deinit()
///       _ = frame.x[0];
///   }
///
/// The reader reuses a single Frame buffer to avoid per-frame allocations.
/// Coordinates are in angstroms (nm * 10.0 conversion applied at read time).
pub const XtcReader = struct {
    inner: XtcReaderInner,
    /// Reused frame buffer. Valid until the next call to next() or deinit().
    frame: types.Frame,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Open an XTC file for reading.
    ///
    /// Allocates a single Frame buffer sized to the number of atoms in the file.
    /// Returns error.FileNotFound if the path does not exist.
    pub fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
        var inner = XtcReaderInner.open(io, allocator, path) catch |err| {
            return switch (err) {
                XtcError.FileNotFound => XtcReadError.FileNotFound,
                XtcError.InvalidMagic => XtcReadError.InvalidMagic,
                XtcError.OutOfMemory => XtcReadError.OutOfMemory,
                else => XtcReadError.ReadError,
            };
        };
        errdefer inner.close();

        const n_atoms: usize = @intCast(inner.natoms);
        const frame = types.Frame.init(allocator, n_atoms) catch |err| {
            inner.close();
            return err;
        };

        return Self{
            .inner = inner,
            .frame = frame,
            .allocator = allocator,
        };
    }

    /// Read the next frame.
    ///
    /// Returns a pointer to the internal frame buffer on success.
    /// Returns null at genuine end of file.
    /// Returns an error for any actual read or decompression failure.
    /// The returned pointer is valid until the next call to next() or deinit().
    pub fn next(self: *Self) !?*const types.Frame {
        var xtc_frame = self.inner.readFrame() catch |err| {
            if (err == XtcError.EndOfFile) return null;
            return switch (err) {
                XtcError.InvalidMagic => XtcReadError.InvalidMagic,
                XtcError.OutOfMemory => XtcReadError.OutOfMemory,
                XtcError.DecompressionError => XtcReadError.DecompressionError,
                else => XtcReadError.ReadError,
            };
        };
        defer xtc_frame.deinit(self.allocator);

        const n_atoms: usize = @intCast(self.inner.natoms);

        // Convert AOS (x0,y0,z0, x1,y1,z1, ...) in nm to SOA in angstroms.
        for (0..n_atoms) |i| {
            self.frame.x[i] = xtc_frame.coords[i * 3 + 0] * 10.0;
            self.frame.y[i] = xtc_frame.coords[i * 3 + 1] * 10.0;
            self.frame.z[i] = xtc_frame.coords[i * 3 + 2] * 10.0;
        }

        self.frame.time = xtc_frame.time;
        self.frame.step = xtc_frame.step;

        // Convert box vectors from nm to angstroms.
        var box: [3][3]f32 = undefined;
        for (0..3) |i| {
            for (0..3) |j| {
                box[i][j] = xtc_frame.box[i][j] * 10.0;
            }
        }
        self.frame.box_vectors = box;

        return &self.frame;
    }

    /// Free the Frame buffer and close the underlying file.
    pub fn deinit(self: *Self) void {
        self.frame.deinit();
        self.inner.close();
    }

    /// Number of atoms in the trajectory.
    pub fn nAtoms(self: *const Self) u32 {
        return @intCast(self.inner.natoms);
    }
};

/// Streaming XTC writer that writes one Frame at a time.
///
/// Usage:
///
///   var writer = try XtcWriter.open(allocator, "trajectory.xtc", n_atoms);
///   defer writer.deinit();
///
///   try writer.writeFrame(frame);
///   try writer.close();
///
/// Coordinates are expected in angstroms and are converted to nanometers
/// (/ 10.0) before writing. Box vectors are also converted from angstroms to nm.
pub const XtcWriter = struct {
    inner: XtcWriterInner,
    coords_buf: []f32,
    allocator: std.mem.Allocator,
    closed: bool = false,

    const Self = @This();

    pub fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8, n_atoms: usize) !Self {
        const natoms_i: i32 = @intCast(n_atoms);
        var inner = try XtcWriterInner.open(io, allocator, path, natoms_i, .write);
        errdefer inner.close() catch {};
        const coords_buf = try allocator.alloc(f32, n_atoms * 3);
        return Self{ .inner = inner, .coords_buf = coords_buf, .allocator = allocator };
    }

    /// Convert a ztraj Frame (Å, SOA) to XTC format (nm, AOS) and write it.
    pub fn writeFrame(self: *Self, frame: types.Frame) !void {
        const n = frame.x.len;
        for (0..n) |i| {
            self.coords_buf[i * 3 + 0] = frame.x[i] / 10.0;
            self.coords_buf[i * 3 + 1] = frame.y[i] / 10.0;
            self.coords_buf[i * 3 + 2] = frame.z[i] / 10.0;
        }
        var box: [3][3]f32 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
        if (frame.box_vectors) |bv| {
            for (0..3) |r| {
                for (0..3) |c| {
                    box[r][c] = bv[r][c] / 10.0;
                }
            }
        }
        try self.inner.writeFrame(.{
            .step = frame.step,
            .time = frame.time,
            .box = box,
            .coords = self.coords_buf,
            .precision = 1000.0,
        });
    }

    /// Flush and close the file, then free all resources.
    pub fn close(self: *Self) !void {
        defer {
            self.allocator.free(self.coords_buf);
            self.coords_buf = &.{};
            self.closed = true;
        }
        try self.inner.close();
    }

    /// Best-effort cleanup. Frees buffer and closes inner if not already closed.
    pub fn deinit(self: *Self) void {
        if (self.coords_buf.len > 0) {
            self.allocator.free(self.coords_buf);
            self.coords_buf = &.{};
        }
        if (!self.closed) {
            self.inner.close() catch {};
            self.closed = true;
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "XtcReader compiles and can be deinitialized with no file" {
    // Verify the struct layout is correct without requiring a real XTC file.
    // Structural integrity only — open() will fail gracefully on missing file.
    const allocator = std.testing.allocator;

    const result = XtcReader.open(allocator, "nonexistent_file.xtc");
    try std.testing.expectError(XtcReadError.FileNotFound, result);
}

test "XtcReader open error is FileNotFound for missing path" {
    const allocator = std.testing.allocator;
    const err = XtcReader.open(allocator, "/no/such/path/trajectory.xtc");
    try std.testing.expectError(XtcReadError.FileNotFound, err);
}
