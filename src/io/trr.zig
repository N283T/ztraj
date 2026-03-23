//! TRR trajectory reader for ztraj.
//!
//! Wraps the zxdrfile TrrReader to yield Frame values in ztraj's SOA layout.
//! TRR files store coordinates, velocities, and forces (all optional).
//! Only coordinates are converted; velocities and forces are discarded.
//! Coordinates are converted from nanometers (TRR native) to angstroms (*10.0)
//! at read time.

const std = @import("std");
const types = @import("../types.zig");
const zxdrfile = @import("zxdrfile");

const TrrReaderInner = zxdrfile.TrrReader;
const TrrWriterInner = zxdrfile.TrrWriter;
const TrrError = zxdrfile.TrrError;

pub const TrrReadError = error{
    FileNotFound,
    InvalidMagic,
    InvalidHeader,
    EndOfFile,
    ReadError,
    OutOfMemory,
};

/// Streaming TRR reader that yields one Frame at a time.
///
/// Usage:
///
///   var reader = try TrrReader.open(allocator, "trajectory.trr");
///   defer reader.deinit();
///
///   while (try reader.next()) |frame| {
///       _ = frame.x[0];
///   }
///
/// The reader reuses a single Frame buffer to avoid per-frame allocations.
/// Coordinates are in angstroms (nm * 10.0 conversion applied at read time).
pub const TrrReader = struct {
    inner: TrrReaderInner,
    frame: types.Frame,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Open a TRR file for reading.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Self {
        var inner = TrrReaderInner.open(allocator, path) catch |err| {
            return switch (err) {
                TrrError.FileNotFound => TrrReadError.FileNotFound,
                TrrError.InvalidMagic => TrrReadError.InvalidMagic,
                TrrError.InvalidHeader => TrrReadError.InvalidHeader,
                TrrError.OutOfMemory => TrrReadError.OutOfMemory,
                else => TrrReadError.ReadError,
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
    /// Returns null at end of file.
    /// Skips frames without coordinates (velocity-only or force-only).
    pub fn next(self: *Self) !?*const types.Frame {
        while (true) {
            var trr_frame = self.inner.readFrame() catch |err| {
                if (err == TrrError.EndOfFile) return null;
                return switch (err) {
                    TrrError.InvalidMagic => TrrReadError.InvalidMagic,
                    TrrError.InvalidHeader => TrrReadError.InvalidHeader,
                    TrrError.OutOfMemory => TrrReadError.OutOfMemory,
                    else => TrrReadError.ReadError,
                };
            };
            defer trr_frame.deinit(self.allocator);

            // Skip frames without coordinates
            const coords = trr_frame.coords orelse continue;

            const n_atoms: usize = @intCast(self.inner.natoms);

            // Convert AOS nm to SOA angstroms
            for (0..n_atoms) |i| {
                self.frame.x[i] = coords[i * 3 + 0] * 10.0;
                self.frame.y[i] = coords[i * 3 + 1] * 10.0;
                self.frame.z[i] = coords[i * 3 + 2] * 10.0;
            }

            self.frame.time = trr_frame.time;
            self.frame.step = trr_frame.step;

            // Convert box vectors from nm to angstroms
            var box: [3][3]f32 = undefined;
            for (0..3) |i| {
                for (0..3) |j| {
                    box[i][j] = trr_frame.box[i][j] * 10.0;
                }
            }
            self.frame.box_vectors = box;

            return &self.frame;
        }
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

/// Streaming TRR writer that writes one Frame at a time.
///
/// Usage:
///
///   var writer = try TrrWriter.open(allocator, "trajectory.trr", n_atoms);
///   defer writer.deinit();
///
///   try writer.writeFrame(frame);
///   try writer.close();
///
/// Coordinates are expected in angstroms and are converted to nanometers
/// (/ 10.0) before writing. Box vectors are also converted from angstroms to nm.
/// Only coordinates are written (has_x = true); velocities and forces are omitted.
pub const TrrWriter = struct {
    inner: TrrWriterInner,
    coords_buf: []f32,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Open a TRR file for writing.
    ///
    /// Allocates an internal AOS conversion buffer sized to n_atoms * 3.
    pub fn open(allocator: std.mem.Allocator, path: []const u8, n_atoms: usize) !Self {
        const natoms_i: i32 = @intCast(n_atoms);
        var inner = try TrrWriterInner.open(allocator, path, natoms_i, .write);
        errdefer inner.close() catch {};
        const coords_buf = try allocator.alloc(f32, n_atoms * 3);
        return Self{ .inner = inner, .coords_buf = coords_buf, .allocator = allocator };
    }

    /// Write a single frame.
    ///
    /// Converts SOA angstroms to AOS nanometers and box vectors from angstroms to nm.
    /// Velocities and forces are not written.
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
            .lambda = 0.0,
            .box = box,
            .has_x = true,
            .has_v = false,
            .has_f = false,
            .coords = self.coords_buf,
            .velocities = null,
            .forces = null,
        });
    }

    /// Flush and close the underlying file, then free the conversion buffer.
    pub fn close(self: *Self) !void {
        self.allocator.free(self.coords_buf);
        try self.inner.close();
    }

    /// Free all resources. Silently ignores close errors.
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.coords_buf);
        self.inner.close() catch {};
    }
};

// ============================================================================
// Tests
// ============================================================================

test "TrrReader: missing file returns FileNotFound" {
    const allocator = std.testing.allocator;
    const result = TrrReader.open(allocator, "nonexistent_file.trr");
    try std.testing.expectError(TrrReadError.FileNotFound, result);
}
