//! Low-level TRR reader/writer integrated from zxdrfile v0.4.0.
//! Coordinates, velocities, and forces remain format-native in this module.
// TRR trajectory file reader
// Zig port of xdrfile_trr from mdtraj:
// https://github.com/mdtraj/mdtraj/tree/master/mdtraj/formats/xtc
//
// TRR is an uncompressed trajectory format storing:
// - Coordinates (x), velocities (v), and/or forces (f)
// - Box matrix, simulation step, time, and lambda
// - Supports both single and double precision
//
// Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel
// Copyright (c) 2014, Robert T. McGibbon (mdtraj modifications)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
const std = @import("std");
const Allocator = std.mem.Allocator;
const native_endian = @import("builtin").cpu.arch.endian();

pub const TrrError = error{
    FileNotFound,
    AccessDenied,
    IsDir,
    NoSpaceLeft,
    IoError,
    InvalidMagic,
    InvalidHeader,
    EndOfFile,
    ReadError,
    OutOfMemory,
    WriteError,
    InvalidAtomCount,
    InvalidFrameData,
};

fn mapOpenError(err: std.Io.File.OpenError) TrrError {
    return switch (err) {
        error.FileNotFound => TrrError.FileNotFound,
        error.AccessDenied, error.PermissionDenied, error.ReadOnlyFileSystem => TrrError.AccessDenied,
        error.IsDir => TrrError.IsDir,
        error.NoSpaceLeft => TrrError.NoSpaceLeft,
        else => TrrError.IoError,
    };
}

const TRR_MAGIC: i32 = 1993;
const DIM: usize = 3;
const VERSION_STRING = "GMX_trn_file";
const READ_BUF_SIZE = 65536;

/// TRR frame header (matches t_trnheader from GROMACS)
const TrrHeader = struct {
    is_double: bool = false,
    ir_size: i32 = 0,
    e_size: i32 = 0,
    box_size: i32 = 0,
    vir_size: i32 = 0,
    pres_size: i32 = 0,
    top_size: i32 = 0,
    sym_size: i32 = 0,
    x_size: i32 = 0,
    v_size: i32 = 0,
    f_size: i32 = 0,
    natoms: i32 = 0,
    step: i32 = 0,
    nre: i32 = 0,
    time: f32 = 0,
    lambda: f32 = 0,
};

/// TRR frame data
pub const TrrFrame = struct {
    step: i32,
    time: f32,
    lambda: f32,
    box: [3][3]f32,
    has_x: bool,
    has_v: bool,
    has_f: bool,
    coords: ?[]f32, // flat array (natoms * 3), null if not present
    velocities: ?[]f32,
    forces: ?[]f32,

    pub fn deinit(self: *TrrFrame, allocator: Allocator) void {
        if (self.coords) |c| allocator.free(c);
        if (self.velocities) |v| allocator.free(v);
        if (self.forces) |f| allocator.free(f);
    }
};

/// TRR file reader
pub const TrrReader = struct {
    io_handle: std.Io,
    file: std.Io.File,
    reader: std.Io.File.Reader,
    read_buf: *[READ_BUF_SIZE]u8,
    allocator: Allocator,
    natoms: i32,

    const Self = @This();

    pub fn open(io_handle: std.Io, allocator: Allocator, path: []const u8) !Self {
        const file = std.Io.Dir.cwd().openFile(io_handle, path, .{ .allow_directory = false }) catch |err| return mapOpenError(err);
        errdefer file.close(io_handle);

        const read_buf = allocator.create([READ_BUF_SIZE]u8) catch return TrrError.OutOfMemory;
        errdefer allocator.destroy(read_buf);

        var self = Self{
            .io_handle = io_handle,
            .file = file,
            .reader = undefined,
            .read_buf = read_buf,
            .allocator = allocator,
            .natoms = 0,
        };
        self.reader = file.reader(io_handle, read_buf);

        // Read first header to get natoms
        const header = try self.readHeader();
        self.natoms = header.natoms;

        // Rewind: we consumed the first header just to learn natoms; replay from byte 0.
        self.reader.seekTo(0) catch return TrrError.ReadError;

        return self;
    }

    pub fn close(self: *Self) void {
        self.allocator.destroy(self.read_buf);
        self.file.close(self.io_handle);
    }

    pub fn getNumAtoms(self: *const Self) i32 {
        return self.natoms;
    }

    /// Read next frame from the TRR file.
    pub fn readFrame(self: *Self) !TrrFrame {
        const header = try self.readHeader();

        if (header.natoms != self.natoms) {
            return TrrError.ReadError;
        }

        const natoms_u: usize = @intCast(header.natoms);
        const size3 = natoms_u * DIM;

        // Read box, virial, pressure (skip virial and pressure)
        var box: [3][3]f32 = std.mem.zeroes([3][3]f32);
        if (header.box_size != 0) {
            if (header.is_double) {
                var dbox: [DIM * DIM]f64 = undefined;
                try self.readDoublesBulk(&dbox);
                for (0..DIM) |i| {
                    for (0..DIM) |j| {
                        box[i][j] = @floatCast(dbox[i * DIM + j]);
                    }
                }
            } else {
                var fbox: [DIM * DIM]f32 = undefined;
                try self.readFloatsBulk(&fbox);
                for (0..DIM) |i| {
                    for (0..DIM) |j| {
                        box[i][j] = fbox[i * DIM + j];
                    }
                }
            }
        }

        // Skip virial
        if (header.vir_size != 0) {
            try self.skipBytes(@intCast(header.vir_size));
        }

        // Skip pressure
        if (header.pres_size != 0) {
            try self.skipBytes(@intCast(header.pres_size));
        }

        // Read coordinates
        var coords: ?[]f32 = null;
        if (header.x_size != 0) {
            coords = try self.readVectors(header.is_double, size3);
        }
        errdefer if (coords) |c| self.allocator.free(c);

        // Read velocities
        var velocities: ?[]f32 = null;
        if (header.v_size != 0) {
            velocities = try self.readVectors(header.is_double, size3);
        }
        errdefer if (velocities) |v| self.allocator.free(v);

        // Read forces
        var forces: ?[]f32 = null;
        if (header.f_size != 0) {
            forces = try self.readVectors(header.is_double, size3);
        }

        return TrrFrame{
            .step = header.step,
            .time = header.time,
            .lambda = header.lambda,
            .box = box,
            .has_x = header.x_size != 0,
            .has_v = header.v_size != 0,
            .has_f = header.f_size != 0,
            .coords = coords,
            .velocities = velocities,
            .forces = forces,
        };
    }

    // ============================================
    // Internal I/O
    // ============================================

    fn readHeader(self: *Self) !TrrHeader {
        // Magic
        const magic = self.readInt() catch return TrrError.EndOfFile;
        if (magic != TRR_MAGIC) return TrrError.InvalidMagic;

        // Version string: slen (int) then XDR string (int length + padded data)
        const slen = try self.readInt();
        const expected_len: i32 = @intCast(VERSION_STRING.len + 1);
        if (slen != expected_len) return TrrError.InvalidHeader;

        // XDR string encoding: 4 bytes for string length + data padded to 4-byte boundary
        const str_len: usize = @intCast(try self.readInt());
        const padded_len = ((str_len + 3) / 4) * 4;
        try self.skipBytes(padded_len);

        // Read header fields
        var header = TrrHeader{};
        header.ir_size = try self.readInt();
        header.e_size = try self.readInt();
        header.box_size = try self.readInt();
        header.vir_size = try self.readInt();
        header.pres_size = try self.readInt();
        header.top_size = try self.readInt();
        header.sym_size = try self.readInt();
        header.x_size = try self.readInt();
        header.v_size = try self.readInt();
        header.f_size = try self.readInt();
        header.natoms = try self.readInt();

        // Determine float size (single or double precision)
        const nflsize = blk: {
            if (header.box_size != 0) {
                break :blk @divTrunc(header.box_size, @as(i32, DIM * DIM));
            } else if (header.x_size != 0) {
                break :blk @divTrunc(header.x_size, header.natoms * @as(i32, DIM));
            } else if (header.v_size != 0) {
                break :blk @divTrunc(header.v_size, header.natoms * @as(i32, DIM));
            } else if (header.f_size != 0) {
                break :blk @divTrunc(header.f_size, header.natoms * @as(i32, DIM));
            } else {
                break :blk @as(i32, 4); // default to float
            }
        };
        header.is_double = (nflsize == 8);

        // Step and nre
        header.step = try self.readInt();
        header.nre = try self.readInt();

        // Time and lambda
        if (header.is_double) {
            const td = try self.readDouble();
            header.time = @floatCast(td);
            const ld = try self.readDouble();
            header.lambda = @floatCast(ld);
        } else {
            header.time = try self.readFloat();
            header.lambda = try self.readFloat();
        }

        return header;
    }

    inline fn io(self: *Self) *std.Io.Reader {
        return &self.reader.interface;
    }

    fn readInt(self: *Self) !i32 {
        const buf = self.io().takeArray(4) catch |err| return mapIoError(err);
        return @bitCast(std.mem.readInt(u32, buf, .big));
    }

    fn readFloat(self: *Self) !f32 {
        const buf = self.io().takeArray(4) catch |err| return mapIoError(err);
        return @bitCast(std.mem.readInt(u32, buf, .big));
    }

    fn readDouble(self: *Self) !f64 {
        const buf = self.io().takeArray(8) catch |err| return mapIoError(err);
        return @bitCast(std.mem.readInt(u64, buf, .big));
    }

    fn readExact(self: *Self, dest: []u8) !void {
        self.io().readSliceAll(dest) catch |err| return mapIoError(err);
    }

    /// Bulk read f32 array: read raw bytes then byte-swap in place.
    fn readFloatsBulk(self: *Self, dest: []f32) !void {
        const bytes: []u8 = @as([*]u8, @ptrCast(dest.ptr))[0 .. dest.len * 4];
        try self.readExact(bytes);
        if (native_endian != .big) {
            for (dest) |*d| {
                d.* = @bitCast(std.mem.bigToNative(u32, @bitCast(d.*)));
            }
        }
    }

    /// Bulk read f64 array: read raw bytes then byte-swap in place.
    fn readDoublesBulk(self: *Self, dest: []f64) !void {
        const bytes: []u8 = @as([*]u8, @ptrCast(dest.ptr))[0 .. dest.len * 8];
        try self.readExact(bytes);
        if (native_endian != .big) {
            for (dest) |*d| {
                d.* = @bitCast(std.mem.bigToNative(u64, @bitCast(d.*)));
            }
        }
    }

    fn skipBytes(self: *Self, count: usize) !void {
        self.io().discardAll(count) catch |err| return mapIoError(err);
    }

    fn mapIoError(err: std.Io.Reader.Error) TrrError {
        return switch (err) {
            error.EndOfStream => TrrError.EndOfFile,
            error.ReadFailed => TrrError.ReadError,
        };
    }

    /// Read a vector array (coords/velocities/forces), handling float/double conversion.
    fn readVectors(self: *Self, is_double: bool, size3: usize) ![]f32 {
        const result = self.allocator.alloc(f32, size3) catch return TrrError.OutOfMemory;
        errdefer self.allocator.free(result);

        if (is_double) {
            // Read doubles in chunks to avoid huge temp allocation
            const chunk_size: usize = 1024;
            var tmp: [chunk_size]f64 = undefined;
            var i: usize = 0;
            while (i < size3) {
                const remaining = size3 - i;
                const n = if (remaining < chunk_size) remaining else chunk_size;
                try self.readDoublesBulk(tmp[0..n]);
                for (0..n) |j| {
                    result[i + j] = @floatCast(tmp[j]);
                }
                i += n;
            }
        } else {
            try self.readFloatsBulk(result);
        }
        return result;
    }
};

// ============================================
// TRR Writer
// ============================================

const WRITE_BUF_SIZE = 65536;

/// TRR file writer (single-precision output)
pub const TrrWriter = struct {
    io_handle: std.Io,
    file: std.Io.File,
    writer: std.Io.File.Writer,
    write_buf: *[WRITE_BUF_SIZE]u8,
    allocator: Allocator,
    natoms: i32,

    const Self = @This();

    pub const Mode = enum { write, append };

    pub fn open(io_handle: std.Io, allocator: Allocator, path: []const u8, natoms: i32, mode: Mode) !Self {
        if (natoms <= 0) return TrrError.InvalidAtomCount;

        switch (mode) {
            .write => {
                const file = std.Io.Dir.cwd().createFile(io_handle, path, .{}) catch |err| return mapOpenError(err);
                errdefer file.close(io_handle);

                const write_buf = allocator.create([WRITE_BUF_SIZE]u8) catch return TrrError.OutOfMemory;
                errdefer allocator.destroy(write_buf);

                return Self{
                    .io_handle = io_handle,
                    .file = file,
                    .writer = file.writer(io_handle, write_buf),
                    .write_buf = write_buf,
                    .allocator = allocator,
                    .natoms = natoms,
                };
            },
            .append => {
                const file = std.Io.Dir.cwd().openFile(io_handle, path, .{ .mode = .read_write }) catch |err| return mapOpenError(err);
                errdefer file.close(io_handle);

                const write_buf = allocator.create([WRITE_BUF_SIZE]u8) catch return TrrError.OutOfMemory;
                errdefer allocator.destroy(write_buf);

                // Validate natoms from existing file and get file size
                const file_size = file.length(io_handle) catch return TrrError.IoError;
                if (file_size > 0) {
                    const read_buf = allocator.create([READ_BUF_SIZE]u8) catch return TrrError.OutOfMemory;
                    defer allocator.destroy(read_buf);

                    var temp_reader = TrrReader{
                        .io_handle = io_handle,
                        .file = file,
                        .reader = file.reader(io_handle, read_buf),
                        .read_buf = read_buf,
                        .allocator = allocator,
                        .natoms = 0,
                    };
                    const header = try temp_reader.readHeader();
                    if (header.natoms != natoms) {
                        return TrrError.InvalidAtomCount;
                    }
                }

                // Use positional writer: set pos to end of file so writes
                // append without disturbing the OS seek position.
                var w = file.writer(io_handle, write_buf);
                w.pos = file_size;

                return Self{
                    .io_handle = io_handle,
                    .file = file,
                    .writer = w,
                    .write_buf = write_buf,
                    .allocator = allocator,
                    .natoms = natoms,
                };
            },
        }
    }

    pub fn close(self: *Self) !void {
        const flush_result = self.writer.interface.flush();
        self.allocator.destroy(self.write_buf);
        self.file.close(self.io_handle);
        flush_result catch return TrrError.WriteError;
    }

    /// Write a single frame to the TRR file.
    /// On write error, the file may contain a partially written frame and
    /// should be considered corrupted.
    pub fn writeFrame(self: *Self, frame: TrrFrame) !void {
        const natoms_u: usize = @intCast(self.natoms);
        const size3 = natoms_u * DIM;

        // Validate frame data consistency
        if (frame.has_x) {
            const coords = frame.coords orelse return TrrError.InvalidFrameData;
            if (coords.len != size3) return TrrError.InvalidFrameData;
        }
        if (frame.has_v) {
            const vels = frame.velocities orelse return TrrError.InvalidFrameData;
            if (vels.len != size3) return TrrError.InvalidFrameData;
        }
        if (frame.has_f) {
            const forces = frame.forces orelse return TrrError.InvalidFrameData;
            if (forces.len != size3) return TrrError.InvalidFrameData;
        }

        // Compute sizes
        const float_size: i32 = @sizeOf(f32);
        const box_is_nonzero = blk: {
            for (0..DIM) |i| {
                for (0..DIM) |j| {
                    if (frame.box[i][j] != 0.0) break :blk true;
                }
            }
            break :blk false;
        };

        const header = TrrHeader{
            .is_double = false,
            .ir_size = 0,
            .e_size = 0,
            .box_size = if (box_is_nonzero) @as(i32, DIM * DIM) * float_size else 0,
            .vir_size = 0,
            .pres_size = 0,
            .top_size = 0,
            .sym_size = 0,
            .x_size = if (frame.has_x) self.natoms * @as(i32, DIM) * float_size else 0,
            .v_size = if (frame.has_v) self.natoms * @as(i32, DIM) * float_size else 0,
            .f_size = if (frame.has_f) self.natoms * @as(i32, DIM) * float_size else 0,
            .natoms = self.natoms,
            .step = frame.step,
            .nre = 0,
            .time = frame.time,
            .lambda = frame.lambda,
        };

        try self.writeHeader(header);

        // Write box matrix
        if (box_is_nonzero) {
            var box_flat: [DIM * DIM]f32 = undefined;
            for (0..DIM) |i| {
                for (0..DIM) |j| {
                    box_flat[i * DIM + j] = frame.box[i][j];
                }
            }
            try self.writeFloatsBulk(&box_flat);
        }

        // Write coordinates
        if (frame.has_x) {
            try self.writeFloatsBulk(frame.coords.?);
        }

        // Write velocities
        if (frame.has_v) {
            try self.writeFloatsBulk(frame.velocities.?);
        }

        // Write forces
        if (frame.has_f) {
            try self.writeFloatsBulk(frame.forces.?);
        }
    }

    // ============================================
    // Internal I/O (optimized for throughput)
    // ============================================

    inline fn io(self: *Self) *std.Io.Writer {
        return &self.writer.interface;
    }

    fn writeRaw(self: *Self, bytes: []const u8) !void {
        self.io().writeAll(bytes) catch return TrrError.WriteError;
    }

    /// Write an entire float array in one bulk operation.
    /// Uses a large stack buffer for byte-swapping to minimize write calls.
    fn writeFloatsBulk(self: *Self, data: []const f32) !void {
        if (native_endian == .big) {
            const bytes: []const u8 = @as([*]const u8, @ptrCast(data.ptr))[0 .. data.len * 4];
            try self.writeRaw(bytes);
        } else {
            // 16KB stack buffer = 4096 floats per chunk
            const chunk_size = 4096;
            var tmp: [chunk_size]u32 = undefined;
            var i: usize = 0;
            while (i < data.len) {
                const remaining = data.len - i;
                const n = if (remaining < chunk_size) remaining else chunk_size;
                const src: [*]const u32 = @ptrCast(data.ptr + i);
                for (0..n) |j| {
                    tmp[j] = @byteSwap(src[j]);
                }
                const bytes: []const u8 = @as([*]const u8, @ptrCast(&tmp))[0 .. n * 4];
                try self.writeRaw(bytes);
                i += n;
            }
        }
    }

    /// Write the entire TRR frame header as a single buffer write.
    fn writeHeader(self: *Self, header: TrrHeader) !void {
        // Pack header into a contiguous buffer:
        // magic(4) + slen(4) + xdr_string_len(4) + "GMX_trn_file"(12) +
        // 10 size fields(40) + natoms(4) + step(4) + nre(4) + time(4) + lambda(4) = 84 bytes
        const version_padded_len = ((VERSION_STRING.len + 3) / 4) * 4;
        // 3 prefix ints + version string + 13 ints (sizes+natoms+step+nre) + 2 floats (time+lambda)
        const header_size = 3 * 4 + version_padded_len + 15 * 4;
        var buf: [header_size]u8 = undefined;
        var pos: usize = 0;

        inline for (.{
            TRR_MAGIC,
            @as(i32, @intCast(VERSION_STRING.len + 1)),
            @as(i32, @intCast(VERSION_STRING.len)),
        }) |val| {
            @as(*align(1) [4]u8, @ptrCast(buf[pos..][0..4])).* =
                std.mem.toBytes(std.mem.nativeToBig(u32, @bitCast(val)));
            pos += 4;
        }

        // Version string + padding
        @memcpy(buf[pos..][0..VERSION_STRING.len], VERSION_STRING);
        pos += VERSION_STRING.len;
        const pad = version_padded_len - VERSION_STRING.len;
        if (pad > 0) {
            @memset(buf[pos..][0..pad], 0);
            pos += pad;
        }

        // 10 size fields + natoms + step + nre + time + lambda
        inline for (.{
            header.ir_size,  header.e_size,    header.box_size,
            header.vir_size, header.pres_size, header.top_size,
            header.sym_size, header.x_size,    header.v_size,
            header.f_size,   header.natoms,    header.step,
            header.nre,
        }) |val| {
            @as(*align(1) [4]u8, @ptrCast(buf[pos..][0..4])).* =
                std.mem.toBytes(std.mem.nativeToBig(u32, @bitCast(val)));
            pos += 4;
        }
        // time and lambda as floats
        @as(*align(1) [4]u8, @ptrCast(buf[pos..][0..4])).* =
            std.mem.toBytes(std.mem.nativeToBig(u32, @bitCast(header.time)));
        pos += 4;
        @as(*align(1) [4]u8, @ptrCast(buf[pos..][0..4])).* =
            std.mem.toBytes(std.mem.nativeToBig(u32, @bitCast(header.lambda)));
        pos += 4;

        try self.writeRaw(buf[0..pos]);
    }
};

// ============================================
// Tests
// ============================================

test "TrrReader open non-existent file" {
    const allocator = std.testing.allocator;
    const result = TrrReader.open(std.testing.io, allocator, "non_existent.trr");
    try std.testing.expectError(TrrError.FileNotFound, result);
}

test "TrrReader open directory returns IsDir" {
    const allocator = std.testing.allocator;
    const result = TrrReader.open(std.testing.io, allocator, "test_data");
    try std.testing.expectError(TrrError.IsDir, result);
}

test "TrrWriter create on directory path returns IsDir" {
    const allocator = std.testing.allocator;
    const result = TrrWriter.open(std.testing.io, allocator, "test_data", 1, .write);
    try std.testing.expectError(TrrError.IsDir, result);
}

test "read frame0.trr first frame" {
    const allocator = std.testing.allocator;

    var reader = try TrrReader.open(std.testing.io, allocator, "test_data/frame0.trr");
    defer reader.close();

    // frame0.trr has 22 atoms
    try std.testing.expectEqual(@as(i32, 22), reader.getNumAtoms());

    var frame = try reader.readFrame();
    defer frame.deinit(allocator);

    try std.testing.expectEqual(@as(i32, 1), frame.step);
    try std.testing.expect(frame.has_x);

    const tolerance: f32 = 0.001;

    // time[0] = 500.00003
    try std.testing.expectApproxEqAbs(@as(f32, 500.0), frame.time, 0.001);
    // lambda[0] = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), frame.lambda, tolerance);

    // box = identity matrix
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), frame.box[0][0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), frame.box[1][1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), frame.box[2][2], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), frame.box[0][1], tolerance);

    const coords = frame.coords.?;

    // atom[0]: [0.429, 1.31, 0.859]
    try std.testing.expectApproxEqAbs(@as(f32, 0.429), coords[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.31), coords[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.859), coords[2], tolerance);

    // atom[21] (last): [1.13, 0.909, 0.83]
    try std.testing.expectApproxEqAbs(@as(f32, 1.13), coords[21 * 3], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.909), coords[21 * 3 + 1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.83), coords[21 * 3 + 2], tolerance);
}

test "read frame0.trr all frames" {
    const allocator = std.testing.allocator;

    var reader = try TrrReader.open(std.testing.io, allocator, "test_data/frame0.trr");
    defer reader.close();

    var frame_count: usize = 0;
    while (true) {
        var frame = reader.readFrame() catch |err| {
            if (err == TrrError.EndOfFile) break;
            return err;
        };
        defer frame.deinit(allocator);
        frame_count += 1;

        try std.testing.expectEqual(@as(usize, 22 * 3), frame.coords.?.len);
    }

    // frame0.trr has 501 frames
    try std.testing.expectEqual(@as(usize, 501), frame_count);
}

test "read frame0.trr last frame" {
    const allocator = std.testing.allocator;

    var reader = try TrrReader.open(std.testing.io, allocator, "test_data/frame0.trr");
    defer reader.close();

    const tolerance: f32 = 0.01;
    var frame_count: usize = 0;

    // Read all frames, keeping only the last one
    var last_coords: ?[]f32 = null;
    defer if (last_coords) |c| allocator.free(c);

    while (true) {
        var frame = reader.readFrame() catch |err| {
            if (err == TrrError.EndOfFile) break;
            return err;
        };
        frame_count += 1;

        // Keep last frame's coords, free previous
        if (last_coords) |c| allocator.free(c);
        last_coords = frame.coords;
        frame.coords = null; // prevent deinit from freeing
        if (frame.velocities) |v| allocator.free(v);
        if (frame.forces) |f| allocator.free(f);
    }

    try std.testing.expectEqual(@as(usize, 501), frame_count);

    const coords = last_coords.?;

    // Frame 501 (index 500): atom[0]: [0.77, 1.01, 0.47]
    try std.testing.expectApproxEqAbs(@as(f32, 0.77), coords[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.01), coords[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.47), coords[2], tolerance);

    // atom[21]: [0.81, 1.4, 1.11]
    try std.testing.expectApproxEqAbs(@as(f32, 0.81), coords[21 * 3], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4), coords[21 * 3 + 1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.11), coords[21 * 3 + 2], tolerance);
}

test "TrrWriter write and read back single frame" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_test.trr";

    // Write a frame
    {
        var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 3, .write);
        defer writer.close() catch {};

        const coords = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
        const frame = TrrFrame{
            .step = 42,
            .time = 10.5,
            .lambda = 0.0,
            .box = [3][3]f32{
                .{ 2.0, 0.0, 0.0 },
                .{ 0.0, 3.0, 0.0 },
                .{ 0.0, 0.0, 4.0 },
            },
            .has_x = true,
            .has_v = false,
            .has_f = false,
            .coords = @constCast(&coords),
            .velocities = null,
            .forces = null,
        };
        try writer.writeFrame(frame);
    }

    // Read it back
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    var reader = try TrrReader.open(std.testing.io, allocator, tmp_path);
    defer reader.close();

    try std.testing.expectEqual(@as(i32, 3), reader.getNumAtoms());

    var frame = try reader.readFrame();
    defer frame.deinit(allocator);

    try std.testing.expectEqual(@as(i32, 42), frame.step);
    try std.testing.expectApproxEqAbs(@as(f32, 10.5), frame.time, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), frame.lambda, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), frame.box[0][0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), frame.box[1][1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), frame.box[2][2], 0.001);
    try std.testing.expect(frame.has_x);

    const coords = frame.coords.?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), coords[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), coords[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), coords[8], 0.001);
}

test "TrrWriter write frame with x, v, f" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_xvf.trr";

    {
        var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 2, .write);
        defer writer.close() catch {};

        const coords = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        const vels = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
        const forces = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 };
        const frame = TrrFrame{
            .step = 1,
            .time = 0.0,
            .lambda = 0.5,
            .box = std.mem.zeroes([3][3]f32),
            .has_x = true,
            .has_v = true,
            .has_f = true,
            .coords = @constCast(&coords),
            .velocities = @constCast(&vels),
            .forces = @constCast(&forces),
        };
        try writer.writeFrame(frame);
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    var reader = try TrrReader.open(std.testing.io, allocator, tmp_path);
    defer reader.close();

    var frame = try reader.readFrame();
    defer frame.deinit(allocator);

    try std.testing.expect(frame.has_x);
    try std.testing.expect(frame.has_v);
    try std.testing.expect(frame.has_f);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), frame.lambda, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), frame.velocities.?[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), frame.forces.?[0], 0.001);
}

test "TrrWriter write multiple frames" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_multi.trr";

    {
        var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 2, .write);
        defer writer.close() catch {};

        for (0..5) |i| {
            const step: i32 = @intCast(i);
            const time: f32 = @as(f32, @floatFromInt(i)) * 0.5;
            const val: f32 = @floatFromInt(i);
            const coords = [_]f32{ val, val + 1.0, val + 2.0, val + 3.0, val + 4.0, val + 5.0 };
            const frame = TrrFrame{
                .step = step,
                .time = time,
                .lambda = 0.0,
                .box = [3][3]f32{ .{ 1.0, 0.0, 0.0 }, .{ 0.0, 1.0, 0.0 }, .{ 0.0, 0.0, 1.0 } },
                .has_x = true,
                .has_v = false,
                .has_f = false,
                .coords = @constCast(&coords),
                .velocities = null,
                .forces = null,
            };
            try writer.writeFrame(frame);
        }
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    var reader = try TrrReader.open(std.testing.io, allocator, tmp_path);
    defer reader.close();

    var count: usize = 0;
    while (true) {
        var frame = reader.readFrame() catch |err| {
            if (err == TrrError.EndOfFile) break;
            return err;
        };
        defer frame.deinit(allocator);
        try std.testing.expectEqual(@as(i32, @intCast(count)), frame.step);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), count);
}

test "TrrWriter append mode" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_append.trr";

    // Write 2 frames
    {
        var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 1, .write);
        defer writer.close() catch {};

        for (0..2) |i| {
            const coords = [_]f32{ @floatFromInt(i), 0.0, 0.0 };
            const frame = TrrFrame{
                .step = @intCast(i),
                .time = 0.0,
                .lambda = 0.0,
                .box = std.mem.zeroes([3][3]f32),
                .has_x = true,
                .has_v = false,
                .has_f = false,
                .coords = @constCast(&coords),
                .velocities = null,
                .forces = null,
            };
            try writer.writeFrame(frame);
        }
    }

    // Append 3 more frames
    {
        var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 1, .append);
        defer writer.close() catch {};

        for (2..5) |i| {
            const coords = [_]f32{ @floatFromInt(i), 0.0, 0.0 };
            const frame = TrrFrame{
                .step = @intCast(i),
                .time = 0.0,
                .lambda = 0.0,
                .box = std.mem.zeroes([3][3]f32),
                .has_x = true,
                .has_v = false,
                .has_f = false,
                .coords = @constCast(&coords),
                .velocities = null,
                .forces = null,
            };
            try writer.writeFrame(frame);
        }
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    // Read all 5 frames
    var reader = try TrrReader.open(std.testing.io, allocator, tmp_path);
    defer reader.close();

    var count: usize = 0;
    while (true) {
        var frame = reader.readFrame() catch |err| {
            if (err == TrrError.EndOfFile) break;
            return err;
        };
        defer frame.deinit(allocator);
        try std.testing.expectEqual(@as(i32, @intCast(count)), frame.step);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), count);
}

test "TrrWriter rejects mismatched array length" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_err.trr";

    var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 2, .write);
    defer {
        writer.close() catch {};
        std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};
    }

    // coords has 3 elements but natoms=2 expects 6
    const bad_coords = [_]f32{ 1.0, 2.0, 3.0 };
    const frame = TrrFrame{
        .step = 0,
        .time = 0.0,
        .lambda = 0.0,
        .box = std.mem.zeroes([3][3]f32),
        .has_x = true,
        .has_v = false,
        .has_f = false,
        .coords = @constCast(&bad_coords),
        .velocities = null,
        .forces = null,
    };
    try std.testing.expectError(TrrError.InvalidFrameData, writer.writeFrame(frame));
}

test "TrrWriter rejects has_x=true with null coords" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_null.trr";

    var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 1, .write);
    defer {
        writer.close() catch {};
        std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};
    }

    const frame = TrrFrame{
        .step = 0,
        .time = 0.0,
        .lambda = 0.0,
        .box = std.mem.zeroes([3][3]f32),
        .has_x = true,
        .has_v = false,
        .has_f = false,
        .coords = null,
        .velocities = null,
        .forces = null,
    };
    try std.testing.expectError(TrrError.InvalidFrameData, writer.writeFrame(frame));
}

test "TrrWriter append rejects natoms mismatch" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_natoms.trr";

    // Write with natoms=2
    {
        var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 2, .write);
        defer writer.close() catch {};
        const coords = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        const frame = TrrFrame{
            .step = 0,
            .time = 0.0,
            .lambda = 0.0,
            .box = std.mem.zeroes([3][3]f32),
            .has_x = true,
            .has_v = false,
            .has_f = false,
            .coords = @constCast(&coords),
            .velocities = null,
            .forces = null,
        };
        try writer.writeFrame(frame);
    }
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    // Try to append with natoms=3 — should fail
    const result = TrrWriter.open(std.testing.io, allocator, tmp_path, 3, .append);
    try std.testing.expectError(TrrError.InvalidAtomCount, result);
}

test "TrrWriter round-trip with frame0.trr" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_roundtrip.trr";

    // Read original and write all frames to new file
    {
        var reader = try TrrReader.open(std.testing.io, allocator, "test_data/frame0.trr");
        defer reader.close();

        const natoms = reader.getNumAtoms();

        var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, natoms, .write);
        defer writer.close() catch {};

        while (true) {
            var frame = reader.readFrame() catch |err| {
                if (err == TrrError.EndOfFile) break;
                return err;
            };
            defer frame.deinit(allocator);
            try writer.writeFrame(frame);
        }
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    // Read back and compare with original
    var reader_orig = try TrrReader.open(std.testing.io, allocator, "test_data/frame0.trr");
    defer reader_orig.close();

    var reader_copy = try TrrReader.open(std.testing.io, allocator, tmp_path);
    defer reader_copy.close();

    try std.testing.expectEqual(reader_orig.getNumAtoms(), reader_copy.getNumAtoms());

    const tolerance: f32 = 0.0001;
    var frame_count: usize = 0;
    while (true) {
        var orig = reader_orig.readFrame() catch |err| {
            if (err == TrrError.EndOfFile) break;
            return err;
        };
        defer orig.deinit(allocator);

        var copy = reader_copy.readFrame() catch |err| {
            if (err == TrrError.EndOfFile) {
                return error.TestUnexpectedResult; // copy has fewer frames
            }
            return err;
        };
        defer copy.deinit(allocator);

        try std.testing.expectEqual(orig.step, copy.step);
        try std.testing.expectApproxEqAbs(orig.time, copy.time, tolerance);
        try std.testing.expectApproxEqAbs(orig.lambda, copy.lambda, tolerance);
        try std.testing.expectEqual(orig.has_x, copy.has_x);
        try std.testing.expectEqual(orig.has_v, copy.has_v);
        try std.testing.expectEqual(orig.has_f, copy.has_f);

        if (orig.coords) |oc| {
            const cc = copy.coords.?;
            for (oc, cc) |o, c| {
                try std.testing.expectApproxEqAbs(o, c, tolerance);
            }
        }
        if (orig.velocities) |ov| {
            const cv = copy.velocities.?;
            for (ov, cv) |o, c| {
                try std.testing.expectApproxEqAbs(o, c, tolerance);
            }
        }
        if (orig.forces) |of| {
            const cf = copy.forces.?;
            for (of, cf) |o, c| {
                try std.testing.expectApproxEqAbs(o, c, tolerance);
            }
        }

        frame_count += 1;
    }

    try std.testing.expectEqual(@as(usize, 501), frame_count);
}

test "TrrWriter rejects natoms <= 0" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(TrrError.InvalidAtomCount, TrrWriter.open(std.testing.io, allocator, "test_data/trr_tmp_zero.trr", 0, .write));
    try std.testing.expectError(TrrError.InvalidAtomCount, TrrWriter.open(std.testing.io, allocator, "test_data/trr_tmp_neg.trr", -1, .write));
}

test "TrrWriter rejects has_v=true with null velocities" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_null_v.trr";

    var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 1, .write);
    defer {
        writer.close() catch {};
        std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};
    }

    const coords = [_]f32{ 1.0, 2.0, 3.0 };
    const frame = TrrFrame{
        .step = 0,
        .time = 0.0,
        .lambda = 0.0,
        .box = std.mem.zeroes([3][3]f32),
        .has_x = true,
        .has_v = true,
        .has_f = false,
        .coords = @constCast(&coords),
        .velocities = null,
        .forces = null,
    };
    try std.testing.expectError(TrrError.InvalidFrameData, writer.writeFrame(frame));
}

test "TrrWriter rejects has_f=true with null forces" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/trr_tmp_write_null_f.trr";

    var writer = try TrrWriter.open(std.testing.io, allocator, tmp_path, 1, .write);
    defer {
        writer.close() catch {};
        std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};
    }

    const coords = [_]f32{ 1.0, 2.0, 3.0 };
    const frame = TrrFrame{
        .step = 0,
        .time = 0.0,
        .lambda = 0.0,
        .box = std.mem.zeroes([3][3]f32),
        .has_x = true,
        .has_v = false,
        .has_f = true,
        .coords = @constCast(&coords),
        .velocities = null,
        .forces = null,
    };
    try std.testing.expectError(TrrError.InvalidFrameData, writer.writeFrame(frame));
}
