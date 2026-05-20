// XTC trajectory file reader
// Zig port of xdrfile library
//
// Based on the xdrfile implementation from mdtraj:
// https://github.com/mdtraj/mdtraj/tree/master/mdtraj/formats/xtc
//
// XTC is a compressed trajectory format using:
// - XDR (External Data Representation) for portable binary I/O
// - Custom 3D coordinate compression with delta encoding
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

pub const XtcError = error{
    FileNotFound,
    AccessDenied,
    IsDir,
    NoSpaceLeft,
    IoError,
    InvalidMagic,
    InvalidHeader,
    EndOfFile,
    ReadError,
    DecompressionError,
    BufferTooSmall,
    OutOfMemory,
    WriteError,
    InvalidAtomCount,
    CompressionError,
};

fn mapOpenError(err: std.Io.File.OpenError) XtcError {
    return switch (err) {
        error.FileNotFound => XtcError.FileNotFound,
        error.AccessDenied, error.PermissionDenied, error.ReadOnlyFileSystem => XtcError.AccessDenied,
        error.IsDir => XtcError.IsDir,
        error.NoSpaceLeft => XtcError.NoSpaceLeft,
        else => XtcError.IoError,
    };
}

/// XTC file magic number
const XTC_MAGIC: i32 = 1995;

/// Magic integers for coordinate compression (from GROMACS xdrfile)
const magicints = [_]u32{
    0,        0,        0,        0,       0,       0,       0,       0,       0,       8,
    10,       12,       16,       20,      25,      32,      40,      50,      64,      80,
    101,      128,      161,      203,     256,     322,     406,     512,     645,     812,
    1024,     1290,     1625,     2048,    2580,    3250,    4096,    5060,    6501,    8192,
    10321,    13003,    16384,    20642,   26007,   32768,   41285,   52015,   65536,   82570,
    104031,   131072,   165140,   208063,  262144,  330280,  416127,  524287,  660561,  832255,
    1048576,  1321122,  1664510,  2097152, 2642245, 3329021, 4194304, 5284491, 6658042, 8388607,
    10568983, 13316085, 16777216,
};

const FIRSTIDX: usize = 9;
const LASTIDX: usize = magicints.len;

/// XTC frame data
pub const XtcFrame = struct {
    step: i32,
    time: f32,
    box: [3][3]f32,
    coords: []f32, // flat array of x,y,z coordinates (length = natoms * 3)
    precision: f32,

    pub fn deinit(self: *XtcFrame, allocator: Allocator) void {
        allocator.free(self.coords);
    }
};

const READ_BUF_SIZE = 65536;

// ============================================
// Bit-level encoding/decoding utilities (module scope, shared by reader and writer)
// ============================================

/// Return the number of bits needed to represent `size` values (0..size-1).
fn sizeofint(size: u32) u32 {
    var num: u32 = 1;
    var num_of_bits: u32 = 0;
    while (size >= num and num_of_bits < 32) {
        num_of_bits += 1;
        num <<= 1;
    }
    return num_of_bits;
}

/// Return the number of bits needed to represent the product of `sizes`.
fn sizeofints(num_of_ints: usize, sizes: []const u32) u32 {
    var bytes: [32]u32 = undefined;
    var num_of_bytes: usize = 1;
    bytes[0] = 1;

    for (0..num_of_ints) |i| {
        var tmp: u32 = 0;
        for (0..num_of_bytes) |bytecnt| {
            tmp = bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
        }
        while (tmp != 0) {
            bytes[num_of_bytes] = tmp & 0xff;
            num_of_bytes += 1;
            tmp >>= 8;
        }
    }

    var num: u32 = 1;
    var num_of_bits: u32 = 0;
    num_of_bytes -= 1;
    while (bytes[num_of_bytes] >= num) {
        num_of_bits += 1;
        num *= 2;
    }
    return num_of_bits + @as(u32, @intCast(num_of_bytes)) * 8;
}

/// Decode `num_of_bits` bits from the compressed buffer.
/// buf[0] = byte counter, buf[1] = lastbits, buf[2] = lastbyte
fn decodebits(buf: []i32, num_of_bits_arg: u32) i32 {
    var num_of_bits = num_of_bits_arg;
    const cbuf: [*]u8 = @ptrCast(@alignCast(buf.ptr + 3));

    var cnt: usize = @intCast(@as(u32, @bitCast(buf[0])));
    var lastbits: u32 = @bitCast(buf[1]);
    var lastbyte: u32 = @bitCast(buf[2]);

    const mask: u32 = if (num_of_bits_arg >= 32)
        0xFFFFFFFF
    else
        (@as(u32, 1) << @as(u5, @intCast(num_of_bits_arg))) - 1;

    var num: u32 = 0;
    while (num_of_bits >= 8) {
        lastbyte = (lastbyte << 8) | cbuf[cnt];
        cnt += 1;
        const shift_r: u5 = @intCast(lastbits & 31);
        const shift_l: u5 = @intCast((num_of_bits - 8) & 31);
        num |= (lastbyte >> shift_r) << shift_l;
        num_of_bits -= 8;
    }
    if (num_of_bits > 0) {
        if (lastbits < num_of_bits) {
            lastbits += 8;
            lastbyte = (lastbyte << 8) | cbuf[cnt];
            cnt += 1;
        }
        lastbits -= num_of_bits;
        const shift: u5 = @intCast(lastbits & 31);
        const mask_bits: u5 = @intCast(num_of_bits & 31);
        num |= (lastbyte >> shift) & ((@as(u32, 1) << mask_bits) -% 1);
    }

    num &= mask;
    buf[0] = @bitCast(@as(u32, @intCast(cnt)));
    buf[1] = @bitCast(lastbits);
    buf[2] = @bitCast(lastbyte);
    return @bitCast(num);
}

/// Decode multiple packed integers from the compressed buffer.
fn decodeints(buf: []i32, num_of_ints: usize, num_of_bits_arg: u32, sizes: []const u32, nums: []i32) void {
    var bytes: [32]i32 = undefined;
    bytes[1] = 0;
    bytes[2] = 0;
    bytes[3] = 0;

    var num_of_bytes: usize = 0;
    var num_of_bits = num_of_bits_arg;
    while (num_of_bits > 8) {
        bytes[num_of_bytes] = decodebits(buf, 8);
        num_of_bytes += 1;
        num_of_bits -= 8;
    }
    if (num_of_bits > 0) {
        bytes[num_of_bytes] = decodebits(buf, num_of_bits);
        num_of_bytes += 1;
    }

    var i = num_of_ints - 1;
    while (i > 0) : (i -= 1) {
        var num: i32 = 0;
        var j = num_of_bytes;
        while (j > 0) {
            j -= 1;
            num = (num << 8) | bytes[j];
            const size_i: i32 = @intCast(sizes[i]);
            const p = @divTrunc(num, size_i);
            bytes[j] = p;
            num = num - p * size_i;
        }
        nums[i] = num;
    }
    nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}

/// Encode `num` using `num_of_bits` bits into the compressed buffer.
/// buf[0] = byte counter, buf[1] = lastbits, buf[2] = lastbyte
fn encodebits(buf: []i32, num_of_bits_arg: u32, num: u32) void {
    const cbuf: [*]u8 = @ptrCast(@alignCast(buf.ptr + 3));

    var cnt: usize = @intCast(@as(u32, @bitCast(buf[0])));
    var lastbits: u32 = @bitCast(buf[1]);
    var lastbyte: u32 = @bitCast(buf[2]);

    var num_of_bits = num_of_bits_arg;
    while (num_of_bits >= 8) {
        lastbyte = (lastbyte << 8) | ((num >> @intCast(num_of_bits - 8)) & 0xff);
        cbuf[cnt] = @intCast((lastbyte >> @intCast(lastbits & 31)) & 0xff);
        cnt += 1;
        num_of_bits -= 8;
    }
    if (num_of_bits > 0) {
        const tail_mask = (@as(u32, 1) << @intCast(num_of_bits & 31)) - 1;
        lastbyte = (lastbyte << @intCast(num_of_bits & 31)) | (num & tail_mask);
        lastbits += num_of_bits;
        if (lastbits >= 8) {
            lastbits -= 8;
            cbuf[cnt] = @intCast((lastbyte >> @intCast(lastbits & 31)) & 0xff);
            cnt += 1;
        }
    }
    buf[0] = @bitCast(@as(u32, @intCast(cnt)));
    buf[1] = @bitCast(lastbits);
    buf[2] = @bitCast(lastbyte);
    if (lastbits > 0) {
        cbuf[cnt] = @intCast((lastbyte << @intCast((8 - lastbits) & 31)) & 0xff);
    }
}

/// Encode multiple integers packed into the compressed buffer.
fn encodeints(buf: []i32, num_of_ints: usize, num_of_bits: u32, sizes: []const u32, nums: []const u32) void {
    var bytes: [32]u32 = undefined;
    var num_of_bytes: usize = 0;

    var tmp: u32 = nums[0];
    while (true) {
        bytes[num_of_bytes] = tmp & 0xff;
        num_of_bytes += 1;
        tmp >>= 8;
        if (tmp == 0) break;
    }

    for (1..num_of_ints) |idx| {
        tmp = nums[idx];
        for (0..num_of_bytes) |bytecnt| {
            tmp = bytes[bytecnt] * sizes[idx] + tmp;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
        }
        var bytecnt = num_of_bytes;
        while (tmp != 0) {
            bytes[bytecnt] = tmp & 0xff;
            bytecnt += 1;
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }

    if (num_of_bits >= num_of_bytes * 8) {
        for (0..num_of_bytes) |idx| {
            encodebits(buf, 8, bytes[idx]);
        }
        encodebits(buf, num_of_bits - @as(u32, @intCast(num_of_bytes)) * 8, 0);
    } else {
        for (0..num_of_bytes - 1) |idx| {
            encodebits(buf, 8, bytes[idx]);
        }
        encodebits(buf, num_of_bits - @as(u32, @intCast(num_of_bytes - 1)) * 8, bytes[num_of_bytes - 1]);
    }
}

/// XTC file reader
pub const XtcReader = struct {
    io_handle: std.Io,
    file: std.Io.File,
    reader: std.Io.File.Reader,
    read_buf: *[READ_BUF_SIZE]u8,
    allocator: Allocator,
    natoms: i32,
    // buf1: integer coordinate buffer for decompression (thiscoord storage)
    // buf2: compressed bitstream with 3-element header for bit decoding state
    buf1: []i32,
    buf2: []i32,

    const Self = @This();

    pub fn open(io_handle: std.Io, allocator: Allocator, path: []const u8) !Self {
        const file = std.Io.Dir.cwd().openFile(io_handle, path, .{ .allow_directory = false }) catch |err| return mapOpenError(err);
        errdefer file.close(io_handle);

        const read_buf = allocator.create([READ_BUF_SIZE]u8) catch return XtcError.OutOfMemory;
        errdefer allocator.destroy(read_buf);

        var self = Self{
            .io_handle = io_handle,
            .file = file,
            .reader = undefined,
            .read_buf = read_buf,
            .allocator = allocator,
            .natoms = 0,
            .buf1 = &[_]i32{},
            .buf2 = &[_]i32{},
        };
        self.reader = file.reader(io_handle, read_buf);

        // Read first frame header to get natoms
        const magic = self.readInt() catch return XtcError.ReadError;
        if (magic != XTC_MAGIC) return XtcError.InvalidMagic;

        self.natoms = self.readInt() catch return XtcError.ReadError;
        if (self.natoms <= 0) return XtcError.InvalidAtomCount;

        // Rewind: we consumed the first header just to learn natoms; replay from byte 0.
        self.reader.seekTo(0) catch return XtcError.ReadError;

        // Allocate decompression buffers
        const natoms_u: usize = @intCast(self.natoms);
        const size3 = std.math.mul(usize, natoms_u, 3) catch return XtcError.ReadError;
        self.buf1 = allocator.alloc(i32, size3) catch return XtcError.OutOfMemory;
        errdefer allocator.free(self.buf1);
        // buf2: size3 * 1.2 for worst-case compression + 3 for bit decoder header
        const buf2_size: usize = size3 + size3 / 5;
        self.buf2 = allocator.alloc(i32, buf2_size + 3) catch return XtcError.OutOfMemory;

        return self;
    }

    pub fn close(self: *Self) void {
        self.allocator.free(self.buf1);
        self.allocator.free(self.buf2);
        self.allocator.destroy(self.read_buf);
        self.file.close(self.io_handle);
    }

    pub fn getNumAtoms(self: *const Self) i32 {
        return self.natoms;
    }

    /// Read next frame from the XTC file.
    pub fn readFrame(self: *Self) !XtcFrame {
        // Read header: magic, natoms, step, time
        const magic = self.readInt() catch return XtcError.EndOfFile;
        if (magic != XTC_MAGIC) {
            return XtcError.InvalidMagic;
        }

        const natoms = try self.readInt();
        if (natoms != self.natoms) {
            return XtcError.ReadError;
        }

        const step = try self.readInt();
        const time = try self.readFloat();

        // Read box (3x3 matrix)
        var box: [3][3]f32 = undefined;
        for (0..3) |i| {
            for (0..3) |j| {
                box[i][j] = try self.readFloat();
            }
        }

        // Allocate output coordinates
        const natoms_u: usize = @intCast(natoms);
        const size3 = std.math.mul(usize, natoms_u, 3) catch return XtcError.ReadError;
        const coords = self.allocator.alloc(f32, size3) catch return XtcError.OutOfMemory;
        errdefer self.allocator.free(coords);

        // Decompress coordinates
        var precision: f32 = 0;
        const result = self.decompressCoords(coords, &precision);
        if (result < 0) {
            return XtcError.DecompressionError;
        }

        return XtcFrame{
            .step = step,
            .time = time,
            .box = box,
            .coords = coords,
            .precision = precision,
        };
    }

    // ============================================
    // XDR I/O (big-endian, 4 bytes per element)
    // ============================================

    inline fn io(self: *Self) *std.Io.Reader {
        return &self.reader.interface;
    }

    fn mapIoError(err: std.Io.Reader.Error) XtcError {
        return switch (err) {
            error.EndOfStream => XtcError.EndOfFile,
            error.ReadFailed => XtcError.ReadError,
        };
    }

    fn readInt(self: *Self) !i32 {
        const buf = self.io().takeArray(4) catch |err| return mapIoError(err);
        return @bitCast(std.mem.readInt(u32, buf, .big));
    }

    fn readFloat(self: *Self) !f32 {
        const buf = self.io().takeArray(4) catch |err| return mapIoError(err);
        return @bitCast(std.mem.readInt(u32, buf, .big));
    }

    fn readOpaque(self: *Self, dest: []u8) !void {
        try self.readExact(dest);
        // XDR opaque data is padded to 4-byte boundary
        const padding = (4 - (dest.len % 4)) % 4;
        if (padding > 0) {
            self.io().discardAll(padding) catch |err| return mapIoError(err);
        }
    }

    fn readExact(self: *Self, dest: []u8) !void {
        self.io().readSliceAll(dest) catch |err| return mapIoError(err);
    }

    // ============================================
    // 3D coordinate decompression (XTC-specific)
    // ============================================

    /// Decompress XTC-compressed 3D coordinates.
    /// Based on xdrfile_decompress_coord_float() from mdtraj/xdrfile.
    fn decompressCoords(self: *Self, coords: []f32, precision: *f32) i32 {
        const buf1 = self.buf1;
        const buf2 = self.buf2;

        // Read number of atoms
        const lsize: i32 = self.readInt() catch return -1;
        if (lsize <= 0) return -1;
        if (lsize != self.natoms) return -1;
        const lsize_u: usize = @intCast(lsize);
        const size3 = std.math.mul(usize, lsize_u, 3) catch return -1;
        if (size3 != coords.len or size3 > buf1.len) return -1;

        // Don't bother with compression for 9 atoms or less
        if (lsize <= 9) {
            for (0..size3) |i| {
                coords[i] = self.readFloat() catch return -1;
            }
            return lsize;
        }

        // Read precision
        precision.* = self.readFloat() catch return -1;

        buf2[0] = 0;
        buf2[1] = 0;
        buf2[2] = 0;

        // Read min/max integer bounds
        var minint: [3]i32 = undefined;
        var maxint: [3]i32 = undefined;
        for (0..3) |i| {
            minint[i] = self.readInt() catch return -1;
        }
        for (0..3) |i| {
            maxint[i] = self.readInt() catch return -1;
        }

        var sizeint: [3]u32 = undefined;
        var bitsizeint: [3]u32 = .{ 0, 0, 0 };
        sizeint[0] = @intCast(maxint[0] - minint[0] + 1);
        sizeint[1] = @intCast(maxint[1] - minint[1] + 1);
        sizeint[2] = @intCast(maxint[2] - minint[2] + 1);

        // Check if one of the sizes is too big to be multiplied
        var bitsize: u32 = 0;
        if ((sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff) {
            bitsizeint[0] = sizeofint(sizeint[0]);
            bitsizeint[1] = sizeofint(sizeint[1]);
            bitsizeint[2] = sizeofint(sizeint[2]);
            bitsize = 0; // flag the use of large sizes
        } else {
            bitsize = sizeofints(3, &sizeint);
        }

        // Read smallidx and compute bounds (mdtraj style)
        var smallidx: i32 = self.readInt() catch return -1;
        if (smallidx < FIRSTIDX or smallidx >= LASTIDX) return -1;

        var tmp: i32 = smallidx + 8;
        const maxidx: usize = if (LASTIDX < @as(usize, @intCast(tmp))) LASTIDX else @intCast(tmp);
        const minidx: usize = maxidx - 8; // often equals smallidx
        _ = minidx;

        tmp = smallidx - 1;
        if (tmp < FIRSTIDX) tmp = @intCast(FIRSTIDX);
        var smaller: i32 = @intCast(magicints[@intCast(tmp)] / 2);
        var smallnum: i32 = @intCast(magicints[@intCast(smallidx)] / 2);
        var sizesmall: [3]u32 = .{
            magicints[@intCast(smallidx)],
            magicints[@intCast(smallidx)],
            magicints[@intCast(smallidx)],
        };

        // Read compressed data length and payload
        const data_len = self.readInt() catch return -1;
        if (data_len < 0) return -1;
        const data_len_u: usize = @intCast(data_len);
        const buf2_payload_capacity = std.math.mul(usize, buf2.len - 3, @sizeOf(i32)) catch return -1;
        if (data_len_u > buf2_payload_capacity) return -1;

        const cbuf: [*]u8 = @ptrCast(@alignCast(buf2.ptr + 3));
        self.readOpaque(cbuf[0..data_len_u]) catch return -1;
        buf2[0] = 0;
        buf2[1] = 0;
        buf2[2] = 0;

        const inv_precision = 1.0 / precision.*;
        var run: i32 = 0;
        var i: usize = 0;
        var lfp: usize = 0;
        var prevcoord: [3]i32 = undefined;

        while (i < @as(usize, @intCast(lsize))) {
            // Use buf1 as thiscoord storage (like C version: thiscoord = (int*)(lip) + i*3)
            const thiscoord = buf1[i * 3 ..][0..3];

            if (bitsize == 0) {
                thiscoord[0] = decodebits(buf2, bitsizeint[0]);
                thiscoord[1] = decodebits(buf2, bitsizeint[1]);
                thiscoord[2] = decodebits(buf2, bitsizeint[2]);
            } else {
                decodeints(buf2, 3, bitsize, &sizeint, thiscoord);
            }

            i += 1;
            thiscoord[0] += minint[0];
            thiscoord[1] += minint[1];
            thiscoord[2] += minint[2];

            prevcoord[0] = thiscoord[0];
            prevcoord[1] = thiscoord[1];
            prevcoord[2] = thiscoord[2];

            const flag = decodebits(buf2, 1);
            var is_smaller: i32 = 0;
            if (flag == 1) {
                run = decodebits(buf2, 5);
                is_smaller = @mod(run, 3);
                run -= is_smaller;
                is_smaller -= 1;
            }

            // Buffer overrun check (mdtraj style)
            if (lfp + @as(usize, @intCast(@max(0, run))) > size3) {
                return -1;
            }

            if (run > 0) {
                // Advance thiscoord pointer into buf1 for run-length decoded atoms
                const run_start = i * 3;
                var k: i32 = 0;
                while (k < run) : (k += 3) {
                    const k_u: usize = @intCast(@divTrunc(k, 3));
                    const rc = buf1[run_start + k_u * 3 ..][0..3];
                    decodeints(buf2, 3, @intCast(smallidx), &sizesmall, rc);
                    i += 1;
                    rc[0] += prevcoord[0] - smallnum;
                    rc[1] += prevcoord[1] - smallnum;
                    rc[2] += prevcoord[2] - smallnum;
                    if (k == 0) {
                        // Interchange first with second atom for better
                        // compression of water molecules
                        const t0 = rc[0];
                        rc[0] = prevcoord[0];
                        prevcoord[0] = t0;
                        const t1 = rc[1];
                        rc[1] = prevcoord[1];
                        prevcoord[1] = t1;
                        const t2 = rc[2];
                        rc[2] = prevcoord[2];
                        prevcoord[2] = t2;

                        coords[lfp] = @as(f32, @floatFromInt(prevcoord[0])) * inv_precision;
                        coords[lfp + 1] = @as(f32, @floatFromInt(prevcoord[1])) * inv_precision;
                        coords[lfp + 2] = @as(f32, @floatFromInt(prevcoord[2])) * inv_precision;
                        lfp += 3;
                    } else {
                        prevcoord[0] = rc[0];
                        prevcoord[1] = rc[1];
                        prevcoord[2] = rc[2];
                    }

                    coords[lfp] = @as(f32, @floatFromInt(rc[0])) * inv_precision;
                    coords[lfp + 1] = @as(f32, @floatFromInt(rc[1])) * inv_precision;
                    coords[lfp + 2] = @as(f32, @floatFromInt(rc[2])) * inv_precision;
                    lfp += 3;
                }
            } else {
                coords[lfp] = @as(f32, @floatFromInt(thiscoord[0])) * inv_precision;
                coords[lfp + 1] = @as(f32, @floatFromInt(thiscoord[1])) * inv_precision;
                coords[lfp + 2] = @as(f32, @floatFromInt(thiscoord[2])) * inv_precision;
                lfp += 3;
            }

            // Adjust smallidx based on compression efficiency
            smallidx += is_smaller;
            if (smallidx < FIRSTIDX or smallidx >= LASTIDX) return -1;
            if (is_smaller < 0) {
                smallnum = smaller;
                if (smallidx > FIRSTIDX) {
                    smaller = @intCast(magicints[@intCast(smallidx - 1)] / 2);
                } else {
                    smaller = 0;
                }
            } else if (is_smaller > 0) {
                smaller = smallnum;
                smallnum = @intCast(magicints[@intCast(smallidx)] / 2);
            }
            sizesmall[0] = magicints[@intCast(smallidx)];
            sizesmall[1] = magicints[@intCast(smallidx)];
            sizesmall[2] = magicints[@intCast(smallidx)];

            // Zero-size check (mdtraj bugfix: prevents divide-by-zero)
            if (sizesmall[0] == 0 or sizesmall[1] == 0 or sizesmall[2] == 0) {
                return -1;
            }
        }

        return lsize;
    }
};

// ============================================
// XTC Writer
// ============================================

const WRITE_BUF_SIZE = 65536;

/// XTC file writer (single-precision with coordinate compression)
pub const XtcWriter = struct {
    io_handle: std.Io,
    file: std.Io.File,
    writer: std.Io.File.Writer,
    write_buf: *[WRITE_BUF_SIZE]u8,
    allocator: Allocator,
    natoms: i32,
    // buf1: integer coordinate buffer (size3 elements)
    // buf2: compressed bitstream with 3-element header + payload
    buf1: []i32,
    buf2: []i32,

    const Self = @This();

    pub const Mode = enum { write, append };

    pub fn open(io_handle: std.Io, allocator: Allocator, path: []const u8, natoms: i32, mode: Mode) !Self {
        if (natoms <= 0) return XtcError.InvalidAtomCount;

        const natoms_u: usize = @intCast(natoms);
        const size3 = std.math.mul(usize, natoms_u, 3) catch return XtcError.InvalidAtomCount;

        // Allocate compression buffers
        const buf1 = allocator.alloc(i32, size3) catch return XtcError.OutOfMemory;
        errdefer allocator.free(buf1);

        // buf2: size3 * 1.2 + 3 for worst-case compression plus 3-element header
        const buf2_size: usize = size3 + size3 / 5 + 3;
        const buf2 = allocator.alloc(i32, buf2_size) catch return XtcError.OutOfMemory;
        errdefer allocator.free(buf2);

        switch (mode) {
            .write => {
                const file = std.Io.Dir.cwd().createFile(io_handle, path, .{}) catch |err| return mapOpenError(err);
                errdefer file.close(io_handle);

                const write_buf = allocator.create([WRITE_BUF_SIZE]u8) catch return XtcError.OutOfMemory;
                errdefer allocator.destroy(write_buf);

                return Self{
                    .io_handle = io_handle,
                    .file = file,
                    .writer = file.writer(io_handle, write_buf),
                    .write_buf = write_buf,
                    .allocator = allocator,
                    .natoms = natoms,
                    .buf1 = buf1,
                    .buf2 = buf2,
                };
            },
            .append => {
                const file = std.Io.Dir.cwd().openFile(io_handle, path, .{ .mode = .read_write }) catch |err| return mapOpenError(err);
                errdefer file.close(io_handle);

                const write_buf = allocator.create([WRITE_BUF_SIZE]u8) catch return XtcError.OutOfMemory;
                errdefer allocator.destroy(write_buf);

                // Validate natoms from existing file, then seek to end
                const file_size = file.length(io_handle) catch return XtcError.IoError;
                if (file_size > 0) {
                    const read_buf = allocator.create([READ_BUF_SIZE]u8) catch return XtcError.OutOfMemory;
                    defer allocator.destroy(read_buf);

                    var temp_reader = file.reader(io_handle, read_buf);
                    const magic_buf = temp_reader.interface.takeArray(4) catch |err| return switch (err) {
                        error.EndOfStream => XtcError.InvalidHeader,
                        error.ReadFailed => XtcError.ReadError,
                    };
                    const magic: i32 = @bitCast(std.mem.readInt(u32, magic_buf, .big));
                    if (magic != XTC_MAGIC) return XtcError.InvalidMagic;

                    const natoms_buf = temp_reader.interface.takeArray(4) catch |err| return switch (err) {
                        error.EndOfStream => XtcError.InvalidHeader,
                        error.ReadFailed => XtcError.ReadError,
                    };
                    const file_natoms: i32 = @bitCast(std.mem.readInt(u32, natoms_buf, .big));
                    if (file_natoms != natoms) return XtcError.InvalidAtomCount;
                }

                // Use positional writer: set pos to end of file so writes append
                var w = file.writer(io_handle, write_buf);
                w.pos = file_size;

                return Self{
                    .io_handle = io_handle,
                    .file = file,
                    .writer = w,
                    .write_buf = write_buf,
                    .allocator = allocator,
                    .natoms = natoms,
                    .buf1 = buf1,
                    .buf2 = buf2,
                };
            },
        }
    }

    /// Flush, release all resources, and close the file.
    /// Resources are always freed even if flush fails.
    pub fn close(self: *Self) !void {
        const flush_result = self.writer.interface.flush();
        self.allocator.free(self.buf1);
        self.allocator.free(self.buf2);
        self.allocator.destroy(self.write_buf);
        self.file.close(self.io_handle);
        flush_result catch return XtcError.WriteError;
    }

    /// Write a single frame to the XTC file.
    /// On write error, the file may contain a partially written frame.
    pub fn writeFrame(self: *Self, frame: XtcFrame) !void {
        const natoms_u: usize = @intCast(self.natoms);
        if (frame.coords.len != natoms_u * 3) return XtcError.CompressionError;

        // Write header: magic, natoms, step, time
        try self.writeInt(XTC_MAGIC);
        try self.writeInt(self.natoms);
        try self.writeInt(frame.step);
        try self.writeFloat(frame.time);

        // Write box (3x3 matrix)
        for (0..3) |i| {
            for (0..3) |j| {
                try self.writeFloat(frame.box[i][j]);
            }
        }

        // Write compressed coordinates
        try self.compressCoords(frame.coords, frame.precision);
    }

    // ============================================
    // Coordinate compression
    // ============================================

    /// Compress and write 3D coordinates using the XTC algorithm.
    /// Ported from xdrfile_compress_coord_float() in xdrfile.c.
    fn compressCoords(self: *Self, coords: []const f32, precision_arg: f32) !void {
        const lsize: i32 = self.natoms;
        const lsize_u: usize = @intCast(lsize);
        const size3 = lsize_u * 3;

        // Write number of atoms
        try self.writeInt(lsize);

        // For 9 atoms or fewer, write raw floats (no compression)
        if (lsize <= 9) {
            for (0..size3) |i| {
                try self.writeFloat(coords[i]);
            }
            return;
        }

        // Validate precision
        if (precision_arg <= 0) return XtcError.CompressionError;
        const precision: f32 = precision_arg;
        try self.writeFloat(precision);

        const buf1 = self.buf1;
        const buf2 = self.buf2;
        buf2[0] = 0;
        buf2[1] = 0;
        buf2[2] = 0;

        // Phase 1: Quantize floats to integers, find min/max per dimension
        var minint: [3]i32 = .{ std.math.maxInt(i32), std.math.maxInt(i32), std.math.maxInt(i32) };
        var maxint: [3]i32 = .{ std.math.minInt(i32), std.math.minInt(i32), std.math.minInt(i32) };
        var mindiff: i32 = std.math.maxInt(i32);
        var oldlint: [3]i32 = .{ 0, 0, 0 };

        var lip: usize = 0;
        var lfp: usize = 0;
        while (lfp < size3) {
            var lint: [3]i32 = undefined;
            for (0..3) |d| {
                const lf: f32 = if (coords[lfp + d] >= 0.0)
                    coords[lfp + d] * precision + 0.5
                else
                    coords[lfp + d] * precision - 0.5;
                lint[d] = @intFromFloat(lf);
                if (lint[d] < minint[d]) minint[d] = lint[d];
                if (lint[d] > maxint[d]) maxint[d] = lint[d];
                buf1[lip + d] = lint[d];
            }
            // Track min inter-atom distance (from second atom onwards)
            if (lfp >= 3) {
                const diff: i32 = @as(i32, @intCast(@abs(oldlint[0] - lint[0]))) +
                    @as(i32, @intCast(@abs(oldlint[1] - lint[1]))) +
                    @as(i32, @intCast(@abs(oldlint[2] - lint[2])));
                if (diff < mindiff) mindiff = diff;
            }
            oldlint = lint;
            lip += 3;
            lfp += 3;
        }

        // Write minint, maxint
        for (0..3) |d| try self.writeInt(minint[d]);
        for (0..3) |d| try self.writeInt(maxint[d]);

        // Phase 2: Compute sizeint, bitsize
        var sizeint: [3]u32 = undefined;
        sizeint[0] = @intCast(maxint[0] - minint[0] + 1);
        sizeint[1] = @intCast(maxint[1] - minint[1] + 1);
        sizeint[2] = @intCast(maxint[2] - minint[2] + 1);

        var bitsizeint: [3]u32 = .{ 0, 0, 0 };
        var bitsize: u32 = 0;
        if ((sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff) {
            bitsizeint[0] = sizeofint(sizeint[0]);
            bitsizeint[1] = sizeofint(sizeint[1]);
            bitsizeint[2] = sizeofint(sizeint[2]);
            bitsize = 0; // flag: use per-dimension sizes
        } else {
            bitsize = sizeofints(3, &sizeint);
        }

        // Phase 3: Find smallidx such that magicints[smallidx] >= mindiff
        var smallidx: i32 = @intCast(FIRSTIDX);
        while (smallidx < @as(i32, @intCast(LASTIDX)) and magicints[@intCast(smallidx)] < @as(u32, @intCast(@max(0, mindiff)))) {
            smallidx += 1;
        }
        if (smallidx >= @as(i32, @intCast(LASTIDX))) return XtcError.CompressionError;
        try self.writeInt(smallidx);

        const tmp_maxidx: i32 = smallidx + 8;
        const maxidx: usize = if (LASTIDX - 1 < @as(usize, @intCast(tmp_maxidx))) LASTIDX - 1 else @intCast(tmp_maxidx);
        const minidx: usize = maxidx - 8;

        var tmp_smaller: i32 = smallidx - 1;
        if (tmp_smaller < @as(i32, @intCast(FIRSTIDX))) tmp_smaller = @intCast(FIRSTIDX);
        var smaller: i32 = @intCast(magicints[@intCast(tmp_smaller)] / 2);
        var smallnum: i32 = @intCast(magicints[@intCast(smallidx)] / 2);
        var sizesmall: [3]u32 = .{
            magicints[@intCast(smallidx)],
            magicints[@intCast(smallidx)],
            magicints[@intCast(smallidx)],
        };
        var larger: i32 = @intCast(magicints[maxidx] / 2);

        // Phase 4: Main compression loop
        var i: usize = 0;
        var prevrun: i32 = -1;
        var prevcoord: [3]i32 = .{ 0, 0, 0 };

        while (i < lsize_u) {
            const thiscoord = buf1[i * 3 ..][0..3];

            // Determine is_smaller: can we shrink the smallidx?
            var is_smaller: i32 = 0;
            if (smallidx < @as(i32, @intCast(maxidx)) and i >= 1 and
                @abs(thiscoord[0] - prevcoord[0]) < larger and
                @abs(thiscoord[1] - prevcoord[1]) < larger and
                @abs(thiscoord[2] - prevcoord[2]) < larger)
            {
                is_smaller = 1;
            } else if (smallidx > @as(i32, @intCast(minidx))) {
                is_smaller = -1;
            }

            // Check is_small: next atom within smallnum delta (water molecule optimization)
            var is_small: i32 = 0;
            if (i + 1 < lsize_u) {
                const nextcoord = buf1[(i + 1) * 3 ..][0..3];
                if (@abs(thiscoord[0] - nextcoord[0]) < smallnum and
                    @abs(thiscoord[1] - nextcoord[1]) < smallnum and
                    @abs(thiscoord[2] - nextcoord[2]) < smallnum)
                {
                    // Swap first and second atom for better water compression
                    const t0 = thiscoord[0];
                    thiscoord[0] = nextcoord[0];
                    nextcoord[0] = t0;
                    const t1 = thiscoord[1];
                    thiscoord[1] = nextcoord[1];
                    nextcoord[1] = t1;
                    const t2 = thiscoord[2];
                    thiscoord[2] = nextcoord[2];
                    nextcoord[2] = t2;
                    is_small = 1;
                }
            }

            // Encode absolute coordinate (relative to minint)
            var tmpcoord: [30]u32 = undefined;
            tmpcoord[0] = @intCast(thiscoord[0] - minint[0]);
            tmpcoord[1] = @intCast(thiscoord[1] - minint[1]);
            tmpcoord[2] = @intCast(thiscoord[2] - minint[2]);
            if (bitsize == 0) {
                encodebits(buf2, bitsizeint[0], tmpcoord[0]);
                encodebits(buf2, bitsizeint[1], tmpcoord[1]);
                encodebits(buf2, bitsizeint[2], tmpcoord[2]);
            } else {
                encodeints(buf2, 3, bitsize, &sizeint, tmpcoord[0..3]);
            }

            prevcoord[0] = thiscoord[0];
            prevcoord[1] = thiscoord[1];
            prevcoord[2] = thiscoord[2];
            i += 1;

            // Build run of "small" delta-encoded atoms
            var run: i32 = 0;
            if (is_small == 0 and is_smaller == -1) {
                is_smaller = 0;
            }
            while (is_small != 0 and run < 8 * 3) {
                const rc = buf1[i * 3 ..][0..3];

                // Check if we should stop shrinking
                if (is_smaller == -1) {
                    const tmpsum: i32 = blk: {
                        var s: i32 = 0;
                        for (0..3) |d| {
                            const dt = rc[d] - prevcoord[d];
                            s += dt * dt;
                        }
                        break :blk s;
                    };
                    if (tmpsum >= smaller * smaller) {
                        is_smaller = 0;
                    }
                }

                // Encode delta relative to prevcoord + smallnum offset
                const run_idx: usize = @intCast(run);
                tmpcoord[run_idx] = @intCast(rc[0] - prevcoord[0] + smallnum);
                tmpcoord[run_idx + 1] = @intCast(rc[1] - prevcoord[1] + smallnum);
                tmpcoord[run_idx + 2] = @intCast(rc[2] - prevcoord[2] + smallnum);

                prevcoord[0] = rc[0];
                prevcoord[1] = rc[1];
                prevcoord[2] = rc[2];

                run += 3;
                i += 1;

                // Check if next atom is also small
                is_small = 0;
                if (i < lsize_u) {
                    const nc = buf1[i * 3 ..][0..3];
                    if (@abs(nc[0] - prevcoord[0]) < smallnum and
                        @abs(nc[1] - prevcoord[1]) < smallnum and
                        @abs(nc[2] - prevcoord[2]) < smallnum)
                    {
                        is_small = 1;
                    }
                }
            }

            // Encode run metadata: 1 flag bit + optional 5-bit run+is_smaller+1
            if (run != prevrun or is_smaller != 0) {
                prevrun = run;
                encodebits(buf2, 1, 1); // flag: run changed
                encodebits(buf2, 5, @intCast(run + is_smaller + 1));
            } else {
                encodebits(buf2, 1, 0); // flag: run unchanged
            }

            // Encode run delta coords
            var k: usize = 0;
            while (k < @as(usize, @intCast(run))) : (k += 3) {
                encodeints(buf2, 3, @intCast(smallidx), &sizesmall, tmpcoord[k..][0..3]);
            }

            // Adjust smallidx
            if (is_smaller != 0) {
                smallidx += is_smaller;
                if (is_smaller < 0) {
                    smallnum = smaller;
                    smaller = @intCast(magicints[@intCast(smallidx - 1)] / 2);
                } else {
                    smaller = smallnum;
                    smallnum = @intCast(magicints[@intCast(smallidx)] / 2);
                }
                sizesmall[0] = magicints[@intCast(smallidx)];
                sizesmall[1] = magicints[@intCast(smallidx)];
                sizesmall[2] = magicints[@intCast(smallidx)];
                larger = @intCast(magicints[maxidx] / 2);
            }
        }

        // Flush: if there are pending bits, round up byte count
        if (buf2[1] != 0) {
            buf2[0] += 1;
        }

        // Write compressed data length (bytes) and payload
        try self.writeInt(buf2[0]);
        const data_len_u: usize = @intCast(buf2[0]);
        const cbuf: [*]const u8 = @ptrCast(@alignCast(buf2.ptr + 3));
        try self.writeOpaque(cbuf[0..data_len_u]);
    }

    // ============================================
    // Internal I/O
    // ============================================

    inline fn io(self: *Self) *std.Io.Writer {
        return &self.writer.interface;
    }

    fn writeInt(self: *Self, value: i32) !void {
        const bytes = std.mem.toBytes(std.mem.nativeToBig(u32, @bitCast(value)));
        self.io().writeAll(&bytes) catch return XtcError.WriteError;
    }

    fn writeFloat(self: *Self, value: f32) !void {
        const bytes = std.mem.toBytes(std.mem.nativeToBig(u32, @bitCast(value)));
        self.io().writeAll(&bytes) catch return XtcError.WriteError;
    }

    fn writeOpaque(self: *Self, data: []const u8) !void {
        self.io().writeAll(data) catch return XtcError.WriteError;
        // XDR opaque data must be padded to 4-byte boundary
        const pad = (4 - (data.len % 4)) % 4;
        if (pad > 0) {
            const zeros = [_]u8{0} ** 4;
            self.io().writeAll(zeros[0..pad]) catch return XtcError.WriteError;
        }
    }
};

// ============================================
// Tests
// ============================================

test "XtcReader open non-existent file" {
    const allocator = std.testing.allocator;
    const result = XtcReader.open(std.testing.io, allocator, "non_existent.xtc");
    try std.testing.expectError(XtcError.FileNotFound, result);
}

test "XtcReader open directory returns IsDir" {
    const allocator = std.testing.allocator;
    const result = XtcReader.open(std.testing.io, allocator, "test_data");
    try std.testing.expectError(XtcError.IsDir, result);
}

test "XtcWriter create on directory path returns IsDir" {
    const allocator = std.testing.allocator;
    const result = XtcWriter.open(std.testing.io, allocator, "test_data", 10, .write);
    try std.testing.expectError(XtcError.IsDir, result);
}

test "sizeofint" {
    try std.testing.expectEqual(@as(u32, 0), sizeofint(0));
    try std.testing.expectEqual(@as(u32, 1), sizeofint(1));
    try std.testing.expectEqual(@as(u32, 8), sizeofint(255));
    try std.testing.expectEqual(@as(u32, 9), sizeofint(256));
}

test "sizeofints" {
    const sizes1 = [_]u32{ 21801, 21008, 15514 };
    try std.testing.expectEqual(@as(u32, 43), sizeofints(3, &sizes1));

    const sizes2 = [_]u32{ 2048, 2048, 2048 };
    try std.testing.expectEqual(@as(u32, 34), sizeofints(3, &sizes2));
}

test "magicints table" {
    try std.testing.expectEqual(@as(u32, 8), magicints[9]);
    try std.testing.expectEqual(@as(u32, 1024), magicints[30]);
    try std.testing.expectEqual(@as(u32, 16777216), magicints[72]);
}

test "read 1l2y.xtc first frame" {
    const allocator = std.testing.allocator;

    var reader = try XtcReader.open(std.testing.io, allocator, "test_data/1l2y.xtc");
    defer reader.close();

    const natoms = reader.getNumAtoms();
    try std.testing.expectEqual(@as(i32, 304), natoms);

    var frame = try reader.readFrame();
    defer frame.deinit(allocator);

    try std.testing.expectEqual(@as(i32, 1), frame.step);

    const tolerance: f32 = 0.0001;

    // atom[0]
    try std.testing.expectApproxEqAbs(@as(f32, -0.8901), frame.coords[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4127), frame.coords[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, -0.0555), frame.coords[2], tolerance);

    // atom[1]
    try std.testing.expectApproxEqAbs(@as(f32, -0.8608), frame.coords[3], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3135), frame.coords[4], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1618), frame.coords[5], tolerance);

    // atom[152]
    try std.testing.expectApproxEqAbs(@as(f32, -0.3502), frame.coords[152 * 3], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, -0.415), frame.coords[152 * 3 + 1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, -0.4813), frame.coords[152 * 3 + 2], tolerance);

    // atom[302]
    try std.testing.expectApproxEqAbs(@as(f32, 0.1636), frame.coords[302 * 3], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1959), frame.coords[302 * 3 + 1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1824), frame.coords[302 * 3 + 2], tolerance);

    // atom[303] (last atom)
    try std.testing.expectApproxEqAbs(@as(f32, 0.2831), frame.coords[303 * 3], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.004), frame.coords[303 * 3 + 1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2676), frame.coords[303 * 3 + 2], tolerance);
}

test "read 1l2y.xtc all frames" {
    const allocator = std.testing.allocator;

    var reader = try XtcReader.open(std.testing.io, allocator, "test_data/1l2y.xtc");
    defer reader.close();

    var frame_count: usize = 0;
    while (true) {
        var frame = reader.readFrame() catch |err| {
            if (err == XtcError.EndOfFile) break;
            return err;
        };
        defer frame.deinit(allocator);
        frame_count += 1;

        try std.testing.expectEqual(@as(usize, 304 * 3), frame.coords.len);
    }

    try std.testing.expectEqual(@as(usize, 38), frame_count);
}

test "read large xtc (6qfk 90MB)" {
    const allocator = std.testing.allocator;

    var reader = XtcReader.open(std.testing.io, allocator, "benchmarks/md_data/6qfk_A_analysis/6qfk_A_R1.xtc") catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer reader.close();

    const natoms: usize = @intCast(reader.getNumAtoms());
    try std.testing.expectEqual(@as(usize, 20391), natoms);

    const tolerance: f32 = 0.0001;
    var frame_count: usize = 0;

    while (true) {
        var frame = reader.readFrame() catch |err| {
            if (err == XtcError.EndOfFile) break;
            return err;
        };
        defer frame.deinit(allocator);
        frame_count += 1;

        if (frame_count == 1) {
            try std.testing.expectApproxEqAbs(@as(f32, 9.561), frame.coords[0], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 0.786), frame.coords[1], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 1.222), frame.coords[2], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 9.318), frame.coords[20390 * 3], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 5.342), frame.coords[20390 * 3 + 1], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 7.609), frame.coords[20390 * 3 + 2], tolerance);
        } else if (frame_count == 100) {
            try std.testing.expectApproxEqAbs(@as(f32, 10.7114), frame.coords[0], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 2.4758), frame.coords[1], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 1.8902), frame.coords[2], tolerance);
        } else if (frame_count == 1000) {
            try std.testing.expectApproxEqAbs(@as(f32, 9.830999), frame.coords[0], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 1.4728), frame.coords[1], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 3.9008), frame.coords[2], tolerance);
        }
    }

    try std.testing.expectEqual(@as(usize, 1001), frame_count);
}

test "read very large xtc (5ltj 511MB)" {
    const allocator = std.testing.allocator;

    var reader = XtcReader.open(std.testing.io, allocator, "benchmarks/md_data/5ltj_A_protein/5ltj_A_prod_R1_fit.xtc") catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer reader.close();

    const natoms: usize = @intCast(reader.getNumAtoms());
    try std.testing.expectEqual(@as(usize, 11487), natoms);

    const tolerance: f32 = 0.0001;
    var frame_count: usize = 0;

    while (true) {
        var frame = reader.readFrame() catch |err| {
            if (err == XtcError.EndOfFile) break;
            return err;
        };
        defer frame.deinit(allocator);
        frame_count += 1;

        if (frame_count == 1) {
            try std.testing.expectApproxEqAbs(@as(f32, 1.626), frame.coords[0], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 5.291), frame.coords[1], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 5.627), frame.coords[2], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 8.840), frame.coords[11486 * 3], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 2.650), frame.coords[11486 * 3 + 1], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, -0.004), frame.coords[11486 * 3 + 2], tolerance);
        } else if (frame_count == 1000) {
            try std.testing.expectApproxEqAbs(@as(f32, 0.902), frame.coords[0], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 5.5208), frame.coords[1], tolerance);
            try std.testing.expectApproxEqAbs(@as(f32, 6.3102), frame.coords[2], tolerance);
        }
    }

    try std.testing.expectEqual(@as(usize, 10001), frame_count);
}

test "XtcWriter round-trip: 3 atoms (small path, no compression)" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/xtc_tmp_xtc_write_3.xtc";

    {
        var writer = try XtcWriter.open(std.testing.io, allocator, tmp_path, 3, .write);
        defer writer.close() catch {};

        var coords = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
        const frame = XtcFrame{
            .step = 42,
            .time = 10.5,
            .box = [3][3]f32{
                .{ 2.0, 0.0, 0.0 },
                .{ 0.0, 3.0, 0.0 },
                .{ 0.0, 0.0, 4.0 },
            },
            .coords = &coords,
            .precision = 1000.0,
        };
        try writer.writeFrame(frame);
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    var reader = try XtcReader.open(std.testing.io, allocator, tmp_path);
    defer reader.close();

    try std.testing.expectEqual(@as(i32, 3), reader.getNumAtoms());

    var frame = try reader.readFrame();
    defer frame.deinit(allocator);

    try std.testing.expectEqual(@as(i32, 42), frame.step);
    try std.testing.expectApproxEqAbs(@as(f32, 10.5), frame.time, 0.001);
    // Small path writes raw floats, so exact match is expected
    const tolerance: f32 = 0.002;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), frame.coords[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), frame.coords[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), frame.coords[2], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), frame.coords[4], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), frame.coords[8], tolerance);
}

test "XtcWriter round-trip: 20 atoms (full compression path)" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/xtc_tmp_xtc_write_20.xtc";

    const natoms = 20;
    var src_coords: [natoms * 3]f32 = undefined;
    for (0..natoms) |a| {
        src_coords[a * 3 + 0] = @as(f32, @floatFromInt(a)) * 0.35 + 0.1;
        src_coords[a * 3 + 1] = @as(f32, @floatFromInt(a)) * 0.12 - 0.5;
        src_coords[a * 3 + 2] = @as(f32, @floatFromInt(a)) * 0.27 + 0.3;
    }

    {
        var writer = try XtcWriter.open(std.testing.io, allocator, tmp_path, natoms, .write);
        defer writer.close() catch {};

        const frame = XtcFrame{
            .step = 1,
            .time = 0.0,
            .box = [3][3]f32{
                .{ 5.0, 0.0, 0.0 },
                .{ 0.0, 5.0, 0.0 },
                .{ 0.0, 0.0, 5.0 },
            },
            .coords = &src_coords,
            .precision = 1000.0,
        };
        try writer.writeFrame(frame);
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    var reader = try XtcReader.open(std.testing.io, allocator, tmp_path);
    defer reader.close();

    try std.testing.expectEqual(@as(i32, natoms), reader.getNumAtoms());

    var frame = try reader.readFrame();
    defer frame.deinit(allocator);

    try std.testing.expectEqual(@as(i32, 1), frame.step);
    // XTC is lossy — tolerance based on precision=1000 (1/1000 nm = 0.001 nm)
    const tolerance: f32 = 0.002;
    for (0..natoms * 3) |k| {
        try std.testing.expectApproxEqAbs(src_coords[k], frame.coords[k], tolerance);
    }
}

test "XtcWriter multi-frame: write 5 frames, verify step and coord values" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/xtc_tmp_xtc_multiframe.xtc";

    const natoms = 20;
    const num_frames = 5;

    // Build source data: each frame has distinct step/time/coords
    var src_coords: [num_frames][natoms * 3]f32 = undefined;
    for (0..num_frames) |f| {
        for (0..natoms) |a| {
            src_coords[f][a * 3 + 0] = @as(f32, @floatFromInt(f * natoms + a)) * 0.1 + 0.05;
            src_coords[f][a * 3 + 1] = @as(f32, @floatFromInt(f * natoms + a)) * 0.15 - 0.3;
            src_coords[f][a * 3 + 2] = @as(f32, @floatFromInt(f * natoms + a)) * 0.08 + 0.2;
        }
    }

    {
        var writer = try XtcWriter.open(std.testing.io, allocator, tmp_path, natoms, .write);
        defer writer.close() catch {};

        for (0..num_frames) |f| {
            const frame = XtcFrame{
                .step = @intCast(f * 100 + 10),
                .time = @as(f32, @floatFromInt(f)) * 2.5,
                .box = [3][3]f32{
                    .{ 4.0, 0.0, 0.0 },
                    .{ 0.0, 4.0, 0.0 },
                    .{ 0.0, 0.0, 4.0 },
                },
                .coords = &src_coords[f],
                .precision = 1000.0,
            };
            try writer.writeFrame(frame);
        }
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    var reader = try XtcReader.open(std.testing.io, allocator, tmp_path);
    defer reader.close();

    try std.testing.expectEqual(@as(i32, natoms), reader.getNumAtoms());

    // Read back all 5 frames and verify step, time, and a sample coordinate
    const tolerance: f32 = 0.002;
    for (0..num_frames) |f| {
        var frame = try reader.readFrame();
        defer frame.deinit(allocator);

        try std.testing.expectEqual(@as(i32, @intCast(f * 100 + 10)), frame.step);
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(f)) * 2.5, frame.time, 0.001);
        try std.testing.expectEqual(@as(usize, natoms * 3), frame.coords.len);
        // Check first and last coordinates of each frame
        try std.testing.expectApproxEqAbs(src_coords[f][0], frame.coords[0], tolerance);
        try std.testing.expectApproxEqAbs(src_coords[f][natoms * 3 - 1], frame.coords[natoms * 3 - 1], tolerance);
    }

    // Verify EOF on the next read
    const eof_err = reader.readFrame();
    try std.testing.expectError(XtcError.EndOfFile, eof_err);
}

test "XtcWriter append mode: write 2 then append 3, read all 5" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/xtc_tmp_xtc_append.xtc";

    const natoms = 20;

    var src_coords: [5][natoms * 3]f32 = undefined;
    for (0..5) |f| {
        for (0..natoms * 3) |k| {
            src_coords[f][k] = @as(f32, @floatFromInt(f * 10 + k)) * 0.01 + 0.1;
        }
    }

    // Write first 2 frames
    {
        var writer = try XtcWriter.open(std.testing.io, allocator, tmp_path, natoms, .write);
        defer writer.close() catch {};

        for (0..2) |f| {
            const frame = XtcFrame{
                .step = @intCast(f + 1),
                .time = @as(f32, @floatFromInt(f)) * 1.0,
                .box = [3][3]f32{
                    .{ 3.0, 0.0, 0.0 },
                    .{ 0.0, 3.0, 0.0 },
                    .{ 0.0, 0.0, 3.0 },
                },
                .coords = &src_coords[f],
                .precision = 1000.0,
            };
            try writer.writeFrame(frame);
        }
    }

    // Append 3 more frames
    {
        var writer = try XtcWriter.open(std.testing.io, allocator, tmp_path, natoms, .append);
        defer writer.close() catch {};

        for (2..5) |f| {
            const frame = XtcFrame{
                .step = @intCast(f + 1),
                .time = @as(f32, @floatFromInt(f)) * 1.0,
                .box = [3][3]f32{
                    .{ 3.0, 0.0, 0.0 },
                    .{ 0.0, 3.0, 0.0 },
                    .{ 0.0, 0.0, 3.0 },
                },
                .coords = &src_coords[f],
                .precision = 1000.0,
            };
            try writer.writeFrame(frame);
        }
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    var reader = try XtcReader.open(std.testing.io, allocator, tmp_path);
    defer reader.close();

    try std.testing.expectEqual(@as(i32, natoms), reader.getNumAtoms());

    // All 5 frames must be readable with correct step numbers
    const tolerance: f32 = 0.002;
    for (0..5) |f| {
        var frame = try reader.readFrame();
        defer frame.deinit(allocator);

        try std.testing.expectEqual(@as(i32, @intCast(f + 1)), frame.step);
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(f)) * 1.0, frame.time, 0.001);
        try std.testing.expectApproxEqAbs(src_coords[f][0], frame.coords[0], tolerance);
    }

    const eof_err = reader.readFrame();
    try std.testing.expectError(XtcError.EndOfFile, eof_err);
}

test "XtcWriter round-trip with 1l2y.xtc (304 atoms, all 38 frames)" {
    const allocator = std.testing.allocator;
    const src_path = "test_data/1l2y.xtc";
    const tmp_path = "test_data/xtc_tmp_xtc_roundtrip_1l2y.xtc";

    // 1l2y.xtc has exactly 38 frames; read and store them all
    const expected_frames = 38;
    var src_frames: [expected_frames]XtcFrame = undefined;
    var src_frame_count: usize = 0;

    {
        var src_reader = try XtcReader.open(std.testing.io, allocator, src_path);
        defer src_reader.close();

        const natoms = src_reader.getNumAtoms();
        try std.testing.expectEqual(@as(i32, 304), natoms);

        while (src_frame_count < expected_frames) {
            const frame = src_reader.readFrame() catch |err| {
                if (err == XtcError.EndOfFile) break;
                return err;
            };
            src_frames[src_frame_count] = frame;
            src_frame_count += 1;
        }
    }

    defer {
        for (0..src_frame_count) |i| src_frames[i].deinit(allocator);
    }

    try std.testing.expectEqual(expected_frames, src_frame_count);

    // Determine tolerance from the precision of the first frame (XTC is lossy)
    const precision = src_frames[0].precision;
    const tolerance: f32 = 1.0 / precision + 0.0005;

    // Write all frames to a temporary file
    {
        var writer = try XtcWriter.open(std.testing.io, allocator, tmp_path, @as(i32, 304), .write);
        defer writer.close() catch {};

        for (0..src_frame_count) |i| {
            try writer.writeFrame(src_frames[i]);
        }
    }

    defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

    // Read back and compare every frame coordinate-by-coordinate
    var dst_reader = try XtcReader.open(std.testing.io, allocator, tmp_path);
    defer dst_reader.close();

    try std.testing.expectEqual(@as(i32, 304), dst_reader.getNumAtoms());

    for (0..src_frame_count) |i| {
        const src_frame = src_frames[i];
        var dst_frame = try dst_reader.readFrame();
        defer dst_frame.deinit(allocator);

        try std.testing.expectEqual(src_frame.step, dst_frame.step);
        try std.testing.expectApproxEqAbs(src_frame.time, dst_frame.time, 0.001);
        try std.testing.expectEqual(src_frame.coords.len, dst_frame.coords.len);

        for (src_frame.coords, dst_frame.coords) |src_c, dst_c| {
            try std.testing.expectApproxEqAbs(src_c, dst_c, tolerance);
        }
    }

    const eof_err = dst_reader.readFrame();
    try std.testing.expectError(XtcError.EndOfFile, eof_err);
}

test "XtcWriter error paths" {
    const allocator = std.testing.allocator;

    // natoms <= 0 must return InvalidAtomCount
    {
        const result = XtcWriter.open(std.testing.io, allocator, "test_data/xtc_tmp_xtc_err_zero.xtc", 0, .write);
        try std.testing.expectError(XtcError.InvalidAtomCount, result);

        const result_neg = XtcWriter.open(std.testing.io, allocator, "test_data/xtc_tmp_xtc_err_neg.xtc", -5, .write);
        try std.testing.expectError(XtcError.InvalidAtomCount, result_neg);
    }

    // coords length mismatch must return CompressionError
    {
        const tmp_path = "test_data/xtc_tmp_xtc_err_coords.xtc";

        var writer = try XtcWriter.open(std.testing.io, allocator, tmp_path, 20, .write);
        defer writer.close() catch {};
        defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

        // Provide only 3 coords instead of 60
        var short_coords = [_]f32{ 1.0, 2.0, 3.0 };
        const bad_frame = XtcFrame{
            .step = 1,
            .time = 0.0,
            .box = [3][3]f32{
                .{ 5.0, 0.0, 0.0 },
                .{ 0.0, 5.0, 0.0 },
                .{ 0.0, 0.0, 5.0 },
            },
            .coords = &short_coords,
            .precision = 1000.0,
        };
        const write_result = writer.writeFrame(bad_frame);
        try std.testing.expectError(XtcError.CompressionError, write_result);
    }

    // Append with mismatched natoms must return InvalidAtomCount
    {
        const tmp_path = "test_data/xtc_tmp_xtc_err_natoms_mismatch.xtc";

        // Create file with natoms=20
        {
            var writer = try XtcWriter.open(std.testing.io, allocator, tmp_path, 20, .write);
            defer writer.close() catch {};

            var coords: [20 * 3]f32 = undefined;
            for (0..20 * 3) |k| coords[k] = @as(f32, @floatFromInt(k)) * 0.1;
            const frame = XtcFrame{
                .step = 1,
                .time = 0.0,
                .box = [3][3]f32{
                    .{ 5.0, 0.0, 0.0 },
                    .{ 0.0, 5.0, 0.0 },
                    .{ 0.0, 0.0, 5.0 },
                },
                .coords = &coords,
                .precision = 1000.0,
            };
            try writer.writeFrame(frame);
        }

        defer std.Io.Dir.cwd().deleteFile(std.testing.io, tmp_path) catch {};

        // Try to append with natoms=10 (mismatch — file has 20)
        const append_result = XtcWriter.open(std.testing.io, allocator, tmp_path, 10, .append);
        try std.testing.expectError(XtcError.InvalidAtomCount, append_result);
    }
}
