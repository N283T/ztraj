//! AMBER NetCDF trajectory reader.
//!
//! Reads AMBER convention NetCDF-3 trajectory files (.nc, .ncdf).
//! Both classic (CDF-1) and 64-bit offset (CDF-2) formats are supported.
//! Coordinates are stored in angstroms — no unit conversion needed.
//!
//! ## NetCDF-3 Format
//!
//! The file uses big-endian byte order throughout:
//! - Magic: "CDF" + version byte (0x01 = classic, 0x02 = 64-bit offset)
//! - Header: dimensions, global attributes, variable definitions
//! - Data: fixed-size and record (unlimited) variables
//!
//! ## AMBER Convention Variables
//!
//! - coordinates (frame, atom, spatial) float32  — atom positions in angstroms
//! - time        (frame)               float32  — required per spec; absent in some files
//! - cell_lengths (frame, cell_spatial) float64  — box edge lengths in angstroms (optional)
//! - cell_angles  (frame, cell_angular) float64  — box angles in degrees (optional)
//!
//! Reference: http://ambermd.org/netcdf/nctraj.xhtml

const std = @import("std");
const types = @import("../types.zig");

pub const NcError = error{
    FileNotFound,
    InvalidMagic,
    BadVersion,
    NotAmberConvention,
    MissingVariable,
    BadDimension,
    ReadError,
    EndOfFile,
    OutOfMemory,
};

// NetCDF-3 header tag constants
const NC_DIMENSION: u32 = 0x0000_000A;
const NC_VARIABLE: u32 = 0x0000_000B;
const NC_ATTRIBUTE: u32 = 0x0000_000C;

// NetCDF-3 type constants
const NC_BYTE: u32 = 1;
const NC_CHAR: u32 = 2;
const NC_SHORT: u32 = 3;
const NC_INT: u32 = 4;
const NC_FLOAT: u32 = 5;
const NC_DOUBLE: u32 = 6;

/// Maximum name length accepted from NetCDF headers (defense against malicious files).
const MAX_NAME_LEN: u32 = 8192;

/// Size in bytes of each NetCDF type. Returns error for unrecognized types.
fn ncTypeSize(nc_type: u32) !u32 {
    return switch (nc_type) {
        NC_BYTE, NC_CHAR => 1,
        NC_SHORT => 2,
        NC_INT, NC_FLOAT => 4,
        NC_DOUBLE => 8,
        else => NcError.ReadError,
    };
}

/// Variable descriptor parsed from the NetCDF header.
const VarInfo = struct {
    /// Data offset in the file (32-bit for CDF-1, 64-bit for CDF-2; stored as u64).
    offset: u64,
    /// NetCDF type (NC_FLOAT, NC_DOUBLE, etc.).
    nc_type: u32,
    /// Size of one record entry in bytes (for record variables).
    /// For non-record variables, this is the total size.
    var_size: u64,
};

/// Streaming AMBER NetCDF trajectory reader.
///
/// Usage:
///
///   var reader = try NcReader.open(testIo(), allocator, "trajectory.nc");
///   defer reader.deinit();
///
///   while (try reader.next()) |frame| {
///       _ = frame.x[0]; // coordinates in angstroms
///   }
///
/// The reader reuses a single Frame buffer. The returned pointer is valid
/// until the next call to next() or deinit().
pub const NcReader = struct {
    io: std.Io,
    file: std.Io.File,
    reader: std.Io.File.Reader,
    read_buffer: []u8,
    allocator: std.mem.Allocator,
    frame: types.Frame,
    /// Per-frame coordinate data buffer (AOS: x0,y0,z0,x1,y1,z1,...)
    coord_buf: []u8,
    n_atoms: u32,
    n_frames: u32,
    frames_read: u32,
    /// Record stride in bytes: total size of one record (recsize in NetCDF spec).
    rec_size: u64,
    /// Variable offsets within the file.
    coords_var: VarInfo,
    time_var: ?VarInfo,
    cell_lengths_var: ?VarInfo,
    cell_angles_var: ?VarInfo,

    const Self = @This();

    /// Open an AMBER NetCDF trajectory for reading.
    pub fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => NcError.FileNotFound,
                else => NcError.ReadError,
            };
        };
        errdefer file.close(io);

        const read_buffer = allocator.alloc(u8, 64 * 1024) catch return NcError.OutOfMemory;
        errdefer allocator.free(read_buffer);

        var reader = file.reader(io, read_buffer);

        // ---- Magic and version ----
        var magic: [4]u8 = undefined;
        readExact(&reader, &magic) catch return NcError.ReadError;
        if (magic[0] != 'C' or magic[1] != 'D' or magic[2] != 'F')
            return NcError.InvalidMagic;
        if (magic[3] != 1 and magic[3] != 2)
            return NcError.BadVersion;

        const is_64bit = magic[3] == 2;

        // ---- Number of records (unlimited dimension length) ----
        const numrecs = fileReadU32(&reader) catch return NcError.ReadError;

        // ---- Parse dimensions ----
        const DimInfo = struct { size: u32, is_unlimited: bool };
        var dims = std.ArrayList(DimInfo).empty;
        defer dims.deinit(allocator);

        var n_atoms: u32 = 0;
        var n_frames: u32 = 0;
        var spatial_size: u32 = 0;

        {
            const tag = fileReadU32(&reader) catch return NcError.ReadError;
            const n_dims = fileReadU32(&reader) catch return NcError.ReadError;

            if (tag == NC_DIMENSION and n_dims > 0) {
                for (0..n_dims) |_| {
                    const name = try readNcName(&reader, allocator);
                    defer allocator.free(name);
                    const dim_len = fileReadU32(&reader) catch return NcError.ReadError;

                    const is_unlim = dim_len == 0;
                    const size = if (is_unlim) numrecs else dim_len;

                    // Match dimension names directly (no hashing)
                    if (std.mem.eql(u8, name, "frame")) n_frames = size;
                    if (std.mem.eql(u8, name, "spatial")) spatial_size = size;
                    if (std.mem.eql(u8, name, "atom")) n_atoms = size;

                    try dims.append(allocator, .{
                        .size = size,
                        .is_unlimited = is_unlim,
                    });
                }
            } else if (tag == 0 and n_dims == 0) {
                // ABSENT — no dimensions
            } else {
                return NcError.ReadError;
            }
        }

        if (n_atoms == 0) return NcError.BadDimension;
        if (spatial_size != 3) return NcError.BadDimension;

        // ---- Parse global attributes ----
        var is_amber = false;
        {
            const tag = fileReadU32(&reader) catch return NcError.ReadError;
            const n_attrs = fileReadU32(&reader) catch return NcError.ReadError;

            if (tag == NC_ATTRIBUTE and n_attrs > 0) {
                for (0..n_attrs) |_| {
                    const name = try readNcName(&reader, allocator);
                    defer allocator.free(name);
                    const attr_type = fileReadU32(&reader) catch return NcError.ReadError;
                    const attr_nelems = fileReadU32(&reader) catch return NcError.ReadError;

                    const type_size = try ncTypeSize(attr_type);
                    const attr_bytes: u64 = @as(u64, attr_nelems) * type_size;
                    const padded: u64 = (attr_bytes + 3) & ~@as(u64, 3); // 4-byte aligned

                    if (std.mem.eql(u8, name, "Conventions")) {
                        if (attr_type == NC_CHAR and attr_nelems <= 64) {
                            var buf: [64]u8 = undefined;
                            readExact(&reader, buf[0..@intCast(padded)]) catch return NcError.ReadError;
                            const val = std.mem.trim(u8, buf[0..attr_nelems], " \x00");
                            if (std.mem.eql(u8, val, "AMBER")) is_amber = true;
                            continue;
                        }
                    }
                    // Skip attribute data
                    reader.seekBy(@intCast(padded)) catch return NcError.ReadError;
                }
            } else if (tag == 0 and n_attrs == 0) {
                // ABSENT
            } else {
                return NcError.ReadError;
            }
        }

        if (!is_amber) return NcError.NotAmberConvention;

        // ---- Parse variables ----
        var coords_var: ?VarInfo = null;
        var time_var: ?VarInfo = null;
        var cell_lengths_var: ?VarInfo = null;
        var cell_angles_var: ?VarInfo = null;
        var rec_size: u64 = 0;

        {
            const tag = fileReadU32(&reader) catch return NcError.ReadError;
            const n_vars = fileReadU32(&reader) catch return NcError.ReadError;

            if (tag == NC_VARIABLE and n_vars > 0) {
                for (0..n_vars) |_| {
                    const name = try readNcName(&reader, allocator);
                    defer allocator.free(name);

                    // Number of dimensions
                    const n_var_dims = fileReadU32(&reader) catch return NcError.ReadError;
                    var is_record_var = false;

                    for (0..n_var_dims) |_| {
                        const dim_id = fileReadU32(&reader) catch return NcError.ReadError;
                        if (dim_id >= dims.items.len) return NcError.BadDimension;
                        if (dims.items[dim_id].is_unlimited) is_record_var = true;
                    }

                    // Skip variable attributes
                    const va_tag = fileReadU32(&reader) catch return NcError.ReadError;
                    const va_count = fileReadU32(&reader) catch return NcError.ReadError;
                    if (va_tag == NC_ATTRIBUTE and va_count > 0) {
                        for (0..va_count) |_| {
                            try skipNcName(&reader);
                            const at = fileReadU32(&reader) catch return NcError.ReadError;
                            const an = fileReadU32(&reader) catch return NcError.ReadError;
                            const ts = try ncTypeSize(at);
                            const ab: u64 = @as(u64, an) * ts;
                            const ap: u64 = (ab + 3) & ~@as(u64, 3);
                            reader.seekBy(@intCast(ap)) catch return NcError.ReadError;
                        }
                    }

                    const nc_type = fileReadU32(&reader) catch return NcError.ReadError;
                    // vsize: per-record size for record vars, total size for fixed vars
                    const vsize = fileReadU32(&reader) catch return NcError.ReadError;

                    // begin: data offset (4 bytes for CDF-1, 8 bytes for CDF-2)
                    const offset: u64 = if (is_64bit)
                        fileReadU64(&reader) catch return NcError.ReadError
                    else
                        fileReadU32(&reader) catch return NcError.ReadError;

                    const info = VarInfo{
                        .offset = offset,
                        .nc_type = nc_type,
                        .var_size = vsize,
                    };

                    // Match variable names directly (no hashing)
                    if (std.mem.eql(u8, name, "coordinates")) coords_var = info;
                    if (std.mem.eql(u8, name, "time")) time_var = info;
                    if (std.mem.eql(u8, name, "cell_lengths")) cell_lengths_var = info;
                    if (std.mem.eql(u8, name, "cell_angles")) cell_angles_var = info;

                    if (is_record_var) {
                        rec_size += vsize;
                    }
                }
            }
        }

        if (coords_var == null) return NcError.MissingVariable;
        // Validate coordinate type is float32
        if (coords_var.?.nc_type != NC_FLOAT) return NcError.MissingVariable;

        // Allocate frame and coordinate buffer (u64 arithmetic to avoid overflow)
        const coord_bytes: usize = @as(usize, n_atoms) * 3 * 4; // float32 AOS
        const coord_buf = allocator.alloc(u8, coord_bytes) catch return NcError.OutOfMemory;
        errdefer allocator.free(coord_buf);

        var frame = types.Frame.init(allocator, n_atoms) catch return NcError.OutOfMemory;
        errdefer frame.deinit();

        return Self{
            .io = io,
            .file = file,
            .reader = reader,
            .read_buffer = read_buffer,
            .allocator = allocator,
            .frame = frame,
            .coord_buf = coord_buf,
            .n_atoms = n_atoms,
            .n_frames = n_frames,
            .frames_read = 0,
            .rec_size = rec_size,
            .coords_var = coords_var.?,
            .time_var = time_var,
            .cell_lengths_var = cell_lengths_var,
            .cell_angles_var = cell_angles_var,
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
        return self.n_atoms;
    }

    /// Number of frames reported in the header.
    pub fn nFrames(self: *const Self) u32 {
        return self.n_frames;
    }

    /// Read the next frame.
    ///
    /// Returns a pointer to the internal SOA frame on success.
    /// Returns null when all frames have been read.
    /// The returned pointer is valid until the next call to next() or deinit().
    pub fn next(self: *Self) !?*const types.Frame {
        if (self.frames_read >= self.n_frames) return null;

        const fi: u64 = self.frames_read;

        // ---- Read coordinates ----
        {
            const offset = self.coords_var.offset + fi * self.rec_size;
            self.reader.seekTo(offset) catch return NcError.ReadError;
            readExact(&self.reader, self.coord_buf) catch return NcError.ReadError;

            // Convert AOS big-endian float32 → SOA native f32
            const n: usize = self.n_atoms;
            for (0..n) |ai| {
                const base = ai * 12; // 3 floats * 4 bytes
                self.frame.x[ai] = readBEf32(self.coord_buf[base..][0..4]);
                self.frame.y[ai] = readBEf32(self.coord_buf[base + 4 ..][0..4]);
                self.frame.z[ai] = readBEf32(self.coord_buf[base + 8 ..][0..4]);
            }
        }

        // ---- Read time (optional) ----
        if (self.time_var) |tv| {
            const offset = tv.offset + fi * self.rec_size;
            self.reader.seekTo(offset) catch return NcError.ReadError;
            var buf: [4]u8 = undefined;
            readExact(&self.reader, &buf) catch return NcError.ReadError;
            self.frame.time = readBEf32(&buf);
        } else {
            self.frame.time = 0.0;
        }

        // ---- Read box vectors (optional) ----
        if (self.cell_lengths_var) |cl| {
            if (self.cell_angles_var) |ca| {
                const cl_offset = cl.offset + fi * self.rec_size;
                self.reader.seekTo(cl_offset) catch return NcError.ReadError;
                var lbuf: [24]u8 = undefined; // 3 * f64
                readExact(&self.reader, &lbuf) catch return NcError.ReadError;

                const ca_offset = ca.offset + fi * self.rec_size;
                self.reader.seekTo(ca_offset) catch return NcError.ReadError;
                var abuf: [24]u8 = undefined;
                readExact(&self.reader, &abuf) catch return NcError.ReadError;

                const a: f32 = @floatCast(readBEf64(lbuf[0..8]));
                const b: f32 = @floatCast(readBEf64(lbuf[8..16]));
                const c: f32 = @floatCast(readBEf64(lbuf[16..24]));

                const alpha_deg: f64 = readBEf64(abuf[0..8]);
                const beta_deg: f64 = readBEf64(abuf[8..16]);
                const gamma_deg: f64 = readBEf64(abuf[16..24]);

                self.frame.box_vectors = cellToVectors(a, b, c, alpha_deg, beta_deg, gamma_deg);
            }
        }
        if (self.cell_lengths_var == null or self.cell_angles_var == null) {
            self.frame.box_vectors = null;
        }

        self.frame.step = @intCast(self.frames_read);
        self.frames_read += 1;
        return &self.frame;
    }

    // ================================================================
    // Helpers
    // ================================================================

    /// Convert cell parameters (a, b, c, alpha, beta, gamma) to box vectors.
    /// Follows the standard convention: a along x, b in xy-plane, c completes the basis.
    /// Returns null for degenerate cells (gamma ≈ 0° or 180°).
    fn cellToVectors(a: f32, b: f32, c: f32, alpha_deg: f64, beta_deg: f64, gamma_deg: f64) ?[3][3]f32 {
        const deg2rad: f64 = std.math.pi / 180.0;
        const alpha = alpha_deg * deg2rad;
        const beta = beta_deg * deg2rad;
        const gamma = gamma_deg * deg2rad;

        const cos_alpha = @cos(alpha);
        const cos_beta = @cos(beta);
        const cos_gamma = @cos(gamma);
        const sin_gamma = @sin(gamma);

        if (@abs(sin_gamma) < 1e-10) return null; // degenerate cell

        const bx: f32 = @floatCast(@as(f64, b) * cos_gamma);
        const by: f32 = @floatCast(@as(f64, b) * sin_gamma);
        const cx: f32 = @floatCast(@as(f64, c) * cos_beta);
        const cy_f64 = @as(f64, c) * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
        const cy: f32 = @floatCast(cy_f64);
        const cz: f32 = @floatCast(@sqrt(@max(0.0, @as(f64, c) * @as(f64, c) - @as(f64, cx) * @as(f64, cx) - cy_f64 * cy_f64)));

        return .{
            .{ a, 0.0, 0.0 },
            .{ bx, by, 0.0 },
            .{ cx, cy, cz },
        };
    }
};

// ============================================================================
// NetCDF-3 binary reading helpers (big-endian)
// ============================================================================

fn readBEf32(bytes: *const [4]u8) f32 {
    return @bitCast(std.mem.readInt(u32, bytes, .big));
}

fn readBEf64(bytes: *const [8]u8) f64 {
    return @bitCast(std.mem.readInt(u64, bytes, .big));
}

/// Read exactly buf.len bytes from a file reader.
fn readExact(reader: *std.Io.File.Reader, buf: []u8) !void {
    reader.interface.readSliceAll(buf) catch return NcError.ReadError;
}

fn fileReadU32(reader: *std.Io.File.Reader) !u32 {
    var buf: [4]u8 = undefined;
    try readExact(reader, &buf);
    return std.mem.readInt(u32, &buf, .big);
}

fn fileReadU64(reader: *std.Io.File.Reader) !u64 {
    var buf: [8]u8 = undefined;
    try readExact(reader, &buf);
    return std.mem.readInt(u64, &buf, .big);
}

/// Read a NetCDF name: u32 length, then chars padded to 4-byte boundary.
fn readNcName(reader: *std.Io.File.Reader, allocator: std.mem.Allocator) ![]u8 {
    const name_len = try fileReadU32(reader);
    if (name_len > MAX_NAME_LEN) return NcError.ReadError;
    const padded_len = (name_len + 3) & ~@as(u32, 3);
    const buf = allocator.alloc(u8, padded_len) catch return NcError.OutOfMemory;
    errdefer allocator.free(buf);
    try readExact(reader, buf);
    const result = allocator.alloc(u8, name_len) catch return NcError.OutOfMemory;
    @memcpy(result, buf[0..name_len]);
    allocator.free(buf);
    return result;
}

/// Skip a NetCDF name without allocating.
fn skipNcName(reader: *std.Io.File.Reader) !void {
    const name_len = try fileReadU32(reader);
    if (name_len > MAX_NAME_LEN) return NcError.ReadError;
    const padded_len = (name_len + 3) & ~@as(u32, 3);
    reader.seekBy(@intCast(padded_len)) catch return NcError.ReadError;
}

// ============================================================================
// NcWriter — streaming AMBER NetCDF trajectory writer
// ============================================================================

pub const NcWriteError = error{
    FileCreateFailed,
    WriteError,
    OutOfMemory,
};

/// Streaming AMBER NetCDF trajectory writer.
///
/// Usage:
///
///   var writer = try NcWriter.open(testIo(), allocator, "output.nc", 100);
///   defer writer.deinit();
///   for (frames) |frame| try writer.writeFrame(frame);
///   try writer.close();
///
/// Writes CDF-2 (64-bit offset) format with AMBER conventions.
/// Coordinates must be in angstroms.
pub const NcWriter = struct {
    io: std.Io,
    file: std.Io.File,
    writer: std.Io.File.Writer,
    write_buffer: []u8,
    allocator: std.mem.Allocator,
    n_atoms: u32,
    frames_written: u32,
    /// Reusable buffer for AOS coordinate encoding.
    coord_buf: []u8,
    /// Whether cell info (box vectors) should be written.
    has_cell: bool,
    /// File offset where the numrecs field lives (to update on close).
    numrecs_offset: u64,
    /// File offset where coordinate data begins.
    data_offset: u64,
    /// Per-record stride in bytes.
    rec_size: u64,
    closed: bool = false,

    const Self = @This();

    /// Create a new AMBER NetCDF trajectory file for writing.
    pub fn open(io: std.Io, allocator: std.mem.Allocator, path: []const u8, n_atoms: u32, has_cell: bool) !Self {
        const file = std.Io.Dir.cwd().createFile(io, path, .{}) catch {
            return NcWriteError.FileCreateFailed;
        };
        errdefer file.close(io);

        const coord_bytes: usize = @as(usize, n_atoms) * 3 * 4;
        const coord_buf = allocator.alloc(u8, coord_bytes) catch return NcWriteError.OutOfMemory;
        errdefer allocator.free(coord_buf);

        const write_buffer = allocator.alloc(u8, 64 * 1024) catch return NcWriteError.OutOfMemory;
        errdefer allocator.free(write_buffer);

        var writer = file.writer(io, write_buffer);

        // Write header
        const header_info = writeHeader(&writer, n_atoms, has_cell) catch return NcWriteError.WriteError;

        return Self{
            .io = io,
            .file = file,
            .writer = writer,
            .write_buffer = write_buffer,
            .allocator = allocator,
            .n_atoms = n_atoms,
            .frames_written = 0,
            .coord_buf = coord_buf,
            .has_cell = has_cell,
            .numrecs_offset = header_info.numrecs_offset,
            .data_offset = header_info.data_offset,
            .rec_size = header_info.rec_size,
        };
    }

    /// Write a single frame.
    pub fn writeFrame(self: *Self, frame: types.Frame) !void {
        const n: usize = self.n_atoms;
        const fi: u64 = self.frames_written;
        const file_offset = self.data_offset + fi * self.rec_size;

        // Encode SOA native f32 → AOS big-endian float32
        for (0..n) |ai| {
            const base = ai * 12;
            writeBEf32(self.coord_buf[base..][0..4], frame.x[ai]);
            writeBEf32(self.coord_buf[base + 4 ..][0..4], frame.y[ai]);
            writeBEf32(self.coord_buf[base + 8 ..][0..4], frame.z[ai]);
        }

        self.writer.seekTo(file_offset) catch return NcWriteError.WriteError;
        self.writer.interface.writeAll(self.coord_buf) catch return NcWriteError.WriteError;

        // Pad coordinates to 4-byte boundary
        const coord_bytes: u64 = @as(u64, n) * 12;
        const coord_padded = (coord_bytes + 3) & ~@as(u64, 3);
        if (coord_padded > coord_bytes) {
            const zeros = [_]u8{ 0, 0, 0 };
            const pad_len: usize = @intCast(coord_padded - coord_bytes);
            self.writer.interface.writeAll(zeros[0..pad_len]) catch return NcWriteError.WriteError;
        }

        // Write cell data if present
        if (self.has_cell) {
            // cell_lengths: 3 x f64 (big-endian)
            var lbuf: [24]u8 = undefined;
            var abuf: [24]u8 = undefined;

            if (frame.box_vectors) |box| {
                // Extract lengths from box vectors (diagonal for orthogonal,
                // vector norms for triclinic)
                const a = @sqrt(@as(f64, box[0][0]) * box[0][0] + @as(f64, box[0][1]) * box[0][1] + @as(f64, box[0][2]) * box[0][2]);
                const b = @sqrt(@as(f64, box[1][0]) * box[1][0] + @as(f64, box[1][1]) * box[1][1] + @as(f64, box[1][2]) * box[1][2]);
                const c = @sqrt(@as(f64, box[2][0]) * box[2][0] + @as(f64, box[2][1]) * box[2][1] + @as(f64, box[2][2]) * box[2][2]);

                writeBEf64(lbuf[0..8], a);
                writeBEf64(lbuf[8..16], b);
                writeBEf64(lbuf[16..24], c);

                // Compute angles from box vectors
                const rad2deg: f64 = 180.0 / std.math.pi;
                const alpha = std.math.acos((@as(f64, box[1][0]) * box[2][0] + @as(f64, box[1][1]) * box[2][1] + @as(f64, box[1][2]) * box[2][2]) / (b * c)) * rad2deg;
                const beta = std.math.acos((@as(f64, box[0][0]) * box[2][0] + @as(f64, box[0][1]) * box[2][1] + @as(f64, box[0][2]) * box[2][2]) / (a * c)) * rad2deg;
                const gamma = std.math.acos((@as(f64, box[0][0]) * box[1][0] + @as(f64, box[0][1]) * box[1][1] + @as(f64, box[0][2]) * box[1][2]) / (a * b)) * rad2deg;

                writeBEf64(abuf[0..8], alpha);
                writeBEf64(abuf[8..16], beta);
                writeBEf64(abuf[16..24], gamma);
            } else {
                // No box: write zeros
                @memset(&lbuf, 0);
                @memset(&abuf, 0);
            }

            self.writer.interface.writeAll(&lbuf) catch return NcWriteError.WriteError;
            self.writer.interface.writeAll(&abuf) catch return NcWriteError.WriteError;
        }

        self.frames_written += 1;
    }

    /// Flush and close the file. Updates the frame count in the header.
    pub fn close(self: *Self) !void {
        defer {
            self.allocator.free(self.coord_buf);
            self.coord_buf = &.{};
            self.allocator.free(self.write_buffer);
            self.write_buffer = &.{};
            self.file.close(self.io);
            self.closed = true;
        }
        // Update numrecs (frame count) in the header
        self.writer.seekTo(self.numrecs_offset) catch return NcWriteError.WriteError;
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, self.frames_written, .big);
        self.writer.interface.writeAll(&buf) catch return NcWriteError.WriteError;
        self.writer.interface.flush() catch return NcWriteError.WriteError;
    }

    /// Best-effort cleanup if close() was not called.
    pub fn deinit(self: *Self) void {
        if (!self.closed) {
            self.close() catch {};
        }
    }
};

const HeaderInfo = struct {
    numrecs_offset: u64,
    data_offset: u64,
    rec_size: u64,
};

/// Write a complete AMBER NetCDF-3 (CDF-2) header.
fn writeHeader(writer: *std.Io.File.Writer, n_atoms: u32, has_cell: bool) !HeaderInfo {

    // Magic: CDF version 2 (64-bit offset)
    try writer.interface.writeAll("CDF\x02");

    // numrecs (placeholder — updated on close)
    const numrecs_offset: u64 = 4;
    try fileWriteU32(writer, 0);

    // ---- Dimensions ----
    const n_dims: u32 = if (has_cell) 6 else 3;
    try fileWriteU32(writer, NC_DIMENSION);
    try fileWriteU32(writer, n_dims);

    // dim 0: frame (unlimited)
    try fileWriteName(writer, "frame");
    try fileWriteU32(writer, 0); // 0 = unlimited

    // dim 1: spatial
    try fileWriteName(writer, "spatial");
    try fileWriteU32(writer, 3);

    // dim 2: atom
    try fileWriteName(writer, "atom");
    try fileWriteU32(writer, n_atoms);

    if (has_cell) {
        // dim 3: cell_spatial
        try fileWriteName(writer, "cell_spatial");
        try fileWriteU32(writer, 3);

        // dim 4: label
        try fileWriteName(writer, "label");
        try fileWriteU32(writer, 5);

        // dim 5: cell_angular
        try fileWriteName(writer, "cell_angular");
        try fileWriteU32(writer, 3);
    }

    // ---- Global attributes ----
    try fileWriteU32(writer, NC_ATTRIBUTE);
    try fileWriteU32(writer, 2); // 2 attributes

    // Conventions = "AMBER"
    try fileWriteName(writer, "Conventions");
    try fileWriteU32(writer, NC_CHAR);
    try fileWriteU32(writer, 5);
    try writer.interface.writeAll("AMBER\x00\x00\x00"); // padded to 8 bytes

    // ConventionVersion = "1.0"
    try fileWriteName(writer, "ConventionVersion");
    try fileWriteU32(writer, NC_CHAR);
    try fileWriteU32(writer, 3);
    try writer.interface.writeAll("1.0\x00"); // padded to 4 bytes

    // ---- Variables ----
    const n_vars: u32 = if (has_cell) 5 else 2;
    try fileWriteU32(writer, NC_VARIABLE);
    try fileWriteU32(writer, n_vars);

    // Compute per-record sizes
    const coord_size: u64 = @as(u64, n_atoms) * 3 * 4;
    const coord_padded = (coord_size + 3) & ~@as(u64, 3);
    const spatial_size: u64 = 4; // 3 chars padded to 4
    const cell_lengths_size: u64 = 24; // 3 x f64
    const cell_angles_size: u64 = 24;

    var rec_size: u64 = coord_padded;
    if (has_cell) {
        rec_size += cell_lengths_size + cell_angles_size;
    }

    // We need to compute the header size to determine data offsets.
    // For simplicity, collect the variable definitions first, then write offsets.

    // Variable 0: spatial (non-record, char, dim=[spatial])
    // Variable 1: coordinates (record, float, dim=[frame, atom, spatial])
    // Variable 2: cell_spatial (non-record, char, dim=[cell_spatial]) — only if has_cell
    // Variable 3: cell_lengths (record, double, dim=[frame, cell_spatial]) — only if has_cell
    // Variable 4: cell_angles (record, double, dim=[frame, cell_angular]) — only if has_cell

    // We need to calculate the header end position for data offsets.
    // Track current position to compute where data starts.

    // Actually, let's compute the full header size first, then seek back to write offsets.
    // Or better: write variable defs with placeholder offsets, note positions, then fix up.

    // Let me use a simpler approach: compute header size analytically.
    // Current position after attrs will be the start of variable section.
    // Each variable def: name(padded) + 4(ndims) + 4*ndims(dimids) + 8(vatt tag+count) + 4(type) + 4(vsize) + 8(begin offset)

    // For now, write all variable defs, track begin-offset positions, then fix them up.
    const VarDef = struct { begin_file_pos: u64 };
    var var_defs: [5]VarDef = undefined;
    var vi: usize = 0;

    // var 0: spatial (non-record)
    try fileWriteName(writer, "spatial");
    try fileWriteU32(writer, 1); // 1 dim
    try fileWriteU32(writer, 1); // dimid=1 (spatial)
    try fileWriteU32(writer, 0); // no attrs tag
    try fileWriteU32(writer, 0); // no attrs count
    try fileWriteU32(writer, NC_CHAR);
    try fileWriteU32(writer, @intCast(spatial_size)); // vsize
    var_defs[vi] = .{ .begin_file_pos = writer.logicalPos() };
    try fileWriteU64(writer, 0); // placeholder offset
    vi += 1;

    // var 1: coordinates (record)
    try fileWriteName(writer, "coordinates");
    try fileWriteU32(writer, 3); // 3 dims
    try fileWriteU32(writer, 0); // dimid=0 (frame, unlimited)
    try fileWriteU32(writer, 2); // dimid=2 (atom)
    try fileWriteU32(writer, 1); // dimid=1 (spatial)
    // 1 attribute: units = "angstrom"
    try fileWriteU32(writer, NC_ATTRIBUTE);
    try fileWriteU32(writer, 1);
    try fileWriteName(writer, "units");
    try fileWriteU32(writer, NC_CHAR);
    try fileWriteU32(writer, 8);
    try writer.interface.writeAll("angstrom"); // 8 bytes, already aligned
    try fileWriteU32(writer, NC_FLOAT);
    try fileWriteU32(writer, @intCast(coord_padded)); // vsize
    var_defs[vi] = .{ .begin_file_pos = writer.logicalPos() };
    try fileWriteU64(writer, 0); // placeholder
    vi += 1;

    if (has_cell) {
        // var 2: cell_spatial (non-record)
        try fileWriteName(writer, "cell_spatial");
        try fileWriteU32(writer, 1); // 1 dim
        try fileWriteU32(writer, 3); // dimid=3 (cell_spatial)
        try fileWriteU32(writer, 0); // no attrs
        try fileWriteU32(writer, 0);
        try fileWriteU32(writer, NC_CHAR);
        try fileWriteU32(writer, 4); // vsize: 3 chars padded to 4
        var_defs[vi] = .{ .begin_file_pos = writer.logicalPos() };
        try fileWriteU64(writer, 0);
        vi += 1;

        // var 3: cell_lengths (record)
        try fileWriteName(writer, "cell_lengths");
        try fileWriteU32(writer, 2); // 2 dims
        try fileWriteU32(writer, 0); // dimid=0 (frame)
        try fileWriteU32(writer, 3); // dimid=3 (cell_spatial)
        try fileWriteU32(writer, NC_ATTRIBUTE);
        try fileWriteU32(writer, 1);
        try fileWriteName(writer, "units");
        try fileWriteU32(writer, NC_CHAR);
        try fileWriteU32(writer, 8);
        try writer.interface.writeAll("angstrom");
        try fileWriteU32(writer, NC_DOUBLE);
        try fileWriteU32(writer, @intCast(cell_lengths_size));
        var_defs[vi] = .{ .begin_file_pos = writer.logicalPos() };
        try fileWriteU64(writer, 0);
        vi += 1;

        // var 4: cell_angles (record)
        try fileWriteName(writer, "cell_angles");
        try fileWriteU32(writer, 2); // 2 dims
        try fileWriteU32(writer, 0); // dimid=0 (frame)
        try fileWriteU32(writer, 5); // dimid=5 (cell_angular)
        try fileWriteU32(writer, NC_ATTRIBUTE);
        try fileWriteU32(writer, 1);
        try fileWriteName(writer, "units");
        try fileWriteU32(writer, NC_CHAR);
        try fileWriteU32(writer, 6);
        try writer.interface.writeAll("degree\x00\x00"); // 6 chars padded to 8
        try fileWriteU32(writer, NC_DOUBLE);
        try fileWriteU32(writer, @intCast(cell_angles_size));
        var_defs[vi] = .{ .begin_file_pos = writer.logicalPos() };
        try fileWriteU64(writer, 0);
        vi += 1;
    }

    // Now we know where data starts
    var data_offset = writer.logicalPos();

    // Non-record variables come first (spatial, cell_spatial)
    // var 0: spatial data at data_offset
    const spatial_offset = data_offset;
    data_offset += spatial_size;

    var cell_spatial_offset: u64 = 0;
    if (has_cell) {
        cell_spatial_offset = data_offset;
        data_offset += 4; // "abc\0"
    }

    // Record data starts here
    const record_start = data_offset;
    const coords_begin = record_start; // first record var

    var cell_lengths_begin: u64 = 0;
    var cell_angles_begin: u64 = 0;
    if (has_cell) {
        cell_lengths_begin = record_start + coord_padded;
        cell_angles_begin = cell_lengths_begin + cell_lengths_size;
    }

    // Write non-record data
    writer.seekTo(spatial_offset) catch return NcWriteError.WriteError;
    try writer.interface.writeAll("xyz\x00"); // spatial labels padded to 4

    if (has_cell) {
        writer.seekTo(cell_spatial_offset) catch return NcWriteError.WriteError;
        try writer.interface.writeAll("abc\x00"); // cell_spatial labels
    }

    // Fix up variable begin offsets
    vi = 0;

    // var 0: spatial
    writer.seekTo(var_defs[vi].begin_file_pos) catch return NcWriteError.WriteError;
    try fileWriteU64(writer, spatial_offset);
    vi += 1;

    // var 1: coordinates
    writer.seekTo(var_defs[vi].begin_file_pos) catch return NcWriteError.WriteError;
    try fileWriteU64(writer, coords_begin);
    vi += 1;

    if (has_cell) {
        // var 2: cell_spatial
        writer.seekTo(var_defs[vi].begin_file_pos) catch return NcWriteError.WriteError;
        try fileWriteU64(writer, cell_spatial_offset);
        vi += 1;

        // var 3: cell_lengths
        writer.seekTo(var_defs[vi].begin_file_pos) catch return NcWriteError.WriteError;
        try fileWriteU64(writer, cell_lengths_begin);
        vi += 1;

        // var 4: cell_angles
        writer.seekTo(var_defs[vi].begin_file_pos) catch return NcWriteError.WriteError;
        try fileWriteU64(writer, cell_angles_begin);
        vi += 1;
    }

    // Seek to record data start for frame writing
    writer.seekTo(record_start) catch return NcWriteError.WriteError;

    return HeaderInfo{
        .numrecs_offset = numrecs_offset,
        .data_offset = record_start,
        .rec_size = rec_size,
    };
}

fn writeBEf32(buf: *[4]u8, val: f32) void {
    std.mem.writeInt(u32, buf, @bitCast(val), .big);
}

fn writeBEf64(buf: *[8]u8, val: f64) void {
    std.mem.writeInt(u64, buf, @bitCast(val), .big);
}

fn fileWriteU32(writer: *std.Io.File.Writer, val: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, val, .big);
    writer.interface.writeAll(&buf) catch return NcWriteError.WriteError;
}

fn fileWriteU64(writer: *std.Io.File.Writer, val: u64) !void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, val, .big);
    writer.interface.writeAll(&buf) catch return NcWriteError.WriteError;
}

fn fileWriteName(writer: *std.Io.File.Writer, name: []const u8) !void {
    try fileWriteU32(writer, @intCast(name.len));
    writer.interface.writeAll(name) catch return NcWriteError.WriteError;
    const pad = (4 - (name.len % 4)) % 4;
    if (pad > 0) {
        const zeros = [_]u8{ 0, 0, 0 };
        writer.interface.writeAll(zeros[0..pad]) catch return NcWriteError.WriteError;
    }
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

test "open and read cpptraj nc (3 frames, 80 atoms, with cell)" {
    const allocator = std.testing.allocator;
    var reader = try NcReader.open(testIo(), allocator, "test_data/cpptraj_traj.nc");
    defer reader.deinit();

    try std.testing.expectEqual(@as(u32, 80), reader.nAtoms());
    try std.testing.expectEqual(@as(u32, 3), reader.nFrames());

    var frame_count: usize = 0;
    while (try reader.next()) |frame| {
        try std.testing.expectEqual(@as(usize, 80), frame.nAtoms());

        if (frame_count == 0) {
            // Frame 0, atom 0: verified against netCDF4/Python
            try std.testing.expectApproxEqAbs(@as(f32, 23.326), frame.x[0], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 21.548), frame.y[0], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 20.000), frame.z[0], 0.01);

            // Frame 0, last atom (79)
            try std.testing.expectApproxEqAbs(@as(f32, 38.391), frame.x[79], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 29.270), frame.y[79], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 20.000), frame.z[79], 0.01);

            // Orthogonal box: cell_lengths=[40, 40, 40], angles=90
            try std.testing.expect(frame.box_vectors != null);
            const box = frame.box_vectors.?;
            try std.testing.expectApproxEqAbs(@as(f32, 40.0), box[0][0], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 40.0), box[1][1], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 40.0), box[2][2], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), box[0][1], 0.001);
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), box[1][0], 0.001);

            // No time variable in this file -> defaults to 0
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), frame.time, 0.001);
        }
        if (frame_count == 1) {
            // All 3 frames have identical coords (replicated from inpcrd)
            try std.testing.expectApproxEqAbs(@as(f32, 23.326), frame.x[0], 0.01);
        }
        frame_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), frame_count);
}

test "open and read mdcrd nc (3 frames, 22 atoms, no cell)" {
    const allocator = std.testing.allocator;
    var reader = try NcReader.open(testIo(), allocator, "test_data/mdcrd.nc");
    defer reader.deinit();

    try std.testing.expectEqual(@as(u32, 22), reader.nAtoms());
    try std.testing.expectEqual(@as(u32, 3), reader.nFrames());

    var frame_count: usize = 0;
    while (try reader.next()) |frame| {
        try std.testing.expectEqual(@as(usize, 22), frame.nAtoms());

        // No cell_lengths/cell_angles variables in this file
        try std.testing.expect(frame.box_vectors == null);

        if (frame_count == 0) {
            // Frame 0, atom 0: verified against netCDF4/Python
            try std.testing.expectApproxEqAbs(@as(f32, 2.0), frame.x[0], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 1.0), frame.y[0], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), frame.z[0], 0.01);

            // Frame 0, last atom (21)
            try std.testing.expectApproxEqAbs(@as(f32, 6.360), frame.x[21], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 8.648), frame.y[21], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, -0.890), frame.z[21], 0.01);
        }
        frame_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), frame_count);
}

test "invalid nc file rejected" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(NcError.FileNotFound, NcReader.open(testIo(), allocator, "test_data/nonexistent.nc"));
    // A PDB file should fail magic check
    try std.testing.expectError(NcError.InvalidMagic, NcReader.open(testIo(), allocator, "test_data/1l2y.pdb"));
}

test "cellToVectors orthogonal box" {
    const box = NcReader.cellToVectors(10.0, 20.0, 30.0, 90.0, 90.0, 90.0).?;
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), box[0][0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), box[0][1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), box[1][0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), box[1][1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), box[2][0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), box[2][1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), box[2][2], 0.001);
}

test "cellToVectors triclinic box" {
    // Rhombohedral: a=b=c=10, alpha=beta=gamma=60 degrees
    const box = NcReader.cellToVectors(10.0, 10.0, 10.0, 60.0, 60.0, 60.0).?;
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), box[0][0], 0.001); // a along x
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), box[1][0], 0.01); // bx = b*cos(60)
    try std.testing.expectApproxEqAbs(@as(f32, 8.660), box[1][1], 0.01); // by = b*sin(60)
    // cz must be positive for a valid cell
    try std.testing.expect(box[2][2] > 0.0);
}

test "cellToVectors degenerate gamma returns null" {
    try std.testing.expect(NcReader.cellToVectors(10.0, 10.0, 10.0, 90.0, 90.0, 0.0) == null);
    try std.testing.expect(NcReader.cellToVectors(10.0, 10.0, 10.0, 90.0, 90.0, 180.0) == null);
}

test "next returns null after all frames read" {
    const allocator = std.testing.allocator;
    var reader = try NcReader.open(testIo(), allocator, "test_data/cpptraj_traj.nc");
    defer reader.deinit();

    while (try reader.next()) |_| {}
    const result = try reader.next();
    try std.testing.expect(result == null);
}

test "NcWriter round-trip without cell" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/_test_nc_roundtrip.nc";

    // Write 2 frames of 3 atoms
    {
        var writer = try NcWriter.open(testIo(), allocator, tmp_path, 3, false);
        defer writer.deinit();

        var frame = try types.Frame.init(allocator, 3);
        defer frame.deinit();

        frame.x[0] = 1.0;
        frame.y[0] = 2.0;
        frame.z[0] = 3.0;
        frame.x[1] = 4.0;
        frame.y[1] = 5.0;
        frame.z[1] = 6.0;
        frame.x[2] = 7.5;
        frame.y[2] = 8.5;
        frame.z[2] = 9.5;
        try writer.writeFrame(frame);

        frame.x[0] = 10.0;
        frame.y[0] = 11.0;
        frame.z[0] = 12.0;
        try writer.writeFrame(frame);
        try writer.close();
    }
    defer std.Io.Dir.cwd().deleteFile(testIo(), tmp_path) catch {};

    // Read back and verify
    var reader = try NcReader.open(testIo(), allocator, tmp_path);
    defer reader.deinit();

    try std.testing.expectEqual(@as(u32, 3), reader.nAtoms());
    try std.testing.expectEqual(@as(u32, 2), reader.nFrames());

    const f0 = (try reader.next()).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f0.x[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), f0.y[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), f0.z[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), f0.x[2], 0.001);
    try std.testing.expect(f0.box_vectors == null);

    const f1 = (try reader.next()).?;
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), f1.x[0], 0.001);

    try std.testing.expect((try reader.next()) == null);
}

test "NcWriter round-trip with cell" {
    const allocator = std.testing.allocator;
    const tmp_path = "test_data/_test_nc_roundtrip_cell.nc";

    {
        var writer = try NcWriter.open(testIo(), allocator, tmp_path, 2, true);
        defer writer.deinit();

        var frame = try types.Frame.init(allocator, 2);
        defer frame.deinit();

        frame.x[0] = 1.0;
        frame.y[0] = 2.0;
        frame.z[0] = 3.0;
        frame.x[1] = 4.0;
        frame.y[1] = 5.0;
        frame.z[1] = 6.0;
        frame.box_vectors = .{
            .{ 50.0, 0.0, 0.0 },
            .{ 0.0, 50.0, 0.0 },
            .{ 0.0, 0.0, 50.0 },
        };
        try writer.writeFrame(frame);
        try writer.close();
    }
    defer std.Io.Dir.cwd().deleteFile(testIo(), tmp_path) catch {};

    var reader = try NcReader.open(testIo(), allocator, tmp_path);
    defer reader.deinit();

    try std.testing.expectEqual(@as(u32, 2), reader.nAtoms());
    try std.testing.expectEqual(@as(u32, 1), reader.nFrames());

    const f0 = (try reader.next()).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f0.x[0], 0.001);
    try std.testing.expect(f0.box_vectors != null);
    const box = f0.box_vectors.?;
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), box[0][0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), box[1][1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), box[2][2], 0.01);
}
