//! AMBER NetCDF trajectory reader.
//!
//! Reads AMBER convention NetCDF-3 (64-bit offset) trajectory files (.nc, .ncdf).
//! Coordinates are stored in angstroms — no unit conversion needed.
//!
//! ## NetCDF-3 64-bit Format
//!
//! The file uses big-endian byte order throughout:
//! - Magic: "CDF\x02" (64-bit offset variant)
//! - Header: dimensions, global attributes, variable definitions
//! - Data: fixed-size and record (unlimited) variables
//!
//! ## AMBER Convention Variables
//!
//! - coordinates (frame, atom, spatial) float32  — atom positions in angstroms
//! - time        (frame)               float32  — simulation time in ps (optional)
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

// NetCDF-3 tag constants (big-endian)
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

/// Size in bytes of each NetCDF type.
fn ncTypeSize(nc_type: u32) u32 {
    return switch (nc_type) {
        NC_BYTE, NC_CHAR => 1,
        NC_SHORT => 2,
        NC_INT, NC_FLOAT => 4,
        NC_DOUBLE => 8,
        else => 0,
    };
}

/// Variable descriptor parsed from the NetCDF header.
const VarInfo = struct {
    /// Data offset in the file (64-bit for CDF-2).
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
///   var reader = try NcReader.open(allocator, "trajectory.nc");
///   defer reader.deinit();
///
///   while (try reader.next()) |frame| {
///       _ = frame.x[0]; // coordinates in angstroms
///   }
///
/// The reader reuses a single Frame buffer. The returned pointer is valid
/// until the next call to next() or deinit().
pub const NcReader = struct {
    file: std.fs.File,
    allocator: std.mem.Allocator,
    frame: types.Frame,
    /// Per-frame coordinate data buffer (AOS: x0,y0,z0,x1,y1,z1,...)
    coord_buf: []u8,
    n_atoms: u32,
    n_frames: u32,
    frames_read: u32,
    /// Byte offset between consecutive records (recsize in NetCDF spec).
    rec_size: u64,
    /// Variable offsets within the file.
    coords_var: VarInfo,
    time_var: ?VarInfo,
    cell_lengths_var: ?VarInfo,
    cell_angles_var: ?VarInfo,

    const Self = @This();

    /// Open an AMBER NetCDF trajectory for reading.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => NcError.FileNotFound,
                else => NcError.ReadError,
            };
        };
        errdefer file.close();

        // ---- Magic and version ----
        var magic: [4]u8 = undefined;
        readExact(file, &magic) catch return NcError.ReadError;
        if (magic[0] != 'C' or magic[1] != 'D' or magic[2] != 'F')
            return NcError.InvalidMagic;
        if (magic[3] != 1 and magic[3] != 2)
            return NcError.BadVersion;

        const is_64bit = magic[3] == 2;

        // ---- Number of records (unlimited dimension length) ----
        const numrecs = fileReadU32(file) catch return NcError.ReadError;

        // ---- Parse dimensions ----
        const DimInfo = struct { name_hash: u64, size: u32, is_unlimited: bool };
        var dims = std.ArrayListUnmanaged(DimInfo){};
        defer dims.deinit(allocator);

        var unlimited_dim_id: ?usize = null;

        {
            const tag = fileReadU32(file) catch return NcError.ReadError;
            const n_dims = fileReadU32(file) catch return NcError.ReadError;

            if (tag == NC_DIMENSION and n_dims > 0) {
                for (0..n_dims) |di| {
                    const name = try readFileName(file, allocator);
                    defer allocator.free(name);
                    const dim_len = fileReadU32(file) catch return NcError.ReadError;

                    const is_unlim = dim_len == 0;
                    if (is_unlim) unlimited_dim_id = di;

                    try dims.append(allocator, .{
                        .name_hash = std.hash.Wyhash.hash(0, name),
                        .size = if (is_unlim) numrecs else dim_len,
                        .is_unlimited = is_unlim,
                    });
                }
            } else if (tag == 0 and n_dims == 0) {
                // ABSENT — no dimensions
            } else {
                return NcError.ReadError;
            }
        }

        // Look up key dimensions by name hash
        const hash_frame = std.hash.Wyhash.hash(0, "frame");
        const hash_spatial = std.hash.Wyhash.hash(0, "spatial");
        const hash_atom = std.hash.Wyhash.hash(0, "atom");

        var n_atoms: u32 = 0;
        var n_frames: u32 = 0;
        var spatial_size: u32 = 0;

        for (dims.items) |d| {
            if (d.name_hash == hash_frame) n_frames = d.size;
            if (d.name_hash == hash_spatial) spatial_size = d.size;
            if (d.name_hash == hash_atom) n_atoms = d.size;
        }

        if (n_atoms == 0) return NcError.BadDimension;
        if (spatial_size != 3) return NcError.BadDimension;

        // ---- Parse global attributes ----
        var is_amber = false;
        {
            const tag = fileReadU32(file) catch return NcError.ReadError;
            const n_attrs = fileReadU32(file) catch return NcError.ReadError;

            if (tag == NC_ATTRIBUTE and n_attrs > 0) {
                for (0..n_attrs) |_| {
                    const name = try readFileName(file, allocator);
                    defer allocator.free(name);
                    const attr_type = fileReadU32(file) catch return NcError.ReadError;
                    const attr_nelems = fileReadU32(file) catch return NcError.ReadError;

                    const attr_bytes = attr_nelems * ncTypeSize(attr_type);
                    const padded = (attr_bytes + 3) & ~@as(u32, 3); // 4-byte aligned

                    if (std.mem.eql(u8, name, "Conventions")) {
                        if (attr_type == NC_CHAR and attr_nelems <= 64) {
                            var buf: [64]u8 = undefined;
                            readExact(file, buf[0..padded]) catch return NcError.ReadError;
                            const val = std.mem.trim(u8, buf[0..attr_nelems], " \x00");
                            if (std.mem.eql(u8, val, "AMBER")) is_amber = true;
                            continue;
                        }
                    }
                    // Skip attribute data
                    file.seekBy(@intCast(padded)) catch return NcError.ReadError;
                }
            } else if (tag == 0 and n_attrs == 0) {
                // ABSENT
            }
        }

        if (!is_amber) return NcError.NotAmberConvention;

        // ---- Parse variables ----
        const hash_coords = std.hash.Wyhash.hash(0, "coordinates");
        const hash_time = std.hash.Wyhash.hash(0, "time");
        const hash_cell_lengths = std.hash.Wyhash.hash(0, "cell_lengths");
        const hash_cell_angles = std.hash.Wyhash.hash(0, "cell_angles");

        var coords_var: ?VarInfo = null;
        var time_var: ?VarInfo = null;
        var cell_lengths_var: ?VarInfo = null;
        var cell_angles_var: ?VarInfo = null;
        var rec_size: u64 = 0;

        {
            const tag = fileReadU32(file) catch return NcError.ReadError;
            const n_vars = fileReadU32(file) catch return NcError.ReadError;

            if (tag == NC_VARIABLE and n_vars > 0) {
                for (0..n_vars) |_| {
                    const name = try readFileName(file, allocator);
                    defer allocator.free(name);
                    const name_hash = std.hash.Wyhash.hash(0, name);

                    // Number of dimensions
                    const n_var_dims = fileReadU32(file) catch return NcError.ReadError;
                    var is_record_var = false;
                    var var_elem_count: u64 = 1;

                    for (0..n_var_dims) |_| {
                        const dim_id = fileReadU32(file) catch return NcError.ReadError;
                        if (dim_id < dims.items.len) {
                            const d = dims.items[dim_id];
                            if (d.is_unlimited) {
                                is_record_var = true;
                            } else {
                                var_elem_count *= d.size;
                            }
                        }
                    }

                    // Skip variable attributes
                    const va_tag = fileReadU32(file) catch return NcError.ReadError;
                    const va_count = fileReadU32(file) catch return NcError.ReadError;
                    if (va_tag == NC_ATTRIBUTE and va_count > 0) {
                        for (0..va_count) |_| {
                            try skipFileName(file);
                            const at = fileReadU32(file) catch return NcError.ReadError;
                            const an = fileReadU32(file) catch return NcError.ReadError;
                            const ab = an * ncTypeSize(at);
                            const ap = (ab + 3) & ~@as(u32, 3);
                            file.seekBy(@intCast(ap)) catch return NcError.ReadError;
                        }
                    }

                    const nc_type = fileReadU32(file) catch return NcError.ReadError;
                    // vsize: per-record size for record vars, total size for fixed vars
                    const vsize = fileReadU32(file) catch return NcError.ReadError;

                    // begin: data offset (4 bytes for CDF-1, 8 bytes for CDF-2)
                    const offset: u64 = if (is_64bit)
                        fileReadU64(file) catch return NcError.ReadError
                    else
                        fileReadU32(file) catch return NcError.ReadError;

                    const info = VarInfo{
                        .offset = offset,
                        .nc_type = nc_type,
                        .var_size = vsize,
                    };

                    if (name_hash == hash_coords) coords_var = info;
                    if (name_hash == hash_time) time_var = info;
                    if (name_hash == hash_cell_lengths) cell_lengths_var = info;
                    if (name_hash == hash_cell_angles) cell_angles_var = info;

                    if (is_record_var) {
                        rec_size += vsize;
                    }
                }
            }
        }

        if (coords_var == null) return NcError.MissingVariable;

        // Allocate frame and coordinate buffer
        const coord_bytes = n_atoms * 3 * 4; // float32 AOS
        const coord_buf = allocator.alloc(u8, coord_bytes) catch return NcError.OutOfMemory;
        errdefer allocator.free(coord_buf);

        var frame = types.Frame.init(allocator, n_atoms) catch return NcError.OutOfMemory;
        errdefer frame.deinit();

        return Self{
            .file = file,
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
        self.file.close();
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
    /// Returns null at end of file.
    /// The returned pointer is valid until the next call to next() or deinit().
    pub fn next(self: *Self) !?*const types.Frame {
        if (self.frames_read >= self.n_frames) return null;

        const fi: u64 = self.frames_read;

        // ---- Read coordinates ----
        {
            const offset = self.coords_var.offset + fi * self.rec_size;
            self.file.seekTo(offset) catch return NcError.ReadError;
            readExact(self.file,self.coord_buf) catch return NcError.ReadError;

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
            self.file.seekTo(offset) catch return NcError.ReadError;
            var buf: [4]u8 = undefined;
            readExact(self.file,&buf) catch return NcError.ReadError;
            self.frame.time = readBEf32(&buf);
        } else {
            self.frame.time = 0.0;
        }

        // ---- Read box vectors (optional) ----
        if (self.cell_lengths_var) |cl| {
            if (self.cell_angles_var) |ca| {
                const cl_offset = cl.offset + fi * self.rec_size;
                self.file.seekTo(cl_offset) catch return NcError.ReadError;
                var lbuf: [24]u8 = undefined; // 3 * f64
                readExact(self.file,&lbuf) catch return NcError.ReadError;

                const ca_offset = ca.offset + fi * self.rec_size;
                self.file.seekTo(ca_offset) catch return NcError.ReadError;
                var abuf: [24]u8 = undefined;
                readExact(self.file,&abuf) catch return NcError.ReadError;

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
    /// Follows the standard convention: a along x, b in xy-plane, c fills z.
    fn cellToVectors(a: f32, b: f32, c: f32, alpha_deg: f64, beta_deg: f64, gamma_deg: f64) [3][3]f32 {
        const deg2rad: f64 = std.math.pi / 180.0;
        const alpha = alpha_deg * deg2rad;
        const beta = beta_deg * deg2rad;
        const gamma = gamma_deg * deg2rad;

        const cos_alpha = @cos(alpha);
        const cos_beta = @cos(beta);
        const cos_gamma = @cos(gamma);
        const sin_gamma = @sin(gamma);

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

/// Read exactly buf.len bytes from a file.
fn readExact(file: std.fs.File, buf: []u8) !void {
    const n = file.readAll(buf) catch return NcError.ReadError;
    if (n != buf.len) return NcError.ReadError;
}

fn fileReadU32(file: std.fs.File) !u32 {
    var buf: [4]u8 = undefined;
    try readExact(file, &buf);
    return std.mem.readInt(u32, &buf, .big);
}

fn fileReadU64(file: std.fs.File) !u64 {
    var buf: [8]u8 = undefined;
    try readExact(file, &buf);
    return std.mem.readInt(u64, &buf, .big);
}

/// Read a NetCDF name: u32 length, then chars padded to 4-byte boundary.
fn readFileName(file: std.fs.File, allocator: std.mem.Allocator) ![]u8 {
    const name_len = try fileReadU32(file);
    const padded_len = (name_len + 3) & ~@as(u32, 3);
    const buf = allocator.alloc(u8, padded_len) catch return NcError.OutOfMemory;
    errdefer allocator.free(buf);
    try readExact(file, buf);
    const result = allocator.alloc(u8, name_len) catch return NcError.OutOfMemory;
    @memcpy(result, buf[0..name_len]);
    allocator.free(buf);
    return result;
}

/// Skip a NetCDF name without allocating.
fn skipFileName(file: std.fs.File) !void {
    const name_len = try fileReadU32(file);
    const padded_len = (name_len + 3) & ~@as(u32, 3);
    file.seekBy(@intCast(padded_len)) catch return NcError.ReadError;
}

// ============================================================================
// Tests
// ============================================================================

test "open and read cpptraj nc (3 frames, 84 atoms, with cell)" {
    const allocator = std.testing.allocator;
    var reader = try NcReader.open(allocator, "test_data/cpptraj_traj.nc");
    defer reader.deinit();

    try std.testing.expectEqual(@as(u32, 84), reader.nAtoms());
    try std.testing.expectEqual(@as(u32, 3), reader.nFrames());

    var frame_count: usize = 0;
    while (try reader.next()) |frame| {
        try std.testing.expectEqual(@as(usize, 84), frame.nAtoms());

        if (frame_count == 0) {
            // First atom, frame 0: coordinates from Python verification
            try std.testing.expectApproxEqAbs(@as(f32, 19.073), frame.x[0], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 31.774), frame.y[0], 0.01);
            try std.testing.expectApproxEqAbs(@as(f32, 59.940), frame.z[0], 0.01);

            // Should have box vectors (cell_lengths/cell_angles present)
            try std.testing.expect(frame.box_vectors != null);
        }
        frame_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), frame_count);
}

test "open and read mdcrd nc (101 frames, 223 atoms, no cell)" {
    const allocator = std.testing.allocator;
    var reader = try NcReader.open(allocator, "test_data/mdcrd.nc");
    defer reader.deinit();

    try std.testing.expectEqual(@as(u32, 223), reader.nAtoms());
    try std.testing.expectEqual(@as(u32, 101), reader.nFrames());

    var frame_count: usize = 0;
    while (try reader.next()) |frame| {
        try std.testing.expectEqual(@as(usize, 223), frame.nAtoms());

        // No cell info in this file
        try std.testing.expect(frame.box_vectors == null);

        // Coordinates should not all be zero
        if (frame_count == 0) {
            var has_nonzero = false;
            for (frame.x) |v| {
                if (v != 0.0) { has_nonzero = true; break; }
            }
            try std.testing.expect(has_nonzero);
        }
        frame_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 101), frame_count);
}

test "invalid nc file rejected" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(NcError.FileNotFound, NcReader.open(allocator, "test_data/nonexistent.nc"));
    // A PDB file should fail magic check
    try std.testing.expectError(NcError.InvalidMagic, NcReader.open(allocator, "test_data/1l2y.pdb"));
}

test "next returns null after all frames read" {
    const allocator = std.testing.allocator;
    var reader = try NcReader.open(allocator, "test_data/cpptraj_traj.nc");
    defer reader.deinit();

    // Exhaust all frames
    while (try reader.next()) |_| {}

    // Should return null now
    const result = try reader.next();
    try std.testing.expect(result == null);
}
