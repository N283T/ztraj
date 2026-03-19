const std = @import("std");
const flate = std.compress.flate;
const Io = std.Io;
const Allocator = std.mem.Allocator;

/// Gzip magic bytes
const GZIP_MAGIC: [2]u8 = .{ 0x1f, 0x8b };

/// Check if data starts with gzip magic bytes
pub fn isGzipCompressed(data: []const u8) bool {
    return data.len >= 2 and data[0] == GZIP_MAGIC[0] and data[1] == GZIP_MAGIC[1];
}

/// Check if file path ends with .gz extension (case-insensitive)
pub fn hasGzipExtension(path: []const u8) bool {
    if (path.len < 3) return false;
    const ext = path[path.len - 3 ..];
    return std.ascii.eqlIgnoreCase(ext, ".gz");
}

/// Decompress gzip data with a specified size limit.
/// Returns the decompressed data allocated with the given allocator.
/// Caller owns the returned memory.
///
/// Errors:
/// - error.OutOfMemory: Allocation failed
/// - Decompression errors from flate.Decompress (e.g., corrupted data)
pub fn decompress(allocator: Allocator, compressed: []const u8, max_decompressed_size: usize) ![]u8 {
    // Create a fixed buffer stream from compressed data
    var fbs = std.io.fixedBufferStream(compressed);
    const generic_reader = fbs.reader();

    // Adapt to new I/O API
    var read_buf: [4096]u8 = undefined;
    var adapter = generic_reader.adaptToNewApi(&read_buf);
    const io_reader = &adapter.new_interface;

    // Create decompressor with gzip container
    var window_buf: [flate.max_window_len]u8 = undefined;
    var decompressor = flate.Decompress.init(io_reader, .gzip, &window_buf);

    // Read all decompressed data
    const decompressed = try decompressor.reader.allocRemaining(
        allocator,
        Io.Limit.limited(max_decompressed_size),
    );

    return decompressed;
}

/// Read a file and decompress if it's gzip compressed.
/// Returns the file contents (decompressed if necessary).
/// Caller owns the returned memory.
pub fn readFileMaybeCompressed(allocator: Allocator, path: []const u8, max_size: usize) ![]u8 {
    // Read raw file contents
    const raw_data = try std.fs.cwd().readFileAlloc(allocator, path, max_size);
    errdefer allocator.free(raw_data);

    // Check if gzip compressed
    if (isGzipCompressed(raw_data)) {
        // Decompress with the same size limit
        const decompressed = try decompress(allocator, raw_data, max_size);
        allocator.free(raw_data);
        return decompressed;
    }

    // Not compressed, return as-is
    return raw_data;
}

/// Get the base filename without .gz extension if present.
/// For example: "file.pdb.gz" -> "file.pdb", "file.cif" -> "file.cif"
pub fn stripGzExtension(path: []const u8) []const u8 {
    if (hasGzipExtension(path)) {
        return path[0 .. path.len - 3];
    }
    return path;
}

test "isGzipCompressed" {
    const testing = std.testing;

    // Gzip magic bytes
    try testing.expect(isGzipCompressed(&[_]u8{ 0x1f, 0x8b, 0x08, 0x00 }));

    // Not gzip
    try testing.expect(!isGzipCompressed(&[_]u8{ 0x00, 0x00, 0x00, 0x00 }));
    try testing.expect(!isGzipCompressed(&[_]u8{0x1f})); // Too short
    try testing.expect(!isGzipCompressed(&[_]u8{})); // Empty
}

test "hasGzipExtension" {
    const testing = std.testing;

    try testing.expect(hasGzipExtension("file.pdb.gz"));
    try testing.expect(hasGzipExtension("file.cif.gz"));
    try testing.expect(hasGzipExtension(".gz"));
    try testing.expect(hasGzipExtension("file.pdb.GZ")); // Case-insensitive
    try testing.expect(hasGzipExtension("file.cif.Gz")); // Mixed case

    try testing.expect(!hasGzipExtension("file.pdb"));
    try testing.expect(!hasGzipExtension("file.gz.bak"));
    try testing.expect(!hasGzipExtension(""));
    try testing.expect(!hasGzipExtension("gz")); // Too short
}

test "stripGzExtension" {
    const testing = std.testing;

    try testing.expectEqualStrings("file.pdb", stripGzExtension("file.pdb.gz"));
    try testing.expectEqualStrings("file.cif", stripGzExtension("file.cif.gz"));
    try testing.expectEqualStrings("file.pdb", stripGzExtension("file.pdb"));
}
