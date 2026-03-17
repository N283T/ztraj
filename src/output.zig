//! Output formatting for ztraj analysis results.
//!
//! Provides functions to write analysis results to any std.io writer in
//! CSV, TSV, and JSON formats.

const std = @import("std");

/// Supported output formats.
pub const Format = enum {
    json,
    csv,
    tsv,
};

// ============================================================================
// Delimited (CSV / TSV)
// ============================================================================

/// Write a table of f64 values as CSV or TSV.
///
/// Parameters:
///   writer    — any writer (file, buffer, etc.)
///   headers   — column header strings
///   rows      — slice of rows; each row is a slice of f64 values
///   delimiter — ',' for CSV, '\t' for TSV
///
/// Each value is written with 6 decimal places. An empty rows slice produces
/// only the header line.
pub fn writeDelimited(
    writer: anytype,
    headers: []const []const u8,
    rows: []const []const f64,
    delimiter: u8,
) !void {
    // Write header row.
    for (headers, 0..) |h, i| {
        if (i > 0) try writer.writeByte(delimiter);
        try writer.writeAll(h);
    }
    try writer.writeByte('\n');

    // Write data rows.
    for (rows) |row| {
        for (row, 0..) |v, j| {
            if (j > 0) try writer.writeByte(delimiter);
            try writer.print("{d:.6}", .{v});
        }
        try writer.writeByte('\n');
    }
}

// ============================================================================
// JSON helpers
// ============================================================================

/// Write a single named JSON array of f64 values.
///
/// Output: `"key": [v0, v1, ...]` (no surrounding braces).
/// Intended for embedding inside a larger JSON object.
pub fn writeJsonArray(
    writer: anytype,
    key: []const u8,
    values: []const f64,
) !void {
    try writer.writeByte('"');
    try writeJsonEscaped(writer, key);
    try writer.writeAll("\": [");
    for (values, 0..) |v, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("{d:.6}", .{v});
    }
    try writer.writeByte(']');
}

/// Write a JSON object mapping string keys to f64 arrays.
///
/// Output:
///   {
///     "key0": [v, ...],
///     "key1": [v, ...]
///   }
///
/// `keys` and `values` must have the same length.
pub fn writeJsonObject(
    writer: anytype,
    keys: []const []const u8,
    values: []const []const f64,
) !void {
    std.debug.assert(keys.len == values.len);

    try writer.writeAll("{\n");
    for (keys, 0..) |key, i| {
        try writer.writeAll("  ");
        try writeJsonArray(writer, key, values[i]);
        if (i + 1 < keys.len) {
            try writer.writeByte(',');
        }
        try writer.writeByte('\n');
    }
    try writer.writeByte('}');
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Write a string with JSON-required escapes (", \, and control characters).
fn writeJsonEscaped(writer: anytype, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(c),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "writeDelimited CSV with header and rows" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);

    const headers = [_][]const u8{ "frame", "rmsd", "rg" };
    const row0 = [_]f64{ 0.0, 1.23456, 7.89012 };
    const row1 = [_]f64{ 1.0, 2.34567, 8.90123 };
    const rows = [_][]const f64{
        &row0,
        &row1,
    };

    try writeDelimited(buf.writer(allocator), &headers, &rows, ',');

    const output = buf.items;
    try std.testing.expect(std.mem.startsWith(u8, output, "frame,rmsd,rg\n"));
    try std.testing.expect(std.mem.indexOf(u8, output, "0.000000,1.234560,7.890120\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "1.000000,2.345670,8.901230\n") != null);
}

test "writeDelimited TSV uses tab delimiter" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);

    const headers = [_][]const u8{ "a", "b" };
    const row = [_]f64{ 1.0, 2.0 };
    const rows = [_][]const f64{&row};

    try writeDelimited(buf.writer(allocator), &headers, &rows, '\t');

    const output = buf.items;
    try std.testing.expect(std.mem.startsWith(u8, output, "a\tb\n"));
    try std.testing.expect(std.mem.indexOf(u8, output, "\t") != null);
    // Must not contain commas.
    try std.testing.expect(std.mem.indexOf(u8, output, ",") == null);
}

test "writeDelimited empty rows only writes header" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);

    const headers = [_][]const u8{ "x", "y" };
    const rows = [_][]const f64{};

    try writeDelimited(buf.writer(allocator), &headers, &rows, ',');

    try std.testing.expectEqualStrings("x,y\n", buf.items);
}

test "writeJsonArray produces valid JSON array" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);

    const values = [_]f64{ 1.0, 2.5, 3.14159 };
    try writeJsonArray(buf.writer(allocator), "rmsd", &values);

    const output = buf.items;
    try std.testing.expect(std.mem.startsWith(u8, output, "\"rmsd\": ["));
    try std.testing.expect(std.mem.endsWith(u8, output, "]"));
    try std.testing.expect(std.mem.indexOf(u8, output, "1.000000") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "3.141590") != null);
}

test "writeJsonObject produces valid JSON object" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);

    const keys = [_][]const u8{ "rmsd", "rg" };
    const rmsd_vals = [_]f64{ 0.5, 1.0 };
    const rg_vals = [_]f64{ 10.0, 11.0 };
    const values = [_][]const f64{ &rmsd_vals, &rg_vals };

    try writeJsonObject(buf.writer(allocator), &keys, &values);

    const output = buf.items;
    try std.testing.expect(std.mem.startsWith(u8, output, "{\n"));
    try std.testing.expect(std.mem.endsWith(u8, output, "}"));
    try std.testing.expect(std.mem.indexOf(u8, output, "\"rmsd\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "\"rg\"") != null);
    // Second key should NOT have a trailing comma.
    const rg_pos = std.mem.indexOf(u8, output, "\"rg\"").?;
    const after_rg = output[rg_pos..];
    try std.testing.expect(std.mem.indexOf(u8, after_rg, "],") == null);
}

test "writeJsonArray with empty values" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);

    const values = [_]f64{};
    try writeJsonArray(buf.writer(allocator), "empty", &values);

    try std.testing.expectEqualStrings("\"empty\": []", buf.items);
}

test "writeJsonObject with single key" {
    const allocator = std.testing.allocator;

    var buf = std.ArrayList(u8){};
    defer buf.deinit(allocator);

    const keys = [_][]const u8{"x"};
    const vals = [_]f64{42.0};
    const values = [_][]const f64{&vals};

    try writeJsonObject(buf.writer(allocator), &keys, &values);

    const output = buf.items;
    // Single key must not have a trailing comma before the closing newline.
    try std.testing.expect(std.mem.indexOf(u8, output, "],") == null);
    try std.testing.expect(std.mem.indexOf(u8, output, "42.000000") != null);
}
