const std = @import("std");
const builtin = @import("builtin");

pub const is_windows = builtin.os.tag == .windows;

/// A read-only file region backed by mmap (POSIX) or heap allocation (Windows).
///
/// Ownership: single-owner, non-copyable by convention. Exactly one `deinit`
/// call must occur per `mmapFile` call.
/// Typical usage: `const mapped = try mmapFile(alloc, path); defer mapped.deinit();`
pub const MappedFile = struct {
    data: if (is_windows) []const u8 else []align(std.heap.page_size_min) const u8,
    allocator: if (is_windows) std.mem.Allocator else void,

    pub fn deinit(self: MappedFile) void {
        if (self.data.len == 0) return;
        if (is_windows) {
            self.allocator.free(self.data);
        } else {
            std.posix.munmap(@alignCast(self.data));
        }
    }
};

pub fn mmapFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !MappedFile {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const size: usize = std.math.cast(usize, stat.size) orelse return error.FileTooBig;
    if (size == 0) return .{ .data = &.{}, .allocator = if (is_windows) allocator else {} };

    if (is_windows) {
        // Read from the already-open handle to avoid a TOCTOU re-open.
        const data = try allocator.alloc(u8, size);
        errdefer allocator.free(data);
        const n = try file.readPositionalAll(io, data, 0);
        return .{ .data = data[0..n], .allocator = allocator };
    } else {
        const mapped = try std.posix.mmap(
            null,
            size,
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        return .{ .data = mapped, .allocator = {} };
    }
}

fn testIo() std.Io {
    const t = struct {
        var threaded: std.Io.Threaded = .init_single_threaded;
    };
    return t.threaded.io();
}

test "mmapFile reads valid PDB" {
    const mapped = mmapFile(testIo(), std.testing.allocator, "test_data/1l2y.pdb") catch |err| {
        if (err == error.FileNotFound) return; // Skip when run from non-project-root cwd
        return err;
    };
    defer mapped.deinit();
    try std.testing.expect(mapped.data.len > 0);
    try std.testing.expect(std.mem.startsWith(u8, mapped.data, "REMARK") or
        std.mem.startsWith(u8, mapped.data, "HEADER") or
        std.mem.startsWith(u8, mapped.data, "ATOM"));
}
