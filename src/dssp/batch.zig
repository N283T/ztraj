const std = @import("std");
const dssp = @import("dssp.zig");
const json_writer = @import("json_writer.zig");
const json_parser = @import("json_parser.zig");
const mmcif_parser = @import("mmcif_parser.zig");
const pdb_parser = @import("pdb_parser.zig");
const gzip = @import("gzip.zig");
const thread_pool_mod = @import("thread_pool.zig");

const Allocator = std.mem.Allocator;
const DsspResult = dssp.DsspResult;
const DsspConfig = dssp.DsspConfig;
const OutputFormat = json_writer.OutputFormat;

/// Parallelism strategy for batch processing
pub const Parallelism = enum {
    auto, // 3-phase: scan → split → process (default)
    file, // File-level: N files in parallel, 1 thread per file
    residue, // Residue-level: 1 file at a time, N threads for accessibility
};

/// Supported input formats for batch processing
pub const InputFormat = enum {
    auto, // Auto-detect from extension
    mmcif,
    pdb,
    json,
};

/// Configuration for batch processing
pub const BatchConfig = struct {
    n_threads: usize = 0, // 0 = auto-detect
    parallelism: Parallelism = .auto,
    /// Files larger than this are processed with residue-level parallelism.
    /// Default 5 MB ≈ ~8000 residues for mmCIF format.
    heavy_file_threshold: u64 = 5 * 1024 * 1024,
    input_format: InputFormat = .auto,
    output_format: OutputFormat = .json,
    pp_stretch: u32 = 3,
    model_num: ?u32 = 1,
    calculate_accessibility: bool = true,
    show_timing: bool = false,
    timing_json_path: ?[]const u8 = null, // Write timing data to JSON file
    quiet: bool = false,
    recursive: bool = false, // Recursively scan subdirectories
};

/// Result for a single file
pub const FileResult = struct {
    filename: []const u8,
    n_residues: usize,
    dssp_time_ns: u64,
    status: Status,
    error_msg: ?[]const u8 = null,

    pub const Status = enum {
        ok,
        err,
    };
};

/// Aggregate result for batch processing
pub const BatchResult = struct {
    total_files: usize,
    successful: usize,
    failed: usize,
    total_dssp_time_ns: u64, // DSSP calculation only
    total_time_ns: u64, // Including I/O
    file_results: []FileResult,
    allocator: Allocator,

    pub fn deinit(self: *BatchResult) void {
        for (self.file_results) |*result| {
            self.allocator.free(result.filename);
            if (result.error_msg) |msg| {
                self.allocator.free(msg);
            }
        }
        self.allocator.free(self.file_results);
    }

    /// Print human-readable summary
    pub fn printSummary(self: BatchResult, show_timing: bool) void {
        const ns_to_ms = 1_000_000.0;
        const total_dssp_ms = @as(f64, @floatFromInt(self.total_dssp_time_ns)) / ns_to_ms;
        const total_ms = @as(f64, @floatFromInt(self.total_time_ns)) / ns_to_ms;
        const throughput = if (total_ms > 0)
            @as(f64, @floatFromInt(self.successful)) / (total_ms / 1000.0)
        else
            0.0;

        std.debug.print("\nBatch Results:\n", .{});
        std.debug.print("  Total files:     {d}\n", .{self.total_files});
        std.debug.print("  Successful:      {d}\n", .{self.successful});
        std.debug.print("  Failed:          {d}\n", .{self.failed});
        std.debug.print("  Total DSSP time: {d:.2} ms\n", .{total_dssp_ms});
        std.debug.print("  Total time:      {d:.2} ms (includes I/O)\n", .{total_ms});
        std.debug.print("  Throughput:      {d:.1} files/sec\n", .{throughput});

        if (show_timing and self.successful > 0) {
            // Calculate timing statistics
            var min_ns: u64 = std.math.maxInt(u64);
            var max_ns: u64 = 0;
            var sum_ns: u64 = 0;
            var ok_count: usize = 0;

            for (self.file_results) |result| {
                if (result.status == .ok) {
                    min_ns = @min(min_ns, result.dssp_time_ns);
                    max_ns = @max(max_ns, result.dssp_time_ns);
                    sum_ns += result.dssp_time_ns;
                    ok_count += 1;
                }
            }

            if (ok_count > 0) {
                const min_ms = @as(f64, @floatFromInt(min_ns)) / ns_to_ms;
                const max_ms = @as(f64, @floatFromInt(max_ns)) / ns_to_ms;
                const mean_ms = @as(f64, @floatFromInt(sum_ns)) / @as(f64, @floatFromInt(ok_count)) / ns_to_ms;

                std.debug.print("\nTiming breakdown (DSSP only):\n", .{});
                std.debug.print("  Min:  {d:.2} ms\n", .{min_ms});
                std.debug.print("  Max:  {d:.2} ms\n", .{max_ms});
                std.debug.print("  Mean: {d:.2} ms\n", .{mean_ms});
            }
        }
    }
};

/// Detect input format from file extension
/// Handles .gz compressed files (e.g., file.pdb.gz is detected as PDB)
fn detectFormat(path: []const u8) InputFormat {
    // Strip .gz extension if present
    const base_path = gzip.stripGzExtension(path);

    // Find the last dot in the path
    var dot_pos: ?usize = null;
    for (base_path, 0..) |c, i| {
        if (c == '.') dot_pos = i;
    }

    if (dot_pos) |pos| {
        const ext = base_path[pos..];
        if (std.mem.eql(u8, ext, ".json")) return .json;
        if (std.mem.eql(u8, ext, ".cif") or std.mem.eql(u8, ext, ".mmcif")) return .mmcif;
        if (std.mem.eql(u8, ext, ".pdb") or std.mem.eql(u8, ext, ".ent")) return .pdb;
    }

    // Default to mmCIF (most common for modern structures)
    return .mmcif;
}

/// Check if filename matches the expected format (supports .gz compressed files)
fn matchesFormat(name: []const u8, format: InputFormat) bool {
    return switch (format) {
        .auto => blk: {
            // Accept all supported formats (including .gz compressed)
            if (std.mem.endsWith(u8, name, ".cif") or
                std.mem.endsWith(u8, name, ".cif.gz") or
                std.mem.endsWith(u8, name, ".mmcif") or
                std.mem.endsWith(u8, name, ".mmcif.gz") or
                std.mem.endsWith(u8, name, ".pdb") or
                std.mem.endsWith(u8, name, ".pdb.gz") or
                std.mem.endsWith(u8, name, ".ent") or
                std.mem.endsWith(u8, name, ".ent.gz") or
                std.mem.endsWith(u8, name, ".json") or
                std.mem.endsWith(u8, name, ".json.gz"))
            {
                break :blk true;
            }
            break :blk false;
        },
        .mmcif => std.mem.endsWith(u8, name, ".cif") or
            std.mem.endsWith(u8, name, ".cif.gz") or
            std.mem.endsWith(u8, name, ".mmcif") or
            std.mem.endsWith(u8, name, ".mmcif.gz"),
        .pdb => std.mem.endsWith(u8, name, ".pdb") or
            std.mem.endsWith(u8, name, ".pdb.gz") or
            std.mem.endsWith(u8, name, ".ent") or
            std.mem.endsWith(u8, name, ".ent.gz"),
        .json => std.mem.endsWith(u8, name, ".json") or std.mem.endsWith(u8, name, ".json.gz"),
    };
}

/// Maximum recursion depth to prevent stack overflow
const MAX_RECURSION_DEPTH: usize = 100;

/// Scan a single directory level and add matching files to the list
fn scanDirectoryLevel(
    allocator: Allocator,
    base_dir: std.fs.Dir,
    relative_prefix: []const u8,
    format: InputFormat,
    recursive: bool,
    files: *std.ArrayListUnmanaged([]const u8),
    depth: usize,
) !void {
    // Prevent stack overflow from deeply nested directories
    if (depth > MAX_RECURSION_DEPTH) return;

    var dir = if (relative_prefix.len == 0)
        base_dir
    else
        try base_dir.openDir(relative_prefix, .{ .iterate = true });
    defer if (relative_prefix.len > 0) dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        const name = entry.name;
        // Skip filenames with path separators (defense in depth)
        if (std.mem.indexOfAny(u8, name, "/\\") != null) continue;

        // Skip hidden files and directories (starting with '.')
        if (name.len > 0 and name[0] == '.') continue;

        // Handle directories recursively (note: directory symlinks are NOT followed to prevent cycles)
        if (entry.kind == .directory) {
            if (recursive) {
                // Build relative path for subdirectory
                const subdir_path = if (relative_prefix.len == 0)
                    try allocator.dupe(u8, name)
                else
                    try std.fs.path.join(allocator, &.{ relative_prefix, name });
                defer allocator.free(subdir_path);

                try scanDirectoryLevel(allocator, base_dir, subdir_path, format, recursive, files, depth + 1);
            }
            continue;
        }

        // Accept regular files and symlinks
        if (entry.kind != .file and entry.kind != .sym_link) continue;

        // Check extension based on format
        if (matchesFormat(name, format)) {
            // Build relative path from base directory
            const relative_path = if (relative_prefix.len == 0)
                try allocator.dupe(u8, name)
            else
                try std.fs.path.join(allocator, &.{ relative_prefix, name });
            try files.append(allocator, relative_path);
        }
    }
}

/// Scan directory for structure files
/// When recursive is true, descends into subdirectories and returns relative paths
pub fn scanDirectory(allocator: Allocator, dir_path: []const u8, format: InputFormat, recursive: bool) ![][]const u8 {
    var files = std.ArrayListUnmanaged([]const u8){};
    errdefer {
        for (files.items) |f| allocator.free(f);
        files.deinit(allocator);
    }

    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| {
        return err;
    };
    defer dir.close();

    try scanDirectoryLevel(allocator, dir, "", format, recursive, &files, 0);

    // Sort for deterministic ordering
    std.mem.sort([]const u8, files.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    return files.toOwnedSlice(allocator);
}

/// Maximum file size for reading (256 MB)
const MAX_FILE_SIZE = 256 * 1024 * 1024;

/// Output write buffer size (256 KB — reduces write syscalls vs 8 KB default)
const OUTPUT_BUF_SIZE = 256 * 1024;

/// Write batch timing data to JSON file (summary only, wall time)
pub fn writeTimingJson(result: BatchResult, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var write_buf: [4096]u8 = undefined;
    var buffered = file.writer(&write_buf);
    const writer = &buffered.interface;

    const ns_to_ms = 1_000_000.0;
    const total_ms = @as(f64, @floatFromInt(result.total_time_ns)) / ns_to_ms;
    const dssp_ms = @as(f64, @floatFromInt(result.total_dssp_time_ns)) / ns_to_ms;

    try writer.writeAll("{");
    try writer.print("\"total_files\":{d}", .{result.total_files});
    try writer.print(",\"successful\":{d}", .{result.successful});
    try writer.print(",\"failed\":{d}", .{result.failed});
    try writer.print(",\"total_dssp_time_ms\":{d:.3}", .{dssp_ms});
    try writer.print(",\"total_time_ms\":{d:.3}", .{total_ms});
    if (result.successful > 0) {
        const throughput = @as(f64, @floatFromInt(result.successful)) / (total_ms / 1000.0);
        try writer.print(",\"throughput_files_per_sec\":{d:.1}", .{throughput});
    }
    try writer.writeAll("}\n");

    try buffered.interface.flush();
}

/// Process a single file and return result
fn processOneFile(
    arena: Allocator,
    input_dir: []const u8,
    output_dir: ?[]const u8,
    filename: []const u8,
    config: BatchConfig,
    n_threads: usize,
) FileResult {
    return processOneFileImpl(arena, input_dir, output_dir, filename, config, n_threads) catch |err| {
        return FileResult{
            .filename = filename,
            .n_residues = 0,
            .dssp_time_ns = 0,
            .status = .err,
            .error_msg = std.fmt.allocPrint(arena, "{}", .{err}) catch null,
        };
    };
}

/// Internal implementation that returns errors
fn processOneFileImpl(
    arena: Allocator,
    input_dir: []const u8,
    output_dir: ?[]const u8,
    filename: []const u8,
    config: BatchConfig,
    n_threads: usize,
) !FileResult {
    // Build input path
    const input_path = try std.fs.path.join(arena, &.{ input_dir, filename });

    // Determine input format
    const format = if (config.input_format == .auto)
        detectFormat(filename)
    else
        config.input_format;

    // Read input file (handles gzip compression automatically)
    const input_data = try gzip.readFileMaybeCompressed(arena, input_path, MAX_FILE_SIZE);

    // Parse input
    const parse_result = switch (format) {
        .auto => unreachable, // Handled above
        .json => blk: {
            break :blk try json_parser.parseJsonInput(arena, input_data);
        },
        .mmcif => blk: {
            var parser = mmcif_parser.MmcifParser.init(arena);
            parser.model_num = config.model_num;
            break :blk try parser.parse(input_data);
        },
        .pdb => blk: {
            var parser = pdb_parser.PdbParser.init(arena);
            parser.model_num = config.model_num;
            break :blk try parser.parse(input_data);
        },
    };

    // Time DSSP calculation only
    var timer = try std.time.Timer.start();

    // Run DSSP calculation
    const dssp_config = DsspConfig{
        .pp_stretch = config.pp_stretch,
        .calculate_accessibility = config.calculate_accessibility,
        .n_threads = n_threads,
    };

    var dssp_result = try dssp.calculateFromParseResult(arena, parse_result, dssp_config);
    defer dssp_result.deinit();

    const dssp_time_ns = timer.read();
    const n_residues = dssp_result.statistics.total_residues;

    // Write output if directory specified
    if (output_dir) |out_dir| {
        // Generate output filename (replace extension with .dssp.json or .dssp)
        const output_ext = switch (config.output_format) {
            .json => ".dssp.json",
            .legacy => ".dssp",
        };

        // Strip .gz extension first if present
        const filename_without_gz = gzip.stripGzExtension(filename);

        // Split into directory and basename
        const dir_part = std.fs.path.dirname(filename_without_gz);
        const basename = std.fs.path.basename(filename_without_gz);

        // Find the first dot in basename to get the stem
        var stem_end: usize = basename.len;
        for (basename, 0..) |c, i| {
            if (c == '.') {
                stem_end = i;
                break;
            }
        }
        const stem = basename[0..stem_end];

        // Build output filename with new extension
        const output_filename = try std.fmt.allocPrint(arena, "{s}{s}", .{ stem, output_ext });

        // Build full output path, preserving subdirectory structure if present
        const output_path = if (dir_part) |dp|
            try std.fs.path.join(arena, &.{ out_dir, dp, output_filename })
        else
            try std.fs.path.join(arena, &.{ out_dir, output_filename });

        // Create parent directories if needed (for recursive mode)
        if (std.fs.path.dirname(output_path)) |parent| {
            try std.fs.cwd().makePath(parent);
        }

        const file = try std.fs.cwd().createFile(output_path, .{});
        defer file.close();

        var write_buf: [OUTPUT_BUF_SIZE]u8 = undefined;
        var writer = file.writer(&write_buf);
        switch (config.output_format) {
            .json => try json_writer.writeJson(&writer.interface, &dssp_result),
            .legacy => try json_writer.writeLegacy(&writer.interface, &dssp_result),
        }
        try writer.interface.flush();
    }

    return FileResult{
        .filename = filename,
        .n_residues = n_residues,
        .dssp_time_ns = dssp_time_ns,
        .status = .ok,
    };
}

/// Run batch processing sequentially
pub fn runBatchSequential(
    allocator: Allocator,
    input_dir: []const u8,
    output_dir: ?[]const u8,
    config: BatchConfig,
) !BatchResult {
    // Start total timer
    var total_timer = try std.time.Timer.start();

    // Scan directory for files
    const files = try scanDirectory(allocator, input_dir, config.input_format, config.recursive);
    defer {
        for (files) |f| allocator.free(f);
        allocator.free(files);
    }

    // Create output directory if specified
    if (output_dir) |out_dir| {
        try std.fs.cwd().makePath(out_dir);
    }

    // Allocate results
    const file_results = try allocator.alloc(FileResult, files.len);
    errdefer allocator.free(file_results);

    // Determine thread count for residue-level parallelism
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const n_threads_per_file = if (config.parallelism == .residue)
        if (config.n_threads == 0) cpu_count else @min(config.n_threads, cpu_count)
    else
        1;

    // Process each file
    var total_dssp_time_ns: u64 = 0;
    var successful: usize = 0;
    var failed: usize = 0;

    // Use arena allocator for each file (reset between files)
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    for (files, 0..) |filename, i| {
        // Copy filename to result allocator
        const filename_copy = try allocator.dupe(u8, filename);

        // Process file
        var result = processOneFile(
            arena.allocator(),
            input_dir,
            output_dir,
            filename,
            config,
            n_threads_per_file,
        );
        result.filename = filename_copy;

        if (result.status == .ok) {
            successful += 1;
            total_dssp_time_ns += result.dssp_time_ns;
        } else {
            failed += 1;
        }

        file_results[i] = result;

        // Reset arena for next file
        _ = arena.reset(.retain_capacity);

        // Progress output
        if (!config.quiet) {
            std.debug.print("\rProcessing: {d}/{d}", .{ i + 1, files.len });
        }
    }

    if (!config.quiet) {
        std.debug.print("\n", .{});
    }

    const total_time_ns = total_timer.read();

    return BatchResult{
        .total_files = files.len,
        .successful = successful,
        .failed = failed,
        .total_dssp_time_ns = total_dssp_time_ns,
        .total_time_ns = total_time_ns,
        .file_results = file_results,
        .allocator = allocator,
    };
}

/// Shared context for parallel batch workers (read-only after initialization).
/// Thread safety: `result_allocator` must be thread-safe (e.g., GPA with default config).
/// `processed_count` uses atomic operations for progress monitoring.
const BatchWorkerContext = struct {
    files: []const []const u8,
    input_dir: []const u8,
    output_dir: ?[]const u8,
    config: BatchConfig,
    /// Must be thread-safe. GPA with default config is thread-safe.
    result_allocator: Allocator,
    processed_count: *std.atomic.Value(usize),
};

/// Worker function for ThreadPool-based parallel batch processing.
/// Each call processes a single file (chunk_size=1).
fn batchWorker(ctx: BatchWorkerContext, start: usize, end: usize) FileResult {
    _ = end; // chunk_size=1, so end == start + 1

    const filename = ctx.files[start];

    // Copy filename to result allocator (thread-safe: GPA).
    // If even zero-byte alloc fails, OOM is catastrophic and unrecoverable.
    const filename_copy = ctx.result_allocator.dupe(u8, filename) catch {
        const empty = ctx.result_allocator.alloc(u8, 0) catch
            @panic("out of memory: cannot allocate 0 bytes for error result");
        _ = ctx.processed_count.fetchAdd(1, .monotonic);
        return FileResult{
            .filename = empty,
            .n_residues = 0,
            .dssp_time_ns = 0,
            .status = .err,
        };
    };

    // Each invocation gets its own arena
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var result = processOneFile(
        arena.allocator(),
        ctx.input_dir,
        ctx.output_dir,
        filename,
        ctx.config,
        1, // single-threaded DSSP per file
    );
    result.filename = filename_copy;

    _ = ctx.processed_count.fetchAdd(1, .monotonic);
    return result;
}

/// Progress monitoring thread — polls processed_count and prints status.
fn progressMonitor(processed_count: *std.atomic.Value(usize), total: usize) void {
    while (processed_count.load(.monotonic) < total) {
        const processed = processed_count.load(.monotonic);
        std.debug.print("\rProcessing: {d}/{d}", .{ processed, total });
        std.Thread.sleep(50 * std.time.ns_per_ms);
    }
    std.debug.print("\rProcessing: {d}/{d}\n", .{ total, total });
}

/// Run batch processing in parallel
pub fn runBatchParallel(
    allocator: Allocator,
    input_dir: []const u8,
    output_dir: ?[]const u8,
    config: BatchConfig,
) !BatchResult {
    // Start total timer
    var total_timer = try std.time.Timer.start();

    // Scan directory for files
    const files = try scanDirectory(allocator, input_dir, config.input_format, config.recursive);
    defer {
        for (files) |f| allocator.free(f);
        allocator.free(files);
    }

    if (files.len == 0) {
        return BatchResult{
            .total_files = 0,
            .successful = 0,
            .failed = 0,
            .total_dssp_time_ns = 0,
            .total_time_ns = total_timer.read(),
            .file_results = try allocator.alloc(FileResult, 0),
            .allocator = allocator,
        };
    }

    // Create output directory if specified
    if (output_dir) |out_dir| {
        try std.fs.cwd().makePath(out_dir);
    }

    // Allocate results
    const file_results = try allocator.alloc(FileResult, files.len);
    errdefer allocator.free(file_results);

    // Determine thread count
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const n_threads = if (config.n_threads == 0)
        cpu_count
    else
        @min(config.n_threads, cpu_count);

    // For single file or single thread, use sequential
    if (files.len == 1 or n_threads <= 1) {
        allocator.free(file_results);
        return runBatchSequential(allocator, input_dir, output_dir, config);
    }

    // Set up shared context
    var processed_count = std.atomic.Value(usize).init(0);
    const ctx = BatchWorkerContext{
        .files = files,
        .input_dir = input_dir,
        .output_dir = output_dir,
        .config = config,
        .result_allocator = allocator,
        .processed_count = &processed_count,
    };

    // Create thread pool (chunk_size=1: one file per work unit)
    const actual_threads = @min(n_threads, files.len);
    var pool = try thread_pool_mod.ThreadPool(BatchWorkerContext, FileResult).init(
        allocator, actual_threads, batchWorker, ctx, files.len, 1,
    );
    defer pool.deinit();

    // Spawn progress monitor before starting pool
    const progress_thread = if (!config.quiet)
        std.Thread.spawn(.{}, progressMonitor, .{ &processed_count, files.len }) catch |err| blk: {
            std.debug.print("Warning: could not start progress monitor: {}\n", .{err});
            break :blk null;
        }
    else
        null;

    try pool.run();

    // Join progress monitor
    if (progress_thread) |pt| {
        pt.join();
    }

    // Copy results from pool (pool.deinit() will free its internal storage)
    const pool_results = pool.getResults();
    std.debug.assert(pool_results.len == file_results.len);
    @memcpy(file_results, pool_results);

    // Aggregate results
    var total_dssp_time_ns: u64 = 0;
    var successful: usize = 0;
    var failed: usize = 0;

    for (file_results) |result| {
        if (result.status == .ok) {
            successful += 1;
            total_dssp_time_ns += result.dssp_time_ns;
        } else {
            failed += 1;
        }
    }

    const total_time_ns = total_timer.read();

    return BatchResult{
        .total_files = files.len,
        .successful = successful,
        .failed = failed,
        .total_dssp_time_ns = total_dssp_time_ns,
        .total_time_ns = total_time_ns,
        .file_results = file_results,
        .allocator = allocator,
    };
}

/// Run 3-phase batch: scan file sizes → split regular/heavy → process with optimal strategy.
/// Regular files: file-parallel (ThreadPool, 1 thread per file)
/// Heavy files: sequential with residue-parallel (all threads for accessibility)
fn runBatchAuto(
    allocator: Allocator,
    input_dir: []const u8,
    output_dir: ?[]const u8,
    config: BatchConfig,
) !BatchResult {
    var total_timer = try std.time.Timer.start();

    // Phase 1: Scan directory
    const files = try scanDirectory(allocator, input_dir, config.input_format, config.recursive);
    defer {
        for (files) |f| allocator.free(f);
        allocator.free(files);
    }

    if (files.len == 0) {
        return BatchResult{
            .total_files = 0,
            .successful = 0,
            .failed = 0,
            .total_dssp_time_ns = 0,
            .total_time_ns = total_timer.read(),
            .file_results = try allocator.alloc(FileResult, 0),
            .allocator = allocator,
        };
    }

    if (output_dir) |out_dir| {
        try std.fs.cwd().makePath(out_dir);
    }

    const cpu_count = std.Thread.getCpuCount() catch 1;
    const n_threads = if (config.n_threads == 0)
        cpu_count
    else
        @min(config.n_threads, cpu_count);

    // Phase 2: Split files into regular/heavy based on file size
    var regular_indices = std.ArrayListUnmanaged(usize){};
    defer regular_indices.deinit(allocator);
    var heavy_indices = std.ArrayListUnmanaged(usize){};
    defer heavy_indices.deinit(allocator);

    var dir = std.fs.cwd().openDir(input_dir, .{}) catch |err| {
        if (!config.quiet) {
            std.debug.print("Warning: could not open '{s}' for stat ({}) — falling back to file-parallel mode\n", .{ input_dir, err });
        }
        return runBatchParallel(allocator, input_dir, output_dir, config);
    };
    defer dir.close();

    for (files, 0..) |filename, i| {
        const stat = dir.statFile(filename) catch |err| {
            if (!config.quiet) {
                std.debug.print("Warning: cannot stat '{s}': {} — treating as regular\n", .{ filename, err });
            }
            try regular_indices.append(allocator, i);
            continue;
        };
        if (stat.size > config.heavy_file_threshold) {
            try heavy_indices.append(allocator, i);
        } else {
            try regular_indices.append(allocator, i);
        }
    }

    if (!config.quiet) {
        std.debug.print("Files: {d} regular, {d} heavy (>{d} MB)\n", .{
            regular_indices.items.len,
            heavy_indices.items.len,
            config.heavy_file_threshold / (1024 * 1024),
        });
    }

    // Allocate and zero-initialize results (defensive against missing index entries)
    const file_results = try allocator.alloc(FileResult, files.len);
    @memset(file_results, FileResult{ .filename = "", .n_residues = 0, .dssp_time_ns = 0, .status = .err });
    errdefer allocator.free(file_results);

    var total_dssp_time_ns: u64 = 0;
    var successful: usize = 0;
    var failed: usize = 0;

    // Phase 3a: Process regular files with file-parallel (ThreadPool)
    if (regular_indices.items.len > 0 and n_threads > 1) {
        var processed_count = std.atomic.Value(usize).init(0);

        // Build file list for regular files
        const regular_files = try allocator.alloc([]const u8, regular_indices.items.len);
        defer allocator.free(regular_files);
        for (regular_indices.items, 0..) |file_idx, i| {
            regular_files[i] = files[file_idx];
        }

        const ctx = BatchWorkerContext{
            .files = regular_files,
            .input_dir = input_dir,
            .output_dir = output_dir,
            .config = config,
            .result_allocator = allocator,
            .processed_count = &processed_count,
        };

        const actual_threads = @min(n_threads, regular_files.len);
        var pool = try thread_pool_mod.ThreadPool(BatchWorkerContext, FileResult).init(
            allocator, actual_threads, batchWorker, ctx, regular_files.len, 1,
        );
        defer pool.deinit();

        const progress_thread = if (!config.quiet)
            std.Thread.spawn(.{}, progressMonitor, .{ &processed_count, regular_files.len }) catch |err| blk: {
                std.debug.print("Warning: could not start progress monitor: {}\n", .{err});
                break :blk null;
            }
        else
            null;

        try pool.run();

        if (progress_thread) |pt| pt.join();

        // Map results back to original file indices
        const pool_results = pool.getResults();
        for (regular_indices.items, 0..) |file_idx, i| {
            file_results[file_idx] = pool_results[i];
        }
    } else if (regular_indices.items.len > 0) {
        // Single-thread: process regular files sequentially
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();

        for (regular_indices.items, 0..) |file_idx, i| {
            const filename = files[file_idx];
            const filename_copy = try allocator.dupe(u8, filename);
            var result = processOneFile(arena.allocator(), input_dir, output_dir, filename, config, 1);
            result.filename = filename_copy;
            file_results[file_idx] = result;
            _ = arena.reset(.retain_capacity);
            if (!config.quiet) {
                std.debug.print("\rProcessing regular: {d}/{d}", .{ i + 1, regular_indices.items.len });
            }
        }
        if (!config.quiet and regular_indices.items.len > 0) {
            std.debug.print("\n", .{});
        }
    }

    // Phase 3b: Process heavy files sequentially with residue-parallel
    if (heavy_indices.items.len > 0) {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();

        for (heavy_indices.items, 0..) |file_idx, i| {
            const filename = files[file_idx];
            const filename_copy = try allocator.dupe(u8, filename);
            // Use all threads for residue-level parallelism on heavy files
            var result = processOneFile(arena.allocator(), input_dir, output_dir, filename, config, n_threads);
            result.filename = filename_copy;
            file_results[file_idx] = result;
            _ = arena.reset(.retain_capacity);
            if (!config.quiet) {
                std.debug.print("\rProcessing heavy: {d}/{d}", .{ i + 1, heavy_indices.items.len });
            }
        }
        if (!config.quiet) {
            std.debug.print("\n", .{});
        }
    }

    // Aggregate
    for (file_results) |result| {
        if (result.status == .ok) {
            successful += 1;
            total_dssp_time_ns += result.dssp_time_ns;
        } else {
            failed += 1;
        }
    }

    return BatchResult{
        .total_files = files.len,
        .successful = successful,
        .failed = failed,
        .total_dssp_time_ns = total_dssp_time_ns,
        .total_time_ns = total_timer.read(),
        .file_results = file_results,
        .allocator = allocator,
    };
}

/// Run batch processing (main entry point)
/// Selects strategy based on config.parallelism:
/// - auto: 3-phase scan/split/process (default)
/// - file: N files in parallel, 1 thread per file
/// - residue: 1 file at a time, N threads for accessibility calculation
pub fn runBatch(
    allocator: Allocator,
    input_dir: []const u8,
    output_dir: ?[]const u8,
    config: BatchConfig,
) !BatchResult {
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const n_threads = if (config.n_threads == 0)
        cpu_count
    else
        config.n_threads;

    return switch (config.parallelism) {
        // 3-phase: scan file sizes, split, process with optimal strategy
        .auto => if (n_threads <= 1)
            runBatchSequential(allocator, input_dir, output_dir, config)
        else
            runBatchAuto(allocator, input_dir, output_dir, config),
        // File-level parallelism: N files in parallel, 1 thread per file
        .file => if (n_threads <= 1)
            runBatchSequential(allocator, input_dir, output_dir, config)
        else
            runBatchParallel(allocator, input_dir, output_dir, config),
        // Residue-level parallelism: sequential files, N threads per file
        .residue => runBatchSequential(allocator, input_dir, output_dir, config),
    };
}

// Tests

test "BatchConfig default values" {
    const config = BatchConfig{};

    try std.testing.expectEqual(@as(usize, 0), config.n_threads);
    try std.testing.expectEqual(Parallelism.auto, config.parallelism);
    try std.testing.expectEqual(InputFormat.auto, config.input_format);
    try std.testing.expectEqual(@as(u32, 3), config.pp_stretch);
    try std.testing.expect(config.calculate_accessibility);
    try std.testing.expect(!config.recursive); // Default: non-recursive
}

test "matchesFormat" {
    // mmCIF format
    try std.testing.expect(matchesFormat("test.cif", .mmcif));
    try std.testing.expect(matchesFormat("test.cif.gz", .mmcif));
    try std.testing.expect(matchesFormat("test.mmcif", .mmcif));
    try std.testing.expect(!matchesFormat("test.pdb", .mmcif));

    // PDB format
    try std.testing.expect(matchesFormat("test.pdb", .pdb));
    try std.testing.expect(matchesFormat("test.pdb.gz", .pdb));
    try std.testing.expect(matchesFormat("test.ent", .pdb));
    try std.testing.expect(!matchesFormat("test.cif", .pdb));

    // JSON format
    try std.testing.expect(matchesFormat("test.json", .json));
    try std.testing.expect(matchesFormat("test.json.gz", .json));
    try std.testing.expect(!matchesFormat("test.cif", .json));

    // Auto format (accepts all)
    try std.testing.expect(matchesFormat("test.cif", .auto));
    try std.testing.expect(matchesFormat("test.pdb", .auto));
    try std.testing.expect(matchesFormat("test.json", .auto));
    try std.testing.expect(!matchesFormat("test.txt", .auto));
}

test "BatchResult deinit" {
    const allocator = std.testing.allocator;

    var results = try allocator.alloc(FileResult, 1);
    results[0] = FileResult{
        .filename = try allocator.dupe(u8, "test.cif"),
        .n_residues = 100,
        .dssp_time_ns = 1000000,
        .status = .ok,
    };

    var batch_result = BatchResult{
        .total_files = 1,
        .successful = 1,
        .failed = 0,
        .total_dssp_time_ns = 1000000,
        .total_time_ns = 2000000,
        .file_results = results,
        .allocator = allocator,
    };

    batch_result.deinit();
}

test "detectFormat" {
    try std.testing.expectEqual(InputFormat.json, detectFormat("test.json"));
    try std.testing.expectEqual(InputFormat.mmcif, detectFormat("test.cif"));
    try std.testing.expectEqual(InputFormat.mmcif, detectFormat("test.mmcif"));
    try std.testing.expectEqual(InputFormat.pdb, detectFormat("test.pdb"));
    try std.testing.expectEqual(InputFormat.pdb, detectFormat("test.ent"));
    try std.testing.expectEqual(InputFormat.mmcif, detectFormat("test")); // default
    // Gzip compressed files
    try std.testing.expectEqual(InputFormat.json, detectFormat("test.json.gz"));
    try std.testing.expectEqual(InputFormat.mmcif, detectFormat("test.cif.gz"));
    try std.testing.expectEqual(InputFormat.mmcif, detectFormat("test.mmcif.gz"));
    try std.testing.expectEqual(InputFormat.pdb, detectFormat("test.pdb.gz"));
    try std.testing.expectEqual(InputFormat.pdb, detectFormat("test.ent.gz"));
}
