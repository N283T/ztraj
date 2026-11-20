const std = @import("std");

const version = "0.9.0";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // When set, only the `ztraj-io` module is configured and the lazy `zsasa`
    // dependency is never requested, so it is never fetched. Package consumers
    // that only need file I/O pass `.io_only = true` to `b.dependency("ztraj")`.
    const io_only = b.option(
        bool,
        "io_only",
        "Only expose the ztraj-io module; do not fetch zsasa or build the CLI/library",
    ) orelse false;

    // Standalone structure/trajectory I/O module (no zsasa dependency).
    const io_mod = b.addModule("ztraj-io", .{
        .root_source_file = b.path("src/io_root.zig"),
        .target = target,
    });

    // The ztraj-io test binary is only compiled, which checks that the module
    // builds on its own without zsasa. Its format tests already run through
    // the ztraj module; running them here too would race on the fixed
    // temporary file paths those tests write under test_data/.
    const test_step = b.step("test", "Run tests");
    const io_tests = b.addTest(.{ .root_module = io_mod });
    test_step.dependOn(&io_tests.step);

    if (io_only) return;

    // zsasa is a lazy dependency. On the first run without it, this returns
    // null; the build runner then fetches it and re-runs this function. The
    // modules and artifacts below are still declared on that first pass (just
    // without the zsasa import) so that consumers looking up `ztraj` do not
    // fail before the fetch happens.
    const zsasa_mod: ?*std.Build.Module = if (b.lazyDependency("zsasa", .{
        .target = target,
        .optimize = optimize,
    })) |dep| dep.module("zsasa") else null;

    // Library module (exposed to package consumers via zig fetch)
    const mod = b.addModule("ztraj", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // Shared library (C API for Python bindings)
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "ztraj",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_api.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    if (zsasa_mod) |zsasa| {
        mod.addImport("zsasa", zsasa);
        lib.root_module.addImport("zsasa", zsasa);
    }
    b.installArtifact(lib);

    // CLI executable
    const options = b.addOptions();
    options.addOption([]const u8, "version", version);

    const exe = b.addExecutable(.{
        .name = "ztraj",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ztraj", .module = mod },
                .{ .name = "build_options", .module = options.createModule() },
            },
        }),
    });
    b.installArtifact(exe);

    // Run step
    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Test step
    const mod_tests = b.addTest(.{ .root_module = mod });
    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    test_step.dependOn(&b.addRunArtifact(mod_tests).step);
    test_step.dependOn(&b.addRunArtifact(exe_tests).step);

    // C API: tests are verified via Python integration tests.
    // The shared library compiles c_api.zig; Zig-level c_api tests cannot run
    // as a separate module because pdb.zig's @embedFile uses relative paths
    // that go outside the c_api module's package root.

    // Docs step (zig autodoc)
    const docs_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "ztraj",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
        }),
    });
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Emit autodoc to zig-out/docs");
    docs_step.dependOn(&install_docs.step);
}
