const std = @import("std");

const version = "0.6.2";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zxdrfile_dep = b.dependency("zxdrfile", .{
        .target = target,
        .optimize = optimize,
    });
    const zxdrfile_mod = zxdrfile_dep.module("zxdrfile");

    const zsasa_dep = b.dependency("zsasa", .{
        .target = target,
        .optimize = optimize,
    });
    const zsasa_mod = zsasa_dep.module("zsasa");

    // Library module (exposed to package consumers via zig fetch)
    const mod = b.addModule("ztraj", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "zxdrfile", .module = zxdrfile_mod },
            .{ .name = "zsasa", .module = zsasa_mod },
        },
    });

    // Shared library (C API for Python bindings)
    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "ztraj",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_api.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zxdrfile", .module = zxdrfile_mod },
                .{ .name = "zsasa", .module = zsasa_mod },
            },
        }),
    });
    lib.linkLibC();
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
                .{ .name = "zxdrfile", .module = zxdrfile_mod },
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
    const test_step = b.step("test", "Run tests");
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
