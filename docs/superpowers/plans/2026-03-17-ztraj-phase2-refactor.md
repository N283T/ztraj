# Phase 2 Refactoring Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `main.zig` and `simd.zig` into focused sub-modules under `src/cli/` and `src/simd/`.

**Architecture:** Pure structural refactoring — move functions into new files, update imports, verify tests pass. No behavioral changes. Three independent commits: CLI split, SIMD split, Vec3Gen dedup.

**Tech Stack:** Zig 0.15.2

**Spec:** `docs/superpowers/specs/2026-03-17-ztraj-phase2-refactor-design.md`

---

## Chunk 1: Split main.zig into src/cli/

### Task 1: Create cli/args.zig

**Files:**
- Create: `src/cli/args.zig`
- Modify: `src/main.zig`

- [ ] **Step 1: Create `src/cli/args.zig`**

Extract from `src/main.zig`:
- Lines 21-35: `Subcommand` enum
- Lines 37-79: `Args` struct
- Lines 81-124: `printUsage`
- Lines 126-138: `ParseArgsError`
- Lines 140-259: `parseArgs`

The file needs these imports at the top:
```zig
const std = @import("std");
```

All types/functions must be `pub`. `printUsage` takes `prog_name: []const u8` and writes to stderr.

- [ ] **Step 2: Run tests**

Run: `zig build test`
Expected: All tests pass (args.zig has no tests, but compilation must succeed)

---

### Task 2: Create cli/loader.zig

**Files:**
- Create: `src/cli/loader.zig`

- [ ] **Step 1: Create `src/cli/loader.zig`**

Extract from `src/main.zig`:
- Lines 260-267: `endsWithCI`
- Lines 269-283: `isPdb`, `isCif`, `isXtc`, `isDcd`
- Lines 285-290: `TopologyFormat` (if exists, or file-type comment)
- Lines 291-306: `loadTopology`
- Lines 470-527: `loadAllFrames`

Imports needed:
```zig
const std = @import("std");
const ztraj = @import("ztraj");
const types = ztraj.types;
const io = ztraj.io;
```

All functions must be `pub`.

- [ ] **Step 2: Run tests**

Run: `zig build test`
Expected: Compilation succeeds

---

### Task 3: Create cli/parsers.zig

**Files:**
- Create: `src/cli/parsers.zig`

- [ ] **Step 1: Create `src/cli/parsers.zig`**

Extract from `src/main.zig`:
- Lines 314-335: `resolveSelection`
- Lines 342-355: `parsePairs`
- Lines 358-377: `parseTriplets`
- Lines 380-399: `parseQuartets`
- Lines 402-414: `validateIndices`

Imports needed:
```zig
const std = @import("std");
const ztraj = @import("ztraj");
const types = ztraj.types;
const select = ztraj.select;
```

All functions must be `pub`.

- [ ] **Step 2: Run tests**

Run: `zig build test`
Expected: Compilation succeeds

---

### Task 4: Create cli/runners.zig

**Files:**
- Create: `src/cli/runners.zig`

- [ ] **Step 1: Create `src/cli/runners.zig`**

Extract from `src/main.zig`:
- Lines 405-462: `flushOutput`, `writeScalarSeriesBuf`
- Lines 533-1278: All 11 `runXxx` functions

Imports needed:
```zig
const std = @import("std");
const ztraj = @import("ztraj");
const types = ztraj.types;
const geometry = ztraj.geometry;
const analysis = ztraj.analysis;
const output = ztraj.output;
const args_mod = @import("args.zig");
const Args = args_mod.Args;
const loader = @import("loader.zig");
const parsers = @import("parsers.zig");
```

Each `runXxx` function signature changes from `fn runXxx(allocator: std.mem.Allocator, args: Args)` to use the imported `Args` type. All functions must be `pub`.

- [ ] **Step 2: Run tests**

Run: `zig build test`
Expected: Compilation succeeds

---

### Task 5: Rewrite main.zig as thin dispatcher

**Files:**
- Modify: `src/main.zig`

- [ ] **Step 1: Replace main.zig contents**

The new `main.zig` should be ~80 lines:
```zig
const std = @import("std");
const build_options = @import("build_options");
const cli_args = @import("cli/args.zig");
const runners = @import("cli/runners.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const raw_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, raw_args);

    if (raw_args.len < 2) {
        cli_args.printUsage(raw_args[0]);
        std.process.exit(1);
    }

    const first = raw_args[1];

    if (std.mem.eql(u8, first, "--version") or std.mem.eql(u8, first, "-V")) {
        const stdout = std.fs.File.stdout();
        var buf: [64]u8 = undefined;
        const line = try std.fmt.bufPrint(&buf, "ztraj {s}\n", .{build_options.version});
        try stdout.writeAll(line);
        return;
    }
    if (std.mem.eql(u8, first, "--help") or std.mem.eql(u8, first, "-h")) {
        cli_args.printUsage(raw_args[0]);
        return;
    }

    const args = cli_args.parseArgs(raw_args) catch |err| {
        // ... same error switch as before, using cli_args.printUsage ...
    };

    // Dispatch — same switch as before, calling runners.runXxx
}
```

Preserve the full error handling switch and subcommand dispatch from the original.

- [ ] **Step 2: Verify build**

Run: `zig build`
Expected: Compiles with no errors

- [ ] **Step 3: Run all tests**

Run: `zig build test`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/main.zig src/cli/
git commit -m "refactor: split main.zig into src/cli/ modules

Extract CLI logic into focused modules:
- cli/args.zig: argument parsing and usage
- cli/loader.zig: topology and trajectory loading
- cli/parsers.zig: index spec parsing and validation
- cli/runners.zig: subcommand implementations

main.zig is now a thin dispatcher (~80 lines)."
```

---

## Chunk 2: Split simd.zig into src/simd/

### Task 6: Create simd/vec.zig

**Files:**
- Create: `src/simd/vec.zig`

- [ ] **Step 1: Create `src/simd/vec.zig`**

Extract from `src/simd.zig`:
- Lines 1-65: All type definitions (`Vec3Gen`, `Vec3`, `Vec3f32`, `Epsilon`, `cpu_features`, `optimal_vector_width`)

Imports needed:
```zig
const std = @import("std");
const builtin = @import("builtin");
```

All types must be `pub`.

- [ ] **Step 2: Run tests**

Run: `zig build test`
Expected: Compilation succeeds

---

### Task 7: Create simd/trig.zig

**Files:**
- Create: `src/simd/trig.zig`

- [ ] **Step 1: Create `src/simd/trig.zig`**

Extract from `src/simd.zig`:
- Lines 70-98: `fastAcos` (f64)
- Lines 100-143: `fastAtan2` (f64)
- Lines 771-789: `fastAcosGen` (generic)
- Lines 792-829: `fastAtan2Gen` (generic)
- Lines 1061-1109: Trig tests (fastAcos accuracy, edge cases, fastAtan2 accuracy, edge cases)
- Lines 1202-1224: Generic trig tests (f32 accuracy)

Imports needed:
```zig
const std = @import("std");
const vec = @import("vec.zig");
const Epsilon = vec.Epsilon;
```

- [ ] **Step 2: Run tests**

Run: `zig build test`
Expected: All trig tests pass

---

### Task 8: Create simd/distance.zig

**Files:**
- Create: `src/simd/distance.zig`

- [ ] **Step 1: Create `src/simd/distance.zig`**

Extract from `src/simd.zig`:
- Lines 145-339: `distanceSquaredBatch{4,8,16}`, `isPointBuriedBatch{4,8,16}` (f64)
- Lines 343-626: All distance/point-buried tests (f64)
- Lines 628-768: Generic variants (`distanceSquaredBatch{4,8,16}Gen`, `isPointBuriedBatch{4,8,16}Gen`)
- Lines 1173-1200: CPU feature detection test
- Lines 1230-1324: Generic distance/point-buried tests (f32)

Imports needed:
```zig
const std = @import("std");
const vec = @import("vec.zig");
pub const Vec3Gen = vec.Vec3Gen;
pub const Vec3 = vec.Vec3;
pub const Vec3f32 = vec.Vec3f32;
pub const cpu_features = vec.cpu_features;
pub const optimal_vector_width = vec.optimal_vector_width;
```

- [ ] **Step 2: Run tests**

Run: `zig build test`
Expected: All distance tests pass

---

### Task 9: Create simd/lee_richards.zig

**Files:**
- Create: `src/simd/lee_richards.zig`

- [ ] **Step 1: Create `src/simd/lee_richards.zig`**

Extract from `src/simd.zig`:
- Lines 831-920: `xyDistanceBatch4`, `sliceRadiiBatch4`, `circlesOverlapBatch4` (f64)
- Lines 922-1007: 8-wide variants (f64)
- Lines 1011-1167: Lee-Richards tests (f64 4-wide, 8-wide)
- Lines 1331-1468: Generic variants + tests

Imports needed:
```zig
const std = @import("std");
```

- [ ] **Step 2: Run tests**

Run: `zig build test`
Expected: All lee_richards tests pass

---

### Task 10: Rewrite simd.zig as re-export hub

**Files:**
- Modify: `src/simd.zig`

- [ ] **Step 1: Replace simd.zig contents**

```zig
//! SIMD utilities for vectorized computation.
//!
//! Re-exports sub-modules. Import this file for the full SIMD API.

pub const vec = @import("simd/vec.zig");
pub const trig = @import("simd/trig.zig");
pub const distance = @import("simd/distance.zig");
pub const lee_richards = @import("simd/lee_richards.zig");

// Re-export commonly used types at the top level for convenience.
pub const Vec3Gen = vec.Vec3Gen;
pub const Vec3 = vec.Vec3;
pub const Vec3f32 = vec.Vec3f32;
pub const Epsilon = vec.Epsilon;
pub const cpu_features = vec.cpu_features;
pub const optimal_vector_width = vec.optimal_vector_width;

// Re-export trig functions.
pub const fastAcos = trig.fastAcos;
pub const fastAtan2 = trig.fastAtan2;
pub const fastAcosGen = trig.fastAcosGen;
pub const fastAtan2Gen = trig.fastAtan2Gen;

// Re-export distance functions.
pub const distanceSquaredBatch4 = distance.distanceSquaredBatch4;
pub const distanceSquaredBatch8 = distance.distanceSquaredBatch8;
pub const distanceSquaredBatch16 = distance.distanceSquaredBatch16;
pub const isPointBuriedBatch4 = distance.isPointBuriedBatch4;
pub const isPointBuriedBatch8 = distance.isPointBuriedBatch8;
pub const isPointBuriedBatch16 = distance.isPointBuriedBatch16;
pub const distanceSquaredBatch4Gen = distance.distanceSquaredBatch4Gen;
pub const distanceSquaredBatch8Gen = distance.distanceSquaredBatch8Gen;
pub const distanceSquaredBatch16Gen = distance.distanceSquaredBatch16Gen;
pub const isPointBuriedBatch4Gen = distance.isPointBuriedBatch4Gen;
pub const isPointBuriedBatch8Gen = distance.isPointBuriedBatch8Gen;
pub const isPointBuriedBatch16Gen = distance.isPointBuriedBatch16Gen;

// Re-export Lee-Richards helpers.
pub const xyDistanceBatch4 = lee_richards.xyDistanceBatch4;
pub const xyDistanceBatch8 = lee_richards.xyDistanceBatch8;
pub const sliceRadiiBatch4 = lee_richards.sliceRadiiBatch4;
pub const sliceRadiiBatch8 = lee_richards.sliceRadiiBatch8;
pub const circlesOverlapBatch4 = lee_richards.circlesOverlapBatch4;
pub const circlesOverlapBatch8 = lee_richards.circlesOverlapBatch8;
pub const xyDistanceBatch4Gen = lee_richards.xyDistanceBatch4Gen;
pub const xyDistanceBatch8Gen = lee_richards.xyDistanceBatch8Gen;
pub const sliceRadiiBatch4Gen = lee_richards.sliceRadiiBatch4Gen;
pub const sliceRadiiBatch8Gen = lee_richards.sliceRadiiBatch8Gen;
pub const circlesOverlapBatch4Gen = lee_richards.circlesOverlapBatch4Gen;
pub const circlesOverlapBatch8Gen = lee_richards.circlesOverlapBatch8Gen;

test {
    @import("std").testing.refAllDecls(@This());
}
```

- [ ] **Step 2: Verify all simd users still compile**

Check that `src/neighbor_list.zig` and any other files importing simd still work.

- [ ] **Step 3: Run all tests**

Run: `zig build test`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/simd.zig src/simd/
git commit -m "refactor: split simd.zig into src/simd/ modules

Extract SIMD utilities into focused sub-modules:
- simd/vec.zig: Vec3Gen types, CPU features, vector widths
- simd/trig.zig: fast acos/atan2 approximations
- simd/distance.zig: batched distance and point-in-sphere
- simd/lee_richards.zig: Lee-Richards SASA helpers

simd.zig is now a re-export hub preserving the full public API."
```

---

## Chunk 3: Vec3Gen deduplication

### Task 11: Deduplicate Vec3Gen in neighbor_list.zig

**Files:**
- Modify: `src/neighbor_list.zig`

- [ ] **Step 1: Replace local Vec3Gen definitions**

In `src/neighbor_list.zig`, remove the local definitions of `Vec3Gen`, `Vec3`, `Vec3f32` (lines 10-21) and replace with imports:

```zig
const simd_vec = @import("simd/vec.zig");
const Vec3Gen = simd_vec.Vec3Gen;
const Vec3 = simd_vec.Vec3;
const Vec3f32 = simd_vec.Vec3f32;
```

- [ ] **Step 2: Run all tests**

Run: `zig build test`
Expected: All tests pass (neighbor_list tests + full suite)

- [ ] **Step 3: Commit**

```bash
git add src/neighbor_list.zig
git commit -m "refactor: deduplicate Vec3Gen — import from simd/vec.zig

neighbor_list.zig now imports Vec3Gen/Vec3/Vec3f32 from simd/vec.zig
instead of defining its own copies."
```

---

## Post-refactor Verification

- [ ] **Final check: line counts**

Run: `wc -l src/main.zig src/cli/*.zig src/simd.zig src/simd/*.zig src/neighbor_list.zig`

Expected:
- `src/main.zig`: ~80 lines
- `src/cli/args.zig`: ~200 lines
- `src/cli/loader.zig`: ~290 lines
- `src/cli/parsers.zig`: ~100 lines
- `src/cli/runners.zig`: ~860 lines
- `src/simd.zig`: ~60 lines
- `src/simd/vec.zig`: ~70 lines
- `src/simd/trig.zig`: ~200 lines
- `src/simd/distance.zig`: ~500 lines
- `src/simd/lee_richards.zig`: ~500 lines

---
- [ ] **DONE** - Phase complete
