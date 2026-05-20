# Integrated XDR Trajectory I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move zxdrfile's XTC/TRR implementation into ztraj so Zig-version maintenance happens in one repository while preserving ztraj's existing high-level XTC/TRR API.

**Architecture:** Keep low-level format-native XDR/XTC/TRR code under `src/io/`, and keep ztraj `types.Frame` conversion in the existing high-level `src/io/xtc.zig` and `src/io/trr.zig` wrappers. Add `src/io/root.zig` as the I/O namespace root, then remove the external `zxdrfile` build dependency.

**Tech Stack:** Zig 0.16 build system, ztraj Zig library/CLI/shared library, existing zxdrfile v0.4.0 source from `zig-pkg/zxdrfile-0.4.0-JIaQ9mIfAgAHB9bW-_p9g6_gfd0tyG3-MmA0LYJKB8rb/src/`.

---

## File Structure

- Create `src/io/xdr.zig`: formerly `zxdrfile`'s `xdrfile.zig`; re-exports low-level `xtc_format` and `trr_format` modules and their reader/writer/frame/error types.
- Create `src/io/xtc_format.zig`: formerly `zxdrfile`'s `xdrfile_xtc.zig`; low-level XTC reader/writer using format-native nm coordinates and packed buffers.
- Create `src/io/trr_format.zig`: formerly `zxdrfile`'s `xdrfile_trr.zig`; low-level TRR reader/writer using format-native optional coordinate/velocity/force buffers.
- Create `src/io/root.zig`: ztraj I/O namespace root, exporting existing parsers and the new low-level XDR module.
- Modify `src/io/xtc.zig`: replace `@import("zxdrfile")` with `@import("xdr.zig")`; keep high-level public API unchanged.
- Modify `src/io/trr.zig`: replace `@import("zxdrfile")` with `@import("xdr.zig")`; keep high-level public API unchanged.
- Modify `src/root.zig`: replace inline `pub const io = struct { ... }` with `pub const io = @import("io/root.zig");`; keep root-level tests pulling the same high-level modules.
- Modify `build.zig`: remove `zxdrfile` dependency loading and module imports; keep `zsasa` imports unchanged.
- Modify `build.zig.zon`: remove `.zxdrfile` dependency entry; keep `.zsasa` unchanged.

## Task 1: Add the I/O namespace root

**Files:**
- Create: `src/io/root.zig`
- Modify: `src/root.zig`

- [ ] **Step 1: Create `src/io/root.zig` with current I/O exports**

Create the file with this content:

```zig
//! ztraj I/O namespace.
//!
//! High-level parsers and trajectory readers/writers live here. Low-level
//! format-native helpers are only exported when they are useful for internal
//! composition or advanced Zig consumers.

pub const pdb = @import("pdb.zig");
pub const mmcif = @import("mmcif.zig");
pub const cif_tokenizer = @import("cif_tokenizer.zig");
pub const xtc = @import("xtc.zig");
pub const trr = @import("trr.zig");
pub const dcd = @import("dcd.zig");
pub const gro = @import("gro.zig");
pub const prmtop = @import("prmtop.zig");
pub const nc = @import("nc.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
```

- [ ] **Step 2: Update `src/root.zig` to import the new namespace root**

Replace the current inline `pub const io = struct { ... };` block with:

```zig
pub const io = @import("io/root.zig");
```

Do not change the existing root test block yet; it should continue to reference `io.gro`, `io.prmtop`, `io.nc`, `io.dcd`, `io.xtc`, `io.trr`, and `io.mmcif`.

- [ ] **Step 3: Run the current test suite to verify this pure namespace refactor**

Run:

```bash
zig build test
```

Expected: PASS. If it fails, the failure should be from the namespace move; fix the import path or missing export before continuing.

- [ ] **Step 4: Commit the namespace root**

Run:

```bash
git add src/io/root.zig src/root.zig
git commit -m "refactor: add io namespace root"
```

## Task 2: Import zxdrfile source as low-level ztraj I/O modules

**Files:**
- Create: `src/io/xdr.zig`
- Create: `src/io/xtc_format.zig`
- Create: `src/io/trr_format.zig`
- Modify: `src/io/root.zig`

- [ ] **Step 1: Copy zxdrfile source files into ztraj**

Run:

```bash
cp zig-pkg/zxdrfile-0.4.0-JIaQ9mIfAgAHB9bW-_p9g6_gfd0tyG3-MmA0LYJKB8rb/src/xdrfile.zig src/io/xdr.zig
cp zig-pkg/zxdrfile-0.4.0-JIaQ9mIfAgAHB9bW-_p9g6_gfd0tyG3-MmA0LYJKB8rb/src/xdrfile_xtc.zig src/io/xtc_format.zig
cp zig-pkg/zxdrfile-0.4.0-JIaQ9mIfAgAHB9bW-_p9g6_gfd0tyG3-MmA0LYJKB8rb/src/xdrfile_trr.zig src/io/trr_format.zig
```

- [ ] **Step 2: Update `src/io/xdr.zig` imports to the new file names**

In `src/io/xdr.zig`, replace the module declarations near the top with:

```zig
pub const xtc = @import("xtc_format.zig");
pub const trr = @import("trr_format.zig");
```

Keep these public aliases unchanged because the high-level wrappers will use them:

```zig
pub const XtcReader = xtc.XtcReader;
pub const XtcWriter = xtc.XtcWriter;
pub const XtcFrame = xtc.XtcFrame;
pub const XtcError = xtc.XtcError;

pub const TrrReader = trr.TrrReader;
pub const TrrWriter = trr.TrrWriter;
pub const TrrFrame = trr.TrrFrame;
pub const TrrError = trr.TrrError;
```

- [ ] **Step 3: Add the low-level module to `src/io/root.zig`**

Add this export after the high-level parser exports or at the end of the file:

```zig
pub const xdr = @import("xdr.zig");
```

- [ ] **Step 4: Run the focused compile check before changing wrappers**

Run:

```bash
zig build test
```

Expected: PASS. This verifies the copied low-level code compiles inside ztraj before it is wired into the high-level wrappers.

- [ ] **Step 5: Commit copied low-level modules**

Run:

```bash
git add src/io/xdr.zig src/io/xtc_format.zig src/io/trr_format.zig src/io/root.zig
git commit -m "feat: vendor xdr trajectory formats into io"
```

## Task 3: Switch high-level XTC/TRR wrappers to integrated low-level modules

**Files:**
- Modify: `src/io/xtc.zig`
- Modify: `src/io/trr.zig`

- [ ] **Step 1: Update `src/io/xtc.zig` import and type aliases**

Replace:

```zig
const zxdrfile = @import("zxdrfile");

const XtcReaderInner = zxdrfile.XtcReader;
const XtcWriterInner = zxdrfile.XtcWriter;
const XtcError = zxdrfile.XtcError;
```

with:

```zig
const xdr = @import("xdr.zig");

const XtcReaderInner = xdr.XtcReader;
const XtcWriterInner = xdr.XtcWriter;
const XtcError = xdr.XtcError;
```

No other public names in `src/io/xtc.zig` should change.

- [ ] **Step 2: Update `src/io/trr.zig` import and type aliases**

Replace:

```zig
const zxdrfile = @import("zxdrfile");

const TrrReaderInner = zxdrfile.TrrReader;
const TrrWriterInner = zxdrfile.TrrWriter;
const TrrError = zxdrfile.TrrError;
```

with:

```zig
const xdr = @import("xdr.zig");

const TrrReaderInner = xdr.TrrReader;
const TrrWriterInner = xdr.TrrWriter;
const TrrError = xdr.TrrError;
```

No other public names in `src/io/trr.zig` should change.

- [ ] **Step 3: Verify no high-level wrapper imports `zxdrfile`**

Run:

```bash
rg -n '@import\("zxdrfile"\)|\bzxdrfile\b' src build.zig build.zig.zon
```

Expected before Task 4: matches may still appear in `build.zig` and `build.zig.zon`, but there should be no matches in `src/`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
zig build test
```

Expected: PASS. This verifies high-level ztraj wrappers work against the integrated low-level modules while the external dependency still exists in build metadata.

- [ ] **Step 5: Commit wrapper switch**

Run:

```bash
git add src/io/xtc.zig src/io/trr.zig
git commit -m "refactor: use integrated xdr trajectory modules"
```

## Task 4: Remove external zxdrfile build dependency

**Files:**
- Modify: `build.zig`
- Modify: `build.zig.zon`

- [ ] **Step 1: Remove dependency loading from `build.zig`**

Delete this block near the top of `build.zig`:

```zig
const zxdrfile_dep = b.dependency("zxdrfile", .{
    .target = target,
    .optimize = optimize,
});
const zxdrfile_mod = zxdrfile_dep.module("zxdrfile");
```

- [ ] **Step 2: Remove `zxdrfile` imports from ztraj module creation**

In the `b.addModule("ztraj", ...)` imports list, remove this entry:

```zig
.{ .name = "zxdrfile", .module = zxdrfile_mod },
```

Keep the `zsasa` entry:

```zig
.{ .name = "zsasa", .module = zsasa_mod },
```

- [ ] **Step 3: Remove `zxdrfile` imports from shared library creation**

In the shared library `b.createModule(...)` imports list for `src/c_api.zig`, remove this entry:

```zig
.{ .name = "zxdrfile", .module = zxdrfile_mod },
```

Keep the `zsasa` entry.

- [ ] **Step 4: Remove `zxdrfile` import from CLI executable creation**

In the CLI executable `b.createModule(...)` imports list for `src/main.zig`, remove this entry:

```zig
.{ .name = "zxdrfile", .module = zxdrfile_mod },
```

Keep the `ztraj` and `build_options` entries.

- [ ] **Step 5: Remove `.zxdrfile` dependency from `build.zig.zon`**

Delete this dependency entry:

```zig
.zxdrfile = .{
    .url = "git+https://github.com/N283T/zxdrfile.git?ref=v0.4.0#47d8c42cff9b098febd1990cc50438c26987f85c",
    .hash = "zxdrfile-0.4.0-JIaQ9mIfAgAHB9bW-_p9g6_gfd0tyG3-MmA0LYJKB8rb",
},
```

Leave the `.zsasa` dependency unchanged.

- [ ] **Step 6: Verify no declared zxdrfile dependency remains**

Run:

```bash
rg -n '@import\("zxdrfile"\)|\bzxdrfile\b' src build.zig build.zig.zon
```

Expected: no output.

- [ ] **Step 7: Run focused build checks**

Run:

```bash
zig build
zig build test
```

Expected: both PASS.

- [ ] **Step 8: Commit dependency removal**

Run:

```bash
git add build.zig build.zig.zon
git commit -m "chore: remove external zxdrfile dependency"
```

## Task 5: Preserve and document attribution for integrated source

**Files:**
- Modify: `src/io/xdr.zig`
- Modify: `src/io/xtc_format.zig`
- Modify: `src/io/trr_format.zig`

- [ ] **Step 1: Check source license compatibility**

Run:

```bash
sed -n '1,120p' zig-pkg/zxdrfile-0.4.0-JIaQ9mIfAgAHB9bW-_p9g6_gfd0tyG3-MmA0LYJKB8rb/LICENSE
sed -n '1,120p' LICENSE
```

Expected: The imported source is compatible with ztraj's repository license. If the licenses differ, preserve the upstream license header or add an attribution note before committing.

- [ ] **Step 2: Add short provenance comments to integrated files**

At the top of `src/io/xdr.zig`, before the original module documentation, add:

```zig
//! Low-level XDR trajectory format API integrated from zxdrfile v0.4.0.
//! This module intentionally stays format-native; ztraj Frame conversion lives
//! in `xtc.zig` and `trr.zig`.
```

At the top of `src/io/xtc_format.zig`, before the original module documentation, add:

```zig
//! Low-level XTC reader/writer integrated from zxdrfile v0.4.0.
//! Coordinates are stored in the file's native nanometer units here.
```

At the top of `src/io/trr_format.zig`, before the original module documentation, add:

```zig
//! Low-level TRR reader/writer integrated from zxdrfile v0.4.0.
//! Coordinates, velocities, and forces remain format-native in this module.
```

- [ ] **Step 3: Run formatting**

Run:

```bash
zig fmt src/io/xdr.zig src/io/xtc_format.zig src/io/trr_format.zig
```

Expected: command exits 0.

- [ ] **Step 4: Run focused tests**

Run:

```bash
zig build test
```

Expected: PASS.

- [ ] **Step 5: Commit attribution comments**

Run:

```bash
git add src/io/xdr.zig src/io/xtc_format.zig src/io/trr_format.zig
git commit -m "docs: note integrated xdr format provenance"
```

## Task 6: Final verification and cleanup check

**Files:**
- No required source edits unless verification exposes an issue.

- [ ] **Step 1: Verify git status only contains intentional changes**

Run:

```bash
git status --short
```

Expected: clean, or only pre-existing unrelated untracked files such as `AGENTS.md`, `leap.log`, or `zig-pkg/`. Do not add unrelated untracked files.

- [ ] **Step 2: Verify no external zxdrfile dependency remains in tracked project files**

Run:

```bash
rg -n '@import\("zxdrfile"\)|\bzxdrfile\b' build.zig build.zig.zon src README.md python docs --glob '!docs/superpowers/**'
```

Expected: no output in `build.zig`, `build.zig.zon`, or `src/`. If README/docs mention historical dependency text, update only if it describes current build dependencies incorrectly.

- [ ] **Step 3: Run final Zig verification**

Run:

```bash
zig build
zig build test
```

Expected: both PASS.

- [ ] **Step 4: Run Python smoke tests if the shared library build changed successfully**

Run:

```bash
cd python && pytest tests -x -q
```

Expected: PASS. If this fails because the local Python environment lacks pytest or package setup, record the exact failure in the final handoff and do not claim Python verification passed.

- [ ] **Step 5: Summarize final dependency state**

Run:

```bash
git log --oneline -5
```

Expected: recent commits include the namespace root, vendored low-level modules, wrapper switch, dependency removal, and provenance comments.

## Self-Review

- Spec coverage: Tasks 1-4 implement the integrated layout, public API preservation, and build dependency removal. Task 5 covers provenance and responsibility boundaries. Task 6 covers acceptance verification.
- Placeholder scan: No `TBD`, `TODO`, `FIXME`, or unspecified implementation steps remain.
- Type consistency: The plan consistently uses `xdr.XtcReader`, `xdr.XtcWriter`, `xdr.XtcError`, `xdr.TrrReader`, `xdr.TrrWriter`, and `xdr.TrrError`; these aliases are defined in Task 2 before wrapper usage in Task 3.
