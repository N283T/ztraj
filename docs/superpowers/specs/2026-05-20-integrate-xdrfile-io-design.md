# Integrate XDR trajectory I/O into ztraj

## Context

ztraj currently depends on the external `zxdrfile` Zig package for low-level XTC and TRR parsing/writing. ztraj then wraps those readers and writers in `src/io/xtc.zig` and `src/io/trr.zig` to convert between format-native data and ztraj's internal `types.Frame` representation.

Maintaining `zxdrfile` as a separate Zig package creates duplicate work when Zig changes its language, standard library, or build system APIs. Because Zig is still evolving, each Zig upgrade can require coordinated changes across both repositories before ztraj builds again.

## Goal

Absorb the `zxdrfile` implementation into the ztraj repository so ztraj owns the complete XTC/TRR implementation lifecycle, while preserving a clean boundary between low-level file format handling and ztraj's high-level frame-oriented API.

## Non-goals

- Do not redesign the public high-level ztraj XTC/TRR API in this change.
- Do not change ztraj's internal unit convention: coordinates remain angstroms at the ztraj API boundary.
- Do not merge low-level XDR parsing directly into the high-level `Frame` wrappers.
- Do not remove or rewrite unrelated parsers/writers.

## Proposed layout

Introduce an explicit I/O module root and move the imported XDR implementation behind it:

```text
src/io/root.zig
src/io/xdr.zig
src/io/xtc_format.zig
src/io/trr_format.zig
src/io/xtc.zig
src/io/trr.zig
src/io/dcd.zig
src/io/gro.zig
src/io/pdb.zig
src/io/mmcif.zig
src/io/prmtop.zig
src/io/nc.zig
```

Responsibilities:

- `src/io/xdr.zig`: low-level XDR primitives originally provided by zxdrfile.
- `src/io/xtc_format.zig`: low-level XTC reader/writer types that operate on format-native data, including nanometer coordinates and packed coordinate buffers.
- `src/io/trr_format.zig`: low-level TRR reader/writer types that operate on format-native data, including optional coordinates, velocities, and forces.
- `src/io/xtc.zig`: ztraj high-level XTC wrapper that returns/writes `types.Frame`, converts nm to/from angstroms, and exposes the existing public API.
- `src/io/trr.zig`: ztraj high-level TRR wrapper with the same boundary responsibilities for TRR.
- `src/io/root.zig`: public namespace for all ztraj I/O modules.

`src/root.zig` should expose I/O through:

```zig
pub const io = @import("io/root.zig");
```

## Public API policy

The first integration should keep low-level XDR/XTC/TRR format modules internal or semi-private. The stable public API remains the existing high-level namespace:

- `ztraj.io.xtc.XtcReader`
- `ztraj.io.xtc.XtcWriter`
- `ztraj.io.trr.TrrReader`
- `ztraj.io.trr.TrrWriter`

If another Zig library later needs direct access to format-native XTC/TRR data, ztraj can intentionally expose a documented `ztraj.io.formats` namespace in a follow-up change.

## Build changes

- Remove the `zxdrfile` dependency from `build.zig.zon`.
- Remove `b.dependency("zxdrfile", ...)` from `build.zig`.
- Remove `zxdrfile` imports from ztraj modules and replace them with relative imports to the integrated low-level modules.
- Keep `zsasa` unchanged.
- Ensure the module, shared library, CLI, tests, and docs still compile.

## Migration strategy

1. Copy the current `zxdrfile` source into ztraj under the new low-level module names.
2. Adjust imports to use relative paths inside `src/io/`.
3. Update `src/io/xtc.zig` and `src/io/trr.zig` to import the integrated low-level modules instead of `@import("zxdrfile")`.
4. Add `src/io/root.zig` and update `src/root.zig` to import it.
5. Remove the external dependency from build files.
6. Run focused Zig build and test commands.

## Testing

Minimum checks for the implementation change:

```bash
zig build
zig build test
```

Recommended additional checks when Python bindings are affected:

```bash
cd python && pytest tests -x -q
```

## Risks and mitigations

- **Risk: accidental public API breakage.** Mitigate by keeping `ztraj.io.xtc` and `ztraj.io.trr` names and types stable.
- **Risk: low-level and high-level responsibilities become tangled.** Mitigate by keeping format-native modules separate from `types.Frame` wrappers.
- **Risk: imported code keeps stale zxdrfile naming.** Mitigate by renaming files and imports during integration, while avoiding broad rewrites beyond the dependency boundary.
- **Risk: hidden build dependency remains.** Mitigate by removing the `zxdrfile` entry from both `build.zig.zon` and `build.zig`, then verifying from a clean build cache if needed.

## Acceptance criteria

- ztraj no longer declares `zxdrfile` as an external dependency.
- XTC and TRR reader/writer wrappers still expose the existing high-level ztraj API.
- Low-level XDR/XTC/TRR implementation lives in the ztraj repository under `src/io/`.
- `zig build test` passes.
