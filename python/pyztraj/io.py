"""I/O functions: structure loading (PDB, GRO, mmCIF) and trajectory streaming (XTC, TRR, DCD)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pyztraj._ffi import get_ffi, get_lib
from pyztraj._helpers import ZtrajError, _check, _load_topology_handle, _ptr_f32, _ptr_f64, _ptr_i32

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Structure:
    """A loaded molecular structure with coordinates and topology."""

    coords: NDArray[np.float32]  # (n_atoms, 3)
    masses: NDArray[np.float64]  # (n_atoms,)
    atom_names: list[str]  # length n_atoms
    residue_names: list[str]  # length n_atoms (per-atom)
    resids: NDArray[np.int32]  # (n_atoms,) per-atom residue IDs
    n_atoms: int
    _path: str = ""  # internal: path for re-loading topology in analysis functions
    _loader: str = "ztraj_load_pdb"  # internal: C API function name for re-loading


def _load_structure(load_fn, path: str | Path, label: str) -> Structure:
    """Internal helper: load a structure file via the given C API function."""
    ffi = get_ffi()
    lib = get_lib()

    path_bytes = str(path).encode("utf-8")
    handle_ptr = ffi.new("void**")

    _check(load_fn(path_bytes, handle_ptr), f"{label}({path})")
    handle = handle_ptr[0]
    if handle == ffi.NULL:
        raise ZtrajError(f"{label}({path}): returned success but handle is null")

    try:
        n_atoms = lib.ztraj_get_n_atoms(handle)

        # Coordinates
        x = np.empty(n_atoms, dtype=np.float32)
        y = np.empty(n_atoms, dtype=np.float32)
        z = np.empty(n_atoms, dtype=np.float32)
        _check(lib.ztraj_get_coords(handle, _ptr_f32(x), _ptr_f32(y), _ptr_f32(z)))
        coords = np.column_stack([x, y, z])

        # Masses
        masses = np.empty(n_atoms, dtype=np.float64)
        _check(lib.ztraj_get_masses(handle, _ptr_f64(masses)))

        # Atom names (4 bytes each)
        name_buf = ffi.new(f"char[{n_atoms * 4}]")
        _check(lib.ztraj_get_atom_names(handle, name_buf))
        atom_names = [
            ffi.string(name_buf + i * 4, 4).decode("utf-8").strip() for i in range(n_atoms)
        ]

        # Residue names (5 bytes each, per-atom)
        res_buf = ffi.new(f"char[{n_atoms * 5}]")
        _check(lib.ztraj_get_residue_names(handle, res_buf))
        residue_names = [
            ffi.string(res_buf + i * 5, 5).decode("utf-8").strip() for i in range(n_atoms)
        ]

        # Residue IDs
        resids = np.empty(n_atoms, dtype=np.int32)
        _check(lib.ztraj_get_resids(handle, _ptr_i32(resids)))

    finally:
        lib.ztraj_free_structure(handle)

    return Structure(
        coords=coords,
        masses=masses,
        atom_names=atom_names,
        residue_names=residue_names,
        resids=resids,
        n_atoms=n_atoms,
        _path=str(path),
        _loader=label.replace("load_", "ztraj_load_"),
    )


def load_pdb(path: str | Path) -> Structure:
    """Load a PDB file and return a Structure with coordinates and topology."""
    return _load_structure(get_lib().ztraj_load_pdb, path, "load_pdb")


def load_gro(path: str | Path) -> Structure:
    """Load a GRO (GROMACS) file and return a Structure with coordinates and topology."""
    return _load_structure(get_lib().ztraj_load_gro, path, "load_gro")


def load_mmcif(path: str | Path) -> Structure:
    """Load an mmCIF file and return a Structure with coordinates and topology."""
    return _load_structure(get_lib().ztraj_load_mmcif, path, "load_mmcif")


class XtcReader:
    """Streaming XTC trajectory reader (context manager).

    Usage::

        with pyztraj.open_xtc("traj.xtc", n_atoms) as reader:
            for frame in reader:
                # frame.coords is (n_atoms, 3) float32
                print(frame.time, frame.coords.shape)
    """

    @dataclass
    class Frame:
        """A single trajectory frame."""

        coords: NDArray[np.float32]  # (n_atoms, 3)
        time: float  # picoseconds
        step: int

    def __init__(self, path: str | Path, n_atoms: int) -> None:
        self._ffi = get_ffi()
        self._lib = get_lib()
        self._path = str(path).encode("utf-8")
        self._expected_n_atoms = n_atoms
        self._handle = None
        self._n_atoms = 0

    def __enter__(self) -> XtcReader:
        n_atoms_out = self._ffi.new("size_t*")
        handle_ptr = self._ffi.new("void**")
        _check(self._lib.ztraj_open_xtc(self._path, n_atoms_out, handle_ptr), "open_xtc")
        self._handle = handle_ptr[0]
        self._n_atoms = n_atoms_out[0]

        if self._n_atoms != self._expected_n_atoms:
            self._lib.ztraj_close_xtc(self._handle)
            self._handle = None
            msg = (
                f"XTC has {self._n_atoms} atoms but expected {self._expected_n_atoms} "
                f"(from topology)"
            )
            raise ValueError(msg)

        return self

    def __exit__(self, *args) -> None:
        if self._handle is not None:
            self._lib.ztraj_close_xtc(self._handle)
            self._handle = None

    def __iter__(self):
        return self

    def __next__(self) -> XtcReader.Frame:
        if self._handle is None:
            raise StopIteration

        x = np.empty(self._n_atoms, dtype=np.float32)
        y = np.empty(self._n_atoms, dtype=np.float32)
        z = np.empty(self._n_atoms, dtype=np.float32)
        time_out = self._ffi.new("float*")
        step_out = self._ffi.new("int32_t*")

        rc = self._lib.ztraj_read_xtc_frame(
            self._handle, _ptr_f32(x), _ptr_f32(y), _ptr_f32(z), time_out, step_out
        )

        if rc == self._lib.ZTRAJ_ERROR_EOF:
            raise StopIteration

        _check(rc, "read_xtc_frame")

        coords = np.column_stack([x, y, z])
        return XtcReader.Frame(coords=coords, time=float(time_out[0]), step=int(step_out[0]))


def open_xtc(path: str | Path, n_atoms: int) -> XtcReader:
    """Open an XTC trajectory file for streaming frame-by-frame reading.

    Args:
        path: Path to the XTC file.
        n_atoms: Expected number of atoms (from topology).

    Returns:
        XtcReader context manager yielding Frame objects.
    """
    return XtcReader(path, n_atoms)


class _TrajectoryReader:
    """Generic streaming trajectory reader (context manager).

    Subclasses specify the C API function names for open/read/close.
    """

    @dataclass
    class Frame:
        """A single trajectory frame."""

        coords: NDArray[np.float32]  # (n_atoms, 3)
        time: float  # picoseconds
        step: int

    def __init__(
        self,
        path: str | Path,
        n_atoms: int,
        open_fn_name: str,
        read_fn_name: str,
        close_fn_name: str,
        label: str,
    ) -> None:
        self._ffi = get_ffi()
        self._lib = get_lib()
        self._path = str(path).encode("utf-8")
        self._expected_n_atoms = n_atoms
        self._handle = None
        self._n_atoms = 0
        self._open_fn = getattr(self._lib, open_fn_name)
        self._read_fn = getattr(self._lib, read_fn_name)
        self._close_fn = getattr(self._lib, close_fn_name)
        self._label = label

    def __enter__(self):
        n_atoms_out = self._ffi.new("size_t*")
        handle_ptr = self._ffi.new("void**")
        _check(self._open_fn(self._path, n_atoms_out, handle_ptr), self._label)
        self._handle = handle_ptr[0]
        if self._handle == self._ffi.NULL:
            raise ZtrajError(f"{self._label}: returned success but handle is null")
        self._n_atoms = n_atoms_out[0]

        if self._n_atoms != self._expected_n_atoms:
            self._close_fn(self._handle)
            self._handle = None
            msg = (
                f"{self._label} has {self._n_atoms} atoms but expected "
                f"{self._expected_n_atoms} (from topology)"
            )
            raise ValueError(msg)

        return self

    def __exit__(self, *args) -> None:
        if self._handle is not None:
            self._close_fn(self._handle)
            self._handle = None

    def __iter__(self):
        return self

    def __next__(self) -> _TrajectoryReader.Frame:
        if self._handle is None:
            raise StopIteration

        x = np.empty(self._n_atoms, dtype=np.float32)
        y = np.empty(self._n_atoms, dtype=np.float32)
        z = np.empty(self._n_atoms, dtype=np.float32)
        time_out = self._ffi.new("float*")
        step_out = self._ffi.new("int32_t*")

        rc = self._read_fn(self._handle, _ptr_f32(x), _ptr_f32(y), _ptr_f32(z), time_out, step_out)

        if rc == self._lib.ZTRAJ_ERROR_EOF:
            raise StopIteration

        if rc != 0:
            # Close handle on read error to prevent broken-stream iteration
            self._close_fn(self._handle)
            self._handle = None
        _check(rc, f"read_{self._label}_frame")

        coords = np.column_stack([x, y, z])
        return _TrajectoryReader.Frame(
            coords=coords, time=float(time_out[0]), step=int(step_out[0])
        )


class TrrReader(_TrajectoryReader):
    """Streaming TRR trajectory reader (context manager).

    Usage::

        with pyztraj.open_trr("traj.trr", n_atoms) as reader:
            for frame in reader:
                print(frame.time, frame.coords.shape)
    """

    def __init__(self, path: str | Path, n_atoms: int) -> None:
        super().__init__(
            path, n_atoms, "ztraj_open_trr", "ztraj_read_trr_frame", "ztraj_close_trr", "trr"
        )


class DcdReader(_TrajectoryReader):
    """Streaming DCD trajectory reader (context manager).

    Usage::

        with pyztraj.open_dcd("traj.dcd", n_atoms) as reader:
            for frame in reader:
                print(frame.time, frame.coords.shape)
    """

    def __init__(self, path: str | Path, n_atoms: int) -> None:
        super().__init__(
            path, n_atoms, "ztraj_open_dcd", "ztraj_read_dcd_frame", "ztraj_close_dcd", "dcd"
        )


def open_trr(path: str | Path, n_atoms: int) -> TrrReader:
    """Open a TRR trajectory file for streaming frame-by-frame reading."""
    return TrrReader(path, n_atoms)


def open_dcd(path: str | Path, n_atoms: int) -> DcdReader:
    """Open a DCD trajectory file for streaming frame-by-frame reading."""
    return DcdReader(path, n_atoms)


# ============================================================================
# Structure writers
# ============================================================================


def write_pdb(
    path: str | Path, structure: Structure, coords: NDArray[np.float32] | None = None
) -> None:
    """Write structure as PDB file.

    Args:
        path: Output file path.
        structure: Loaded structure (topology source).
        coords: Coordinates (n_atoms, 3). If None, uses structure.coords.
    """
    _write_structure("ztraj_write_pdb", path, structure, coords)


def write_gro(
    path: str | Path, structure: Structure, coords: NDArray[np.float32] | None = None
) -> None:
    """Write structure as GRO file.

    Args:
        path: Output file path.
        structure: Loaded structure (topology source).
        coords: Coordinates (n_atoms, 3). If None, uses structure.coords.
    """
    _write_structure("ztraj_write_gro", path, structure, coords)


def _write_structure(
    fn_name: str,
    path: str | Path,
    structure: Structure,
    coords: NDArray[np.float32] | None = None,
) -> None:
    """Internal helper for writing structure files."""
    ffi = get_ffi()
    lib = get_lib()

    if coords is None:
        coords = structure.coords
    coords = np.ascontiguousarray(coords, dtype=np.float32)

    x, y, z = coords[:, 0].copy(), coords[:, 1].copy(), coords[:, 2].copy()
    n_atoms = len(x)

    handle = _load_topology_handle(structure, lib, ffi, fn_name)

    try:
        path_bytes = str(path).encode("utf-8")
        write_fn = getattr(lib, fn_name)
        _check(
            write_fn(handle, _ptr_f32(x), _ptr_f32(y), _ptr_f32(z), n_atoms, path_bytes),
            fn_name,
        )
    finally:
        lib.ztraj_free_structure(handle)


# ============================================================================
# Trajectory writers
# ============================================================================


class _TrajectoryWriter:
    """Generic streaming trajectory writer (context manager).

    Subclasses specify the C API function names for open/write/close.
    """

    def __init__(
        self,
        path: str | Path,
        n_atoms: int,
        open_fn_name: str,
        write_fn_name: str,
        close_fn_name: str,
        label: str,
    ) -> None:
        self._ffi = get_ffi()
        self._lib = get_lib()
        self._path = str(path).encode("utf-8")
        self._n_atoms = n_atoms
        self._handle = None
        self._open_fn = getattr(self._lib, open_fn_name)
        self._write_fn = getattr(self._lib, write_fn_name)
        self._close_fn = getattr(self._lib, close_fn_name)
        self._label = label

    def __enter__(self):
        handle_ptr = self._ffi.new("void**")
        _check(self._open_fn(self._path, self._n_atoms, handle_ptr), self._label)
        self._handle = handle_ptr[0]
        if self._handle == self._ffi.NULL:
            raise ZtrajError(f"{self._label}: returned success but handle is null")
        return self

    def __exit__(self, *args) -> None:
        if self._handle is not None:
            self._close_fn(self._handle)
            self._handle = None

    def write_frame(
        self,
        coords: NDArray[np.float32],
        time: float = 0.0,
        step: int = 0,
    ) -> None:
        """Write a single frame.

        Args:
            coords: Atom coordinates (n_atoms, 3) in Angstroms.
            time: Simulation time in picoseconds.
            step: Step counter.
        """
        if self._handle is None:
            raise ZtrajError(f"{self._label}: writer is not open")

        coords = np.ascontiguousarray(coords, dtype=np.float32)
        x, y, z = coords[:, 0].copy(), coords[:, 1].copy(), coords[:, 2].copy()

        _check(
            self._write_fn(
                self._handle,
                _ptr_f32(x),
                _ptr_f32(y),
                _ptr_f32(z),
                self._n_atoms,
                time,
                step,
            ),
            f"{self._label}/write_frame",
        )


class XtcWriter(_TrajectoryWriter):
    """Streaming XTC trajectory writer (context manager).

    Usage::

        with pyztraj.XtcWriter("output.xtc", n_atoms) as writer:
            writer.write_frame(coords, time=0.0, step=0)
    """

    def __init__(self, path: str | Path, n_atoms: int) -> None:
        super().__init__(
            path,
            n_atoms,
            "ztraj_open_xtc_writer",
            "ztraj_write_xtc_frame",
            "ztraj_close_xtc_writer",
            "xtc_writer",
        )


class TrrWriter(_TrajectoryWriter):
    """Streaming TRR trajectory writer (context manager).

    Usage::

        with pyztraj.TrrWriter("output.trr", n_atoms) as writer:
            writer.write_frame(coords, time=0.0, step=0)
    """

    def __init__(self, path: str | Path, n_atoms: int) -> None:
        super().__init__(
            path,
            n_atoms,
            "ztraj_open_trr_writer",
            "ztraj_write_trr_frame",
            "ztraj_close_trr_writer",
            "trr_writer",
        )
