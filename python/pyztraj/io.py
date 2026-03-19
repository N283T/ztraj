"""I/O functions: PDB loading and XTC streaming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pyztraj._ffi import get_ffi, get_lib
from pyztraj._helpers import ZtrajError, _check, _ptr_f32, _ptr_f64, _ptr_i32

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
    _pdb_path: str = ""  # internal: path for re-loading topology in analysis functions


def load_pdb(path: str | Path) -> Structure:
    """Load a PDB file and return a Structure with coordinates and topology.

    Args:
        path: Path to the PDB file.

    Returns:
        Structure with coords (Angstroms), masses, atom/residue names.
    """
    ffi = get_ffi()
    lib = get_lib()

    path_bytes = str(path).encode("utf-8")
    handle_ptr = ffi.new("void**")

    _check(lib.ztraj_load_pdb(path_bytes, handle_ptr), f"load_pdb({path})")
    handle = handle_ptr[0]
    if handle == ffi.NULL:
        raise ZtrajError(f"load_pdb({path}): returned success but handle is null")

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
        _pdb_path=str(path),
    )


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
