"""Tests for pyztraj core API."""

from pathlib import Path

import numpy as np
import pytest

import pyztraj


class TestVersion:
    def test_returns_nonempty_string(self):
        ver = pyztraj.get_version()
        assert isinstance(ver, str)
        assert len(ver) > 0

    def test_semantic_version_format(self):
        ver = pyztraj.get_version()
        parts = ver.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestDistances:
    def test_3_4_5_triangle(self):
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
        pairs = np.array([[0, 1]], dtype=np.uint32)
        result = pyztraj.compute_distances(coords, pairs)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 5.0, atol=1e-5)

    def test_multiple_pairs(self):
        coords = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32
        )
        pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.uint32)
        result = pyztraj.compute_distances(coords, pairs)
        assert result.shape == (3,)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-5)
        np.testing.assert_allclose(result[1], 2.0, atol=1e-5)
        np.testing.assert_allclose(result[2], np.sqrt(5.0), atol=1e-5)

    def test_zero_distance(self):
        coords = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        pairs = np.array([[0, 1]], dtype=np.uint32)
        result = pyztraj.compute_distances(coords, pairs)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-7)


class TestAngles:
    def test_right_angle(self):
        coords = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        triplets = np.array([[0, 1, 2]], dtype=np.uint32)
        result = pyztraj.compute_angles(coords, triplets)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], np.pi / 2, atol=1e-5)

    def test_straight_angle(self):
        coords = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        )
        triplets = np.array([[0, 1, 2]], dtype=np.uint32)
        result = pyztraj.compute_angles(coords, triplets)
        np.testing.assert_allclose(result[0], np.pi, atol=1e-5)


class TestDihedrals:
    def test_90_degree(self):
        # i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,0,1)
        coords = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        quartets = np.array([[0, 1, 2, 3]], dtype=np.uint32)
        result = pyztraj.compute_dihedrals(coords, quartets)
        assert result.shape == (1,)
        np.testing.assert_allclose(abs(result[0]), np.pi / 2, atol=1e-5)


class TestRMSD:
    def test_identical_structures(self):
        coords = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        rmsd = pyztraj.compute_rmsd(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_translated(self):
        ref = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        # QCP centers both structures, so pure translation -> RMSD 0
        shifted = ref + np.array([10.0, 20.0, 30.0], dtype=np.float32)
        rmsd = pyztraj.compute_rmsd(shifted, ref)
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_with_atom_indices(self):
        coords = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        # Using only first 2 atoms
        rmsd = pyztraj.compute_rmsd(coords, coords, atom_indices=np.array([0, 1], dtype=np.uint32))
        assert rmsd == pytest.approx(0.0, abs=1e-10)


class TestRg:
    def test_two_atoms_symmetric(self):
        coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([1.0, 1.0], dtype=np.float64)
        rg = pyztraj.compute_rg(coords, masses)
        assert rg == pytest.approx(1.0, abs=1e-10)

    def test_single_atom(self):
        coords = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        masses = np.array([12.0], dtype=np.float64)
        rg = pyztraj.compute_rg(coords, masses)
        assert rg == pytest.approx(0.0, abs=1e-10)


class TestCenterOfMass:
    def test_equal_masses(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([1.0, 1.0], dtype=np.float64)
        com = pyztraj.compute_center_of_mass(coords, masses)
        np.testing.assert_allclose(com, [1.0, 0.0, 0.0], atol=1e-10)

    def test_unequal_masses(self):
        coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([3.0, 1.0], dtype=np.float64)
        com = pyztraj.compute_center_of_mass(coords, masses)
        np.testing.assert_allclose(com, [1.0, 0.0, 0.0], atol=1e-10)


class TestCenterOfGeometry:
    def test_mean_position(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        cog = pyztraj.compute_center_of_geometry(coords)
        np.testing.assert_allclose(cog, [1.0, 2.0, 3.0], atol=1e-6)


class TestInertia:
    def test_two_atoms_on_x_axis(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        masses = np.array([1.0, 1.0], dtype=np.float64)
        tensor = pyztraj.compute_inertia(coords, masses)
        assert tensor.shape == (3, 3)
        # Ixx = 0 (on x-axis), Iyy = Izz = 2
        np.testing.assert_allclose(tensor[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(tensor[1, 1], 2.0, atol=1e-10)
        np.testing.assert_allclose(tensor[2, 2], 2.0, atol=1e-10)

    def test_principal_moments(self):
        tensor = np.diag([1.0, 2.0, 3.0])
        moments = pyztraj.compute_principal_moments(tensor)
        np.testing.assert_allclose(moments, [1.0, 2.0, 3.0], atol=1e-10)


class TestLoadPDB:
    def test_load_structure(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        assert struct.n_atoms > 0
        assert struct.coords.shape == (struct.n_atoms, 3)
        assert struct.masses.shape == (struct.n_atoms,)
        assert len(struct.atom_names) == struct.n_atoms
        assert len(struct.residue_names) == struct.n_atoms
        assert struct.resids.shape == (struct.n_atoms,)

    def test_atom_names_not_empty(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        assert all(len(name) > 0 for name in struct.atom_names)

    def test_masses_positive(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        assert np.all(struct.masses > 0)

    def test_file_not_found(self):
        with pytest.raises(pyztraj.ZtrajError, match="File I/O"):
            pyztraj.load_pdb("/nonexistent/path.pdb")


class TestXtcReader:
    def test_read_frames(self, pdb_path: Path, xtc_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        frame_count = 0
        with pyztraj.open_xtc(xtc_path, struct.n_atoms) as reader:
            for frame in reader:
                assert frame.coords.shape == (struct.n_atoms, 3)
                frame_count += 1
                if frame_count >= 5:
                    break
        assert frame_count == 5

    def test_atom_count_mismatch(self, xtc_path: Path):
        with pytest.raises(ValueError, match="atoms"):
            with pyztraj.open_xtc(xtc_path, 999) as _reader:
                pass
