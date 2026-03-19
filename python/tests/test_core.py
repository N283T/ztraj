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
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
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
        coords = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        triplets = np.array([[0, 1, 2]], dtype=np.uint32)
        result = pyztraj.compute_angles(coords, triplets)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], np.pi / 2, atol=1e-5)

    def test_straight_angle(self):
        coords = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
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
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        rmsd = pyztraj.compute_rmsd(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_translated(self):
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        # QCP centers both structures, so pure translation -> RMSD 0
        shifted = ref + np.array([10.0, 20.0, 30.0], dtype=np.float32)
        rmsd = pyztraj.compute_rmsd(shifted, ref)
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_with_atom_indices(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
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


class TestRMSF:
    def test_static_structure(self):
        # All frames identical -> RMSF should be 0
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        frames = [coords, coords, coords]
        result = pyztraj.compute_rmsf(frames)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)

    def test_oscillating_atom(self):
        # Atom 0 oscillates: x = -1, +1, -1 -> mean = -1/3, RMSF > 0
        f1 = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        f2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        f3 = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        result = pyztraj.compute_rmsf([f1, f2, f3])
        assert result[0] > 0.0  # Atom 0 fluctuates
        np.testing.assert_allclose(result[1], 0.0, atol=1e-10)  # Atom 1 static

    def test_with_atom_indices(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        frames = [coords, coords]
        result = pyztraj.compute_rmsf(frames, atom_indices=np.array([0, 2], dtype=np.uint32))
        assert result.shape == (2,)  # Only 2 atoms selected

    def test_empty_frames_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pyztraj.compute_rmsf([])


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


class TestRDF:
    def test_uniform_distribution(self):
        # Two selections at known positions — just verify shape and no crash
        sel1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        sel2 = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        r, g_r = pyztraj.compute_rdf(sel1, sel2, box_volume=1000.0, r_max=5.0, n_bins=50)
        assert r.shape == (50,)
        assert g_r.shape == (50,)
        assert np.all(r >= 0)

    def test_invalid_params(self):
        sel = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        with pytest.raises(pyztraj.ZtrajError):
            pyztraj.compute_rdf(sel, sel, box_volume=-1.0)


class TestHBonds:
    def test_detect_from_pdb(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        hbonds = pyztraj.detect_hbonds(struct, struct.coords)
        assert isinstance(hbonds, list)
        if len(hbonds) > 0:
            hb = hbonds[0]
            assert isinstance(hb, pyztraj.HBond)
            assert hb.distance > 0
            assert hb.angle > 0

    def test_returns_expected_fields(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        hbonds = pyztraj.detect_hbonds(struct, struct.coords)
        for hb in hbonds:
            assert hb.donor >= 0
            assert hb.hydrogen >= 0
            assert hb.acceptor >= 0
            assert hb.distance <= 2.5  # default cutoff
            assert hb.angle >= 120.0  # default cutoff


class TestContacts:
    def test_detect_from_pdb(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        contacts = pyztraj.compute_contacts(struct, struct.coords, cutoff=8.0)
        assert isinstance(contacts, list)
        assert len(contacts) > 0  # 38 residue protein should have many contacts
        c = contacts[0]
        assert isinstance(c, pyztraj.Contact)
        assert c.residue_i < c.residue_j
        assert c.distance > 0
        assert c.distance < 8.0

    def test_schemes(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        for scheme in ["closest", "ca", "closest_heavy"]:
            contacts = pyztraj.compute_contacts(struct, struct.coords, cutoff=8.0, scheme=scheme)
            assert isinstance(contacts, list)

    def test_invalid_scheme(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        with pytest.raises(ValueError, match="Unknown scheme"):
            pyztraj.compute_contacts(struct, struct.coords, scheme="invalid")


class TestAnalyzeAll:
    def test_single_frame(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        result = pyztraj.analyze_all(struct, [struct.coords])
        assert result["n_frames"] == 1
        assert result["n_atoms"] == struct.n_atoms
        assert result["rmsd"].shape == (1,)
        assert result["rmsd"][0] == pytest.approx(0.0, abs=1e-6)  # self vs self
        assert result["rg"].shape == (1,)
        assert result["rg"][0] > 0
        assert result["sasa"].shape == (1,)
        assert result["sasa"][0] > 0
        assert result["center_of_mass"].shape == (1, 3)
        assert result["rmsf"].shape == (struct.n_atoms,)
        assert result["n_hbonds"].shape == (1,)
        assert result["n_contacts"].shape == (1,)
        expected_keys = {
            "n_frames",
            "n_atoms",
            "rmsd",
            "rmsf",
            "rg",
            "sasa",
            "center_of_mass",
            "n_hbonds",
            "n_contacts",
        }
        assert set(result.keys()) == expected_keys


class TestSASA:
    def test_compute_from_pdb(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        result = pyztraj.compute_sasa(struct, struct.coords)
        assert isinstance(result, pyztraj.SasaResult)
        assert result.total_area > 0
        assert result.atom_areas.shape == (struct.n_atoms,)
        assert np.all(result.atom_areas >= 0)

    def test_total_equals_sum_of_atoms(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        result = pyztraj.compute_sasa(struct, struct.coords, n_points=100)
        np.testing.assert_allclose(result.total_area, result.atom_areas.sum(), rtol=1e-6)

    def test_custom_parameters(self, pdb_path: Path):
        struct = pyztraj.load_pdb(pdb_path)
        r1 = pyztraj.compute_sasa(struct, struct.coords, n_points=100, probe_radius=1.4)
        r2 = pyztraj.compute_sasa(struct, struct.coords, n_points=100, probe_radius=2.0)
        # Larger probe should give larger SASA
        assert r2.total_area > r1.total_area
