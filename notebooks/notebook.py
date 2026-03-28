"""Interactive Molecular Dynamics Trajectory Analysis Dashboard.

A comprehensive interactive notebook for analyzing MD trajectories using pyztraj.

Run from the notebooks/ directory:
    cd notebooks && uv run marimo edit notebook.py
    cd notebooks && uv run marimo run notebook.py
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full", app_title="pyztraj — Trajectory Analysis")


@app.cell
def header():
    import marimo as mo

    mo.md(
        """
        # pyztraj
        **Interactive Molecular Dynamics Trajectory Analysis**
        """
    )
    return (mo,)


@app.cell
def load_data(mo):
    from pathlib import Path

    import altair as alt
    import numpy as np
    import pandas as pd
    import pyztraj

    _data_dir = Path("../validation/test_data")
    pdb_path = _data_dir.joinpath("3tvj_I.pdb")
    xtc_path = _data_dir.joinpath("3tvj_I_R1.xtc")

    structure = pyztraj.load_pdb(str(pdb_path))

    frames = []
    times = []
    with pyztraj.open_xtc(str(xtc_path), structure.n_atoms) as _reader:
        for _frame in _reader:
            frames.append(_frame.coords.copy())
            times.append(_frame.time)

    times = np.array(times)
    n_frames = len(frames)

    mo.md(
        f"""
        **Files loaded:**
        - Topology: `{pdb_path.name}` ({structure.n_atoms} atoms)
        - Trajectory: `{xtc_path.name}` ({n_frames} frames)
        """
    ).callout(kind="success")
    return alt, frames, n_frames, np, pd, pyztraj, structure, times


@app.cell
def selections(structure):
    backbone_idx = structure.select("backbone")
    ca_idx = structure.select("name CA")

    _residue_ids = structure.resids[ca_idx]
    _residue_names = [structure.residue_names[i] for i in ca_idx]
    residue_labels = [f"{name}{rid}" for name, rid in zip(_residue_names, _residue_ids)]
    n_residues = len(ca_idx)
    return backbone_idx, ca_idx, n_residues, residue_labels


@app.cell
def combined_analysis(frames, pyztraj, structure):
    results = pyztraj.analyze_all(structure, frames, ref_frame=0)
    return (results,)


@app.cell
def system_overview(mo, n_frames, n_residues, np, results, structure, times):
    _dt_ns = (times[-1] - times[0]) / 1000.0
    _dt_step = times[1] - times[0]

    _stats = mo.hstack(
        [
            mo.stat(value=structure.n_atoms, label="Atoms"),
            mo.stat(value=n_residues, label="Residues"),
            mo.stat(value=n_frames, label="Frames"),
            mo.stat(value=f"{_dt_ns:.0f} ns", label="Sim. Time"),
            mo.stat(value=f"{_dt_step:.0f} ps", label="Timestep"),
            mo.stat(value=f"{np.mean(results['rmsd']):.2f} Å", label="Mean RMSD"),
            mo.stat(value=f"{np.mean(results['rg']):.1f} Å", label="Mean Rg"),
            mo.stat(value=f"{np.mean(results['n_hbonds']):.0f}", label="Mean H-bonds"),
        ],
        justify="center",
        gap=1,
    )

    mo.vstack([mo.md("## System Overview"), _stats])
    return


@app.cell
def rmsd_section(mo):
    mo.md("""
    ## RMSD
    """)
    return


@app.cell
def rmsd_controls(mo):
    rmsd_selection = mo.ui.dropdown(
        options={"Backbone": "backbone", "CA only": "ca", "All atoms": "all"},
        value="Backbone",
        label="Atom selection",
    )
    rmsd_selection
    return (rmsd_selection,)


@app.cell
def rmsd_plot(
    alt,
    backbone_idx,
    ca_idx,
    frames,
    mo,
    np,
    pd,
    pyztraj,
    rmsd_selection,
    times,
):
    _sel_map = {"backbone": backbone_idx, "ca": ca_idx, "all": None}
    _idx = _sel_map[rmsd_selection.value]
    _ref = frames[0]
    _rmsd = np.array([pyztraj.compute_rmsd(f, _ref, atom_indices=_idx) for f in frames])
    _t = times / 1000.0

    _df = pd.DataFrame({"Time (ns)": _t, "RMSD (Å)": _rmsd})
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X("Time (ns):Q"),
            y=alt.Y("RMSD (Å):Q", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("RMSD (Å)", format=".3f"),
            ],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def rmsf_section(mo):
    mo.md("""
    ## RMSF (Cα)
    """)
    return


@app.cell
def rmsf_plot(alt, ca_idx, frames, mo, pd, pyztraj, residue_labels, structure):
    _rmsf = pyztraj.compute_rmsf(frames, atom_indices=ca_idx)
    _resids = [int(structure.resids[i]) for i in ca_idx]
    _df = pd.DataFrame(
        {"Residue ID": _resids, "Residue": residue_labels, "RMSF (Å)": _rmsf}
    )
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line(point=alt.OverlayMarkDef(size=20))
        .encode(
            x=alt.X("Residue ID:Q", title="Residue"),
            y=alt.Y("RMSF (Å):Q"),
            tooltip=["Residue", alt.Tooltip("RMSF (Å)", format=".3f")],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def rg_section(mo):
    mo.md("""
    ## Radius of Gyration
    """)
    return


@app.cell
def rg_plot(alt, mo, pd, results, times):
    _t = times / 1000.0
    _df = pd.DataFrame({"Time (ns)": _t, "Rg (Å)": results["rg"]})
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y=alt.Y("Rg (Å):Q", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("Rg (Å)", format=".2f"),
            ],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def sasa_section(mo):
    mo.md("""
    ## SASA — Solvent Accessible Surface Area
    """)
    return


@app.cell
def sasa_timeseries(alt, mo, pd, results, times):
    _t = times / 1000.0
    _df = pd.DataFrame({"Time (ns)": _t, "SASA (Å²)": results["sasa"]})
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y=alt.Y("SASA (Å²):Q", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("SASA (Å²)", format=".1f"),
            ],
        )
        .properties(width="container", height=300)
        .interactive()
    )
    return


@app.cell
def sasa_precompute(ca_idx, frames, np, pyztraj, residue_labels, structure):
    # Pre-compute per-residue SASA for all frames
    _resids = structure.resids
    _unique_resids = _resids[ca_idx]
    _n_res = len(residue_labels)
    sasa_per_res_all = np.zeros((len(frames), _n_res))
    for _fi, _f in enumerate(frames):
        _result = pyztraj.compute_sasa(structure, _f)
        for _ri, _rid in enumerate(_unique_resids):
            sasa_per_res_all[_fi, _ri] = _result.atom_areas[_resids == _rid].sum()
    return (sasa_per_res_all,)


@app.cell
def sasa_controls(mo, n_frames):
    sasa_frame_slider = mo.ui.slider(
        start=0,
        stop=n_frames - 1,
        step=1,
        value=0,
        label="Per-residue SASA frame",
        show_value=True,
    )
    sasa_frame_slider
    return (sasa_frame_slider,)


@app.cell
def sasa_per_residue(
    alt,
    mo,
    np,
    pd,
    residue_labels,
    sasa_frame_slider,
    sasa_per_res_all,
):
    _fi = sasa_frame_slider.value
    _y_max = float(np.max(sasa_per_res_all))
    _df = pd.DataFrame(
        {
            "Residue": residue_labels,
            "SASA (Å²)": sasa_per_res_all[_fi],
            "index": range(len(residue_labels)),
        }
    )
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_bar(color="coral")
        .encode(
            x=alt.X(
                "Residue:N", sort=alt.SortField("index"), axis=alt.Axis(labelAngle=-45)
            ),
            y=alt.Y("SASA (Å²):Q", scale=alt.Scale(domain=[0, _y_max])),
            tooltip=["Residue", alt.Tooltip("SASA (Å²)", format=".1f")],
        )
        .properties(width="container", height=300)
    )
    return


@app.cell
def hbond_section(mo):
    mo.md("""
    ## Hydrogen Bonds
    """)
    return


@app.cell
def hbond_controls(mo):
    hbond_dist = mo.ui.slider(
        debounce=True,
        start=1.5,
        stop=3.5,
        step=0.1,
        value=2.5,
        label="Distance cutoff (Å)",
        show_value=True,
    )
    hbond_angle = mo.ui.slider(
        debounce=True,
        start=90.0,
        stop=170.0,
        step=5.0,
        value=120.0,
        label="Angle cutoff (°)",
        show_value=True,
    )
    mo.hstack([hbond_dist, hbond_angle])
    return hbond_angle, hbond_dist


@app.cell
def hbond_plot(
    alt,
    frames,
    hbond_angle,
    hbond_dist,
    mo,
    pd,
    pyztraj,
    structure,
    times,
):
    _counts = [
        len(
            pyztraj.detect_hbonds(
                structure,
                f,
                dist_cutoff=hbond_dist.value,
                angle_cutoff=hbond_angle.value,
            )
        )
        for f in frames
    ]
    _t = times / 1000.0
    _df = pd.DataFrame({"Time (ns)": _t, "H-bonds": _counts})
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y="H-bonds:Q",
            tooltip=[alt.Tooltip("Time (ns)", format=".2f"), "H-bonds"],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def contact_section(mo):
    mo.md("""
    ## Contact Map
    """)
    return


@app.cell
def contact_controls(mo):
    contact_cutoff = mo.ui.slider(
        debounce=True,
        start=3.0,
        stop=8.0,
        step=0.5,
        value=4.5,
        label="Cutoff (Å)",
        show_value=True,
    )
    contact_scheme = mo.ui.dropdown(
        options={"Closest heavy": "closest_heavy", "CA": "ca", "Closest": "closest"},
        value="Closest heavy",
        label="Scheme",
    )
    mo.hstack([contact_cutoff, contact_scheme])
    return contact_cutoff, contact_scheme


@app.cell
def contact_plot(
    alt,
    contact_cutoff,
    contact_scheme,
    frames,
    n_residues,
    np,
    pd,
    pyztraj,
    residue_labels,
    structure,
):
    # Count contact frequency across all frames
    _freq = np.zeros((n_residues, n_residues))
    _resid_to_idx = {
        rid: i for i, rid in enumerate(structure.resids[structure.select("name CA")])
    }
    for _f in frames:
        _contacts = pyztraj.compute_contacts(
            structure,
            _f,
            cutoff=contact_cutoff.value,
            scheme=contact_scheme.value,
        )
        for _c in _contacts:
            _i = _resid_to_idx.get(_c.residue_i)
            _j = _resid_to_idx.get(_c.residue_j)
            if _i is not None and _j is not None:
                _freq[_i, _j] += 1
                _freq[_j, _i] += 1
    _freq /= len(frames)

    _rows = []
    for _i in range(n_residues):
        for _j in range(n_residues):
            _rows.append(
                {
                    "Residue i": residue_labels[_i],
                    "Residue j": residue_labels[_j],
                    "Frequency": round(_freq[_i, _j], 3),
                    "idx_i": _i,
                    "idx_j": _j,
                }
            )

    _df = pd.DataFrame(_rows)
    _chart = (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X("Residue i:N", sort=alt.SortField("idx_i")),
            y=alt.Y("Residue j:N", sort=alt.SortField("idx_j")),
            color=alt.Color(
                "Frequency:Q",
                scale=alt.Scale(scheme="reds", domain=[0, 1]),
                legend=alt.Legend(title="Frequency"),
            ),
            tooltip=["Residue i", "Residue j", alt.Tooltip("Frequency", format=".2f")],
        )
        .properties(width=500, height=500, title="Contact Frequency Map")
    )
    _chart
    return


@app.cell
def contact_frame_section(mo):
    mo.md("""
    ## Contact Map (Single Frame)
    """)
    return


@app.cell
def contact_frame_controls(mo, n_frames):
    contact_single_frame = mo.ui.slider(
        debounce=False,
        start=0,
        stop=n_frames - 1,
        step=1,
        value=0,
        label="Frame",
        show_value=True,
    )
    contact_single_cutoff = mo.ui.slider(
        debounce=True,
        start=3.0,
        stop=8.0,
        step=0.5,
        value=4.5,
        label="Cutoff (Å)",
        show_value=True,
    )
    contact_single_scheme = mo.ui.dropdown(
        options={"Closest heavy": "closest_heavy", "CA": "ca", "Closest": "closest"},
        value="Closest heavy",
        label="Scheme",
    )
    mo.hstack([contact_single_frame, contact_single_cutoff, contact_single_scheme])
    return contact_single_cutoff, contact_single_frame, contact_single_scheme


@app.cell(hide_code=True)
def contact_frame_plot(
    alt,
    contact_single_cutoff,
    contact_single_frame,
    contact_single_scheme,
    frames,
    pd,
    pyztraj,
    residue_labels,
    structure,
):
    _contacts = pyztraj.compute_contacts(
        structure,
        frames[contact_single_frame.value],
        cutoff=contact_single_cutoff.value,
        scheme=contact_single_scheme.value,
    )
    _resid_to_idx = {
        rid: i for i, rid in enumerate(structure.resids[structure.select("name CA")])
    }
    _rows = []
    for _c in _contacts:
        _i = _resid_to_idx.get(_c.residue_i)
        _j = _resid_to_idx.get(_c.residue_j)
        if _i is not None and _j is not None:
            _rows.append(
                {
                    "Residue i": residue_labels[_i],
                    "Residue j": residue_labels[_j],
                    "Distance (Å)": round(_c.distance, 2),
                    "idx_i": _i,
                    "idx_j": _j,
                }
            )
            _rows.append(
                {
                    "Residue i": residue_labels[_j],
                    "Residue j": residue_labels[_i],
                    "Distance (Å)": round(_c.distance, 2),
                    "idx_i": _j,
                    "idx_j": _i,
                }
            )

    _df = pd.DataFrame(_rows)
    _chart = (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X("Residue i:N", sort=alt.SortField("idx_i")),
            y=alt.Y("Residue j:N", sort=alt.SortField("idx_j")),
            color=alt.Color(
                "Distance (Å):Q", scale=alt.Scale(scheme="viridis", reverse=True)
            ),
            tooltip=["Residue i", "Residue j", "Distance (Å)"],
        )
        .properties(
            width=500,
            height=500,
            title=f"Contact Map (frame {contact_single_frame.value})",
        )
    )
    _chart
    return


@app.cell
def native_q_section(mo):
    mo.md("""
    ## Native Contacts — Fraction Q
    """)
    return


@app.cell
def native_q_controls(mo):
    q_cutoff = mo.ui.slider(
        debounce=True,
        start=3.0,
        stop=8.0,
        step=0.5,
        value=4.5,
        label="Contact cutoff (Å)",
        show_value=True,
    )
    q_cutoff
    return (q_cutoff,)


@app.cell
def native_q_plot(alt, ca_idx, frames, mo, np, pd, pyztraj, q_cutoff, times):
    _ref = frames[0]
    _q = np.array(
        [
            pyztraj.compute_native_contacts_q(
                _ref, f, ca_idx, ca_idx, cutoff=q_cutoff.value
            )
            for f in frames
        ]
    )
    _t = times / 1000.0
    _df = pd.DataFrame({"Time (ns)": _t, "Q": _q})
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y=alt.Y(
                "Q:Q",
                title="Q (fraction of native contacts)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("Q", format=".3f"),
            ],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def rama_section(mo):
    mo.md("""
    ## Ramachandran Plot
    Overlap of φ/ψ dihedral angles for all residues across all frames.
    """)
    return


@app.cell
def rama_plot(alt, frames, mo, np, pd, pyztraj, residue_labels, structure):
    # Collect phi/psi from all frames
    _stride = 1
    _all_phi = []
    _all_psi = []
    _all_res = []
    _all_frame = []
    for _fi in range(0, len(frames), _stride):
        _phi = np.degrees(pyztraj.compute_phi(structure, frames[_fi]))
        _psi = np.degrees(pyztraj.compute_psi(structure, frames[_fi]))
        _mask = ~(np.isnan(_phi) | np.isnan(_psi))
        _all_phi.extend(_phi[_mask])
        _all_psi.extend(_psi[_mask])
        _all_res.extend(
            [residue_labels[i] for i in range(len(residue_labels)) if _mask[i]]
        )
        _all_frame.extend([_fi] * int(_mask.sum()))

    _phi_arr = np.array(_all_phi)
    _psi_arr = np.array(_all_psi)

    # Pre-compute 2D histogram in Python (avoids Vega bin signal conflicts)
    _nbins = 120
    _edges = np.linspace(-180, 180, _nbins + 1)
    _hist, _xedges, _yedges = np.histogram2d(_phi_arr, _psi_arr, bins=[_edges, _edges])
    _heat_rows = []
    for _ix in range(_nbins):
        for _iy in range(_nbins):
            if _hist[_ix, _iy] > 0:
                _heat_rows.append(
                    {
                        "phi0": _xedges[_ix],
                        "phi1": _xedges[_ix + 1],
                        "psi0": _yedges[_iy],
                        "psi1": _yedges[_iy + 1],
                        "count": int(_hist[_ix, _iy]),
                    }
                )
    _df_heat = pd.DataFrame(_heat_rows)
    _max_count = _df_heat["count"].max()
    _df_heat["Frequency"] = _df_heat["count"] / _max_count

    _scale_x = alt.Scale(domain=[-180, 180])
    _scale_y = alt.Scale(domain=[-180, 180])
    # Colorscale: navy → cyan → yellow → red → dark red
    _rama_colors = alt.Scale(
        domain=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        range=["#00008B", "#0066FF", "#00CCCC", "#FFFF00", "#FF3300", "#8B0000"],
    )

    _chart = (
        alt.Chart(_df_heat)
        .mark_rect()
        .encode(
            x=alt.X("phi0:Q", scale=_scale_x, title="φ (degrees)"),
            x2="phi1:Q",
            y=alt.Y("psi0:Q", scale=_scale_y, title="ψ (degrees)"),
            y2="psi1:Q",
            color=alt.Color(
                "Frequency:Q",
                scale=_rama_colors,
                legend=alt.Legend(title="Frequency"),
            ),
            tooltip=[
                alt.Tooltip("phi0", format=".0f", title="φ"),
                alt.Tooltip("psi0", format=".0f", title="ψ"),
                alt.Tooltip("Frequency", format=".2f"),
            ],
        )
    ).properties(width=500, height=500, title="Total Ramachandran Plot")

    mo.ui.altair_chart(_chart)
    return


@app.cell
def dssp_section(mo):
    mo.md("""
    ## DSSP — Secondary Structure
    """)
    return


@app.cell
def dssp_controls(mo):
    dssp_stride = mo.ui.slider(
        debounce=True,
        start=1,
        stop=50,
        step=1,
        value=10,
        label="Frame stride",
        show_value=True,
    )
    dssp_stride
    return (dssp_stride,)


@app.cell
def dssp_plot(
    alt,
    dssp_stride,
    frames,
    n_frames,
    pd,
    pyztraj,
    residue_labels,
    structure,
):
    _stride_val = dssp_stride.value
    _frame_indices = list(range(0, n_frames, _stride_val))
    _n_res = len(residue_labels)
    _rows = []
    for _fi in _frame_indices:
        _assignments = pyztraj.compute_dssp(structure, frames[_fi])
        for _col, _code in enumerate(_assignments[:_n_res]):
            _rows.append(
                {
                    "Residue": residue_labels[_col],
                    "Frame": _fi,
                    "SS": _code if _code.strip() else "C",
                    "res_idx": _col,
                }
            )

    _df = pd.DataFrame(_rows)
    _ss_order = ["H", "G", "I", "E", "B", "T", "S", "P", "C"]
    _ss_colors = [
        "#e41a1c",
        "#ff7f00",
        "#984ea3",
        "#377eb8",
        "#4daf4a",
        "#a6d854",
        "#ffff33",
        "#f781bf",
        "#999999",
    ]
    _chart = (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X("Residue:N", sort=alt.SortField("res_idx")),
            y=alt.Y("Frame:O", sort="ascending"),
            color=alt.Color(
                "SS:N",
                scale=alt.Scale(domain=_ss_order, range=_ss_colors),
                legend=alt.Legend(title="SS"),
            ),
            tooltip=["Residue", "Frame", "SS"],
        )
        .properties(width="container", height=400)
    )
    _chart
    return


@app.cell
def dihedral_section(mo):
    mo.md("""
    ## Backbone Dihedral Angles
    """)
    return


@app.cell
def dihedral_controls(mo, residue_labels):
    dihedral_res = mo.ui.dropdown(
        options={label: i for i, label in enumerate(residue_labels)},
        value=residue_labels[min(5, len(residue_labels) - 1)],
        label="Residue",
    )
    dihedral_type = mo.ui.dropdown(
        options={"Phi (φ)": "phi", "Psi (ψ)": "psi", "Omega (ω)": "omega"},
        value="Phi (φ)",
        label="Dihedral type",
    )
    mo.hstack([dihedral_res, dihedral_type])
    return dihedral_res, dihedral_type


@app.cell
def dihedral_plot(
    alt,
    dihedral_res,
    dihedral_type,
    frames,
    mo,
    np,
    pd,
    pyztraj,
    residue_labels,
    structure,
    times,
):
    _fn = {
        "phi": pyztraj.compute_phi,
        "psi": pyztraj.compute_psi,
        "omega": pyztraj.compute_omega,
    }[dihedral_type.value]
    _ri = dihedral_res.value
    _angles = np.array([np.degrees(_fn(structure, f)[_ri]) for f in frames])
    _t = times / 1000.0
    _df = pd.DataFrame({"Time (ns)": _t, "Angle (deg)": _angles})
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y=alt.Y(
                "Angle (deg):Q",
                scale=alt.Scale(domain=[-180, 180]),
                title="Angle (degrees)",
            ),
            tooltip=[
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("Angle (deg)", format=".1f"),
            ],
        )
        .properties(
            width="container",
            height=350,
            title=f"{dihedral_type.value} — {residue_labels[_ri]}",
        )
        .interactive()
    )
    return


@app.cell
def pca_section(mo):
    mo.md("""
    ## PCA — Principal Component Analysis
    """)
    return


@app.cell
def pca_compute(ca_idx, frames, pyztraj):
    pca_result = pyztraj.compute_pca(frames, atom_indices=ca_idx, n_components=10)
    return (pca_result,)


@app.cell
def pca_plot(alt, ca_idx, frames, mo, np, pca_result, pd):
    _var = pca_result.variance_ratio
    _cum = np.cumsum(_var)
    _n = min(10, len(_var))
    _df_var = pd.DataFrame(
        {
            "Component": [f"PC{i + 1}" for i in range(_n)],
            "Variance (%)": _var[:_n] * 100,
            "Cumulative (%)": _cum[:_n] * 100,
            "order": list(range(_n)),
        }
    )
    _bars = (
        alt.Chart(_df_var)
        .mark_bar()
        .encode(
            x=alt.X("Component:N", sort=alt.SortField("order")),
            y="Variance (%):Q",
            tooltip=[
                "Component",
                alt.Tooltip("Variance (%)", format=".1f"),
                alt.Tooltip("Cumulative (%)", format=".1f"),
            ],
        )
    )
    _line = (
        alt.Chart(_df_var)
        .mark_line(color="coral", point=True)
        .encode(
            x=alt.X("Component:N", sort=alt.SortField("order")),
            y="Cumulative (%):Q",
        )
    )
    _chart_var = (_bars + _line).properties(
        width=400, height=350, title="Variance Explained"
    )

    _mean = np.mean([f[ca_idx].flatten() for f in frames], axis=0)
    _proj = np.array(
        [(f[ca_idx].flatten() - _mean) @ pca_result.eigenvectors for f in frames]
    )
    _df_pc = pd.DataFrame(
        {"PC1": _proj[:, 0], "PC2": _proj[:, 1], "Frame": np.arange(len(frames))}
    )
    _chart_pc = (
        alt.Chart(_df_pc)
        .mark_circle(size=20, opacity=0.7)
        .encode(
            x="PC1:Q",
            y="PC2:Q",
            color=alt.Color("Frame:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=[
                "Frame",
                alt.Tooltip("PC1", format=".2f"),
                alt.Tooltip("PC2", format=".2f"),
            ],
        )
        .properties(width=450, height=350, title="PC1 vs PC2")
    )

    mo.hstack([_chart_var, _chart_pc])
    return


@app.cell
def msd_section(mo):
    mo.md("""
    ## MSD — Mean Square Displacement
    """)
    return


@app.cell
def msd_plot(alt, ca_idx, frames, mo, pd, pyztraj, times):
    _msd = pyztraj.compute_msd(frames, atom_indices=ca_idx)
    _t = times / 1000.0
    _df = pd.DataFrame({"Time (ns)": _t, "MSD (Å²)": _msd})
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y="MSD (Å²):Q",
            tooltip=[
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("MSD (Å²)", format=".1f"),
            ],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def rdf_section(mo):
    mo.md("""
    ## RDF — Radial Distribution Function (Cα–Cα)
    """)
    return


@app.cell
def rdf_controls(mo):
    rdf_r_max = mo.ui.slider(
        debounce=True,
        start=5.0,
        stop=20.0,
        step=1.0,
        value=12.0,
        label="r_max (Å)",
        show_value=True,
    )
    rdf_n_bins = mo.ui.slider(
        debounce=True,
        start=50,
        stop=300,
        step=10,
        value=100,
        label="Bins",
        show_value=True,
    )
    mo.hstack([rdf_r_max, rdf_n_bins])
    return rdf_n_bins, rdf_r_max


@app.cell
def rdf_plot(alt, ca_idx, frames, mo, np, pd, pyztraj, rdf_n_bins, rdf_r_max):
    _all_r = None
    _all_gr = None
    for _i in range(len(frames)):
        _ca_coords = frames[_i][ca_idx]
        _extent = _ca_coords.max(axis=0) - _ca_coords.min(axis=0) + 10.0
        _box_vol = float(np.prod(_extent))
        _r, _gr = pyztraj.compute_rdf(
            _ca_coords,
            _ca_coords,
            box_volume=_box_vol,
            r_min=0.0,
            r_max=rdf_r_max.value,
            n_bins=int(rdf_n_bins.value),
        )
        if _all_gr is None:
            _all_r = np.array(_r)
            _all_gr = np.array(_gr)
        else:
            _all_gr += np.array(_gr)
    _all_gr /= len(frames)

    _df = pd.DataFrame({"r (Å)": _all_r, "g(r)": _all_gr})
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="r (Å):Q",
            y="g(r):Q",
            tooltip=[
                alt.Tooltip("r (Å)", format=".2f"),
                alt.Tooltip("g(r)", format=".3f"),
            ],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def com_section(mo):
    mo.md("""
    ## Center of Mass
    """)
    return


@app.cell
def com_plot(alt, mo, np, pd, results, times):
    _com = results["center_of_mass"]
    _t = times / 1000.0
    _df = pd.DataFrame(
        {
            "Time (ns)": np.tile(_t, 3),
            "Position (Å)": np.concatenate([_com[:, 0], _com[:, 1], _com[:, 2]]),
            "Axis": ["x"] * len(_t) + ["y"] * len(_t) + ["z"] * len(_t),
        }
    )
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y=alt.Y("Position (Å):Q", scale=alt.Scale(zero=False)),
            color="Axis:N",
            tooltip=[
                "Axis",
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("Position (Å)", format=".2f"),
            ],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def inertia_section(mo):
    mo.md("""
    ## Principal Moments of Inertia
    """)
    return


@app.cell
def inertia_plot(alt, frames, mo, np, pd, pyztraj, structure, times):
    _moments = np.array(
        [
            pyztraj.compute_principal_moments(
                pyztraj.compute_inertia(f, structure.masses)
            )
            for f in frames
        ]
    )
    _t = times / 1000.0
    _df = pd.DataFrame(
        {
            "Time (ns)": np.tile(_t, 3),
            "Moment (amu·Å²)": np.concatenate(
                [_moments[:, 0], _moments[:, 1], _moments[:, 2]]
            ),
            "Component": ["I₁ (smallest)"] * len(_t)
            + ["I₂"] * len(_t)
            + ["I₃ (largest)"] * len(_t),
        }
    )
    mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y=alt.Y("Moment (amu·Å²):Q", scale=alt.Scale(zero=False)),
            color="Component:N",
            tooltip=[
                "Component",
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("Moment (amu·Å²)", format=".0f"),
            ],
        )
        .properties(width="container", height=350)
        .interactive()
    )
    return


@app.cell
def distance_section(mo):
    mo.md("""
    ## Interactive Pairwise Distances
    """)
    return


@app.cell
def distance_controls(mo, n_residues, residue_labels):
    dist_res_i = mo.ui.dropdown(
        options={label: i for i, label in enumerate(residue_labels)},
        value=residue_labels[0],
        label="Residue i (CA)",
    )
    dist_res_j = mo.ui.dropdown(
        options={label: i for i, label in enumerate(residue_labels)},
        value=residue_labels[min(10, n_residues - 1)],
        label="Residue j (CA)",
    )
    mo.hstack([dist_res_i, dist_res_j])
    return dist_res_i, dist_res_j


@app.cell
def distance_plot(
    alt,
    ca_idx,
    dist_res_i,
    dist_res_j,
    frames,
    mo,
    np,
    pd,
    pyztraj,
    residue_labels,
    times,
):
    _ai = int(ca_idx[dist_res_i.value])
    _aj = int(ca_idx[dist_res_j.value])
    _pairs = np.array([[_ai, _aj]], dtype=np.uint32)
    _dists = np.array([pyztraj.compute_distances(f, _pairs)[0] for f in frames])
    _t = times / 1000.0

    _label_i = residue_labels[dist_res_i.value]
    _label_j = residue_labels[dist_res_j.value]

    _df = pd.DataFrame({"Time (ns)": _t, "Distance (Å)": _dists})
    _chart = mo.ui.altair_chart(
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="Time (ns):Q",
            y=alt.Y("Distance (Å):Q", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Time (ns)", format=".2f"),
                alt.Tooltip("Distance (Å)", format=".2f"),
            ],
        )
        .properties(
            width="container", height=350, title=f"Cα distance: {_label_i} — {_label_j}"
        )
        .interactive()
    )

    _stats = mo.hstack(
        [
            mo.stat(value=f"{np.mean(_dists):.2f} Å", label="Mean"),
            mo.stat(value=f"{np.std(_dists):.2f} Å", label="Std"),
            mo.stat(value=f"{np.min(_dists):.2f} Å", label="Min"),
            mo.stat(value=f"{np.max(_dists):.2f} Å", label="Max"),
        ],
        justify="center",
        gap=1,
    )

    mo.vstack([_chart, _stats])
    return


@app.cell
def summary_section(mo):
    mo.md("""
    ## Summary
    """)
    return


@app.cell
def summary_table(mo, np, pd, results):
    _data = {
        "Metric": ["RMSD (Å)", "Rg (Å)", "SASA (Å²)", "H-bonds", "Contacts"],
        "Mean": [
            f"{np.mean(results['rmsd']):.3f}",
            f"{np.mean(results['rg']):.2f}",
            f"{np.mean(results['sasa']):.1f}",
            f"{np.mean(results['n_hbonds']):.1f}",
            f"{np.mean(results['n_contacts']):.1f}",
        ],
        "Std": [
            f"{np.std(results['rmsd']):.3f}",
            f"{np.std(results['rg']):.2f}",
            f"{np.std(results['sasa']):.1f}",
            f"{np.std(results['n_hbonds']):.1f}",
            f"{np.std(results['n_contacts']):.1f}",
        ],
        "Min": [
            f"{np.min(results['rmsd']):.3f}",
            f"{np.min(results['rg']):.2f}",
            f"{np.min(results['sasa']):.1f}",
            f"{int(np.min(results['n_hbonds']))}",
            f"{int(np.min(results['n_contacts']))}",
        ],
        "Max": [
            f"{np.max(results['rmsd']):.3f}",
            f"{np.max(results['rg']):.2f}",
            f"{np.max(results['sasa']):.1f}",
            f"{int(np.max(results['n_hbonds']))}",
            f"{int(np.max(results['n_contacts']))}",
        ],
    }
    mo.ui.table(pd.DataFrame(_data), selection=None)
    return


@app.cell
def footer(mo, pyztraj):
    mo.md(f"""
    *pyztraj {pyztraj.get_version()} — 3TVJ chain I (38 residues, 531 atoms, 1000 frames)*
    """)
    return


if __name__ == "__main__":
    app.run()
