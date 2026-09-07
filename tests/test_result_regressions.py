"""Regression cases for changing results and rendering independent browser sessions."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import gradio as gr
import pandas as pd
import pytest

from pyscf_webui import result, visualization
from pyscf_webui.utils import write_json


def test_loader_resets_ecd_plot_and_wires_it_as_an_output(tmp_path, water_result):
    write_json(str(tmp_path / "sp.result.json"), water_result)
    loaded = result.on_load_result_file(str(tmp_path), "sp.result.json")
    with gr.Blocks() as app:
        result.result_tab_content(gr.State(), gr.State(), gr.Markdown())
    loader = next(fn for fn in app.fns.values() if fn.fn is result.on_load_result_file)
    assert len(loader.outputs) == len(loaded) == result.RESULT_OUTPUT_COUNT
    assert loader.outputs[-1].label == "ECD spectrum"
    assert loaded[-1] is None
    assert result._blank_outputs("error")[-1] is None


def test_non_object_json_reports_a_load_error(tmp_path):
    (tmp_path / "bad.result.json").write_text("[]")
    outputs = result.on_load_result_file(str(tmp_path), "bad.result.json")
    assert len(outputs) == result.RESULT_OUTPUT_COUNT
    assert "JSON object" in outputs[0]


def test_unconverged_geometry_remains_visible_with_a_warning(tmp_path, water_result):
    water_result["optimization"] = {"energies_hartree": [-76.3, -76.4], "converged": False}
    write_json(str(tmp_path / "opt.result.json"), water_result)
    outputs = result.on_load_result_file(str(tmp_path), "opt.result.json")
    assert "orange" in outputs[0]
    assert "Geometry optimization did not converge" in outputs[0]
    assert outputs[6]["visible"] is True
    assert outputs[3].endswith("hartree")


def test_frequency_note_warns_about_unconverged_geometry(tmp_path, water_result):
    water_result["optimization"] = {"energies_hartree": [-76.3, -76.4], "converged": False}
    water_result["frequency"] = {"frequencies_cm1": [1700.0, 3700.0, 3800.0],
                                 "ir_intensities_km_mol": None}
    water_result["thermochemistry"] = {"temperature_k": 298.15, "zpve_hartree": 0.02}
    write_json(str(tmp_path / "freq.result.json"), water_result)
    outputs = result.on_load_result_file(str(tmp_path), "freq.result.json")
    assert "orange" in outputs[0]
    assert "not confirmed to be stationary" in outputs[12]
    assert "thermochemistry" in outputs[12]
    assert outputs[8]["visible"] is True
    assert outputs[11] is not None


def test_unconverged_emission_remains_visible_with_a_warning(tmp_path, water_result):
    water_result["emission"] = {
        "converged": False, "root": 1, "emission_energy_ev": 3.1, "wavelength_nm": 397.7,
        "excited_energy_hartree": -114.0, "ground_energy_at_excited_geometry_hartree": -114.1,
    }
    write_json(str(tmp_path / "emission.result.json"), water_result)
    outputs = result.on_load_result_file(str(tmp_path), "emission.result.json")
    assert "orange" in outputs[0]
    assert "Excited-state optimization did not converge" in outputs[0]
    assert outputs[20]["visible"] is True
    assert outputs[21] is not None


def test_legacy_completed_result_with_failed_scf_is_rejected(tmp_path, water_result):
    water_result["converged"] = False
    write_json(str(tmp_path / "legacy.result.json"), water_result)
    outputs = result.on_load_result_file(str(tmp_path), "legacy.result.json")
    assert "SCF calculation did not converge" in outputs[0]
    assert "red" in outputs[0]
    assert outputs[1] is None
    assert outputs[2]["visible"] is False


def test_missing_transition_strengths_leave_blanks_and_only_omit_affected_bands(
        tmp_path, partial_excitation_result):
    write_json(str(tmp_path / "td.result.json"), partial_excitation_result)
    outputs = result.on_load_result_file(str(tmp_path), "td.result.json")
    assert "orange" in outputs[0]
    assert "transition strengths are unavailable" in outputs[0]
    assert outputs[13]["visible"] is True
    table = outputs[14]
    assert len(table) == 2
    assert table["Oscillator strength"].tolist() == ["", "0.300000"]
    assert table["Rotatory strength"].tolist() == ["1.200000", ""]
    assert [annotation.x for annotation in outputs[15].layout.annotations] == [227.82]
    ecd = result.on_show_ecd_spectrum(outputs[1])
    assert [annotation.x for annotation in ecd.layout.annotations] == [303.76]


def test_excitation_energies_stay_visible_when_all_strengths_are_missing(
        tmp_path, partial_excitation_result, monkeypatch):
    for state in partial_excitation_result["excitations"]:
        state["oscillator_strength"] = state["rotatory_strength"] = None
    write_json(str(tmp_path / "td.result.json"), partial_excitation_result)
    outputs = result.on_load_result_file(str(tmp_path), "td.result.json")
    assert outputs[13]["visible"] is True
    assert outputs[14]["Energy (eV)"].tolist() == ["4.0817", "5.4423"]
    assert outputs[15] is None
    assert outputs[16]["interactive"] is False
    warnings = []
    monkeypatch.setattr(gr, "Warning", warnings.append)
    assert result.on_show_ecd_spectrum(outputs[1]) is None
    assert "no available rotatory strengths" in warnings[0]


def test_zero_transition_strength_is_distinct_from_unavailable(partial_excitation_result):
    states = partial_excitation_result["excitations"]
    states[1]["oscillator_strength"] = 0.0
    table = result._excitation_dataframe(states)
    assert table["Oscillator strength"].tolist() == ["", "0.000000"]
    assert result._states_with_strength(states, "oscillator_strength") == [states[1]]


@pytest.mark.parametrize("change", ["file", "directory", "rerun"])
def test_visualization_rejects_changed_result_before_launching(
        tmp_path, water_result, monkeypatch, change):
    source = tmp_path / "sp.result.json"
    write_json(str(source), water_result)
    loaded = result.on_load_result_file(str(tmp_path), source.name)[1]
    directory, name = str(tmp_path), source.name
    if change == "file":
        name = "other.result.json"
    elif change == "directory":
        directory = str(tmp_path / "other")
    else:
        import os
        stamp = source.stat().st_mtime_ns + 1_000_000
        os.utime(source, ns=(stamp, stamp))
    warnings = []
    monkeypatch.setattr(gr, "Warning", warnings.append)
    monkeypatch.setattr(result.subprocess, "run", lambda *a, **k: pytest.fail("launched stale result"))
    assert result.on_visualize(directory, name, loaded, "MO 1", "blue", "red", 0.8, 0.02, 0.2) is None
    assert "load the selected result" in warnings[0]


def test_cube_failure_includes_stderr(tmp_path, water_result, monkeypatch):
    from subprocess import CompletedProcess

    write_json(str(tmp_path / "sp.result.json"), water_result)
    loaded = result.on_load_result_file(str(tmp_path), "sp.result.json")[1]
    warnings = []
    monkeypatch.setattr(gr, "Warning", warnings.append)
    monkeypatch.setattr(result.subprocess, "run", lambda *a, **k: CompletedProcess(
        [], 1, stdout="", stderr="Checkpoint could not be read"))
    assert result.on_visualize(str(tmp_path), "sp.result.json", loaded, "MO 1",
                               "blue", "red", 0.8, 0.02, 0.2) is None
    assert "Checkpoint could not be read" in warnings[0]


def test_viewer_urls_keep_previous_render_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization, "_STATIC_DIR", str(tmp_path))
    monkeypatch.setattr(visualization.nglview, "write_html",
                        lambda path, views: Path(path).write_text(views[0]))
    first = visualization._write_view("first molecule", "structure.html")
    second = visualization._write_view("second molecule", "structure.html")
    assert first != second
    for iframe, expected in [(first, "first molecule"), (second, "second molecule")]:
        url = iframe.split('src="')[1].split('"')[0].split("?")[0]
        assert (tmp_path / url.removeprefix("/static/")).read_text() == expected


def test_parallel_widget_renders_do_not_share_or_unregister_each_other(monkeypatch):
    from ipywidgets.widgets import widget as widget_module

    registry = {"existing": object()}
    monkeypatch.setattr(widget_module, "_instances", registry)
    start = threading.Barrier(2)

    def render(key):
        start.wait(timeout=5)
        with visualization._transient_widgets():
            registry[key] = object()
            time.sleep(0.03)
            return set(registry)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(render, ("first", "second")))
    assert first == {"existing", "first"}
    assert second == {"existing", "second"}
    assert set(registry) == {"existing"}


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_nmr_references_fall_back(value):
    assert result._parse_reference(value, 31.57) == 31.57


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "nested\\outside", "", ".."])
def test_csv_export_names_cannot_escape_working_directory(tmp_path, name):
    status, files = result.on_export_data(str(tmp_path), name, pd.DataFrame({"energy": [-76.4]}))
    assert "red" in status
    assert files == []
    assert list(tmp_path.iterdir()) == []


def test_csv_export_does_not_follow_a_symlink_outside_the_directory(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    target = tmp_path / "outside.csv"
    target.write_text("preserve this file")
    (workdir / "data.csv").symlink_to(target)
    status, _ = result.on_export_data(str(workdir), "data", pd.DataFrame({"energy": [-76.4]}))
    assert "red" in status
    assert target.read_text() == "preserve this file"


def test_csv_export_refreshes_file_list(tmp_path):
    status, files = result.on_export_data(str(tmp_path), "energies", pd.DataFrame({"energy": [-76.4]}))
    assert "green" in status
    assert files == ["energies.csv"]
    assert pd.read_csv(tmp_path / "energies.csv")["energy"].tolist() == [-76.4]
