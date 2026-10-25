"""Validate the SMILES boundary and the conformer tab's sketcher integration."""
from pathlib import Path
import shutil
import subprocess

import gradio as gr
import pytest
from rdkit import Chem

from pyscf_webui import conformer_generation, molecular_editor as editor


@pytest.fixture(autouse=True)
def warnings(monkeypatch):
    messages = []
    monkeypatch.setattr(gr, "Warning", messages.append)
    return messages


@pytest.mark.parametrize("smiles", ["OCC", "C[C@H](O)F", "F/C=C/F", "[13CH3][NH3+]"])
def test_editor_round_trip_preserves_molecular_identity(smiles):
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    command = editor.on_load_smiles_into_editor(smiles)
    assert command["smiles"] == canonical
    assert command["load_id"]
    assert editor.on_smiles_from_editor(command) == canonical


def test_repeated_explicit_loads_have_distinct_ids():
    first = editor.on_load_smiles_into_editor("CCO")
    second = editor.on_load_smiles_into_editor("CCO")
    assert first["smiles"] == second["smiles"]
    assert first["load_id"] != second["load_id"]


@pytest.mark.parametrize("smiles", [" ", "", None])
def test_empty_smiles_explicitly_clears_drawing(smiles):
    command = editor.on_load_smiles_into_editor(smiles)
    assert command["smiles"] == ""
    assert command["load_id"]


@pytest.mark.parametrize("value", ["not smiles", "<script>alert(1)</script>", [], 42])
def test_invalid_smiles_keeps_existing_drawing(value, warnings):
    assert editor.on_load_smiles_into_editor(value) == gr.skip()
    assert "has not been changed" in warnings[-1]


@pytest.mark.parametrize("value", [None, "CCO", {}, {"smiles": []}, {"smiles": "C1CC"}])
def test_invalid_editor_payload_keeps_existing_smiles(value, warnings):
    assert editor.on_smiles_from_editor(value) == gr.skip()
    assert warnings


def test_empty_drawing_clears_smiles():
    assert editor.on_smiles_from_editor({"smiles": "  ", "load_id": None}) == ""


def test_editor_ignores_unrelated_browser_fields():
    assert editor.on_smiles_from_editor({"smiles": "OCC", "html": "<script>"}) == "CCO"


def test_component_uses_packaged_code_and_constant_template(tmp_path, monkeypatch):
    monkeypatch.setattr(editor, "KETCHER_DIR", tmp_path)
    component = editor.molecular_editor()
    config = component.get_config()
    assert config["value"] == {"smiles": "", "load_id": None}
    assert config["props"]["editor_available"] is False
    assert config["js_on_load"] == editor._FRONTEND_PATH.read_text(encoding="utf-8")
    assert "${value}" not in config["html_template"]


def test_conformer_tab_uses_explicit_editor_actions_and_shared_status():
    with gr.Blocks() as blocks:
        directory = gr.State()
        files = gr.State()
        shared_status = gr.Markdown()
        conformer_generation.conformer_generation_tab_content(directory, files, shared_status)
    events = {event.fn.__name__: event for event in blocks.fns.values()}
    exporting = events["on_smiles_from_editor"]
    loading = events["on_load_smiles_into_editor"]
    generating = events["on_generate_conformers"]
    assert loading.inputs[0] is exporting.outputs[0]
    assert exporting.inputs[0] is loading.outputs[0]
    assert exporting.outputs[0] is generating.inputs[1]
    assert generating.outputs[0] is shared_status
    dependencies = blocks.get_config_file()["dependencies"]
    assert any((exporting.inputs[0]._id, "input") in dependency["targets"]
               for dependency in dependencies)
    assert not any((exporting.inputs[0]._id, "change") in dependency["targets"]
                   for dependency in dependencies)


def test_invalid_smiles_error_cannot_inject_status_html(tmp_path):
    status, _, _ = conformer_generation.on_generate_conformers(
        str(tmp_path), "<script>alert(1)</script>", 0, 1, 1, 0.1, 0.5, "molecule", "xyz",
    )
    assert "<script>" not in status
    assert "&lt;script&gt;" in status


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for frontend tests")
def test_editor_frontend_serialization_and_failure_recovery():
    harness = Path(__file__).with_name("molecular_editor_frontend.cjs")
    result = subprocess.run(
        [shutil.which("node"), str(harness), str(editor._FRONTEND_PATH)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
