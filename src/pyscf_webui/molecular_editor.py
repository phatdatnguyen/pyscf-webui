"""Local Ketcher sketcher built with Gradio's HTML component API.

Only validated SMILES crosses the Python boundary. The iframe and its undo history
stay alive when a molecule is loaded or exported; molecule text is never HTML.
"""
from pathlib import Path
from uuid import uuid4

import gradio as gr
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from rdkit import Chem

from .install_molecular_editor import KETCHER_DIR


_FRONTEND_PATH = Path(__file__).resolve().parent / "frontend" / "molecular_editor.js"

# A constant template lets Gradio update the component value without replacing the
# iframe. Interpolating SMILES here would reset the drawing and expose input as HTML.
EDITOR_TEMPLATE = """
<div class="molecule-editor">
  <iframe title="Molecule drawing editor" class="molecule-canvas"></iframe>
  <div class="molecule-actions">
    <button type="button" class="use-drawing" disabled>Use drawing</button>
    <span class="editor-status" role="status" aria-live="polite">Loading molecular editor…</span>
  </div>
</div>
"""

EDITOR_CSS = """
.molecule-canvas { width: 100%; height: 580px; border: 1px solid var(--border-color-primary); border-radius: 6px; background: white; }
.molecule-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; padding-top: 10px; }
.use-drawing { padding: 8px 16px; border: 1px solid var(--button-primary-border-color); border-radius: 6px; background: var(--button-primary-background-fill); color: var(--button-primary-text-color); cursor: pointer; font: inherit; }
.use-drawing:disabled { opacity: 0.5; cursor: default; }
.editor-status { font-size: 0.9em; color: var(--body-text-color-subdued); }
.editor-status[data-error="true"] { color: var(--error-text-color, #b42318); }
"""


def mount_molecular_editor(app):
    """Serve installed editor assets separately from disposable /static viewers."""
    if (KETCHER_DIR / "index.html").is_file():
        app.mount("/molecule-editor", StaticFiles(directory=KETCHER_DIR, html=True),
                  name="molecule-editor")
    else:
        @app.get("/molecule-editor/{asset_path:path}", include_in_schema=False)
        def editor_not_installed(asset_path: str):
            return HTMLResponse(
                "<p>The drawing editor is not installed. You can still enter SMILES directly.</p>",
                status_code=503,
            )
    return app


def molecular_editor():
    """Build the sketcher; a plain SMILES field remains usable without its assets."""
    return gr.HTML(
        value={"smiles": "", "load_id": None},
        label="Molecule",
        html_template=EDITOR_TEMPLATE,
        css_template=EDITOR_CSS,
        js_on_load=_FRONTEND_PATH.read_text(encoding="utf-8"),
        editor_available=(KETCHER_DIR / "index.html").is_file(),
        elem_id="molecular-editor",
    )


def on_load_smiles_into_editor(smiles):
    """Validate an explicit load and give it an ID, including repeated molecules."""
    try:
        if smiles is None:
            smiles = ""
        if not isinstance(smiles, str):
            raise ValueError("SMILES must be text.")
        smiles = smiles.strip()
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not mol.GetNumAtoms():
                raise ValueError("Invalid SMILES.")
            smiles = Chem.MolToSmiles(mol, canonical=True)
        return {"smiles": smiles, "load_id": uuid4().hex}
    except Exception:
        gr.Warning("The SMILES string is invalid. The drawing has not been changed.")
        return gr.skip()


def on_smiles_from_editor(value):
    """Validate browser input before replacing the SMILES used for calculations."""
    try:
        if not isinstance(value, dict) or not isinstance(value.get("smiles"), str):
            raise ValueError("The drawing did not contain SMILES text.")
        smiles = value["smiles"].strip()
        if not smiles:
            return ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms():
            raise ValueError("Invalid molecule.")
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        gr.Warning("The drawing does not describe a valid molecule. Please check its atoms and bonds.")
        return gr.skip()
