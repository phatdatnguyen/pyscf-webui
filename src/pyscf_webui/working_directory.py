"""Working-directory column: directory selection, file browser, and file viewers.

Owns the left-hand column of the UI. A "working directory" is a subfolder under
``data/`` in which all structure files, job specs, logs, results and exports live.
This module also holds the two shared ``gr.State`` objects (current path and file list)
that the other tabs subscribe to; any handler that changes files on disk returns a
fresh file list into that state so every tab's dropdowns stay in sync.
"""
import os
from pathlib import Path
import shutil
import time
import pandas as pd
import gradio as gr
from .utils import JOB_SUFFIX, RESULT_SUFFIX, get_files_in_working_directory, mol_from_structure_file
from .visualization import render_structure_html

def get_working_directories(base_path="./data/") -> list[str]:
    """List the existing working-directory names (immediate subfolders of ``data/``)."""
    base_path = Path(base_path).resolve()
    if not base_path.is_dir():
        return []
    return sorted(d.name for d in base_path.iterdir()
                  if d.is_dir() and d.resolve().parent == base_path)

def _file_path(working_directory_path, file_name):
    """Resolve a file inside the selected directory before reading or writing it."""
    if not working_directory_path:
        raise ValueError("Please open a working directory first.")
    directory = Path(working_directory_path).resolve()
    if not directory.is_dir():
        raise ValueError("The working directory no longer exists.")
    if (not isinstance(file_name, str) or not file_name.strip()
            or file_name in (".", "..") or "/" in file_name or "\\" in file_name):
        raise ValueError("Please select a file in the working directory.")
    path = directory / file_name
    if path.resolve().parent != directory:
        raise ValueError("The selected file is outside the working directory.")
    return path


def on_open_working_directory(working_directory, base_path="./data/"):
    """Create (if needed) and open ``working_directory`` under ``data/``.

    Returns updates for the directory dropdown, the path/file-list states, and the
    two buttons that become interactive once a directory is open. Warns and no-ops
    when the name is blank.
    """
    try:
        if not isinstance(working_directory, str) or not working_directory.strip():
            raise ValueError("Please specify a working directory.")
        if working_directory in (".", "..") or "/" in working_directory or "\\" in working_directory:
            raise ValueError("Use a directory name without path separators.")
        base_path = Path(base_path).resolve()
        working_directory_path = (base_path / working_directory).resolve()
        if working_directory_path.parent != base_path:
            raise ValueError("The working directory must be inside the data directory.")
        working_directory_path.mkdir(parents=True, exist_ok=True)
        files = get_files_in_working_directory(working_directory_path)
        return (gr.update(choices=get_working_directories(base_path), value=working_directory),
                str(working_directory_path), files, gr.update(interactive=True),
                gr.update(interactive=True))
    except Exception as exc:
        gr.Warning(str(exc))
        # An invalid dropdown entry must not clear the path while leaving the old
        # file actions enabled. Keep the previous directory and its state together.
        return tuple(gr.update() for _ in range(5))

def on_file_list_change(working_directory_path) -> pd.DataFrame:
    """Build the file-browser DataFrame (File / Type / Modified) for the directory.

    Each file is classified by extension into a human-readable type, and rows are
    ordered by actual modification time (newest first). Yields an empty table when no
    working directory is open, since this fires on the shared state's ``.change`` and
    that state is ``None`` before the first directory is opened.
    """
    if not working_directory_path:
        return pd.DataFrame(columns=["File", "Type", "Modified"])

    files = get_files_in_working_directory(working_directory_path)

    # Update the file dataframe
    file_info = []
    for f in files:
        if f.endswith('.xyz') or f.endswith('.pdb') or f.endswith('.mol') or f.endswith('.mol2'):
            file_type = "Structure file"
        elif f.endswith(RESULT_SUFFIX):
            file_type = "Result file"
        elif f.endswith(JOB_SUFFIX):
            file_type = "Job spec file"
        elif f.endswith('.chk'):
            file_type = "Wavefunction file"
        elif f.endswith('.freq.molden'):
            file_type = "Normal mode file"
        elif f.endswith('.molden'):
            file_type = "Molden orbital file"
        elif f.endswith('.h5'):
            file_type = "Array data file"
        elif f.endswith('.log'):
            file_type = "Log file"
        elif f.endswith('.cube'):
            file_type = "Cube file"
        elif f.endswith('.csv'):
            file_type = "Exported data file"
        elif f.endswith('.txt'):
            file_type = "Text file"
        elif f == 'timer.dat' or f.endswith('.clean'):
            # Left behind by an older Psi4-era run; "Clean Working Directory" removes them.
            file_type = "Scratch file"
        else:
            file_type = "Other File"
        
        file_path = os.path.join(working_directory_path, f)
        try:
            mtime = os.path.getmtime(file_path)
        except FileNotFoundError:
            continue  # a runner or another browser session may have removed it
        file_info.append([f, file_type, mtime])
    # Sort by the numeric modification time (newest first), then format for display.
    # Sorting the ctime *string* would order by weekday name, not chronologically.
    file_info.sort(key=lambda x: x[2], reverse=True)
    file_info = [[f, file_type, time.ctime(mtime)] for f, file_type, mtime in file_info]
    file_df = pd.DataFrame(file_info, columns=["File", "Type", "Modified"])

    return file_df

def on_select_file(evt: gr.SelectData):
    """Handle a row click in the file browser.

    Returns the selected file name plus two derived selections (a viewable structure
    file and a viewable text file, each ``None`` when the type doesn't apply) and
    enables the delete button. These drive which viewer buttons become active.
    """
    selected_file_name = evt.row_value[0]
    # The .log is plain prose, not a coordinate file, so (unlike ORCA's) it is not
    # offered to the 3D viewer -- optimized geometries are written as .xyz instead.
    if selected_file_name.endswith(('.xyz', '.pdb', '.mol', '.mol2')):
        selected_structure_file = selected_file_name
    else:
        selected_structure_file = None
    if selected_file_name.endswith(('.xyz', '.pdb', '.mol', '.mol2', '.log', '.json', '.csv',
                                    '.txt', '.molden')):
        selected_text_file = selected_file_name
    else:
        selected_text_file = None
    
    return selected_file_name, selected_structure_file, selected_text_file, gr.update(interactive=True)

def on_selected_structure_file_state_change(state):
    """Enable the "View Structure File" button only when a structure file is selected."""
    return gr.update(interactive=(state is not None))

def on_selected_text_file_state_change(state):
    """Reset the editor when selecting another file, until that file is loaded."""
    return (gr.update(interactive=(state is not None)),
            gr.update(label="Text File Viewer", value="", interactive=False),
            gr.update(interactive=False), None)


def on_directory_change():
    """Discard selections and edits belonging to the previous working directory."""
    return (None, None, None, None, gr.update(interactive=False), None,
            gr.update(label="Text File Viewer", value="", interactive=False),
            gr.update(interactive=False))


def on_open_working_directory_and_reset(working_directory, base_path):
    """Change the directory and clear its old selections in the same UI event."""
    updates = on_open_working_directory(working_directory, base_path)
    reset = (on_directory_change() if isinstance(updates[1], str)
             else tuple(gr.update() for _ in range(8)))
    return (*updates, *reset)


def on_upload_file(working_directory_path, file_path):
    """Copy an uploaded file into the working directory; return the refreshed file list."""
    try:
        target = _file_path(working_directory_path, os.path.basename(file_path))
        shutil.copy2(file_path, target)
    except Exception as exc:
        gr.Warning("Error uploading file!\n" + str(exc))
    return get_files_in_working_directory(working_directory_path)

def on_delete_file(working_directory_path, selected_file_name):
    """Delete the selected file; return the refreshed file list (warns on error)."""
    if selected_file_name is None:
        return get_files_in_working_directory(working_directory_path)
    
    try:
        file_path = _file_path(working_directory_path, selected_file_name)
        os.remove(file_path)
        status = "File deleted successfully."
    except Exception as exc:
        status = "Error deleting file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_clean_working_directory(working_directory_path):
    """Remove scratch and leftover files from the working directory.

    Covers PySCF's temporary files, the ``psi.*.clean``/``timer.dat`` droppings a working
    directory carried over from the Psi4 app may still hold, and Windows
    ``Zone.Identifier`` streams. Returns the refreshed file list; warns on error.
    """
    try:
        if not working_directory_path:
            raise ValueError("Please open a working directory first.")
        # Engine scratch: PySCF tmp files, plus psi.*.clean / timer.dat from an older run.
        files_to_clean = [f for f in os.listdir(working_directory_path)
                          if f.startswith('#') or f.endswith(".tmp") or f.endswith(".clean")
                          or f == "timer.dat" or f.endswith("Zone.Identifier") or ".tmp." in f]
        for f in files_to_clean:
            file_path = os.path.join(working_directory_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        status = "Working directory cleaned successfully."
    except Exception as exc:
        status = "Error cleaning working directory!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_view_structure_file(working_directory_path, file_name):
    """Render a 3D viewer for the selected structure/output file.

    Loads the geometry with :func:`mol_from_structure_file` — the same loader the
    Calculation tab uses to build the geometry the engine receives, so the atom indices
    labelled here are the indices PySCF works in. Rendering is delegated to
    :func:`pyscf_webui.visualization.render_structure_html`, which the orbital viewer in
    the Result tab shares. Returns ``None`` and warns on error.
    """
    try:
        file_path = _file_path(working_directory_path, file_name)
        return render_structure_html(mol_from_structure_file(str(file_path)))
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None

def on_view_text_file(working_directory_path, text_file_name):
    """Load a text-viewable file into the editable text viewer.

    Returns updates for the text area, save button, and the absolute path of the
    loaded file. Save uses this path, so selecting a different row cannot redirect it.
    """
    try:
        text_file_path = _file_path(working_directory_path, text_file_name)
        with open(text_file_path, 'r') as file:
            content = file.read()
        return (gr.update(label=f"Text File Viewer - {text_file_name}", value=content, interactive=True),
                gr.update(interactive=True), str(text_file_path))
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return (gr.update(label="Text File Viewer", value="", interactive=False),
                gr.update(interactive=False), None)

def on_save_text_file(working_directory_path, loaded_text_file_path, text_content):
    """Write the text-viewer contents back to the selected file.

    Returns the refreshed file list; warns if no file is selected or on write error.
    """
    if loaded_text_file_path is None:
        gr.Warning("Please load a text file to save.")
        return get_files_in_working_directory(working_directory_path)
    
    try:
        text_file_path = _file_path(working_directory_path, Path(loaded_text_file_path).name)
        if str(text_file_path) != loaded_text_file_path:
            raise ValueError("The loaded file belongs to a different working directory. Load it again.")
        if not text_file_path.is_file():
            raise ValueError("The loaded file no longer exists.")
        with open(text_file_path, 'w') as file:
            file.write(text_content)
        status = "File saved successfully."
    except Exception as exc:
        status = "Error saving file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def working_directory_blocks(base_path="./data/"):
    """Build the working-directory column and wire its events.

    Returns the two shared ``gr.State`` objects — ``(working_directory_path_state,
    working_directory_file_list_state)`` — which the calculation and result tabs
    subscribe to.
    """
    with gr.Column(scale=1):
        base_path_state = gr.State(str(Path(base_path).resolve()))
        working_directory_dropdown = gr.Dropdown(label="Working Directory", choices=get_working_directories(base_path), value="wd", allow_custom_value=True)
        working_directory_path_state = gr.State()
        open_working_directory_button = gr.Button(value="Create/Open Working Directory")
        working_directory_file_list_state = gr.State()
        working_directory_file_dataframe = gr.Dataframe(label="Files in Working Directory", headers=["File", "Type", "Modified"], max_height=360, wrap=True, interactive=False)
        selected_file_state = gr.State()
        selected_structure_file_state = gr.State()
        selected_text_file_state = gr.State()
        loaded_text_file_path_state = gr.State()
        with gr.Row():
            add_file_upload_button = gr.UploadButton(label="Add File", file_types=[".xyz", ".pdb", ".mol", ".mol2"], interactive=False)
            delete_file_button = gr.Button(value="Delete Selected File", interactive=False)
            clean_working_directory_button = gr.Button(value="Clean Working Directory", interactive=False)
        view_structure_button = gr.Button(value="View Structure File", interactive=False)
        structure_viewer_html = gr.HTML()
        view_text_file_button = gr.Button(value="View Text File", interactive=False)
        text_file_viewer_textarea = gr.TextArea(label="Text File Viewer", lines=20, elem_id="textfile_viewer", interactive=False)
        save_text_file_button = gr.Button(value="Save Text File", interactive=False)
    
    directory_inputs = [working_directory_dropdown, base_path_state]
    directory_outputs = [working_directory_dropdown, working_directory_path_state, working_directory_file_list_state, add_file_upload_button, clean_working_directory_button]
    directory_outputs += [selected_file_state, selected_structure_file_state, selected_text_file_state, loaded_text_file_path_state, delete_file_button, structure_viewer_html, text_file_viewer_textarea, save_text_file_button]
    working_directory_dropdown.change(on_open_working_directory_and_reset, directory_inputs, directory_outputs)
    open_working_directory_button.click(on_open_working_directory_and_reset, directory_inputs, directory_outputs)
    working_directory_file_list_state.change(on_file_list_change, working_directory_path_state, working_directory_file_dataframe)
    working_directory_file_dataframe.select(on_select_file, [], [selected_file_state, selected_structure_file_state, selected_text_file_state, delete_file_button])
    selected_structure_file_state.change(on_selected_structure_file_state_change, selected_structure_file_state, view_structure_button)
    selected_text_file_state.change(on_selected_text_file_state_change, selected_text_file_state, [view_text_file_button, text_file_viewer_textarea, save_text_file_button, loaded_text_file_path_state])
    add_file_upload_button.upload(on_upload_file, [working_directory_path_state, add_file_upload_button], working_directory_file_list_state)
    delete_file_button.click(on_delete_file, [working_directory_path_state, selected_file_state], working_directory_file_list_state)
    clean_working_directory_button.click(on_clean_working_directory, working_directory_path_state, working_directory_file_list_state)
    view_structure_button.click(on_view_structure_file, [working_directory_path_state, selected_structure_file_state], structure_viewer_html)
    view_text_file_button.click(on_view_text_file, [working_directory_path_state, selected_text_file_state], [text_file_viewer_textarea, save_text_file_button, loaded_text_file_path_state])
    save_text_file_button.click(on_save_text_file, [working_directory_path_state, loaded_text_file_path_state, text_file_viewer_textarea], working_directory_file_list_state)

    return working_directory_path_state, working_directory_file_list_state
