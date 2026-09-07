"""File actions must stay attached to the directory and text file the user loaded."""
from pathlib import Path

import gradio as gr
import pytest

from pyscf_webui import working_directory as wd


@pytest.fixture(autouse=True)
def silence_notifications(monkeypatch):
    monkeypatch.setattr(gr, "Warning", lambda message: None)


def test_save_stays_attached_to_loaded_file(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first original")
    second.write_text("second original")

    editor, _, loaded_path = wd.on_view_text_file(str(tmp_path), first.name)
    assert editor["value"] == "first original"
    # A subsequent row selection must never make the old editor overwrite that row.
    reset = wd.on_selected_text_file_state_change(second.name)
    assert reset[1]["value"] == ""
    assert reset[2]["interactive"] is False
    assert reset[3] is None
    wd.on_save_text_file(str(tmp_path), loaded_path, "first edited")
    assert first.read_text() == "first edited"
    assert second.read_text() == "second original"


def test_save_rejects_editor_from_previous_directory(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "input.xyz").write_text("old geometry")
    (new / "input.xyz").write_text("new geometry")
    _, _, loaded_path = wd.on_view_text_file(str(old), "input.xyz")
    wd.on_save_text_file(str(new), loaded_path, "stale editor contents")
    assert (new / "input.xyz").read_text() == "new geometry"
    assert (old / "input.xyz").read_text() == "old geometry"


def test_deleted_text_file_is_not_recreated_by_stale_editor(tmp_path):
    target = tmp_path / "input.txt"
    target.write_text("original")
    _, _, loaded_path = wd.on_view_text_file(str(tmp_path), target.name)
    target.unlink()
    wd.on_save_text_file(str(tmp_path), loaded_path, "stale edit")
    assert not target.exists()


def test_file_editor_events_save_the_loaded_path(tmp_path):
    with gr.Blocks() as blocks:
        wd.working_directory_blocks(tmp_path)
    events = {event.fn.__name__: event for event in blocks.fns.values()}
    view = events["on_view_text_file"]
    save = events["on_save_text_file"]
    reset = events["on_open_working_directory_and_reset"]
    assert save.inputs[1] is view.outputs[2]
    assert save.inputs[1] is not view.inputs[1]
    assert save.inputs[1] in reset.outputs
    assert events["on_delete_file"].inputs[1] in reset.outputs


def test_directory_change_clears_old_delete_selection(tmp_path):
    target = tmp_path / "input.xyz"
    target.write_text("new directory geometry")
    selected_file, _, _, loaded_path, delete_button, *_ = wd.on_directory_change()
    assert loaded_path is None
    assert delete_button["interactive"] is False
    wd.on_delete_file(str(tmp_path), selected_file)
    assert target.exists()


def test_clean_without_open_directory_does_not_clean_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "unrelated.tmp"
    target.write_text("keep me")
    assert wd.on_clean_working_directory(None) == []
    assert target.read_text() == "keep me"


@pytest.mark.parametrize("name", ["..", "../outside", "/tmp/outside", "nested/name", "a\\b"])
def test_directory_name_cannot_escape_data(tmp_path, name):
    data = tmp_path / "data"
    updates = wd.on_open_working_directory(name, data)
    assert all(update == gr.update() for update in updates)
    assert not data.exists()


def test_directory_symlink_cannot_escape_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (data / "escape").symlink_to(outside, target_is_directory=True)
    assert wd.get_working_directories(data) == []
    updates = wd.on_open_working_directory("escape", data)
    assert all(update == gr.update() for update in updates)


def test_open_directory_uses_configured_base(tmp_path):
    data = tmp_path / "custom" / "data"
    dropdown, path, files, *_ = wd.on_open_working_directory("water", data)
    assert Path(path) == data / "water"
    assert Path(path).is_dir()
    assert dropdown["choices"] == ["water"]
    assert files == []


def test_delete_cannot_escape_selected_directory(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me")
    wd.on_delete_file(str(working), "../outside.txt")
    assert outside.read_text() == "keep me"


def test_upload_does_not_follow_symlink_outside_directory(tmp_path):
    source = tmp_path / "source" / "input.xyz"
    source.parent.mkdir()
    source.write_text("uploaded")
    working = tmp_path / "working"
    working.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me")
    (working / source.name).symlink_to(outside)
    wd.on_upload_file(str(working), str(source))
    assert outside.read_text() == "keep me"


def test_failed_text_load_disables_stale_editor(tmp_path):
    editor, button, loaded_path = wd.on_view_text_file(str(tmp_path), "missing.txt")
    assert editor["value"] == ""
    assert editor["interactive"] is False
    assert button["interactive"] is False
    assert loaded_path is None


def test_upload_without_open_directory_returns_empty_list(tmp_path):
    source = tmp_path / "input.xyz"
    source.write_text("geometry")
    assert wd.on_upload_file(None, str(source)) == []


def test_file_removed_while_refreshing_browser_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "get_files_in_working_directory", lambda path: ["vanished.txt"])
    assert wd.on_file_list_change(str(tmp_path)).empty
