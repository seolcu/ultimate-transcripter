from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

from ultimate_transcripter.pipeline import run_transcription
from ultimate_transcripter.types import PipelineConfig, RunSummary


_GUI_IMPORT_ERROR: Exception | None = None
Adw: Any
Gio: Any
GLib: Any
Gtk: Any

try:
    import gi

    gi.require_version("Adw", "1")
    gi.require_version("Gtk", "4.0")
    from gi.repository import Adw, Gio, GLib, Gtk  # type: ignore
except Exception as exc:  # pragma: no cover - depends on local desktop deps
    _GUI_IMPORT_ERROR = exc
    Adw = None
    Gio = None
    GLib = None
    Gtk = None


PROVIDERS = ("openai", "assemblyai")
PROVIDER_MODELS: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"),
    "assemblyai": ("universal-2",),
}
DEFAULT_MODELS = {provider: models[0] for provider, models in PROVIDER_MODELS.items()}


if _GUI_IMPORT_ERROR is None:

    class TranscriberWindow(Adw.ApplicationWindow):
        def __init__(self, app: Any) -> None:
            super().__init__(application=app)
            self.set_title("Ultimate Transcripter")
            self.set_default_size(980, 760)

            self._running = False
            self._pulse_source_id: int | None = None

            self.toast_overlay = Adw.ToastOverlay()
            self.set_content(self.toast_overlay)

            toolbar_view = Adw.ToolbarView()
            self.toast_overlay.set_child(toolbar_view)

            header = Adw.HeaderBar()
            toolbar_view.add_top_bar(header)

            scroller = Gtk.ScrolledWindow()
            scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            toolbar_view.set_content(scroller)

            outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
            outer.set_margin_top(18)
            outer.set_margin_bottom(18)
            outer.set_margin_start(18)
            outer.set_margin_end(18)
            scroller.set_child(outer)

            hero = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            hero_title = Gtk.Label(label="Fast, clean transcription")
            hero_title.set_xalign(0.0)
            hero_title.add_css_class("title-1")
            hero_subtitle = Gtk.Label(
                label=(
                    "Linux desktop app for long-form audio transcription. "
                    "API keys are used in-memory only."
                )
            )
            hero_subtitle.set_xalign(0.0)
            hero_subtitle.add_css_class("dim-label")
            hero.append(hero_title)
            hero.append(hero_subtitle)
            outer.append(hero)

            form_frame = Gtk.Frame()
            form_frame.add_css_class("card")
            outer.append(form_frame)

            form_grid = Gtk.Grid(column_spacing=12, row_spacing=10)
            form_grid.set_margin_top(14)
            form_grid.set_margin_bottom(14)
            form_grid.set_margin_start(14)
            form_grid.set_margin_end(14)
            form_frame.set_child(form_grid)

            self.audio_entry = Gtk.Entry(hexpand=True)
            self.audio_entry.set_placeholder_text("Select an audio file (.m4a, .mp3, .wav ...)")
            audio_browse = Gtk.Button(label="Browse")
            audio_browse.connect("clicked", self._on_browse_audio)
            audio_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            audio_box.append(self.audio_entry)
            audio_box.append(audio_browse)
            self._attach_row(form_grid, 0, "Audio file", audio_box)

            self.output_entry = Gtk.Entry(hexpand=True)
            self.output_entry.set_placeholder_text("Output directory (optional)")
            output_browse = Gtk.Button(label="Browse")
            output_browse.connect("clicked", self._on_browse_output)
            output_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            output_box.append(self.output_entry)
            output_box.append(output_browse)
            self._attach_row(form_grid, 1, "Output dir", output_box)

            self.provider_dropdown = Gtk.DropDown.new_from_strings(PROVIDERS)
            self.provider_dropdown.connect("notify::selected", self._on_provider_changed)
            self._attach_row(form_grid, 2, "Provider", self.provider_dropdown)

            self.model_dropdown = Gtk.DropDown()
            self._set_model_choices_for_provider(self._selected_provider())
            self._attach_row(form_grid, 3, "Model", self.model_dropdown)

            self.language_entry = Gtk.Entry()
            self.language_entry.set_placeholder_text("ko, en, ... (optional)")
            self.language_entry.set_text("ko")
            self._attach_row(form_grid, 4, "Language", self.language_entry)

            self.key_entry = Gtk.Entry()
            self.key_entry.set_visibility(False)
            self.key_entry.set_invisible_char("*")
            self.key_entry.set_placeholder_text("API key for selected provider")
            self._attach_row(form_grid, 5, "API key", self.key_entry)

            format_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            self.txt_check = Gtk.CheckButton(label="txt")
            self.txt_check.set_active(True)
            self.srt_check = Gtk.CheckButton(label="srt")
            self.srt_check.set_active(True)
            self.json_check = Gtk.CheckButton(label="json")
            self.json_check.set_active(True)
            format_box.append(self.txt_check)
            format_box.append(self.srt_check)
            format_box.append(self.json_check)
            self._attach_row(form_grid, 6, "Formats", format_box)

            advanced = Gtk.Expander(label="Advanced")
            advanced.set_expanded(False)
            advanced_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

            chunk_label = Gtk.Label(label="Chunk seconds")
            chunk_label.add_css_class("dim-label")
            self.chunk_spin = Gtk.SpinButton()
            self.chunk_spin.set_range(30, 7200)
            self.chunk_spin.set_increments(30, 60)
            self.chunk_spin.set_value(300)

            overlap_label = Gtk.Label(label="Overlap seconds")
            overlap_label.add_css_class("dim-label")
            self.overlap_spin = Gtk.SpinButton()
            self.overlap_spin.set_range(0, 120)
            self.overlap_spin.set_increments(1, 4)
            self.overlap_spin.set_value(8)

            advanced_box.append(chunk_label)
            advanced_box.append(self.chunk_spin)
            advanced_box.append(overlap_label)
            advanced_box.append(self.overlap_spin)
            advanced.set_child(advanced_box)
            self._attach_row(form_grid, 7, "", advanced)

            action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            self.start_button = Gtk.Button(label="Start Transcription")
            self.start_button.add_css_class("suggested-action")
            self.start_button.connect("clicked", self._on_start_clicked)
            action_box.append(self.start_button)

            self.progress_bar = Gtk.ProgressBar(show_text=True)
            self.progress_bar.set_hexpand(True)
            self.progress_bar.set_text("Idle")
            action_box.append(self.progress_bar)
            self._attach_row(form_grid, 8, "", action_box)

            log_frame = Gtk.Frame()
            log_frame.add_css_class("card")
            outer.append(log_frame)

            log_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            log_box.set_margin_top(14)
            log_box.set_margin_bottom(14)
            log_box.set_margin_start(14)
            log_box.set_margin_end(14)
            log_frame.set_child(log_box)

            log_title = Gtk.Label(label="Run log")
            log_title.set_xalign(0.0)
            log_title.add_css_class("heading")
            log_box.append(log_title)

            log_scroller = Gtk.ScrolledWindow()
            log_scroller.set_min_content_height(260)
            log_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            log_box.append(log_scroller)

            self.log_view = Gtk.TextView()
            self.log_view.set_editable(False)
            self.log_view.set_cursor_visible(False)
            self.log_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            self.log_view.add_css_class("monospace")
            log_scroller.set_child(self.log_view)
            self.log_buffer = self.log_view.get_buffer()

            self.audio_file_dialog = self._build_audio_file_dialog()
            self.output_folder_dialog = self._build_output_folder_dialog()

            self._append_log("Ready. Select audio, enter key, then start.")

        def _attach_row(self, grid: Any, row: int, title: str, widget: Any) -> None:
            if title:
                label = Gtk.Label(label=title)
                label.set_xalign(0.0)
                label.add_css_class("dim-label")
                grid.attach(label, 0, row, 1, 1)
            else:
                spacer = Gtk.Label(label="")
                grid.attach(spacer, 0, row, 1, 1)
            grid.attach(widget, 1, row, 1, 1)

        def _selected_provider(self) -> str:
            index = int(self.provider_dropdown.get_selected())
            if index < 0 or index >= len(PROVIDERS):
                return "openai"
            return PROVIDERS[index]

        def _on_provider_changed(self, *_args: object) -> None:
            provider = self._selected_provider()
            current = self._selected_model()
            self._set_model_choices_for_provider(provider, preferred_model=current)
            self.key_entry.set_placeholder_text(f"API key for {provider}")

        def _set_model_choices_for_provider(
            self, provider: str, preferred_model: str | None = None
        ) -> None:
            model_choices = PROVIDER_MODELS.get(provider, (DEFAULT_MODELS[provider],))
            model_store = Gtk.StringList.new(list(model_choices))
            self.model_dropdown.set_model(model_store)

            selected_index = 0
            if preferred_model and preferred_model in model_choices:
                selected_index = model_choices.index(preferred_model)
            self.model_dropdown.set_selected(selected_index)

        def _selected_model(self) -> str:
            selected_item = self.model_dropdown.get_selected_item()
            if selected_item is not None:
                value = str(selected_item.get_string() or "").strip()
                if value:
                    return value
            provider = self._selected_provider()
            return DEFAULT_MODELS[provider]

        def _build_audio_file_dialog(self) -> Any | None:
            file_dialog_cls = getattr(Gtk, "FileDialog", None)
            if file_dialog_cls is None:
                return None

            dialog = file_dialog_cls(title="Select audio file")
            filters = Gio.ListStore.new(Gtk.FileFilter)

            audio_filter = Gtk.FileFilter()
            audio_filter.set_name("Audio files")
            audio_filter.add_mime_type("audio/*")
            audio_filter.add_pattern("*.m4a")
            audio_filter.add_pattern("*.mp3")
            audio_filter.add_pattern("*.wav")
            audio_filter.add_pattern("*.flac")

            all_filter = Gtk.FileFilter()
            all_filter.set_name("All files")
            all_filter.add_pattern("*")

            filters.append(audio_filter)
            filters.append(all_filter)
            dialog.set_filters(filters)
            dialog.set_default_filter(audio_filter)
            return dialog

        def _build_output_folder_dialog(self) -> Any | None:
            file_dialog_cls = getattr(Gtk, "FileDialog", None)
            if file_dialog_cls is None:
                return None
            return file_dialog_cls(title="Select output directory")

        def _on_browse_audio(self, *_args: object) -> None:
            if self.audio_file_dialog is None:
                self._show_error(
                    "This GTK runtime does not provide the modern file dialog. "
                    "Please use GTK 4.10+ or type the path manually."
                )
                return
            self.audio_file_dialog.open(self, None, self._on_audio_dialog_done)

        def _on_audio_dialog_done(self, dialog: Any, result: Any) -> None:
            try:
                selected = dialog.open_finish(result)
            except Exception as exc:
                if _is_dialog_cancelled(exc):
                    return
                self._append_log(f"File dialog error: {exc}")
                self._show_error("Could not open file picker.")
                return

            if selected is None:
                return
            path = selected.get_path()
            if not path:
                return
            self.audio_entry.set_text(path)
            if not self.output_entry.get_text().strip():
                input_path = Path(path)
                default_output = input_path.with_name(f"{input_path.stem}_transcript")
                self.output_entry.set_text(str(default_output))

        def _on_browse_output(self, *_args: object) -> None:
            if self.output_folder_dialog is None:
                self._show_error(
                    "This GTK runtime does not provide the modern file dialog. "
                    "Please use GTK 4.10+ or type the path manually."
                )
                return
            self.output_folder_dialog.select_folder(self, None, self._on_output_dialog_done)

        def _on_output_dialog_done(self, dialog: Any, result: Any) -> None:
            try:
                selected = dialog.select_folder_finish(result)
            except Exception as exc:
                if _is_dialog_cancelled(exc):
                    return
                self._append_log(f"Folder dialog error: {exc}")
                self._show_error("Could not open folder picker.")
                return

            if selected is None:
                return
            path = selected.get_path()
            if not path:
                return
            self.output_entry.set_text(path)

        def _on_start_clicked(self, *_args: object) -> None:
            if self._running:
                return

            try:
                config, api_key = self._build_config()
            except ValueError as exc:
                self._show_error(str(exc))
                return

            self._append_log(f"Input: {config.input_path}")
            self._append_log(f"Output dir: {config.output_dir}")
            self._append_log(f"Provider: {config.provider} | model: {config.model}")

            self._set_running(True)

            worker = threading.Thread(
                target=self._run_pipeline_worker,
                args=(config, api_key),
                daemon=True,
            )
            worker.start()

        def _build_config(self) -> tuple[PipelineConfig, str]:
            audio_raw = self.audio_entry.get_text().strip()
            if not audio_raw:
                raise ValueError("Select an audio file first.")
            input_path = Path(audio_raw).expanduser()
            if not input_path.exists() or not input_path.is_file():
                raise ValueError(f"Audio file not found: {input_path}")

            output_raw = self.output_entry.get_text().strip()
            output_dir = (
                Path(output_raw).expanduser()
                if output_raw
                else input_path.with_name(f"{input_path.stem}_transcript")
            )
            if not output_raw:
                self.output_entry.set_text(str(output_dir))

            formats: set[str] = set()
            if self.txt_check.get_active():
                formats.add("txt")
            if self.srt_check.get_active():
                formats.add("srt")
            if self.json_check.get_active():
                formats.add("json")
            if not formats:
                raise ValueError("Choose at least one output format.")

            provider = self._selected_provider()
            model = self._selected_model()
            language = self.language_entry.get_text().strip() or None

            chunk_seconds = int(self.chunk_spin.get_value_as_int())
            overlap_seconds = int(self.overlap_spin.get_value_as_int())
            if chunk_seconds <= 0:
                raise ValueError("Chunk seconds must be greater than zero.")
            if overlap_seconds < 0:
                raise ValueError("Overlap seconds must be zero or more.")
            if overlap_seconds >= chunk_seconds:
                raise ValueError("Overlap seconds must be smaller than chunk seconds.")

            api_key = self.key_entry.get_text().strip()
            if not api_key:
                raise ValueError("Enter API key for the selected provider.")

            config = PipelineConfig(
                input_path=input_path,
                output_dir=output_dir,
                provider=provider,
                model=model,
                formats=formats,
                language=language,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                keep_temp_chunks=False,
                resume=True,
                api_base=None,
                timeout_seconds=300,
                max_retries=6,
            )
            return config, api_key

        def _run_pipeline_worker(self, config: PipelineConfig, api_key: str) -> None:
            def _log_from_worker(message: str) -> None:
                GLib.idle_add(self._append_log, message)

            try:
                summary = run_transcription(
                    config=config,
                    api_key=api_key,
                    logger=_log_from_worker,
                )
            except Exception as exc:
                GLib.idle_add(self._on_transcription_error, str(exc))
                return
            GLib.idle_add(self._on_transcription_done, summary)

        def _on_transcription_done(self, summary: RunSummary) -> None:
            self._set_running(False)
            self.key_entry.set_text("")
            self._append_log("Done.")
            self._append_log(f"Chunks processed: {summary.chunk_count}")
            if summary.text_path is not None:
                self._append_log(f"TXT:  {summary.text_path}")
            if summary.srt_path is not None:
                self._append_log(f"SRT:  {summary.srt_path}")
            if summary.json_path is not None:
                self._append_log(f"JSON: {summary.json_path}")
            toast = Adw.Toast.new("Transcription completed")
            self.toast_overlay.add_toast(toast)

        def _on_transcription_error(self, message: str) -> None:
            self._set_running(False)
            self._append_log(f"Error: {message}")
            self._show_error("Transcription failed. Check the log for details.")

        def _set_running(self, running: bool) -> None:
            self._running = running
            self.start_button.set_sensitive(not running)
            if running:
                self.progress_bar.set_text("Running...")
                if self._pulse_source_id is None:
                    self._pulse_source_id = GLib.timeout_add(120, self._on_progress_pulse)
                return

            if self._pulse_source_id is not None:
                GLib.source_remove(self._pulse_source_id)
                self._pulse_source_id = None
            self.progress_bar.set_fraction(0.0)
            self.progress_bar.set_text("Idle")

        def _on_progress_pulse(self) -> bool:
            if not self._running:
                return False
            self.progress_bar.pulse()
            return True

        def _append_log(self, line: str) -> None:
            end = self.log_buffer.get_end_iter()
            self.log_buffer.insert(end, f"{line.rstrip()}\n")
            self.log_view.scroll_to_iter(self.log_buffer.get_end_iter(), 0.0, False, 0.0, 1.0)

        def _show_error(self, message: str) -> None:
            toast = Adw.Toast.new(message)
            toast.set_timeout(6)
            self.toast_overlay.add_toast(toast)


    class UltimateTranscripterApp(Adw.Application):
        def __init__(self) -> None:
            super().__init__(
                application_id="io.github.seolcu.UltimateTranscripter",
                flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
            )

        def do_activate(self) -> None:
            window = self.props.active_window
            if window is None:
                window = TranscriberWindow(self)
            window.present()


def main(argv: Sequence[str] | None = None) -> int:
    if _GUI_IMPORT_ERROR is not None:
        fallback_code = _run_system_python_fallback(argv=argv, error=_GUI_IMPORT_ERROR)
        if fallback_code is not None:
            return fallback_code
        print(
            "Error: GTK4/libadwaita dependencies are missing. "
            "Install python3-gi, GTK4, and libadwaita."
        )
        print(f"Detail: {_GUI_IMPORT_ERROR}")
        return 1

    app = UltimateTranscripterApp()
    run_argv = sys.argv if argv is None else [sys.argv[0], *list(argv)]
    return int(app.run(run_argv))


def _run_system_python_fallback(*, argv: Sequence[str] | None, error: Exception) -> int | None:
    if not _is_missing_gi_error(error):
        return None

    current_python = Path(sys.executable).absolute()
    candidates: list[str] = []

    preferred = Path("/usr/bin/python3")
    if preferred.exists():
        candidates.append(str(preferred))

    discovered = shutil.which("python3")
    if discovered:
        candidates.append(discovered)

    source_root = _discover_source_root()

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        candidate_path = Path(candidate)
        try:
            if candidate_path.absolute() == current_python:
                continue
        except OSError:
            continue

        if not _python_has_gtk(candidate):
            continue

        env = os.environ.copy()
        if source_root:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{source_root}{os.pathsep}{existing}" if existing else source_root
            )

        command = [candidate, "-m", "ultimate_transcripter.gui"]
        if argv:
            command.extend(list(argv))

        try:
            completed = subprocess.run(command, env=env)
        except OSError:
            continue
        return int(completed.returncode)

    return None


def _discover_source_root() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "src"
    if source_root.exists():
        return str(source_root)
    return None


def _python_has_gtk(python_executable: str) -> bool:
    probe = (
        "import gi; "
        "gi.require_version('Gtk','4.0'); "
        "gi.require_version('Adw','1'); "
        "from gi.repository import Gtk, Adw"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _is_missing_gi_error(error: Exception) -> bool:
    if isinstance(error, ModuleNotFoundError) and getattr(error, "name", "") == "gi":
        return True
    return "No module named 'gi'" in str(error)


def _is_dialog_cancelled(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "dismissed" in message
        or "cancelled" in message
        or "canceled" in message
        or "operation was cancelled" in message
    )


if __name__ == "__main__":
    raise SystemExit(main())
