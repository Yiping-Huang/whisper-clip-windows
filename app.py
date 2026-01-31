import os
import sys
import time
import numpy as np
import sounddevice as sd
import whisper
from PySide6 import QtCore, QtGui, QtWidgets


APP_TITLE = "Whisper Clip"
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")
AVAILABLE_MODELS = ["base", "small", "medium", "large"]
SAMPLE_RATE = 16000
APP_ICON = os.path.join(os.path.dirname(__file__), "assets", "whisper.ico")


def _build_app_icon(style: QtWidgets.QStyle) -> QtGui.QIcon:
    if os.path.exists(APP_ICON):
        return QtGui.QIcon(APP_ICON)

    base_icon = style.standardIcon(QtWidgets.QStyle.SP_MediaVolume)
    pixmap = base_icon.pixmap(64, 64)
    if pixmap.isNull():
        return base_icon

    tinted = QtGui.QPixmap(pixmap.size())
    tinted.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(tinted)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), QtGui.QColor(230, 230, 230))
    painter.end()
    return QtGui.QIcon(tinted)


class ModelLoader(QtCore.QObject):
    loaded = QtCore.Signal(object, str)
    error = QtCore.Signal(str, str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    @QtCore.Slot()
    def run(self):
        try:
            download_root = os.path.join(os.path.dirname(__file__), "model")
            os.makedirs(download_root, exist_ok=True)
            model = whisper.load_model(self.model_name, download_root=download_root)
            self.loaded.emit(model, self.model_name)
        except Exception as exc:
            self.error.emit(str(exc), self.model_name)


class Transcriber(QtCore.QObject):
    finished = QtCore.Signal(str, float)
    error = QtCore.Signal(str)

    def __init__(self, model, audio: np.ndarray):
        super().__init__()
        self.model = model
        self.audio = audio

    @QtCore.Slot()
    def run(self):
        try:
            start = time.time()
            result = self.model.transcribe(self.audio, fp16=False)
            text = (result.get("text") or "").strip()
            elapsed = time.time() - start
            self.finished.emit(text, elapsed)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setMinimumSize(280, 80)
        self.resize(300, 100)

        self.model = None
        self.model_name = DEFAULT_MODEL
        self.recording = False
        self.frames = []
        self.stream = None
        self.latest_text = ""
        self.transcript_visible = False
        self._collapsed_min_height = 80
        self._expanded_min_height = 300
        self._collapsed_size = None
        self._loading_model = False
        self._pending_model_name = None

        self._build_ui()
        self._load_model_async()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.model_combo = QtWidgets.QComboBox()
        model_items = list(AVAILABLE_MODELS)
        if self.model_name not in model_items:
            model_items.insert(0, self.model_name)
        self.model_combo.addItems(model_items)
        self.model_combo.setCurrentText(self.model_name)
        self.model_combo.currentTextChanged.connect(self._on_model_change)
        self.model_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.model_combo.setMinimumContentsLength(4)
        self.model_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )

        self.text_area = QtWidgets.QPlainTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setPlaceholderText("Transcribed text will appear here.")
        self.text_area.setObjectName("Output")
        self.text_area.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.text_area.setVisible(False)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)

        self._record_icon = self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        self._stop_icon = self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop)
        themed_copy = QtGui.QIcon.fromTheme("edit-copy")
        if themed_copy.isNull():
            themed_copy = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        self._copy_icon = themed_copy
        self._transcript_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_FileDialogDetailedView
        )
        self._transcript_hide_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_FileDialogListView
        )

        self.record_button = QtWidgets.QToolButton()
        self.record_button.setCheckable(True)
        self.record_button.setObjectName("RecordButton")
        self.record_button.clicked.connect(self._toggle_recording)
        self.record_button.setIcon(self._record_icon)
        self.record_button.setIconSize(QtCore.QSize(18, 18))
        self.record_button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.record_button.setToolTip("Record / Stop")

        self.copy_button = QtWidgets.QToolButton()
        self.copy_button.setObjectName("CopyButton")
        self.copy_button.clicked.connect(self._copy_latest)
        self.copy_button.setEnabled(False)
        self.copy_button.setIcon(self._copy_icon)
        self.copy_button.setIconSize(QtCore.QSize(18, 18))
        self.copy_button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.copy_button.setToolTip("Copy latest transcript")

        self.transcript_toggle = QtWidgets.QToolButton()
        self.transcript_toggle.setCheckable(True)
        self.transcript_toggle.setChecked(False)
        self.transcript_toggle.setObjectName("TranscriptToggle")
        self.transcript_toggle.clicked.connect(self._toggle_transcript)
        self.transcript_toggle.setIcon(self._transcript_icon)
        self.transcript_toggle.setIconSize(QtCore.QSize(18, 18))
        self.transcript_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.transcript_toggle.setToolTip("Show/Hide transcript")

        controls.addWidget(self.model_combo)
        controls.addWidget(self.record_button)
        controls.addWidget(self.copy_button)
        controls.addWidget(self.transcript_toggle)
        controls.addStretch(1)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(6)

        self.status_label = QtWidgets.QLabel("Loading model\u2026")
        self.status_label.setObjectName("Status")

        self.clipboard_label = QtWidgets.QLabel("Clipboard: idle")
        self.clipboard_label.setObjectName("ClipboardStatus")
        self._set_clipboard_state("idle", "Clipboard: idle")

        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        status_row.addWidget(self.clipboard_label)

        root.addWidget(self.text_area, 1)
        root.addLayout(controls)
        root.addLayout(status_row)

        self.setCentralWidget(central)
        self._apply_styles()
        self._apply_window_icon()

    def _apply_window_icon(self):
        self.setWindowIcon(_build_app_icon(self.style()))

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #111827, stop:0.6 #1f2937, stop:1 #0f172a);
                color: #f9fafb;
            }
            QComboBox {
                background-color: #0b1220;
                border: 1px solid #374151;
                padding: 4px 15px 4px 8px;
                border-radius: 8px;
                color: #e5e7eb;
                font-size: 12px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #374151;
            }
            QComboBox::down-arrow {
                image: url(assets/chevron-down.svg);
                width: 10px;
                height: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #0b1220;
                border: 1px solid #374151;
                color: #e5e7eb;
                selection-background-color: #1f2937;
            }
            #Output {
                background-color: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 10px;
                padding: 10px;
                color: #e5e7eb;
                font-size: 13px;
            }
            #RecordButton {
                background-color: #ef4444;
                color: #ffffff;
                padding: 6px;
                border-radius: 10px;
            }
            #RecordButton:checked {
                background-color: #22c55e;
            }
            #CopyButton, #TranscriptToggle {
                background-color: #111827;
                border: 1px solid #374151;
                color: #e5e7eb;
                padding: 6px;
                border-radius: 10px;
            }
            #Status {
                color: #cbd5f5;
                font-size: 12px;
            }
            #ClipboardStatus {
                font-size: 12px;
                color: #9ca3af;
            }
            #ClipboardStatus[state="ready"] {
                color: #22c55e;
            }
            #ClipboardStatus[state="working"] {
                color: #fbbf24;
            }
            #ClipboardStatus[state="empty"] {
                color: #f87171;
            }
            """
        )

    def _set_status(self, message: str):
        self.status_label.setText(message)

    def _set_clipboard_state(self, state: str, message: str):
        self.clipboard_label.setText(message)
        self.clipboard_label.setProperty("state", state)
        self.clipboard_label.style().unpolish(self.clipboard_label)
        self.clipboard_label.style().polish(self.clipboard_label)

    def _on_model_change(self, model_name: str):
        if self.recording:
            self._set_status("Stop recording to change the model.")
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentText(self.model_name)
            self.model_combo.blockSignals(False)
            return
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self._load_model_async()

    def _load_model_async(self):
        if self._loading_model:
            self._pending_model_name = self.model_name
            return

        self._set_status("Loading model\u2026")
        self._set_clipboard_state("idle", "Clipboard: idle")
        self.record_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self._loading_model = True
        self._pending_model_name = None

        self.loader_thread = QtCore.QThread(self)
        self.loader_worker = ModelLoader(self.model_name)
        self.loader_worker.moveToThread(self.loader_thread)

        self.loader_thread.started.connect(self.loader_worker.run)
        self.loader_worker.loaded.connect(self._on_model_loaded)
        self.loader_worker.error.connect(self._on_model_error)
        self.loader_worker.loaded.connect(self.loader_thread.quit)
        self.loader_worker.error.connect(self.loader_thread.quit)

        self.loader_thread.start()

    @QtCore.Slot(object, str)
    def _on_model_loaded(self, model, model_name: str):
        self._loading_model = False
        if model_name != self.model_name:
            if self._pending_model_name:
                self._load_model_async()
            return
        self.model = model
        self.record_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self._set_status("Ready")

    @QtCore.Slot(str, str)
    def _on_model_error(self, message: str, model_name: str):
        self._loading_model = False
        if model_name != self.model_name:
            if self._pending_model_name:
                self._load_model_async()
            return
        self.record_button.setEnabled(False)
        self.model_combo.setEnabled(True)
        self._set_status(f"Model load failed: {message}")

    def _toggle_recording(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self.model is None:
            self._set_status("Model not loaded yet.")
            self.record_button.setChecked(False)
            return

        try:
            self.frames = []
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._on_audio,
            )
            self.stream.start()
            self.recording = True
            self.record_button.setIcon(self._stop_icon)
            self.model_combo.setEnabled(False)
            self._set_status("Recording\u2026")
            self._set_clipboard_state("working", "Clipboard: recording")
        except Exception as exc:
            self.record_button.setChecked(False)
            self._set_status(f"Microphone error: {exc}")

    def _stop_recording(self):
        self.record_button.setEnabled(False)
        self.record_button.setChecked(False)
        self.record_button.setIcon(self._record_icon)
        self.recording = False

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

        if not self.frames:
            self._set_status("No audio captured.")
            self.record_button.setEnabled(True)
            self.model_combo.setEnabled(True)
            self._set_clipboard_state("empty", "Clipboard: empty")
            return

        audio = np.concatenate(self.frames, axis=0).flatten()
        self._start_transcription(audio)

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            pass
        if self.recording:
            self.frames.append(indata.copy())

    def _start_transcription(self, audio: np.ndarray):
        self._set_status("Transcribing\u2026")
        self._set_clipboard_state("working", "Clipboard: transcribing")

        self.worker_thread = QtCore.QThread(self)
        self.worker = Transcriber(self.model, audio)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_transcription_finished)
        self.worker.error.connect(self._on_transcription_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)

        self.worker_thread.finished.connect(self._on_worker_finished)
        self.worker_thread.start()

    @QtCore.Slot(str, float)
    def _on_transcription_finished(self, text: str, elapsed: float):
        self.latest_text = text
        self.text_area.setPlainText(text)
        self.copy_button.setEnabled(bool(text))
        QtWidgets.QApplication.clipboard().setText(text)
        if text:
            self._set_status(f"Done in {elapsed:.1f}s. Copied to clipboard.")
            self._set_clipboard_state("ready", "Clipboard: ready")
        else:
            self._set_status("Done, but no speech detected.")
            self._set_clipboard_state("empty", "Clipboard: empty")

    @QtCore.Slot(str)
    def _on_transcription_error(self, message: str):
        self._set_status(f"Transcription failed: {message}")
        self._set_clipboard_state("empty", "Clipboard: error")

    def _on_worker_finished(self):
        self.record_button.setEnabled(True)
        self.model_combo.setEnabled(True)

    def _copy_latest(self):
        if not self.latest_text:
            return
        QtWidgets.QApplication.clipboard().setText(self.latest_text)
        self._set_status("Copied latest text to clipboard.")
        self._set_clipboard_state("ready", "Clipboard: ready")

    def _toggle_transcript(self):
        self.transcript_visible = not self.transcript_visible
        self.text_area.setVisible(self.transcript_visible)
        if self.transcript_visible:
            self.transcript_toggle.setIcon(self._transcript_hide_icon)
            self.setMinimumHeight(self._expanded_min_height)
            if self._collapsed_size is None:
                self._collapsed_size = self.size()
        else:
            self.transcript_toggle.setIcon(self._transcript_icon)
            self.setMinimumHeight(self._collapsed_min_height)
            if self._collapsed_size is not None:
                self.resize(self._collapsed_size)
        self._resize_to_content()

    def _resize_to_content(self):
        self.adjustSize()
        desired_height = max(self.sizeHint().height(), self.minimumHeight())
        self.resize(self.width(), desired_height)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(_build_app_icon(app.style()))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
