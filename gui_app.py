#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small tray-friendly control window: pick camera, output device and voice, then start/stop translation."""
import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from pynput import keyboard

from audio.output_router import list_output_devices
from engine.recognizer import list_camera_indices
from service.fingerspell_buffer import DEFAULT_CONFIRM_DELAY, DEFAULT_FINALIZE_TIMEOUT
from service.gloss_buffer import DEFAULT_FINALIZE_TIMEOUT as DEFAULT_SENTENCE_PAUSE
from service.translator_service import TranslatorService

HOTKEY_TOGGLE_LETTERS = '<ctrl>+<alt>+l'
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
LOGO_BANNER_PATH = os.path.join(ASSETS_DIR, 'logo_banner.png')
LOGO_ICON_PATH = os.path.join(ASSETS_DIR, 'logo_icon.png')

MODE_NOTIFICATION_LABELS = {
    'words': 'Words (gestures)',
    'letters': 'Letters (fingerspelling)',
    'both': 'Both (letters + words)',
}

BLUE = '#2E5AA8'
PURPLE = '#7B3F98'

STYLESHEET = f"""
QWidget {{
    background-color: #f7f7fb;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
    color: #2b2b33;
}}
QGroupBox {{
    border: 1px solid #dcdce6;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px;
    font-weight: 600;
    color: {PURPLE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QComboBox, QLineEdit {{
    border: 1px solid #cfcfe0;
    border-radius: 5px;
    padding: 4px 6px;
    background-color: white;
}}
QRadioButton {{
    padding: 2px 6px;
}}
QPushButton#startButton {{
    background-color: {BLUE};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: 600;
}}
QPushButton#startButton:hover {{
    background-color: #244a8a;
}}
QPushButton#startButton[running="true"] {{
    background-color: #a8324a;
}}
QPushButton#startButton[running="true"]:hover {{
    background-color: #8a2a3e;
}}
QLabel#statusLabel {{
    background-color: white;
    border: 1px solid #dcdce6;
    border-radius: 6px;
    padding: 8px;
}}
QLabel#hotkeyHint {{
    color: #6b6b7a;
    font-size: 11px;
}}
"""


class MainWindow(QtWidgets.QWidget):
    status_changed = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('DeafEar - RSL Translator')
        self.setFixedWidth(420)
        self.setStyleSheet(STYLESHEET)
        if os.path.exists(LOGO_ICON_PATH):
            self.setWindowIcon(QtGui.QIcon(LOGO_ICON_PATH))
        self.service = None

        self.logo_label = QtWidgets.QLabel()
        if os.path.exists(LOGO_BANNER_PATH):
            banner = QtGui.QPixmap(LOGO_BANNER_PATH)
            self.logo_label.setPixmap(banner.scaledToWidth(320, QtCore.Qt.SmoothTransformation))
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)

        self.camera_combo = QtWidgets.QComboBox()
        camera_indices = list_camera_indices()
        if not camera_indices:
            camera_indices = [0]
        for index in camera_indices:
            self.camera_combo.addItem(f'Camera {index}', index)

        self.output_combo = QtWidgets.QComboBox()
        for device in list_output_devices():
            self.output_combo.addItem(device['name'], device['index'])

        self.male_radio = QtWidgets.QRadioButton('Male')
        self.female_radio = QtWidgets.QRadioButton('Female')
        self.female_radio.setChecked(True)
        voice_group = QtWidgets.QButtonGroup(self)
        voice_group.addButton(self.male_radio)
        voice_group.addButton(self.female_radio)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem('Words only (gestures)', 'words')
        self.mode_combo.addItem('Letters only (alphabet)', 'letters')
        self.mode_combo.addItem('Both (letters + words)', 'both')
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.letter_hold_spin = QtWidgets.QDoubleSpinBox()
        self.letter_hold_spin.setRange(0.1, 3.0)
        self.letter_hold_spin.setSingleStep(0.1)
        self.letter_hold_spin.setSuffix(' s')
        self.letter_hold_spin.setValue(DEFAULT_CONFIRM_DELAY)

        self.word_pause_spin = QtWidgets.QDoubleSpinBox()
        self.word_pause_spin.setRange(0.5, 10.0)
        self.word_pause_spin.setSingleStep(0.1)
        self.word_pause_spin.setSuffix(' s')
        self.word_pause_spin.setValue(DEFAULT_FINALIZE_TIMEOUT)

        self.sentence_pause_spin = QtWidgets.QDoubleSpinBox()
        self.sentence_pause_spin.setRange(0.5, 10.0)
        self.sentence_pause_spin.setSingleStep(0.1)
        self.sentence_pause_spin.setSuffix(' s')
        self.sentence_pause_spin.setValue(DEFAULT_SENTENCE_PAUSE)

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.setObjectName('startButton')
        self.start_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.start_button.clicked.connect(self.toggle_service)

        self.status_label = QtWidgets.QLabel('Stopped')
        self.status_label.setObjectName('statusLabel')
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(40)

        voice_row = QtWidgets.QHBoxLayout()
        voice_row.addWidget(self.male_radio)
        voice_row.addWidget(self.female_radio)
        voice_row.addStretch(1)

        settings_group = QtWidgets.QGroupBox('Setup')
        settings_form = QtWidgets.QFormLayout()
        settings_form.addRow('Camera:', self.camera_combo)
        settings_form.addRow('Voice output device:', self.output_combo)
        settings_form.addRow('Voice:', voice_row)
        settings_form.addRow('Recognition mode:', self.mode_combo)
        settings_group.setLayout(settings_form)

        timing_group = QtWidgets.QGroupBox('Timing')
        timing_form = QtWidgets.QFormLayout()
        timing_form.addRow('Letter hold time:', self.letter_hold_spin)
        timing_form.addRow('Pause between words:', self.word_pause_spin)
        timing_form.addRow('Pause before speaking sentence:', self.sentence_pause_spin)
        timing_group.setLayout(timing_form)

        hotkey_hint = QtWidgets.QLabel(f'Hotkey {HOTKEY_TOGGLE_LETTERS} toggles Letters/Words anywhere')
        hotkey_hint.setObjectName('hotkeyHint')
        hotkey_hint.setAlignment(QtCore.Qt.AlignCenter)

        status_group = QtWidgets.QGroupBox('Status')
        status_layout = QtWidgets.QVBoxLayout()
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(self.logo_label)
        layout.addWidget(settings_group)
        layout.addWidget(timing_group)
        layout.addWidget(hotkey_hint)
        layout.addWidget(self.start_button)
        layout.addWidget(status_group)

        self.status_changed.connect(self._handle_status_signal)

        tray_icon_source = (
            QtGui.QIcon(LOGO_ICON_PATH) if os.path.exists(LOGO_ICON_PATH)
            else self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        )
        self.tray_icon = QtWidgets.QSystemTrayIcon(tray_icon_source, self)
        tray_menu = QtWidgets.QMenu()
        show_action = tray_menu.addAction('Show')
        show_action.triggered.connect(self._show_window)
        quit_action = tray_menu.addAction('Quit')
        quit_action.triggered.connect(self.quit_app)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

        self._hotkey_listener = keyboard.GlobalHotKeys({
            HOTKEY_TOGGLE_LETTERS: self._toggle_letters_hotkey,
        })
        self._hotkey_listener.start()

    def _show_window(self):
        self.showNormal()
        self.activateWindow()

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData()
        if self.service and self.service.is_running:
            self.service.set_mode(mode)
        self._notify_mode(mode)

    def _notify_mode(self, mode):
        label = MODE_NOTIFICATION_LABELS.get(mode, mode)
        self.tray_icon.showMessage(
            'DeafEar', f'Mode switched to: {label}', QtWidgets.QSystemTrayIcon.Information, 1500
        )

    def _toggle_letters_hotkey(self):
        # Runs on the pynput listener thread; queued signal keeps Qt widget updates on the main thread.
        self.status_changed.emit('__toggle_letters__')

    @QtCore.pyqtSlot(str)
    def _handle_status_signal(self, text):
        if text == '__toggle_letters__':
            current = self.mode_combo.currentData()
            next_mode = 'words' if current == 'letters' else 'letters'
            index = self.mode_combo.findData(next_mode)
            if index >= 0:
                self.mode_combo.blockSignals(True)
                self.mode_combo.setCurrentIndex(index)
                self.mode_combo.blockSignals(False)
            if self.service and self.service.is_running:
                self.service.set_mode(next_mode)
            self.status_label.setText(f'Mode: {next_mode}')
            self._notify_mode(next_mode)
            return
        self.status_label.setText(text)

    def _on_tray_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self._show_window()

    def toggle_service(self):
        if self.service and self.service.is_running:
            self.service.stop()
            self.start_button.setText('Start')
            self._set_button_running(False)
            self.status_changed.emit('Stopped')
            return

        camera_index = self.camera_combo.currentData()
        output_index = self.output_combo.currentData()
        gender = 'male' if self.male_radio.isChecked() else 'female'
        mode = self.mode_combo.currentData()

        self.service = TranslatorService(
            camera_index=camera_index if camera_index is not None else 0,
            voice_gender=gender,
            output_device_index=output_index,
            mode=mode,
            on_status=self.status_changed.emit,
            letter_hold_delay=self.letter_hold_spin.value(),
            word_pause=self.word_pause_spin.value(),
            sentence_pause=self.sentence_pause_spin.value(),
        )
        self.service.start()
        self.start_button.setText('Stop')
        self._set_button_running(True)
        self.status_changed.emit('Running...')

    def _set_button_running(self, running):
        self.start_button.setProperty('running', 'true' if running else 'false')
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            'DeafEar', 'Still running in background', QtWidgets.QSystemTrayIcon.Information, 2000
        )

    def quit_app(self):
        if self.service:
            self.service.stop()
        self._hotkey_listener.stop()
        QtWidgets.QApplication.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
