"""Settings dialog."""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig, find_stockfish


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Réglages")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QLabel("Réglages")
        font = title.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)

        engine_row = QHBoxLayout()
        self.engine_edit = QLineEdit(config.stockfish_path)
        self.engine_edit.setPlaceholderText(
            "Chemin de stockfish.exe (vide = détection automatique)"
        )
        browse = QPushButton("Parcourir…")
        detect = QPushButton("Détecter")
        browse.clicked.connect(self._browse_engine)
        detect.clicked.connect(self._detect_engine)
        engine_row.addWidget(self.engine_edit, 1)
        engine_row.addWidget(browse)
        engine_row.addWidget(detect)
        form.addRow("Moteur Stockfish :", engine_row)

        self.movetime_spin = QSpinBox()
        self.movetime_spin.setRange(200, 5000)
        self.movetime_spin.setSingleStep(100)
        self.movetime_spin.setSuffix(" ms")
        self.movetime_spin.setValue(config.movetime_ms)
        form.addRow("Temps d'analyse :", self.movetime_spin)

        self.poll_spin = QSpinBox()
        self.poll_spin.setRange(300, 3000)
        self.poll_spin.setSingleStep(100)
        self.poll_spin.setSuffix(" ms")
        self.poll_spin.setValue(config.poll_ms)
        form.addRow("Lecture de l'écran :", self.poll_spin)

        self.advice_combo = QComboBox()
        self.advice_combo.addItem("Seulement quand c'est mon tour", "human")
        self.advice_combo.addItem("Toujours (trait aux Blancs ou aux Noirs)", "always")
        index = self.advice_combo.findData(config.advice_mode)
        self.advice_combo.setCurrentIndex(max(0, index))
        self.advice_combo.currentIndexChanged.connect(self._sync_human_controls)
        form.addRow("Afficher les conseils :", self.advice_combo)

        self.color_combo = QComboBox()
        self.color_combo.addItem("Blancs", "white")
        self.color_combo.addItem("Noirs", "black")
        color_index = self.color_combo.findData(config.human_color)
        self.color_combo.setCurrentIndex(max(0, color_index))
        form.addRow("Je joue avec :", self.color_combo)
        layout.addLayout(form)

        self.show_label = QCheckBox("Afficher le coup et l'évaluation sur la flèche")
        self.show_label.setChecked(config.show_label)
        layout.addWidget(self.show_label)

        self.hide_on_start = QCheckBox(
            "Masquer cette fenêtre quand l'analyse démarre "
            "(icône dans la zone de notification)"
        )
        self.hide_on_start.setChecked(config.hide_on_start)
        layout.addWidget(self.hide_on_start)

        note = QLabel(
            "L'assistant lit uniquement les pixels de l'écran. Il ne contrôle ni "
            "la souris ni le clavier."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6B7280;")
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Annuler")
        save = QPushButton("Enregistrer")
        save.setDefault(True)
        save.setStyleSheet(
            "QPushButton { background:#166534; color:white; padding:8px 18px; "
            "border-radius:6px; font-weight:600; }"
            "QPushButton:hover { background:#15803D; }"
        )
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self._accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

        self._sync_human_controls()

    def _sync_human_controls(self) -> None:
        self.color_combo.setEnabled(self.advice_combo.currentData() == "human")

    def _browse_engine(self) -> None:
        start = self.engine_edit.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir Stockfish",
            start,
            "Exécutables (*.exe);;Tous les fichiers (*)",
        )
        if path:
            self.engine_edit.setText(path)

    def _detect_engine(self) -> None:
        found = find_stockfish(self.engine_edit.text().strip())
        if found is None:
            QMessageBox.warning(
                self,
                "Stockfish introuvable",
                "Aucun Stockfish n'a été trouvé automatiquement. Téléchargez-le "
                "puis choisissez son exécutable avec « Parcourir… ».",
            )
            return
        self.engine_edit.setText(str(found))

    def _accept(self) -> None:
        self.accept()

    def result_config(self) -> AppConfig:
        advice_mode = str(self.advice_combo.currentData() or "human")
        human_color = str(self.color_combo.currentData() or "white")
        cfg = replace(
            self._config,
            stockfish_path=self.engine_edit.text().strip(),
            movetime_ms=self.movetime_spin.value(),
            poll_ms=self.poll_spin.value(),
            advice_mode=advice_mode,
            human_color=human_color,
            show_label=self.show_label.isChecked(),
            hide_on_start=self.hide_on_start.isChecked(),
        ).normalized()
        return cfg
