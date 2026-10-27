"""
gui.py - BUNYIM Dubbing Studio (Desktop, PySide6)
ដំណើរការ:  python main.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QRectF, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QFontMetrics, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

import engine as E

APP_TITLE = "BUNYIM DUBBING STUDIO"

STYLE = """
* { font-family: 'Khmer UI', 'Leelawadee UI', 'Segoe UI', sans-serif; font-size: 13px; color: #EDE7FA; }
QMainWindow, QWidget#root { background: #0D0A14; }
QFrame#header { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2A1650, stop:0.55 #3A1747, stop:1 #4A1A3C);
  border: 1px solid #5B3A8C; border-radius: 16px; }
QGroupBox { border: 1px solid #2E2347; border-radius: 14px; margin-top: 16px; padding: 12px 8px 8px 8px; background: #151021; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 8px; color: #F5B83D; font-weight: 700; }
QLabel#sub { color: #A99CC8; font-size: 11px; }
QLabel#pill { background: #1E1633; border: 1px solid #3B2D5C; border-radius: 10px; padding: 3px 10px; color: #CDBBFF; }
QLabel#subtitle { background: #05030A; color: #FFD479; font-size: 16px; font-weight: 700; border: 1px solid #6B4A12; border-radius: 10px; padding: 8px 12px; }
QPushButton { background: #231A38; border: 1px solid #3F3260; border-radius: 10px; padding: 8px 14px; }
QPushButton:hover { background: #2F2350; border-color: #6D55B0; }
QPushButton:disabled { color: #6C6286; background: #130F1E; border-color: #231B36; }
QPushButton#step1 { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7C3AED, stop:1 #A855F7); border: 1px solid #C4A0FF; font-weight: 800; color: white; }
QPushButton#step2 { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #DB2777, stop:1 #FB7185); border: 1px solid #FDA4AF; font-weight: 800; color: white; }
QPushButton#step3 { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #F59E0B, stop:1 #FCD34D); border: 1px solid #FDE68A; font-weight: 800; color: #2A1A00; }
QPushButton#step1:hover, QPushButton#step2:hover, QPushButton#step3:hover { border-color: #FFFFFF; }
QPushButton#step1:disabled, QPushButton#step2:disabled, QPushButton#step3:disabled { background: #1A1429; color: #6C6286; border-color: #2A2140; }
QPushButton#play { background: #F43F5E; border: 1px solid #FDA4AF; font-weight: 800; color: white; border-radius: 16px; padding: 6px 16px; }
QPushButton#play:hover { background: #FB7185; }
QPushButton#export { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #10B981, stop:1 #2DD4BF); border: 1px solid #99F6E4; color: #01221A; font-weight: 800; }
QPushButton#export:hover { border-color: #FFFFFF; }
QPushButton#export:disabled { background: #130F1E; color: #6C6286; border-color: #231B36; }
QLineEdit, QComboBox, QSpinBox { background: #100C1B; border: 1px solid #3F3260; border-radius: 8px; padding: 6px 10px; }
QLineEdit:focus, QComboBox:focus { border-color: #F5B83D; }
QComboBox QAbstractItemView { background: #151021; selection-background-color: #5B21B6; }
QTableWidget { background: #0F0B19; gridline-color: #231B36; border: 1px solid #2E2347; border-radius: 12px;
  selection-background-color: #4C1D95; alternate-background-color: #130E20; }
QHeaderView::section { background: #1A1429; color: #F5B83D; border: none; border-bottom: 2px solid #3B2D5C; padding: 8px; font-weight: 700; }
QProgressBar { background: #100C1B; border: 1px solid #2E2347; border-radius: 8px; text-align: center; height: 16px; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #A855F7, stop:0.5 #FB7185, stop:1 #FCD34D); border-radius: 7px; }
QSlider::groove:horizontal { height: 6px; background: #2A2140; border-radius: 3px; }
QSlider::handle:horizontal { background: #F5B83D; width: 16px; margin: -6px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: #FB7185; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #0F0B19; width: 10px; }
QScrollBar::handle:vertical { background: #3B2D5C; border-radius: 5px; min-height: 24px; }
QCheckBox::indicator { width: 16px; height: 16px; }
"""

SPEAKER_COLORS = {"ប្រុស": "#2563EB", "ស្រី": "#DB2777", "អ្នកនាំរឿង": "#7C3AED"}
GEMINI_MODELS = ["gemini-flash-latest", "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-2.5-flash"]
SOURCE_LANG_LABELS = list(E.SOURCE_LANGS.keys())


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class Worker(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[Callable[[float, str], None]], object]):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            res = self.fn(lambda p, t: self.progress.emit(float(p), str(t)))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.done.emit(res)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
class BrandLogo(QWidget):
    """រូបសញ្ញា BUNYIM: ការ៉េមូលពណ៌ gradient + ត្រីកោណ play + រលកសំឡេង"""

    def __init__(self, size: int = 54):
        super().__init__()
        self._s = size
        self.setFixedSize(size, size)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        s = float(self._s)
        g = QLinearGradient(0, 0, s, s)
        g.setColorAt(0.0, QColor("#FCD34D"))
        g.setColorAt(0.5, QColor("#FB7185"))
        g.setColorAt(1.0, QColor("#7C3AED"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRoundedRect(QRectF(0, 0, s, s), s * 0.28, s * 0.28)
        p.setBrush(QColor("#FFFFFF"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.30, s * 0.24), QPointF(s * 0.30, s * 0.76), QPointF(s * 0.66, s * 0.50)]))
        p.setBrush(QColor(255, 255, 255, 200))
        for i, h in enumerate((0.16, 0.30, 0.20)):
            x = s * (0.72 + i * 0.07)
            p.drawRoundedRect(QRectF(x, s * (0.5 - h / 2), s * 0.035, s * h), 2, 2)
        p.end()


class Timeline(QWidget):
    seek = Signal(float)
    LEFT = 96
    TOP = 26
    TRACK_H = 46
    GAP = 8

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(self.TOP + 3 * (self.TRACK_H + self.GAP) + 10)
        self.total = 60.0
        self.pos = 0.0
        self.df: Optional[pd.DataFrame] = None
        self.env: Optional[np.ndarray] = None
        self.media_name = ""
        self.view_start = 0.0
        self.view_len = 60.0
        self._drag = False
        self._pan_x: Optional[float] = None

    # ---- data
    def set_total(self, t: float):
        self.total = max(float(t), 1.0)
        self.view_start = 0.0
        self.view_len = self.total
        self.update()

    def set_df(self, df):
        self.df = df
        self.update()

    def set_env(self, env, name: str = ""):
        self.env = env
        self.media_name = name
        self.update()

    def set_pos(self, t: float):
        self.pos = float(t)
        if self.pos > self.view_start + self.view_len or self.pos < self.view_start:
            self.view_start = min(max(0.0, self.pos - self.view_len * 0.1), max(0.0, self.total - self.view_len))
        self.update()

    # ---- geometry
    def _w(self) -> float:
        return max(10.0, self.width() - self.LEFT - 8)

    def _x(self, t: float) -> float:
        return self.LEFT + (t - self.view_start) / self.view_len * self._w()

    def _t(self, x: float) -> float:
        return self.view_start + (x - self.LEFT) / self._w() * self.view_len

    def _track_top(self, i: int) -> float:
        return self.TOP + i * (self.TRACK_H + self.GAP)

    # ---- paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0F0B19"))
        fm = QFontMetrics(p.font())

        # ruler
        pps = self._w() / self.view_len
        step = next((s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600) if s * pps >= 80), 3600)
        t = (int(self.view_start // step)) * step
        p.setPen(QPen(QColor("#3B2D5C"), 1))
        while t <= self.view_start + self.view_len:
            x = self._x(t)
            if self.LEFT <= x <= self.width():
                p.drawLine(int(x), self.TOP - 8, int(x), self.TOP - 2)
                p.setPen(QColor("#A99CC8"))
                p.drawText(int(x) + 3, self.TOP - 10, E.fmt_clock(t))
                p.setPen(QPen(QColor("#3B2D5C"), 1))
            t += step

        labels = ["វីដេអូ", "សំឡេងដើម", "សំឡេងខ្មែរ AI"]
        label_colors = ["#C4A0FF", "#FB7185", "#FCD34D"]
        for i in range(3):
            top = self._track_top(i)
            p.setPen(QColor(label_colors[i]))
            p.drawText(QRectF(8, top, self.LEFT - 12, self.TRACK_H), Qt.AlignmentFlag.AlignVCenter.value, labels[i])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#171126"))
            p.drawRoundedRect(QRectF(self.LEFT, top, self._w(), self.TRACK_H), 6, 6)

        left, right = float(self.LEFT), float(self.LEFT) + self._w()

        # video track
        top = self._track_top(0)
        x0, x1 = max(left, self._x(0.0)), min(right, self._x(self.total))
        if x1 > x0:
            p.setBrush(QColor("#3B1F6E"))
            p.setPen(QPen(QColor("#6D45B8"), 1))
            p.drawRoundedRect(QRectF(x0, top + 4, x1 - x0, self.TRACK_H - 8), 5, 5)
            p.setPen(QColor("#E9D8FF"))
            p.drawText(QRectF(x0 + 8, top, x1 - x0 - 12, self.TRACK_H), Qt.AlignmentFlag.AlignVCenter.value,
                       fm.elidedText(self.media_name or "(មិនទាន់មានវីដេអូ)", Qt.TextElideMode.ElideRight, int(x1 - x0 - 16)))

        # original waveform
        top = self._track_top(1)
        if self.env is not None and len(self.env) > 1:
            n = len(self.env)
            mid = top + self.TRACK_H / 2
            half = self.TRACK_H / 2 - 4
            p.setPen(QPen(QColor("#FB7185"), 1))
            px = int(max(left, self._x(0.0)))
            end_px = int(min(right, self._x(self.total)))
            for x in range(px, end_px):
                t0, t1 = self._t(x), self._t(x + 1)
                i0 = int(min(max(t0 / self.total * n, 0), n - 1))
                i1 = int(min(max(t1 / self.total * n, i0 + 1), n))
                h = float(self.env[i0:i1].max()) * half
                p.drawLine(x, int(mid - h), x, int(mid + h))

        # khmer dub blocks
        top = self._track_top(2)
        if self.df is not None and len(self.df):
            for _, r in self.df.iterrows():
                if not str(r["khmer"]).strip():
                    continue
                xs, xe = self._x(float(r["start"])), self._x(float(r["end"]))
                if xe < left or xs > right:
                    continue
                xs, xe = max(xs, left), min(max(xe, xs + 3), right)
                p.setBrush(QColor(SPEAKER_COLORS.get(str(r["speaker"]), "#047857")))
                p.setPen(QPen(QColor(255, 255, 255, 70), 1))
                p.drawRoundedRect(QRectF(xs, top + 4, xe - xs, self.TRACK_H - 8), 4, 4)
                if xe - xs > 24:
                    p.setPen(QColor("#FFFFFF"))
                    p.drawText(QRectF(xs + 4, top, xe - xs - 6, self.TRACK_H), Qt.AlignmentFlag.AlignVCenter.value,
                               fm.elidedText(str(r["khmer"]), Qt.TextElideMode.ElideRight, int(xe - xs - 8)))

        # playhead
        x = self._x(self.pos)
        if left <= x <= right:
            p.setPen(QPen(QColor("#FCD34D"), 2))
            p.drawLine(int(x), self.TOP - 4, int(x), self.height() - 4)
            p.setBrush(QColor("#FCD34D"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([QPointF(x - 6, self.TOP - 12), QPointF(x + 6, self.TOP - 12), QPointF(x, self.TOP - 2)]))
        p.end()

    # ---- mouse
    def _seek_x(self, x: float):
        self.seek.emit(float(min(max(self._t(x), 0.0), self.total)))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self._seek_x(ev.position().x())
        else:
            self._pan_x = ev.position().x()

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._seek_x(ev.position().x())
        elif self._pan_x is not None:
            dx = ev.position().x() - self._pan_x
            self._pan_x = ev.position().x()
            self.view_start = min(max(0.0, self.view_start - dx / self._w() * self.view_len),
                                  max(0.0, self.total - self.view_len))
            self.update()

    def mouseReleaseEvent(self, _ev):
        self._drag = False
        self._pan_x = None

    def wheelEvent(self, ev):
        mouse_t = self._t(ev.position().x())
        factor = 0.8 if ev.angleDelta().y() > 0 else 1.25
        new_len = min(max(self.view_len * factor, 3.0), self.total)
        ratio = (mouse_t - self.view_start) / self.view_len
        self.view_len = new_len
        self.view_start = min(max(0.0, mouse_t - ratio * new_len), max(0.0, self.total - new_len))
        self.update()

    def zoom(self, factor: float):
        centre = self.view_start + self.view_len / 2
        new_len = min(max(self.view_len * factor, 3.0), self.total)
        self.view_len = new_len
        self.view_start = min(max(0.0, centre - new_len / 2), max(0.0, self.total - new_len))
        self.update()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
COL_NUM, COL_START, COL_END, COL_SPK, COL_KH, COL_ORIG = range(6)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BUNYIM Dubbing Studio")
        self.resize(1480, 900)
        self.cfg = E.load_config()
        self.workdir = Path(tempfile.mkdtemp(prefix="mts_"))
        self.df = E.empty_df()
        self.media: Optional[Path] = None
        self.duration = 0.0
        self.dub_path: Optional[Path] = None
        self.worker: Optional[Worker] = None
        self._loading = False
        self._seeking = False
        self._last_row = -1
        self._build_ui()
        self._init_player()
        self._apply_config()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        # header
        header = QFrame()
        header.setObjectName("header")
        head = QHBoxLayout(header)
        head.setContentsMargins(14, 10, 14, 10)
        head.setSpacing(10)
        head.addWidget(BrandLogo(54))
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        brand = QLabel('<span style="font-size:26px;font-weight:900;color:#FCD34D">BUNYIM</span>'
                       '<span style="font-size:26px;font-weight:300;color:#F5EEFF">  DUBBING STUDIO</span>')
        sub = QLabel("កម្មវិធី AI បកប្រែ និងបញ្ចូលសំឡេងខ្មែរលើរឿង និងវីដេអូ")
        sub.setObjectName("sub")
        title_box.addWidget(brand)
        title_box.addWidget(sub)
        head.addLayout(title_box)
        head.addStretch(1)
        self.btn_open = QPushButton("📂 បើកវីដេអូ")
        self.btn_open.clicked.connect(self.open_media)
        self.btn_srt_in = QPushButton("⬆ នាំចូល SRT")
        self.btn_srt_in.clicked.connect(self.import_srt)
        self.btn_srt_out = QPushButton("⬇ ទាញ SRT ខ្មែរ")
        self.btn_srt_out.clicked.connect(self.export_srt)
        self.btn_proj_save = QPushButton("💾 រក្សាទុក")
        self.btn_proj_save.clicked.connect(self.save_project)
        self.btn_proj_open = QPushButton("📁 បើក Project")
        self.btn_proj_open.clicked.connect(self.open_project)
        self.btn_export = QPushButton("🚀 Export វីដេអូ")
        self.btn_export.setObjectName("export")
        self.btn_export.clicked.connect(self.export_video)
        for b in (self.btn_open, self.btn_srt_in, self.btn_srt_out, self.btn_proj_save, self.btn_proj_open, self.btn_export):
            head.addWidget(b)
        outer.addWidget(header)

        # middle: left player | centre table | right settings
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)

        # ---- left: player
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.video = QVideoWidget()
        self.video.setMinimumSize(300, 260)
        lv.addWidget(self.video, 1)
        self.sub_label = QLabel("[ មិនទាន់មានអត្ថបទ ]")
        self.sub_label.setObjectName("subtitle")
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_label.setWordWrap(True)
        lv.addWidget(self.sub_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.slider.sliderReleased.connect(self._slider_released)
        self.slider.sliderMoved.connect(lambda v: self.seek_to(v / 1000.0))
        lv.addWidget(self.slider)
        ctl = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setObjectName("play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.chk_dub = QCheckBox("ស្តាប់សំឡេងខ្មែរ")
        self.chk_dub.setChecked(True)
        self.chk_dub.toggled.connect(self._update_volumes)
        ctl.addWidget(self.btn_play)
        ctl.addWidget(self.time_label, 1)
        ctl.addWidget(self.chk_dub)
        lv.addLayout(ctl)
        self.meta_label = QLabel("ឯកសារ: -")
        self.meta_label.setObjectName("sub")
        self.meta_label.setWordWrap(True)
        lv.addWidget(self.meta_label)
        split.addWidget(left)

        # ---- centre: actions + table
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        self.btn_transcribe = QPushButton("①  ចាប់សំឡេង")
        self.btn_transcribe.setObjectName("step1")
        self.btn_transcribe.clicked.connect(self.do_transcribe)
        self.btn_translate = QPushButton("②  បកប្រែជាខ្មែរ")
        self.btn_translate.setObjectName("step2")
        self.btn_translate.clicked.connect(self.do_translate)
        self.btn_gen = QPushButton("③  បង្កើតសំឡេងខ្មែរ")
        self.btn_gen.setObjectName("step3")
        self.btn_gen.clicked.connect(self.do_generate)
        self.btn_test = QPushButton("🎧 សាកល្បងសំឡេង")
        self.btn_test.clicked.connect(self.do_test_voice)
        for b in (self.btn_transcribe, self.btn_translate, self.btn_gen, self.btn_test):
            actions.addWidget(b)
        cv.addLayout(actions)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "ចាប់ផ្តើម", "បញ្ចប់", "តួអង្គ", "អត្ថបទបកប្រែខ្មែរ (ស្គ្រីបសំឡេង)", "អត្ថបទដើម"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_NUM, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_START, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_END, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_SPK, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_KH, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(COL_ORIG, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        cv.addWidget(self.table, 1)

        rowbar = QHBoxLayout()
        self.btn_add = QPushButton("➕ បន្ថែមជួរ (ត្រង់ playhead)")
        self.btn_add.clicked.connect(self.add_row)
        self.btn_del = QPushButton("🗑 លុបជួរដែលជ្រើស")
        self.btn_del.clicked.connect(self.delete_rows)
        self.count_label = QLabel("សរុប 0 ឃ្លា")
        self.count_label.setObjectName("sub")
        rowbar.addWidget(self.btn_add)
        rowbar.addWidget(self.btn_del)
        rowbar.addStretch(1)
        rowbar.addWidget(self.count_label)
        cv.addLayout(rowbar)
        split.addWidget(centre)

        # ---- right: settings
        split.addWidget(self._build_settings())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 2)
        split.setSizes([420, 760, 320])

        # bottom: timeline
        tl_head = QHBoxLayout()
        tl_head.addWidget(QLabel("〰 Audio Timeline Multi-Track"))
        tl_head.addStretch(1)
        zin = QPushButton("＋")
        zout = QPushButton("－")
        zin.setFixedWidth(36)
        zout.setFixedWidth(36)
        tl_head.addWidget(QLabel("Zoom:"))
        tl_head.addWidget(zout)
        tl_head.addWidget(zin)
        outer.addLayout(tl_head)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek_to)
        zin.clicked.connect(lambda: self.timeline.zoom(0.7))
        zout.clicked.connect(lambda: self.timeline.zoom(1.4))
        outer.addWidget(self.timeline)

        # status
        st = QHBoxLayout()
        self.status = QLabel("ត្រៀមខ្លួន។ ចុច «បើកវីដេអូ» ដើម្បីចាប់ផ្តើម។")
        self.status.setObjectName("sub")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setFixedWidth(320)
        st.addWidget(self.status, 1)
        st.addWidget(self.bar)
        outer.addLayout(st)

    def _build_settings(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 4, 0)

        g1 = QGroupBox("ការចាប់សំឡេង (Whisper លើកុំព្យូទ័រ)")
        f1 = QFormLayout(g1)
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(SOURCE_LANG_LABELS)
        self.cmb_whisper = QComboBox()
        self.cmb_whisper.addItems(E.WHISPER_MODELS)
        self.cmb_whisper.setCurrentText("small")
        f1.addRow("ភាសាដើម", self.cmb_lang)
        f1.addRow("ម៉ូដែល", self.cmb_whisper)
        hint = QLabel("tiny/base លឿន · small ល្អ · medium/large ច្បាស់ជាង តែយឺតលើ CPU។ លើកដំបូងត្រូវទាញម៉ូដែល។")
        hint.setObjectName("sub")
        hint.setWordWrap(True)
        f1.addRow(hint)
        v.addWidget(g1)

        g2 = QGroupBox("ការបកប្រែ")
        f2 = QFormLayout(g2)
        self.ed_gemini = QLineEdit()
        self.ed_gemini.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_gemini.setPlaceholderText("AIza... (ជម្រើស)")
        self.cmb_gmodel = QComboBox()
        self.cmb_gmodel.setEditable(True)
        self.cmb_gmodel.addItems(GEMINI_MODELS)
        self.chk_google = QCheckBox("ប្រើ Google Translate បើ Gemini បរាជ័យ/គ្មាន key")
        self.chk_google.setChecked(True)
        f2.addRow("Gemini Key", self.ed_gemini)
        f2.addRow("ម៉ូដែល", self.cmb_gmodel)
        f2.addRow(self.chk_google)
        v.addWidget(g2)

        g3 = QGroupBox("សំឡេងខ្មែរ (Edge TTS)")
        f3 = QFormLayout(g3)
        self.cmb_voice = {}
        for key, label, idx in (("male", "♂ តួប្រុស", 0), ("female", "♀ តួស្រី", 1), ("narrator", "🎤 អ្នកនាំរឿង", 0)):
            c = QComboBox()
            for vid, name in E.EDGE_VOICES.items():
                c.addItem(name, vid)
            c.setCurrentIndex(idx)
            self.cmb_voice[key] = c
            f3.addRow(label, c)
        self.sl_rate = self._slider(-30, 50, 0)
        self.sl_pitch = self._slider(-30, 30, -10)
        self.sl_speedup = self._slider(100, 200, 140)
        self.sl_orig = self._slider(0, 100, 12)
        self.sl_orig.valueChanged.connect(self._update_volumes)
        self.lb_rate = QLabel()
        self.lb_pitch = QLabel()
        self.lb_speedup = QLabel()
        self.lb_orig = QLabel()
        f3.addRow("ល្បឿននិយាយ", self._with_label(self.sl_rate, self.lb_rate, lambda x: f"{x:+d}%"))
        f3.addRow("Pitch អ្នកនាំរឿង", self._with_label(self.sl_pitch, self.lb_pitch, lambda x: f"{x:+d}Hz"))
        f3.addRow("បង្កើនល្បឿនអតិបរមា", self._with_label(self.sl_speedup, self.lb_speedup, lambda x: f"{x / 100:.2f}x"))
        f3.addRow("សំឡេងដើម (ខាងក្រោយ)", self._with_label(self.sl_orig, self.lb_orig, lambda x: f"{x}%"))
        v.addWidget(g3)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        scroll.setMinimumWidth(300)
        return scroll

    def _slider(self, lo: int, hi: int, val: int) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def _with_label(self, slider: QSlider, label: QLabel, fmt) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        label.setMinimumWidth(48)
        h.addWidget(label)
        slider.valueChanged.connect(lambda x: label.setText(fmt(x)))
        label.setText(fmt(slider.value()))
        return w

    # -------------------------------------------------------------- player
    def _init_player(self):
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.dub_player = QMediaPlayer()
        self.dub_audio = QAudioOutput()
        self.dub_player.setAudioOutput(self.dub_audio)
        self.test_player = QMediaPlayer()
        self.test_audio = QAudioOutput()
        self.test_player.setAudioOutput(self.test_audio)
        self.player.positionChanged.connect(self.on_position)
        self.player.durationChanged.connect(self.on_duration)
        self.player.playbackStateChanged.connect(self.on_state)
        self._update_volumes()

    def _dub_active(self) -> bool:
        return self.dub_path is not None and self.chk_dub.isChecked()

    def _update_volumes(self, *_):
        if self._dub_active():
            self.audio.setVolume(self.sl_orig.value() / 100.0)
            self.dub_audio.setVolume(1.0)
        else:
            self.audio.setVolume(1.0)
            self.dub_audio.setVolume(0.0)

    def toggle_play(self):
        if self.media is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def on_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.btn_play.setText("⏸ Pause" if playing else "▶ Play")
        if self.dub_path is None:
            return
        if playing:
            self.dub_player.setPosition(self.player.position())
            self.dub_player.play()
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.dub_player.pause()
        else:
            self.dub_player.stop()

    def on_duration(self, ms: int):
        self.slider.setRange(0, int(ms))

    def _slider_released(self):
        self._seeking = False
        self.seek_to(self.slider.value() / 1000.0)

    def seek_to(self, sec: float):
        ms = int(max(0.0, sec) * 1000)
        self.player.setPosition(ms)
        if self.dub_path is not None:
            self.dub_player.setPosition(ms)
        self.on_position(ms)

    def on_position(self, ms: int):
        t = ms / 1000.0
        if not self._seeking:
            self.slider.setValue(int(ms))
        self.time_label.setText(f"{E.fmt_ts(t)} / {E.fmt_ts(self.duration)}")
        self.timeline.set_pos(t)
        # drift correction
        if (self.dub_path is not None
                and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
                and abs(self.dub_player.position() - ms) > 300):
            self.dub_player.setPosition(ms)
        # subtitle + highlight
        text, row = "", -1
        if len(self.df):
            hit = self.df[(self.df["start"] <= t) & (self.df["end"] >= t)]
            if len(hit):
                row = int(hit.index[0])
                text = str(hit.iloc[0]["khmer"]) or str(hit.iloc[0]["original"])
        self.sub_label.setText(text if text else " ")
        if (row >= 0 and row != self._last_row
                and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState):
            self._last_row = row
            self._loading = True
            self.table.selectRow(row)
            self._loading = False

    # -------------------------------------------------------------- config
    def _apply_config(self):
        c = self.cfg
        self.ed_gemini.setText(c.get("gemini_key", ""))
        if c.get("gemini_model"):
            self.cmb_gmodel.setCurrentText(c["gemini_model"])
        if c.get("whisper_model") in E.WHISPER_MODELS:
            self.cmb_whisper.setCurrentText(c["whisper_model"])
        if c.get("lang") in SOURCE_LANG_LABELS:
            self.cmb_lang.setCurrentText(c["lang"])
        self.chk_google.setChecked(bool(c.get("google_fallback", True)))
        for key, combo in self.cmb_voice.items():
            i = combo.findData(c.get(f"voice_{key}"))
            if i >= 0:
                combo.setCurrentIndex(i)
        for name, sl in (("rate", self.sl_rate), ("pitch", self.sl_pitch), ("speedup", self.sl_speedup), ("orig", self.sl_orig)):
            if isinstance(c.get(name), int):
                sl.setValue(c[name])

    def _collect_config(self) -> dict:
        c = {
            "gemini_key": self.ed_gemini.text().strip(),
            "gemini_model": self.cmb_gmodel.currentText().strip(),
            "whisper_model": self.cmb_whisper.currentText(),
            "lang": self.cmb_lang.currentText(),
            "google_fallback": self.chk_google.isChecked(),
            "rate": self.sl_rate.value(), "pitch": self.sl_pitch.value(),
            "speedup": self.sl_speedup.value(), "orig": self.sl_orig.value(),
            "last_dir": self.cfg.get("last_dir", ""),
        }
        for key, combo in self.cmb_voice.items():
            c[f"voice_{key}"] = combo.currentData()
        return c

    def closeEvent(self, ev):
        self.cfg = self._collect_config()
        E.save_config(self.cfg)
        try:
            self.player.stop()
            self.dub_player.stop()
            if self.worker is not None and self.worker.isRunning():
                self.worker.requestInterruption()
                if not self.worker.wait(1500):
                    self.worker.terminate()
                    self.worker.wait(1000)
        except Exception:
            pass
        super().closeEvent(ev)

    # ------------------------------------------------------------- helpers
    def _lang_name(self) -> str:
        return E.SOURCE_LANGS[self.cmb_lang.currentText()]

    def _voices(self) -> dict:
        return {k: c.currentData() for k, c in self.cmb_voice.items()}

    def _start_dir(self) -> str:
        return self.cfg.get("last_dir", "") or str(Path.home())

    def _remember_dir(self, path: str):
        self.cfg["last_dir"] = str(Path(path).parent)

    def warn(self, text: str, title: str = "ចំណាំ"):
        QMessageBox.warning(self, title, text)

    def set_status(self, text: str, frac: Optional[float] = None):
        self.status.setText(text)
        if frac is not None:
            self.bar.setValue(int(min(max(frac, 0.0), 1.0) * 1000))

    def set_busy(self, busy: bool):
        for b in (self.btn_open, self.btn_srt_in, self.btn_proj_open, self.btn_transcribe,
                  self.btn_translate, self.btn_gen, self.btn_test, self.btn_export):
            b.setEnabled(not busy)

    def run_task(self, fn, on_done, label: str):
        if self.worker is not None and self.worker.isRunning():
            self.warn("កំពុងដំណើរការការងារមួយទៀត សូមរង់ចាំ។")
            return
        self.set_busy(True)
        self.set_status(label, 0.0)
        w = Worker(fn)
        w.progress.connect(lambda p, t: self.set_status(t, p))

        def finished(res):
            self.set_busy(False)
            self.bar.setValue(1000)
            on_done(res)

        def failed(msg):
            self.set_busy(False)
            self.bar.setValue(0)
            self.set_status("មានបញ្ហា: " + msg[:200])
            self.warn(msg, "មានបញ្ហា")

        w.done.connect(finished)
        w.failed.connect(failed)
        self.worker = w
        w.start()

    # ---------------------------------------------------------------- table
    def refresh_table(self):
        self._loading = True
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.df))
        for r, row in self.df.iterrows():
            num = QTableWidgetItem(str(r + 1))
            num.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(r, COL_NUM, num)
            self.table.setItem(r, COL_START, QTableWidgetItem(E.fmt_ts(row["start"])))
            self.table.setItem(r, COL_END, QTableWidgetItem(E.fmt_ts(row["end"])))
            combo = QComboBox()
            combo.addItems(list(E.SPEAKER_KEYS))
            combo.setCurrentText(str(row["speaker"]))
            combo.currentTextChanged.connect(lambda txt, rr=r: self.set_speaker(rr, txt))
            self.table.setCellWidget(r, COL_SPK, combo)
            self.table.setItem(r, COL_KH, QTableWidgetItem(str(row["khmer"])))
            self.table.setItem(r, COL_ORIG, QTableWidgetItem(str(row["original"])))
        self.table.blockSignals(False)
        self.table.resizeRowsToContents()
        self._loading = False
        self.count_label.setText(f"សរុប {len(self.df)} ឃ្លា · មានអត្ថបទខ្មែរ {self._khmer_count()} ឃ្លា")
        self.timeline.set_df(self.df)

    def _khmer_count(self) -> int:
        if not len(self.df):
            return 0
        return int((self.df["khmer"].astype(str).str.strip() != "").sum())

    def set_speaker(self, row: int, text: str):
        if 0 <= row < len(self.df) and text in E.SPEAKER_KEYS:
            self.df.at[row, "speaker"] = text
            self.timeline.set_df(self.df)

    def on_item_changed(self, item):
        if self._loading:
            return
        r, c = item.row(), item.column()
        if not (0 <= r < len(self.df)):
            return
        if c in (COL_START, COL_END):
            col = "start" if c == COL_START else "end"
            try:
                val = E.parse_ts(item.text())
            except ValueError:
                self._loading = True
                item.setText(E.fmt_ts(self.df.at[r, col]))
                self._loading = False
                self.set_status("ពេលវេលាមិនត្រឹមត្រូវ (ឧ. 01:23.5)")
                return
            self.df.at[r, col] = val
            if self.df.at[r, "end"] <= self.df.at[r, "start"]:
                self.df.at[r, "end"] = self.df.at[r, "start"] + 0.3
        elif c == COL_KH:
            self.df.at[r, "khmer"] = item.text().strip()
        elif c == COL_ORIG:
            self.df.at[r, "original"] = item.text().strip()
        self.count_label.setText(f"សរុប {len(self.df)} ឃ្លា · មានអត្ថបទខ្មែរ {self._khmer_count()} ឃ្លា")
        self.timeline.set_df(self.df)

    def on_cell_double_clicked(self, row: int, _col: int):
        if 0 <= row < len(self.df):
            self.seek_to(float(self.df.at[row, "start"]))

    def add_row(self):
        t = self.player.position() / 1000.0
        new = pd.DataFrame([{"start": t, "end": t + 2.0, "speaker": E.DEFAULT_SPEAKER,
                             "original": "", "khmer": "ឃ្លាថ្មី"}], columns=E.COLUMNS)
        self.df = pd.concat([self.df, new], ignore_index=True).sort_values("start", kind="stable").reset_index(drop=True)
        self.refresh_table()

    def delete_rows(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows:
            self.warn("សូមជ្រើសជួរក្នុងតារាងជាមុនសិន។")
            return
        self.df = self.df.drop(index=rows).reset_index(drop=True)
        self.refresh_table()

    # ---------------------------------------------------------------- media
    def open_media(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "បើកវីដេអូ ឬ Audio", self._start_dir(),
            "Media (*.mp4 *.mkv *.mov *.ts *.avi *.webm *.mp3 *.wav *.m4a *.aac *.ogg);;All files (*)")
        if path:
            self._remember_dir(path)
            self.load_media(Path(path))

    def load_media(self, path: Path):
        if not E.ffmpeg_available():
            self.warn("រកមិនឃើញ ffmpeg ទេ។ សូមដំឡើង ffmpeg ឬប្រើកំណែ .exe ដែលមាន ffmpeg ភ្ជាប់មកជាមួយ។")
            return
        dur = E.probe_duration(path)
        if dur <= 0:
            self.warn("អានឯកសារនេះមិនបាន។")
            return
        self.media = path
        self.duration = dur
        self.dub_path = None
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.dub_player.setSource(QUrl())
        self._update_volumes()
        self.timeline.set_total(dur)
        self.timeline.set_env(None, path.name)
        size = E.video_size(path) if E.has_video(path) else "(Audio)"
        self.meta_label.setText(f"ឯកសារ: {path.name}\nរយៈពេល: {E.fmt_clock(dur)} · {size}")
        self.setWindowTitle(f"BUNYIM Dubbing Studio - {path.name}")
        self.set_status("កំពុងគូរ waveform...", 0.0)

        def work(_p):
            return E.waveform_envelope(path)

        def done(env):
            self.timeline.set_env(env, path.name)
            self.set_status("ត្រៀមខ្លួន។ ចុច «① ចាប់សំឡេង» ឬនាំចូល SRT។", 0.0)

        self.run_task(work, done, "កំពុងគូរ waveform...")

    # -------------------------------------------------------------- actions
    def do_transcribe(self):
        if self.media is None:
            self.warn("សូមបើកវីដេអូជាមុនសិន។")
            return
        if len(self.df) and QMessageBox.question(self, "បញ្ជាក់", "តារាងមានអត្ថបទរួចហើយ។ ជំនួសដោយការចាប់សំឡេងថ្មី?") != QMessageBox.StandardButton.Yes:
            return
        media, lang, model = self.media, self._lang_name(), self.cmb_whisper.currentText()

        def work(prog):
            return E.whisper_local(media, lang, self.workdir, model, progress=prog)

        def done(df):
            if df.empty:
                self.warn("រកមិនឃើញសំឡេងនិយាយក្នុងឯកសារនេះទេ។")
                return
            self.df = df
            self.refresh_table()
            self.set_status(f"ចាប់បាន {len(df)} ឃ្លា។ ចុច «② បកប្រែជាខ្មែរ»។", 1.0)

        self.run_task(work, done, "កំពុងចាប់សំឡេង...")

    def do_translate(self):
        idx = [i for i in self.df.index if str(self.df.at[i, "original"]).strip()]
        if not idx:
            self.warn("មិនមានអត្ថបទដើមសម្រាប់បកប្រែទេ។ សូមចាប់សំឡេង ឬនាំចូល SRT ជាមុន។")
            return
        lines = [str(self.df.at[i, "original"]).strip() for i in idx]
        key = self.ed_gemini.text().strip()
        if key.startswith("gsk_"):
            self.warn("ប្រអប់ Gemini ដាក់ key របស់ Groq (gsk_...) ខុសហើយ។ Gemini key ចាប់ផ្តើមដោយ AIza...")
            return
        model = self.cmb_gmodel.currentText().strip() or GEMINI_MODELS[0]
        models = [model] + [m for m in GEMINI_MODELS if m != model]
        use_google = self.chk_google.isChecked()
        lang = self._lang_name()

        def work(prog):
            client = E.get_client(key) if key else None
            res, engine_name = E.translate_smart(client, models, lines, lang, use_google_fallback=use_google, progress=prog)
            return idx, res, engine_name

        def done(out):
            ids, res, engine_name = out
            for i, t in zip(ids, res):
                self.df.at[i, "khmer"] = t
            self.refresh_table()
            self.set_status(f"បកប្រែរួច {len(ids)} ឃ្លា (ដោយ {engine_name})។ ពិនិត្យ/កែ រួចចុច «③ បង្កើតសំឡេងខ្មែរ»។", 1.0)

        self.run_task(work, done, "កំពុងបកប្រែ...")

    def _dub_args(self):
        return (self._voices(), self.sl_rate.value(), self.sl_pitch.value(), self.sl_speedup.value() / 100.0)

    def do_generate(self):
        if self.media is None:
            self.warn("សូមបើកវីដេអូជាមុនសិន។")
            return
        if self._khmer_count() == 0:
            self.warn("មិនទាន់មានអត្ថបទខ្មែរ។ សូមចុច «② បកប្រែជាខ្មែរ» ជាមុន។")
            return
        df = self.df.copy()
        voices, rate, pitch, speedup = self._dub_args()
        dur = self.duration
        wav = self.workdir / "khmer_dub.wav"

        def work(prog):
            E.build_dub_track(df, wav, self.workdir / "tts_cache", voices, rate, pitch, speedup, dur, progress=prog)
            return wav

        def done(path):
            self.dub_path = Path(path)
            self.dub_player.setSource(QUrl.fromLocalFile(str(path)))
            self.chk_dub.setChecked(True)
            self._update_volumes()
            self.set_status("បង្កើតសំឡេងខ្មែររួចរាល់។ ចុច Play ដើម្បីស្តាប់ ឬ Export។", 1.0)

        self.run_task(work, done, "កំពុងបង្កើតសំឡេងខ្មែរ...")

    def do_test_voice(self):
        rows = self.df[self.df["khmer"].astype(str).str.strip() != ""]
        text = str(rows.iloc[0]["khmer"]) if len(rows) else "សួស្តី នេះគឺជាការសាកល្បងសំឡេង"
        spk = str(rows.iloc[0]["speaker"]) if len(rows) else E.DEFAULT_SPEAKER
        key = E.SPEAKER_KEYS.get(spk, "narrator")
        voice = self._voices()[key]
        rate, pitch = self.sl_rate.value(), (self.sl_pitch.value() if key == "narrator" else 0)

        def work(_p):
            return E.tts_clip(text, voice, rate, pitch, self.workdir / "tts_cache")

        def done(path):
            self.test_audio.setVolume(1.0)
            self.test_player.setSource(QUrl.fromLocalFile(str(path)))
            self.test_player.play()
            self.set_status("កំពុងចាក់សំឡេងសាកល្បង...", 1.0)

        self.run_task(work, done, "កំពុងបង្កើតសំឡេងសាកល្បង...")

    def export_video(self):
        if self.media is None or self._khmer_count() == 0:
            self.warn("ត្រូវការវីដេអូ និងអត្ថបទខ្មែរយ៉ាងហោចណាស់ ១ ឃ្លា។")
            return
        is_video = E.has_video(self.media)
        ext = ".mp4" if is_video else ".m4a"
        default = str(Path(self._start_dir()) / (self.media.stem + "_khmer" + ext))
        path, _ = QFileDialog.getSaveFileName(self, "Export", default, "MP4 (*.mp4)" if is_video else "M4A (*.m4a)")
        if not path:
            return
        self._remember_dir(path)
        out = Path(path)
        media = self.media
        df = self.df.copy()
        voices, rate, pitch, speedup = self._dub_args()
        dur = self.duration
        orig_vol = self.sl_orig.value() / 100.0
        wav = self.workdir / "khmer_dub.wav"

        def work(prog):
            E.build_dub_track(df, wav, self.workdir / "tts_cache", voices, rate, pitch, speedup, dur,
                              progress=lambda p, t: prog(p * 0.8, t))
            prog(0.85, "កំពុងលាយសំឡេងចូលវីដេអូ...")
            E.mux_dub(media, wav, out, orig_vol)
            return out

        def done(p):
            self.dub_path = wav
            self.set_status(f"Export រួចរាល់៖ {p}", 1.0)
            QMessageBox.information(self, "រួចរាល់", f"បានរក្សាទុក៖\n{p}")

        self.run_task(work, done, "កំពុង Export...")

    # ------------------------------------------------------------ file ops
    def import_srt(self):
        path, _ = QFileDialog.getOpenFileName(self, "នាំចូល SRT", self._start_dir(), "SRT (*.srt)")
        if not path:
            return
        self._remember_dir(path)
        raw = Path(path).read_bytes()
        text = None
        for enc in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        items = E.parse_srt(text or "")
        if not items:
            self.warn("អានឯកសារ SRT មិនបាន។")
            return
        is_khmer = any("ក" <= ch <= "៿" for it in items[:5] for ch in it["text"])
        rows = [{"start": it["start"], "end": it["end"], "speaker": E.DEFAULT_SPEAKER,
                 "original": "" if is_khmer else it["text"], "khmer": it["text"] if is_khmer else ""} for it in items]
        self.df = E.clean_df(pd.DataFrame(rows, columns=E.COLUMNS))
        self.refresh_table()
        self.set_status(f"នាំចូល {len(items)} ឃ្លា" + ("" if is_khmer else " (ភាសាដើម → ចុច «② បកប្រែ»)"), 1.0)

    def export_srt(self):
        if not len(self.df):
            self.warn("តារាងទទេ។")
            return
        default = str(Path(self._start_dir()) / ((self.media.stem if self.media else "khmer") + "_khmer.srt"))
        path, _ = QFileDialog.getSaveFileName(self, "ទាញយក SRT", default, "SRT (*.srt)")
        if path:
            Path(path).write_text(E.to_srt(self.df, "khmer"), encoding="utf-8-sig")
            self.set_status(f"រក្សាទុក SRT៖ {path}", 1.0)

    def save_project(self):
        if not len(self.df):
            self.warn("តារាងទទេ។")
            return
        default = str(Path(self._start_dir()) / ((self.media.stem if self.media else "project") + ".mts.json"))
        path, _ = QFileDialog.getSaveFileName(self, "រក្សាទុក Project", default, "Project (*.mts.json *.json)")
        if path:
            E.save_project(Path(path), self.df, {"media": str(self.media) if self.media else ""})
            self.set_status(f"រក្សាទុក Project៖ {path}", 1.0)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "បើក Project", self._start_dir(), "Project (*.json)")
        if not path:
            return
        try:
            df, meta = E.load_project(Path(path))
        except Exception as e:  # noqa: BLE001
            self.warn(f"បើក Project មិនបាន: {e}")
            return
        self.df = df
        self.refresh_table()
        m = meta.get("media", "")
        if m and Path(m).exists():
            self.load_media(Path(m))
        self.set_status(f"បើក Project៖ {len(df)} ឃ្លា", 1.0)


def run():
    import sys

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
