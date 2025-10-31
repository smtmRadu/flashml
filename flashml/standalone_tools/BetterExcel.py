# datasheet_diff_app.py
# Single-file desktop app using PyQt6 and pandas.
# Run locally:
#   pip install PyQt6 pandas openpyxl
#   python datasheet_diff_app.py

from pathlib import Path
import sys
import typing as t
import difflib
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets


APP_NAME = "DataSheet & Diff — Single File App"


# -------------------- Utility: Dark/Light palettes & styles --------------------
def apply_dark_palette(app: QtWidgets.QApplication):
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(22, 22, 22))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(0, 122, 204))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(60, 100, 160))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.white)
    app.setPalette(palette)
    app.setStyle("Fusion")


def apply_light_palette(app: QtWidgets.QApplication):
    app.setPalette(app.style().standardPalette())
    app.setStyle("Fusion")


# Subtle column zebra shading
def column_shade_color(is_dark: bool, index: int) -> QtGui.QColor:
    if is_dark:
        base = 35 if index % 2 else 40
        return QtGui.QColor(base, base, base)
    else:
        base = 246 if index % 2 else 240
        return QtGui.QColor(base, base, base)


# -------------------- Data Model for DataFrame --------------------
class DataFrameModel(QtCore.QAbstractTableModel):
    dataChangedHard = QtCore.pyqtSignal()      # for full repaint (e.g., search highlight)
    matchesChanged = QtCore.pyqtSignal(int)    # emits total number of matches

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        self.search_term: str = ""
        self.replace_term: str = ""
        self._is_dark = True
        self._matches: list[QtCore.QModelIndex] = []   # cached matches

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=None):
        return len(self._df.index)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        val = self._df.iat[r, c]

        if role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            if pd.isna(val):
                return ""
            return str(val)

        if role == QtCore.Qt.ItemDataRole.BackgroundRole:
            text = "" if pd.isna(val) else str(val).lower()

            # Row highlight: any cell in row matches?
            row_has_match = any(m.row() == r for m in self._matches)
            if row_has_match:
                return QtGui.QBrush(QtGui.QColor(255, 220, 180, 60))  # light orange

            # Cell highlight: this cell contains the term
            if self.search_term and self.search_term.lower() in text:
                return QtGui.QBrush(QtGui.QColor(255, 255, 150, 80))  # light yellow

            # Default zebra shading
            return QtGui.QBrush(column_shade_color(self._is_dark, c))

        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        return (
            QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsEditable
        )

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if role == QtCore.Qt.ItemDataRole.EditRole and index.isValid():
            r, c = index.row(), index.column()
            self._df.iat[r, c] = value
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole])
            self._rebuild_matches()  # update matches after edit
            self.matchesChanged.emit(len(self._matches))
            return True
        return False

    # --------------------------------------------------------------------- #
    #  SEARCH & MATCH LOGIC
    # --------------------------------------------------------------------- #
    def set_search_term(self, term: str):
        term = term.strip()
        if term == self.search_term:
            return
        self.search_term = term
        self._rebuild_matches()
        self.dataChangedHard.emit()
        self.matchesChanged.emit(len(self._matches))

    def _rebuild_matches(self):
        self._matches = []
        if not self.search_term:
            return
        st = self.search_term.lower()
        for r in range(self.rowCount()):
            for c in range(self.columnCount()):
                val = self._df.iat[r, c]
                if pd.isna(val):
                    continue
                if st in str(val).lower():
                    self._matches.append(self.index(r, c))

    def match_at(self, pos: int) -> t.Optional[QtCore.QModelIndex]:
        if 0 <= pos < len(self._matches):
            return self._matches[pos]
        return None

    def total_matches(self) -> int:
        return len(self._matches)

    # --------------------------------------------------------------------- #
    #  REPLACE LOGIC (unchanged from your original)
    # --------------------------------------------------------------------- #
    def replace_once(self, start_after: t.Optional[QtCore.QModelIndex]):
        if not self.search_term or self.replace_term is None:
            return None
        matches = self._matches
        if not matches:
            return None
        if start_after and start_after.isValid():
            try:
                idx = matches.index(start_after) + 1
            except ValueError:
                idx = 0
                for i, m in enumerate(matches):
                    if (m.row(), m.column()) > (start_after.row(), start_after.column()):
                        idx = i
                        break
        else:
            idx = 0
        m = matches[idx % len(matches)]
        r, c = m.row(), m.column()
        val = str(self._df.iat[r, c])
        new_val = self._replace_case_insensitive(val, self.search_term, self.replace_term, count=1)
        self._df.iat[r, c] = new_val
        self.dataChanged.emit(m, m, [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole])
        self._rebuild_matches()
        self.dataChangedHard.emit()
        self.matchesChanged.emit(len(self._matches))
        return m

    def replace_all(self) -> int:
        if not self.search_term:
            return 0
        count = 0
        st = self.search_term
        for r in range(self.rowCount()):
            for c in range(self.columnCount()):
                val = self._df.iat[r, c]
                if pd.isna(val):
                    continue
                s = str(val)
                if st.lower() in s.lower():
                    new_val = self._replace_case_insensitive(s, st, self.replace_term, count=0)
                    if new_val != s:
                        self._df.iat[r, c] = new_val
                        count += 1
        if count:
            tl = self.index(0, 0)
            br = self.index(self.rowCount() - 1, self.columnCount() - 1)
            self.dataChanged.emit(tl, br, [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole])
            self._rebuild_matches()
            self.dataChangedHard.emit()
            self.matchesChanged.emit(len(self._matches))
        return count

    @staticmethod
    def _replace_case_insensitive(text: str, pattern: str, repl: str, count: int) -> str:
        import re
        flags = re.IGNORECASE
        return re.sub(re.escape(pattern), repl, text, count=count if count else 0, flags=flags)

class HighlightDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, model: DataFrameModel):
        super().__init__()
        self.model = model

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        painter.save()

        text = index.data(QtCore.Qt.ItemDataRole.DisplayRole) or ""
        st = self.model.search_term
        has_term = st and st.lower() in text.lower()

        # Draw default item background (includes row/cell highlights from model)
        style = QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, option, painter)

        rect = option.rect.adjusted(4, 2, -4, -2)

        # Highlight only the matching word(s) in green
        if has_term:
            doc = QtGui.QTextDocument()
            text_option = QtGui.QTextOption()
            text_option.setWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            text_option.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
            doc.setDefaultTextOption(text_option)
            doc.setPlainText(text)

            lower = text.lower()
            start = 0
            st_lower = st.lower()
            highlights = []
            while True:
                i = lower.find(st_lower, start)
                if i == -1:
                    break
                highlights.append((i, len(st)))
                start = i + len(st)

            if highlights:
                doc.setTextWidth(rect.width())
                layout = doc.documentLayout()

                for pos, length in highlights:
                    cursor = QtGui.QTextCursor(doc)
                    cursor.setPosition(pos)
                    cursor.setPosition(pos + length, QtGui.QTextCursor.MoveMode.KeepAnchor)

                    block_rect = layout.blockBoundingRect(cursor.block())
                    selection_rect = QtCore.QRectF(block_rect)
                    selection_rect.translate(QtCore.QPointF(rect.topLeft()))

                    painter.fillRect(selection_rect, QtGui.QColor(0, 200, 0, 140))  # green word

        # Draw the text itself
        painter.translate(rect.topLeft())
        clip = QtCore.QRectF(0, 0, rect.width(), rect.height())

        doc = QtGui.QTextDocument()
        opt = QtGui.QTextOption()
        opt.setWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        opt.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        doc.setDefaultTextOption(opt)
        doc.setPlainText(text)
        doc.drawContents(painter, clip)

        painter.restore()

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QPlainTextEdit(parent)
        editor.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        editor.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        editor.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        editor.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return editor

    def setEditorData(self, editor, index):
        editor.setPlainText(index.data(QtCore.Qt.ItemDataRole.DisplayRole) or "")

    def setModelData(self, editor, model, index):
        model.setData(index, editor.toPlainText(), QtCore.Qt.ItemDataRole.EditRole)

class DragDropWidget(QtWidgets.QFrame):
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        # Try to call parent's open dialog when clicked
        parent = self.parent()
        if parent and hasattr(parent, 'open_file_dialog'):
            parent.open_file_dialog()
        super().mousePressEvent(e)

    fileDropped = QtCore.pyqtSignal(Path)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.icon = QtWidgets.QLabel("🗂️\\nDrop a CSV / XLS / XLSX / JSONL file here")
        self.icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.icon.setStyleSheet("font-size: 22px;")
        layout.addWidget(self.icon)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    suffix = Path(url.toLocalFile()).suffix.lower()
                    if suffix in [".csv", ".xls", ".xlsx", ".jsonl"]:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in [".csv", ".xls", ".xlsx", ".jsonl"]:
                    self.fileDropped.emit(path)
                    return


class WindowControls(QtWidgets.QWidget):
    minimizeRequested = QtCore.pyqtSignal()
    maximizeRestoreRequested = QtCore.pyqtSignal()
    closeRequested = QtCore.pyqtSignal()
    backRequested = QtCore.pyqtSignal()

    def __init__(self, title: str = ""):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.back_btn = QtWidgets.QToolButton()
        self.back_btn.setText("← Back")
        self.back_btn.clicked.connect(self.backRequested.emit)

        self.title_lbl = QtWidgets.QLabel(title)
        self.title_lbl.setStyleSheet("font-weight: 600;")

        layout.addWidget(self.back_btn)
        layout.addStretch(1)

        self.min_btn = QtWidgets.QToolButton()
        self.min_btn.setText("—")
        self.min_btn.setToolTip("Minimize")
        self.min_btn.clicked.connect(self.minimizeRequested.emit)

        self.max_btn = QtWidgets.QToolButton()
        self.max_btn.setText("▢")
        self.max_btn.setToolTip("Maximize/Restore")
        self.max_btn.clicked.connect(self.maximizeRestoreRequested.emit)

        self.close_btn = QtWidgets.QToolButton()
        self.close_btn.setText("✕")
        self.close_btn.setToolTip("Close")
        self.close_btn.clicked.connect(self.closeRequested.emit)

        layout.addWidget(self.title_lbl)
        layout.addStretch(10)
        for b in (self.min_btn, self.max_btn, self.close_btn):
            layout.addWidget(b)


class DataViewerPage(QtWidgets.QWidget):
    backRequested = QtCore.pyqtSignal()

    def __init__(self, app_ref: 'MainWindow'):
        super().__init__()
        self.setAcceptDrops(True)
        self.app_ref = app_ref
        self.current_path: t.Optional[Path] = None
        self.model: t.Optional[DataFrameModel] = None
        self.current_match_pos: int = -1

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.controls = WindowControls("Viewer")
        self.controls.backRequested.connect(self.backRequested.emit)
        self.controls.minimizeRequested.connect(lambda: self.window().showMinimized())
        self.controls.maximizeRestoreRequested.connect(self._toggle_max_restore)
        self.controls.closeRequested.connect(self.window().close)

        toolbar = QtWidgets.QToolBar()
        toolbar.setIconSize(QtCore.QSize(16, 16))

        # --- Actions ---
        self.btn_open = QtGui.QAction("Open", self)
        self.btn_open.triggered.connect(self.open_file_dialog)
        toolbar.addAction(self.btn_open)

        self.btn_save = QtGui.QAction("Save", self)
        self.btn_save.triggered.connect(self.save_file)
        toolbar.addAction(self.btn_save)

        self.btn_save_copy = QtGui.QAction("Save Copy", self)
        self.btn_save_copy.triggered.connect(self.save_copy)
        toolbar.addAction(self.btn_save_copy)

        toolbar.addSeparator()

        # --- Dark/Light Toggle ---
        self.mode_toggle = QtWidgets.QToolButton()
        self.mode_toggle.setText("Dark")
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setChecked(True)
        self.mode_toggle.toggled.connect(self.toggle_mode)
        toolbar.addWidget(self.mode_toggle)

        toolbar.addSeparator()

        # --- Search Field ---
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search keyword…")
        self.search_edit.textChanged.connect(self.on_search_changed)
        toolbar.addWidget(self.search_edit)

        # --- Navigation Buttons ---
        self.btn_prev = QtWidgets.QToolButton()
        self.btn_prev.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
        self.btn_prev.setToolTip("Previous match (Shift+Enter)")
        self.btn_prev.clicked.connect(self.go_prev_match)

        self.btn_next = QtWidgets.QToolButton()
        self.btn_next.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.btn_next.setToolTip("Next match (Enter)")
        self.btn_next.clicked.connect(self.go_next_match)

        self.match_label = QtWidgets.QLabel("Matches: 0")

        toolbar.addWidget(self.btn_prev)
        toolbar.addWidget(self.btn_next)
        toolbar.addWidget(self.match_label)

        # --- Replace Section ---
        self.replace_container = QtWidgets.QWidget()
        repl_layout = QtWidgets.QHBoxLayout(self.replace_container)
        repl_layout.setContentsMargins(0, 0, 0, 0)
        self.replace_edit = QtWidgets.QLineEdit()
        self.replace_edit.setPlaceholderText("Replace with…")
        self.replace_edit.textChanged.connect(self._update_replace_buttons)
        self.btn_replace_next = QtWidgets.QToolButton()
        self.btn_replace_next.setText("Replace Next")
        self.btn_replace_next.clicked.connect(self.on_replace_next)
        self.btn_replace_all = QtWidgets.QToolButton()
        self.btn_replace_all.setText("Replace All")
        self.btn_replace_all.clicked.connect(self.on_replace_all)
        repl_layout.addWidget(self.replace_edit)
        repl_layout.addWidget(self.btn_replace_next)
        repl_layout.addWidget(self.btn_replace_all)
        self.replace_container.setVisible(False)
        toolbar.addWidget(self.replace_container)

        toolbar.addSeparator()

        # --- Other Actions ---
        self.btn_autosize = QtGui.QAction("Autosize Columns", self)
        self.btn_autosize.triggered.connect(lambda: self.autosize_columns(force=True))
        toolbar.addAction(self.btn_autosize)

        self.btn_add_row = QtGui.QAction("Add Row", self)
        self.btn_add_row.triggered.connect(self.add_row)
        toolbar.addAction(self.btn_add_row)

        self.btn_delete_rows = QtGui.QAction("Delete Selected Rows", self)
        self.btn_delete_rows.triggered.connect(self.delete_selected_rows)
        toolbar.addAction(self.btn_delete_rows)

        # --- Table & Drop Zone ---
        self.table = QtWidgets.QTableView()
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.setWordWrap(True)
        self.table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

        self.dd = DragDropWidget()
        self.dd.fileDropped.connect(self.load_path)

        outer.addWidget(self.controls)
        outer.addWidget(toolbar)
        outer.addWidget(self.dd)
        outer.addWidget(self.table)
        self.table.hide()

        self.status = QtWidgets.QStatusBar()
        outer.addWidget(self.status)

    def _toggle_max_restore(self):
        if self.window().isMaximized() or self.window().isFullScreen():
            self.window().showNormal()
        else:
            self.window().showMaximized()

    def load_path(self, path: Path):
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, low_memory=False)  # Fixed DtypeWarning
            elif path.suffix.lower() in [".xls", ".xlsx"]:
                df = pd.read_excel(path)
            elif path.suffix.lower() == ".jsonl":
                df = pd.read_json(path, lines=True)
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported", f"Unsupported file: {path}")
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open file:\n{e}")
            return

        self.current_path = path
        self.model = DataFrameModel(df)
        self.model._is_dark = self.mode_toggle.isChecked()
        self.model.dataChangedHard.connect(self.table.viewport().update)
        self.model.matchesChanged.connect(self._on_matches_changed)

        self.table.setModel(self.model)
        self.table.setItemDelegate(HighlightDelegate(self.model))

        if self.model._is_dark:
            self.table.setStyleSheet("QTableView { gridline-color: rgb(80, 80, 80); }")
        else:
            self.table.setStyleSheet("QTableView { gridline-color: rgb(0, 0, 0); }")

        self.table.show()
        self.dd.hide()
        self.status.showMessage(f"Loaded: {path}  |  {df.shape[0]} rows × {df.shape[1]} cols")
        self.autosize_columns(force=True)
        self.autosize_rows()

        # Reset search state
        self.current_match_pos = -1
        self.model.set_search_term("")
        self._on_matches_changed(0)

    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", str(Path.home()), "Data Files (*.csv *.xls *.xlsx *.jsonl)"
        )
        if path:
            self.load_path(Path(path))

    def save_file(self):
        if not self.model or not self.current_path:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Open a file first.")
            return
        try:
            self._write_dataframe(self.current_path, self.model.df)
            self.status.showMessage(f"Saved to {self.current_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def save_copy(self):
        if not self.model:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Copy As",
            str(self.current_path or (Path.home() / "data.csv")),
            "CSV (*.csv);;Excel (*.xlsx);;JSON Lines (*.jsonl)"
        )
        if not path:
            return
        try:
            self._write_dataframe(Path(path), self.model.df)
            self.status.showMessage(f"Saved copy to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save copy:\n{e}")

    def _write_dataframe(self, path: Path, df: pd.DataFrame):
        suf = path.suffix.lower()
        if suf == ".csv":
            df.to_csv(path, index=False)
        elif suf in [".xls", ".xlsx"]:
            df.to_excel(path, index=False)
        elif suf == ".jsonl":
            df.to_json(path, orient="records", lines=True, force_ascii=False)
        else:
            df.to_csv(path, index=False)

    def on_search_changed(self, text: str):
        if not self.model:
            return
        self.model.set_search_term(text)
        self.current_match_pos = -1
        self.replace_container.setVisible(bool(text.strip()))
        self._update_replace_buttons()

    def _update_replace_buttons(self):
        has_repl = bool(self.replace_edit.text().strip())
        self.btn_replace_next.setEnabled(has_repl)
        self.btn_replace_all.setEnabled(has_repl)
        if self.model:
            self.model.replace_term = self.replace_edit.text()

    def on_replace_next(self):
        if not self.model:
            return
        idx = self.model.replace_once(self.current_match_index())
        if idx:
            self.current_match_pos = self.model._matches.index(idx)
            self.table.scrollTo(idx, QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible)
            self.table.setCurrentIndex(idx)
            self._on_matches_changed(self.model.total_matches())

    def on_replace_all(self):
        if not self.model:
            return
        n = self.model.replace_all()
        QtWidgets.QMessageBox.information(self, "Replace All", f"Replaced in {n} cell(s).")

    def current_match_index(self) -> t.Optional[QtCore.QModelIndex]:
        if self.current_match_pos >= 0 and self.model:
            return self.model.match_at(self.current_match_pos)
        return None

    def _on_matches_changed(self, total: int):
        current = self.current_match_pos + 1 if total > 0 and self.current_match_pos >= 0 else 0
        self.match_label.setText(f"Matches: {current} / {total}")
        self.btn_prev.setEnabled(total > 0)
        self.btn_next.setEnabled(total > 0)

    def go_next_match(self):
        if not self.model or self.model.total_matches() == 0:
            return
        self.current_match_pos = (self.current_match_pos + 1) % self.model.total_matches()
        self._scroll_to_current()

    def go_prev_match(self):
        if not self.model or self.model.total_matches() == 0:
            return
        self.current_match_pos = (self.current_match_pos - 1) % self.model.total_matches()
        self._scroll_to_current()

    def _scroll_to_current(self):
        idx = self.model.match_at(self.current_match_pos)
        if idx:
            self.table.scrollTo(idx, QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible)
            self.table.setCurrentIndex(idx)
            self._on_matches_changed(self.model.total_matches())

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self.go_prev_match()
            else:
                self.go_next_match()
            return
        super().keyPressEvent(event)

    # --- Rest of methods (autosize, add_row, etc.) unchanged ---
    def autosize_columns(self, force=False):
        if not self.model:
            return
        font = self.table.font()
        metrics = QtGui.QFontMetrics(font)
        min_w, max_w = 80, 650
        for c in range(self.model.columnCount()):
            header = str(self.model.df.columns[c])
            col = self.model.df.iloc[:, c]
            sample = col.dropna().astype(str).head(1000)
            avg_len = sample.str.len().mean() if not sample.empty else 5
            base = 8 if "id" in header.strip().lower() and avg_len <= 16 else avg_len
            w = int(metrics.averageCharWidth() * (base + len(header) * 0.15)) + 28
            w = max(min_w, min(max_w, w))
            self.table.setColumnWidth(c, w)

    def autosize_rows(self):
        if not self.model:
            return
        delegate = self.table.itemDelegate()
        font_metrics = QtGui.QFontMetrics(self.table.font())
        default_height = font_metrics.lineSpacing() + 8
        for row in range(self.model.rowCount()):
            max_height = default_height
            for col in range(self.model.columnCount()):
                index = self.model.index(row, col)
                text = str(self.model.data(index, QtCore.Qt.ItemDataRole.DisplayRole) or "")
                layout = QtGui.QTextLayout(text, self.table.font())
                layout.beginLayout()
                height = 0
                while True:
                    line = layout.createLine()
                    if not line.isValid():
                        break
                    line.setLineWidth(self.table.columnWidth(col))
                    height += font_metrics.lineSpacing()
                layout.endLayout()
                max_height = max(max_height, height + 8)
            current_height = self.table.rowHeight(row)
            if current_height != max_height:
                self.table.setRowHeight(row, int(max_height))

    def add_row(self):
        if not self.model:
            return
        r = self.model.rowCount()
        self.model.beginInsertRows(QtCore.QModelIndex(), r, r)
        empty = {col: "" for col in self.model.df.columns}
        self.model.df = pd.concat([self.model.df, pd.DataFrame([empty])], ignore_index=True)
        self.model.endInsertRows()

    def delete_selected_rows(self):
        if not self.model:
            return
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        rows = sorted([i.row() for i in sel], reverse=True)
        for r in rows:
            self.model.beginRemoveRows(QtCore.QModelIndex(), r, r)
            self.model.df.drop(self.model.df.index[r], inplace=True)
            self.model.df.reset_index(drop=True, inplace=True)
            self.model.endRemoveRows()

    def toggle_mode(self, checked: bool):
        if self.model:
            self.model._is_dark = checked
        if checked:
            apply_dark_palette(QtWidgets.QApplication.instance())
            self.mode_toggle.setText("Dark")
            self.table.setStyleSheet("QTableView { gridline-color: rgb(80, 80, 80); }")
        else:
            apply_light_palette(QtWidgets.QApplication.instance())
            self.mode_toggle.setText("Light")
            self.table.setStyleSheet("QTableView { gridline-color: rgb(0, 0, 0); }")
        self.table.viewport().update()

class DiffTextEdit(QtWidgets.QTextEdit):
    def __init__(self):
        super().__init__()
        self.setAcceptRichText(False)
        self.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)


class DiffPage(QtWidgets.QWidget):
    backRequested = QtCore.pyqtSignal()

    def __init__(self, app_ref: 'MainWindow'):
        super().__init__()
        self.app_ref = app_ref

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Top bar
        self.controls = WindowControls("Compare Texts")
        self.controls.backRequested.connect(self.backRequested.emit)
        self.controls.minimizeRequested.connect(lambda: self.window().showMinimized())
        self.controls.maximizeRestoreRequested.connect(self._toggle_max_restore)
        self.controls.closeRequested.connect(self.window().close)

        # Small toolbar with live stats (no button)
        toolbar = QtWidgets.QToolBar()
        self.similarity_lbl = QtWidgets.QLabel("Similarity: —")
        self.stats_lbl = QtWidgets.QLabel("Δ: —")
        toolbar.addWidget(self.similarity_lbl)
        toolbar.addWidget(self.stats_lbl)

        # Editors side by side
        editors = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(editors)
        hl.setContentsMargins(8, 8, 8, 8)

        self.left_edit = QtWidgets.QTextEdit()
        self.right_edit = QtWidgets.QTextEdit()
        for e, ph in [(self.left_edit, "Left text (original)…"),
                      (self.right_edit, "Right text (modified)…")]:
            e.setPlaceholderText(ph)
            e.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
            e.textChanged.connect(self.run_compare)   # 🔥 auto-compare
        hl.addWidget(self.left_edit)
        hl.addWidget(self.right_edit)

        outer.addWidget(self.controls)
        outer.addWidget(toolbar)
        outer.addWidget(editors)

    # ---------- window helpers ----------
    def _toggle_max_restore(self):
        if self.window().isMaximized() or self.window().isFullScreen():
            self.window().showNormal()
        else:
            self.window().showMaximized()

    # ---------- live diff ----------
    def run_compare(self):
        from html import escape as esc
        import difflib, re

        left = self.left_edit.toPlainText()
        right = self.right_edit.toPlainText()
        if not left and not right:
            self.similarity_lbl.setText("Similarity: —")
            self.stats_lbl.setText("Δ: —")
            return

        def words(s: str):
            return re.findall(r"\s+|[^\s]+", s)

        l_words, r_words = words(left), words(right)
        sm = difflib.SequenceMatcher(a=l_words, b=r_words)
        opcodes = sm.get_opcodes()

        left_html_parts, right_html_parts = [], []
        dels = adds = 0

        for tag, i1, i2, j1, j2 in opcodes:
            l_chunk = "".join(l_words[i1:i2])
            r_chunk = "".join(r_words[j1:j2])
            if tag == "equal":
                left_html_parts.append(esc(l_chunk))
                right_html_parts.append(esc(r_chunk))
            elif tag == "delete":
                left_html_parts.append(f'<span style="background-color: rgba(255,0,0,0.35);">{esc(l_chunk)}</span>')
                dels += 1
            elif tag == "insert":
                right_html_parts.append(f'<span style="background-color: rgba(0,200,0,0.35);">{esc(r_chunk)}</span>')
                adds += 1
            elif tag == "replace":
                left_html_parts.append(f'<span style="background-color: rgba(255,0,0,0.35);">{esc(l_chunk)}</span>')
                right_html_parts.append(f'<span style="background-color: rgba(0,200,0,0.35);">{esc(r_chunk)}</span>')
                dels += 1
                adds += 1

        # ✅ Use <div> with pre-wrap so QTextEdit shows colors and layout correctly
        left_html = "<div style='font-family: Consolas, monospace; white-space: pre-wrap;'>" + "".join(left_html_parts) + "</div>"
        right_html = "<div style='font-family: Consolas, monospace; white-space: pre-wrap;'>" + "".join(right_html_parts) + "</div>"

        self.left_edit.blockSignals(True)
        self.right_edit.blockSignals(True)
        self.left_edit.setHtml(left_html)
        self.right_edit.setHtml(right_html)
        self.left_edit.blockSignals(False)
        self.right_edit.blockSignals(False)

        sim = sm.ratio()
        self.similarity_lbl.setText(f"Similarity: {sim*100:.1f}%")
        self.stats_lbl.setText(f"Δ: removals (left) {dels} | additions (right) {adds}")
        
        
class StartPage(QtWidgets.QWidget):
    viewRequested = QtCore.pyqtSignal()
    compareRequested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl = QtWidgets.QLabel("Welcome")
        lbl.setStyleSheet("font-size: 20px; font-weight: 600;")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        btns = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(btns)
        h.setSpacing(20)
        view = QtWidgets.QPushButton("View File")
        diff = QtWidgets.QPushButton("Compare Texts")
        view.clicked.connect(self.viewRequested.emit)
        diff.clicked.connect(self.compareRequested.emit)

        h.addWidget(view)
        h.addWidget(diff)

        hint = QtWidgets.QLabel("Tip: You can drag & drop your file after choosing 'View File'.")
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(lbl)
        layout.addWidget(btns)
        layout.addSpacing(10)
        layout.addWidget(hint)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QtGui.QIcon())

        central = QtWidgets.QStackedWidget()
        self.setCentralWidget(central)

        self.start = StartPage()
        self.viewer = DataViewerPage(self)
        self.diff = DiffPage(self)

        central.addWidget(self.start)
        central.addWidget(self.viewer)
        central.addWidget(self.diff)

        self.start.viewRequested.connect(self.enter_viewer)
        self.start.compareRequested.connect(self.enter_diff)
        self.viewer.backRequested.connect(self.back_to_start)
        self.diff.backRequested.connect(self.back_to_start)

        self.resize(520, 320)
        self.center_on_screen()

        apply_dark_palette(QtWidgets.QApplication.instance())

        self._create_shortcuts()

    def _create_shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Find, self, activated=self._focus_search)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=lambda: self.viewer.save_file())
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, activated=lambda: self.viewer.open_file_dialog())
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self._escape_behavior)

    def _focus_search(self):
        if self.centralWidget().currentWidget() is self.viewer:
            self.viewer.search_edit.setFocus()

    def _escape_behavior(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            cur = self.centralWidget().currentWidget()
            if cur is self.viewer or cur is self.diff:
                self.back_to_start()

    def center_on_screen(self):
        screen = QtGui.QGuiApplication.primaryScreen().geometry()
        self.move(int((screen.width() - self.width()) / 2), int((screen.height() - self.height()) / 2))

    def enter_viewer(self):
        self.centralWidget().setCurrentWidget(self.viewer)
        self.showFullScreen()

    def enter_diff(self):
        self.centralWidget().setCurrentWidget(self.diff)
        self.showFullScreen()

    def back_to_start(self):
        self.centralWidget().setCurrentWidget(self.start)
        self.showNormal()
        self.resize(520, 320)
        self.center_on_screen()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
