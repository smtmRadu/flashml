from pathlib import Path
import sys
import typing as t
import re
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
import time

APP_NAME = "Better Excel"


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
    """FINAL VERSION – subtle tinted whites"""
    cycle_len = 12
    step = index % cycle_len
    hue = int((step / cycle_len) * 360)
    saturation = 100  # visible but not garish
    lightness = 30
    return QtGui.QColor.fromHsl(hue, saturation, lightness)

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
        self.case_sensitive: bool = False

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
        # Allow spaces, but treat empty or whitespace-only as empty search
        if not term:  # if term is empty string
            term = ""
        # Optional: if you want to treat pure-whitespace as no search:
        # if not term.strip():
        #     term = ""
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
        
        if self.case_sensitive:
            # Case-sensitive search
            st = self.search_term
            for r in range(self.rowCount()):
                for c in range(self.columnCount()):
                    val = self._df.iat[r, c]
                    if pd.isna(val):
                        continue
                    if st in str(val):
                        self._matches.append(self.index(r, c))
        else:
            # Original case-insensitive search
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
        # Cache font metrics
        self._metrics_cache = {}
    
    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtCore.QSize:
        text = index.data(QtCore.Qt.ItemDataRole.DisplayRole) or ""
        
        # Fast path: short text without newlines
        if len(text) < 100 and '\n' not in text:
            font = option.font
            metrics = QtGui.QFontMetrics(font)
            width = metrics.horizontalAdvance(text) + 28  # padding
            height = metrics.lineSpacing() + 8
            return QtCore.QSize(width, height)
        
        # Slow path for multiline/long text
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(option.font)
        doc.setPlainText(text)
        doc.setTextWidth(option.rect.width())
        
        content_height = doc.size().height()
        total_height = int(content_height + 6)
        
        return QtCore.QSize(int(doc.idealWidth()), total_height)


    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        painter.save()
        
        # --- 1. Draw column-tinted background from the model ---
        bg_brush = index.data(QtCore.Qt.ItemDataRole.BackgroundRole)
        if bg_brush:
            painter.fillRect(option.rect, bg_brush)
        elif option.state & QtWidgets.QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        
        # --- 2. Now draw the content ---
        text = index.data(QtCore.Qt.ItemDataRole.DisplayRole) or ""
        st = self.model.search_term
        if st:
            if self.model.case_sensitive:
                has_term = st in text
            else:
                has_term = st.lower() in text.lower()
        else:
            has_term = False
        
        rect = option.rect.adjusted(4, 2, -4, -2)
        
        # Create document for text (rest of your existing logic)
        doc = QtGui.QTextDocument()
        doc.setDefaultFont(option.font)
        text_option = QtGui.QTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        text_option.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        doc.setDefaultTextOption(text_option)
        doc.setPlainText(text)
        doc.setTextWidth(rect.width())
        
        # Highlight matching words (existing logic unchanged)
        if has_term:
            start = 0
            highlights = []
            
            if self.model.case_sensitive:
                search_text = st
                search_in = text
            else:
                search_text = st.lower()
                search_in = text.lower()
            
            text_len = len(st)
            
            while True:
                i = search_in.find(search_text, start)
                if i == -1:
                    break
                highlights.append((i, text_len))
                start = i + text_len
            
            if highlights:
                layout = doc.documentLayout()
                for pos, length in highlights:
                    cursor = QtGui.QTextCursor(doc)
                    cursor.setPosition(pos)
                    cursor.setPosition(pos + length, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    block_rect = layout.blockBoundingRect(cursor.block())
                    selection_rect = QtCore.QRectF(block_rect)
                    selection_rect.translate(QtCore.QPointF(rect.topLeft()))
                    painter.fillRect(selection_rect, QtGui.QColor(0, 200, 0, 140))
        
        # Draw the text
        painter.translate(rect.topLeft())
        clip = QtCore.QRectF(0, 0, rect.width(), rect.height())
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
        self.icon = QtWidgets.QLabel("🗂️Drop a CSV / XLS / XLSX / JSONL file here (open file if WSL)")
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
        self._drag_start_pos = None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        
        self.back_btn = QtWidgets.QToolButton()
        self.back_btn.setText("← Back")
        self.back_btn.clicked.connect(self.backRequested.emit)

        self.title_lbl = QtWidgets.QLabel(title)
        self.title_lbl.setStyleSheet("font-weight: 600;")
        self.title_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

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

        # Left side: back button
        layout.addWidget(self.back_btn)
        layout.addStretch(1)
        
        # Center: title
        layout.addWidget(self.title_lbl)
        layout.addStretch(1)
        
        # Right side: window controls
        for b in (self.min_btn, self.max_btn, self.close_btn):
            layout.addWidget(b)
            
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._drag_start_pos and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_start_pos
            self.window().move(self.window().pos() + delta)
            self._drag_start_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start_pos = None
        super().mouseReleaseEvent(event)
        
class AnimatedScrollBar(QtWidgets.QScrollBar):
    """A scrollbar that animates value changes for buttery-smooth scrolling"""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._animation = QtCore.QPropertyAnimation(self, b"value")
        self._animation.setDuration(300)  # 180ms for smooth but responsive feel
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)  # Natural deceleration
        self._is_dragging = False
        
    def setValue(self, value):
        # If user is dragging the handle, update instantly for immediate feedback
        if self._is_dragging:
            super().setValue(value)
            return
            
        # If same value, nothing to do
        if value == self.value():
            return
            
        # Stop any ongoing animation before starting new one
        if self._animation.state() == QtCore.QPropertyAnimation.State.Running:
            self._animation.stop()
            
        # Animate from current to target value
        self._animation.setStartValue(self.value())
        self._animation.setEndValue(value)
        self._animation.start()
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._animation.stop()  # Stop animation when user grabs handle
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._is_dragging = False
        super().mouseReleaseEvent(event)


class SmoothScrollTableView(QtWidgets.QTableView):
    """QTableView with smooth animated scrolling for both wheel and scrollbar interactions"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable per-pixel scrolling (already smooth for drag operations)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        
        # Replace default scrollbars with animated versions
        v_scrollbar = AnimatedScrollBar(QtCore.Qt.Orientation.Vertical, self)
        h_scrollbar = AnimatedScrollBar(QtCore.Qt.Orientation.Horizontal, self)
        
        self.setVerticalScrollBar(v_scrollbar)
        self.setHorizontalScrollBar(h_scrollbar)
        
    def wheelEvent(self, event):
        # Animate mouse wheel scrolling for vertical movement
        if event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier:
            delta = event.angleDelta().y()
            scrollbar = self.verticalScrollBar()
            
            # Calculate scroll distance (3 lines per wheel step for reasonable speed)
            single_step = scrollbar.singleStep()
            if delta > 0:
                target_value = max(scrollbar.minimum(), scrollbar.value() - single_step * 0.75)
            else:
                target_value = min(scrollbar.maximum(), scrollbar.value() + single_step * 0.75)
                
            # Animate to target value (this triggers AnimatedScrollBar.setValue)
            scrollbar.setValue(target_value)
            event.accept()
        else:
            # Handle horizontal scrolling or Ctrl+wheel (zoom) normally
            super().wheelEvent(event)
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
        outer.setSpacing(0)

        # Window controls bar
        self.controls = WindowControls("Viewer")
        self.controls.backRequested.connect(self.backRequested.emit)
        self.controls.minimizeRequested.connect(lambda: self.window().showMinimized())
        self.controls.maximizeRestoreRequested.connect(self._toggle_max_restore)
        self.controls.closeRequested.connect(QtWidgets.QApplication.instance().quit)
        self.controls.setFixedHeight(40)

        # Toolbar
        toolbar = QtWidgets.QToolBar()
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

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
        self.search_edit.setMinimumWidth(150)
        self.search_edit.returnPressed.connect(self.on_search_return_pressed)
        
        toolbar.addWidget(self.search_edit)
        
        # --- Search Button (magnifier) ---
        self.search_btn = QtWidgets.QToolButton()
        self.search_btn.setText("🔍")
        self.search_btn.setToolTip("Search")
        self.search_btn.clicked.connect(self.on_search_return_pressed)
        toolbar.addWidget(self.search_btn)


        # --- Case Sensitive Toggle ---
        # --- Case Sensitive Toggle ---
        self.case_sensitive_btn = QtWidgets.QToolButton()
        self.case_sensitive_btn.setText("Aa")
        self.case_sensitive_btn.setCheckable(True)
        self.case_sensitive_btn.setChecked(False)
        self.case_sensitive_btn.setToolTip("Case sensitive search (off)")
        self.case_sensitive_btn.toggled.connect(self.on_case_sensitive_toggled)

        # Add blue background when checked
        self.case_sensitive_btn.setStyleSheet("""
            QToolButton:checked {
                background-color: rgb(60, 100, 160);
                border: 1px solid rgb(80, 130, 200);
            }
        """)

        toolbar.addWidget(self.case_sensitive_btn)

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
        self.match_label.setMinimumWidth(100)

        toolbar.addWidget(self.btn_prev)
        toolbar.addWidget(self.btn_next)
        toolbar.addWidget(self.match_label)

        toolbar.addSeparator()

        # --- Replace Section ---
        self.replace_container = QtWidgets.QWidget()
        repl_layout = QtWidgets.QHBoxLayout(self.replace_container)
        repl_layout.setContentsMargins(0, 0, 0, 0)
        self.replace_edit = QtWidgets.QLineEdit()
        self.replace_edit.setPlaceholderText("Replace with…")
        self.replace_edit.setMinimumWidth(150)
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

        # --- Font Size Scaler ---
        font_container = QtWidgets.QWidget()
        font_layout = QtWidgets.QHBoxLayout(font_container)
        font_layout.setContentsMargins(0, 0, 0, 0)
        font_layout.setSpacing(2)

        self.font_decrease_btn = QtWidgets.QToolButton()
        self.font_decrease_btn.setText("−")
        self.font_decrease_btn.setToolTip("Decrease font size")
        self.font_decrease_btn.clicked.connect(lambda: self.change_font_size(-1))

        self.font_size_label = QtWidgets.QLabel("100%")
        self.font_size_label.setMinimumWidth(45)
        self.font_size_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.font_increase_btn = QtWidgets.QToolButton()
        self.font_increase_btn.setText("+")
        self.font_increase_btn.setToolTip("Increase font size")
        self.font_increase_btn.clicked.connect(lambda: self.change_font_size(1))

        font_layout.addWidget(self.font_decrease_btn)
        font_layout.addWidget(self.font_size_label)
        font_layout.addWidget(self.font_increase_btn)

        toolbar.addWidget(font_container)
        toolbar.addSeparator()

        # --- Other Actions ---
        self.btn_autosize = QtGui.QAction("Autosize", self)
        self.btn_autosize.triggered.connect(lambda: self.autosize_columns(force=True))
        toolbar.addAction(self.btn_autosize)

        # --- Table View ---
        self.table = SmoothScrollTableView()
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        
        # === KEY CHANGE: Set vertical header to Fixed mode for manual management ===
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setWordWrap(True)
        
        # === NEW: Enable column dragging ===
        self.table.horizontalHeader().setSectionsMovable(True)
        self.table.horizontalHeader().setDragEnabled(True)
        self.table.horizontalHeader().setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.table.horizontalHeader().sectionMoved.connect(self._on_column_moved)
        
                # === NEW: Visual feedback for column dragging ===
        self.table.horizontalHeader().setDragDropOverwriteMode(False)
        self.table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        
        self._apply_scrollbar_style()
        
        # === NEW: Debounced row resizing system ===
        # Timer to debounce rapid column resize events
        self._row_resize_timer = QtCore.QTimer()
        self._row_resize_timer.setSingleShot(True)
        self._row_resize_timer.setInterval(120)  # 120ms debounce for responsiveness
        self._row_resize_timer.timeout.connect(self._resize_visible_rows)
        
        # Track which columns were resized since last update
        self._columns_resized = set()
        
        # Connect column resize signal
        self.table.horizontalHeader().sectionResized.connect(self._on_column_resized)
        
        # === NEW: Track scroll position for on-demand row sizing ===
        self._last_scroll_value = 0
        self.table.verticalScrollBar().valueChanged.connect(self._on_scroll)
        
        # === NEW: Handle cell edits ===
        # This will be connected after model is created
        
        # --- Drop Zone ---
        self.dd = DragDropWidget()
        self.dd.fileDropped.connect(self.load_path)
        
        # Style the drop zone better
        self.dd.setStyleSheet("""
            DragDropWidget {
                background-color: rgba(60, 60, 60, 100);
                border: 2px dashed rgba(100, 100, 100, 150);
                border-radius: 12px;
            }
            DragDropWidget:hover {
                background-color: rgba(70, 70, 70, 120);
                border-color: rgba(120, 120, 120, 200);
            }
        """)

        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.status.setFixedHeight(24)

        # Background task indicator (bottom-left corner)
        self.bg_indicator = QtWidgets.QLabel("")
        self.bg_indicator.setStyleSheet("""
            QLabel {
                background-color: #FFC107;
                color: black;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: 600;
                border: 1px solid #E65100;
            }
        """)
        self.bg_indicator.hide()
        self.status.addWidget(self.bg_indicator)  # Left side of status bar

        # Timer for updating elapsed time & progress
        self.bg_timer = QtCore.QTimer()
        self.bg_timer.setInterval(500)  # Update every 500ms for smoother progress
        self.bg_timer.timeout.connect(self._update_bg_indicator)

        # Progress tracking
        self.bg_start_time = None
        self.bg_task_name = ""
        self.bg_total_rows = 0
        self.bg_current_row = 0

        # Add everything to layout
        outer.addWidget(self.controls)
        outer.addWidget(toolbar)
        outer.addWidget(self.dd, 1)  # Give it stretch factor
        outer.addWidget(self.table, 1)  # Give it stretch factor
        outer.addWidget(self.status)
        
        self.table.hide()
    def on_case_sensitive_toggled(self, checked: bool):
        """Toggle case sensitivity and refresh search"""
        if self.model:
            self.model.case_sensitive = checked
            self.case_sensitive_btn.setToolTip(f"Case sensitive search ({'on' if checked else 'off'})")
            
            # Re-run search with new setting
            search_text = self.search_edit.text()
            if search_text:
                current_term = self.model.search_term
                self.model.set_search_term("")
                self.model.set_search_term(current_term)
                self.table.viewport().update()  # 👈 Force immediate repaint
                
                
    def _show_background_indicator(self, task_name: str, total_rows: int, current_row: int = 0):
        """Show the background processing indicator with progress and elapsed time"""
        self.bg_task_name = task_name
        self.bg_start_time = time.time()
        self.bg_total_rows = total_rows
        self.bg_current_row = current_row
        self.bg_indicator.show()
        self._update_bg_indicator()
        self.bg_timer.start()

    def _hide_background_indicator(self):
        """Hide the background processing indicator"""
        self.bg_timer.stop()
        self.bg_indicator.hide()
        self.bg_start_time = None
        self.bg_task_name = ""
        self.bg_total_rows = 0
        self.bg_current_row = 0

    def _update_bg_indicator(self):
        """Update the progress and elapsed time in the indicator"""
        if self.bg_start_time is None or not self.isVisible():
            return
        
        elapsed = int(time.time() - self.bg_start_time)
        
        # Show both progress % and elapsed time
        if self.bg_total_rows > 0:
            progress = (self.bg_current_row / self.bg_total_rows) * 100
            self.bg_indicator.setText(f"{self.bg_task_name}… ({progress:.1f}% - {elapsed}s)")
        else:
            self.bg_indicator.setText(f"{self.bg_task_name}… ({elapsed}s)")
    
    def _on_column_moved(self, logical_index, old_visual_index, new_visual_index):
        """Reorder DataFrame columns when user drags column header"""
        if not self.model or old_visual_index == new_visual_index:
            return
        
        try:
            # Get current visual order
            header = self.table.horizontalHeader()
            visual_order = [header.logicalIndex(i) for i in range(header.count())]
            
            # Use layout change signals for efficient update (no flicker)
            self.model.layoutAboutToBeChanged.emit()
            
            # Reorder DataFrame to match visual order
            new_columns = [self.model.df.columns[i] for i in visual_order]
            self.model.df = self.model.df[new_columns]
            
            self.model.layoutChanged.emit()
            
            # Reapply search highlights (column indices changed)
            search_text = self.search_edit.text()
            if search_text:
                self.model.set_search_term("")
                self.model.set_search_term(search_text)
            
            # Resize rows (column widths are preserved by the header)
            self._autosize_all_rows()
            
            # Update status
            col_name = new_columns[new_visual_index]
            self.status.showMessage(f"Moved column '{col_name}' to position {new_visual_index + 1}")
            
        except Exception as e:
            print(f"Error moving column: {e}")
            self.status.showMessage("Error moving column")
            
    def _toggle_max_restore(self):
        if self.window().isMaximized() or self.window().isFullScreen():
            self.window().showNormal()
        else:
            self.window().showMaximized()
            
    def _apply_scrollbar_style(self):
        """Apply custom scrollbar styling with gray sliders"""
        scrollbar_style = """
            QScrollBar:vertical {
                border: none;
                background: rgb(30, 30, 30);
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgb(120, 120, 120);
                min-height: 30px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgb(140, 140, 140);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            
            QScrollBar:horizontal {
                border: none;
                background: rgb(30, 30, 30);
                height: 14px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: rgb(120, 120, 120);
                min-width: 30px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgb(140, 140, 140);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """
        self.table.setStyleSheet(self.table.styleSheet() + scrollbar_style)
                             
    def on_search_return_pressed(self):
        """Triggered when Enter is pressed in the search field."""
        search_text = self.search_edit.text()
        if not self.model:
            return
        self.model.set_search_term(search_text)
        self.current_match_pos = -1
        self.replace_container.setVisible(bool(search_text.strip()))
        self._update_replace_buttons()
        
    def load_path(self, path: Path):
        self._hide_background_indicator()
    
        original_text = self.dd.icon.text()
        self.dd.icon.setText("Loading…")
        QtWidgets.QApplication.processEvents()
        
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, low_memory=False)
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
        
        # === NEW: Connect dataChanged for cell edit handling ===
        self.model.dataChanged.connect(self._on_cell_data_changed)
        
        self.table.setModel(self.model)
        self.table.setItemDelegate(HighlightDelegate(self.model))
        
        # Style setup
        if self.model._is_dark:
            self.table.setStyleSheet("QTableView { gridline-color: rgb(80, 80, 80); }")
            self._apply_scrollbar_style()
        else:
            self.table.setStyleSheet("QTableView { gridline-color: rgb(0, 0, 0); }")
            self._apply_scrollbar_style()
        
        # === NEW: Efficient initialization ===
        self.table.show()
        self.dd.hide()
        self.status.showMessage(f"Loaded: {path}  |  {df.shape[0]} rows × {df.shape[1]} cols")
        
        # Size columns first (fast)
        self.autosize_columns(force=True)
        
        # Defer row sizing - let UI paint first
        QtCore.QTimer.singleShot(50, self._autosize_all_rows)
        
        # Reset search state
        self.current_match_pos = -1
        self.base_font_size = 10
        self.font_scale = 1.0
        self.model.set_search_term("")
        self._on_matches_changed(0)
        
        # Enter full screen AFTER file is loaded
        self.window().showFullScreen()

    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", str(Path.home()), "Data Files (*.csv *.xls *.xlsx *.jsonl)"
        )
        if path:
            self.load_path(Path(path))
            
    def change_font_size(self, delta: int):
        # Adjust scale by 10% per step
        self.font_scale += delta * 0.1
        self.font_scale = max(0.5, min(2.0, self.font_scale))  # 50% to 200%
        
        # Update label
        self.font_size_label.setText(f"{int(self.font_scale * 100)}%")
        
        # Apply new font size to table
        new_size = int(self.base_font_size * self.font_scale)
        font = self.table.font()
        font.setPointSize(new_size)
        self.table.setFont(font)
        
        # Refresh table to apply changes
        self.table.viewport().update()
        self._autosize_all_rows()  # Recalculate all rows after font change
        
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

    def autosize_columns(self, force=False):
        if not self.model:
            return
        
        # Cache metrics and calculate once
        font = self.table.font()
        metrics = QtGui.QFontMetrics(font)
        avg_char_width = metrics.averageCharWidth()
        min_w, max_w = 80, 650
        
        # Sample size: 200 rows is enough for most cases
        SAMPLE_SIZE = 200
        df_sample = self.model.df.head(SAMPLE_SIZE) if len(self.model.df) > SAMPLE_SIZE else self.model.df
        
        for c in range(self.model.columnCount()):
            header = str(self.model.df.columns[c])
            col = df_sample.iloc[:, c]
            
            # Initialize avg_len with a default value
            avg_len = 8  # ← FIX: Define it here
            
            # Fast path: numeric columns need less width
            if pd.api.types.is_numeric_dtype(col):
                # Find max numeric width
                max_val = col.dropna().abs().max() if not col.dropna().empty else 0
                avg_len = len(f"{max_val:.2f}") if pd.notna(max_val) else 8  # ← Assign here
            else:
                # Use vectorized string length calculation
                sample = col.dropna().astype(str)
                if not sample.empty:
                    avg_len = sample.str.len().mean()
                    avg_len = min(avg_len, 100)  # Cap extremely long text
                else:
                    avg_len = 8  # ← Fallback here
            
            # Now avg_len is always defined
            is_id_col = "id" in header.strip().lower() and avg_len <= 16
            base = 8 if is_id_col else int(avg_len)
            w = int(avg_char_width * (base + len(header) * 0.15)) + 28
            w = max(min_w, min(max_w, w))
            
            # Set width without triggering multiple signals
            self.table.horizontalHeader().resizeSection(c, w)
    
    # === NEW: Column resize handler (debounced) ===
    def _on_column_resized(self, logical_index: int, old_size: int, new_size: int):
        """Triggered when any column is resized. Starts debounced row update."""
        # Ignore tiny adjustments (less than 5 pixels) to avoid micro-calculations
        if abs(new_size - old_size) > 5:
            self._columns_resized.add(logical_index)
            self._row_resize_timer.start()

    # === NEW: Efficient row height updater for visible area ===
    def _resize_visible_rows(self):
        """Recalculates row heights for visible area based on resized columns."""
        if not self.model or not self._columns_resized:
            return
        
        # Get viewport boundaries for visible rows
        viewport = self.table.viewport()
        top_row = self.table.rowAt(0)
        bottom_row = self.table.rowAt(viewport.height())
        
        # Handle edge cases
        if top_row == -1: 
            top_row = 0
        if bottom_row == -1: 
            bottom_row = self.model.rowCount() - 1
        
        # Add buffer rows for smoother scrolling (25 rows above/below)
        BUFFER_ROWS = 25
        top_row = max(0, top_row - BUFFER_ROWS)
        bottom_row = min(self.model.rowCount() - 1, bottom_row + BUFFER_ROWS)
        
        # Calculate new heights only for rows in visible+buffer zone
        for row in range(top_row, bottom_row + 1):
            self.table.resizeRowToContents(row)
        
        # Clear the tracking set
        self._columns_resized.clear()
        
        # Background full update for very large tables (>1000 rows)
        if self.model.rowCount() > 1000:
            QtCore.QTimer.singleShot(800, self._background_resize_rows)

    # === NEW: Background processing for off-screen rows ===
    def _background_resize_rows(self):
        """Low-priority update for rows outside viewport (runs after idle)."""
        if not self.model:
            return
        
        total_rows = self.model.rowCount()
        
        # Process in batches of 50 to maintain UI responsiveness
        BATCH_SIZE = 50
        delay = 15  # milliseconds between batches
        
        # Use closure to maintain state
        def create_batch_processor():
            current_row = 0
            def process_next_batch():
                nonlocal current_row
                if not self.model:  # Safety check
                    return
                    
                end_row = min(current_row + BATCH_SIZE, total_rows)
                for row in range(current_row, end_row):
                    self.table.resizeRowToContents(row)
                
                current_row = end_row
                if current_row < total_rows:
                    QtCore.QTimer.singleShot(delay, process_next_batch)
            
            return process_next_batch
        
        # Start processing from first row
        processor = create_batch_processor()
        processor()

    # === NEW: Handle cell data changes ===
    def _on_cell_data_changed(self, top_left, bottom_right, roles):
        """Recalculate row height when cell data changes via editing."""
        if not roles or QtCore.Qt.ItemDataRole.DisplayRole in roles:
            # Recalculate just the changed rows
            for row in range(top_left.row(), bottom_right.row() + 1):
                self.table.resizeRowToContents(row)

    # === NEW: Handle scrolling for on-demand row sizing ===
    def _on_scroll(self, value):
        """Precalculate rows as they come into view - optimized version"""
        if not self.model:
            return
        
        # More aggressive throttling
        if abs(value - self._last_scroll_value) < 150:
            return
        
        self._last_scroll_value = value
        
        # Get viewport boundaries more efficiently
        viewport_height = self.table.viewport().height()
        if viewport_height <= 0:
            return
        
        top_row = self.table.rowAt(0)
        bottom_row = self.table.rowAt(viewport_height)
        
        # Validate row indices
        if top_row == -1:
            top_row = max(0, int(value / self.table.verticalHeader().defaultSectionSize()) - 10)
        if bottom_row == -1:
            bottom_row = min(self.model.rowCount() - 1, top_row + 100)
        
        # Add smaller buffer for better performance
        BUFFER_ROWS = 20
        top_row = max(0, top_row - BUFFER_ROWS)
        bottom_row = min(self.model.rowCount() - 1, bottom_row + BUFFER_ROWS)
        
        # Check if rows already have reasonable height
        font_metrics = QtGui.QFontMetrics(self.table.font())
        min_expected_height = font_metrics.lineSpacing() + 8
        
        # Process only rows that need it
        for row in range(top_row, bottom_row + 1):
            if self.table.rowHeight(row) < min_expected_height:
                # Double-check if this row actually needs resizing
                needs_resize = False
                for c in range(self.model.columnCount()):
                    val = self.model.df.iat[row, c]
                    if pd.notna(val):
                        s = str(val)
                        if '\n' in s or len(s) > 100:  # Only resize if multiline or very long
                            needs_resize = True
                            break
                
                if needs_resize:
                    self.table.resizeRowToContents(row)
    # === NEW: Full recalculation on initial load ===
    def _autosize_all_rows(self):
        """Full recalculation of all row heights - called only on file load."""
        if not self.model:
            return
        
        row_count = self.model.rowCount()
        if row_count == 0:
            return
        
        # Calculate a smart default height based on font
        font = self.table.font()
        metrics = QtGui.QFontMetrics(font)
        default_height = metrics.lineSpacing() + 8
        self.table.verticalHeader().setDefaultSectionSize(default_height)
        
        # Identify columns that might contain multiline content
        text_columns = []
        for c in range(self.model.columnCount()):
            col_data = self.model.df.iloc[:, c]
            if col_data.dtype == object:
                sample = col_data.dropna().head(100)
                if sample.astype(str).str.contains('\n').any() or sample.astype(str).str.len().max() > 200:
                    text_columns.append(c)
        
        if not text_columns:
            return  # Fast exit: no columns need row height adjustment
        
        # Show indicator for large tables
        show_indicator = row_count > 500
        if show_indicator:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self._show_background_indicator("Autosizing rows", row_count, 0)
        
        # Process initial rows (limit to 500 for initial load)
        initial_limit = min(500, row_count)
        
        for start_row in range(0, initial_limit, 50):
            end_row = min(start_row + 50, initial_limit)
            
            # Update progress
            if show_indicator:
                self.bg_current_row = end_row
                self._update_bg_indicator()
            
            for row in range(start_row, end_row):
                # Quick check before resizing
                needs_resize = False
                for col in text_columns:
                    val = self.model.df.iat[row, col]
                    if pd.notna(val):
                        s = str(val)
                        if '\n' in s or len(s) > 200:
                            needs_resize = True
                            break
                
                if needs_resize:
                    self.table.resizeRowToContents(row)
            
            # Keep UI responsive
            if show_indicator and start_row % 200 == 0:
                QtWidgets.QApplication.processEvents()
        
        if show_indicator:
            QtWidgets.QApplication.restoreOverrideCursor()
        
        # Start background processing for remaining rows
        remaining_rows = row_count - initial_limit
        if remaining_rows > 0:
            # Continue progress from where we left off
            self.bg_current_row = initial_limit
            self._update_bg_indicator()
            QtCore.QTimer.singleShot(100, lambda: self._background_resize_remaining(initial_limit))
        else:
            self._hide_background_indicator()  # Hide if no background work needed
        
    def _background_resize_remaining(self, start_row):
        """Process remaining rows in very small batches"""
        if not self.model:
            self._hide_background_indicator()
            return
        
        total_rows = self.model.rowCount()
        if start_row >= total_rows:
            self._hide_background_indicator()
            return
        
        # Process in tiny batches to avoid any UI lag
        BATCH_SIZE = 25
        end_row = min(start_row + BATCH_SIZE, total_rows)
        
        # Update progress
        self.bg_current_row = end_row
        
        for row in range(start_row, end_row):
            self.table.resizeRowToContents(row)
        
        # Update progress display
        self._update_bg_indicator()
        
        # Schedule next batch or hide indicator
        if end_row < total_rows:
            QtCore.QTimer.singleShot(10, lambda: self._background_resize_remaining(end_row))
        else:
            self._hide_background_indicator()   
                
    def toggle_mode(self, checked: bool):
        if self.model:
            self.model._is_dark = checked
        if checked:
            apply_dark_palette(QtWidgets.QApplication.instance())
            self.mode_toggle.setText("Dark")
            self.table.setStyleSheet("QTableView { gridline-color: rgb(80, 80, 80); }")
            self._apply_scrollbar_style()
            # Dark mode style for case-sensitive button
            self.case_sensitive_btn.setStyleSheet("""
                QToolButton:checked {
                    background-color: rgb(60, 100, 160);
                    border: 1px solid rgb(80, 130, 200);
                }
            """)
        else:
            apply_light_palette(QtWidgets.QApplication.instance())
            self.mode_toggle.setText("Light")
            # Light mode style for case-sensitive button
            self.case_sensitive_btn.setStyleSheet("""
                QToolButton:checked {
                    background-color: rgb(0, 122, 204);
                    border: 1px solid rgb(0, 100, 180);
                }
            """)
            light_scrollbar_style = """
                QTableView { gridline-color: rgb(0, 0, 0); }
                QScrollBar:vertical {
                    border: none;
                    background: rgb(240, 240, 240);
                    width: 14px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background: rgb(150, 150, 150);
                    min-height: 30px;
                    border-radius: 7px;
                }
                QScrollBar::handle:vertical:hover {
                    background: rgb(120, 120, 120);
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                }
                
                QScrollBar:horizontal {
                    border: none;
                    background: rgb(240, 240, 240);
                    height: 14px;
                    margin: 0px;
                }
                QScrollBar::handle:horizontal {
                    background: rgb(150, 150, 150);
                    min-width: 30px;
                    border-radius: 7px;
                }
                QScrollBar::handle:horizontal:hover {
                    background: rgb(120, 120, 120);
                }
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                    width: 0px;
                }
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                    background: none;
                }
            """
            self.table.setStyleSheet(light_scrollbar_style)
        self.table.viewport().update()
                
        
class DiffTextEdit(QtWidgets.QTextEdit):
    def __init__(self):
        super().__init__()
        self.setAcceptRichText(False)
        self.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        self._unescaping = False
        self.textChanged.connect(self._auto_unescape)

    def _auto_unescape(self):
        """Convert literal \n, \t, \\, etc. into real control characters."""
        if self._unescaping:
            return
            
        text = self.toPlainText()
        if '\\' not in text:  # Fast exit if nothing to do
            return
            
        # Only convert these specific escape sequences
        pattern = re.compile(r'\\([ntr\\"\'])')
        escape_map = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
        
        unescaped = pattern.sub(lambda m: escape_map.get(m.group(1), m.group(0)), text)
        
        if unescaped != text:
            self._unescaping = True
            cursor = self.textCursor()
            scroll_pos = self.verticalScrollBar().value()
            self.setPlainText(unescaped)
            self.setTextCursor(cursor)
            self.verticalScrollBar().setValue(scroll_pos)
            self._unescaping = False

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
        self.controls.closeRequested.connect(QtWidgets.QApplication.instance().quit)

        # Small toolbar with live stats
        toolbar = QtWidgets.QToolBar()
        self.similarity_lbl = QtWidgets.QLabel("Similarity: —")
        self.stats_lbl = QtWidgets.QLabel("Δ: —")
        toolbar.addWidget(self.similarity_lbl)
        toolbar.addWidget(self.stats_lbl)

        # Editors side by side
        editors = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(editors)
        hl.setContentsMargins(8, 8, 8, 8)

        self.left_edit = DiffTextEdit()
        self.right_edit = DiffTextEdit()
        for e, ph in [(self.left_edit, "Left text (original)…"),
                      (self.right_edit, "Right text (modified)…")]:
            e.setPlaceholderText(ph)
            e.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
            e.textChanged.connect(self.schedule_compare)
        hl.addWidget(self.left_edit)
        hl.addWidget(self.right_edit)

        outer.addWidget(self.controls)
        outer.addWidget(toolbar)
        outer.addWidget(editors)

        # Enable undo and redo functionality
        self.left_edit.setUndoRedoEnabled(True)
        self.right_edit.setUndoRedoEnabled(True)

        # Store base font size for zooming
        self.base_font_size = self.left_edit.font().pointSize()
        self.zoom_factor = 1.0

        # Timer for debounced comparison
        self.compare_timer = QtCore.QTimer()
        self.compare_timer.setSingleShot(True)
        self.compare_timer.setInterval(100)
        self.compare_timer.timeout.connect(self.run_compare)

        # Set up keyboard shortcuts for undo/redo
        self.set_up_shortcuts()

        # Selection synchronization
        self._syncing_selection = False
        self.left_edit.cursorPositionChanged.connect(lambda: self._sync_selection(self.left_edit, self.right_edit))
        self.right_edit.cursorPositionChanged.connect(lambda: self._sync_selection(self.right_edit, self.left_edit))

    def _sync_selection(self, source_edit: QtWidgets.QTextEdit, target_edit: QtWidgets.QTextEdit):
        """Synchronize text selection from source to target editor with vertical alignment"""
        if self._syncing_selection:
            return
        
        # Always clear target selection first
        self._clear_selection(target_edit)
        
        cursor = source_edit.textCursor()
        if not cursor.hasSelection():
            return
        
        selected_text = cursor.selectedText()
        if not selected_text:
            return

        # Detect if selection is in diff-only region
        start_cursor = QtGui.QTextCursor(cursor)
        start_cursor.setPosition(cursor.selectionStart())
        char_format = start_cursor.charFormat()
        bg_color = char_format.background().color()
        
        # Check if background is red (deleted) or green (inserted)
        is_red = bg_color.red() > 150 and bg_color.green() < 80 and bg_color.blue() < 80
        is_green = bg_color.green() > 120 and bg_color.red() < 80 and bg_color.blue() < 80
        
        if is_red or is_green:
            # Don't sync diff-only text - just keep cleared state
            return
        
        # Find and highlight matching text in target
        target_doc = target_edit.document()
        find_cursor = target_doc.find(selected_text, 0)
        
        if find_cursor.isNull():
            return
            
        self._syncing_selection = True
        
        # Calculate relative position for scroll alignment
        source_rect = source_edit.cursorRect(cursor)
        source_viewport_height = source_edit.viewport().height()
        
        if source_viewport_height > 0:
            source_rel_y = source_rect.center().y() / source_viewport_height
            source_rel_y = max(0.0, min(1.0, source_rel_y))
            
            target_edit.setTextCursor(find_cursor)
            target_rect = target_edit.cursorRect(find_cursor)
            target_viewport_height = target_edit.viewport().height()
            
            if target_viewport_height > 0:
                target_scrollbar = target_edit.verticalScrollBar()
                current_scroll = target_scrollbar.value()
                desired_viewport_y = source_rel_y * target_viewport_height
                delta_y = target_rect.top() - desired_viewport_y
                new_scroll = current_scroll + delta_y
                new_scroll = max(0, min(new_scroll, target_scrollbar.maximum()))
                target_scrollbar.setValue(int(new_scroll))
        
        # Create and apply new highlight
        selection = QtWidgets.QTextEdit.ExtraSelection()
        selection.cursor = find_cursor
        selection.format.setBackground(QtGui.QColor(0, 150, 255, 100))
        selection.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, False)
        
        target_edit.setExtraSelections([selection])
        target_edit.viewport().update()
        
        self._syncing_selection = False

    def _clear_selection(self, edit: QtWidgets.QTextEdit):
        """Clear extra selection highlights in the given editor"""
        self._syncing_selection = True
        edit.setExtraSelections([])
        edit.viewport().update()  # Force immediate repaint
        self._syncing_selection = False

    def set_up_shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self, self.undo_text)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self, self.redo_text)

    def undo_text(self):
        focused = QtWidgets.QApplication.focusWidget()
        if focused in (self.left_edit, self.right_edit):
            if focused.isUndoAvailable():
                focused.undo()

    def redo_text(self):
        focused = QtWidgets.QApplication.focusWidget()
        if focused in (self.left_edit, self.right_edit):
            if focused.isRedoAvailable():
                focused.redo()

    def schedule_compare(self):
        """Debounce comparison to avoid running on every keystroke"""
        self.compare_timer.start()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() / 120
            
            self.zoom_factor += delta * 0.1
            self.zoom_factor = max(0.5, min(3.0, self.zoom_factor))
            
            new_size = int(self.base_font_size * self.zoom_factor)
            font = self.left_edit.font()
            font.setPointSize(new_size)
            self.left_edit.setFont(font)
            self.right_edit.setFont(font)
            
            event.accept()
        else:
            super().wheelEvent(event)

    def run_compare(self):
        from html import escape as esc
        import difflib, re

        left = self.left_edit.toPlainText()
        right = self.right_edit.toPlainText()
        
        if not left and not right:
            self.similarity_lbl.setText("Similarity: —")
            self.stats_lbl.setText("Δ: —")
            return

        left_cursor = self.left_edit.textCursor()
        right_cursor = self.right_edit.textCursor()
        left_scroll = self.left_edit.verticalScrollBar().value()
        right_scroll = self.right_edit.verticalScrollBar().value()

        def words(s: str):
            return re.findall(r"\s+|[a-zA-Z0-9]+|[^a-zA-Z0-9\s]", s)

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

        left_html = "<div style='font-family: Consolas, monospace; white-space: pre-wrap;'>" + "".join(left_html_parts) + "</div>"
        right_html = "<div style='font-family: Consolas, monospace; white-space: pre-wrap;'>" + "".join(right_html_parts) + "</div>"

        self.left_edit.blockSignals(True)
        self.right_edit.blockSignals(True)
        
        self.left_edit.setHtml(left_html)
        self.right_edit.setHtml(right_html)
        
        self.left_edit.setTextCursor(left_cursor)
        self.right_edit.setTextCursor(right_cursor)
        
        self.left_edit.verticalScrollBar().setValue(left_scroll)
        self.right_edit.verticalScrollBar().setValue(right_scroll)
        
        self.left_edit.blockSignals(False)
        self.right_edit.blockSignals(False)

        sim = sm.ratio()
        self.similarity_lbl.setText(f"Similarity: {sim*100:.1f}%")
        self.stats_lbl.setText(f"Δ: removals (left) {dels} | additions (right) {adds}")

    def _toggle_max_restore(self):
        if self.window().isMaximized() or self.window().isFullScreen():
            self.window().showNormal()
        else:
            self.window().showMaximized()
class StartPage(QtWidgets.QWidget):
    viewRequested = QtCore.pyqtSignal()
    compareRequested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        
        # Main layout with no margins/spacing for the window controls
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Add window controls bar - hide back button on start page
        self.controls = WindowControls("Better Excel")
        self.controls.back_btn.hide()
        self.controls.minimizeRequested.connect(lambda: self.window().showMinimized())
        self.controls.maximizeRestoreRequested.connect(self._toggle_max_restore)
        self.controls.closeRequested.connect(QtWidgets.QApplication.instance().quit)
        
        # Content container for centered elements
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Welcome label
        lbl = QtWidgets.QLabel("Welcome")
        lbl.setStyleSheet("font-size: 20px; font-weight: 600;")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Buttons container
        btns = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(btns)
        h.setSpacing(20)
        
        view = QtWidgets.QPushButton("View File")
        diff = QtWidgets.QPushButton("Compare Texts")
        
        # Style the buttons
        button_style = """
            QPushButton {
                min-height: 80px;
                font-size: 16px;
                font-weight: 600;
                border-radius: 12px;
                padding: 10px 20px;
                background-color: rgb(45, 45, 45);
                border: 1px solid rgb(60, 60, 60);
            }
            QPushButton:hover {
                background-color: rgba(60, 100, 160, 180);
            }
        """
        view.setStyleSheet(button_style)
        diff.setStyleSheet(button_style)
        
        view.clicked.connect(self.viewRequested.emit)
        diff.clicked.connect(self.compareRequested.emit)
        
        h.addWidget(view)
        h.addWidget(diff)
        
        # Assemble content
        content_layout.addWidget(lbl)
        content_layout.addWidget(btns)
        content_layout.addSpacing(10)
        
        # Add to main layout
        layout.addWidget(self.controls)
        layout.addWidget(content, 1)  # Stretch factor for content

    def _toggle_max_restore(self):
        """Toggle between maximized and normal window state"""
        if self.window().isMaximized():
            self.window().showNormal()
        else:
            self.window().showMaximized()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QtGui.QIcon())
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
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
        # Get the screen that contains the mouse cursor
        cursor_pos = QtGui.QCursor.pos()
        screen = QtWidgets.QApplication.screenAt(cursor_pos)
        
        # Fallback to primary screen if screenAt returns None
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        
        screen_geometry = screen.availableGeometry()  # Use availableGeometry to account for taskbars
        
        # Calculate center position
        x = screen_geometry.x() + (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.y() + (screen_geometry.height() - self.height()) // 2
        
        self.move(x, y)
        
    def enter_viewer(self):
        self.centralWidget().setCurrentWidget(self.viewer)
        if self.viewer.model is not None:
            self.showFullScreen()
        else:
            self.showNormal()
            self.resize(520, 320)
            self.center_on_screen()

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
    win.center_on_screen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
