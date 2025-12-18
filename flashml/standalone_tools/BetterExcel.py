from pathlib import Path
import sys
import typing as t
import re
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
import time

APP_NAME = "Better Excel v1.1"

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
    dataChangedHard = QtCore.pyqtSignal()     
    matchesChanged = QtCore.pyqtSignal(int)    
    dirtyStateChanged = QtCore.pyqtSignal(bool)
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        self.search_term: str = ""
        self.replace_term: str = ""
        self._is_dark = True
        self._matches: list[QtCore.QModelIndex] = []
        self.case_sensitive: bool = False
        self._is_dirty = False
        
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        self._df = value

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    @is_dirty.setter
    def is_dirty(self, value: bool):
        """Automatically emit signal when dirty state changes"""
        if self._is_dirty != value:
            self._is_dirty = value
            self.dirtyStateChanged.emit(value)

    def rowCount(self, parent=None):
        return len(self._df.index)

    def columnCount(self, parent=None):
        return len(self._df.columns) + 1

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        
        # Handle the "+" column
        if c == len(self._df.columns):
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return ""
            elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
                return QtGui.QBrush(QtGui.QColor(60, 60, 60))
            return None
        
        # Existing logic for regular columns
        val = self._df.iat[r, c]
        if role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            if pd.isna(val):
                return ""
            return str(val)

        if role == QtCore.Qt.ItemDataRole.BackgroundRole:
            text = "" if pd.isna(val) else str(val).lower()
            if self.search_term and self.search_term.lower() in text:
                return QtGui.QBrush(QtGui.QColor(255, 255, 150, 80))

            base_color = column_shade_color(self._is_dark, c)
            if r % 2 == 1:
                base_color = base_color.darker(150)
            
            return QtGui.QBrush(base_color)
        return None
    
    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                if section == len(self._df.columns):  # Last column
                    return "+"
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        if index.column() == len(self._df.columns):
            return QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        return (
            QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsEditable
        )

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if role == QtCore.Qt.ItemDataRole.EditRole and index.isValid():
            r, c = index.row(), index.column()
            self._df.iat[r, c] = value
            self.is_dirty = True
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
            st = self.search_term
            for r in range(self.rowCount()):
                for c in range(self.columnCount() - 1):  # Exclude "+" column
                    val = self._df.iat[r, c]
                    if pd.isna(val):
                        continue
                    if st in str(val):
                        self._matches.append(self.index(r, c))
        else:
            st = self.search_term.lower()
            for r in range(self.rowCount()):
                for c in range(self.columnCount() - 1):  # Exclude "+" column
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
    
    def add_new_column(self, col_name: str):
        """Add a new column to the DataFrame"""
        if col_name and col_name not in self._df.columns:
            self.beginInsertColumns(QtCore.QModelIndex(), self.columnCount() - 1, self.columnCount() - 1)
            self._df[col_name] = ""
            self.is_dirty = True
            self.endInsertColumns()

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
        self.is_dirty = True
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
            self.is_dirty = True
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
        
        # Get base background color
        bg_brush = index.data(QtCore.Qt.ItemDataRole.BackgroundRole)
        
        # Check if item is selected
        is_selected = option.state & QtWidgets.QStyle.StateFlag.State_Selected
        
        if bg_brush:
            base_color = bg_brush.color()
            if is_selected:
                # Make background lighter when selected (add 40 to each RGB component)
                lighter_color = QtGui.QColor(
                    min(255, base_color.red() - 20),
                    min(255, base_color.green() - 20),
                    min(255, base_color.blue() - 20),
                    base_color.alpha()
                )
                painter.fillRect(option.rect, lighter_color)
            else:
                painter.fillRect(option.rect, bg_brush)
        elif is_selected:
            # Fallback to standard selection color if no background
            painter.fillRect(option.rect, option.palette.highlight())
        
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
                highlight_format = QtGui.QTextCharFormat()
                highlight_format.setBackground(QtGui.QBrush(QtGui.QColor(0, 200, 0, 140)))
                highlight_format.setForeground(QtGui.QBrush(QtCore.Qt.GlobalColor.white)) # Optional: ensure text is readable

                # APPLY FORMAT TO EACH MATCH
                for pos, length in highlights:
                    cursor = QtGui.QTextCursor(doc)
                    cursor.setPosition(pos)
                    cursor.setPosition(pos + length, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    cursor.mergeCharFormat(highlight_format)

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

    def setValueInstant(self, value):
            """Force set value immediately, cancelling any animation."""
            if self._animation.state() == QtCore.QPropertyAnimation.State.Running:
                self._animation.stop()
            super().setValue(value)


class SmoothScrollTableView(QtWidgets.QTableView):
    """QTableView with smooth animated scrolling for both wheel and scrollbar interactions
    
    Features:
    - Animated scrolling via AnimatedScrollBar
    - Wheel event smoothing
    - 2D Middle-click autoscroll (NEW) - hold middle button and move mouse to scroll in any direction
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable per-pixel scrolling
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        
        # Replace default scrollbars with animated versions
        v_scrollbar = AnimatedScrollBar(QtCore.Qt.Orientation.Vertical, self)
        h_scrollbar = AnimatedScrollBar(QtCore.Qt.Orientation.Horizontal, self)
        
        self.setVerticalScrollBar(v_scrollbar)
        self.setHorizontalScrollBar(h_scrollbar)
        
        # --- 2D Middle-click autoscroll support ---
        self._autoscroll_active = False
        self._autoscroll_origin = None
        self._autoscroll_speed_x = 0
        self._autoscroll_speed_y = 0
        self._autoscroll_deadzone = 15  # pixels before scrolling starts
        self._autoscroll_sensitivity = 0.35  # INCREASED for faster scrolling
        
        # Timer for smooth autoscroll updates
        self._autoscroll_timer = QtCore.QTimer()
        self._autoscroll_timer.setInterval(16)  # ~60fps
        self._autoscroll_timer.timeout.connect(self._update_autoscroll)
        
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        # Any button press while autoscrolling should stop it
        if self._autoscroll_active:
            self._stop_autoscroll()
            
        # Middle button starts 2D autoscroll
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            v_scrollbar = self.verticalScrollBar()
            h_scrollbar = self.horizontalScrollBar()
            
            # Only enable if there's actual content to scroll (vertical OR horizontal)
            has_vertical = v_scrollbar.maximum() > v_scrollbar.minimum()
            has_horizontal = h_scrollbar.maximum() > h_scrollbar.minimum()
            
            if has_vertical or has_horizontal:
                self._autoscroll_active = True
                self._autoscroll_origin = event.position().toPoint()
                self.viewport().setCursor(QtCore.Qt.CursorShape.SizeAllCursor)  # 2D cursor
                self._autoscroll_timer.start()
                event.accept()
                return
            
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        # Middle button stops autoscroll
        if event.button() == QtCore.Qt.MouseButton.MiddleButton and self._autoscroll_active:
            self._stop_autoscroll()
            event.accept()
            return
            
        super().mouseReleaseEvent(event)
        
    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        # Update autoscroll speed based on mouse position (2D)
        if self._autoscroll_active and self._autoscroll_origin:
            # Calculate distance from origin for both axes
            delta_x = event.position().x() - self._autoscroll_origin.x()
            delta_y = event.position().y() - self._autoscroll_origin.y()
            
            # Apply deadzone to horizontal
            if abs(delta_x) < self._autoscroll_deadzone:
                self._autoscroll_speed_x = 0
            else:
                effective_delta_x = delta_x - (self._autoscroll_deadzone * (1 if delta_x > 0 else -1))
                self._autoscroll_speed_x = effective_delta_x * self._autoscroll_sensitivity
            
            # Apply deadzone to vertical
            if abs(delta_y) < self._autoscroll_deadzone:
                self._autoscroll_speed_y = 0
            else:
                effective_delta_y = delta_y - (self._autoscroll_deadzone * (1 if delta_y > 0 else -1))
                self._autoscroll_speed_y = effective_delta_y * self._autoscroll_sensitivity
            
        super().mouseMoveEvent(event)
        
    def leaveEvent(self, event: QtCore.QEvent):
        # Stop autoscroll if mouse leaves viewport
        if self._autoscroll_active:
            self._stop_autoscroll()
        super().leaveEvent(event)
        
    def _update_autoscroll(self):
        """Update scroll position based on autoscroll speed (both axes)"""
        if not self._autoscroll_active:
            return
            
        # Update vertical scrollbar - bypass animation
        if self._autoscroll_speed_y != 0:
            v_scrollbar = self.verticalScrollBar()
            current_value_y = v_scrollbar.value()
            new_value_y = current_value_y + self._autoscroll_speed_y
            new_value_y = max(v_scrollbar.minimum(), min(new_value_y, v_scrollbar.maximum()))
            
            # Stop any ongoing animation and set directly
            if hasattr(v_scrollbar, '_animation'):
                v_scrollbar._animation.stop()
            QtWidgets.QScrollBar.setValue(v_scrollbar, int(new_value_y))  # Call parent class method
        
        # Update horizontal scrollbar - bypass animation
        if self._autoscroll_speed_x != 0:
            h_scrollbar = self.horizontalScrollBar()
            current_value_x = h_scrollbar.value()
            new_value_x = current_value_x + self._autoscroll_speed_x
            new_value_x = max(h_scrollbar.minimum(), min(new_value_x, h_scrollbar.maximum()))
            
            # Stop any ongoing animation and set directly
            if hasattr(h_scrollbar, '_animation'):
                h_scrollbar._animation.stop()
            QtWidgets.QScrollBar.setValue(h_scrollbar, int(new_value_x))  # Call parent class method
            
    def _stop_autoscroll(self):
        """Stop autoscroll and reset state"""
        if not self._autoscroll_active:
            return
            
        self._autoscroll_active = False
        self._autoscroll_origin = None
        self._autoscroll_speed_x = 0
        self._autoscroll_speed_y = 0
        self._autoscroll_timer.stop()
        self.viewport().unsetCursor()
        self.viewport().update()  # Clear any marker
        
    def paintEvent(self, event: QtGui.QPaintEvent):
        """Custom paint to draw 2D autoscroll marker when active"""
        super().paintEvent(event)
        
        if self._autoscroll_active and self._autoscroll_origin:
            painter = QtGui.QPainter(self.viewport())
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            
            marker_size = 20
            rect = QtCore.QRect(
                self._autoscroll_origin.x() - marker_size // 2,
                self._autoscroll_origin.y() - marker_size // 2,
                marker_size,
                marker_size
            )
            
            # Draw crosshair circle
            painter.setPen(QtGui.QPen(QtGui.QColor(100, 150, 255, 180), 2))
            painter.setBrush(QtGui.QColor(100, 150, 255, 60))
            painter.drawEllipse(rect)
            
            # Draw direction arrows with higher visibility
            arrow_size = 7  # Slightly larger
            painter.setBrush(QtGui.QColor(100, 200, 255, 220))  # Brighter
            
            # Vertical arrow
            if abs(self._autoscroll_speed_y) > 2:
                if self._autoscroll_speed_y < 0:  # Up arrow
                    points = [
                        QtCore.QPoint(self._autoscroll_origin.x(), self._autoscroll_origin.y() - marker_size // 2 - 3),
                        QtCore.QPoint(self._autoscroll_origin.x() - arrow_size, self._autoscroll_origin.y() - marker_size // 2 + 4),
                        QtCore.QPoint(self._autoscroll_origin.x() + arrow_size, self._autoscroll_origin.y() - marker_size // 2 + 4)
                    ]
                else:  # Down arrow
                    points = [
                        QtCore.QPoint(self._autoscroll_origin.x(), self._autoscroll_origin.y() + marker_size // 2 + 3),
                        QtCore.QPoint(self._autoscroll_origin.x() - arrow_size, self._autoscroll_origin.y() + marker_size // 2 - 4),
                        QtCore.QPoint(self._autoscroll_origin.x() + arrow_size, self._autoscroll_origin.y() + marker_size // 2 - 4)
                    ]
                painter.drawPolygon(QtGui.QPolygon(points))
            
            # Horizontal arrow
            if abs(self._autoscroll_speed_x) > 2:
                if self._autoscroll_speed_x < 0:  # Left arrow
                    points = [
                        QtCore.QPoint(self._autoscroll_origin.x() - marker_size // 2 - 3, self._autoscroll_origin.y()),
                        QtCore.QPoint(self._autoscroll_origin.x() - marker_size // 2 + 4, self._autoscroll_origin.y() - arrow_size),
                        QtCore.QPoint(self._autoscroll_origin.x() - marker_size // 2 + 4, self._autoscroll_origin.y() + arrow_size)
                    ]
                else:  # Right arrow
                    points = [
                        QtCore.QPoint(self._autoscroll_origin.x() + marker_size // 2 + 3, self._autoscroll_origin.y()),
                        QtCore.QPoint(self._autoscroll_origin.x() + marker_size // 2 - 4, self._autoscroll_origin.y() - arrow_size),
                        QtCore.QPoint(self._autoscroll_origin.x() + marker_size // 2 - 4, self._autoscroll_origin.y() + arrow_size)
                    ]
                painter.drawPolygon(QtGui.QPolygon(points))
        
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
        self.controls.backRequested.connect(lambda: self.app_ref.back_to_start())
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
        toolbar.addSeparator()

        self.btn_save = QtGui.QAction("Save", self)
        self.btn_save.triggered.connect(self.save_file)
        toolbar.addAction(self.btn_save)
        toolbar.addSeparator()

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
        #toolbar.addWidget(self.mode_toggle) # we don't really need this shit (dark is fine).
        #toolbar.addSeparator()

        # --- Search Field ---
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search keyword…")
        self.search_edit.setMinimumWidth(150)
        self.search_edit.returnPressed.connect(self.on_search_return_pressed)
        self.search_edit.installEventFilter(self)
        toolbar.addWidget(self.search_edit)
        
        # --- Search Button (magnifier) ---
        self.search_btn = QtWidgets.QToolButton()
        self.search_btn.setText("🔍")
        self.search_btn.setToolTip("Search")
        self.search_btn.clicked.connect(self.on_search_return_pressed)
        toolbar.addWidget(self.search_btn)

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

        self.match_label = QtWidgets.QLabel("Matching cells: 0")
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
        
        self.table.horizontalHeader().sectionClicked.connect(self._select_column)
        self.table.verticalHeader().sectionClicked.connect(self._select_row)
        
        
                # === NEW: Visual feedback for column dragging ===
        self.table.horizontalHeader().setDragDropOverwriteMode(False)
        self.table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self._rename_column)
        self.table.clicked.connect(self._handle_plus_column_click)
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

        self.dirty_indicator = QtWidgets.QLabel("")
        self.dirty_indicator.setStyleSheet("""
            QLabel {
                color: #FFC107;
                font-weight: 600;
                padding: 2px 8px;
            }
        """)
        self.dirty_indicator.hide()
        self.status.addPermanentWidget(self.dirty_indicator)  # Right side of status bar

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
                
    def reset_viewer(self):
        """Clear all loaded data and return to drag-and-drop screen"""
        # Clear the model
        if self.model:
            self.table.setModel(None)
            self.model = None
        
        # Reset UI state
        self.current_path = None
        self.current_match_pos = -1
        self.search_edit.clear()
        self.replace_edit.clear()
        self.replace_container.setVisible(False)
        
        # Reset window title
        self.controls.title_lbl.setText("Viewer")
        self.window().setWindowTitle(APP_NAME)
        
        # Reset indicators
        self.match_label.setText("Matching cells: 0")
        self.status.clearMessage()
        self.dirty_indicator.hide()
        self._hide_background_indicator()
        
        # Show drag-and-drop, hide table
        self.table.hide()
        self.dd.show()
        self.dd.icon.setText("🗂️Drop a CSV / XLS / XLSX / JSONL file here")     
        
               
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
            self.model.is_dirty = True
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
        search_text = self.search_edit.text()
        if not self.model:
            return
            
        current_model_term = self.model.search_term or ""
        
        # CHECK: Is this a new search term?
        if search_text != current_model_term:
            # --- CASE 1: New Term (First Enter) ---
            
            # A. Capture current scroll position (Pixel-perfect)
            current_v_scroll = self.table.verticalScrollBar().value()
            
            # B. Update the Search (Highlights appear)
            self.model.set_search_term(search_text)
            
            # C. Show Replace bar
            self.replace_container.setVisible(bool(search_text.strip()))
            self._update_replace_buttons()
            
            # D. FORCE INSTANT RESTORE (No animation, no jumping)
            sb = self.table.verticalScrollBar()
            if hasattr(sb, 'setValueInstant'):
                sb.setValueInstant(current_v_scroll)
            else:
                sb.setValue(current_v_scroll)
            
            # E. Smart Start: Set to -1 so next press goes to first match
            self.current_match_pos = -1
        
        # Always navigate to next match when Enter is pressed
        self.go_next_match()
               
    def _select_column(self, logical_index: int):
        """Select entire column when header is clicked"""
        if not self.model:
            return
        
        # Check if the clicked header is the "+" column
        if logical_index == len(self.model.df.columns):
            # Trigger the New Column dialog (same logic as clicking the cells)
            new_name, ok = QtWidgets.QInputDialog.getText(
                self, "New Column", "Enter column name:"
            )
            if ok and new_name.strip():
                self.model.add_new_column(new_name.strip())
                self.autosize_columns(force=True)
                self._autosize_all_rows()
                self.status.showMessage(f"Added column: {new_name}")
            return
        
        self.table.selectColumn(logical_index)

    def _select_row(self, logical_index: int):
        """Select entire row when row header is clicked"""
        if not self.model:
            return
        
        self.table.selectRow(logical_index)
    
    
    def load_path(self, path: Path):
        self._hide_background_indicator()
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
        filename = path.name
        self.controls.title_lbl.setText(filename)
        self.window().setWindowTitle(f"{filename}")
        
        self.model = DataFrameModel(df)
        self.model._is_dark = self.mode_toggle.isChecked()
        self.model.dataChangedHard.connect(self.table.viewport().update)
        self.model.matchesChanged.connect(self._on_matches_changed)
        self.model.dirtyStateChanged.connect(self.update_dirty_indicator)
        self.update_dirty_indicator()
        
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
        self.window().showMaximized()

    def open_file_dialog(self):
        """Open new file with unsaved changes check"""
        # Check dirty state BEFORE closing
        if self.model and self.model.is_dirty:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before opening a new file?",
                (QtWidgets.QMessageBox.StandardButton.Yes | 
                QtWidgets.QMessageBox.StandardButton.No | 
                QtWidgets.QMessageBox.StandardButton.Cancel)
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                return  # Abort - keep current file open
            elif reply == QtWidgets.QMessageBox.StandardButton.Yes:
                if not self.save_file():
                    return  # Abort if save failed
        
        # Only proceed if user didn't cancel
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
            return False
        try:
            self._write_dataframe(self.current_path, self.model.df)
            self.model.is_dirty = False
            self.status.showMessage(f"Saved to {self.current_path}")
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
            return False
    
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
            
            # Smooth scroll to replaced cell
            row = idx.row()
            v_header = self.table.verticalHeader()
            row_top = v_header.sectionPosition(row)
            row_height = v_header.sectionSize(row)
            viewport_height = self.table.viewport().height()
            
            target_v_scroll = row_top - (viewport_height - row_height) // 2
            
            v_sb = self.table.verticalScrollBar()
            target_v_scroll = max(0, min(target_v_scroll, v_sb.maximum()))
            
            v_sb.setValue(target_v_scroll)
            
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
        self.match_label.setText(f"Matching cells: {current} / {total}")
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

    def eventFilter(self, obj, event):
        """Capture Shift+Enter in search box for backwards navigation"""
        if obj == self.search_edit and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    # Shift+Enter: go to previous match
                    self.go_prev_match()
                    return True  # Event handled
        return super().eventFilter(obj, event)

    def _scroll_to_current(self):
        idx = self.model.match_at(self.current_match_pos)
        if idx:
            row = idx.row()
            col = idx.column()
            
            # === VERTICAL SCROLLING (Animated) ===
            v_header = self.table.verticalHeader()
            row_top = v_header.sectionPosition(row)
            row_height = v_header.sectionSize(row)
            viewport_height = self.table.viewport().height()
            
            target_v_scroll = row_top - (viewport_height - row_height) // 2
            v_sb = self.table.verticalScrollBar()
            target_v_scroll = max(0, min(target_v_scroll, v_sb.maximum()))
            
            # Use animated setValue (this triggers AnimatedScrollBar's animation)
            v_sb.setValue(int(target_v_scroll))
            
            # === HORIZONTAL SCROLLING (Animated) ===
            h_header = self.table.horizontalHeader()
            col_left = h_header.sectionPosition(col)
            col_width = h_header.sectionSize(col)
            viewport_width = self.table.viewport().width()
            
            target_h_scroll = col_left - (viewport_width - col_width) // 2
            h_sb = self.table.horizontalScrollBar()
            target_h_scroll = max(0, min(target_h_scroll, h_sb.maximum()))
            
            # Use animated setValue for horizontal too
            h_sb.setValue(int(target_h_scroll))
            
            self.table.setCurrentIndex(idx)
            self._on_matches_changed(self.model.total_matches())

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Existing Enter key logic
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self.go_prev_match()
            else:
                self.go_next_match()
            return
        
        # === NEW: Handle Delete Key ===
        if event.key() == QtCore.Qt.Key.Key_Delete:
            self._handle_delete_key()
            return
            
        super().keyPressEvent(event)
        
    def _handle_delete_key(self):
            """Handles deletion of Columns, Rows, or Cell Contents based on selection."""
            if not self.model:
                return

            # --- Case 1: Delete Selected Columns ---
            sel_cols = self.table.selectionModel().selectedColumns()
            if sel_cols:
                # Get indices and filter out the "+" column
                col_indices = [c.column() for c in sel_cols]
                valid_indices = [i for i in col_indices if i < len(self.model._df.columns)]
                
                if not valid_indices:
                    return

                msg = f"Delete {len(valid_indices)} column(s)?"
                reply = QtWidgets.QMessageBox.question(
                    self, "Delete Columns", msg,
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
                )
                
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.model.layoutAboutToBeChanged.emit()
                    # Drop columns by name
                    cols_to_drop = [self.model._df.columns[i] for i in valid_indices]
                    self.model._df.drop(columns=cols_to_drop, inplace=True)
                    self.model.is_dirty = True
                    self.model.layoutChanged.emit()
                return

            # --- Case 2: Delete Selected Rows ---
            sel_rows = self.table.selectionModel().selectedRows()
            if sel_rows:
                row_indices = [r.row() for r in sel_rows]
                
                msg = f"Delete {len(row_indices)} row(s)?"
                reply = QtWidgets.QMessageBox.question(
                    self, "Delete Rows", msg,
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
                )
                
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.model.layoutAboutToBeChanged.emit()
                    # Drop rows by index
                    self.model._df.drop(self.model._df.index[row_indices], inplace=True)
                    # Reset index to keep row numbers sequential
                    self.model._df.reset_index(drop=True, inplace=True)
                    self.model.is_dirty = True
                    self.model.layoutChanged.emit()
                    
                    # Refresh search matches since row indices have shifted
                    if self.model.search_term:
                        self.model._rebuild_matches()
                return

            # --- Case 3: Clear Cell Contents (Fallback) ---
            # If no full row/col is selected, just empty the selected cells
            if self.table.selectionModel().hasSelection():
                self.model.layoutAboutToBeChanged.emit()
                for idx in self.table.selectedIndexes():
                    if idx.isValid() and idx.column() < len(self.model._df.columns):
                        # Set value to empty string
                        self.model._df.iat[idx.row(), idx.column()] = ""
                self.model.is_dirty = True
                self.model.layoutChanged.emit()
            
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
        
        for c in range(len(self.model.df.columns)):
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
            self._resize_row_with_padding(row)
        
        # Clear the tracking set
        self._columns_resized.clear()
        
        # Background full update for very large tables (>1000 rows)
        if self.model.rowCount() > 1000:
            QtCore.QTimer.singleShot(800, self._background_resize_rows)
            
    def _resize_row_with_padding(self, row: int):
        """
        Resize row to fit content and add one extra line of space for better readability.
        This ensures all text is visible with comfortable padding at the bottom.
        """
        # Resize to fit content first
        self.table.resizeRowToContents(row)
        
        # Calculate height of one text line based on current font
        font = self.table.font()
        metrics = QtGui.QFontMetrics(font)
        line_height = metrics.lineSpacing() * 2
        
        # Add one extra line of space
        current_height = self.table.rowHeight(row)
        new_height = current_height + line_height
        self.table.setRowHeight(row, new_height)
        
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
                    self._resize_row_with_padding(row)
                
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
                self._resize_row_with_padding(row)

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
                    self._resize_row_with_padding(row)
                    
    # === NEW: Full recalculation on initial load ===
    def _autosize_all_rows(self):
        """
        Optimized: Sets a default height for all rows instantly, then uses 
        Pandas vectorization to find ONLY the complex rows that need 
        calculation.
        """
        if not self.model:
            return
        
        row_count = self.model.rowCount()
        if row_count == 0:
            return

        # 1. Set global default height (Instant)
        # This handles 90-99% of rows without ANY calculation overhead
        font = self.table.font()
        metrics = QtGui.QFontMetrics(font)
        default_height = metrics.lineSpacing() + 8
        self.table.verticalHeader().setDefaultSectionSize(default_height)
        
        # 2. Identify "Complex" rows using Vectorized Pandas (Fast)
        # We only care about columns that are 'object' type (text)
        text_cols = self.model.df.select_dtypes(include=['object']).columns
        
        complex_rows_set = set()
        
        if len(text_cols) > 0:
            # Show busy cursor if file is huge
            if row_count > 10000:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

            try:
                # Check each text column for newlines or excessive length
                # We do this column by column to save memory
                for col in text_cols:
                    series = self.model.df[col].dropna().astype(str)
                    
                    # Criteria: Contains newline OR is longer than 100 chars
                    # We get the boolean mask and then the indices
                    mask = (series.str.contains('\n')) | (series.str.len() > 100)
                    
                    # Add indices of complex rows to our set
                    complex_rows_set.update(series[mask].index.tolist())
            finally:
                 if row_count > 10000:
                    QtWidgets.QApplication.restoreOverrideCursor()

        # Convert to a sorted list for processing
        self._pending_resize_rows = sorted(list(complex_rows_set))
        total_complex = len(self._pending_resize_rows)
        
        # If very few complex rows, just do it now and finish
        if total_complex < 50:
            for row in self._pending_resize_rows:
                self.table.resizeRowToContents(row)
            return

        # 3. Setup Background Processing for the Complex Rows ONLY
        self._show_background_indicator("Autosizing rows", total_complex, 0)
        
        # Process first batch immediately to give instant feedback
        BATCH_SIZE = 20
        initial_batch = self._pending_resize_rows[:BATCH_SIZE]
        self._pending_resize_rows = self._pending_resize_rows[BATCH_SIZE:]
        
        for row in initial_batch:
            self.table.resizeRowToContents(row)
            
        # Schedule the rest
        if self._pending_resize_rows:
            QtCore.QTimer.singleShot(50, self._background_resize_remaining)
        else:
            self._hide_background_indicator()

    def _background_resize_remaining(self):
        """
        Optimized: Process the queue of specific rows that need resizing.
        Does NOT iterate through the whole table, only the necessary rows.
        """
        # Safety check: ensure the queue exists and model is loaded
        if not self.model or not hasattr(self, '_pending_resize_rows') or not self._pending_resize_rows:
            self._hide_background_indicator()
            return
            
        # Process in batches
        BATCH_SIZE = 50 
        
        # Take the next batch of indices from the queue
        current_batch = self._pending_resize_rows[:BATCH_SIZE]
        self._pending_resize_rows = self._pending_resize_rows[BATCH_SIZE:]
        
        # Resize only these specific rows
        for row in current_batch:
            self.table.resizeRowToContents(row)
            
        # Update progress based on how many are LEFT
        if self.bg_total_rows > 0:
            remaining = len(self._pending_resize_rows)
            done = self.bg_total_rows - remaining
            self.bg_current_row = done
            self._update_bg_indicator()
        
        # If there are still rows left, schedule the next batch
        if self._pending_resize_rows:
            # Short delay (10ms) to keep UI responsive
            QtCore.QTimer.singleShot(10, self._background_resize_remaining)
        else:
            self._hide_background_indicator()
            self.status.showMessage(f"Finished formatting.")    
        
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
                
    def _handle_plus_column_click(self, index: QtCore.QModelIndex):
        """Handle clicks on the '+' column to add a new column"""
        if not self.model:
            return
        
        if index.column() == self.model.columnCount() - 1:
            new_name, ok = QtWidgets.QInputDialog.getText(
                self, "New Column", "Enter column name:"
            )
            if ok and new_name.strip():
                self.model.add_new_column(new_name.strip())
                self.autosize_columns(force=True)
                self._autosize_all_rows()
                self.status.showMessage(f"Added column: {new_name}")

    def _rename_column(self, logical_index: int):
        """Rename a column via double-click on header"""
        if not self.model or logical_index >= len(self.model.df.columns):
            return
        
        current_name = self.model.df.columns[logical_index]
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Column", f"Rename '{current_name}' to:", text=current_name
        )
        if ok and new_name.strip() and new_name != current_name:
            self.model.df.rename(columns={current_name: new_name}, inplace=True)
            self.model.is_dirty = True
            self.model.headerDataChanged.emit(
                QtCore.Qt.Orientation.Horizontal, logical_index, logical_index
            )
            self.status.showMessage(f"Renamed column to: {new_name}")
        
    def update_dirty_indicator(self, is_dirty: bool = None):
        """Update the unsaved changes indicator based on model state"""
        if not self.model:
            self.dirty_indicator.hide()
            return
        
        if is_dirty is None:
            is_dirty = self.model.is_dirty
        
        if is_dirty:
            self.dirty_indicator.setText("⚠️ Unsaved changes")
            self.dirty_indicator.show()
        else:
            self.dirty_indicator.hide()
            
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
        self.controls.closeRequested.connect(self.window().close)

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
        """Synchronize text selection using Regex for multi-line and forced cleanup for stuck highlights."""
        if self._syncing_selection:
            return

        self._syncing_selection = True
        try:
            # --- FIX 1: STUCK SELECTION ---
            # Immediately clear highlights on BOTH sides. 
            # This ensures that if you just clicked to deselect (hasSelection is False),
            # the old artifacts are removed immediately.
            source_edit.setExtraSelections([])
            target_edit.setExtraSelections([])

            cursor = source_edit.textCursor()
            if not cursor.hasSelection():
                return

            text = cursor.selectedText()
            if not text.strip():
                return

            # Detect if selection is in diff-only region (red/green background)
            start_cursor = QtGui.QTextCursor(cursor)
            start_cursor.setPosition(cursor.selectionStart())
            char_format = start_cursor.charFormat()
            bg_color = char_format.background().color()

            # Check strictly for the diff colors defined in run_compare
            is_red = bg_color.red() > 150 and bg_color.green() < 80 and bg_color.blue() < 80
            is_green = bg_color.green() > 120 and bg_color.red() < 80 and bg_color.blue() < 80

            if is_red or is_green:
                return

            # --- FIX 2: MULTI-LINE SELECTION ---
            # Split selection into words and create a Regex pattern.
            # This finds the words regardless of whether the gap is a space, \n, or \u2029
            parts = text.split()
            if not parts:
                return
                
            # Escape parts to handle special chars like brackets or dots in the text
            safe_parts = [QtCore.QRegularExpression.escape(p) for p in parts]
            
            # Join words with \s+ (matches any whitespace sequence including newlines)
            pattern = r"\s+".join(safe_parts)
            regex = QtCore.QRegularExpression(pattern)
            
            target_doc = target_edit.document()
            
            # Determine search direction
            sel_start = cursor.selectionStart()
            source_mid = source_edit.document().characterCount() // 2

            if sel_start < source_mid:
                find_cursor = target_doc.find(regex, 0)
            else:
                text_length = target_doc.characterCount()
                find_cursor = target_doc.find(regex, text_length - 1,
                                              QtGui.QTextDocument.FindFlag.FindBackward)

            if find_cursor.isNull():
                return

            # Scroll alignment
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
                    desired_y = source_rel_y * target_viewport_height
                    delta_y = target_rect.top() - desired_y
                    new_scroll = current_scroll + delta_y
                    new_scroll = max(0, min(new_scroll, target_scrollbar.maximum()))
                    target_scrollbar.setValue(int(new_scroll))

            # Apply blue highlight
            selection = QtWidgets.QTextEdit.ExtraSelection()
            selection.cursor = find_cursor
            selection.format.setBackground(QtGui.QColor(0, 150, 255, 100))
            selection.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, False)

            target_edit.setExtraSelections([selection])
            
        finally:
            # Ensure flag is reset even if we returned early
            self._syncing_selection = False
    def _clear_selection(self, edit: QtWidgets.QTextEdit):
        """Clear extra selection highlights in the given editor"""
        if self._syncing_selection:
            return
        self._syncing_selection = True
        edit.setExtraSelections([])
        edit.viewport().update()
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
        import difflib
        import re

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
        self.controls = WindowControls(APP_NAME)
        self.controls.back_btn.hide()
        self.controls.minimizeRequested.connect(lambda: self.window().showMinimized())
        self.controls.maximizeRestoreRequested.connect(self._toggle_max_restore)
        self.controls.closeRequested.connect(QtWidgets.QApplication.instance().quit)
        
        # Content container for centered elements
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # Welcome label
        lbl = QtWidgets.QLabel("Choose your mode")
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
        
        self._last_screen = None
        self._screen_change_timer = QtCore.QTimer()
        self._screen_change_timer.setSingleShot(True)
        self._screen_change_timer.setInterval(100)
        self._is_initial_move = True

    def closeEvent(self, event):
        """Intercept close to check for unsaved changes"""
        current_widget = self.centralWidget().currentWidget()
        
        if current_widget == self.viewer and self.viewer.model and self.viewer.model.is_dirty:
            reply = QtWidgets.QMessageBox.question(
                self, 
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                (QtWidgets.QMessageBox.StandardButton.Yes | 
                QtWidgets.QMessageBox.StandardButton.No | 
                QtWidgets.QMessageBox.StandardButton.Cancel)
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            elif reply == QtWidgets.QMessageBox.StandardButton.Yes:
                if not self.viewer.save_file():
                    event.ignore()
                    return
        
        event.accept()
        
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

    
    
    def back_to_start(self):
        """Return to start screen with unsaved changes check"""
        # Check if in viewer with unsaved changes
        if self.centralWidget().currentWidget() == self.viewer and self.viewer.model and self.viewer.model.is_dirty:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before leaving?",
                (QtWidgets.QMessageBox.StandardButton.Yes | 
                QtWidgets.QMessageBox.StandardButton.No | 
                QtWidgets.QMessageBox.StandardButton.Cancel)
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                return  # Abort - stay on current screen
            elif reply == QtWidgets.QMessageBox.StandardButton.Yes:
                if not self.viewer.save_file():
                    return  # Abort if save failed
        
        # Clear viewer state (NEW)
        self.viewer.reset_viewer()
        
        # Only proceed if user didn't cancel
        self.centralWidget().setCurrentWidget(self.start)
        self.showNormal()
        self.resize(520, 320)
        self.center_on_screen()
        

        
    def moveEvent(self, event):
        """Detect when window moves and check for screen changes"""
        super().moveEvent(event)
        # Trigger screen change check with debounce
        self._screen_change_timer.start()

    def enter_viewer(self):
        self.centralWidget().setCurrentWidget(self.viewer)
        # --- CHANGE: Use maximized instead of full-screen to avoid DPI issues ---
        if self.viewer.model is not None:
            self.showMaximized()  # Better than fullScreen for multi-screen
        else:
            self.showNormal()
            self.resize(520, 320)
            self.center_on_screen()

    def enter_diff(self):
        self.centralWidget().setCurrentWidget(self.diff)
        self.showMaximized()  # Better than fullScreen for multi-screen

def main():
    from PyQt6 import QtCore
    
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.Round
    )
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
