# BetterExcel.py

Single-file (~11,300 lines) PyQt6 desktop spreadsheet viewer/editor. Built with PyInstaller `--onefile` in mind.

## Stack
- **UI**: PyQt6 (frameless window, custom title bar, dark palette default)
- **Data**: pandas DataFrames as the internal data model
- **XLSX**: openpyxl (lazy-loaded) for rich text, cell styles, fonts, fills
- **Google Sheets**: google-api-python-client + google-auth (lazy-loaded)
- **Collab**: raw TCP sockets + threading for LAN collaboration

## Architecture (all in BetterExcel.py)

Entry: `main()` → `MainWindow` (QStackedWidget with 3 pages).

### Pages
| Class | Purpose |
|---|---|
| `StartPage` | Mode chooser: "View File" or "Compare Texts" |
| `DataViewerPage` | Main spreadsheet viewer/editor (~5000 lines, line 5855) |
| `DiffPage` | Side-by-side text comparison with live similarity stats |

### Key Classes
| Class | Line | Role |
|---|---|---|
| `DataFrameModel` | 3298 | QAbstractTableModel wrapping pandas DataFrame + cell styles + QA spans + edit metadata |
| `ColumnStatsHeader` | 4897 | Custom QHeaderView — per-column profiling (histograms, type counts, missing %). Animated collapse. |
| `HighlightDelegate` | 4056 | QStyledItemDelegate — renders search highlights, QA error spans, remote presence cursors, cell background colors |
| `QAContextTextEdit` | 3812 | Rich text editor panel for reviewing/annotating cell content with QA markup |
| `CollaborationServer` | 2055 | TCP socket host — accepts guests, broadcasts snapshots |
| `CollaborationClient` | 2261 | TCP socket guest — receives snapshots, sends edits |
| `DragDropWidget` | 4324 | Drop zone for CSV/XLS/XLSX/JSONL files |
| `WindowControls` | 4379 | Custom frameless title bar (back, share, summarize, min/max/close) |
| `SmoothScrollTableView` | 4553 | QTableView with animated scrollbar and row height caching |

### Major Feature Blocks
- **File I/O** (lines ~9440–9600): open/save/save-copy, supports CSV/XLS/XLSX/JSONL
- **XLSX styles** (lines ~3126–3290): `load_xlsx_dataframe_with_styles`, `write_xlsx_with_styles` — round-trips cell formatting through openpyxl
- **Google Sheets sync** (lines ~280–1415): OAuth2 flow, bidirectional sync with 3-way merge, presence/cursor sharing via hidden metadata sheets (`__betterexcel_*__`)
- **LAN collaboration** (lines ~2055–2410): host/guest TCP sessions with `betterexcel://` URI links and token auth
- **QA annotations** (lines ~2680–3090): mark text as big/small errors, serialize to XLSX hidden sheets, rich text rendering
- **Column stats** (lines ~4897–5855): profiling engine with histograms, type detection, missing value analysis
- **Search/replace**: regex-capable find with match-by-match navigation
- **LLM summarize**: sends QA annotations to an LLM via `SUMMARIZE_SYSTEM_PROMPT`

## Key Constants (top of file)
- `APP_NAME = "Better Excel v2.1"`
- `COMPACT_WINDOW_WIDTH/HEIGHT` = 470 x 430 (start page size)
- `QA_METADATA_SHEET_NAME`, `EDIT_METADATA_SHEET_NAME`, `CELL_STYLES_METADATA_SHEET_NAME`, `PRESENCE_METADATA_SHEET_NAME` — hidden XLSX sheet names for persisting metadata
- `GOOGLE_SHEETS_*` — sync intervals, batch sizes, retry config
- `COLLAB_*` — LAN collaboration protocol settings

## Keyboard Shortcuts
- `Ctrl+F` — focus search
- `Ctrl+S` — save
- `Ctrl+O` — open file
- `Esc` — exit fullscreen

## Lazy Imports
openpyxl, Google API libs, and pandas are all lazy-loaded via `_ensure_*_loaded()` functions. This keeps startup fast and allows the app to run with reduced functionality if optional deps are missing.

## Build
```
pyinstaller --noconsole --onefile --clean --strip --icon=BetterExcel.ico --optimize=2 BetterExcel.py
```
