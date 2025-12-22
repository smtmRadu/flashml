def inspect_tokenizer_window(tokenizer):
    import tkinter as tk
    from tkinter import ttk
    import threading
    import queue
    import time
    import math

    # --- Dark Mode Style (unchanged) ---
    BG_COLOR = "#2E2E2E"
    FG_COLOR = "#FFFFFF"
    TEXT_BG_COLOR = "#3C3C3C"
    BUTTON_BG_COLOR = "#555555"
    BUTTON_FG_COLOR = "#FFFFFF"
    HIGHLIGHT_COLOR = "#4A4A4A"
    ENTRY_BORDER_COLOR = "#777777"

    # --- Create Main Window ---
    window = tk.Tk()
    window.title(f"Tokenizer Inspector: {tokenizer.__class__.__name__}")
    window.configure(bg=BG_COLOR)
    window.geometry("1300x800+0+0")

    # --- Style Configuration ---
    style = ttk.Style(window)
    style.theme_use("clam")
    style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, padding=5, font=("Segoe UI", 10))
    style.configure("TFrame", background=BG_COLOR)
    style.configure("TButton", background=BUTTON_BG_COLOR, foreground=BUTTON_FG_COLOR, font=("Segoe UI", 10, "bold"), borderwidth=1)
    style.map("TButton", background=[("active", HIGHLIGHT_COLOR)])
    style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
    style.configure("TSeparator", background=HIGHLIGHT_COLOR)
    style.configure("TEntry", fieldbackground=TEXT_BG_COLOR, foreground=FG_COLOR, insertcolor=FG_COLOR,
                    bordercolor=ENTRY_BORDER_COLOR, lightcolor=TEXT_BG_COLOR, darkcolor=TEXT_BG_COLOR, padding=5)
    style.configure("Vertical.TScrollbar", background=BUTTON_BG_COLOR, troughcolor=BG_COLOR, arrowcolor=FG_COLOR,
                    bordercolor=HIGHLIGHT_COLOR, lightcolor=BUTTON_BG_COLOR, darkcolor=BUTTON_BG_COLOR)
    style.map("Vertical.TScrollbar", background=[("active", HIGHLIGHT_COLOR)])
    style.configure("Dark.Treeview", background=TEXT_BG_COLOR, fieldbackground=TEXT_BG_COLOR,
                    foreground=FG_COLOR, bordercolor=HIGHLIGHT_COLOR, borderwidth=1)
    style.configure("Dark.Treeview.Heading", background=BUTTON_BG_COLOR, foreground=FG_COLOR,
                    font=("Segoe UI", 10, "bold"))
    style.map("Dark.Treeview.Heading", background=[("active", HIGHLIGHT_COLOR)])

    # --- Main Frame & Panes ---
    main_frame = ttk.Frame(window, padding="10")
    main_frame.pack(expand=True, fill=tk.BOTH)
    left_pane = ttk.Frame(main_frame, padding=(0, 0, 5, 0))
    left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    right_pane = ttk.Frame(main_frame, padding=(5, 0, 0, 0))
    right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    for p in (left_pane, right_pane):
        for i in range(2):
            p.grid_rowconfigure(i, weight=1)
        p.grid_columnconfigure(0, weight=1)

    def add_section_heading(parent, title):
        ttk.Label(parent, text=title, style="Header.TLabel").pack(
            pady=(10, 5), anchor="w"
        )
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(0, 10))

    # --- Left: Basic Info (unchanged) ---
    basic_frame = ttk.Frame(left_pane)
    basic_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
    add_section_heading(basic_frame, "Basic Information")
    canvas = tk.Canvas(basic_frame, bg=BG_COLOR, highlightthickness=0)
    vsb = ttk.Scrollbar(basic_frame, orient="vertical", command=canvas.yview, style="Vertical.TScrollbar")
    content = ttk.Frame(canvas)
    content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)

    basic_info = {
        "Tokenizer Class": tokenizer.__class__.__name__,
        "Name or Path": getattr(tokenizer, "name_or_path", "N/A"),
        "Vocabulary Size": tokenizer.vocab_size,
        "Tokenizer Length": len(tokenizer),
        "Model Max Length": getattr(tokenizer, "model_max_length", "N/A"),
        "Padding Side": getattr(tokenizer, "padding_side", "N/A"),
        "Truncation Side": getattr(tokenizer, "truncation_side", "N/A"),
        "Model Input Names": getattr(tokenizer, "model_input_names", "N/A"),
    }
    for tk_name in ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]:
        val = getattr(tokenizer, f"{tk_name}_token", None)
        vid = getattr(tokenizer, f"{tk_name}_token_id", None)
        if val is not None:
            basic_info[f"{tk_name.upper()} Token"] = (
                f"'{val}' (ID: {vid if vid is not None else 'N/A'})"
            )

    for key, val in basic_info.items():
        row = ttk.Frame(content)
        row.pack(fill=tk.X, pady=1, anchor="w")
        ttk.Label(
            row, text=f"{key}:", font=("Segoe UI", 9, "bold"), wraplength=160
        ).pack(side=tk.LEFT, padx=(0, 5), anchor="nw")
        ttk.Label(
            row, text=str(val), wraplength=320, justify=tk.LEFT, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, anchor="nw")

    # --- Special Tokens Table ---
    special_frame = ttk.Frame(left_pane)
    special_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
    add_section_heading(special_frame, f"Special Tokens ({len(tokenizer) - tokenizer.vocab_size})")
    cols = ("Token", "ID")
    table = ttk.Treeview(
        special_frame, columns=cols, show="headings", height=18, style="Dark.Treeview"
    )
    for c in cols:
        table.heading(c, text=c, anchor="center")
        table.column(c, anchor="center")
    table.pack(fill=tk.BOTH, expand=True)

    uniq = {}
    if hasattr(tokenizer, "all_special_tokens"):
        for t, i in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids):
            uniq[t] = i
    for pfx in ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]:
        t = getattr(tokenizer, f"{pfx}_token", None)
        i = getattr(tokenizer, f"{pfx}_token_id", None)
        if t and i is not None:
            uniq.setdefault(t, i)
    if hasattr(tokenizer, "added_tokens_decoder"):
        for i, obj in tokenizer.added_tokens_decoder.items():
            uniq.setdefault(obj.content, i)
    if not uniq:
        table.insert("", "end", values=("No special tokens found", ""))
    else:
        for t, i in sorted(uniq.items(), key=lambda x: (x[1], x[0])):
            table.insert("", "end", values=(repr(t)[1:-1], i))

    # --- Right: Live Tokenization & Finder ---

    # TOP: Live Tokenization
    live_frame = ttk.Frame(right_pane)
    live_frame.grid(row=0, column=0, sticky="nsew")
    add_section_heading(live_frame, "Live Tokenization")
    live_cont = ttk.Frame(live_frame)
    live_cont.pack(fill=tk.BOTH, expand=True)
    input_hl_text = tk.Text(
        live_cont,
        wrap=tk.WORD,
        height=14,
        relief=tk.FLAT,
        bg=TEXT_BG_COLOR,
        fg=FG_COLOR,
        insertbackground=FG_COLOR,
        font=("Segoe UI", 11),
        borderwidth=1,
        highlightthickness=1,
        highlightbackground=HIGHLIGHT_COLOR,
    )
    input_hl_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
    token_count_label = ttk.Label(live_cont, text="", anchor="w", style="TLabel")
    token_count_label.pack(fill=tk.X, padx=(0,5))

    ids_disp_text = tk.Text(
        live_cont,
        wrap=tk.WORD,
        height=14,
        relief=tk.FLAT,
        bg=TEXT_BG_COLOR,
        fg=FG_COLOR,
        insertbackground=FG_COLOR,
        font=("Courier New", 10),
        borderwidth=1,
        highlightthickness=1,
        highlightbackground=HIGHLIGHT_COLOR,
    )
    ids_disp_text.pack(fill=tk.BOTH, expand=True)
    ids_disp_text.configure(state="disabled")

    # --- Background Tokenization Setup ---
    token_colors = {}
    hover_idx = [None]
    _tokenization_job = None
    _tokenization_queue = queue.Queue(maxsize=1)
    _shutdown_thread = threading.Event()

    def tokenization_worker():
        """Runs tokenization in background thread"""
        while not _shutdown_thread.is_set():
            try:
                text = _tokenization_queue.get(timeout=0.1)
                if text is None:  # Shutdown signal
                    break
                
                start_time = time.time()
                if not text.strip():
                    result = ("empty",)
                else:
                    try:
                        out = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
                        result = ("success", out, len(text))
                    except Exception as e:
                        result = ("error", str(e))
                
                processing_time = time.time() - start_time
                window.after(0, lambda: process_tokenization_result(result, processing_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")

    # Start worker thread
    _tokenization_thread = threading.Thread(target=tokenization_worker, daemon=True)
    _tokenization_thread.start()

    def update_live_hl_tokenize(event=None):
        """Debounced entry point: waits 300ms before tokenizing"""
        nonlocal _tokenization_job
        
        if _tokenization_job:
            window.after_cancel(_tokenization_job)
        
        def queue_tokenization():
            # Clear old queue items
            while not _tokenization_queue.empty():
                _tokenization_queue.get_nowait()
            try:
                _tokenization_queue.put(input_hl_text.get("1.0", "end-1c"), block=False)
            except queue.Full:
                pass
        
        _tokenization_job = window.after(300, queue_tokenization)
        token_count_label.config(text="Processing...")

    def process_tokenization_result(result, processing_time):
        """Update UI with results (runs on main thread)"""
        nonlocal token_colors
        
        # Batch tag removal
        input_hl_text.tag_remove("tok_", "1.0", tk.END)
        for tag in input_hl_text.tag_names():
            if tag.startswith("tok_"):
                input_hl_text.tag_delete(tag)
        
        ids_disp_text.configure(state="normal")
        ids_disp_text.delete("1.0", tk.END)
        token_colors.clear()
        
        if result[0] == "empty":
            ids_disp_text.insert(tk.END, "Token IDs will appear here.")
            token_count_label.config(text="")
        elif result[0] == "error":
            ids_disp_text.insert(tk.END, f"Error: {result[1][:100]}...")
            token_count_label.config(text="")
        else:
            out, text_len = result[1], result[2]
            tids = out.input_ids
            offsets = out.offset_mapping
            
            # Deterministic color generation (faster than random)
            for idx, (start, end) in enumerate(offsets):
                if start < end:
                    hue = (idx * 137.508) % 1.0  # Golden angle
                    r, g, b = [int(200 + 55 * (0.5 + 0.5 * math.sin(hue * 2 * math.pi + i))) for i in (0, 2, 4)]
                    bgc = f"#{r:02x}{g:02x}{b:02x}"
                    tag = f"tok_{idx}"
                    token_colors[idx] = bgc
                    input_hl_text.tag_configure(tag, background=bgc, foreground="black")
                    input_hl_text.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")
            
            # Batch ID insertion
            id_parts = []
            for idx, tid in enumerate(tids):
                bgc = token_colors.get(idx, "#eee")
                tag = f"id_{idx}"
                ids_disp_text.tag_configure(
                    tag,
                    background=bgc,
                    foreground="black",
                    relief="raised",
                    borderwidth=1,
                    lmargin1=2,
                    lmargin2=2,
                    rmargin=2,
                    spacing1=1,
                    spacing3=1,
                )
                id_parts.append((f"{tid} ", tag))
            
            for text_part, tag in id_parts:
                ids_disp_text.insert(tk.END, text_part, tag)
            
            token_count_label.config(
                text=f"Total tokens: {len(tids)} (in {processing_time:.3f}s)"
            )
        
        ids_disp_text.configure(state="disabled")

    # --- Hover Logic (unchanged) ---
    def set_token_highlight(idx, active):
        if active:
            bg = ""
            fg = "white"
        else:
            bg = token_colors.get(idx, "")
            fg = "black"
        tag_tok = f"tok_{idx}"
        tag_id = f"id_{idx}"
        input_hl_text.tag_configure(tag_tok, background=bg, foreground=fg)
        ids_disp_text.tag_configure(tag_id, background=bg, foreground=fg)

    def on_motion_input(event):
        idx = None
        for tag in input_hl_text.tag_names(f"@{event.x},{event.y}"):
            if tag.startswith("tok_"):
                idx = int(tag.split("_")[1])
                break
        if hover_idx[0] is not None and hover_idx[0] != idx:
            set_token_highlight(hover_idx[0], False)
            hover_idx[0] = None
        if idx is not None and hover_idx[0] != idx:
            set_token_highlight(idx, True)
            hover_idx[0] = idx

    def on_leave_input(event):
        if hover_idx[0] is not None:
            set_token_highlight(hover_idx[0], False)
            hover_idx[0] = None

    def on_motion_ids(event):
        idx = None
        index = ids_disp_text.index(f"@{event.x},{event.y}")
        tags = ids_disp_text.tag_names(index)
        for tag in tags:
            if tag.startswith("id_"):
                idx = int(tag.split("_")[1])
                break
        if hover_idx[0] is not None and hover_idx[0] != idx:
            set_token_highlight(hover_idx[0], False)
            hover_idx[0] = None
        if idx is not None and hover_idx[0] != idx:
            set_token_highlight(idx, True)
            hover_idx[0] = idx

    def on_leave_ids(event):
        if hover_idx[0] is not None:
            set_token_highlight(hover_idx[0], False)
            hover_idx[0] = None

    input_hl_text.bind("<KeyRelease>", update_live_hl_tokenize)
    input_hl_text.bind("<Motion>", on_motion_input)
    input_hl_text.bind("<Leave>", on_leave_input)
    ids_disp_text.bind("<Motion>", on_motion_ids)
    ids_disp_text.bind("<Leave>", on_leave_ids)

    # --- Find Token (unchanged) ---
    finder_frame = ttk.Frame(right_pane)
    finder_frame.grid(row=1, column=0, sticky="ew", pady=(10,0))
    add_section_heading(finder_frame, "Find Token (ID or Text)")
    finder_row = ttk.Frame(finder_frame)
    finder_row.pack(fill=tk.X, padx=(0,0))
    ttk.Label(finder_row, text="ID or Text:").pack(side=tk.LEFT, padx=(0, 5))
    tf_entry = ttk.Entry(finder_row, width=32, style="TEntry")
    tf_entry.pack(side=tk.LEFT, padx=(0, 5), expand=False)
    tf_output = ttk.Label(finder_row, text="", foreground=FG_COLOR, background=BG_COLOR, anchor="w")
    tf_output.pack(side=tk.LEFT, padx=(6,0), fill=tk.X, expand=True)

    def perform_find(event=None):
        q = tf_entry.get().strip()
        if not q:
            tf_output.config(text="Please enter a token ID or text.")
        else:
            try:
                if q.lstrip("-").isdigit():
                    tid = int(q)
                    text = f"ID {tid}: "
                    try:
                        dec = tokenizer.decode([tid])
                        text += f"{repr(dec)} | "
                    except:
                        text += "(dec err) | "
                    try:
                        tok = tokenizer.convert_ids_to_tokens([tid])[0]
                        text += f"token {repr(tok)}"
                    except:
                        text += "(tok err)"
                else:
                    tid = tokenizer.convert_tokens_to_ids(q)
                    if tid == tokenizer.unk_token_id and q not in getattr(tokenizer, "vocab", {}):
                        text = f"Text {repr(q)}: Not a known token"
                    else:
                        text = f"Text {repr(q)}: ID {tid}"
                tf_output.config(text=text)
            except Exception as e:
                tf_output.config(text=f"Error: {str(e)[:90]}...")

    tf_entry.bind("<KeyRelease>", perform_find)

    # --- Cleanup on close ---
    def on_closing():
        _shutdown_thread.set()
        _tokenization_queue.put(None)  # Wake up thread
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    # --- Initialize ---
    input_hl_text.insert("1.0", "Type text here to see live tokenization...")
    update_live_hl_tokenize()
    
    window.mainloop()