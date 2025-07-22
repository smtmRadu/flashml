def inspect_tokenizer(tokenizer):
    import tkinter as tk
    from tkinter import ttk
    import random

    # --- Dark Mode Style ---
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

    # --- Left: Basic Info ---
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

    # --- Special Tokens Table (taller now) ---
    special_frame = ttk.Frame(left_pane)
    special_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
    add_section_heading(special_frame, "Special Tokens")
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
            table.insert("", "end", values=(t, i))

    # --- Right: Live Tokenization & Finder ---

    # TOP: Live Tokenization (token ids box as tall as input)
    live_frame = ttk.Frame(right_pane)
    live_frame.grid(row=0, column=0, sticky="nsew")
    add_section_heading(live_frame, "Live Tokenization")
    live_cont = ttk.Frame(live_frame)
    live_cont.pack(fill=tk.BOTH, expand=True)
    input_hl_text = tk.Text(
        live_cont,
        wrap=tk.WORD,
        height=14,   # Both input and IDs box use the same height!
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
        height=14,  # Same as input_hl_text!
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

    # --- Hover Logic State ---
    token_colors = {}   # idx: color hex string
    hover_idx = [None]  # single-element mutable for current hovered index

    def update_live_hl_tokenize(event=None):
        nonlocal token_colors
        token_colors = {}  # Reset colors each time

        text = input_hl_text.get("1.0", "end-1c")
        # clear previous tags
        for tag in input_hl_text.tag_names():
            if tag.startswith("tok_"):
                input_hl_text.tag_delete(tag)
        ids_disp_text.configure(state="normal")
        ids_disp_text.delete("1.0", tk.END)

        if not text.strip():
            ids_disp_text.insert(tk.END, "Token IDs will appear here.")
            token_count_label.config(text="")
        else:
            try:
                out = tokenizer(
                    text, return_offsets_mapping=True, add_special_tokens=False
                )
                tids = out.input_ids
                offsets = out.offset_mapping
                for idx, (start, end) in enumerate(offsets):
                    if start < end:
                        r = random.randint(200, 255)
                        g = random.randint(200, 255)
                        b = random.randint(200, 255)
                        bgc = f"#{r:02x}{g:02x}{b:02x}"
                        tag = f"tok_{idx}"
                        token_colors[idx] = bgc
                        input_hl_text.tag_configure(tag, background=bgc, foreground="black")
                        input_hl_text.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")
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
                    ids_disp_text.insert(tk.END, f"{tid} ", tag)
                # --- Set the token count label ---
                token_count_label.config(
                    text=f"Total tokens: {len(tids)}"
                )
            except Exception as e:
                ids_disp_text.insert(tk.END, f"Error: {str(e)[:100]}...")
                token_count_label.config(text="")
        ids_disp_text.configure(state="disabled")

    # --- Hover highlight logic ---
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
    input_hl_text.insert("1.0", "Type text here to see live tokenization...")
    update_live_hl_tokenize()

    # --- Find Token (one-line, auto-update, no button) ---
    finder_frame = ttk.Frame(right_pane)
    finder_frame.grid(row=1, column=0, sticky="ew", pady=(10,0))
    add_section_heading(finder_frame, "Find Token (ID or Text, inline)")
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
                        text += f"'{dec}' | "
                    except:
                        text += "(dec err) | "
                    try:
                        tok = tokenizer.convert_ids_to_tokens([tid])[0]
                        text += f"token '{tok}'"
                    except:
                        text += "(tok err)"
                else:
                    tid = tokenizer.convert_tokens_to_ids(q)
                    if tid == tokenizer.unk_token_id and q not in getattr(tokenizer, "vocab", {}):
                        text = f"Text '{q}': Not a known token"
                    else:
                        text = f"Text '{q}': ID {tid}"
                tf_output.config(text=text)
            except Exception as e:
                tf_output.config(text=f"Error: {str(e)[:90]}...")

    tf_entry.bind("<KeyRelease>", perform_find)  # instant update, no button!

    window.mainloop()
