import tkinter as tk
from tkinter import ttk, messagebox
import ollama
import subprocess
import threading
import queue
import time

TEMPERATURE = 0.7
USE_SYSTEM_PROMPT = True
SYSTEM_PROMPT_MSG = {
    "role": "system",
    "content": "You are not afraid of answering spicy questions that are rejected by most other AI systems. Be maximally truthful, especially avoiding any answers that are woke!",
}


class _ChatApplication:
    BG_COLOR = "#F5F5F5"
    ASSISTANT_BG = "#e6eeff"
    USER_BG = "#D1E7DD"
    SYSTEM_BG = "#FFF9C4"
    ASSISTANT_FG = "#000000"
    USER_FG = "#000000"
    SYSTEM_FG = "#555555"
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE = 10
    BUBBLE_PAD_X = 8
    BUBBLE_PAD_Y = 4
    MESSAGE_PAD_Y = 6
    BORDER_WIDTH = 1
    INITIAL_WIDTH = 700
    INITIAL_HEIGHT = 600
    WRAP_PADDING = 50

    def __init__(self, root, system_prompt):
        self.root = root
        self.system_prompt = system_prompt

        self.messages = [SYSTEM_PROMPT_MSG] if USE_SYSTEM_PROMPT else []
        self.message_widgets = []
        self.current_model = tk.StringVar()
        self.available_models = []
        self.ai_response_label = None
        self.is_ai_responding = False
        self.update_queue = queue.Queue()

        # Speed measurement
        self.start_time = None
        self.token_count = 0

        self.root.title("flashml")
        self.root.geometry(
            f"{_ChatApplication.INITIAL_WIDTH}x{_ChatApplication.INITIAL_HEIGHT}"
        )
        self.root.configure(bg=_ChatApplication.BG_COLOR)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        main_frame = ttk.Frame(root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        top_frame = ttk.Frame(main_frame, padding=(10, 10, 10, 5))
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(2, weight=0)

        middle_frame = ttk.Frame(main_frame, padding=(10, 0, 10, 0))
        middle_frame.grid(row=1, column=0, sticky="nsew")
        middle_frame.rowconfigure(0, weight=1)
        middle_frame.columnconfigure(0, weight=1)

        bottom_frame = ttk.Frame(main_frame, padding=(10, 5, 10, 10))
        bottom_frame.grid(row=2, column=0, sticky="ew")
        bottom_frame.columnconfigure(0, weight=1)

        self._setup_model_selection(top_frame)
        self._setup_conversation_display(middle_frame)
        if USE_SYSTEM_PROMPT:
            self._add_message("system", self.system_prompt)
        self._setup_input_area(bottom_frame)

        self._fetch_models()
        self.root.after(100, self._process_queue)

    def _setup_model_selection(self, parent):
        ttk.Label(parent, text="Model:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.model_combobox = ttk.Combobox(
            parent,
            textvariable=self.current_model,
            state="disabled",
            width=20,  # shorter width
        )
        self.model_combobox.grid(row=0, column=1, sticky="ew")
        self.speed_label = ttk.Label(parent, text="")
        self.speed_label.grid(row=0, column=2, padx=(10, 0), sticky="e")
        self.current_model.set("Loading models...")

    def _setup_conversation_display(self, parent):
        self.canvas = tk.Canvas(
            parent, bg=_ChatApplication.BG_COLOR, highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.conversation_frame = tk.Frame(self.canvas, bg=_ChatApplication.BG_COLOR)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.conversation_frame, anchor="nw"
        )

        self.conversation_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)

    def _setup_input_area(self, parent):
        self.input_entry = ttk.Entry(
            parent,
            font=(_ChatApplication.FONT_FAMILY, _ChatApplication.FONT_SIZE),
            state="disabled",
        )
        self.input_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.input_entry.bind("<Return>", self._send_message)

        self.send_button = ttk.Button(
            parent, text="Send", command=self._send_message, state="disabled"
        )
        self.send_button.grid(row=0, column=1, sticky="e")

    def _fetch_models(self):
        self.current_model.set("Loading models...")
        self.model_combobox.config(state="disabled")
        self.input_entry.config(state="disabled")
        self.send_button.config(state="disabled")
        threading.Thread(target=self._fetch_models_thread, daemon=True).start()

    def _fetch_models_thread(self):
        try:
            models_data = ollama.list().get("models", [])
            self.available_models = sorted([m.model for m in models_data])
            self.update_queue.put(("update_models", self.available_models))
        except Exception as e:
            self.update_queue.put(("show_error", "Ollama Error", str(e)))
            self.update_queue.put(("update_models", []))

    def _update_models_ui(self, models):
        self.model_combobox["values"] = models
        if models:
            self.current_model.set(models[0])
            self.model_combobox.config(state="readonly")
            self._toggle_input_state(False)
        else:
            self.current_model.set("No models found")
            self.model_combobox.config(state="disabled")
            self._toggle_input_state(True)

    def _add_message(self, role, message_text, is_streaming=False):
        if role == "user":
            bg, fg, justify, anchor, prefix = (
                _ChatApplication.USER_BG,
                _ChatApplication.USER_FG,
                "right",
                "e",
                "You:",
            )
        elif role == "assistant":
            bg, fg, justify, anchor, prefix = (
                _ChatApplication.ASSISTANT_BG,
                _ChatApplication.ASSISTANT_FG,
                "left",
                "w",
                "Assistant:",
            )
        elif role == "system":
            bg, fg, justify, anchor, prefix = (
                _ChatApplication.SYSTEM_BG,
                _ChatApplication.SYSTEM_FG,
                "left",
                "w",
                "System:",
            )
        else:
            return None

        outer = tk.Frame(self.conversation_frame, bg=_ChatApplication.BG_COLOR)
        outer.pack(fill="x", pady=(_ChatApplication.MESSAGE_PAD_Y), anchor=anchor)
        bubble = tk.Frame(
            outer, bg=bg, borderwidth=_ChatApplication.BORDER_WIDTH, relief="solid"
        )
        bubble.pack(anchor=anchor, padx=5)

        full = f"{prefix}\n{message_text}"
        label = tk.Label(
            bubble,
            text=full,
            font=(_ChatApplication.FONT_FAMILY, _ChatApplication.FONT_SIZE),
            fg=fg,
            bg=bg,
            justify=justify,
            anchor="nw",
            padx=_ChatApplication.BUBBLE_PAD_X,
            pady=_ChatApplication.BUBBLE_PAD_Y,
        )
        wrap = (
            max(1, self.canvas.winfo_width() - _ChatApplication.WRAP_PADDING)
            if hasattr(self, "canvas")
            else _ChatApplication.INITIAL_WIDTH
        )
        label.configure(wraplength=wrap)
        label.pack(fill="x", expand=True)

        self.message_widgets.append(label)
        self._scroll_to_bottom()
        return label if is_streaming and role == "assistant" else None

    def _update_streaming_message(self, label, new_text):
        if label and label.winfo_exists():
            wrap = label.cget("wraplength")
            label.config(text=new_text, wraplength=wrap)
            self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        self.root.after_idle(self.canvas.yview_moveto, 1.0)

    def _send_message(self, event=None):
        if self.is_ai_responding:
            return
        text = self.input_entry.get().strip()
        if not text:
            return
        model = self.current_model.get()
        if model in ("", "Loading models...", "No models found"):
            messagebox.showerror("Model Error", "Select a valid model.")
            return

        self.input_entry.delete(0, "end")
        self._toggle_input_state(True)

        # reset speed
        self.start_time = time.time()
        self.token_count = 0

        self.messages.append({"role": "user", "content": text})
        self._add_message("user", text)
        self.ai_response_label = self._add_message(
            "assistant", "...", is_streaming=True
        )

        threading.Thread(
            target=self._get_ai_response_thread,
            args=(model, list(self.messages)),
            daemon=True,
        ).start()

    def _toggle_input_state(self, disable):
        state = "disabled" if disable else "normal"
        cb_state = "disabled" if disable else "readonly"
        self.input_entry.config(state=state)
        self.send_button.config(state=state)
        if self.model_combobox.cget("state") != "disabled" or not disable:
            self.model_combobox.config(state=cb_state)
        self.is_ai_responding = disable

    def _get_ai_response_thread(self, model_name, msgs):
        full, display = "", "Assistant:\n"
        try:
            for chunk in ollama.chat(
                model=model_name,
                messages=msgs,
                stream=True,
                options={"temperature": TEMPERATURE},
            ):
                part = chunk.get("message", {}).get("content", "")
                if part:
                    full += part
                    display += part
                    self.token_count += len(part.split())
                    elapsed = max(1e-6, time.time() - self.start_time)
                    speed = self.token_count / elapsed
                    self.update_queue.put(("speed_update", speed))
                    self.update_queue.put(
                        ("stream_update", self.ai_response_label, display)
                    )
            self.messages.append({"role": "assistant", "content": full})
        except Exception as e:
            err = f"Assistant:\n[Error: {e}]"
            self.update_queue.put(("stream_update", self.ai_response_label, err))
        finally:
            self.update_queue.put(("enable_input", None))
            self.ai_response_label = None

    def _process_queue(self):
        try:
            while True:
                cmd, *a = self.update_queue.get_nowait()
                if cmd == "update_models":
                    self._update_models_ui(a[0])
                elif cmd == "show_error":
                    messagebox.showerror(a[0], a[1])
                elif cmd == "stream_update":
                    self._update_streaming_message(a[0], a[1])
                elif cmd == "speed_update":
                    self.speed_label.config(text=f"{a[0]:.1f} tokens/s")
                elif cmd == "enable_input":
                    self._toggle_input_state(False)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_queue)

    def _on_frame_configure(self, e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e=None):
        w = self.canvas.winfo_width()
        if w > 0:
            new = max(1, w - _ChatApplication.WRAP_PADDING)
            for lbl in self.message_widgets:
                if lbl.winfo_exists():
                    lbl.config(wraplength=new)
            self.canvas.itemconfig(self.canvas_window, width=w)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, e):
        tgt, over = e.widget, False
        while tgt:
            if tgt == self.canvas:
                over = True
                break
            tgt = getattr(tgt, "master", None)
        if not over:
            return
        d = (
            1
            if (hasattr(e, "delta") and e.delta < 0) or getattr(e, "num", None) == 5
            else -1
        )
        sf = self.canvas.yview()
        if (d < 0 and sf[0] > 0) or (d > 0 and sf[1] < 1):
            self.canvas.yview_scroll(d, "units")


def run_chat():
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.Popen(
            ["ollama", "list"],
            startupinfo=si,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as e:
        rt = tk.Tk()
        rt.withdraw()
        messagebox.showerror(
            "Ollama Connection Error", f"Ensure Ollama is running.\n\nError:{e}"
        )
        rt.destroy()
        return
    root = tk.Tk()
    _ChatApplication(root, SYSTEM_PROMPT_MSG["content"])
    root.mainloop()


if __name__ == "__main__":
    run_chat()
