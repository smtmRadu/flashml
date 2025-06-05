### begin of file


class _ConversationViewer:
    """
    Internal class to manage the conversation display window.
    Configuration constants are defined within this class.
    """

    # --- Configuration ---
    BG_COLOR = "#F5F5F5"
    ASSISTANT_BG = "#e6eeff"
    USER_BG = "#D1E7DD"
    SYSTEM_BG = "#FFF9C4"
    ASSISTANT_FG = "#000000"
    USER_FG = "#000000"
    SYSTEM_FG = "#120AE8"
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE = 10
    BUBBLE_PAD_X = 8
    BUBBLE_PAD_Y = 4
    MESSAGE_PAD_Y = 6
    BORDER_WIDTH = 1
    INITIAL_WIDTH = 600
    INITIAL_HEIGHT = 500
    WRAP_PADDING = 40  # Pixels subtracted from canvas width for wraplength

    def __init__(self, root, conversation):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.conversation = conversation
        self.message_labels = []  # Keep track of labels to update wraplength

        self.root.title("flashml")
        self.root.geometry(
            f"{_ConversationViewer.INITIAL_WIDTH}x{_ConversationViewer.INITIAL_HEIGHT}"
        )
        self.root.configure(bg=_ConversationViewer.BG_COLOR)

        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # --- Canvas for Scrolling ---
        self.canvas = tk.Canvas(
            main_frame, bg=_ConversationViewer.BG_COLOR, highlightthickness=0
        )

        # --- Scrollbar ---
        scrollbar = ttk.Scrollbar(
            main_frame, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Grid layout for canvas and scrollbar
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # --- Frame inside Canvas to Hold Messages ---
        self.conversation_frame = tk.Frame(self.canvas, bg=_ConversationViewer.BG_COLOR)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.conversation_frame, anchor="nw"
        )

        # --- Bind Events ---
        self.conversation_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # Bind mouse wheel scrolling universally within the root window
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        self.root.bind_all("<Button-4>", self._on_mousewheel)  # Linux scroll up
        self.root.bind_all("<Button-5>", self._on_mousewheel)  # Linux scroll down

        # --- Add Messages ---
        self._populate_messages()

        # Update geometry and scroll region after initial population
        # Need to call update_idletasks to ensure winfo_width is accurate initially
        self.root.update_idletasks()
        self._on_canvas_configure()  # Call explicitly to set initial wraplength and scroll region

    def _populate_messages(self):
        """Adds all messages from the conversation."""
        for message in self.conversation:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            self._add_message(role, content)

    def _add_message(self, role, message_text):
        import tkinter as tk

        """Adds a single message bubble to the conversation frame."""
        # Determine styling based on role using class constants
        if role == "user":
            bg_color = _ConversationViewer.USER_BG
            fg_color = _ConversationViewer.USER_FG
            justify = "right"
            anchor = "e"
        elif role == "assistant":
            bg_color = _ConversationViewer.ASSISTANT_BG
            fg_color = _ConversationViewer.ASSISTANT_FG
            justify = "left"
            anchor = "w"
        else:  # system or other
            bg_color = _ConversationViewer.SYSTEM_BG
            fg_color = _ConversationViewer.SYSTEM_FG
            justify = "left"
            anchor = "w"

        # Outer frame for alignment and vertical padding
        bubble_outer_frame = tk.Frame(
            self.conversation_frame, bg=_ConversationViewer.BG_COLOR
        )
        bubble_outer_frame.pack(
            fill="x", pady=(0, _ConversationViewer.MESSAGE_PAD_Y), anchor=anchor
        )

        # Inner bubble frame with background and border
        bubble_frame = tk.Frame(
            bubble_outer_frame,
            bg=bg_color,
            borderwidth=_ConversationViewer.BORDER_WIDTH,
            relief="solid",
            padx=0,
            pady=0,
        )
        bubble_frame.pack(anchor=anchor, padx=5, pady=0)

        # Label for the text
        display_text = f"{role.capitalize()}:\n{message_text}"
        label = tk.Label(
            bubble_frame,
            text=display_text,
            font=(_ConversationViewer.FONT_FAMILY, _ConversationViewer.FONT_SIZE),
            fg=fg_color,
            bg=bg_color,
            justify=justify,
            anchor="nw",
            padx=_ConversationViewer.BUBBLE_PAD_X,
            pady=_ConversationViewer.BUBBLE_PAD_Y,
            # wraplength is set dynamically
        )
        label.pack(fill="x", expand=True)
        self.message_labels.append(label)

    def _on_frame_configure(self, event=None):
        """Update scroll region when the conversation frame's size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """Update wraplength and scroll region when canvas width changes."""
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 0:
            return  # Avoid calculation if width isn't determined yet

        new_wraplength = max(1, canvas_width - _ConversationViewer.WRAP_PADDING)

        for label in self.message_labels:
            if label.winfo_exists():  # Check if widget still exists
                label.configure(wraplength=new_wraplength)

        # Ensure the frame width matches the canvas width for proper wrapping
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        # Update scroll region after potential height changes due to wrapping
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        # Determine scroll direction based on platform specifics
        if event.num == 5 or event.delta < 0:
            delta = 1  # Scroll down
        elif event.num == 4 or event.delta > 0:
            delta = -1  # Scroll up
        else:
            delta = 0

        # Check if the canvas is scrollable vertically
        scroll_info = self.canvas.yview()
        if scroll_info[0] > 0.0 or scroll_info[1] < 1.0:
            self.canvas.yview_scroll(delta, "units")


def plot_chat(conversation: list[dict]):
    """
    Creates and displays a window showing the provided conversation.

    Args:
        conversation (list): A list of dictionaries, where each dictionary
                             has 'role' (str) and 'content' (str) keys.

    Example:
        >>> example_conversation = [
             {'role': 'system', 'content': 'You are a helpful assistant designed to provide detailed explanations.'},
             {'role': 'user', 'content': 'What is the capital of France?'},
             {'role': 'assistant', 'content': 'The capital and largest city of France is Paris. It is known for its art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine.'},
             {'role': 'user', 'content': 'What is the tallest mountain in the world and where is it located?'},
             {'role': 'assistant', 'content': 'The tallest mountain in the world is Mount Everest (also known as Sagarmatha in Nepali and Chomolungma in Tibetan). It is located in the Mahalangur Himalayas sub-range of the Himalayas, straddling the border between Nepal and the Tibet Autonomous Region of China.'}]
        >>> plot_chat(example_conversation)
    """
    import tkinter as tk

    root = tk.Tk()
    app = _ConversationViewer(root, conversation)
    root.mainloop()
