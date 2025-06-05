import tkinter as tk
from tkinter import scrolledtext, simpledialog, font as tkfont, messagebox, ttk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_WINDOW_SIZE = 50
DEFAULT_LOW_PPL_PERCENTILE = 20
DEFAULT_HIGH_PPL_PERCENTILE = 80

DARK_BACKGROUND = "#2E2E2E"
DARK_FOREGROUND = "#DCDCDC"
TEXT_AREA_BACKGROUND = "#1E1E1E"
BUTTON_BACKGROUND = "#555555"
BUTTON_FOREGROUND = "#FFFFFF"
ENTRY_BACKGROUND = "#3C3F41"
HIGHLIGHT_GREEN_BG = "#005000"
HIGHLIGHT_RED_BG = "#700000"
LABEL_FRAME_BG = "#3C3F41"

model = None
tokenizer = None


# --- Model Loading ---
def load_model_and_tokenizer_with_gui_feedback(root_window, loading_label_var):
    global model, tokenizer
    try:
        loading_label_var.set(f"Loading model: {MODEL_NAME}...")
        root_window.update_idletasks()

        # Try to load with bfloat16 if supported
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        loading_label_var.set(f"Model {MODEL_NAME} loaded.")
        return True
    except Exception as e:
        messagebox.showerror(
            "Model Loading Error", f"Failed to load model '{MODEL_NAME}'.\nError: {e}"
        )
        loading_label_var.set("Model loading failed.")
        return False


# --- Statistics Variables ---
stat_overall_ppl = None
stat_avg_window_ppl = None
stat_min_window_ppl = None
stat_max_window_ppl = None
stat_low_ppl_thresh_val = None
stat_high_ppl_thresh_val = None
stat_green_segments = None
stat_red_segments = None


def initialize_stats_vars(root):
    global \
        stat_overall_ppl, \
        stat_avg_window_ppl, \
        stat_min_window_ppl, \
        stat_max_window_ppl
    global \
        stat_low_ppl_thresh_val, \
        stat_high_ppl_thresh_val, \
        stat_green_segments, \
        stat_red_segments

    stat_overall_ppl = tk.StringVar(root, value="N/A")
    stat_avg_window_ppl = tk.StringVar(root, value="N/A")
    stat_min_window_ppl = tk.StringVar(root, value="N/A")
    stat_max_window_ppl = tk.StringVar(root, value="N/A")
    stat_low_ppl_thresh_val = tk.StringVar(root, value="N/A")
    stat_high_ppl_thresh_val = tk.StringVar(root, value="N/A")
    stat_green_segments = tk.StringVar(root, value="0")
    stat_red_segments = tk.StringVar(root, value="0")


def reset_stats_display():
    stat_overall_ppl.set("N/A")
    stat_avg_window_ppl.set("N/A")
    stat_min_window_ppl.set("N/A")
    stat_max_window_ppl.set("N/A")
    stat_low_ppl_thresh_val.set("N/A")
    stat_high_ppl_thresh_val.set("N/A")
    stat_green_segments.set("0")
    stat_red_segments.set("0")


def clear_highlights_and_stats():
    text_input.tag_remove("low_ppl", "1.0", tk.END)
    text_input.tag_remove("high_ppl", "1.0", tk.END)
    reset_stats_display()
    processing_status_var.set("Highlights and stats cleared.")


# --- Core Logic ---
def analyze_text_and_highlight():
    if not model or not tokenizer:
        messagebox.showerror("Error", "Model not loaded.")
        return

    input_text_content = text_input.get("1.0", tk.END).strip()
    if not input_text_content:
        messagebox.showinfo("Info", "Input text is empty.")
        return

    clear_highlights_and_stats()  # Clear previous results first

    try:
        window_size = int(window_size_entry.get())
        low_percentile = int(low_percentile_entry.get())
        high_percentile = int(high_percentile_entry.get())

        if not (0 <= low_percentile < high_percentile <= 100):
            messagebox.showerror("Error", "Percentiles must be 0 <= Low < High <= 100.")
            return
        if window_size <= 1:
            messagebox.showerror("Error", "Window size must be greater than 1.")
            return
    except ValueError:
        messagebox.showerror("Error", "Invalid numeric input for controls.")
        return

    processing_status_var.set("Processing... This may take a while.")
    root.update_idletasks()

    try:
        tokenized_inputs = tokenizer(
            input_text_content, return_tensors="pt", return_offsets_mapping=True
        )
        input_ids = tokenized_inputs.input_ids
        offset_mapping = tokenized_inputs.offset_mapping[0].tolist()

        if input_ids.shape[1] < 2:
            messagebox.showinfo(
                "Info", "Text too short for perplexity (need at least 2 tokens)."
            )
            processing_status_var.set("Text too short.")
            reset_stats_display()
            return

        current_device = model.device
        input_ids = input_ids.to(current_device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        neg_log_likelihoods = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        overall_ppl_val = torch.exp(neg_log_likelihoods.mean()).item()
        stat_overall_ppl.set(f"{overall_ppl_val:.2f}")

        num_tokens = input_ids.shape[1]
        num_nlls = len(neg_log_likelihoods)
        window_perplexities_data = []

        if num_tokens < window_size:
            processing_status_var.set("Text shorter than window size.")
            stat_avg_window_ppl.set(f"Text too short for ws={window_size}")
            return

        for i in range(num_nlls - (window_size - 1) + 1):
            window_nlls = neg_log_likelihoods[i : i + window_size - 1]
            if len(window_nlls) == 0:
                continue

            window_ppl = torch.exp(window_nlls.mean()).item()
            start_token_idx_in_input_ids = i
            end_token_idx_in_input_ids = i + window_size - 1

            if start_token_idx_in_input_ids < len(
                offset_mapping
            ) and end_token_idx_in_input_ids < len(offset_mapping):
                char_start = offset_mapping[start_token_idx_in_input_ids][0]
                char_end = offset_mapping[end_token_idx_in_input_ids][1]
                # Ensure char_end is not beyond the actual text length due to special tokens/padding offsets
                char_end = min(char_end, len(input_text_content))
                char_start = min(char_start, char_end)

                if char_start < char_end:  # only consider valid character spans
                    window_perplexities_data.append(
                        {
                            "ppl": window_ppl,
                            "char_start": char_start,
                            "char_end": char_end,
                        }
                    )

        if window_perplexities_data:
            all_ppls = np.array([wp["ppl"] for wp in window_perplexities_data])
            low_threshold_val = np.percentile(all_ppls, low_percentile)
            high_threshold_val = np.percentile(all_ppls, high_percentile)

            stat_low_ppl_thresh_val.set(f"<= {low_threshold_val:.2f}")
            stat_high_ppl_thresh_val.set(f">= {high_threshold_val:.2f}")
            stat_avg_window_ppl.set(f"{np.mean(all_ppls):.2f}")
            stat_min_window_ppl.set(f"{np.min(all_ppls):.2f}")
            stat_max_window_ppl.set(f"{np.max(all_ppls):.2f}")

            green_c = 0
            red_c = 0
            for wp in window_perplexities_data:
                start_idx_tk = f"1.0+{wp['char_start']}c"
                end_idx_tk = f"1.0+{wp['char_end']}c"

                # Check if the range is valid before applying tag
                try:
                    actual_end_offset = text_input.index(end_idx_tk)  # verify end index
                    actual_start_offset = text_input.index(start_idx_tk)
                    if text_input.compare(actual_start_offset, "<", actual_end_offset):
                        if wp["ppl"] <= low_threshold_val and wp["ppl"] > 0:
                            text_input.tag_add("low_ppl", start_idx_tk, end_idx_tk)
                            green_c += 1
                        elif wp["ppl"] >= high_threshold_val:
                            text_input.tag_add("high_ppl", start_idx_tk, end_idx_tk)
                            red_c += 1
                except tk.TclError:
                    print(
                        f"Warning: Invalid text index range for highlighting: {start_idx_tk} to {end_idx_tk}"
                    )

            stat_green_segments.set(str(green_c))
            stat_red_segments.set(str(red_c))
        else:
            stat_avg_window_ppl.set("N/A (no valid windows)")

        processing_status_var.set("Analysis complete.")

    except Exception as e:
        messagebox.showerror("Processing Error", f"An error occurred: {e}")
        processing_status_var.set("Error during analysis.")
        import traceback

        traceback.print_exc()


# --- GUI Setup ---
root = tk.Tk()
root.title(f"Text Perplexity Analyzer")
root.configure(bg=DARK_BACKGROUND)
root.geometry("1000x850")

default_font = tkfont.nametofont("TkDefaultFont")
default_font.configure(size=10)
text_font = tkfont.Font(family="Consolas", size=11)

# --- Model Loading Status (Top) ---
model_status_frame = tk.Frame(root, bg=DARK_BACKGROUND)
model_status_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
model_loading_label_var = tk.StringVar(root, value="Initializing...")
model_loading_label = tk.Label(
    model_status_frame,
    textvariable=model_loading_label_var,
    bg=DARK_BACKGROUND,
    fg=DARK_FOREGROUND,
    font=default_font,
)
model_loading_label.pack(side=tk.LEFT)


# --- Input Area ---
tk.Label(
    root,
    text="Enter Text for Analysis:",
    bg=DARK_BACKGROUND,
    fg=DARK_FOREGROUND,
    font=default_font,
).pack(anchor="w", padx=10, pady=(10, 0))
text_input = scrolledtext.ScrolledText(
    root,
    height=15,
    wrap=tk.WORD,
    bg=TEXT_AREA_BACKGROUND,
    fg=DARK_FOREGROUND,
    insertbackground=DARK_FOREGROUND,
    font=text_font,
    relief=tk.FLAT,
    borderwidth=1,
)
text_input.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
text_input.configure(
    selectbackground=BUTTON_BACKGROUND, selectforeground=DARK_FOREGROUND
)
text_input.tag_configure(
    "low_ppl", background=HIGHLIGHT_GREEN_BG
)  # Foreground will be DARK_FOREGROUND by default
text_input.tag_configure("high_ppl", background=HIGHLIGHT_RED_BG)


# --- Controls Area ---
controls_frame = tk.Frame(root, bg=DARK_BACKGROUND, pady=5)
controls_frame.pack(fill=tk.X, padx=10)

tk.Label(
    controls_frame,
    text="Window (tokens):",
    bg=DARK_BACKGROUND,
    fg=DARK_FOREGROUND,
    font=default_font,
).pack(side=tk.LEFT, padx=(0, 2))
window_size_entry = tk.Entry(
    controls_frame,
    width=4,
    bg=ENTRY_BACKGROUND,
    fg=DARK_FOREGROUND,
    insertbackground=DARK_FOREGROUND,
    relief=tk.FLAT,
    font=default_font,
)
window_size_entry.insert(0, str(DEFAULT_WINDOW_SIZE))
window_size_entry.pack(side=tk.LEFT, padx=(0, 10))

tk.Label(
    controls_frame,
    text="Low PPL %ile:",
    bg=DARK_BACKGROUND,
    fg=DARK_FOREGROUND,
    font=default_font,
).pack(side=tk.LEFT, padx=(0, 2))
low_percentile_entry = tk.Entry(
    controls_frame,
    width=3,
    bg=ENTRY_BACKGROUND,
    fg=DARK_FOREGROUND,
    insertbackground=DARK_FOREGROUND,
    relief=tk.FLAT,
    font=default_font,
)
low_percentile_entry.insert(0, str(DEFAULT_LOW_PPL_PERCENTILE))
low_percentile_entry.pack(side=tk.LEFT, padx=(0, 10))

tk.Label(
    controls_frame,
    text="High PPL %ile:",
    bg=DARK_BACKGROUND,
    fg=DARK_FOREGROUND,
    font=default_font,
).pack(side=tk.LEFT, padx=(0, 2))
high_percentile_entry = tk.Entry(
    controls_frame,
    width=3,
    bg=ENTRY_BACKGROUND,
    fg=DARK_FOREGROUND,
    insertbackground=DARK_FOREGROUND,
    relief=tk.FLAT,
    font=default_font,
)
high_percentile_entry.insert(0, str(DEFAULT_HIGH_PPL_PERCENTILE))
high_percentile_entry.pack(side=tk.LEFT, padx=(0, 15))

analyze_button = tk.Button(
    controls_frame,
    text="Analyze Text",
    command=analyze_text_and_highlight,
    bg=BUTTON_BACKGROUND,
    fg=BUTTON_FOREGROUND,
    relief=tk.FLAT,
    font=default_font,
    padx=10,
)
analyze_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(
    controls_frame,
    text="Clear",
    command=clear_highlights_and_stats,
    bg=BUTTON_BACKGROUND,
    fg=BUTTON_FOREGROUND,
    relief=tk.FLAT,
    font=default_font,
    padx=10,
)
clear_button.pack(side=tk.LEFT, padx=5)

processing_status_var = tk.StringVar(root, value="Ready.")
processing_status_label = tk.Label(
    controls_frame,
    textvariable=processing_status_var,
    bg=DARK_BACKGROUND,
    fg=DARK_FOREGROUND,
    font=default_font,
)
processing_status_label.pack(side=tk.LEFT, padx=10)

# --- Statistics Area ---
stats_outer_frame = tk.LabelFrame(
    root,
    text="Analysis Statistics",
    bg=DARK_BACKGROUND,
    fg=DARK_FOREGROUND,
    font=default_font,
    relief=tk.GROOVE,
    padx=10,
    pady=10,
)
stats_outer_frame.pack(fill=tk.X, padx=10, pady=10)

initialize_stats_vars(root)  # Initialize the StringVars

stats_grid_frame = tk.Frame(stats_outer_frame, bg=DARK_BACKGROUND)
stats_grid_frame.pack(fill=tk.X)


def create_stat_label(parent, text, textvariable, row, col_label, col_value):
    tk.Label(
        parent,
        text=text,
        bg=DARK_BACKGROUND,
        fg=DARK_FOREGROUND,
        font=default_font,
        anchor="e",
    ).grid(row=row, column=col_label, sticky="e", padx=(0, 5), pady=2)
    tk.Label(
        parent,
        textvariable=textvariable,
        bg=DARK_BACKGROUND,
        fg=DARK_FOREGROUND,
        font=default_font,
        anchor="w",
    ).grid(row=row, column=col_value, sticky="w", padx=(0, 10), pady=2)


create_stat_label(stats_grid_frame, "Overall Perplexity:", stat_overall_ppl, 0, 0, 1)
create_stat_label(stats_grid_frame, "Avg. Window PPL:", stat_avg_window_ppl, 1, 0, 1)
create_stat_label(stats_grid_frame, "Min Window PPL:", stat_min_window_ppl, 2, 0, 1)
create_stat_label(stats_grid_frame, "Max Window PPL:", stat_max_window_ppl, 3, 0, 1)

create_stat_label(
    stats_grid_frame, "Low PPL Threshold:", stat_low_ppl_thresh_val, 0, 2, 3
)
create_stat_label(
    stats_grid_frame, "High PPL Threshold:", stat_high_ppl_thresh_val, 1, 2, 3
)
create_stat_label(
    stats_grid_frame, "Green Segments (Low PPL):", stat_green_segments, 2, 2, 3
)
create_stat_label(
    stats_grid_frame, "Red Segments (High PPL):", stat_red_segments, 3, 2, 3
)

# Configure grid columns to space out
stats_grid_frame.columnconfigure(0, weight=1)
stats_grid_frame.columnconfigure(1, weight=2)
stats_grid_frame.columnconfigure(2, weight=1)
stats_grid_frame.columnconfigure(3, weight=2)


# --- Load Model on Startup ---
if __name__ == "__main__":
    if not load_model_and_tokenizer_with_gui_feedback(root, model_loading_label_var):
        # Optionally close the app if model loading fails critically,
        # or allow it to run if some functionality is possible without a model (not the case here)
        if messagebox.askretrycancel(
            "Model Error",
            f"Model {MODEL_NAME} failed to load. The application may not work. Retry or Cancel to exit?",
        ):
            if not load_model_and_tokenizer_with_gui_feedback(
                root, model_loading_label_var
            ):
                root.destroy()  # Exit if retry also fails
            else:
                root.mainloop()  # Start if retry succeeds
        else:
            root.destroy()  # Exit if user cancels
    else:
        root.mainloop()
