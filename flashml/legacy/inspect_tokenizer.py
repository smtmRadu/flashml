def inspect_tokenizer(tokenizer, renderer="vscode"):
    """
    Analyzes a Hugging Face tokenizer and generates a sleek, comprehensive Plotly dashboard.

    This function creates a compact multi-part visualization showing tokenizer configuration,
    all special tokens (including model-specific ones), and detailed statistics.

    Args:
        tokenizer (PreTrainedTokenizerFast): An instance of a Hugging Face tokenizer.
        renderer (str, optional): The Plotly renderer to use for displaying the figure.
                                  Defaults to "vscode". Common options include "notebook",
                                  "browser", or None for default behavior.
    """
    import re

    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    
    pio.templates.default = "plotly_dark"

    # --- 1. Extract Core Data ---
    vocab_size = tokenizer.vocab_size
    model_name = tokenizer.name_or_path or "Unknown Model"
    max_len = tokenizer.model_max_length

    # --- 2. Comprehensive Special Token Detection ---
    def get_all_special_tokens(tokenizer):
        """Extract ALL special tokens from the tokenizer, including hidden ones."""
        special_info = []

        # Get the obvious special tokens
        if hasattr(tokenizer, "special_tokens_map"):
            for token_type, token_value in tokenizer.special_tokens_map.items():
                if isinstance(token_value, str):
                    token_id = tokenizer.convert_tokens_to_ids(token_value)
                    special_info.append((token_type.upper(), token_value, token_id))
                elif isinstance(token_value, dict) and "content" in token_value:
                    token_id = tokenizer.convert_tokens_to_ids(token_value["content"])
                    special_info.append(
                        (token_type.upper(), token_value["content"], token_id)
                    )

        # Get additional special tokens from all_special_tokens
        if hasattr(tokenizer, "all_special_tokens"):
            for token in tokenizer.all_special_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                # Check if we already have this token
                if not any(info[1] == token for info in special_info):
                    special_info.append(("SPECIAL", token, token_id))

        # Look for common special token patterns in the vocabulary
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
            special_patterns = [
                r"<\|.*?\|>",  # <|im_start|>, <|im_end|>, etc.
                r"<.*?>",  # <think>, <eos>, <bos>, etc.
                r"\[.*?\]",  # [INST], [/INST], etc.
                r"<<.*?>>",  # <<SYS>>, <</SYS>>, etc.
            ]

            for token, token_id in vocab.items():
                for pattern in special_patterns:
                    if re.match(pattern, token):
                        # Check if we already have this token
                        if not any(info[1] == token for info in special_info):
                            special_info.append(("PATTERN", token, token_id))
                        break

        # Sort by token ID for consistent display
        special_info.sort(key=lambda x: x[2] if x[2] is not None else float("inf"))
        return special_info

    special_tokens_info = get_all_special_tokens(tokenizer)

    # --- 3. Additional Configuration ---
    padding_side = getattr(tokenizer, "padding_side", "N/A")
    truncation_side = getattr(tokenizer, "truncation_side", "N/A")

    # Get tokenizer class name
    tokenizer_class = tokenizer.__class__.__name__

    # Chat template handling
    try:
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            # Truncate if too long and format for display
            if len(chat_template) > 200:
                chat_template = chat_template[:200] + "..."
            chat_template = chat_template.replace("\n", "<br>").replace(
                "    ", "&nbsp;&nbsp;"
            )
        else:
            chat_template = "Not defined"
    except:
        chat_template = "N/A"

    # --- 4. Create Compact Layout ---
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "table", "colspan": 3}, None, None],
        ],
        vertical_spacing=0.12,
        row_heights=[0.35, 0.65],
        subplot_titles=[
            f"<b style='color:#00d4ff'>Vocabulary Size</b>",
            f"<b style='color:#ff6b6b'>Max Length</b>",
            f"<b style='color:#4ecdc4'>Special Tokens</b>",
        ],
    )

    # --- 5. Add Indicator Traces ---
    # Vocab size with gradient effect
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=vocab_size,
            number={
                "font": {"size": 36, "color": "#00d4ff"},
                "suffix": " tokens",
                "valueformat": ",",
            },
            domain={"row": 0, "column": 0},
        ),
        row=1,
        col=1,
    )

    # Max length with context
    max_len_display = max_len if max_len != float("inf") else "∞"
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=max_len if max_len != float("inf") else None,
            number={
                "font": {"size": 36, "color": "#ff6b6b"},
                "suffix": " tokens",
                "valueformat": ",",
            },
            domain={"row": 0, "column": 1},
        ),
        row=1,
        col=2,
    )

    # Special tokens count
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=len(special_tokens_info),
            number={
                "font": {"size": 36, "color": "#4ecdc4"},
                "suffix": " special tokens",
                "valueformat": ",",
            },
            domain={"row": 0, "column": 2},
        ),
        row=1,
        col=3,
    )

    # --- 6. Enhanced Special Tokens Table ---
    if special_tokens_info:
        token_types = [info[0] for info in special_tokens_info]
        token_values = [f"<code>{info[1]}</code>" for info in special_tokens_info]
        token_ids = [
            str(info[2]) if info[2] is not None else "N/A"
            for info in special_tokens_info
        ]

        # Color code by type
        type_colors = {
            "PAD": "#ff9999",
            "UNK": "#ffcc99",
            "CLS": "#99ff99",
            "SEP": "#99ccff",
            "MASK": "#cc99ff",
            "BOS": "#ffff99",
            "EOS": "#ff99cc",
            "SPECIAL": "#cccccc",
            "PATTERN": "#55C96A",
        }

        cell_colors = []
        for token_type in token_types:
            color = type_colors.get(token_type, "#f0f0f0")
            cell_colors.append(color)
    else:
        token_types = ["No special tokens found"]
        token_values = [""]
        token_ids = [""]
        cell_colors = ["#f0f0f0"]

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "<b>Type</b>",
                    "<b>Token</b>",
                    "<b>ID</b>",
                    "<b>Config Info</b>",
                ],
                fill_color="#2c3e50",
                align="center",
                font=dict(size=13, color="white"),
                height=35,
            ),
            cells=dict(
                values=[
                    token_types,
                    token_values,
                    token_ids,
                    [
                        f"<b>Class:</b> {tokenizer_class}<br><b>Padding:</b> {padding_side}<br><b>Truncation:</b> {truncation_side}"
                    ]
                    + [""] * (len(token_types) - 1),
                ],
                fill_color=[cell_colors + ["#ecf0f1"] * len(token_types)],
                align=["center", "left", "center", "left"],
                font=dict(size=11, color="#2c3e50"),
                height=28,
            ),
        ),
        row=2,
        col=1,
    )

    # --- 7. Final Layout Styling ---
    fig.update_layout(
        title={
            "text": f"🔍 Tokenizer of: <b>{model_name}</b>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#d1d4d6"},
        },
        height=550,  # Much more compressed!
        showlegend=False,
        template="plotly_white",
        margin=dict(l=15, r=15, t=80, b=15),
        font=dict(family="SF Pro Display, -apple-system, system-ui, sans-serif"),
        paper_bgcolor="#1e252c",
        plot_bgcolor="#333030",
    )

    # Add subtle styling
    fig.update_annotations(font_size=12, font_color="#34495e")

    # # Add a subtle footer with chat template info if available
    # if chat_template and chat_template != "Not defined":
    #     fig.add_annotation(
    #         text=f"<i>Chat Template: {chat_template}</i>",
    #         xref="paper",
    #         yref="paper",
    #         x=1,
    #         y=0,
    #         showarrow=False,
    #         font=dict(size=9, color="#7f8c8d"),
    #         bgcolor="rgba(255,255,255,0.8)",
    #         bordercolor="#bdc3c7",
    #         borderwidth=1,
    #     )

    # Show with style
    fig.show(renderer=renderer)
