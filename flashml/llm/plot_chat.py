from typing import Dict, List



class ChatPlotter:
    """
    A chat conversation plotter using Plotly with VS Code-style theming.
    Displays actual messages in a scrollable chat-like interface.
    """

    # VS Code-inspired color scheme
    THEME_CONFIG = {
        "bg_color": "#1c1c1f",  # VS Code dark background
        "paper_color": "#2C2C2E",  # Slightly lighter for paper
        "text_color": "#cccccc",  # VS Code light text
        "grid_color": "#3e3e3e",  # Subtle grid lines
        # Message bubble colors
        "user_bg": "#0e639c",  # VS Code blue
        "user_text": "#ffffff",
        "user_border": "#264f78",
        "assistant_bg": "#107c10",  # VS Code green
        "assistant_text": "#ffffff",
        "assistant_border": "#14432a",
        "system_bg": "#ca5010",  # VS Code orange
        "system_text": "#ffffff",
        "system_border": "#5d2e0a",
    }

    def __init__(self, renderer: str = "vscode"):
        """
        Initialize the chat plotter.

        Args:
            renderer: Plotly renderer to use (default: "vscode")
        """
        self.renderer = renderer
        self._setup_theme()

    def _setup_theme(self):
        import plotly.graph_objects as go
        import plotly.io as pio

        """Setup the VS Code-inspired theme for Plotly."""
        # Create custom template
        vscode_template = go.layout.Template(
            layout=go.Layout(
                paper_bgcolor=self.THEME_CONFIG["paper_color"],
                plot_bgcolor=self.THEME_CONFIG["bg_color"],
                font=dict(
                    color=self.THEME_CONFIG["text_color"],
                    family="'Cascadia Code', 'Fira Code', 'Consolas', monospace",
                    size=12,
                ),
                margin=dict(l=10, r=10, t=50, b=10),
            )
        )

        # Register the template
        pio.templates["vscode"] = vscode_template
        pio.templates.default = "vscode"

    def _wrap_text(self, text: str, width: int = 80) -> str:
        import textwrap
        # Split on existing line breaks first, wrap each separately
        paragraphs = text.split('\n')
        wrapped_lines = []
        for para in paragraphs:
            wrapped = textwrap.wrap(para, width=width) or ['']
            wrapped_lines.extend(wrapped)
            # Add a blank line to indicate paragraph break, except after last
            wrapped_lines.append('<br>')
        if wrapped_lines and wrapped_lines[-1] == '<br>':
            wrapped_lines.pop()  # Remove trailing <br>
        return ' '.join(wrapped_lines)

    def _get_role_config(self, role: str) -> Dict:
        """Get styling configuration for a specific role."""
        role_configs = {
            "user": {
                "bg_color": self.THEME_CONFIG["user_bg"],
                "text_color": self.THEME_CONFIG["user_text"],
                "border_color": self.THEME_CONFIG["user_border"],
                "align": "right",
                "x_pos": 0.95,
                "x_anchor": "right",
            },
            "assistant": {
                "bg_color": self.THEME_CONFIG["assistant_bg"],
                "text_color": self.THEME_CONFIG["assistant_text"],
                "border_color": self.THEME_CONFIG["assistant_border"],
                "align": "left",
                "x_pos": 0.05,
                "x_anchor": "left",
            },
            "system": {
                "bg_color": self.THEME_CONFIG["system_bg"],
                "text_color": self.THEME_CONFIG["system_text"],
                "border_color": self.THEME_CONFIG["system_border"],
                "align": "center",
                "x_pos": 0.5,
                "x_anchor": "center",
            },
        }
        return role_configs.get(role, role_configs["system"])

    def _create_message_annotation(self, message: Dict, y_position: float) -> Dict:
        """Create a Plotly annotation for a single message."""
        role = message.get("role", "system")
        content = message.get("content", "")
        config = self._get_role_config(role)

        # Wrap the content
        wrapped_content = self._wrap_text(content, width=60)

        # Create the message text with role header
        message_text = f"<b>{role.upper()}</b><br>{wrapped_content}"

        # Create annotation
        annotation = dict(
            x=config["x_pos"],
            y=y_position,
            text=message_text,
            showarrow=False,
            xref="paper",
            yref="y",
            xanchor=config["x_anchor"],
            yanchor="top",  # Anchor annotation to its top edge
            align=config["align"],
            font=dict(
                size=11,
                color=config["text_color"],
                family="'Cascadia Code', 'Fira Code', 'Consolas', monospace",
            ),
            bgcolor=config["bg_color"],
            bordercolor=config["border_color"],
            borderwidth=1,
            borderpad=8,
            opacity=0.9,
        )

        return annotation

    def plot_chat(
        self,
        conversation: List[Dict],
        title: str = "Chat Conversation",
        width: int = 900,
        height: int = 600,
    ):
        import plotly.graph_objects as go
        # import plotly.io as pio

        """
        Create and display a scrollable chat conversation plot.
        This version dynamically calculates message positions to prevent overlap on scroll.

        Args:
            conversation: List of message dictionaries with 'role' and 'content'
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly Figure object
        """

        if not conversation:
            raise ValueError("Conversation cannot be empty")

        fig = go.Figure()

        annotations = []
        y_scatter_positions = []
        current_y = 0.0

        # Define arbitrary units for layout. These control the scale of the y-axis.
        LINE_HEIGHT_UNIT = (
            1.0  # The vertical space each line of text occupies in y-axis units.
        )
        PADDING_UNIT = 2.5  # The vertical padding between messages in y-axis units.

        # Estimate the height of each message and position it accordingly, from top to bottom.
        for message in conversation:
            # Determine the number of lines in the message for height calculation
            wrapped_content = self._wrap_text(message.get("content", ""), width=60)
            num_lines = wrapped_content.count("<br>") + 1

            # Calculate the total height of the message box in data units
            message_height = num_lines * LINE_HEIGHT_UNIT

            # Position the top of the current message bubble at current_y
            y_pos = current_y
            y_scatter_positions.append(y_pos)

            # Create and store the annotation for the current message
            annotation = self._create_message_annotation(message, y_pos)
            annotations.append(annotation)

            # Update current_y for the *next* message, moving downwards on the y-axis
            current_y -= message_height + PADDING_UNIT

        # The total y-range now spans from the bottom-most point to the top
        y_range_bottom = current_y
        y_range_top = PADDING_UNIT  # Add a little padding at the top

        # Add invisible scatter points to establish the y-axis coordinate system
        # This is crucial for making the y-axis exist and be scrollable.
        fig.add_trace(
            go.Scatter(
                x=[0.5] * len(conversation),
                y=y_scatter_positions,
                mode="markers",
                marker=dict(size=0.1, color="rgba(0,0,0,0)"),  # Invisible markers
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Update layout with all annotations and correct axis configuration
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color=self.THEME_CONFIG["text_color"]),
                x=0.5,
            ),
            width=width,
            height=height,
            annotations=annotations,
            xaxis=dict(
                range=[0, 1],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False,
                fixedrange=True,  # Prevent horizontal scrolling
            ),
            yaxis=dict(
                # The range encompasses the entire conversation.
                range=[y_range_bottom, y_range_top],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False,
                # IMPORTANT: This allows panning/scrolling on the y-axis
                fixedrange=False,
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            dragmode="pan",  # Set default interaction to 'pan' for scrolling
            paper_bgcolor=self.THEME_CONFIG["paper_color"],
            plot_bgcolor=self.THEME_CONFIG["bg_color"],
        )

        # Configure the mode bar and interactions
        config = {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "select2d",
                "lasso2d",
                "autoScale2d",
                "zoomIn2d",
                "zoomOut2d",
            ],
            "scrollZoom": True,  # Enable zooming/scrolling with the mouse wheel
            "doubleClick": "reset",
        }

        # Show the plot
        fig.show(renderer=self.renderer, config=config)
        return fig


def plot_chat(
    conversation: List[Dict],
    renderer: str = "notebook",
    title: str = "Chat Conversation",
    width: int = 900,
    height: int = 600,
):
    """
    Create and display a scrollable (with click and drag) chat conversation plot using Plotly.

    Args:
        conversation: List of dictionaries with 'role' and 'content' keys
        renderer: Plotly renderer to use (default: "notebook")
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly Figure object

    Example:
        >>> example_conversation = [
              {'role': 'system', 'content': 'You are a helpful assistant designed to provide detailed explanations.'},
              {'role': 'user', 'content': 'What is the capital of France?'},
              {'role': 'assistant', 'content': 'The capital and largest city of France is Paris. It is known for its art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine.'},
              {'role': 'user', 'content': 'What is the tallest mountain in the world and where is it located?'},
              {'role': 'assistant', 'content': 'The tallest mountain in the world is Mount Everest (also known as Sagarmatha in Nepali and Chomolungma in Tibetan). It is located in the Mahalangur Himalayas sub-range of the Himalayas, straddling the border between Nepal and the Tibet Autonomous Region of China.'}
        ]
        >>> # To run the example, uncomment the following line:
        >>> # plot_chat(example_conversation)
    """
    plotter = ChatPlotter(renderer=renderer)
    plotter.plot_chat(conversation, title, width, height)
