_COLOR_MAP = None

def ansi_of(color_name) -> str:
    """
    Returns the ANSI escape sequence for the given color name. Note that if printed, the console color will change.
    """
    global _COLOR_MAP
    if _COLOR_MAP is None:
        _COLOR_MAP = _init_color_map()

    color_name = color_name.lower()
    assert color_name in _COLOR_MAP, f"Color {color_name} not found in color map."
    return _COLOR_MAP[color_name][0]  # Index 0 for ANSI

def hex_of(color_name) -> str:
    """
    Returns the hexadecimal representation of the given color name (e.g., '#FF0000').
    """
    global _COLOR_MAP
    if _COLOR_MAP is None:
        _COLOR_MAP = _init_color_map()

    color_name = color_name.lower()
    assert color_name in _COLOR_MAP, f"Color {color_name} not found in color map."
    return _COLOR_MAP[color_name][1]  # Index 1 for HEX

def _init_color_map() -> dict:
    """
    Initializes and returns a single color map with tuples containing (ANSI, HEX) values.
    """
    return {
        'black': ('\033[30m', '#000000'),
        'red': ('\033[31m', '#FF0000'),
        'green': ('\033[32m', '#008000'),
        'yellow': ('\033[33m', '#FFFF00'),
        'blue': ('\033[34m', '#0000FF'),
        'purple': ('\033[35m', '#800080'),
        'cyan': ('\033[36m', '#00FFFF'),
        'white': ('\033[37m', '#FFFFFF'),
        'gray': ('\033[90m', '#808080'),
        'light_red': ('\033[91m', '#FF5555'),
        'light_green': ('\033[92m', '#55FF55'),
        'light_yellow': ('\033[93m', '#FFFF55'),
        'light_blue': ('\033[94m', '#5555FF'),
        'magenta': ('\033[95m', '#FF00FF'),
        'light_cyan': ('\033[96m', '#55FFFF'),
        'pure_white': ('\033[97m', '#FFFFFF'),
        'orange': ('\033[38;2;255;165;0m', '#FFA500'),
        'pink': ('\033[38;2;255;105;180m', '#FF69B4'),
        'lime': ('\033[38;2;0;255;0m', '#00FF00'),
        'teal': ('\033[38;2;0;128;128m', '#008080'),
        'violet': ('\033[38;2;238;130;238m', '#EE82EE'),
        'indigo': ('\033[38;2;75;0;130m', '#4B0082'),
        'gold': ('\033[38;2;255;215;0m', '#FFD700'),
        'silver': ('\033[38;2;192;192;192m', '#C0C0C0'),
        'brown': ('\033[38;2;165;42;42m', '#A52A2A'),
        'maroon': ('\033[38;2;128;0;0m', '#800000'),
        'olive': ('\033[38;2;128;128;0m', '#808000'),
        'navy': ('\033[38;2;0;0;128m', '#000080'),
        'coral': ('\033[38;2;255;127;80m', '#FF7F50'),
        'turquoise': ('\033[38;2;64;224;208m', '#40E0D0'),
        'salmon': ('\033[38;2;250;128;114m', '#FA8072'),
        'plum': ('\033[38;2;221;160;221m', '#DDA0DD'),
        'orchid': ('\033[38;2;218;112;214m', '#DA70D6'),
        'sienna': ('\033[38;2;160;82;45m', '#A0522D'),
        'khaki': ('\033[38;2;240;230;140m', '#F0E68C'),
        'crimson': ('\033[38;2;220;20;60m', '#DC143C'),
        'lavender': ('\033[38;2;230;230;250m', '#E6E6FA'),
        'beige': ('\033[38;2;245;245;220m', '#F5F5DC'),
        'mint': ('\033[38;2;189;252;201m', '#BDFCC9'),
        'peach': ('\033[38;2;255;218;185m', '#FFDAB9'),
        'aqua': ('\033[38;2;0;255;255m', '#00FFFF'),
        'chartreuse': ('\033[38;2;127;255;0m', '#7FFF00'),
        'tan': ('\033[38;2;210;180;140m', '#D2B48C'),
        'rose': ('\033[38;2;255;0;127m', '#FF007F'),
        'emerald': ('\033[38;2;80;200;120m', '#50C878'),
        'amber': ('\033[38;2;255;191;0m', '#FFBF00'),
        'jade': ('\033[38;2;0;168;107m', '#00A86B'),
        'fuchsia': ('\033[38;2;255;0;255m', '#FF00FF'),
        'slate': ('\033[38;2;112;128;144m', '#708090'),
        'ivory': ('\033[38;2;255;255;240m', '#FFFFF0'),
        'sand': ('\033[38;2;194;178;128m', '#C2B280'),
        'lilac': ('\033[38;2;200;162;200m', '#C8A2C8')
    }
