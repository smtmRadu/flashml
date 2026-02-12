import yaml
from pathlib import Path
from typing import Any


# ANSI color codes
class Colors:
    BRIGHT_CYAN = '\033[96m'      # Tree lines
    WHITE = '\033[97m'            # Field names
    YELLOW = '\033[33m'  # Values (bold)
    THIN_MAGENTA = '\033[2;35m'   # Types (dim/thin)
    RESET = '\033[0m'             # Reset to default


def _auto_convert_type(value: Any) -> Any:
    """
    Automatically convert string representations to appropriate types.
    Tries: bool -> int -> float -> keeps as string
    """
    if not isinstance(value, str):
        return value
    
    # Try boolean conversion
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Try integer conversion
    try:
        # Check if it looks like an int (no decimal point or scientific notation)
        if '.' not in value and 'e' not in value.lower():
            return int(value)
    except (ValueError, AttributeError):
        pass
    
    # Try float conversion (handles scientific notation like 1e-3)
    try:
        return float(value)
    except (ValueError, AttributeError):
        pass
    
    # Keep as string if all conversions fail
    return value


class ConfigObject:
    """
    An object that allows attribute-style access to nested dictionaries.
    Similar to Hydra's OmegaConf but simpler. Preserves original types.
    """
    
    def __init__(self, data: dict):
        """Initialize ConfigObject from a dictionary, preserving types."""
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts to ConfigObjects
                setattr(self, key, ConfigObject(value))
            elif isinstance(value, list):
                # Handle lists that might contain dicts
                setattr(self, key, self._process_list(value))
            else:
                # Auto-convert strings to appropriate types, keep others as-is
                setattr(self, key, _auto_convert_type(value))
    
    def _process_list(self, lst: list) -> list:
        """Process list items, converting dicts to ConfigObjects and strings to appropriate types."""
        result = []
        for item in lst:
            if isinstance(item, dict):
                result.append(ConfigObject(item))
            elif isinstance(item, list):
                result.append(self._process_list(item))
            else:
                result.append(_auto_convert_type(item))
        return result
    
    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"ConfigObject({attrs})"

    def __str__(self) -> str:
        """Return a pretty, fully-expanded tree of the whole config with colors."""
        lines = []
        self._build_tree(lines, prefix="", is_last=True)
        return "\n".join(lines)

    def _build_tree(
        self,
        lines: list[str],
        prefix: str = "",
        is_last: bool = True,
    ) -> None:
        """
        Recursively build an ASCII tree similar to the Linux `tree` command with colors.
        
        Args:
            lines: List to accumulate output lines
            prefix: Indentation string built up while descending
            is_last: True if this node is the last sibling
        """
        items = [(k, v) for k, v in self.__dict__.items() if not k.startswith("_")]

        for idx, (key, value) in enumerate(items):
            is_last_item = idx == len(items) - 1
            corner = "└── " if is_last_item else "├── "
            
            if isinstance(value, ConfigObject):
                # Add the key as a branch node
                # Tree lines in bright cyan, field name in white
                lines.append(
                    f"{Colors.BRIGHT_CYAN}{prefix}{corner}{Colors.RESET}"
                    f"{Colors.WHITE}{key}{Colors.RESET}"
                )
                # Recursively build the tree for nested ConfigObject
                extension = "    " if is_last_item else "│   "
                value._build_tree(lines, prefix + extension, is_last_item)
            elif isinstance(value, list):
                # Add the key as a branch node
                lines.append(
                    f"{Colors.BRIGHT_CYAN}{prefix}{corner}{Colors.RESET}"
                    f"{Colors.WHITE}{key}{Colors.RESET}"
                )
                # Print list elements
                extension = "    " if is_last_item else "│   "
                for i, elem in enumerate(value):
                    elem_is_last = i == len(value) - 1
                    elem_corner = "└── " if elem_is_last else "├── "
                    
                    if isinstance(elem, ConfigObject):
                        lines.append(
                            f"{Colors.BRIGHT_CYAN}{prefix}{extension}{elem_corner}{Colors.RESET}"
                            f"{Colors.WHITE}[{i}]{Colors.RESET}"
                        )
                        elem_extension = "    " if elem_is_last else "│   "
                        elem._build_tree(lines, prefix + extension + elem_extension, elem_is_last)
                    else:
                        # Show type and value for leaf elements
                        type_str = type(elem).__name__
                        lines.append(
                            f"{Colors.BRIGHT_CYAN}{prefix}{extension}{elem_corner}{Colors.RESET}"
                            f"{Colors.WHITE}[{i}]: {Colors.RESET}"
                            f"{Colors.YELLOW}{elem!r}{Colors.RESET} "
                            f"{Colors.THIN_MAGENTA}({type_str}){Colors.RESET}"
                        )
            else:
                # Leaf value - show type to verify it's preserved
                type_str = type(value).__name__
                lines.append(
                    f"{Colors.BRIGHT_CYAN}{prefix}{corner}{Colors.RESET}"
                    f"{Colors.WHITE}{key}: {Colors.RESET}"
                    f"{Colors.YELLOW}{value!r}{Colors.RESET} "
                    f"{Colors.THIN_MAGENTA}({type_str}){Colors.RESET}"
                )
                
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return getattr(self, key)
    
    def to_dict(self) -> dict:
        """Convert ConfigObject back to a regular dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, ConfigObject):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    result[key] = self._list_to_dict(value)
                else:
                    result[key] = value
        return result
    
    def _list_to_dict(self, lst: list) -> list:
        """Convert list items back to dicts if they're ConfigObjects."""
        result = []
        for item in lst:
            if isinstance(item, ConfigObject):
                result.append(item.to_dict())
            elif isinstance(item, list):
                result.append(self._list_to_dict(item))
            else:
                result.append(item)
        return result
    
    def to_list(self) -> list[tuple[str, Any]]:
        """
        Flatten the entire configuration into a list of (path, value) tuples.
        
        Example output:
            [
                ("model.layers[0].units", 256),
                ("training.lr", 0.001),
            ]
        
        Lists produce paths like: key[0], key[1], etc.
        """
        flattened = []
        self._flatten("", self, flattened)
        return flattened

    def _flatten(self, prefix: str, value: Any, out: list):
        """
        Recursive helper that walks ConfigObject, dicts, lists, and primitives.
        """
        if isinstance(value, ConfigObject):
            # Iterate all attributes
            for key, val in value.__dict__.items():
                if key.startswith("_"):
                    continue
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._flatten(new_prefix, val, out)

        elif isinstance(value, list):
            # Handle indexed elements
            for i, item in enumerate(value):
                new_prefix = f"{prefix}[{i}]"
                self._flatten(new_prefix, item, out)

        else:
            # Primitive value → store
            out.append((prefix, value))


def load_yaml_configs(config_path: str = "configs") -> ConfigObject:
    """
    Load YAML configuration files from a directory structure and return
    a ConfigObject with attribute-style access, similar to Hydra.
    
    Args:
        config_path: Path to the configs folder
        
    Returns:
        ConfigObject with nested attributes matching the folder/file structure
        
    Example:
        Given this structure:
            configs/
            ├── model_config.yaml
            └── special_configs/
                └── special.yaml
        
        Usage:
            args = load_yaml_configs("configs")
            print(args.model_config.name)
            print(args.special_configs.special.field_a)
            
    Notes:
        - YAML files become attributes named after the file (without .yaml extension)
        - Folders become nested ConfigObjects
        - Both .yaml and .yml extensions are supported
        - Original types (int, float, str, bool) are preserved
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")
    
    if not config_path.is_dir():
        raise ValueError(f"Config path must be a directory: {config_path}")
    
    config_dict = _load_config_recursive(config_path)
    
    return ConfigObject(config_dict)


def _load_config_recursive(path: Path) -> dict:
    """
    Recursively load YAML files from a directory structure.
    
    Args:
        path: Path to process
        
    Returns:
        Dictionary with loaded configs (types preserved from YAML)
    """
    result = {}
    
    # Process all items in the directory
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix in ['.yaml', '.yml']:
            # Load YAML file - yaml.safe_load preserves types
            config_name = item.stem  # filename without extension
            with open(item, 'r') as f:
                content = yaml.safe_load(f)
                # Handle empty files
                result[config_name] = content if content is not None else {}
                
        elif item.is_dir() and not item.name.startswith('.'):
            # Recursively process subdirectories
            folder_name = item.name
            result[folder_name] = _load_config_recursive(item)
    
    return result