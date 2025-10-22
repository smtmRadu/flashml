import yaml
from pathlib import Path
from typing import Any


class ConfigObject:
    """
    A object that allows attribute-style access to nested dictionaries.
    Similar to Hydra's OmegaConf but simpler.
    """
    
    def __init__(self, data: dict):
        """Initialize ConfigObject from a dictionary."""
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts to ConfigObjects
                setattr(self, key, ConfigObject(value))
            elif isinstance(value, list):
                # Handle lists that might contain dicts
                setattr(self, key, self._process_list(value))
            else:
                setattr(self, key, value)
    
    def _process_list(self, lst: list) -> list:
        """Process list items, converting dicts to ConfigObjects."""
        result = []
        for item in lst:
            if isinstance(item, dict):
                result.append(ConfigObject(item))
            elif isinstance(item, list):
                result.append(self._process_list(item))
            else:
                result.append(item)
        return result
    
    def __repr__(self) -> str:
        """String representation of the config object."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"ConfigObject({attrs})"
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access as well."""
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
        Dictionary with loaded configs
    """
    result = {}
    
    # Process all items in the directory
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix in ['.yaml', '.yml']:
            # Load YAML file
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
