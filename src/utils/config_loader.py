import yaml

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    def get(self, key_path: str, default=None):
        """
        Get a nested value from the config using dot notation, e.g., 'data.real_dir'
        """
        keys = key_path.split(".")
        value = self.config
        try:
            for key in keys:
                value = value[key]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(f"Key '{key_path}' not found in config.")
        return value
