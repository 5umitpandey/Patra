import yaml

def load_config(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
