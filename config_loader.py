import os
import json

# Determine the project's root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the config.json file
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

# Load the JSON configuration
with open(CONFIG_PATH, "r") as config_file:
    CONFIG = json.load(config_file)
