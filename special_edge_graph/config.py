import json

CONFIG = None
with open('config.json') as f:
    CONFIG = json.load(f)