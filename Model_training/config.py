import json
def load_config():
    with open('config_template.json') as f:
        data = json.load(f)
    return data