import json

def read_json(json_fn):
    with open('interactive.json', 'r') as f:
        dict = json.load(f)
        f.close()

    return dict
        
def write_json(dict, json_fn):
    with open(json_fn, 'w') as f:
        json.dump(dict, f, indent=6)
        f.close()