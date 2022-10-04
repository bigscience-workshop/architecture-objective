import json


def extract_json(json_file_path: str) -> dict:
    with open(json_file_path) as json_file:
        return json.load(json_file)
