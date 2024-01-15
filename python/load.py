import toml
import json


def load_simulation_setting_toml(file_path):
    with open(file_path, "r") as toml_file:
        data = toml.load(toml_file)
    return data


def load_result_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data
