import json


class Config:
    def __init__(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        self.__dict__ = config
