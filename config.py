import json


class Config:
    def __init__(self, config_path='config.json'):
        with open(config_path) as f:
            config = json.load(f)
        self.__dict__ = config
