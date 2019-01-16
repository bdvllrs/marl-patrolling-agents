import os
import yaml


def update_config(conf, new_conf):
    for item in new_conf.keys():
        if type(new_conf[item]) == dict and item in conf.keys():
            conf[item] = update_config(conf[item], new_conf[item])
        else:
            conf[item] = new_conf[item]
    return conf


class Config:
    def __init__(self, path=None, config=None):
        self.__is_none = False
        self.__data = config if config is not None else {}
        if path is not None:
            self.__path = os.path.abspath(os.path.join(os.curdir, path))
            with open(os.path.join(self.__path, "default.yaml"), "rb") as default_config:
                self.__data.update(yaml.load(default_config))
            for config in sorted(os.listdir(self.__path)):
                if config != "default.yaml" and config[-4:] in ["yaml", "yml"]:
                    with open(os.path.join(self.__path, config), "rb") as config_file:
                        self.__data = update_config(self.__data, yaml.load(config_file))

    def set(self, key, value):
        self.__data[key] = value

    def __getattr__(self, item):
        if type(self.__data[item]) == dict:
            return Config(config=self.__data[item])
        return self.__data[item]

    def __getitem__(self, item):
        return self.__data[item]
