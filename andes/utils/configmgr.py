"""
Module for config manager
"""

import os
import logging

from collections import OrderedDict

from andes.utils.paths import confirm_overwrite
from typing import Optional, List
import configparser

logger = logging.getLogger(__name__)


class ConfigManager:

    def __init__(self) -> None:

        self.conf_path = None
        self._registered = dict()
        self._config_obj = None

        self._store = dict()

    def set_path(self, path=None, file_name='andes.rc'):
        """
        Set path to `andes.rc` config file.

        Using `default_config`, one should not call this function.
        """

        search_path = [path,
                       os.getcwd(),
                       os.path.join(os.path.expanduser('~'), ".andes"),
                       ]

        for p in search_path:
            if p is None:
                continue

            if os.path.isfile(os.path.join(p, file_name)):
                self.conf_path = os.path.join(p, file_name)
                break

    def register(self, field_name, create_func):
        """
        Register a config field and its creation function.
        """

        self._registered[field_name] = create_func

    def load_config_rc(self):
        """
        Load config file into a ConfigParser object and store it in
        ``self._config_object``.
        """

        conf = configparser.ConfigParser()
        conf.read(self.conf_path)
        logger.info('> Loaded config from file "%s"', self.conf_path)

        self._config_obj = conf

    def parse_options(self, config_option: Optional[List[str]] = None):
        """
        Parse command-line options and update ``self.config_object``.
        """

        if config_option is None:
            return

        if len(config_option) == 0:
            return

        newobj = False

        if self._config_obj is None:
            self._config_obj = configparser.ConfigParser()
            newobj = True

        for item in config_option:

            # check the validity of the config field
            # each field follows the format `SECTION.FIELD = VALUE`

            if item.count('=') != 1:
                raise ValueError(
                    'config_option "{}" must be an assignment expression'.format(item))

            field, value = item.split("=")

            if field.count('.') != 1:
                raise ValueError(
                    'config_option LHS "{}" must use format SECTION.FIELD'.format(field))

            section, key = field.split(".")

            section = section.strip()
            key = key.strip()
            value = value.strip()

            if not newobj:
                self._config_obj.set(section, key, value)
                logger.debug("Existing config option set: %s.%s=%s", section, key, value)

            else:
                self._config_obj.add_section(section)
                self._config_obj.set(section, key, value)
                logger.debug("New config option added: %s.%s=%s", section, key, value)

    def create(self):
        """
        Create Config object for the given field name.
        """

        for field in self._registered:
            self._store[field] = self._registered[field](self._config_obj)
            self._store[field].check()

    def __getattr__(self, field_name):
        return self._store[field_name]

    def save_config(self, file_path=None, overwrite=False):
        """
        Save all system, model, and routine configurations to an rc-formatted
        file.

        Parameters
        ----------
        file_path : str, optional
            path to the configuration file default to `~/andes/andes.rc`.
        overwrite : bool, optional
            If file exists, True to overwrite without confirmation. Otherwise
            prompt for confirmation.

        Warnings
        --------
        Saved config is loaded back and populated *at system instance creation
        time*. Configs from the config file takes precedence over default config
        values.
        """
        if file_path is None:
            andes_path = os.path.join(os.path.expanduser('~'), '.andes')
            os.makedirs(andes_path, exist_ok=True)
            file_path = os.path.join(andes_path, 'andes.rc')

        elif os.path.isfile(file_path):
            if not confirm_overwrite(file_path, overwrite=overwrite):
                return

        conf = self.collect_config()
        with open(file_path, 'w') as f:
            conf.write(f)

        logger.info('Config written to "%s"', file_path)
        return file_path

    def collect_config(self):
        """
        Collect config data from models.

        Returns
        -------
        dict
            a dict containing the config from devices; class names are keys and
            configs in a dict are values.
        """
        config_dict = configparser.ConfigParser()
        config_dict[self.__class__.__name__] = self.config.as_dict()

        all_with_config = OrderedDict(list(self.routines.items()) +
                                      list(self.models.items()))

        for name, instance in all_with_config.items():
            cfg = instance.config.as_dict()
            if len(cfg) > 0:
                config_dict[name] = cfg
        return config_dict
