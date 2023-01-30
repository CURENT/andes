"""
Module for config manager
"""

import os
import logging

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
