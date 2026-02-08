"""
System config runtime helpers for System.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import configparser
import logging
import os
from collections import OrderedDict

from andes.core import Config
from andes.shared import np
from andes.utils.paths import confirm_overwrite, get_config_path

logger = logging.getLogger(__name__)


class SystemConfigRuntime:
    """
    Manage system-level configuration bootstrapping and persistence.
    """

    def __init__(self, system):
        self.system = system

    def initialize(self, config=None, config_path=None, default_config=False):
        """
        Initialize and validate ``system.config`` and runtime NumPy options.
        """
        system = self.system

        # get and load default config file
        system._config_path = get_config_path()
        if config_path is not None:
            system._config_path = config_path
        if default_config is True:
            system._config_path = None

        system._config_object = self.load_config_rc(system._config_path)
        self.update_config_object()
        system.config = Config(system.__class__.__name__, dct=config)
        system.config.load(system._config_object)

        # custom configuration for system goes after this line
        self._add_system_defaults()

        system.config._deprecated.update({'warn_limits'})

        system.config.check()
        self.configure_numpy(seed=system.config.seed,
                             divide=system.config.np_divide,
                             invalid=system.config.np_invalid,
                             )

    def _add_system_defaults(self):
        """
        Add default system config entries and metadata.
        """
        system = self.system
        system.config.add(OrderedDict((('freq', 60),
                                       ('mva', 100),
                                       ('ipadd', 1),
                                       ('seed', 'None'),
                                       ('diag_eps', 1e-8),
                                       ('warn_abnormal', 1),
                                       ('dime_enabled', 0),
                                       ('dime_name', 'andes'),
                                       ('dime_address', 'ipc:///tmp/dime2'),
                                       ('numba', 0),
                                       ('numba_parallel', 0),
                                       ('numba_nopython', 1),
                                       ('yapf_pycode', 0),
                                       ('save_stats', 0),
                                       ('np_divide', 'warn'),
                                       ('np_invalid', 'warn'),
                                       )))
        system.config.add_extra("_help",
                                freq='base frequency [Hz]',
                                mva='system base MVA',
                                ipadd='use spmatrix.ipadd if available',
                                seed='seed (or None) for random number generator',
                                diag_eps='small value for Jacobian diagonals',
                                warn_abnormal='warn initialization out of normal values',
                                numba='use numba for JIT compilation',
                                numba_parallel='enable parallel for numba.jit',
                                numba_nopython='nopython mode for numba',
                                yapf_pycode='format generated code with yapf',
                                save_stats='store statistics of function calls',
                                np_divide='treatment for division by zero',
                                np_invalid='treatment for invalid floating-point ops.',
                                )
        system.config.add_extra("_alt",
                                freq="float",
                                mva="float",
                                ipadd=(0, 1),
                                seed='int or None',
                                warn_abnormal=(0, 1),
                                numba=(0, 1),
                                numba_parallel=(0, 1),
                                numba_nopython=(0, 1),
                                yapf_pycode=(0, 1),
                                save_stats=(0, 1),
                                np_divide={'ignore', 'warn', 'raise', 'call', 'print', 'log'},
                                np_invalid={'ignore', 'warn', 'raise', 'call', 'print', 'log'},
                                )

    def update_config_object(self):
        """
        Change config on the fly based on command-line options.
        """
        system = self.system
        config_option = system.options.get('config_option', None)
        if config_option is None:
            return

        if len(config_option) == 0:
            return

        newobj = False
        if system._config_object is None:
            system._config_object = configparser.ConfigParser()
            newobj = True

        for item in config_option:

            # check the validity of the config field
            # each field follows the format `SECTION.FIELD = VALUE`

            if item.count('=') != 1:
                raise ValueError('config_option "{}" must be an assignment expression'.format(item))

            field, value = item.split("=")

            if field.count('.') != 1:
                raise ValueError('config_option left-hand side "{}" must use format SECTION.FIELD'.format(field))

            section, key = field.split(".")

            section = section.strip()
            key = key.strip()
            value = value.strip()

            if not newobj:
                system._config_object.set(section, key, value)
                logger.debug("Existing config option set: %s.%s=%s", section, key, value)
            else:
                system._config_object.add_section(section)
                system._config_object.set(section, key, value)
                logger.debug("New config option added: %s.%s=%s", section, key, value)

    def set_config(self, config=None):
        """
        Set configuration for the System object.

        Config for models are routines are passed directly to their
        constructors.
        """
        system = self.system
        if config is not None:
            # set config for system
            if system.__class__.__name__ in config:
                system.config.add(config[system.__class__.__name__])
                logger.debug("Config: set for System")

    def collect_config(self):
        """
        Collect config data from models.

        Returns
        -------
        dict
            a dict containing the config from devices; class names are keys and
            configs in a dict are values.
        """
        system = self.system
        config_dict = configparser.ConfigParser()
        config_dict[system.__class__.__name__] = system.config.as_dict()

        all_with_config = OrderedDict(list(system.routines.items()) +
                                      list(system.models.items()))

        for name, instance in all_with_config.items():
            cfg = instance.config.as_dict()
            if len(cfg) > 0:
                config_dict[name] = cfg
        return config_dict

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

    @staticmethod
    def configure_numpy(seed='None', divide='warn', invalid='warn'):
        """
        Configure NumPy based on Config.
        """

        # set up numpy random seed
        if isinstance(seed, int):
            np.random.seed(seed)
            logger.debug("Random seed set to <%d>.", seed)

        # set levels
        np.seterr(divide=divide,
                  invalid=invalid,
                  )

    @staticmethod
    def load_config_rc(conf_path=None):
        """
        Load config from an rc-formatted file.

        Parameters
        ----------
        conf_path : None or str
            Path to the config file. If is `None`, the function body will not
            run.

        Returns
        -------
        configparse.ConfigParser
        """
        if conf_path is None:
            return

        conf = configparser.ConfigParser()
        conf.read(conf_path)
        logger.info('> Loaded config from file "%s"', conf_path)
        return conf
