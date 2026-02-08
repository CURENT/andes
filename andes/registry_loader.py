"""
Registry loading helpers for System.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import importlib
import inspect
import andes.models
from andes.models import file_classes
from andes.models.group import GroupBase
from andes.routines import all_routines


class RegistryLoader:
    """
    Manage imports and registration for groups, models and routines.
    """

    def __init__(self, system):
        self.system = system

    def load_all(self):
        """
        Load groups, models and routines in dependency order.
        """
        self.import_groups()
        self.import_models()
        self.import_routines()  # routine imports come after models

    def check_group_common(self):
        """
        Check if all group common variables and parameters are met.

        This function is called at the end of code generation by `prepare`.

        Raises
        ------
        KeyError if any parameter or value is not provided
        """
        system = self.system
        for group in system.groups.values():
            for item in group.common_params:
                for model in group.models.values():
                    # the below includes all of BaseParam (NumParam, DataParam and ExtParam)
                    if item not in model.__dict__:
                        if item in model.group_param_exception:
                            continue
                        raise KeyError(f'Group <{group.class_name}> common param <{item}> does not exist '
                                       f'in model <{model.class_name}>')
            for item in group.common_vars:
                for model in group.models.values():
                    if item not in model.cache.all_vars:
                        if item in model.group_var_exception:
                            continue
                        raise KeyError(f'Group <{group.class_name}> common var <{item}> does not exist '
                                       f'in model <{model.class_name}>')

    def import_groups(self):
        """
        Import all groups classes defined in ``models/group.py``.

        Groups will be stored as instances with the name as class names.
        All groups will be stored to dictionary ``System.groups``.
        """
        system = self.system
        module = importlib.import_module('andes.models.group')

        for m in inspect.getmembers(module, inspect.isclass):

            name, cls = m
            if name == 'GroupBase':
                continue
            if not issubclass(cls, GroupBase):
                # skip other imported classes such as `OrderedDict`
                continue

            system.__dict__[name] = cls()
            system.groups[name] = system.__dict__[name]

    def import_models(self):
        """
        Import and instantiate models as System member attributes.

        Models defined in ``models/__init__.py`` will be instantiated `sequentially` as attributes with the same
        name as the class name.
        In addition, all models will be stored in dictionary ``System.models`` with model names as
        keys and the corresponding instances as values.

        Examples
        --------
        ``system.Bus`` stores the `Bus` object, and ``system.GENCLS`` stores the classical
        generator object,

        ``system.models['Bus']`` points the same instance as ``system.Bus``.
        """
        system = self.system
        for fname, cls_list in file_classes:
            for model_name in cls_list:
                the_module = importlib.import_module('andes.models.' + fname)
                the_class = getattr(the_module, model_name)
                system.__dict__[model_name] = the_class(system=system, config=system._config_object)
                system.models[model_name] = system.__dict__[model_name]
                system.models[model_name].config.check()

                # link to the group
                group_name = system.__dict__[model_name].group
                system.__dict__[group_name].add_model(model_name, system.__dict__[model_name])

        for key, val in andes.models.model_aliases.items():
            system.model_aliases[key] = system.models[val]
            system.__dict__[key] = system.models[val]

    def import_routines(self):
        """
        Import routines as defined in ``routines/__init__.py``.

        Routines will be stored as instances with the name as class names.
        All routines will be stored to dictionary ``System.routines``.

        Examples
        --------
        ``System.PFlow`` is the power flow routine instance, and ``System.TDS`` and ``System.EIG`` are
        time-domain analysis and eigenvalue analysis routines, respectively.
        """
        system = self.system
        for file, cls_list in all_routines.items():
            for cls_name in cls_list:
                file = importlib.import_module('andes.routines.' + file)
                the_class = getattr(file, cls_name)
                attr_name = cls_name
                system.__dict__[attr_name] = the_class(system=system, config=system._config_object)
                system.routines[attr_name] = system.__dict__[attr_name]
                system.routines[attr_name].config.check()
