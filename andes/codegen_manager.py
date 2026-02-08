"""
Code generation and pycode loading helpers for System.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import importlib
import importlib.util
import inspect
import logging
import os
import sys
import time
from collections import OrderedDict
from typing import Union

from andes.core import Model
from andes.models import file_classes
from andes.shared import NCPUS_PHYSICAL, Pool, Process, dilled_vars, numba
from andes.utils.misc import elapsed
from andes.utils.paths import andes_root, get_pycode_path

logger = logging.getLogger(__name__)


class CodegenManager:
    """
    Manage symbolic code generation, loading and pre-compilation.
    """

    def __init__(self, system):
        self.system = system

    def prepare(self, quick=False, incremental=False, models=None, nomp=False, ncpu=NCPUS_PHYSICAL):
        """
        Generate numerical functions from symbolically defined models.

        All procedures in this function must be independent of test case.

        Parameters
        ----------
        quick : bool, optional
            True to skip pretty-print generation to reduce code generation time.
        incremental : bool, optional
            True to generate only for modified models, incrementally.
        models : list, OrderedDict, None
            List or OrderedList of models to prepare
        nomp : bool
            True to disable multiprocessing

        Notes
        -----
        Option ``incremental`` compares the md5 checksum of all var and
        service strings, and only regenerate for updated models.

        Examples
        --------
        If one needs to print out LaTeX-formatted equations in a Jupyter Notebook, one need to generate such
        equations with ::

            import andes
            sys = andes.prepare()

        Alternatively, one can explicitly create a System and generate the code ::

            import andes
            sys = andes.System()
            sys.prepare()

        Warnings
        --------
        Generated lambda functions will be serialized to file, but pretty prints (SymPy objects) can only exist in
        the System instance on which prepare is called.
        """
        system = self.system
        if incremental is True:
            mode_text = 'rapid incremental mode'
        elif quick is True:
            mode_text = 'quick mode'
        else:
            mode_text = 'full mode'

        logger.info('Numerical code generation (%s) started...', mode_text)

        t0, _ = elapsed()

        # consistency check for group parameters and variables
        system.check_group_common()

        # get `pycode` folder path without automatic creation
        pycode_path = get_pycode_path(system.options.get("pycode_path"), mkdir=False)

        # determine which models to prepare based on mode and `models` list.
        if incremental and models is None:
            if not system.with_calls:
                self._load_calls()
            models = self._find_stale_models()
        elif not incremental and models is None:
            models = system.models
        else:
            models = system._get_models(models)

        total = len(models)
        width = len(str(total))

        if nomp is False:
            print(f"Generating code for {total} models on {ncpu} processes.")
            self._mp_prepare(models, quick, pycode_path, ncpu=ncpu)

        else:
            for idx, (name, model) in enumerate(models.items()):
                print(f"\r\x1b[K Generating code for {name} ({idx+1:>{width}}/{total:>{width}}).",
                      end='\r', flush=True)
                model.prepare(quick=quick, pycode_path=pycode_path)

        if len(models) > 0:
            self._finalize_pycode(pycode_path)
            self._store_calls(models)

        _, s = elapsed(t0)
        logger.info('Generated numerical code for %d models in %s.', len(models), s)

    def _mp_prepare(self, models, quick, pycode_path, ncpu):
        """
        Wrapper for multiprocessed code generation.

        Parameters
        ----------
        models : OrderedDict
            model name : model instance pairs
        quick : bool
            True to skip LaTeX string generation
        pycode_path : str
            Path to store `pycode` folder
        ncpu : int
            Number of processors to use
        """

        # create empty models without dependency
        if len(models) == 0:
            return

        system = self.system
        model_names = list(models.keys())
        model_list = list()

        for fname, cls_list in file_classes:
            for model_name in cls_list:
                if model_name not in model_names:
                    continue
                the_module = importlib.import_module('andes.models.' + fname)
                the_class = getattr(the_module, model_name)
                model_list.append(the_class(system=None, config=system._config_object))

        yapf_pycode = system.config.yapf_pycode

        def _prep_model(model: Model):
            """
            Wrapper function to call prepare on a model.
            """
            model.prepare(quick=quick,
                          pycode_path=pycode_path,
                          yapf_pycode=yapf_pycode
                          )

        Pool(ncpu).map(_prep_model, model_list)

    def _finalize_pycode(self, pycode_path):
        """
        Helper function for finalizing pycode generation by
        writing ``__init__.py`` and reloading ``pycode`` package.
        """
        import andes
        system = self.system
        init_path = os.path.join(pycode_path, '__init__.py')
        with open(init_path, 'w') as f:
            f.write(f"__version__ = '{andes.__version__}'\n\n")

            for name in system.models.keys():
                f.write(f"from . import {name:20s}  # NOQA\n")
            f.write('\n')

        logger.info('Saved generated pycode to "%s"', pycode_path)

        # RELOAD REQUIRED as the generated Jacobian arguments may be in a different order
        importlib.invalidate_caches()
        self._load_calls()

    def _find_stale_models(self):
        """
        Find models whose ModelCall are stale using md5 checksum.
        """
        out = OrderedDict()
        for model in self.system.models.values():
            calls_md5 = getattr(model.calls, 'md5', None)
            if calls_md5 != model.get_md5():
                out[model.class_name] = model

        return out

    def _init_numba(self, models: OrderedDict):
        """
        Helper function to compile all functions with Numba before init.
        """
        system = self.system
        if not system.config.numba:
            return

        try:
            getattr(numba, '__version__')
        except ImportError:
            # numba not installed
            logger.warning("numba is enabled but not installed. Please install numba manually.")
            system.config.numba = 0
            return False

        use_parallel = bool(system.config.numba_parallel)
        nopython = bool(system.config.numba_nopython)

        logger.info("Numba compilation initiated with caching.")

        for mdl in models.values():
            mdl.numba_jitify(parallel=use_parallel,
                             nopython=nopython,
                             )

        return True

    def precompile(self,
                   models: Union[OrderedDict, None] = None,
                   nomp: bool = False,
                   ncpu: int = NCPUS_PHYSICAL):
        """
        Trigger precompilation for the given models.

        Arguments are the same as ``prepare``.
        """
        system = self.system
        t0, _ = elapsed()

        if models is None:
            models = system.models
        else:
            models = system._get_models(models)

        # turn on numba for precompilation
        system.config.numba = 1

        system.setup()
        numba_ok = self._init_numba(models)

        if not numba_ok:
            return

        def _precompile_model(model: Model):
            model.precompile()

        logger.info("Compilation in progress. This might take a minute...")

        if nomp is True:
            for name, mdl in models.items():
                _precompile_model(mdl)
                logger.debug("Model <%s> compiled.", name)

        # multi-processed implementation. `Pool.map` runs very slow somehow.
        else:
            jobs = []
            for idx, (name, mdl) in enumerate(models.items()):
                job = Process(
                    name='Process {0:d}'.format(idx),
                    target=_precompile_model,
                    args=(mdl,),
                )
                jobs.append(job)
                job.start()

                if (idx % ncpu == ncpu - 1) or (idx == len(models) - 1):
                    time.sleep(0.02)
                    for job in jobs:
                        job.join()
                    jobs = []

        _, s = elapsed(t0)
        logger.info('Numba compiled %d model%s in %s.',
                    len(models),
                    '' if len(models) == 1 else 's',
                    s)

    def undill(self, autogen_stale=True):
        """
        Reload generated function functions, from either the
        ``$HOME/.andes/pycode`` folder.

        If no change is made to models, future calls to ``prepare()`` can be
        replaced with ``undill()`` for acceleration.

        Parameters
        ----------
        autogen_stale: bool
            True to automatically call code generation if stale code is
            detected. Regardless of this option, codegen is trigger if importing
            existing code fails.
        """
        # load equations and jacobian from saved code
        loaded = self._load_calls()

        stale_models = self._find_stale_models()

        if loaded is False:
            self.prepare(quick=True, incremental=False)
            loaded = True
        elif len(stale_models) == 0:
            pass
        elif autogen_stale is False:
            # NOTE: incremental code generation may be triggered due to Python
            # not invalidating ``.pyc`` caches. If multiprocessing is being
            # used, code generation will cause nested multiprocessing, which is
            # not allowed.
            # The flag ``autogen_stale=False`` is used to prevent nested codegen
            # and is intended to be used only in multiprocessing.
            logger.info("Generated code for <%s> is stale.", ', '.join(stale_models.keys()))
            logger.info("Automatic code re-generation manually skipped")
            loaded = True
        else:
            logger.info("Generated code for <%s> is stale.", ', '.join(stale_models.keys()))
            self.prepare(quick=True, incremental=True, models=stale_models)
            loaded = True

        return loaded

    def _load_calls(self):
        """
        Helper function for loading generated numerical functions from the ``pycode`` module.
        """
        loaded = False
        user_pycode_path = self.system.options.get("pycode_path")
        pycode = import_pycode(user_pycode_path=user_pycode_path)

        if pycode:
            try:
                self._expand_pycode(pycode)
                loaded = True
            except KeyError:
                logger.error("Your generated pycode is broken. Run `andes prep` to re-generate. ")

        self.system.with_calls = loaded
        return loaded

    def _expand_pycode(self, pycode_module):
        """
        Expand imported ``pycode`` module to model calls.

        Parameters
        ----------
        pycode : module
            The module for generated code for models.
        """
        system = self.system
        for name, model in system.models.items():
            if name not in pycode_module.__dict__:
                logger.debug("Model %s does not exist in pycode", name)
                continue

            pycode_model = pycode_module.__dict__[model.class_name]

            # md5
            model.calls.md5 = getattr(pycode_model, 'md5', None)

            # reload stored variables
            for item in dilled_vars:
                model.calls.__dict__[item] = pycode_model.__dict__[item]

            # equations
            model.calls.f = pycode_model.__dict__.get("f_update")
            model.calls.g = pycode_model.__dict__.get("g_update")

            # services
            for instance in model.services.values():
                if (instance.v_str is not None) and instance.sequential is True:
                    sv_name = f'{instance.name}_svc'
                    model.calls.s[instance.name] = pycode_model.__dict__[sv_name]

            # services - non sequential
            model.calls.sns = pycode_model.__dict__.get("sns_update")

            # load initialization; assignment
            for instance in model.cache.all_vars.values():
                if instance.v_str is not None:
                    ia_name = f'{instance.name}_ia'
                    model.calls.ia[instance.name] = pycode_model.__dict__[ia_name]

            # load initialization: iterative
            for item in model.calls.init_seq:
                if isinstance(item, list):
                    name_concat = '_'.join(item)
                    model.calls.ii[name_concat] = pycode_model.__dict__[name_concat + '_ii']
                    model.calls.ij[name_concat] = pycode_model.__dict__[name_concat + '_ij']

            # load Jacobian functions
            for jname in model.calls.j_names:
                model.calls.j[jname] = pycode_model.__dict__.get(f'{jname}_update')

    def _store_calls(self, models: OrderedDict):
        """
        Collect and store model calls into system.
        """
        import andes
        system = self.system
        logger.debug("Collecting Model.calls into System.")

        system.calls['__version__'] = andes.__version__

        for name, mdl in models.items():
            system.calls[name] = mdl.calls


def import_pycode(user_pycode_path=None):
    """
    Helper function to import generated pycode in the following priority:

    1. a user-provided path from CLI. Currently, this is only for specifying the
       path to store the generated pycode via ``andes prepare``.
    2. ``~/.andes/pycode``. This is where pycode is stored by default.
    3. ``<andes_package_root>/pycode``. One can store pycode in the ANDES
       package folder and ship a full package, which does not require code generation.
    """

    pycode_path = get_pycode_path(user_pycode_path, mkdir=False)
    sources = (
        lambda: reload_submodules('pycode'),
        lambda: _import_pycode_from(pycode_path),
        lambda: reload_submodules('andes.pycode'),
        lambda: _import_pycode_from(os.path.join(andes_root(), 'pycode')),
    )

    for source in sources:
        pycode = source()
        if pycode:
            return pycode

    return None


def _import_pycode_from(pycode_path):
    """
    Helper function to load pycode from ``.andes``.
    """

    module_path = os.path.join(pycode_path, '__init__.py')
    module_name = 'pycode'

    pycode = None
    if os.path.isfile(module_path):
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            pycode = importlib.util.module_from_spec(spec)  # NOQA
            sys.modules[spec.name] = pycode
            spec.loader.exec_module(pycode)
            logger.info('> Loaded generated Python code in "%s".', pycode_path)
        except ImportError:
            logger.debug('> Failed loading generated Python code in "%s".', pycode_path)

    return pycode


def reload_submodules(module_name):
    """
    Helper function for reloading an existing module and its submodules.

    It is used to reload the ``pycode`` module after regenerating code.
    """

    if module_name in sys.modules:
        pycode = sys.modules[module_name]
        for _, m in inspect.getmembers(pycode, inspect.ismodule):
            importlib.reload(m)

        logger.info('> Reloaded generated Python code of module "%s".', module_name)
        return pycode

    return None
