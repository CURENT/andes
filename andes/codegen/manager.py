import importlib
import inspect
import os
import logging
import sys
import time
from collections import OrderedDict
from typing import Union, Optional

import andes
from andes.core import Model, ModelCall
from andes.models import file_classes
from andes.core.symprocessor import SymProcessor
from andes.shared import NCPUS_PHYSICAL, Pool, Process, numba, dilled_vars
from andes.utils.misc import elapsed
from andes.utils.paths import get_dot_andes_path, get_pycode_path, andes_root

logger = logging.getLogger(__name__)


class PyCodeGenerator:
    """
    Class for managing code generation and loading.
    """

    def __init__(self,
                 models: OrderedDict,
                 options=None) -> None:

        self.models = models
        self.options = {} if options is None else options

        self.path = None

    def run(self,
            path: str,
            quick: bool = True,
            models: bool = None,
            nomp: bool = False,
            ncpu: int = NCPUS_PHYSICAL,
            with_yapf: bool = False,
            ):
        """
        Generate numerical functions from symbolically defined models.

        All procedures in this function must be independent of test case.

        Parameters
        ----------
        path : str
            Path to the pycode folder
        quick : bool, optional
            True to skip pretty-print generation to reduce code generation time.
        models : list, OrderedDict, None
            List or OrderedList of models to prepare
        nomp : bool
            True to disable multiprocessing
        with_yapf : bool
            True to run yapf on generated code

        Notes
        -----
        Option ``incremental`` compares the md5 checksum of all var and service
        strings, and only regenerate for updated models.

        Examples
        --------
        If one needs to print out LaTeX-formatted equations in a Jupyter
        Notebook, one need to generate such equations with ::

            import andes sys = andes.prepare()

        Alternatively, one can explicitly create a System and generate the code
        ::

            import andes sys = andes.System() sys.prepare()

        Warnings
        --------
        Generated lambda functions will be stored in Python modules.Pretty
        prints (SymPy objects) can only exist in the System instance on which
        prepare is called.
        """
        t0, _ = elapsed()
        total = len(models)
        width = len(str(total))

        if nomp is False:
            print(f"Generating code for {total} models on {ncpu} processes.")
            self._mp_prepare(models, quick, path, ncpu=ncpu, with_yapf=with_yapf)

        else:
            for idx, (name, model) in enumerate(models.items()):
                print(f"\r\x1b[K Generating code for {name} ({idx+1:>{width}}/{total:>{width}}).",
                      end='\r', flush=True)
                # model.prepare(quick=quick, pycode_path=path, yapf_pycode=with_yapf)
                syms = SymProcessor(model)
                syms.run(quick=quick, pycode_path=path, yapf_pycode=with_yapf)

        if len(models) > 0:
            self._finalize_pycode(path)

        _, s = elapsed(t0)
        logger.info('Generated numerical code for %d models in %s.', len(models), s)

    def _mp_prepare(self, models, quick, pycode_path, ncpu, with_yapf):
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

        model_names = list(models.keys())
        model_list = list()
        # model_list = list(models.values())

        for fname, cls_list in file_classes:
            for model_name in cls_list:
                if model_name not in model_names:
                    continue
                the_module = importlib.import_module('andes.models.' + fname)
                the_class = getattr(the_module, model_name)

                mdl = the_class(system=None, config=None)
                # mdl.set_config(mdl.create_config())

                model_list.append(mdl)

        def _prep_model(model: Model, ):
            """
            Wrapper function to call prepare on a model.
            """
            # model.prepare(quick=quick,
            #               pycode_path=pycode_path,
            #               yapf_pycode=with_yapf
            #               )
            syms = SymProcessor(model)
            syms.run(quick=quick, pycode_path=pycode_path, yapf_pycode=with_yapf)

        Pool(ncpu).map(_prep_model, model_list)

    def _finalize_pycode(self, pycode_path):
        """
        Helper function for finalizing pycode generation by writing
        ``__init__.py``.
        """

        init_path = os.path.join(pycode_path, '__init__.py')

        with open(init_path, 'w') as f:
            f.write(f"__version__ = '{andes.__version__}'\n\n")

            for name in self.models.keys():
                f.write(f"from . import {name:20s}  # NOQA\n")
            f.write('\n')

        logger.info('Saved generated pycode to "%s"', pycode_path)

    def precompile(self,
                   models: Union[OrderedDict, None] = None,
                   nomp: bool = False,
                   ncpu: int = NCPUS_PHYSICAL):
        """
        Trigger precompilation for the given models.

        Arguments are the same as ``prepare``.
        """

        t0, _ = elapsed()

        if models is None:
            models = self.models
        else:
            models = self._get_models(models)

        self.setup()
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

    def _init_numba(self, models: OrderedDict):
        """
        Helper function to compile all functions with Numba before init.
        """

        try:
            getattr(numba, '__version__')
        except ImportError:
            # numba not installed
            logger.warning("numba is enabled but not installed. Please install numba manually.")
            return False

        use_parallel = False  # TODO
        nopython = True  # TODO

        logger.info("Numba compilation initiated with caching.")

        for mdl in models.values():
            mdl.numba_jitify(parallel=use_parallel,
                             nopython=nopython,
                             )

        return True

    def _get_models(self, models):
        """
        Helper function for sanitizing the ``models`` input.

        The output is an OrderedDict of model names and instances.
        """
        out = OrderedDict()

        if isinstance(models, OrderedDict):
            out.update(models)

        elif models is None:
            out.update(self.exist.pflow)

        elif isinstance(models, str):
            out[models] = self.__dict__[models]

        elif isinstance(models, Model):
            out[models.class_name] = models

        elif isinstance(models, list):
            for item in models:
                if isinstance(item, Model):
                    out[item.class_name] = item
                elif isinstance(item, str):
                    out[item] = self.__dict__[item]
                else:
                    raise TypeError(f'Unknown type {type(item)}')

        return out


class PyCodeManager:

    def __init__(self,
                 models: OrderedDict,
                 path=None,
                 ) -> None:

        self.models = models
        self.path: Optional[str] = None

        self.generator = PyCodeGenerator(models)
        self.loader = PyCodeLoader(models)

        self.set_path(path)

    def set_path(self, path: Optional[str] = None) -> None:
        """
        Get and set path to the ``pycode`` folder.
        """

        if path is None:
            path = os.path.join(get_dot_andes_path(), 'pycode')

        os.makedirs(path, exist_ok=True)
        self.path = path
        logger.debug("Using pycode_path: %s", self.path)

    def load(self,
             path: Optional[str] = None,
             ) -> OrderedDict:
        """
        Function for loading ``pycode`` into a dict of calls.
        """

        self.set_path(path)

        if self.regenerate_stale() is True:

            self.loader.load_calls(self.path)

        return self.loader.get_calls()

    def regenerate_stale(self):
        """
        Regenerate stale models.

        Returns
        -------
        bool
            True if any model is regenerated.
        """

        stale_models = OrderedDict()

        try:
            self.loader.load_calls(self.path)
            stale_models = self.loader.find_stale()
        except ImportError:
            stale_models = self.models
            logger.debug("No pycode found. Regenerating all models.")

        if len(stale_models) > 0:
            logger.info("Generated code for <%s> is stale.",
                        ', '.join(stale_models.keys()))

            self.generator.run(self.path,
                               quick=True,
                               models=stale_models,
                               )

            return True

        else:
            logger.info("All models are up to date. No code regeneration needed.")

            return False

    def generate(self,
                 quick=False,
                 incremental=False,
                 ) -> bool:
        """
        Function for generating code and store them into the ``pycode`` module.
        """

        if incremental is True:
            mode_text = 'rapid incremental mode'
        elif quick is True:
            mode_text = 'quick mode'
        else:
            mode_text = 'full mode'

        logger.info('Numerical code generation (%s) started...', mode_text)

        # determine which models to prepare based on mode and `models` list.
        if incremental:
            self.regenerate_stale()

        else:
            self.generator.run(self.path,
                               quick=quick,
                               models=self.models)


class PyCodeLoader:

    def __init__(self, models) -> None:
        self.models = models
        self.path = None

        self.pycode_dict = None

    def get_calls(self):
        """
        Return the loaded pycode dict.
        """

        return self.pycode_dict

    def load_calls(self, pycode_path: str):
        """
        Helper function for loading generated numerical functions from the
        ``pycode`` module.
        """

        pycode_module = import_pycode(user_pycode_path=pycode_path)

        try:
            self.pycode_dict = self._expand_pycode(pycode_module)
        except KeyError:
            logger.error("Your generated pycode is broken. Run `andes prep` to re-generate. ")

        return self.pycode_dict

    def _expand_pycode(self, pycode_module):
        """
        Expand imported ``pycode`` module to model calls.

        Parameters
        ----------
        pycode : module
            The module for generated code for models.
        """

        pycode_dict = dict()

        for name, model in self.models.items():
            if name not in pycode_module.__dict__:
                logger.debug("Model %s does not exist in pycode", name)
                continue

            pycode_model = pycode_module.__dict__[model.class_name]

            calls = ModelCall()

            # md5
            calls.md5 = getattr(pycode_model, 'md5', None)

            # reload stored variables
            for item in dilled_vars:
                calls.__dict__[item] = pycode_model.__dict__[item]

            # equations
            calls.f = pycode_model.__dict__.get("f_update")
            calls.g = pycode_model.__dict__.get("g_update")

            # services
            for instance in model.services.values():
                if (instance.v_str is not None) and instance.sequential is True:
                    sv_name = f'{instance.name}_svc'
                    calls.s[instance.name] = pycode_model.__dict__[sv_name]

            # services - non sequential
            calls.sns = pycode_model.__dict__.get("sns_update")

            # load initialization; assignment
            for instance in model.cache.all_vars.values():
                if instance.v_str is not None:
                    ia_name = f'{instance.name}_ia'
                    calls.ia[instance.name] = pycode_model.__dict__[ia_name]

            # load initialization: iterative
            for item in calls.init_seq:
                if isinstance(item, list):
                    name_concat = '_'.join(item)
                    calls.ii[name_concat] = pycode_model.__dict__[name_concat + '_ii']
                    calls.ij[name_concat] = pycode_model.__dict__[name_concat + '_ij']

            # load Jacobian functions
            for jname in calls.j_names:
                calls.j[jname] = pycode_model.__dict__.get(f'{jname}_update')

            pycode_dict[name] = calls

        return pycode_dict

    def find_stale(self):
        """
        Find models whose ModelCall are stale using md5 checksum.
        """
        out = OrderedDict()
        for model in self.models.values():
            calls_md5 = getattr(self.pycode_dict[model.class_name], 'md5', None)

            if calls_md5 != model.get_md5():
                out[model.class_name] = model

        return out


def import_pycode(user_pycode_path=None):
    """
    Helper function to import generated pycode in the following priority:

    1. a user-provided path from CLI. Currently, this is only for specifying the
       path to store the generated pycode via ``andes prepare``.
    2. ``~/.andes/pycode``. This is where pycode is stored by default.
    3. ``<andes_package_root>/pycode``. One can store pycode in the ANDES
       package folder and ship a full package, which does not require code generation.
    """

    # below are executed serially because of priority
    pycode = reload_submodules('pycode')

    if pycode is None:
        pycode_path = get_pycode_path(user_pycode_path, mkdir=False)
        pycode = _import_pycode_from(pycode_path)

    if pycode is None:
        pycode = reload_submodules('andes.pycode')

    if pycode is None:
        pycode = _import_pycode_from(os.path.join(andes_root(), 'pycode'))

    if pycode is None:
        raise ImportError("Pycode module import failed")

    return pycode


def _import_pycode_from(pycode_path):
    """
    Helper function to load pycode from ``.andes``.
    """

    MODULE_PATH = os.path.join(pycode_path, '__init__.py')
    MODULE_NAME = 'pycode'

    pycode = None
    if os.path.isfile(MODULE_PATH):
        try:
            spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
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


if __name__ == '__main__':

    from andes.system import ModelManager
    andes.config_logger(20)

    mm = ModelManager()
    pcm = PyCodeManager(models=mm.models)
    pcm.generate(quick=True, incremental=False)
