import importlib
import logging

logger = logging.getLogger(__name__)


class JIT(object):
    """Dummy Just-in-Time initialization class"""

    def __init__(self, system, model, device, name):
        """constructor of a dummy JIT class with basic information"""
        self.system = system
        self.model = model
        self.device = device
        self.name = name
        self.loaded = 0

    def jit_load(self):
        """
        Import and instantiate this JIT object

        Returns
        -------

        """
        try:
            model = importlib.import_module('.' + self.model, 'andes.models')
            device = getattr(model, self.device)
            self.system.__dict__[self.name] = device(self.system, self.name)

            g = self.system.__dict__[self.name]._group
            self.system.group_add(g)
            self.system.__dict__[g].register_model(self.name)

            # register device after loading
            self.system.devman.register_device(self.name)
            self.loaded = 1
            logger.debug('Imported model <{:s}.{:s}>.'.format(
                self.model, self.device))
        except ImportError:
            logger.error(
                'non-JIT model <{:s}.{:s}> import error'
                .format(self.model, self.device))
        except AttributeError:
            logger.error(
                'model <{:s}.{:s}> not exist. Check models/__init__.py'
                .format(self.model, self.device))

    def __getattr__(self, attr):
        if not self.loaded:
            self.jit_load()
        if attr in self.system.__dict__[self.name].__dict__:
            return self.system.__dict__[self.name].__dict__[attr]
        else:
            logger.warning(
                'Instance <{:s}> does not have <{:s}> attribute.'.format(
                    self.name, attr))

    def elem_add(self, idx=None, name=None, **kwargs):
        """overloading elem_add function of a JIT class"""
        self.jit_load()
        if self.loaded:
            return self.system.__dict__[self.name].elem_add(
                idx, name, **kwargs)

    def doc(self, **kwargs):
        self.jit_load()
        if self.loaded:
            return self.system.__dict__[self.name].doc(**kwargs)
