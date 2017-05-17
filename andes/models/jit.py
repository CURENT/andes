import importlib


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
        """import and instantiate this JIT object"""
        try:
            model = importlib.import_module('.'+self.model, 'andes.models')
            device = getattr(model, self.device)
            self.system.__dict__[self.name] = device(self.system, self.name)
            self.system.DevMan.register_device(self.name)  # register device after loading
            self.loaded = 1
            self.system.Log.debug('Imported model <{:s}.{:s}>.'.format(self.model, self.device))
        except ImportError:
            self.system.Log.error('Error importing non-JIT model <{:s}.{:s}> while instantiating powersystem class'
                                  .format(self.model, self.device))
        except AttributeError:
            self.system.Log.error('Error importing a non-existent model <{:s}.{:s}>. Check __init__.py in models'
                                  .format(self.model, self.device))
        except:
            self.system.Log.error('Unknown error importing <{:s}.{:s}>.'.format(self.model, self.device))

    def __getattr__(self, attr):
        if not self.loaded:
            self.jit_load()
        if attr in self.system.__dict__[self.name].__dict__:
            return self.system.__dict__[self.name].__dict__[attr]
        else:
            self.system.Log.warning('Instance <{:s}> does not have <{:s}> attribute.'.format(self.name, attr))

    def add(self, idx=None, name=None, **kwargs):
        """overloading add function of a JIT class"""
        self.jit_load()
        if self.loaded:
            self.system.__dict__[self.name].add(idx, name, **kwargs)

    def help_doc(self, **kwargs):
        self.jit_load()
        if self.loaded:
            self.system.__dict__[self.name].help_doc(**kwargs)

