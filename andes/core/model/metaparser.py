"""
Meta parser for defined models.
"""

import logging

from collections import OrderedDict

logger = logging.getLogger(__name__)


class ModelMetaParser:
    """
    This class should output an intermediate representation of the model.
    """

    def __init__(self, model) -> None:
        self.model = model

        self.states = OrderedDict()
        self.algebs = OrderedDict()
        self.states_ext = OrderedDict()
        self.algebs_ext = OrderedDict()

        self.params = OrderedDict()
        self.num_params = OrderedDict()
        self.idx_params = OrderedDict()
        self.timer_params = OrderedDict()
        self.params_ext = OrderedDict()

        self.discrete = OrderedDict()

        self.services_const = OrderedDict()
        self.services_var = OrderedDict()
        self.services_var_seq = OrderedDict()
        self.services_var_nonseq = OrderedDict()
        self.services_post = OrderedDict()
        self.services_subs = OrderedDict()
        self.services_fnd = OrderedDict()
        self.services_ref = OrderedDict()
        self.services_ext = OrderedDict()
        self.services_ops = OrderedDict()
        self.services_icheck = OrderedDict()

        self.blocks = OrderedDict()

        self._skip_classes = ("ModelCall", "ConfigManager")

    def process_attr(self):
        for name, attr in self.model.__dict__.items():
            class_module = attr.__class__.__module__.split('.')[0]
            class_name = attr.__class__.__name__

            if class_module != "andes":
                continue
            elif class_name in self._skip_classes:
                continue

            self.process_one_attr(name, attr)

    def process_one_attr(self, name, attr):

        class_bases = attr.__class__.__bases__
        class_name = attr.__class__.__name__

        if self._call_processor(name, attr, class_name):
            return

        for base in class_bases:
            ret = self._call_processor(name, attr, base.__name__)
            if ret:
                break

    def _call_processor(self, name, attr, class_name):

        if hasattr(self, f'_process_{class_name}'):

            getattr(self, f'_process_{class_name}')(name, attr)
            logger.debug("Processed <%s> as class %s" % (name, class_name))
            return True

        return False

    def _process_State(self, name, attr):
        self.states[name] = attr

    def _process_Algeb(self, name, attr):
        self.algebs[name] = attr

    def _process_ExtState(self, name, attr):
        self.states_ext[name] = attr

    def _process_ExtAlgeb(self, name, attr):
        self.algebs_ext[name] = attr

    def _process_Discrete(self, name, attr):
        self.discrete[name] = attr

    def _process_DataParam(self, name, attr):
        self.params[name] = attr

    def _process_NumParam(self, name, attr):
        self.params[name] = attr
        self.num_params[name] = attr

    def _process_IdxParam(self, name, attr):
        self.params[name] = attr
        self.idx_params[name] = attr

    def _process_TimerParam(self, name, attr):
        self.params[name] = attr
        self.num_params[name] = attr
        self.timer_params[name] = attr

    def _process_ModelFlags(self, name, attr):
        pass

    def _process_ConstService(self, name, attr):
        self.services_const[name] = attr

    def _process_VarService(self, name, attr):
        self.services_const[name] = attr
        self.services_var[name] = attr

        if attr.sequential:
            self.services_var_seq[name] = attr
        else:
            self.services_var_nonseq[name] = attr

    def _process_PostInitService(self, name, attr):
        self.services_const[name] = attr
        self.services_post[name] = attr

    def _process_SubService(self, name, attr):
        self.services_subs[name] = attr

    def _process_DeviceFinder(self, name, attr):
        self.services_fnd[name] = attr

    def _process_BackRef(self, name, attr):
        self.services_ref[name] = attr

    def _process_ExtService(self, name, attr):
        self.services_ext[name] = attr

    def _process_NumReplace(self, name, attr):
        self.services_ops[name] = attr

    def _process_NumReduce(self, name, attr):
        self.services_ops[name] = attr

    def _process_NumSelect(self, name, attr):
        self.services_ops[name] = attr

    def _process_FlagValue(self, name, attr):
        self.services_ops[name] = attr

    def _process_RandomService(self, name, attr):
        self.services_ops[name] = attr

    def _process_SwBlock(self, name, attr):
        self.blocks[name] = attr

    def _process_ParamCalc(self, name, attr):
        self.services_ops[name] = attr

    def _process_ReplCalc(self, name, attr):
        self.services_ops[name] = attr

    def _process_ApplyFunc(self, name, attr):
        self.services_ops[name] = attr

    def _process_InitChecker(self, name, attr):
        self.services.icheck[name] = attr

    def _process_Block(self, name, attr):
        self.blocks[name] = attr
        if attr.namespace == 'local':
            prepend = attr.name + '_'
            tex_append = attr.tex_name
        else:
            prepend = ''
            tex_append = ''

        for var_name, var_instance in attr.export().items():
            var_instance.name = f'{prepend}{var_name}'
            var_instance.tex_name = f'{var_instance.tex_name}_{{{tex_append}}}'
            self.__setattr__(var_instance.name, var_instance)

            self.process_one_attr(var_instance.name, var_instance)
