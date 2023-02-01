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

        self.all_vars = OrderedDict()
        self.states = OrderedDict()
        self.algebs = OrderedDict()
        self.vars_int = OrderedDict()
        self.states_ext = OrderedDict()
        self.algebs_ext = OrderedDict()
        self.states_and_ext = OrderedDict()
        self.algebs_and_ext = OrderedDict()

        self.all_params = OrderedDict()
        self.params = OrderedDict()
        self.num_params = OrderedDict()
        self.idx_params = OrderedDict()
        self.timer_params = OrderedDict()
        self.params_ext = OrderedDict()

        self.discrete = OrderedDict()

        self.services = OrderedDict()
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
        self.dummy = OrderedDict()

        self.all_params_names = list()

        self._skip_classes = ("ModelCall", "ConfigManager")

    def process_attr(self):
        """
        Entry function to process all attributes in the model.
        """

        for name, attr in self.model.__dict__.items():
            class_module = attr.__class__.__module__.split('.')[0]
            class_name = attr.__class__.__name__

            if class_module != "andes":
                continue
            elif class_name in self._skip_classes:
                continue

            self.process_one_attr(name, attr)

        self.all_params_names = self._process_all_params_names()

    def process_one_attr(self, name, attr):
        """
        Function to dispatch an attribute to the correct processor.

        Parameters
        ----------
        name : str
            Name of the attribute
        attr : object
            Attribute object
        """

        cls = attr.__class__

        while True:
            cls_name = cls.__name__

            if self._call_processor(name, attr, cls_name) is True:
                break

            base_class = cls.__bases__[0]

            if base_class.__name__ == cls_name:
                break

            if base_class.__name__ == 'object':
                raise NotImplementedError(
                    f"Cannot find processor for {self.model.class_name}.{name} of {cls_name}")

            cls = base_class

    def _process_all_params_names(self):

        out = list()
        for instance in self.all_params.values():
            out += instance.get_names()
        return out

    def _call_processor(self, name, attr, class_name):

        if hasattr(self, f'_process_{class_name}'):

            getattr(self, f'_process_{class_name}')(name, attr)
            logger.debug("Processed <%s> as class %s" % (name, class_name))
            return True

        return False

    def get_md5(self):
        """
        Return the md5 hash of concatenated equation strings.
        """
        import hashlib
        md5 = hashlib.md5()

        for name in self.all_params.keys():
            md5.update(str(name).encode())

        # TODO
        # for name in self.config.as_dict().keys():
        #     md5.update(str(name).encode())

        for name, item in self.all_vars.items():
            md5.update(str(name).encode())

            if item.v_str is not None:
                md5.update(str(item.v_str).encode())
            if item.v_iter is not None:
                md5.update(str(item.v_iter).encode())
            if item.e_str is not None:
                md5.update(str(item.e_str).encode())
            if item.diag_eps is not None:
                md5.update(str(item.diag_eps).encode())

        for name, item in self.services_const.items():
            md5.update(str(name).encode())

            if item.v_str is not None:
                md5.update(str(item.v_str).encode())

            md5.update(str(int(item.sequential)).encode())

        for name, item in self.discrete.items():
            md5.update(str(name).encode())
            md5.update(str(','.join(item.export_flags)).encode())

        return md5.hexdigest()

    def _process_State(self, name, attr):
        self.all_vars[name] = attr
        self.vars_int[name] = attr
        self.states[name] = attr
        self.states_and_ext[name] = attr

    def _process_Algeb(self, name, attr):
        self.all_vars[name] = attr
        self.vars_int[name] = attr
        self.algebs[name] = attr
        self.algebs_and_ext[name] = attr

    def _process_ExtState(self, name, attr):
        self.all_vars[name] = attr
        self.states_ext[name] = attr
        self.states_and_ext[name] = attr

    def _process_ExtAlgeb(self, name, attr):
        self.all_vars[name] = attr
        self.algebs_ext[name] = attr
        self.algebs_and_ext[name] = attr

    def _process_Discrete(self, name, attr):
        self.discrete[name] = attr
        self.all_params[name] = attr

    def _process_DataParam(self, name, attr):
        self.params[name] = attr

    def _process_NumParam(self, name, attr):
        self.all_params[name] = attr
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
        self.all_params[name] = attr
        self.services_const[name] = attr

    def _process_VarService(self, name, attr):
        self.all_params[name] = attr
        self.services_const[name] = attr
        self.services_var[name] = attr

        if attr.sequential:
            self.services_var_seq[name] = attr
        else:
            self.services_var_nonseq[name] = attr

    def _process_PostInitService(self, name, attr):
        self.all_params[name] = attr
        self.services_const[name] = attr
        self.services_post[name] = attr

    def _process_SubService(self, name, attr):
        self.services_subs[name] = attr

    def _process_DeviceFinder(self, name, attr):
        self.services_fnd[name] = attr

    def _process_BackRef(self, name, attr):
        self.services_ref[name] = attr

    def _process_ExtService(self, name, attr):
        self.all_params[name] = attr
        self.services_ext[name] = attr

    def _process_NumReplace(self, name, attr):
        self.all_params[name] = attr
        self.services_ops[name] = attr

    def _process_NumReduce(self, name, attr):
        self.all_params[name] = attr
        self.services_ops[name] = attr

    def _process_NumSelect(self, name, attr):
        self.all_params[name] = attr
        self.services_ops[name] = attr

    def _process_FlagValue(self, name, attr):
        self.all_params[name] = attr
        self.services_ops[name] = attr

    def _process_RandomService(self, name, attr):
        self.all_params[name] = attr
        self.services_ops[name] = attr

    def _process_SwBlock(self, name, attr):
        self.blocks[name] = attr

    def _process_ParamCalc(self, name, attr):
        self.all_params[name] = attr
        self.services_ops[name] = attr

    def _process_Replace(self, name, attr):
        self.all_params[name] = attr
        self.services_ops[name] = attr

    def _process_ApplyFunc(self, name, attr):
        self.all_params[name] = attr
        self.services_ops[name] = attr

    def _process_InitChecker(self, name, attr):
        self.services_icheck[name] = attr

    def _process_Block(self, name, attr):
        self.blocks[name] = attr
        logger.debug("processing block %s", attr.__class__.__name__)

        # if attr.namespace == 'local':
        #     prepend = attr.name + '_'
        #     tex_append = attr.tex_name
        # else:
        #     prepend = ''
        #     tex_append = ''

        # for var_name, var_instance in attr.export().items():
        #     var_instance.name = f'{prepend}{var_name}'
        #     var_instance.tex_name = f'{var_instance.tex_name}_{{{tex_append}}}'
        #     self.__setattr__(var_instance.name, var_instance)

        #     logger.debug(" processing block", var_instance.name, var_instance.__class__.__name__)
        #     self.process_one_attr(var_instance.name, var_instance)

    def _process_BaseService(self, name, attr):
        self.all_params[name] = attr
        pass

    def _process_AliasAlgeb(self, name, attr):
        self.all_vars[name] = attr

    def _process_AliasState(self, name, attr):
        self.all_vars[name] = attr

    def _process_DummyValue(self, name, attr):
        self.dummy[name] = attr
