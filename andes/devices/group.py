import logging

logger = logging.getLogger(__name__)


class GroupBase(object):
    """
    Base class for groups
    """

    def __init__(self):
        self.common_params = []
        self.common_vars = []

        self.models = {}  # model name, model instance
        self.elem2model = {}  # element idx, model name

    def add_model(self, name, instance):
        if name not in self.models:
            self.models[name] = instance
        else:
            raise KeyError(f"Duplicate model registration if {name}")

    def add(self, idx, model_name):
        """
        Register an idx from model_name to the group

        Parameters
        ----------
        idx: Union[str, float]
            Register an element to a model

        model_name: str
            Name of the model

        Returns
        -------

        """
        self.elem2model['idx'] = model_name

    def get_next_idx(self, idx=None):
        """
        Return the auto-generated next idx

        Parameters
        ----------
        idx

        Returns
        -------

        """
        need_new = False

        if idx is not None:
            if idx not in self.elem2model:
                # name is good
                pass
            else:
                logger.debug(f"{self.__class__.__name__}: conflict idx {idx}. Data may be inconsistent.")
                need_new = True

        if idx is None:
            need_new = True

        if need_new is True:
            count = len(self.elem2model)
            while True:
                idx = self.name + str(count)
                if idx not in self.elem2model:
                    break
                else:
                    count += 1

        return idx


class Undefined(GroupBase):
    pass


class AcTopology(GroupBase):
    pass


class StaticGen(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params = ('p0', 'q0')
        self.common_vars = ('p', 'q')


class AcLine(GroupBase):
    pass


class StaticLoad(GroupBase):
    pass


class StaticShunt(GroupBase):
    pass


class SynGen(GroupBase):
    pass
