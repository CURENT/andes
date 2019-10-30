class GroupBase(object):
    """
    Base class for groups
    """

    def __init__(self):
        self.shared_parameters = []
        self.shared_variables = []

        self.models = {}  # model name, model instance
        self.elem2model = {}  # element idx, model name

    def register(self, name, model_instance):
        if name not in self.models:
            self.models[name] = model_instance
        else:
            raise KeyError(f"Duplicate model registration if {name}")

    def register_element(self, idx, model_name):
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
                # logger.warning(f"{self.name}: conflict idx {idx}. Data may be inconsistent.")
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
