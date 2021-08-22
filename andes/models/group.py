import logging
import numpy as np
from collections import OrderedDict
from andes.core.service import BackRef
from typing import Sized

from andes.utils.func import list_flatten

logger = logging.getLogger(__name__)


class GroupBase:
    """
    Base class for groups.
    """

    def __init__(self):
        self.common_params = ['u', 'name']
        self.common_vars = []

        self.models = OrderedDict()        # model name: model instance
        self._idx2model = OrderedDict()    # element idx: model instance
        self.uid = {}                      # idx - group internal 0-indexed uid
        self.services_ref = OrderedDict()  # BackRef

    def __setattr__(self, key, value):
        if hasattr(value, 'owner'):
            if value.owner is None:
                value.owner = self
        if hasattr(value, 'name'):
            if value.name is None:
                value.name = key

        if isinstance(value, BackRef):
            self.services_ref[key] = value

        super().__setattr__(key, value)

    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def n(self):
        """
        Total number of devices.
        """
        return len(self._idx2model)

    def add_model(self, name: str, instance):
        """
        Add a Model instance to group.

        Parameters
        ----------
        name : str
            Model name
        instance : Model
            Model instance

        Returns
        -------
        None
        """
        if name not in self.models:
            self.models[name] = instance
        else:
            raise KeyError(f"{self.class_name}: Duplicate model registration of {name}")

    def add(self, idx, model):
        """
        Register an idx from model_name to the group

        Parameters
        ----------
        idx: Union[str, float, int]
            Register an element to a model

        model: Model
            instance of the model

        Returns
        -------

        """
        if idx in self._idx2model:
            raise KeyError(f'Group <{self.class_name}> already contains <{repr(idx)}> from '
                           f'<{self._idx2model[idx].class_name}>')
        self.uid[idx] = self.n
        self._idx2model[idx] = model

    def idx2model(self, idx, allow_none=False):
        """
        Find model name for the given idx.

        Parameters
        ----------
        idx : float, int, str, array-like
            idx or idx-es of devices.
        allow_none : bool
           If True, return `None` at the positions where idx is not found.

        Returns
        -------
        If `idx` is a list, return a list of model instances.
        If `idx` is a single element, return a model instance.
        """

        ret = []
        single = False

        if not isinstance(idx, (list, tuple, np.ndarray)):
            single = True
            idx = (idx,)
        elif len(idx) > 0 and isinstance(idx[0], (list, tuple, np.ndarray)):
            idx = list_flatten(idx)

        for i in idx:
            try:
                if i is None and allow_none:
                    ret.append(None)
                else:
                    ret.append(self._idx2model[i])
            except KeyError:
                raise KeyError(f'Group <{self.class_name}> does not contain device with idx={i}')

        if single:
            ret = ret[0]
        return ret

    def idx2uid(self, idx):
        """
        Convert idx to the 0-indexed unique index.

        Parameters
        ----------
        idx : array-like, numbers, or str
            idx of devices

        Returns
        -------
        list
            A list containing the unique indices of the devices
        """
        if idx is None:
            logger.debug("idx2uid returned None for idx None")
            return None
        if isinstance(idx, (float, int, str, np.integer, np.floating)):
            return self.uid[idx]
        elif isinstance(idx, Sized):
            if len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
                idx = list_flatten(idx)
            return [self.uid[i] if i is not None else None
                    for i in idx]
        else:
            raise NotImplementedError(f'Unknown idx type {type(idx)}')

    def get(self, src: str, idx, attr: str = 'v', allow_none=False, default=0.0):
        """
        Based on the indexer, get the `attr` field of the `src` parameter or variable.

        Parameters
        ----------
        src : str
            param or var name
        idx : array-like
            device idx
        attr
            The attribute of the param or var to retrieve
        allow_none : bool
            True to allow None values in the indexer
        default : float
            If `allow_none` is true, the default value to use for None indexer.

        Returns
        -------
        The requested param or variable attribute. If `idx` is a list, return a list of values.
        If `idx` is a single element, return a single value.
        """
        self._check_src(src)
        self._check_idx(idx)

        single = False
        if not isinstance(idx, (list, np.ndarray)):
            idx = [idx]
            single = True

        n = len(idx)
        if n == 0:
            return np.zeros(0)

        ret = [''] * n
        _type_set = False

        models = self.idx2model(idx, allow_none=allow_none)

        for i, idx in enumerate(idx):
            if models[i] is not None:
                uid = models[i].idx2uid(idx)
                instance = models[i].__dict__[src]
                val = instance.__dict__[attr][uid]
            else:
                val = default

            # deduce the type for ret
            if not _type_set:
                if isinstance(val, str):
                    ret = [''] * n
                else:
                    ret = np.zeros(n)
                _type_set = True

            ret[i] = val

        if single:
            ret = ret[0]

        return ret

    def set(self, src: str, idx, attr, value):
        """
        Set the value of an attribute of a group property.
        Performs ``self.<src>.<attr>[idx] = value``.

        The user needs to ensure that the property is shared by all models
        in this group.

        Parameters
        ----------
        src : str
            Name of property.
        idx : str, int, float, array-like
            Indices of devices.
        attr : str, optional, default='v'
            The internal attribute of the property to get.
            ``v`` for values, ``a`` for address, and ``e`` for equation value.
        value : array-like
            New values to be set

        Returns
        -------
        bool
            True when successful.
        """
        self._check_src(src)
        self._check_idx(idx)

        if not isinstance(idx, (list, np.ndarray)):
            idx = [idx]

        if isinstance(value, (float, str, int)):
            value = [value] * len(idx)

        models = self.idx2model(idx)

        for i, idx in enumerate(idx):
            model = models[i]
            uid = model.idx2uid(idx)
            instance = model.__dict__[src]
            instance.__dict__[attr][uid] = value[i]

        return True

    def find_idx(self, keys, values, allow_none=False, default=None):
        """
        Find indices of devices that satisfy the given `key=value` condition.

        This method iterates over all models in this group.
        """
        indices_found = []
        # `indices_found` contains found indices returned from all models of this group
        for model in self.models.values():
            indices_found.append(model.find_idx(keys, values, allow_none=True, default=default))

        out = []
        for idx, idx_found in enumerate(zip(*indices_found)):
            if not allow_none:
                if idx_found.count(None) == len(idx_found):
                    missing_values = [item[idx] for item in values]
                    raise IndexError(f'{list(keys)} = {missing_values} not found in {self.class_name}')

            real_idx = None
            for item in idx_found:
                if item is not None:
                    real_idx = item
                    break
            out.append(real_idx)
        return out

    def _check_src(self, src: str):
        if src not in self.common_vars + self.common_params:
            # raise AttributeError(f'Group <{self.class_name}> does not share property <{src}>.')
            logger.debug(f'Group <{self.class_name}> does not share property <{src}>.')
            pass

    def _check_idx(self, idx):
        if idx is None:
            raise IndexError(f'{self.class_name}: idx cannot be None')

    def get_next_idx(self, idx=None, model_name=None):
        """
        Get a no-conflict idx for a new device.
        Use the provided ``idx`` if no conflict.
        Generate a new one otherwise.

        Parameters
        ----------
        idx : str or None
            Proposed idx. If None, assign a new one.
        model_name : str or None
            Model name. If not, prepend the group name.

        Returns
        -------
        str
            New device name.

        """
        if model_name is None:
            model_name = self.class_name

        need_new = False

        if idx is not None:
            if idx not in self._idx2model:
                # name is good
                pass
            else:
                logger.warning(f"Group {self.class_name}: idx={idx} is used by {self.idx2model(idx).class_name}. "
                               f"Data may be inconsistent.")
                need_new = True
        else:
            need_new = True

        if need_new is True:
            count = self.n
            while True:
                # IMPORTANT: automatically assigned index is 1-indexed. Namely, `GENCLS_1` is the first generator.
                # This is because when we say, for example, `GENCLS_10`, people usually assume it starts at 1.
                idx = model_name + '_' + str(count + 1)
                if idx not in self._idx2model:
                    break
                else:
                    count += 1

        return idx

    def doc(self, export='plain'):
        """
        Return the documentation of the group in a string.
        """
        out = ''
        if export == 'rest':
            out += f'.. _{self.class_name}:\n\n'
            group_header = '================================================================================\n'
        else:
            group_header = ''

        if export == 'rest':
            out += group_header + f'{self.class_name}\n' + group_header
        else:
            out += group_header + f'Group <{self.class_name}>\n' + group_header

        if self.__doc__:
            out += str(self.__doc__) + '\n\n'

        if len(self.common_params):
            out += 'Common Parameters: ' + ', '.join(self.common_params)
            out += '\n\n'
        if len(self.common_vars):
            out += 'Common Variables: ' + ', '.join(self.common_vars)
            out += '\n\n'
        if len(self.models):
            out += 'Available models:\n'
            model_name_list = list(self.models.keys())

            if export == 'rest':
                def add_reference(name_list):
                    return [f'{item}_' for item in name_list]

                model_name_list = add_reference(model_name_list)

            out += ',\n'.join(model_name_list) + '\n'

        return out

    def doc_all(self, export='plain'):
        """
        Return documentation of the group and its models.

        Parameters
        ----------
        export : 'plain' or 'rest'
            Export format, plain-text or RestructuredText

        Returns
        -------
        str

        """
        out = self.doc(export=export)
        out += '\n'
        for instance in self.models.values():
            out += instance.doc(export=export)
            out += '\n'
        return out


class Undefined(GroupBase):
    """
    The undefined group. Holds models with no ``group``.
    """
    pass


class ACTopology(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_vars.extend(('a', 'v'))


class DCTopology(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_vars.extend(('v',))


class Collection(GroupBase):
    """Collection of topology models"""
    pass


class Calculation(GroupBase):
    """Group of classes that calculates based on other models."""
    pass


class StaticGen(GroupBase):
    """
    Static generator group for power flow calculation
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('Sn', 'Vn', 'p0', 'q0', 'ra', 'xs', 'subidx'))
        self.common_vars.extend(('p', 'q', 'a', 'v'))

        self.SynGen = BackRef()


class ACLine(GroupBase):
    def __init__(self):
        super(ACLine, self).__init__()
        self.common_params.extend(('bus1', 'bus2', 'r', 'x'))
        self.common_vars.extend(('v1', 'v2', 'a1', 'a2'))


class ACShort(GroupBase):
    def __init__(self):
        super(ACShort, self).__init__()
        self.common_params.extend(('bus1', 'bus2'))
        self.common_vars.extend(('v1', 'v2', 'a1', 'a2'))


class StaticLoad(GroupBase):
    """
    Static load group.
    """
    pass


class StaticShunt(GroupBase):
    """
    Static shunt compensator group.
    """
    pass


class DynLoad(GroupBase):
    """
    Dynamic load group.
    """
    pass


class SynGen(GroupBase):
    """
    Synchronous generator group.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('Sn', 'Vn', 'fn', 'bus', 'M', 'D'))
        self.common_vars.extend(('omega', 'delta', 'tm', 'te', 'vf', 'XadIfd', 'vd', 'vq', 'Id', 'Iq',
                                 'a', 'v'))

        self.TurbineGov = BackRef()
        self.Exciter = BackRef()


class RenGen(GroupBase):
    """
    Renewable generator (converter) group.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'gen', 'Sn'))
        self.common_vars.extend(('Pe', 'Qe'))


class RenExciter(GroupBase):
    """
    Renewable electrical control (exciter) group.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('reg',))
        self.common_vars.extend(('Pref', 'Qref', 'wg', 'Pord'))


class RenPlant(GroupBase):
    """
    Renewable plant control group.
    """

    def __init__(self):
        super().__init__()


class RenGovernor(GroupBase):
    """
    Renewable turbine governor group.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('ree', 'w0', 'Sn', 'Pe0'))
        self.common_vars.extend(('Pm', 'wr0', 'wt', 'wg', 's3_y'))


class RenAerodynamics(GroupBase):
    """
    Renewable aerodynamics group.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('rego',))
        self.common_vars.extend(('theta',))


class RenPitch(GroupBase):
    """
    Renewable generator pitch controller group.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('rea',))


class RenTorque(GroupBase):
    """
    Renewable torque (Pref) controller.
    """

    def __init__(self):
        super().__init__()


class DG(GroupBase):
    """
    Distributed generation (small-scale).
    """

    def __init__(self):
        super().__init__()


class TurbineGov(GroupBase):
    """
    Turbine governor group for synchronous generator.
    """

    def __init__(self):
        super().__init__()
        self.common_vars.extend(('pout',))


class Exciter(GroupBase):
    """
    Exciter group for synchronous generators.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('syn',))
        self.common_vars.extend(('vout', 'vi',))

        self.PSS = BackRef()


class PSS(GroupBase):
    """Power system stabilizer group."""

    def __init__(self):
        super().__init__()
        self.common_vars.extend(('vsout',))


class Experimental(GroupBase):
    """Experimental group"""
    pass


class DCLink(GroupBase):
    """Basic DC links"""
    pass


class StaticACDC(GroupBase):
    """AC DC device for power flow"""
    pass


class TimedEvent(GroupBase):
    """Timed event group"""
    pass


class FreqMeasurement(GroupBase):
    """Frequency measurements."""

    def __init__(self):
        super().__init__()
        self.common_vars.extend(('f',))


class PhasorMeasurement(GroupBase):
    """Phasor measurements"""

    def __init__(self):
        super().__init__()
        self.common_vars.extend(('am', 'vm'))


class Motor(GroupBase):
    """Induction Motor group
    """

    def __init__(self):
        super().__init__()


class Information(GroupBase):
    """
    Group for information container models.
    """

    def __init__(self):
        GroupBase.__init__(self)
        self.common_params = []
