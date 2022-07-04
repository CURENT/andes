import logging
import inspect
from collections import OrderedDict

import numpy as np

from andes.core.service import BackRef
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
        idx, single = self._1d_vectorize(idx)

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
        vec_idx, single = self._1d_vectorize(idx)

        out = [self.uid[i] if i is not None else None for i in vec_idx]
        if single:
            out = out[0]

        return out

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
        idx, single = self._1d_vectorize(idx)

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

        idx, _ = self._1d_vectorize(idx)
        models = self.idx2model(idx)

        if isinstance(value, (str, int, float, np.integer, np.floating)):
            value = [value] * len(idx)

        for mdl, ii, val in zip(models, idx, value):
            uid = mdl.idx2uid(ii)
            mdl.__dict__[src].__dict__[attr][uid] = val

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

            real_idx = default
            for item in idx_found:
                if item is not None:
                    real_idx = item
                    break
            out.append(real_idx)
        return out

    def _check_src(self, src: str):
        """
        Helper function for checking if ``src`` is a shared field.

        The requirement is not strictly enforced and is only for debugging purposed.
        """
        if src not in self.common_vars + self.common_params:
            logger.debug(f'Group <{self.class_name}> does not share property <{src}>.')

    def _check_idx(self, idx):
        """
        Helper function for checking if ``idx`` is None.

        Raises IndexError if idx is None.
        """

        if idx is None:
            raise IndexError(f'{self.class_name}: idx cannot be None')

    def _1d_vectorize(self, idx):
        """
        Helper function to convert a single element, list, or nested lists
        into a list.

        If the input is a nested list, flatten it into a 1-dimensional
        list.

        Returns
        -------
        idx : list
            List of indices.
        single : bool
            True if the input is a single element.
        """
        single = False
        list_alike = (list, tuple, np.ndarray)

        if not isinstance(idx, list_alike):
            idx = [idx]
            single = True
        elif len(idx) > 0 and isinstance(idx[0], list_alike):
            idx = list_flatten(idx)

        return idx, single

    def get_field(self, src: str, idx, field: str):
        """
        Helper function for retrieving an attribute of a member variable shared
        by models in this group.

        Returns
        -------
        list
            A list with the length equal to ``len(idx)``.
        """

        self._check_src(src)
        self._check_idx(idx)

        idx, _ = self._1d_vectorize(idx)
        models = self.idx2model(idx, allow_none=True)

        ret = [None] * len(models)
        for ii, model in enumerate(models):
            if model is not None:
                ret[ii] = getattr(model.__dict__[src], field)

        return ret

    def set_backref(self, name, from_idx, to_idx):
        """
        Set idxes to ``BackRef``, and set them to models.
        """

        uid = self.idx2uid(to_idx)
        self.services_ref[name].v[uid].append(from_idx)

        model = self.idx2model(to_idx)
        model.set_backref(name, from_idx, to_idx)

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
                logger.warning("Group <%s>: idx=%s is used by %s. Data may be inconsistent.",
                               self.class_name, idx, self.idx2model(idx).class_name)
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
            group_header = '=' * 80 + '\n'
        else:
            group_header = ''

        if export == 'rest':
            out += group_header + f'{self.class_name}\n' + group_header
        else:
            out += group_header + f'Group <{self.class_name}>\n' + group_header

        if self.__doc__ is not None:
            out += inspect.cleandoc(self.__doc__) + '\n\n'

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
    Static generator group.

    Static generators will be replaced by dynamic generators, either synchronous
    generators or inverter-based resources upon the initialization for dynamics.
    It is implemented by setting the connectivity status ``u`` of the replaced
    StaticGen to 0.

    See the notes in :ref:`SynGen` for replacing one StaticGen with multiple
    dynamic ones.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('Sn', 'Vn', 'p0', 'q0', 'ra', 'xs', 'subidx'))
        self.common_vars.extend(('q', 'a', 'v'))

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

    SynGen replaces StaticGen upon the initialization of dynamic studies. SynGen
    and inverter-based resources contain parameters ``gammap`` and ``gammaq``
    for splitting the initial power of a StaticGen into multiple dynamic ones.

    ``gammap``, for example, is the active power ratio of the dynamic generator
    to the static one. If a StaticGen is supposed to be replaced by one SynGen,
    the ``gammap`` and ``gammaq`` should both be ``1``.

    It is critical to ensure that ``gammap`` and ``gammaq``, respectively, of
    all dynamic power sources sum up to 1.0. Otherwise, the initial power
    injections imposed by dynamic sources will differ from the static ones. The
    initialization will then fail with mismatches power injection equations
    corresponding to bus ``a`` and ``v``.

    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('Sn', 'Vn', 'fn', 'bus', 'M', 'D', 'subidx'))
        self.common_vars.extend(('omega', 'delta', ))
        self.idx_island = []
        self.uid_island = []
        self.delta_addr = []

        self.TurbineGov = BackRef()
        self.Exciter = BackRef()

    def store_idx_island(self, bus_idx):
        """
        Get ``idx`` of generators in the given islanded. Also store the
        addresses of the ``delta`` variable in the largest island.

        This function can only be called after initializing dynamic devices.

        Parameters
        ----------
        bus_idx : list
            A list of bus idx in the largest island
        """

        idx_gen = list(self.uid.keys())
        bus_gen = self.get(src='bus', idx=idx_gen, attr='v')

        def intersect(lst1, lst2):
            return list(set(lst1) & set(lst2))

        bus_gen_island = intersect(bus_idx, bus_gen)
        self.idx_island = self.find_idx(keys='bus',
                                        values=bus_gen_island)
        self.uid_island = self.idx2uid(self.idx_island)

        self.delta_addr = self.get('delta', self.idx_island, 'a').astype(int)


class RenGen(GroupBase):
    """
    Renewable generator (converter) group.

    See :ref:`SynGen` for the notes on replacing StaticGen and setting the power
    ratio parameters.
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

    See :ref:`SynGen` for the notes on replacing StaticGen and setting the power
    ratio parameters.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'fn'))


class DGProtection(GroupBase):
    """
    Protection model for DG.
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

        self.VoltComp = BackRef()
        self.PSS = BackRef()


class VoltComp(GroupBase):
    """
    Voltage compensator group for synchronous generators.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('rc', 'xc',))
        self.common_vars.extend(('vcomp',))


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


class PLL(GroupBase):
    """Phase-locked loop models."""

    def __init__(self):
        super().__init__()
        self.common_vars.extend(('am',))


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


class OutputSelect(GroupBase):
    """
    Group for selecting outputs.
    """

    def __init__(self):
        super().__init__()
        self.common_params = []


class Interface(GroupBase):
    """
    Group for interface models.
    """

    def __init__(self):
        super().__init__()
        self.common_params = []


class DataSeries(GroupBase):
    """
    Group for TimeSeries models.
    """

    def __init__(self):
        super().__init__()
        self.common_params = []
