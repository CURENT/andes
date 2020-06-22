from typing import Optional, Union, Callable, Type
from andes.core.param import BaseParam
from andes.utils.func import list_flatten
from andes.core.common import dummify
from andes.shared import np, ndarray
import logging
from collections import OrderedDict
from andes.utils.tab import Tab

logger = logging.getLogger(__name__)


class BaseService(object):
    """
    Base class for Service.

    Service is a v-provider type for holding internal and temporary values. Subclasses need to implement ``v``
    as a member attribute or using a property decorator.

    Parameters
    ----------
    name : str
        Instance name

    Attributes
    ----------
    owner : Model
        The hosting/owner model instance
    """

    def __init__(self, name: str = None, tex_name: str = None, info: str = None, vtype: Type = None):
        self.name = name
        self.tex_name = tex_name if tex_name else name
        self.info = info
        self.vtype = vtype if vtype is not None else np.float  # type for `v`. NOT IN USE.
        self.owner = None

    def get_names(self):
        """
        Return `name` in a list

        Returns
        -------
        list
            A list only containing the name of the service variable
        """
        return [self.name]

    @property
    def n(self):
        """
        Return the count of values in ``self.v``.

        Needs to be overloaded if ``v`` of subclasses is not a 1-dimensional array.

        Returns
        -------
        int
            The count of elements in this variable
        """
        if isinstance(self.v, (list, np.ndarray)):
            return len(self.v)
        else:
            return 1

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__


class ConstService(BaseService):
    """
    A type of Service that stays constant once initialized.

    ConstService are usually constants calculated from parameters. They are only evaluated once in the
    initialization phase before variables are initialized. Therefore, uninitialized variables must not be
    used in `v_str``.

    Parameters
    ----------
    name : str
        Name of the ConstService
    v_str : str
        An equation string to calculate the variable value.
    v_numeric : Callable, optional
        A callable which returns the value of the ConstService

    Attributes
    ----------
    v : array-like or a scalar
        ConstService value
    """

    def __init__(self,
                 v_str: Optional[str] = None,
                 v_numeric: Optional[Callable] = None,
                 vtype: Optional[type] = None,
                 name: Optional[str] = None, tex_name=None, info=None):
        super().__init__(name=name, vtype=vtype, tex_name=tex_name, info=info)
        self.v_str = v_str
        self.v_numeric = v_numeric
        self.v: Union[float, int, ndarray] = np.array([0.])

    def assign_memory(self, n):
        """Assign memory for ``self.v`` and set the array to zero."""
        self.v = np.zeros(n, dtype=self.vtype)


class VarService(ConstService):
    """
    Variable service that gets updated in each step/loop as variables change.

    This class is useful when one has non-differentiable algebraic equations,
    which make use of `abs()`, `re` and `im`.
    Instead of creating `Algeb`, one can put the equation in `VarService`,
    which will be updated before solving algebraic equations.

    Examples
    --------
    In ESST3A model, the voltage and current sensors (vd + jvq), (Id + jIq)
    estimate the sensed VE using equation

    .. math ::

        VE = | K_{PC}*(v_d + 1j v_q) + 1j (K_I + K_{PC}*X_L)*(I_d + 1j I_q)|

    One can use `VarService` to implement this equation ::

        self.VE = VarService(tex_name='V_E',
                             info='VE',
                             v_str='Abs(KPC*(vd + 1j*vq) + 1j*(KI + KPC*XL)*(Id + 1j*Iq))',
                             )

    Warnings
    --------
    `VarService` is not solved with other algebraic equations, meaning that
    there is one step "delay" between the algebraic variables and `VarService`.
    Use an algebraic variable whenever possible.
    """

    pass


class PostInitService(ConstService):
    """
    Constant service that gets stored once after init.

    This service is useful when one need to store initialization
    values stored in variables.

    Examples
    --------
    In ESST3A model, the `vf` variable is initialized followed by other
    variables. One can store the initial `vf` into `vf0` so that equation
    ``vf - vf0 = 0`` will hold. ::

        self.vref0 = PostInitService(info='Initial reference voltage input',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

    Since all `ConstService` are evaluated before equation evaluation,
    without using PostInitService, one will need to create lots
    of `ConstService` to store values in the initialization path
    towards `vf0`, in order to correctly initialize `vf`.
    """

    pass


class BackRef(BaseService):
    """
    A special type of reference collector.

    `BackRef` is used for collecting device indices of other models referencing the parent model of the
    `BackRef`. The `v``field will be a list of lists, each containing the `idx` of other models
    referencing each device of the parent model.

    BackRef can be passed as indexer for params and vars, or shape for `NumReduce` and
    `NumRepeat`. See examples for illustration.

    Examples
    --------
    A Bus device has an `IdxParam` of `area`, storing the `idx` of area to which the bus device belongs.
    In ``Bus.__init__()``, one has ::

        self.area = IdxParam(model='Area')

    Suppose `Bus` has the following data

        ====   ====  ====
        idx    area  Vn
        ----   ----  ----
        1      1     110
        2      2     220
        3      1     345
        4      1     500
        ====   ====  ====

    The Area model wants to collect the indices of Bus devices which points to the corresponding Area device.
    In ``Area.__init__``, one defines ::

        self.Bus = BackRef()

    where the member attribute name `Bus` needs to match exactly model name that `Area` wants to collect
    `idx` for.
    Similarly, one can define ``self.ACTopology = BackRef()`` to collect devices in the `ACTopology` group
    that references Area.

    The collection of `idx` happens in :py:func:`andes.system.System._collect_ref_param`.
    It has to be noted that the specific `Area` entry must exist to collect model idx-dx referencing it.
    For example, if `Area` has the following data ::

        idx
        1

    Then, only Bus 1, 3, and 4 will be collected into `self.Bus.v`, namely, ``self.Bus.v == [ [1, 3, 4] ]``.

    If `Area` has data ::

        idx
        1
        2

    Then, `self.Bus.v` will end up with ``[ [1, 3, 4], [2] ]``.

    See Also
    --------
    andes.core.service.NumReduce : A more complete example using BackRef to build the COI model

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.export = False
        self.v = list()


class ExtService(BaseService):
    """
    Service constants whose value is from an external model or group.

    Parameters
    ----------
    src : str
        Variable or parameter name in the source model or group
    model : str
        A model name or a group name
    indexer : IdxParam or BaseParam
        An "Indexer" instance whose ``v`` field contains the ``idx`` of devices in the model or group.

    Examples
    --------
    A synchronous generator needs to retrieve the ``p`` and ``q`` values from static generators
    for initialization. ``ExtService`` is used for this purpose.

    In a synchronous generator, one can define the following to retrieve ``StaticGen.p`` as ``p0``::

        class GENCLSModel(Model):
            def __init__(...):
                ...
                self.p0 = ExtService(src='p',
                                     model='StaticGen',
                                     indexer=self.gen,
                                     tex_name='P_0')

    """

    def __init__(self,
                 model: str,
                 src: str,
                 indexer: Union[BaseParam, BaseService],
                 attr: str = 'v',
                 allow_none: bool = False,
                 default=0,
                 name: str = None,
                 tex_name: str = None,
                 vtype=None,
                 info: str = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info, vtype=vtype)
        self.model = model
        self.src = src
        self.indexer = indexer
        self.attr = attr
        self.allow_none = allow_none
        self.default = default
        self.v = np.array([0.])

    def assign_memory(self, n):
        """Assign memory for ``self.v`` and set the array to zero."""
        self.v = np.zeros(n, dtype=self.vtype)

    def link_external(self, ext_model):
        """
        Method to be called by ``System`` for getting values from the external model or group.

        Parameters
        ----------
        ext_model
            An instance of a model or group provided by System
        """
        # set initial v values to zero
        self.v = np.zeros(self.n)
        if self.n == 0:
            return

        # the same `get` api for Group and Model
        self.v = ext_model.get(src=self.src, idx=self.indexer.v, attr=self.attr,
                               allow_none=self.allow_none,
                               default=self.default,
                               )


class DataSelect(BaseService):
    """
    Class for selecting values for optional DataParam or NumParam.

    This service is a v-provider that uses optional DataParam if available with a fallback.

    DataParam will be tested for `None`, and NumParam will be tested with `np.isnan()`.

    Notes
    -----
    An use case of DataSelect is remote bus. One can do ::

        self.buss = DataSelect(option=self.busr, fallback=self.bus)

    Then, pass ``self.buss`` instead of ``self.bus`` as indexer to retrieve voltages.

    Another use case is to allow an optional turbine rating. One can do ::

        self.Tn = NumParam(default=None)
        self.Sg = ExtParam(...)
        self.Sn = DataSelect(Tn, Sg)

    """

    def __init__(self,
                 optional,
                 fallback,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info, )
        self.optional = optional
        self.fallback = fallback
        self._v = None

    @property
    def v(self):
        if self._v is None:
            self._v = [v1 if v1 is not None and not np.isnan(v1)
                       else v2
                       for v1, v2 in zip(self.optional.v, self.fallback.v)]

        return self._v


class DeviceFinder(BaseService):
    """
    Service for finding indices of optionally linked devices.

    If not provided, `DeviceFinder` will add devices at the beginning of `System.setup`.

    Examples
    --------
    IEEEST stabilizer takes an optional `busf` (IdxParam) for specifying the connected BusFreq,
    which is needed for mode 6. To avoid reimplementing `BusFreq` within IEEEST, one can do

    .. code-block :: python

        self.busfreq = DeviceFinder(self.busf, link=self.buss, idx_name='bus')

    where `self.busf` is the optional input, `self.buss` is the bus indices that `busf` should measure,
    and `idx_name` is the name of a BusFreq parameter through which the measured bus indices are specified.
    For each `None` values in `self.busf`, a `BusFreq` is created to measure the corresponding bus in `self.buss`.

    That is, ``BusFreq.[idx_name].v = [link]``. `DeviceFinder` will find / create `BusFreq` devices so that
    the returned list of `BusFreq` indices are connected to `self.buss`, respectively.
    """

    def __init__(self, u, link, idx_name, name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)

        self.u = u
        self.model = u.model
        self.idx_name = idx_name

        if self.model is None:
            raise ValueError(f'{u.owner.class_name}.{u.name} must contain "model".')

        self.link = link

    def find_or_add(self, system):
        mdl = system.models[self.model]
        found_idx = mdl.find_idx((self.idx_name,), (self.link.v,),
                                 allow_none=True, default=None)

        action = False
        for ii, idx in enumerate(found_idx):
            if idx is None:
                action = True
                new_idx = system.add(self.model, {self.idx_name: self.link.v[ii]})
                self.u.v[ii] = new_idx

                logger.info(f"{self.owner.class_name} <{self.owner.idx.v[ii]}> "
                            f"added {self.model} <{new_idx}> "
                            f"on {self.idx_name} <{self.link.v[ii]}>")
            else:
                action = True
                self.u.v[ii] = idx

        if action:
            mdl.list2array()
            mdl.refresh_inputs()

    @property
    def v(self):
        return self.u.v


class OperationService(BaseService):
    """
    Base class for a type of Service which performs specific operations

    This class cannot be used by itself.

    See Also
    --------
    NumReduce : Service for Reducing linearly stored 2-D services into 1-D

    NumRepeat : Service for repeating 1-D NumParam/ v-array following a sub-pattern

    IdxRepeat : Service for repeating 1-D IdxParam/ v-list following a sub-pattern
    """

    def __init__(self,
                 name=None,
                 tex_name=None,
                 info=None,
                 ):
        self._v = None
        super().__init__(name=name, tex_name=tex_name, info=info, )
        self.v_str = None

    @property
    def v(self):
        """
        Return values stored in `self._v`. May be overloaded by subclasses.
        """
        return self._v

    @v.setter
    def v(self, value):
        self._v = value


class NumReduce(OperationService):
    """
    A helper Service type which reduces a linearly stored 2-D ExtParam into 1-D Service.

    NumReduce works with ExtParam whose `v` field is a list of lists. A reduce function
    which takes an array-like and returns a scalar need to be supplied. NumReduce calls the reduce
    function on each of the lists and return all the scalars in an array.

    Parameters
    ----------
    u : ExtParam
        Input ExtParam whose ``v`` contains linearly stored 2-dimensional values
    ref : BackRef
        The BackRef whose 2-dimensional shapes are used for indexing
    fun : Callable
        The callable for converting a 1-D array-like to a scalar

    Examples
    --------
    Suppose one wants to calculate the mean value of the ``Vn`` in one Area. In the ``Area`` class, one defines ::

        class AreaModel(...):
            def __init__(...):
                ...
                # backward reference from `Bus`
                self.Bus = BackRef()

                # collect the Vn in an 1-D array
                self.Vn = ExtParam(model='Bus',
                    src='Vn',
                    indexer=self.Bus)

                self.Vn_mean = NumReduce(u=self.Vn,
                    fun=np.mean,
                    ref=self.Bus)

    Suppose we define two areas, 1 and 2, the Bus data looks like

        ===   =====  ====
        idx    area  Vn
        ---   -----  ----
        1      1     110
        2      2     220
        3      1     345
        4      1     500
        ===   =====  ====

    Then, `self.Bus.v` is a list of two lists ``[ [1, 3, 4], [2] ]``.
    `self.Vn.v` will be retrieved and linearly stored as ``[110, 345, 500, 220]``.
    Based on the shape from `self.Bus`, :py:func:`numpy.mean`
    will be called on ``[110, 345, 500]`` and ``[220]`` respectively.
    Thus, `self.Vn_mean.v` will become ``[318.33, 220]``.

    """

    def __init__(self,
                 u,
                 ref: BackRef,
                 fun: Callable,
                 name=None,
                 tex_name=None,
                 info=None,
                 cache=True,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.u = u
        self.ref = ref
        self.fun = fun
        self.cache = cache

    @property
    def v(self):
        """
        Return the reduced values from the reduction function in an array

        Returns
        -------
        The array ``self._v`` storing the reduced values
        """
        if self._v is not None and self.cache is True:
            return self._v

        if self._v is None:
            self._v = np.zeros(len(self.ref.v))

        idx = 0
        for i, v in enumerate(self.ref.v):
            self._v[i] = self.fun(self.u.v[idx:idx + len(v)])
            idx += len(v)
        return self._v


class NumRepeat(OperationService):
    r"""
    A helper Service type which repeats a v-provider's value based on the shape from a BackRef

    Examples
    --------
    NumRepeat was originally designed for computing the inertia-weighted average rotor speed (center of
    inertia speed). COI speed is computed with

    .. math ::
        \omega_{COI} = \frac{ \sum{M_i * \omega_i} } {\sum{M_i}}

    The numerator can be calculated with a mix of BackRef, ExtParam and ExtState. The denominator needs to be
    calculated with NumReduce and Service Repeat. That is, use NumReduce to calculate the sum,
    and use NumRepeat to repeat the summed value for each device.

    In the COI class, one would have

    .. code-block :: python

        class COIModel(...):
            def __init__(...):
                ...
                self.SynGen = BackRef()
                self.SynGenIdx = RefFlatten(ref=self.SynGen)
                self.M = ExtParam(model='SynGen',
                                  src='M',
                                  indexer=self.SynGenIdx)

                self.wgen = ExtState(model='SynGen',
                                     src='omega',
                                     indexer=self.SynGenIdx)

                self.Mt = NumReduce(u=self.M,
                                         fun=np.sum,
                                         ref=self.SynGen)

                self.Mtr = NumRepeat(u=self.Mt,
                                       ref=self.SynGen)

                self.pidx = IdxRepeat(u=self.idx,ref=self.SynGen)

    Finally, one would define the center of inertia speed as

    .. code-block :: python

        self.wcoi = Algeb(v_str='1', e_str='-wcoi')

        self.wcoi_sub = ExtAlgeb(model='COI',
                                 src='wcoi',
                                 e_str='M * wgen / Mtr',
                                 v_str='M / Mtr',
                                 indexer=self.pidx,
                                 )

    It is very worth noting that the implementation uses a trick to separate the average weighted sum into `n`
    sub-equations, each calculating the :math:`(M_i * \omega_i) / (\sum{M_i})`. Since all the variables are
    preserved in the sub-equation, the derivatives can be calculated correctly.

    """

    def __init__(self,
                 u,
                 ref,
                 **kwargs):
        super().__init__(**kwargs)
        self.u = u
        self.ref = ref

    @property
    def v(self):
        """
        Return the values of the repeated values in a sequential 1-D array

        Returns
        -------
        The array, ``self._v`` storing the repeated values
        """
        if self._v is None:
            self._v = np.zeros(len(list_flatten(self.ref.v)))

            idx = 0
            for i, v in enumerate(self.ref.v):
                self._v[idx:idx + len(v)] = self.u.v[i]
                idx += len(v)
            return self._v
        else:
            return self._v


class IdxRepeat(OperationService):
    """
    Helper class to repeat IdxParam.

    This class has the same functionality as :py:class:`andes.core.service.NumRepeat`
    but only operates on IdxParam, DataParam or NumParam.
    """

    def __init__(self,
                 u,
                 ref,
                 **kwargs):
        super().__init__(**kwargs)
        self.u = u
        self.ref = ref

    @property
    def v(self):
        if self._v is None:
            self._v = [''] * len(list_flatten(self.ref.v))
            idx = 0
            for i, v in enumerate(self.ref.v):
                for jj in range(idx, idx + len(v)):
                    self._v[jj] = self.u.v[i]
                idx += len(v)
            return self._v
        else:
            return self._v


class RefFlatten(OperationService):
    """
    A service type for flattening :py:class:`andes.core.service.BackRef` into a 1-D list.

    Examples
    --------
    This class is used when one wants to pass `BackRef` values as indexer.

    :py:class:`andes.models.coi.COI` collects referencing
    :py:class:`andes.models.group.SynGen` with

    .. code-block :: python

        self.SynGen = BackRef(info='SynGen idx lists', export=False)

    After collecting BackRefs, `self.SynGen.v` will become a two-level list of indices,
    where the first level correspond to each COI and the second level correspond to generators
    of the COI.

    Convert `self.SynGen` into 1-d as `self.SynGenIdx`, which can be passed as indexer for
    retrieving other parameters and variables

    .. code-block :: python

        self.SynGenIdx = RefFlatten(ref=self.SynGen)

        self.M = ExtParam(model='SynGen', src='M',
                          indexer=self.SynGenIdx, export=False,
                          )
    """

    def __init__(self, ref, **kwargs):
        super().__init__(**kwargs)
        self.ref = ref

    @property
    def v(self):
        return list_flatten(self.ref.v)


class NumSelect(OperationService):
    """
    Class for selecting values for optional NumParam.

    Notes
    -----
    One use case is to allow an optional turbine rating. One can do ::

        self.Tn = NumParam(default=None)
        self.Sg = ExtParam(...)
        self.Sn = DataSelect(Tn, Sg)

    """

    def __init__(self,
                 optional,
                 fallback,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.optional = optional
        self.fallback = fallback
        self._v = None

    @property
    def v(self):
        if self._v is None:
            self._v = [v1 if not np.isnan(v1)
                       else v2
                       for v1, v2 in zip(self.optional.v, self.fallback.v)]

            self._v = np.array(self._v)

        return self._v


class InitChecker(OperationService):
    """
    Class for checking init values against known typical values.

    Instances will be stored in `Model.services_post` and
    `Model.services_icheck`, which will be checked in
    `Model.post_init_check()` after initialization.

    Parameters
    ----------
    u
        v-provider to be checked
    lower : float, BaseParam, BaseVar, BaseService
        lower bound
    upper : float, BaseParam, BaseVar, BaseService
        upper bound
    equal : float, BaseParam, BaseVar, BaseService
        values that the value from `v_str` should equal
    not_equal : float, BaseParam, BaseVar, BaseService
        values that should not equal
    enable : bool
        True to enable checking

    Examples
    --------
    Let's say generator excitation voltages are known to be in
    the range of 1.6 - 3.0 per unit. One can add the following
    instance to `GENBase` ::

        self._vfc = InitChecker(u=self.vf,
                                info='vf range',
                                lower=1.8,
                                upper=3.0,
                                )

    `lower` and `upper` can also take v-providers instead of
    float values.

    One can also pass float values from Config to make it
    adjustable as in our implementation of ``GENBase._vfc``.
    """

    def __init__(self, u, lower=None, upper=None, equal=None, not_equal=None,
                 enable=True, error_out=False, **kwargs):
        super().__init__(**kwargs)
        self.u = u
        self.lower = dummify(lower) if lower is not None else None
        self.upper = dummify(upper) if upper is not None else None
        self.equal = dummify(equal) if equal is not None else None
        self.not_equal = dummify(not_equal) if not_equal is not None else None
        self.enable = enable
        self.error_out = error_out

    def check(self):
        """
        Check the bounds and equality conditions.
        """
        if not self.enable:
            return

        def _not_all_close(a, b):
            return np.logical_not(np.isclose(a, b))

        if self._v is None:
            self._v = np.zeros_like(self.u.v)

        checks = [(self.lower, np.less_equal, "violation of the lower limit", "limit"),
                  (self.upper, np.greater_equal, "violation of the upper limit", "limit"),
                  (self.equal, _not_all_close, 'should be equal', "expected"),
                  (self.not_equal, np.equal, 'should not be equal', "not expected")
                  ]

        for check in checks:
            limit = check[0]
            func = check[1]
            text = check[2]
            text2 = check[3]
            if limit is None:
                continue

            self.v[:] = np.logical_or(self.v, func(self.u.v, limit.v))

            pos = np.argwhere(func(self.u.v, limit.v)).ravel()

            if len(pos) == 0:
                continue
            idx = [self.owner.idx.v[i] for i in pos]
            lim_v = limit.v * np.ones(self.n)

            title = f'{self.owner.class_name} {self.info} {text}.'

            err_dict = OrderedDict([('idx', idx),
                                    ('values', self.u.v[pos]),
                                    (f'{text2}', lim_v[pos]),
                                    ])
            data = list(map(list, zip(*err_dict.values())))

            tab = Tab(title=title, data=data, header=list(err_dict.keys()))
            if self.error_out:
                logger.error(tab.draw())
            else:
                logger.warning(tab.draw())

        self.v[:] = np.logical_not(self.v)


class FlagNotNone(BaseService):
    """
    Class for flagging non-None indices as 1 and None indices as 0 in a numpy array.

    Warnings
    --------
    FlagNotNone can only be applied to BaseParam with `cache=True`.
    Applying to Service will fail until cache is False (at a performance cost).
    """

    def __init__(self: BaseParam, indexer, to_flag=None, name=None, tex_name=None, info=None, cache=True):
        BaseService.__init__(self, name=name, tex_name=tex_name, info=info)
        self.cache = cache
        self.to_flag = to_flag
        self.indexer = indexer
        self._v = None

    @property
    def v(self):
        new = False
        if self._v is None:
            self._v = np.zeros_like(self.indexer.v, dtype=float)
            new = True

        if not self.cache or new:
            self._v[:] = np.array([0 if i == self.to_flag else 1
                                   for i in self.indexer.v])
        return self._v


class Replace(BaseService):
    """
    Replace parameters with new values if the function returns True
    """

    def __init__(self, old_val, flt, new_val, name=None, tex_name=None, info=None, cache=True):
        BaseService.__init__(self, name=name, tex_name=tex_name, info=info)
        self.cache = cache
        self.filter = flt  # function
        self.old_val = old_val
        self.new_val = dummify(new_val)
        self._v = None

    @property
    def v(self):
        new = False
        if self._v is None or not self.cache:
            self._v = np.zeros_like(self.old_val.v, dtype=float)
            new = True

        if not self.cache or new:
            new_v = self.new_val.v * np.ones_like(self.old_val.v)
            flt = self.filter(self.old_val.v)
            self._v[:] = new_v * flt + self.old_val.v * (1 - flt)

        return self._v


class ParamCalc(BaseService):
    """
    Parameter calculation service.

    Useful to create parameters calculated instantly from existing ones.
    """

    def __init__(self, param1, param2, func, name=None, tex_name=None, info=None,
                 cache=True):
        BaseService.__init__(self, name=name, tex_name=tex_name, info=info)
        self.param1 = param1
        self.param2 = param2
        self.func = func
        self.cache = cache
        self._v = None

    @property
    def v(self):
        new = False
        if self._v is None:
            new = True
            self._v = np.zeros_like(self.param1.v, dtype=float)

        if not self.cache or new:
            self._v[:] = self.func(self.param1.v,
                                   self.param2.v)

        return self._v


class RandomService(BaseService):
    """
    A service type for generating random numbers.

    Parameters
    ----------
    name : str
        Name
    func : Callable
        A callable for generating the random variable.

    Warnings
    --------
    The value will be randomized every time it is accessed. Do not use it if the value needs to be stable for
    each simulation step.
    """

    def __init__(self, func=np.random.rand, **kwargs):
        super(RandomService, self).__init__(**kwargs)
        self.func = func

    @property
    def v(self):
        """
        This class has `v` wrapped by a property decorator.

        Returns
        -------
        array-like
            Randomly generated service variables
        """
        return np.random.rand(self.n)
