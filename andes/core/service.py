from typing import Optional, Union, Callable
from andes.core.param import RefParam, BaseParam
from andes.utils.func import list_flatten
from andes.shared import np, ndarray


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
    def __init__(self, name: str = None, tex_name: str = None, info: str = None):
        self.name = name
        self.tex_name = tex_name if tex_name else name
        self.info = info
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
        if isinstance(self.v, np.ndarray):
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
    Service "variables" that stays constant.

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
                 name=None, tex_name=None, info=None):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.v_str = v_str
        self.v_numeric = v_numeric
        self.v: Union[float, int, ndarray] = 0.


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
                 src: str,
                 model: str,
                 indexer: BaseParam,
                 name: str = None,
                 tex_name: str = None,
                 info=None,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info)
        self.src = src
        self.model = model
        self.indexer = indexer
        self.v = 0

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
        self.v = ext_model.get(src=self.src, idx=self.indexer.v, attr='v')


class OperationService(BaseService):
    """
    Base class for a type of Service which performs specific operations

    This class cannot be used by itself.

    See Also
    --------
    ReducerService : Service for Reducing linearly stored 2-D services into 1-D

    RepeaterService : Service for repeating 1-D services following a sub-pattern
    """
    def __init__(self,
                 u,
                 ref: RefParam,
                 name=None,
                 tex_name=None,
                 ):
        self._v = None
        super().__init__(name=name, tex_name=tex_name)
        self.u = u
        self.ref = ref
        self.v_str = None

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value


class ReducerService(OperationService):
    """
    A helper Service type which reduces a linearly stored 2-D ExtParam into 1-D Service.

    ReducerService works with ExtParam whose ``v`` field is a list of lists. A reduce function
    which takes an array-like and returns a scalar need to be supplied. ReducerService calls the reduce
    function on each of the lists and return all the scalars in an array.

    Parameters
    ----------
    u : ExtParam
        Input ExtParam whose ``v`` contains linearly stored 2-dimensional values
    ref : RefParam
        The RefParam whose 2-dimensional shapes are used for indexing
    fun : Callable
        The callable for converting a 1-D array-like to a scalar

    Examples
    --------
    Suppose one wants to calculate the mean value of the ``Vn`` in one Area. In the ``Area`` class, one defines ::

        class AreaModel(...):
            def __init__(...):
                ...
                # backward reference from `Bus`
                self.Bus = RefParam()

                # collect the Vn in an 1-D array
                self.Vn = ExtParam(model='Bus',
                    src='Vn',
                    indexer=self.Bus)

                self.Vn_mean = ReducerService(u=self.Vn,
                    fun=np.mean,
                    ref=self.Bus)

    Suppose we define two areas, 1 and 2, the Bus data looks like ::

        idx    area  Vn
        1      1     110
        2      2     220
        3      1     345
        4      1     500

    Then, ``self.Bus.v`` is a list of two lists ``[ [1, 3, 4], [2] ]``. ``self.Vn.v`` will be retrieved and
    linearly stored as ``[110, 345, 500, 220]``. Based on the shape from ``self.Bus``, ``np.mean`` will be
    called on ``[110, 345, 500]`` and ``[220]`` respectively. Thus, ``self.Vn_mean.v`` will become
    ``[318.33, 220]``.

    """
    def __init__(self,
                 u,
                 ref: RefParam,
                 fun: Callable,
                 name=None,
                 tex_name=None,
                 ):
        super().__init__(u=u, ref=ref, name=name, tex_name=tex_name)
        self.fun = fun

    @property
    def v(self):
        """
        Return the reduced values from the reduction function in an array

        Returns
        -------
        The array ``self._v`` storing the reduced values
        """
        if self._v is None:
            self._v = np.zeros(len(self.ref.v))
            idx = 0
            for i, v in enumerate(self.ref.v):
                self._v[i] = self.fun(self.u.v[idx:idx + len(v)])
                idx += len(v)
            return self._v
        else:
            return self._v


class RepeaterService(OperationService):
    r"""
    A helper Service type which repeats a v-provider's value based on the shape from a RefParam

    Examples
    --------
    RepeaterService was originally designed for computing the inertia-weighted average rotor speed (center of
    inertia speed). COI speed is computed with

    .. math ::
        \omega_{COI} = \frac{ \sum{M_i * \omega_i} } {\sum{M_i}}

    The numerator can be calculated with a mix of RefParam, ExtParam and ExtState. The denominator needs to be
    calculated with ReducerService and Service Repeat. That is, use ReducerService to calculate the sum,
    and use RepeaterService to repeat the summed value for each device.

    In the COI class, one would have ::

        class COIModel(...):
            def __init__(...):
                ...
                self.SynGen = RefParam()

                self.M = ExtParam(model='SynGen',
                                  src='M',
                                  indexer=self.SynGen)

                self.w_sg = ExtState(model='SynGen',
                                     src='omega',
                                     indexer=self.SynGen)

                self.Mt = ReducerService(u=self.M,
                                        fun=np.sum,
                                        ref=self.SynGen)

                self.Mtr = RepeaterService(u=self.M_sym,
                                         ref=self.SynGen)

    Finally, one would define the center of inertia speed as ::

        self.w_coi = Algeb(v_str='1', e_str='-w_coi')

        self.w_coi_sub = ExtAlgeb(model='COI',
                                  src='w_coi',
                                  e_str='M * w_sg / Mtr',
                                  v_str='M / Mtr',
                                  indexer=self.padded_idx,  # TODO
                                  )

    It is very worth noting that the implementation uses a trick to separate the average weighted sum into ``n``
    sub-equations, each calculating the :math:`(M_i * \omega_i) / (\sum{M_i})`. Since all the variables are
    preserved in the sub-equation, the derivatives can be calculated correctly.

    """
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    @property
    def v(self):
        """
        Return the values of the repeated values in a sequantial 1-D array

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


class RandomService(BaseService):
    """
    A service variable for generating random numbers.

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


# TODO: SafeInverse
