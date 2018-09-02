"""
pymatbridge
===========

This is a module for communicating and running Matlab from within python

Example
-------

>>> import pymatbridge
>>> m = pymatbridge.Matlab()
>>> m.start()
Starting MATLAB on ZMQ socket ipc:///tmp/pymatbridge
Send 'exit' command to kill the server
.MATLAB started and connected!
True
>>> m.run_code('a=1;')
{'content': {'stdout': '', 'datadir': '/private/tmp/MatlabData/',
 'code': 'a=1;', 'figures': []}, 'success': True}
>>> m.get_variable('a')
1

"""

import atexit
import os
import time
import base64
import zmq
import subprocess
import sys
import json
import types
import weakref
from uuid import uuid4

from numpy import ndarray, generic, float64, frombuffer, asfortranarray

try:
    from scipy.sparse import spmatrix
except ImportError:

    class spmatrix:
        pass


def encode_ndarray(obj):
    """Write a numpy array and its shape to base64 buffers"""
    shape = obj.shape
    if len(shape) == 1:
        shape = (1, obj.shape[0])
    if obj.flags.c_contiguous:
        obj = obj.T
    elif not obj.flags.f_contiguous:
        obj = asfortranarray(obj.T)
    else:
        obj = obj.T
    try:
        data = obj.astype(float64).tobytes()
    except AttributeError:
        data = obj.astype(float64).tostring()

    data = base64.b64encode(data).decode('utf-8')
    return data, shape


# JSON encoder extension to handle complex numbers and numpy arrays
class PymatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray) and obj.dtype.kind in 'uif':
            data, shape = encode_ndarray(obj)
            return {'ndarray': True, 'shape': shape, 'data': data}
        elif isinstance(obj, ndarray) and obj.dtype.kind == 'c':
            real, shape = encode_ndarray(obj.real.copy())
            imag, _ = encode_ndarray(obj.imag.copy())
            return {
                'ndarray': True,
                'shape': shape,
                'real': real,
                'imag': imag
            }
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, generic):
            return obj.item()
        # Handle the default case
        return json.JSONEncoder.default(self, obj)


def decode_arr(data):
    """Extract a numpy array from a base64 buffer"""
    data = data.encode('utf-8')
    return frombuffer(base64.b64decode(data), float64)


# JSON decoder for arrays and complex numbers
def decode_pymat(dct):
    if 'ndarray' in dct and 'data' in dct:
        value = decode_arr(dct['data'])
        shape = dct['shape']
        if type(dct['shape']) is not list:
            shape = decode_arr(dct['shape']).astype(int)
        return value.reshape(shape, order='F')
    elif 'ndarray' in dct and 'imag' in dct:
        real = decode_arr(dct['real'])
        imag = decode_arr(dct['imag'])
        shape = decode_arr(dct['shape']).astype(int)
        data = real + 1j * imag
        return data.reshape(shape, order='F')
    elif 'real' in dct and 'imag' in dct:
        return complex(dct['real'], dct['imag'])
    return dct


MATLAB_FOLDER = '%s/matlab' % os.path.realpath(os.path.dirname(__file__))


class _Session(object):
    """
    A class for communicating with a MATLAB session. It provides the behavior
    common across different MATLAB implementations. You shouldn't instantiate
    this directly; rather, use the Matlab or Octave subclasses.
    """

    def __init__(self,
                 executable,
                 socket_addr=None,
                 id='python-matlab-bridge',
                 log=False,
                 maxtime=60,
                 platform=None,
                 startup_options=None):
        """
        Initialize this thing.

        Parameters
        ----------

        executable : str
            A string that would start the session at the terminal.

        socket_addr : str
            A string that represents a valid ZMQ socket address, such as
            "ipc:///tmp/pymatbridge", "tcp://127.0.0.1:55555", etc. Default is
            to choose a random IPC file name, or a random socket (for TCP).

        id : str
            An identifier for this instance of the pymatbridge.

        log : bool
            Whether to save a log file in some known location.

        maxtime : float
           The maximal time to wait for a response from the session (optional,
           Default is 10 sec)

        platform : string
           The OS of the machine on which this is running. Per default this
           will be taken from sys.platform.

        startup_options : string
           Command line options to include in the executable's invocation.
           Optional; sensible defaults are used if this is not provided.
        """
        self.started = False
        self.executable = executable
        self.socket_addr = socket_addr
        self.id = id
        self.log = log
        self.maxtime = maxtime
        self.platform = platform if platform is not None else sys.platform
        self.startup_options = startup_options

        if socket_addr is None:
            if self.platform == "win32":
                self.socket_addr = "tcp://127.0.0.1"
            else:
                self.socket_addr = "ipc:///tmp/pymatbridge-%s" % str(uuid4())

        if self.log:
            startup_options += ' > ./pymatbridge/logs/bashlog_%s.txt' % self.id

        self.context = None
        self.socket = None
        atexit.register(self.stop)

    def _program_name(self):  # pragma: no cover
        raise NotImplemented

    def _preamble_code(self):
        # suppress warnings while loading the path, in the case of
        # overshadowing a built-in function on a newer version of
        # Matlab (e.g. isrow)
        return [
            "old_warning_state = warning('off','all');",
            "addpath(genpath('%s'));" % MATLAB_FOLDER,
            "warning(old_warning_state);", "clear('old_warning_state');",
            "cd('%s');" % os.getcwd()
        ]

    def _execute_flag(self):  # pragma: no cover
        raise NotImplemented

    def _run_server(self):
        code = self._preamble_code()
        code.extend(["matlabserver('%s')" % self.socket_addr])
        command = '%s %s %s "%s"' % (self.executable, self.startup_options,
                                     self._execute_flag(), ','.join(code))
        subprocess.Popen(
            command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # Start server/client session and make the connection
    def start(self, run_matlab_instance, act_as_server, address, context=None):
        # Setup socket
        self.context = context or zmq.Context()

        if act_as_server is True:
            self.socket = self.context.socket(zmq.REP)
        else:
            self.socket = self.context.socket(zmq.REQ)

        if self.platform == "win32":
            port = self.socket.bind_to_random_port(self.socket_addr)
            self.socket_addr = self.socket_addr + ":%s" % port
            self.socket.unbind(self.socket_addr)

        if run_matlab_instance is False:
            # Not starting instance
            self.socket_addr = address
        else:
            # Start the MATLAB server in a new process
            print("Starting %s on ZMQ socket %s" % (self._program_name(),
                                                    self.socket_addr))
            print("Send 'exit' command to kill the server")
            self._run_server()

        if act_as_server is True:
            # Start the server
            self.socket.connect(self.socket_addr)
        else:
            # Start the client
            print("Acting as client")
            self.socket.connect(self.socket_addr)

        self.started = True
        """
        # Test if connection is established
        if self.is_connected():
            print("%s started and connected!" % self._program_name())
            self.set_plot_settings()
            return self
        else:
            raise ValueError("%s failed to start" % self._program_name())
        """

    def _response(self, **kwargs):
        req = json.dumps(kwargs, cls=PymatEncoder)
        return self.socket.send(req)
        # resp = self.socket.recv_multipart()
        # return resp

    # Stop the Matlab server
    def stop(self):
        if not self.started:
            return True

        # Matlab should respond with "exit" if successful
        if self._response(cmd='exit') == "exit":
            print("%s closed" % self._program_name())

        self.started = False
        return True

    # To test if the client can talk to the server
    def is_connected(self):
        if not self.started:
            time.sleep(2)
            return False

        req = json.dumps(dict(cmd="connect"), cls=PymatEncoder)
        self.socket.send_string(req)

        start_time = time.time()
        while True:
            try:
                resp = self.socket.recv_string(flags=zmq.NOBLOCK)
                return resp == "connected"
            except zmq.ZMQError:
                sys.stdout.write('.')
                time.sleep(1)
                if time.time() - start_time > self.maxtime:
                    print("%s session timed out after %d seconds" %
                          (self._program_name(), self.maxtime))
                    return False

    def is_function_processor_working(self):
        result = self.run_func('%s/usrprog/test_sum.m' % MATLAB_FOLDER, {
            'echo':
            '%s: Function processor is working!' % self._program_name()
        })
        return result['success']

    def _json_response(self, **kwargs):
        return self._response(**kwargs)
        # return json.loads(, object_hook=decode_pymat)

    def json_decode(self, msg):
        return json.loads(msg, object_hook=decode_pymat)

    def json_encode(self, msg):
        return json.dumps(msg, cls=PymatEncoder)

    def run_func(self, func_path, *func_args, **kwargs):
        """Run a function in Matlab and return the result.

        Parameters
        ----------
        func_path: str
            Name of function to run or a path to an m-file.
        func_args: object, optional
            Function args to send to the function.
        nargout: uid, optional
            Desired number of return arguments.
        kwargs:
            Keyword arguments are passed to Matlab in the form [key, val] so
            that matlab.plot(x, y, '--', LineWidth=2) would be translated into
            plot(x, y, '--', 'LineWidth', 2)

        Returns
        -------
        Result dictionary with keys: 'log', 'result', and 'success'
        """
        if not self.started:
            raise ValueError('Session not started, use start()')

        nargout = kwargs.pop('nargout', 1)
        func_args += tuple(
            item for pair in zip(kwargs.keys(), kwargs.values())
            for item in pair)
        dname = os.path.dirname(func_path)
        fname = os.path.basename(func_path)
        func_name, ext = os.path.splitext(fname)
        if ext and not ext == '.m':
            raise TypeError('Need to give path to .m file')
        return self._json_response(
            cmd='eval',
            func_name=func_name,
            func_args=func_args or '',
            dname=dname,
            nargout=nargout)

    def run_code(self, code):
        """Run some code in Matlab command line provide by a string

        Parameters
        ----------
        code : str
            Code to send for evaluation.
        """
        return self.run_func('evalin', 'base', code, nargout=0)

    def get_variable(self, varname, default=None):
        resp = self.run_func('evalin', 'base', varname)
        return resp
        # return resp['result'] if resp['success'] else default

    def set_variable(self, varname, value):
        if isinstance(value, spmatrix):
            return self._set_sparse_variable(varname, value)
        return self.run_func('assignin', 'base', varname, value, nargout=0)

    def set_plot_settings(self, width=512, height=384, inline=True):
        if inline:
            code = ["set(0, 'defaultfigurevisible', 'off')"]
        else:
            code = ["set(0, 'defaultfigurevisible', 'on')"]
        size = "set(0, 'defaultfigurepaperposition', [0 0 %s %s])"
        code += [
            "set(0, 'defaultfigurepaperunits', 'inches')",
            "set(0, 'defaultfigureunits', 'inches')",
            size % (int(width) / 150., int(height) / 150.)
        ]
        self.run_code(';'.join(code))

    def _set_sparse_variable(self, varname, value):
        value = value.todok()
        prefix = 'pymatbridge_temp_sparse_%s_' % uuid4().hex
        self.set_variable(prefix + 'keys', value.keys())
        # correct for 1-indexing in MATLAB
        self.run_code('{0}keys = {0}keys + 1;'.format(prefix))
        self.set_variable(prefix + 'values', value.values())
        cmd = "{1} = sparse({0}keys(:, 1), {0}keys(:, 2), {0}values');"
        result = self.run_code(cmd.format(prefix, varname))
        self.run_code('clear {0}keys {0}values'.format(prefix))
        return result

    def __getattr__(self, name):
        """If an attribute is not found, try to create a bound method"""
        return self._bind_method(name)

    def _bind_method(self, name, unconditionally=False):
        """Generate a Matlab function and bind it to the instance

        This is where the magic happens. When an unknown attribute of the
        Matlab class is requested, it is assumed to be a call to a
        Matlab function, and is generated and bound to the instance.

        This works because getattr() falls back to __getattr__ only if no
        attributes of the requested name can be found through normal
        routes (__getattribute__, __dict__, class tree).

        bind_method first checks whether the requested name is a callable
        Matlab function before generating a binding.

        Parameters
        ----------
        name : str
            The name of the Matlab function to call
                e.g. 'sqrt', 'sum', 'svd', etc
        unconditionally : bool, optional
            Bind the method without performing
                checks. Used to bootstrap methods that are required and
                know to exist

        Returns
        -------
        MatlabFunction
            A reference to a newly bound MatlabFunction instance if the
                requested name is determined to be a callable function

        Raises
        ------
        AttributeError: if the requested name is not a callable
                Matlab function

        """
        # TODO: This does not work if the function is a mex function inside a
        #   folder of the same name
        exists = self.run_func('exist', name)['result'] in [2, 3, 5]
        if not unconditionally and not exists:
            raise AttributeError(
                "'Matlab' object has no attribute '%s'" % name)

        # create a new method instance
        method_instance = MatlabFunction(weakref.ref(self), name)
        method_instance.__name__ = name

        # bind to the Matlab instance with a weakref (to avoid circular
        #   references)
        if sys.version.startswith('3'):
            method = types.MethodType(method_instance, weakref.ref(self))
        else:
            method = types.MethodType(method_instance, weakref.ref(self),
                                      _Session)
        setattr(self, name, method)
        return getattr(self, name)


class Matlab(_Session):
    def __init__(self,
                 executable='matlab',
                 socket_addr=None,
                 id='python-matlab-bridge',
                 log=False,
                 maxtime=60,
                 platform=None,
                 startup_options=None):
        """
        Initialize this thing.

        Parameters
        ----------

        executable : str
            A string that would start Matlab at the terminal. Per default, this
            is set to 'matlab', so that you can alias in your bash setup

        socket_addr : str
            A string that represents a valid ZMQ socket address, such as
            "ipc:///tmp/pymatbridge", "tcp://127.0.0.1:55555", etc.

        id : str
            An identifier for this instance of the pymatbridge.

        log : bool
            Whether to save a log file in some known location.

        maxtime : float
           The maximal time to wait for a response from matlab (optional,
           Default is 10 sec)

        platform : string
           The OS of the machine on which this is running. Per default this
           will be taken from sys.platform.

        startup_options : string
           Command line options to pass to MATLAB. Optional; sensible defaults
           are used if this is not provided.
        """
        if platform is None:
            platform = sys.platform
        if startup_options is None:
            if platform == 'win32':
                startup_options = ' -automation -nosplash'
            else:
                startup_options = ' -nodesktop -nosplash'
        if log:
            startup_options += ' -logfile ./pymatbridge/logs/matlablog_%s.txt'\
                               % id
        super(Matlab, self).__init__(executable, socket_addr, id, log, maxtime,
                                     platform, startup_options)

    def _program_name(self):
        return 'MATLAB'

    def _execute_flag(self):
        return '-r'


class Octave(_Session):
    def __init__(self,
                 executable='octave',
                 socket_addr=None,
                 id='python-matlab-bridge',
                 log=False,
                 maxtime=60,
                 platform=None,
                 startup_options=None):
        """
        Initialize this thing.

        Parameters
        ----------

        executable : str
            A string that would start Octave at the terminal. Per default, this
            is set to 'octave', so that you can alias in your bash setup

        socket_addr : str
            A string that represents a valid ZMQ socket address, such as
            "ipc:///tmp/pymatbridge", "tcp://127.0.0.1:55555", etc.

        id : str
            An identifier for this instance of the pymatbridge.

        log : bool
            Whether to save a log file in some known location.

        maxtime : float
           The maximal time to wait for a response from octave (optional,
           Default is 10 sec)

        platform : string
           The OS of the machine on which this is running. Per default this
           will be taken from sys.platform.

        startup_options : string
           Command line options to pass to Octave. Optional; sensible defaults
           are used if this is not provided.
        """
        if startup_options is None:
            startup_options = '--silent --no-gui'
        super(Octave, self).__init__(executable, socket_addr, id, log, maxtime,
                                     platform, startup_options)

    def _program_name(self):
        return 'Octave'

    def _preamble_code(self):
        code = super(Octave, self)._preamble_code()
        if self.log:
            code.append(
                "diary('./pymatbridge/logs/octavelog_%s.txt')" % self.id)
        code.append("graphics_toolkit('gnuplot')")
        return code

    def _execute_flag(self):
        return '--eval'


class MatlabFunction(object):
    def __init__(self, parent, name):
        """An object representing a Matlab function

        Methods are dynamically bound to instances of Matlab objects and
        represent a callable function in the Matlab subprocess.

        Parameters
        ----------
        parent: Matlab instance
            A reference to the parent (Matlab instance) to which the
                MatlabFunction is being bound
        name: str
            The name of the Matlab function this represents
        """
        self.name = name
        self._parent = parent
        self.doc = None

    def __call__(self, unused_parent_weakref, *args, **kwargs):
        """call a function with the supplied arguments in the Matlab subprocess

        Passes parameters to `run_func`.

        """
        return self.parent.run_func(self.name, *args, **kwargs)

    @property
    def parent(self):
        """Get the actual parent from the stored weakref

        The parent (Matlab instance) is stored as a weak reference
        to eliminate circular references from dynamically binding Methods
        to Matlab.

        """
        parent = self._parent()
        if parent is None:
            raise AttributeError(
                'Stale reference to attribute of non-existent Matlab object')
        return parent

    @property
    def __doc__(self):
        """Fetch the docstring from Matlab

        Get the documentation for a Matlab function by calling Matlab's builtin
        help() then returning it as the Python docstring. The result is cached
        so Matlab is only ever polled on the first request

        """
        if self.doc is None:
            self.doc = self.parent.help(self.name)['result']
        return self.doc
