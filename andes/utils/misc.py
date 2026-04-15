import logging
import os
import platform
import sys
from time import time

from decimal import ROUND_DOWN, Decimal

logger = logging.getLogger(__name__)
_missing = object()


def elapsed(t0=0.0):
    """
    Get the elapsed time from the give time.
    If the start time is not given, returns the unix-time.

    Returns
    -------
    t : float
        Elapsed time from the given time; Otherwise the epoch time.
    s : str
        The elapsed time in seconds in a string
    """
    t = time()
    dt = t - t0
    dt_sec = Decimal(str(dt)).quantize(Decimal('.0001'), rounding=ROUND_DOWN)
    if dt_sec == 1:
        s = str(dt_sec) + ' second'
    else:
        s = str(dt_sec) + ' seconds'
    return t, s


def to_number(s):
    """
    Convert a string to a number. If unsuccessful, return the de-blanked string.
    """
    ret = s

    # remove single quotes
    if "'" in ret:
        ret = ret.strip("'").strip()

    # try converting to booleans / None
    if ret == 'True':
        return True
    elif ret == 'False':
        return False
    elif ret == 'None':
        return None

    # try converting to float or int
    try:
        ret = int(ret)
    except ValueError:
        try:
            ret = float(ret)
        except ValueError:
            pass

    return ret


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qt-console
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def is_interactive():
    """
    Check if is in an interactive shell (python or ipython).

    Returns
    -------
    bool

    """
    ipython = False
    try:
        cls_name = get_ipython().__class__.__name__

        if cls_name in ('InteractiveShellEmbed', 'TerminalInteractiveShell'):
            ipython = True
    except NameError:
        pass

    import __main__ as main

    return not hasattr(main, '__file__') or ipython


def supports_color(stream=None):
    """Check whether the given stream (default stderr) supports ANSI colors.

    Respects the ``NO_COLOR`` environment variable (https://no-color.org)
    and the ``FORCE_COLOR`` override used by many CLI tools.
    """
    if os.environ.get('FORCE_COLOR') is not None:
        return True
    if os.environ.get('NO_COLOR') is not None:
        return False

    if stream is None:
        stream = sys.stderr
    if not hasattr(stream, 'isatty') or not stream.isatty():
        return False

    # 'dumb' terminal universally means no color support
    if os.environ.get('TERM') == 'dumb':
        return False

    if platform.system() == 'Windows':
        # Windows 10 build 10586+ supports ANSI natively in conhost/WT
        try:
            win_build = sys.getwindowsversion().build
        except AttributeError:
            win_build = 0
        return (os.environ.get('ANSICON')
                or os.environ.get('WT_SESSION')
                or 'TERM_PROGRAM' in os.environ
                or win_build >= 10586)

    return True


class ColoredFormatter(logging.Formatter):
    """Logging formatter that adds ANSI color codes to log output."""

    COLORS = {
        'DEBUG':    '\033[36m',     # cyan
        'INFO':     '\033[32m',     # green
        'WARNING':  '\033[33m',     # yellow
        'ERROR':    '\033[31m',     # red
        'CRITICAL': '\033[1;31m',   # bold red
    }
    RESET = '\033[0m'

    def format(self, record):
        result = super().format(record)
        color = self.COLORS.get(record.levelname, '')
        if color:
            result = f"{color}{result}{self.RESET}"
        return result


def is_tty():
    """
    Check if stdout is connected to a TTY (interactive terminal).

    This is useful for detecting batch execution environments like Sphinx
    latexpdf builds where notebooks are executed programmatically but
    `is_notebook()` still returns True.

    Returns
    -------
    bool
        True if stdout is a TTY, False otherwise.
    """
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def is_nbconvert():
    """
    Check if running inside nbconvert (e.g., during Sphinx doc builds).

    Detection methods:
    1. Parent process contains 'nbconvert'
    2. Kernel's parent header has 'execute_request' but no frontend comms

    Returns
    -------
    bool
        True if running in nbconvert, False otherwise.
    """
    import os
    import re

    try:
        with open(f'/proc/{os.getppid()}/cmdline') as f:
            cmdline = f.read()
            if re.search(r'(?<![_a-zA-Z])nbconvert(?![_a-zA-Z])', cmdline):
                return True
    except Exception:
        pass

    return False


def has_notebook_frontend():
    """
    Check if running in a Jupyter notebook with an active frontend connection.

    This distinguishes between:
    - Interactive Jupyter Lab/Notebook (has frontend) -> Returns True
    - nbconvert/Sphinx execution (no frontend) -> Returns False

    Returns
    -------
    bool
        True if in an interactive notebook with frontend, False otherwise.
    """
    if not is_notebook():
        return False

    # Explicitly check for nbconvert (most reliable)
    if is_nbconvert():
        return False

    # If we're in a notebook and not in nbconvert, assume we have a frontend
    return True


class cached:
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::

        class Foo:

            @cached
            def foo(self):
                # calculate something important here
                return 42

    The class has to have a `__dict__` in order for this property to
    work.
    See for details:
    http://stackoverflow.com/questions/17486104/python-lazy-loading-of-class-attributes
    """

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value
