from time import time
from decimal import Decimal, ROUND_DOWN


def elapsed(t0=0.0):
    """get elapsed time from the give time

    Returns:
        now: the absolute time now
        dt_str: elapsed time in string
        """
    now = time()
    dt = now - t0
    dt_sec = Decimal(str(dt)).quantize(Decimal('.0001'), rounding=ROUND_DOWN)
    if dt_sec <= 1:
        dt_str = str(dt_sec) + ' second'
    else:
        dt_str = str(dt_sec) + ' seconds'
    return now, dt_str
