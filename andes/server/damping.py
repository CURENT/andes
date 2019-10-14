import json
import logging
import sys
import time  # NOQA
from andes.utils.math import to_number
from urllib.parse import urljoin

import scipy.io
import numpy as np
import queue

# 100ms - 4
# 200ms - 7
# 300ms - 10
yq = queue.Queue(maxsize=10)
mat = scipy.io.loadmat("Controller_new.mat")


try:
    import requests
except ImportError:
    print("Requests import error. Install optional package `requests`")
    sys.exit(1)


logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
logger.addHandler(sh)
logger.setLevel(logging.INFO)


class ANDESClient(object):

    def __init__(self, server_url):
        self.sysid = None
        self.server_url = server_url
        self.r = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_case()

    def load_case(self, case, with_dime=0, tf=20):
        """
        Use the REST API to load a data in andes server

        Parameters
        ----------
        case

        Returns
        -------

        """
        if self.sysid is not None:
            logger.error('sysid <{}> is not none. unload first.'.format(self.sysid))

        url = urljoin(self.server_url, 'load')

        params = {'name': case, 'with_dime': with_dime, 'tf': tf}
        self.r = requests.get(url=url, params=params)
        logger.debug('Load case <{}> requested'.format(case))
        if not self.r.ok:
            logger.error('load <{}> error. file may not exist in server root.'.format(case))
            return False
        else:
            self.sysid = int(self.r.text)
            logger.info('laod_case returned sysid is <{}>'.format(self.sysid))
            return True

    def unload_case(self, force=False):
        """
        Unload the case previously loaded by this client

        Returns
        -------

        """
        url = urljoin(self.server_url, 'unload')

        params = {'sysid': self.sysid, 'force': force}
        self.r = requests.get(url=url, params=params)
        logger.debug('Unload sysid <{}> requested'.format(self.sysid))

        if not self.r.ok:
            logger.error('Unload <{}> error. System might not have been loaded.'.format(self.sysid))
            return False
        else:
            self.sysid = None
            logger.info('Unload sent successful')
            return True

    def run_case(self):
        """
        Run a loaded system
        Parameters
        ----------

        Returns
        -------

        """

        url = urljoin(self.server_url, 'run')
        params = {'sysid': self.sysid}

        self.r = requests.get(url=url, params=params)
        logger.debug('Run case requested')
        if not self.r.ok:
            logger.error('Run simulation error. System may not have been loaded.')
            return False
        else:
            ret = self.r.text.strip()
            logger.info('run_case returns value {}'.format(ret))
            return True

    def param_get(self, model, var_name=None, idx=None, sysbase=False):
        """
        Get parameter from the loaded system

        Parameters
        ----------
        server_url
        model
        var_name
        idx
        sysbase

        Returns
        -------

        """
        url = urljoin(self.server_url, 'param')
        params = {'sysid': self.sysid, 'name': model, 'var': var_name, 'idx': idx, 'sysbase': sysbase}

        self.r = requests.get(url=url, params={"args_json": json.dumps(params)})

        logger.debug('param_get requested with {}'.format(params))

        if not self.r.ok:
            logger.error('Get parameter error. Check if the params are correct.')
            return False
        else:
            ret = to_number(self.r.text)
            logger.info('param_get returns value <{}>'.format(ret))
            return ret

    def param_set(self, model, var_name, idx, value, sysbase=False):
        """
        Set parameter for the loaded system

        Parameters
        ----------
        server_url
        model
        var_name
        idx
        value
        base

        Returns
        -------

        """
        url = urljoin(self.server_url, 'param')
        params = {'sysid': self.sysid,
                  'name': model,
                  'var': var_name,
                  'idx': idx,
                  'value': value,
                  'sysbase': sysbase}

        self.r = requests.post(url=url, params={"args_json": json.dumps(params)})
        logger.debug('param_set called with {}'.format(params))
        if not self.r.ok:
            logger.error('Set parameter error. Check if the params are correct.')
            return False
        else:
            ret = self.r.text.strip()
            logger.debug('param_set returns value <{}>'.format(ret))
            return True

    def is_loaded(self):
        """
        Check if the server is loaded with a case

        Parameters
        ----------
        server_url

        Returns
        -------

        """
        url = urljoin(self.server_url, 'status')
        params = {'sysid': self.sysid}

        self.r = requests.get(url=url, params=params)
        logger.debug('is_loaded requested for sysid <{}>'.format(self.sysid))
        status = int(json.loads(self.r.text))

        if status == 0:
            logger.debug('is_loaded returned False')
            return 0
        elif status == 1:
            logger.debug('is_loaded returned True')
            return 1
        elif status == 2:
            logger.debug('is_loaded returned 2 (done)')
            return 2
        else:
            logger.error('Unknown status {}'.format(status))

    def get_sim_time(self):
        """
        Get current simulation time
        """
        url = urljoin(self.server_url, 'sim_time')
        params = {'sysid': self.sysid}

        self.r = requests.get(url=url, params=params)

        return float(json.loads(self.r.text))

    def get_streaming(self):
        """
        Get streaming data
        """
        url = urljoin(self.server_url, 'streaming')
        params = {'sysid': self.sysid}

        self.r = requests.get(url=url, params=params)

        return json.loads(self.r.text)


if __name__ == '__main__':

    server = 'http://192.168.1.200:7000'

    with ANDESClient(server) as client:
        r = client.load_case('WECC_WIND10_Damping_Coord.dm', with_dime=1, tf=20)

        mat = globals()['mat']

        omega_idx = 0
        t0 = 0.0
        x = np.zeros((6, 1), dtype="float64")
        vref0 = None

        if client.is_loaded() is not False:

            client.run_case()

            while 1 <= client.is_loaded() < 2:
                if (0 <= client.get_sim_time() < 0.1) and vref0 is None:
                    vref0 = client.param_get(model="WTG3", var_name="vref0")
                    omega_idx = client.param_get(model="Syn6a", var_name="omega", idx=3)

                    vref0 = [float(i) for i in vref0[1:-1].split(',')]
                    vref0 = np.array(vref0, dtype="float64").reshape((-1, 1))

                t = client.get_sim_time()
                if t != t0:
                    streaming_data_dict = client.get_streaming()
                    streaming_data = np.array(streaming_data_dict['var'], dtype="float64")

                    omega = (streaming_data[omega_idx] - 1.0)

                    h = t - t0
                    t0 = t

                    # calculate controller output

                    x_new = mat['Ac'] @ x * h + mat['Bc'] * omega
                    y = mat['Cc'] @ x_new

                    y40 = mat['Binv'] @ y
                    yq.put({"t_compute": t, "value": vref0 + y40})

                    x = x_new

                    if not yq.full():
                        pass
                    else:
                        y_got = yq.get()
                        y_set = y_got["value"]
                        t_compute = y_got["t_compute"]

                        wtg_idx = list(range(1, 41))

                        client.param_set(model="WTG3",
                                         var_name="vref0",
                                         idx=wtg_idx,
                                         value=y_set.reshape((-1, )).tolist(),
                                         sysbase=True
                                         )

                        print(f"Set y computed at {t_compute} when t={t}")
