import requests
import logging
import time  # NOQA
import os
import json
from andes.utils.math import to_number

logger = logging.getLogger(__name__)


class ANDESClient(object):

    def __init__(self, server_url):
        self.sysid = None
        self.server_url = server_url
        self.r = None

    def load_case(self, case):
        """
        Use the REST API to load a data in andes server

        Parameters
        ----------
        case

        Returns
        -------

        """
        url = os.path.join(self.server_url, 'load')

        params = {'name': case}
        self.r = requests.get(url=url, params=params)
        if not self.r.ok:
            logger.error('load <{}> error. file may not exist in server root.'.format(case))
            return False
        else:
            self.sysid = int(self.r.text)
            return True

    def run_case(self, simulation_time):
        """
        Run a loaded system
        Parameters
        ----------
        simulation_time

        Returns
        -------

        """
        if simulation_time <= 0:
            logger.error('Simulation time must be greater than 0')
            return

        url = os.path.join(self.server_url, 'run')
        params = {'sysid': self.sysid, 'time': simulation_time}

        self.r = requests.get(url=url, params=params)

        if not self.r.ok:
            logger.error('Run simulation error. System may not have been loaded.')
            return False
        else:
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
        url = os.path.join(self.server_url, 'param')
        params = {'sysid': self.sysid, 'name': model, 'var': var_name, 'idx': idx, 'sysbase': sysbase}

        self.r = requests.get(url=url, params=params)

        if not self.r.ok:
            logger.error('Get parameter error. Check if the params are correct.')
            return False
        else:
            return to_number(self.r.text)

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
        url = os.path.join(self.server_url, 'param')
        params = {'sysid': self.sysid,
                  'name': model,
                  'var': var_name,
                  'idx': idx,
                  'value': value,
                  'sysbase': sysbase}

        self.r = requests.post(url=url, params=params)

        if not self.r.ok:
            logger.error('Set parameter error. Check if the params are correct.')
            return False
        else:
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
        url = os.path.join(self.server_url, 'status')
        params = {'sysid': self.sysid}

        self.r = requests.get(url=url, params=params)

        status = int(json.loads(self.r.text))

        if status == 0:
            return False
        elif status == 1:
            return True
        else:
            logger.error('Unknown status {}'.format(status))


if __name__ == '__main__':

    local_server = 'http://127.0.0.1:5000'
    client = ANDESClient(server_url=local_server)
    r = client.load_case('ieee14_syn.dm')

    if client.is_loaded():
        client.run_case(20)

    if client.param_get(model='AVR1', var_name='Ka', idx=1):
        print(client.r.text)

    time.sleep(5)
    if client.param_set(model='AVR1', var_name='Ka', idx=1, value=50, sysbase=False):
        print(client.r.text)

    if client.param_get(model='AVR1', var_name='Ka', idx=1):
        print(client.r.text)
