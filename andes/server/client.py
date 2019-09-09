import json
import logging
import os
import sys
import time  # NOQA
from andes.utils.math import to_number

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

    def load_case(self, case):
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

        url = os.path.join(self.server_url, 'load')

        params = {'name': case}
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
        url = os.path.join(self.server_url, 'unload')

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
        logger.debug('Run case requested for <{}>s'.format(simulation_time))
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
        url = os.path.join(self.server_url, 'param')
        params = {'sysid': self.sysid, 'name': model, 'var': var_name, 'idx': idx, 'sysbase': sysbase}

        self.r = requests.get(url=url, params=params)

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
        url = os.path.join(self.server_url, 'param')
        params = {'sysid': self.sysid,
                  'name': model,
                  'var': var_name,
                  'idx': idx,
                  'value': value,
                  'sysbase': sysbase}

        self.r = requests.post(url=url, params=params)
        logger.debug('param_set requested with {}'.format(params))
        if not self.r.ok:
            logger.error('Set parameter error. Check if the params are correct.')
            return False
        else:
            ret = self.r.text.strip()
            logger.info('param_set returns value <{}>'.format(ret))
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
        logger.debug('is_loaded requested for sysid <{}>'.format(self.sysid))
        status = int(json.loads(self.r.text))

        if status == 0:
            logger.info('is_loaded returned False')
            return False
        elif status == 1:
            logger.info('is_loaded returned True')
            return True
        else:
            logger.error('Unknown status {}'.format(status))


def fault_test(client):

    if client.param_get(model='Fault', var_name='u', idx=1) is not False:
        logger.info('Fault.1.u before: {}'.format(client.r.text))

    time.sleep(1)

    if client.param_set('Fault', 'u', '1', value=1, sysbase=True) is not False:
        logger.info('Fault.1.u after: {}'.format(client.r.text))


def avr_ka_test(client):
    if client.param_get(model='AVR1', var_name='Ka', idx=1) is not False:
        logger.info('AVR1.1.Ka before: {}'.format(client.r.text))

    time.sleep(5)
    if client.param_set(model='AVR1', var_name='Ka', idx=1, value=50, sysbase=False) is not False:
        pass

    if client.param_get(model='AVR1', var_name='Ka', idx=1) is not False:
        logger.info('AVR1.1.Ka after: {}'.format(client.r.text))


if __name__ == '__main__':

    local_server = 'http://127.0.0.1:5000'
    with ANDESClient(local_server) as client:
        r = client.load_case('ieee14_syn.dm')

        if client.is_loaded() is not False:
            client.run_case(20)

        fault_test(client)

        avr_ka_test(client)
