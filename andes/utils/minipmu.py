"""Python module to request PMU data from a running ANDES
"""

import logging
import time

from andes.utils.dime import Dime

import numpy as np
from numpy import array, ndarray

from pypmu import Pmu
from pypmu.frame import ConfigFrame2, HeaderFrame


class MiniPMU(object):

    def __init__(self, name:str='', dime_address:str='ipc:///tmp/dime',
                 pmu_idx:list=list(), max_store=1000, pmu_ip='0.0.0.0', pmu_port=1410,
                 ):
        assert name, 'PMU Receiver name is empty'
        assert pmu_idx, 'PMU idx is empty'

        self.dime_address = dime_address
        self.pmu_idx = pmu_idx
        self.max_store = max_store

        self.bus_name = []
        self.var_idx = {'am': [],
                        'vm': [],
                        'w': [],
                        }

        self.Varheader = list()
        self.Idxvgs = dict()
        self.SysParam = dict()
        self.Varvgs = ndarray([])

        self.t = ndarray([])
        self.data = ndarray([])
        self.count = 0

        self.dimec = Dime(name, dime_address)
        self.pmu = Pmu(ip="0.0.0.0", port=pmu_port)

    def start_dime(self):
        """Starts the dime client stored in `self.dimec`
        """
        # self.dimec.exit()
        assert self.dimec.start()
        logging.info('DiME client connected to {}'.format(self.dime_address))

    def respond_to_sim(self):
        """Respond with data streaming configuration to the simulator"""
        pass

    def get_bus_name(self):
        """Return bus names based on ``self.pmu_idx`` and store bus names to ``self.bus_name``
        """
        # TODO: implement method to read bus names from Varheader
        self.bus_name = list(self.pmu_idx)
        for i in range(len(self.bus_name)):
            self.bus_name[i] = 'Bus_' + str(self.bus_name[i])

        return self.bus_name

    def config_pmu(self):
        """Sets the ConfigFrame2 of the PMU
        """

        self.cfg = ConfigFrame2(self.pmu_idx[0],  # PMU_ID
                           1000000,  # TIME_BASE
                           1,  # Number of PMUs included in data frame
                           self.bus_name[0],  # Station name
                           self.pmu_idx[0],  # Data-stream ID(s)
                           (True, True, True, True),  # Data format - POLAR; PH - REAL; AN - REAL; FREQ - REAL;
                           1,  # Number of phasors
                           1,  # Number of analog values
                           1,  # Number of digital status words
                           ["VA", "ANALOG1", "BREAKER 1 STATUS",
                            "BREAKER 2 STATUS", "BREAKER 3 STATUS", "BREAKER 4 STATUS", "BREAKER 5 STATUS",
                            "BREAKER 6 STATUS", "BREAKER 7 STATUS", "BREAKER 8 STATUS", "BREAKER 9 STATUS",
                            "BREAKER A STATUS", "BREAKER B STATUS", "BREAKER C STATUS", "BREAKER D STATUS",
                            "BREAKER E STATUS", "BREAKER F STATUS", "BREAKER G STATUS"],  # Channel Names
                           [(0, 'v')],  # Conversion factor for phasor channels - (float representation, not important)
                           [(1, 'pow')],  # Conversion factor for analog channels
                           [(0x0000, 0xffff)],  # Mask words for digital status words
                           60,  # Nominal frequency
                           1,  # Configuration change count
                           30)  # Rate of phasor data transmission)

        self.hf = HeaderFrame(self.pmu_idx[0],  # PMU_ID
                              "Hello I'm a MiniPMU!")  # Header Message

        self.pmu.set_configuration(self.cfg)
        self.pmu.set_header(self.hf)

        self.pmu.run()

    def find_var_idx(self):
        """Returns a dictionary of the indices into Varheader based on
        `self.pmu_idx`. Items in `self.pmu_idx` uses 1-indexing.

        For example, if `self.pmu_idx` == [1, 2], this function will return
        the indices of
         - Idxvgs.Pmu.vm[0] and Idxvgs.Pmu.vm[1] as vm
         - Idxvgs.Pmu.am[0] and Idxvgs.Pmu.am[1] as am
         - Idxvgs.Bus.w_Busfreq[0] and Idxvgs.Bus.w_Busfreq[1] as w
        in the dictionary `self. var_idx` with the above fields.

        """
        for item in self.pmu_idx:
            self.var_idx['am'].append(self.Idxvgs['Pmu']['am'][0, item - 1])
            self.var_idx['vm'].append(self.Idxvgs['Pmu']['vm'][0, item - 1])
            self.var_idx['w'].append(self.Idxvgs['Bus']['w_Busfreq'][0, item - 1])

    @property
    def vgsvaridx(self):
        return self.var_idx['am'] + self.var_idx['vm'] + self.var_idx['w']

    def init_storage(self):
        """Initialize data storage `self.t` and `self.data`
        """
        if self.count % self.max_store == 0:
            self.t = ndarray(shape=(self.max_store, 1), dtype=float)
            self.data = ndarray(shape=(self.max_store, len(self.pmu_idx)), dtype=float)
            self.count = 0
            return True
        else:
            return False

    def sync_measurement_data(self):
        """Store synced data into self.data and return in a tuple of (t, values)
        """
        self.init_storage()

        var = self.dimec.sync()
        ws = self.dimec.workspace

        if var == 'Varvgs':
            self.data[self.count, :] = ws[var]['vars']
            self.t[self.count, :] = ws[var]['t']
            self.count += 1
            return ws[var]['t'], ws[var]['vars']
        else:
            return None, None

    def sync_initialization(self):
        """Sync for ``SysParam``, ``Idxvgs`` and ``Varheader`` until all are received
        """
        logging.info('Syncing for SysParam, Idxvgs and Varheader')
        ret = False
        count = 0
        while True:
            var = self.dimec.sync()
            if var is False:
                time.sleep(0.05)
                continue
            if var in ('SysParam', 'Idxvgs', 'Varheader'):
                self.__dict__[var] = self.dimec.workspace[var]
                count += 1
                logging.info('{} synced.'.format(var))
            if count == 3:
                ret = True
                break

        return ret

    def run(self):
        """Process control function
        """
        self.start_dime()
        if self.sync_initialization():
            self.respond_to_sim()
        self.find_var_idx()
        self.get_bus_name()
        self.config_pmu()

        while True:
            t, var = self.sync_measurement_data()
            # if t:
            if self.pmu.clients:
                self.pmu.send_data(phasors=[(var[1], var[0])],
                                   analog=[9.99],
                                   digital=[0x0001],
                                   freq=var[2])
            # else:
            #     time.sleep(0.005)

        self.pmu.join()


if __name__ == "__main__":
    mini = MiniPMU(name='TestPMU', dime_address='ipc:///tmp/dime', pmu_idx=[1],
                   pmu_port=1412)
    mini.run()
