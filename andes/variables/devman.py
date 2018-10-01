from ..models import order, all_models
from numpy import ndarray
from cvxopt import matrix
import logging

logger = logging.getLogger(__name__)


class DevMan(object):
    """
    Device Manager class.
    Maintains the loaded model list, groups and categories

    """

    def __init__(self, system=None):
        """constructor for devman class"""
        self.system = system
        self.devices = []
        self.group = {}

    def register_device(self, dev_name):
        """register a device to the device list"""
        if dev_name not in self.devices:
            self.devices.append(dev_name)
        group_name = self.system.__dict__[dev_name]._group
        if group_name not in self.group.keys():
            self.group[group_name] = {}

    def register_element(self, dev_name, idx=None):
        """
        Register a device element to the group list

        Parameters
        ----------
        dev_name : str
            model name
        idx : str
            element idx

        Returns
        -------
        str
            assigned idx
        """
        if dev_name not in self.devices:
            logger.error(
                'Device {} missing. call add_device before adding elements'.
                format(dev_name))
            return
        group_name = self.system.__dict__[dev_name]._group
        if idx is None:  # "if not idx" will fail for idx==0.0
            idx = dev_name + '_' + str(len(self.group[group_name].keys()))
        self.group[group_name][idx] = dev_name
        return idx

    def sort_device(self):
        """
        Sort device to follow the order of initialization

        :return: None
        """

        self.devices.sort()
        # idx: the indices of order-sensitive models
        # names: an ordered list of order-sensitive models
        idx = []
        names = []
        for dev in order:
            # if ``dev`` in ``order`` is a model file name:
            #   initialize the models in alphabet order
            if dev in all_models:
                all_dev = list(sorted(all_models[dev].keys()))
                for item in all_dev:
                    if item in self.devices:
                        idx.append(self.devices.index(item))
                        names.append(item)

            # if ``dev`` presents as a model name
            elif dev in self.devices:
                idx.append(self.devices.index(dev))
                names.append(dev)

        idx = sorted(idx)
        for id, name in zip(idx, names):
            self.devices[id] = name

    def get_param(self, group, param, fkey):
        ret = []
        ret_list = False
        if type(fkey) == matrix:
            fkey = list(fkey)
        elif type(fkey) == ndarray:
            fkey = fkey.tolist()

        for key, item in self.group.items():
            if key != group:
                continue
            if type(fkey) != list:
                fkey = [fkey]
            else:
                ret_list = True

            for k in fkey:
                for name, dev in item.items():
                    if name == k:
                        int_id = self.system.__dict__[dev].uid[name]
                        ret.append(
                            self.system.__dict__[dev].__dict__[param][int_id])
                        continue
            if not ret_list:
                ret = ret[0]

        return ret
