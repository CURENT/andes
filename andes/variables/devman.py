from ..models import order, jits, non_jits
from numpy import ndarray
from cvxopt import matrix

class DevMan(object):
    """Device Manager class. Maintains the loaded model list, groups and categories"""
    def __init__(self, system=None):
        """constructor for DevMan class"""
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
        """register a device element to the group list
        Args:
            dev_name: model name
            idx (optional): element external idx

        Returns:
            idx: assigned element index
            """
        if dev_name not in self.devices:
            self.system.Log.error('Device {} missing. Call add_device before adding elements'.format(dev_name))
            return
        group_name = self.system.__dict__[dev_name]._group
        if idx is None:  # "if not idx" will fail for idx==0.0
            idx = dev_name + '_' + str(len(self.group[group_name].keys()))
        self.group[group_name][idx] = dev_name
        return idx

    def sort_device(self):
        """sort device to meet device prerequisites (initialize devices before controllers)"""
        self.devices.sort()
        mapping = non_jits
        mapping.update(jits)
        idx = []
        names = []
        for dev in order:
            if dev in mapping.keys():
                all_dev = list(sorted(mapping[dev].keys()))
                for item in all_dev:
                    if item in self.devices:
                        idx.append(self.devices.index(item))
                        names.append(item)
            elif dev in self.devices:
                idx.append(self.devices.index(dev))
                names.append(dev)

        idx = sorted(idx)
        for id, name in zip(idx, names):
            self.devices[id] = name

    def swap_device(self, front, back):
        if front in self.devices and back in self.devices:
            m = self.devices.index(front)
            n = self.devices.index(back)
            if m > n:
                self.devices[n] = front
                self.devices[m] = back

    def get_param(self, group, param, fkey):
        ret = []
        ret_list = False
        if type(fkey) == matrix:
            fkey = list(fkey)
        elif type(fkey) == ndarray:
            fkey = fkey.tolist()

        for key, item in self.system.DevMan.group.items():
            if key != group:
                continue
            if type(fkey) != list:
                fkey = [fkey]
            else:
                ret_list = True

            for k in fkey:
                for name, dev in item.items():
                    if name == k:
                        int_id = self.system.__dict__[dev].int[name]
                        ret.append(self.system.__dict__[dev].__dict__[param][int_id])
                        continue
            if not ret_list:
                ret = ret[0]

        return ret
