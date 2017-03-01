
class DevMan(object):
    """Device Manager class. Maintains the loaded model list, groups and categories"""
    def __init__(self, system=None):
        """constructor for DevMan class"""
        self.system = system
        self.device = []
        self.group = {}

    def add_device(self, dev_name):
        """register a device to the device list"""
        if dev_name not in self.device:
            self.device.append(dev_name)
        group_name = self.system.__dict__[dev_name]._group
        if group_name not in self.group.keys():
            self.group[group_name] = {}

    def add_element(self, dev_name, idx=None):
        """register a device element to the group list
        Args:
            dev_name: model name
            idx (optional): element external idx

        Returns:
            idx: assigned element index
            """
        if dev_name not in self.device:
            self.system.Log.error('Device {} missing. Call add_device before adding elements'.format(dev_name))
            return
        group_name = self.system.__dict__[dev_name]._group
        if not idx:
            idx = len(self.group[group_name].keys())
        self.group[group_name][idx] = dev_name
        return idx

    def sort_device(self):
        """sort device to meet device prerequisites (initialize devices before controllers)"""
        pass
