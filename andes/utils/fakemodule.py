from andes.utils import dime
from numpy import array, arange


class FakeModule(object):
    def __init__(self,
                 name,
                 address='tcp://127.0.0.1:5000',
                 varout=None,
                 varname=None,
                 idx=None):
        self.name = name
        self.dimec = dime.Dime(name, address)
        self.varout = varout
        self.varname = varname
        self.idx = idx
        self.set_header()
        self.dimec.start()

    def set_header(self):
        if not self.idx:
            return
        varheader = self.varname.unamex + self.varname.unamey
        self.header = [varheader[i] for i in self.idx]

    def set_outgoing(self):
        if not self.idx:
            return
        all_values = self.varout.vars[-1]
        self.outgoing = {
            'vars': array([all_values[i] for i in self.idx]),
            't': self.varout.t[-1],
        }

    def send_header(self):
        self.dimec.send_var('geovis', self.name + '_header', self.header)

    def send_idx(self):
        pass

    def send_init(self):
        pass

    def init_on_geovis(self):
        self.send_header()
        self.send_idx()

    def stream_to_geovis(self):
        self.set_outgoing()
        self.dimec.send_var('geovis', self.name + '_vars', self.outgoing)
        print('Module info sent at t = {}'.format(self.outgoing['t']))


class EAGC(FakeModule):
    def send_idx(self):
        idx = {
            'ACE': arange(1, 4).reshape((1, 3)),
            'P_Area': arange(4, 8).reshape((1, 4)),
        }
        self.dimec.send_var('geovis', self.name + '_idx', idx)

    def send_init(self):
        EAGC = {
            'params': ['Bus'],
            'vgsvaridx': arange(1, 4).reshape((1, 3)),
            'limitsample': 10,
            'usepmu': 1,
        }
        self.dimec.broadcast('EAGC', EAGC)


# for use in System class

# def hack_EAGC(self):
#     self.EAGC_module = EAGC(name='EAGC',
#                             address=self.Config.dime_server,
#                             idx=[671, 672, 673, 1154, 1155, 1156, 1157],
#                             varout=self.varout, varname=self.varname)
#     self.EAGC_module.init_on_geovis()
