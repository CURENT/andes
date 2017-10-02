from numpy import ndarray, array, concatenate, delete
from cvxopt import matrix
from . import dime
from time import sleep


class Streaming(object):
    """Data streaming class for LTB"""
    def __init__(self, system):
        self.system = system
        self.dimec = dime.Dime(system.Settings.dime_name, system.Settings.dime_server)
        self.params_built = False

        _dicts = ['SysParam', 'SysName', 'Idxvgs', 'ModuleInfo']
        _lists = ['Varheader', 'last_devices']

        for item in _dicts:
            self.__dict__[item] = dict()
        for item in _lists:
            self.__dict__[item] = list()

        if self.system.Settings.dime_enable:
            self.system.Log.debug('Trying to connect to dime server {}.'.format(system.Settings.dime_server))
            try:
                self.dimec.start()
                self.system.Log.debug('DiME connection established.')
            except:
                self.dimec.exit()
                self.dimec.start()
                self.system.Log.debug('DiME connection established.')

    def _build_SysParam(self):
        if self.system.Bus.n:
            params = ['idx', 'Vn', 'voltage', 'angle', 'area', 'region', 'xcoord', 'ycoord']
            data_list = self._build_list('Bus', params)
            self.SysParam.update({'Bus': array(data_list).T})

        if self.system.Line.n:
            params = ['bus1', 'bus2', 'Sn', 'Vn', 'fn', 0, 1, 'r', 'x', 'b', 'tap', 'phi', 0, 0, 0, 1]
            data_list = self._build_list('Line', params)
            self.SysParam.update({'Line': array(data_list).T})

        if self.system.PQ.n:
            params = ['bus', 'Sn', 'Vn', 'p', 'q', 'vmax', 'vmin', 1, 'u']
            data_list = self._build_list('PQ', params)
            self.SysParam.update({'PQ': array(data_list).T})

        if self.system.PV.n:
            params = ['bus', 'Sn', 'Vn', 'pg', 'v0', 'qmax', 'qmin', 'vmax', 'vmin', 0, 'u']
            data_list = self._build_list('PV', params)
            self.SysParam.update({'PV': array(data_list).T})

        if self.system.SW.n:
            params = ['bus', 'Sn', 'Vn', 'v0', 'a0', 'qmax', 'qmin', 'vmax', 'vmin', 'pg', 0, 1, 'u']
            data_list = self._build_list('SW', params)
            self.SysParam.update({'SW': array(data_list).T})

        if self.system.Shunt.n:
            params = ['bus', 'Sn', 'Vn', 'fn', 'g', 'b', 'u']
            data_list = self._build_list('Shunt', params)

            self.SysParam.update({'Shunt': array(data_list).T})

        if self.system.PMU.n:
            params = ['bus', 'Vn', 'fn', 'Tv', 'Ta', 'u']
            data_list = self._build_list('PMU', params)

            self.SysParam.update({'Pmu': array(data_list).T})

        if self.system.BusFreq.n:
            params = ['bus', 'Tf', 'Tw', 'u']
            data_list = self._build_list('BusFreq', params)

            self.SysParam.update({'Busfreq': array(data_list).T})

        if self.system.Syn2.n:
            syn_params = ['bus', 'Sn', 'Vn', 'fn', 2, 'xl', 'ra',
                          0, 'xd1', 0, 0, 0,
                          0, 0, 0, 0, 0,
                          'M', 'D', 0, 0,
                          'gammap', 'gammaq', 0,
                          0, 0, 'coi', 'u'
                          ]
            data_list2 = self._build_list('Syn2', syn_params)
            data_array2 = array(data_list2).T
        else:
            data_array2 = array([]).reshape(0, 28)

        if self.system.Syn6a.n:
            syn_params = ['bus', 'Sn', 'Vn', 60, 6, 'xl', 'ra',
                          'xd', 'xd1', 'xd2', 'Td10', 'Td20',
                          'xq', 'xq1', 'xq2', 'Tq10', 'Tq20',
                          'M', 'D', 0, 0,
                          'gammap', 'gammaq', 'Taa',
                          'S10', 'S12', 'coi', 'u']
            data_list6 = self._build_list('Syn6a', syn_params)
            data_array6 = array(data_list6).T
        else:
            data_array6 = array([]).reshape(0, 28)

        data_array = concatenate((data_array2, data_array6), axis=0)
        self.SysParam.update({'Syn': data_array})

        if self.system.AVR1.n or self.system.AVR2.n or self.system.AVR3.n:

            if self.system.AVR1.n:
                params = ['syn', 2, 'vrmax', 'vrmin', 'Ka', 'Ta', 'Kf', 'Tf', 'Ke', 'Te',
                          'Tr', 'Ae', 'Be', 'u']
                data_list_avr1 = self._build_list('AVR1', params)
                data_array_avr1 = array(data_list_avr1).T
            else:
                data_array_avr1 = array([]).reshape(0, 14)

            if self.system.AVR2.n:
                params = ['syn', 1, 'vrmax', 'vrmin', 'K0', 'T1', 'T2', 'T3', 'T4',
                          'Te', 'Tr', 'Ae', 'Be', 'u']
                data_list_avr2 = self._build_list('AVR2', params)
                data_array_avr2 = array(data_list_avr2).T
            else:
                data_array_avr2 = array([]).reshape(0, 14)

            if self.system.AVR3.n:
                params = ['syn', 3, 'vfmax', 'vfmin', 'K0', 'T2', 'T1', 1, 0, 'Te', 'Tr', 0, 0, 'u']
                data_list_avr3 = self._build_list('AVR3', params)
                data_array_avr3 = array(data_list_avr3).T

            else:
                data_array_avr3 = array([]).reshape(0, 14)

            data_array = concatenate((data_array_avr1, data_array_avr2, data_array_avr3), axis=0)
            self.SysParam.update({'Exc': data_array})

            # find bus of the Syn based on AVR.syn idx
            syn_bus = self.system.DevMan.get_param('Synchronous', param='bus', fkey=self.SysParam['Exc'][:, 0])
            # find position of the Syn based on Syn_bus
            self.SysParam['Exc'][:, 0] = self._find_pos('Syn', syn_bus)
            self.SysParam['Exc'][:, 0] += 1

        if self.system.TG1.n or self.system.TG2.n:
            if self.system.TG1.n:
                params = ['gen', 1, 'wref0', 'R', 'pmax', 'pmin', 'Ts', 'Tc', 'T3', 'T4', 'T5', 'u']
                data_list_tg1 = self._build_list('TG1', params)
                data_array_tg1 = array(data_list_tg1).T
            else:
                data_array_tg1 = array([]).reshape(0, 12)

            if self.system.TG2.n:
                params = ['syn', 2, 'wref0', 'R', 'pmax', 'pmin', 'T2', 'T1', 0, 0, 0, 'u']
                data_list_tg2 = self._build_list('TG2', params)
                data_array_tg2 = array(data_list_tg2).T
            else:
                data_array_tg2 = array([]).reshape(0, 12)

            data_array = concatenate((data_array_tg1, data_array_tg2), axis=0)
            self.SysParam.update({'Tg': data_array})

        if self.system.PSS1.n or self.system.PSS2.n:
            if self.system.PSS1.n:
                params = ['avr', 1, 'Ic1', 'vcu', 'vcl', 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                data_list_pss1 = self._build_list('PSS1', params)
                data_array_pss1 = array(data_list_pss1).T
            else:
                data_array_pss1 = array([]).reshape(0, 23)

            if self.system.PSS2.n:
                params = ['avr', 2, 'Ic', 'vcu', 'vcl', 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                data_list_pss2 = self._build_list('PSS2', params)
                data_array_pss2 = array(data_list_pss2).T
            else:
                data_array_pss2 = array([]).reshape(0, 23)

            data_array = concatenate((data_array_pss1, data_array_pss2), axis=0)
            self.SysParam.update({'Pss': data_array})

            avr_gen = self.system.DevMan.get_param('AVR', param='syn', fkey=self.SysParam['Pss'][:, 0])
            self.SysParam['Pss'][:, 0] = self._find_pos('Exc', avr_gen)
            self.SysParam['Pss'][:, 0] += 1

        if self.system.WTG3.n:
            params = ['bus', 'wind', 'Sn', 'Vn', 'fn', 'rs', 'xs', 'rr', 'xr', 'xmu', 'H',
                      'Kp', 'Tp', 'KV', 'Te', 'R', 'npole', 'nblade', 'ngb', 'pmax', 'pmin', 'qmax', 'qmin', 'u']
            data_list = self._build_list('WTG3', params)
            self.SysParam.update({'Dfig': array(data_list).T})

        if self.system.Node.n:
            """Idx, Vdcn, area, region, xcoord, ycoord"""
            params = ['idx', 'Vdcn', 'area', 'region', 'xcoord', 'ycoord']
            data_list = self._build_list('Node', params)
            self.SysParam.update({'Node': array(data_list).T})

        if self.system.R.n or self.system.C.n or self.system.L.n or \
                self.system.RCp.n or self.system.RCs.n or self.system.RLCp.n or self.system.RLCs.n or self.system.RLs.n:

            ground_idx = self.system.Ground.node

            data_array = array([]).reshape(0, 4)
            dev_id = {1: 'R', 2: 'C', 3: 'L', 4: 'RCp',
                      5: 'RCs', 6: 'RLCp', 7: 'RLCs', 8: 'RLs'}
            for id, dev in dev_id.items():
                if self.system.__dict__[dev].n:
                    params = ['node1', 'node2', id, 'u']
                    data_list = self._build_list(dev, params)
                    data_array = concatenate((data_array, array(data_list).T), axis=0)
            to_delete = []
            for row in range(data_array.shape[0]):
                if data_array[row, 0] in ground_idx or data_array[row, 1] in ground_idx:
                    to_delete.append(row)
            data_array = delete(data_array, tuple(to_delete), axis=0)

            self.SysParam.update({'DCLine': data_array})

    def _build_SysName(self):
        self.SysName['Bus'] = self.system.Bus.name
        if self.system.Area.n:
            self.SysName['Areas'] = self.system.Area.name
        if self.system.Region.n:
            self.SysName['Regions'] = self.system.Region.name
        if self.system.Node.n:
            self.SysName['Node'] = self.system.Node.name

    def _build_Varheader(self):
        self.Varheader = self.system.VarName.unamex + self.system.VarName.unamey

    def _build_Idxvgs(self):
        m = self.system.DAE.m
        n = self.system.DAE.n
        mn = m + n

        self.Idxvgs['System'] = {'nBus': self.system.Bus.n,
                                 'nLine': self.system.Line.n,
                                 }
        self.Idxvgs['Bus'] = {'theta': 1 + n + array(self.system.Bus.a),
                              'V': 1 + n + array(self.system.Bus.v),
                              'w_Busfreq': 1 + array(self.system.BusFreq.w),
                              'P': 1 + mn + array(range(self.system.Bus.n)),
                              'Q': 1 + mn + self.system.Bus.n + array(range(self.system.Bus.n)),
                              }
        line0 = 1 + mn + 2 * self.system.Bus.n
        self.Idxvgs['Line'] = {'Pij': line0 + array(range(self.system.Line.n)),
                               'Pji': line0 + self.system.Line.n + array(range(self.system.Line.n)),
                               'Qij': line0 + 2 * self.system.Line.n + array(range(self.system.Line.n)),
                               'Qji': line0 + 3 * self.system.Line.n + array(range(self.system.Line.n)),
                               # 'Iij': array([]),
                               # 'Iji': array([]),
                               # 'Sij': array([]),
                               # 'Sji': array([]),
                               }
        self.Idxvgs['Syn'] = {'delta': 1 + array(self.system.Syn6a.delta),
                              'omega': 1 + array(self.system.Syn6a.omega),
                              'e1d': 1 + array(self.system.Syn6a.e1d),
                              'e1q': 1 + array(self.system.Syn6a.e1q),
                              'e2d': 1 + array(self.system.Syn6a.e2d),
                              'e2q': 1 + array(self.system.Syn6a.e2q),
                              'psid': array(self.system.Syn6a.psid),
                              'psiq': array(self.system.Syn6a.psiq),
                              'p': array(self.system.Syn6a.p),
                              'q': array(self.system.Syn6a.q),
                              }
        self.Idxvgs['Tg'] = {'pm': array(self.system.TG1.pout + self.system.TG2.pout),
                             'wref': array(self.system.TG1.wref + self.system.TG2.wref),
                             }
        self.Idxvgs['Exc'] = {'vf': array(self.system.AVR1.vfout + self.system.AVR2.vfout + self.system.AVR3.vfout),
                              'vm': array(self.system.AVR1.vm + self.system.AVR2.vm + self.system.AVR3.vm),
                              }
        if self.system.WTG3.n:
            self.Idxvgs['Dfig'] = {'omega_m': array(self.system.WTG3.omega_m),
                                   'theta_p': array(self.system.WTG3.theta_p),
                                   'idr': array(self.system.WTG3.ird),
                                   'iqr': array(self.system.WTG3.irq),
                                   }
        if self.system.Node.n:
            self.Idxvgs['Node'] = {'v': array(self.system.Node.v)}

        dev_id = {1: 'R', 2: 'C', 3: 'L', 4: 'RCp',
                  5: 'RCs', 6: 'RLCp', 7: 'RLCs', 8: 'RLs'}
        if 'DCLine' in self.SysParam:
            DCLine_types = set(self.SysParam['DCLine'][:, 2])
            idx = []
            for item in DCLine_types:
                item = int(item)
                idx.extend(self.system.__dict__[dev_id[item]].Idc)
            self.Idxvgs['DCLine'] = {'Idc': array(idx)}
        else:
            DCLine_types = ()
            # self.Idxvgs['DCLine'] = {}


    def _build_list(self, model, params, ret=None):
        if not ret:
            ret = []
        else:
            ret = list(ret)

        for p in params:
            if type(p) in (int, float):
                ret.append([p] * len(ret[0]))
            elif type(p) == list:
                assert len(p) == len(ret[0])
                ret.append(p)
            else:
                ret.append(list(self.system.__dict__[model].__dict__[p]))

        return ret

    def _find_pos(self, model, fkey, src_col=0):
        """Find the positions of foreign keys in the source model index list"""
        if type(fkey) == ndarray:
            fkey = fkey.tolist()
        elif type(fkey) in (int, float):
            fkey = [fkey]

        ret = []
        model_idx_list = self.SysParam[model][:, src_col].tolist()
        for item in fkey:
            ret.append(model_idx_list.index(item) if item in model_idx_list else 0)

        return ret

    def build_init(self):
        """Build `Varheader`, `Idxvgs` and `SysParam` after power flow routine"""
        self._build_SysParam()
        self._build_SysName()
        self._build_Idxvgs()
        self._build_Varheader()

    def send_init(self, recepient='all'):
        """Broadcast `Varheader`, `Idxvgs` and `SysParam` to all DiME clients after power flow routine"""
        if not self.system.Settings.dime_enable:
            return
        if not self.params_built:
            self.build_init()
        if recepient == 'all':
            self.system.Log.debug('Broadcasting Varheader, Idxvgs, SysParam and SysName...')
            sleep(0.5)
            self.dimec.broadcast('Varheader', self.Varheader)
            sleep(0.5)
            self.dimec.broadcast('Idxvgs', self.Idxvgs)
            sleep(0.5)
            try:
                self.dimec.broadcast('SysParam', self.SysParam)
                self.dimec.broadcast('SysName', self.SysName)
            except:
                self.system.Log.warning('SysParam or SysName broadcast error. Check bus coordinates.')
            sleep(0.5)
        else:
            if type(recepient) != list:
                recepient = [recepient]
            for item in recepient:
                self.dimec.send_var(item, 'Varheader', self.Varheader)
                self.dimec.send_var(item, 'Idxvgs', self.Idxvgs)
                self.dimec.send_var(item, 'SysParam', self.SysParam)
                self.dimec.send_var(item, 'SysName', self.SysName)

    def record_module_init(self, module_name, init_var):
        """Record the variable requests from modules"""
        if module_name in self.ModuleInfo:
            self.ModuleInfo[module_name].update(init_var)
        else:
            self.ModuleInfo[module_name] = init_var

        if type(self.ModuleInfo[module_name]['vgsvaridx']) == int:
            self.ModuleInfo[module_name]['vgsvaridx'] = array(self.ModuleInfo[module_name]['vgsvaridx'])

        vgsvaridx = self.ModuleInfo[module_name]['vgsvaridx'].tolist()
        if type(vgsvaridx[0]) == list:
            self.ModuleInfo[module_name]['vgsvaridx'] = vgsvaridx[0]
        else:
            self.ModuleInfo[module_name]['vgsvaridx'] = vgsvaridx

        self.ModuleInfo[module_name]['vgsvaridx'] = [int(i)-1 for i in self.ModuleInfo[module_name]['vgsvaridx']]

    def handle_event(self, Event):
        """Handle Fault, Breaker, Syn and Load Events"""
        pass

    def sync_and_handle(self):
        """Sync until the queue is empty"""
        if not self.system.Settings.dime_enable:
            return

        while True:
            var_name = self.dimec.sync()
            if not var_name:
                break

            current_devices = self.dimec.get_devices()
            workspace = self.dimec.workspace
            var_value = workspace[var_name]

            # send Varheader, SysParam and Idxvgs to modules on the fly
            if set(current_devices) != set(self.last_devices):
                new_devices = list(current_devices)
                new_devices.remove('sim')
                for item in self.last_devices:
                    new_devices.remove(item)
                self.send_init(new_devices)

                self.last_devices = current_devices

            if var_name in current_devices:
                self.record_module_init(var_name, var_value)

            elif var_name == 'Event':
                self.handle_event(var_value)

            else:
                self.system.Log.warning('Synced variable {} not handled'.format(var_name))

    def vars_to_modules(self):
        """Stream the last results to the modules"""
        if not self.system.Settings.dime_enable:
            return

        for mod in self.ModuleInfo.keys():
            idx = self.ModuleInfo[mod]['vgsvaridx']
            values = self.system.VarOut.vars[-1][idx]
            t = self.system.VarOut.t[-1]
            k = self.system.VarOut.k[-1]
            Varvgs = {'t': t,
                      'k': k,
                      'vars': array(values).T,
                      'accurate': array(values).T,
                      }
            self.dimec.send_var(mod, 'Varvgs', Varvgs)
