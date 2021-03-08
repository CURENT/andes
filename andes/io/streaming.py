import logging
from time import sleep
import numpy as np
from numpy import ndarray, array

logger = logging.getLogger(__name__)

try:
    from dime import DimeClient
except ImportError:
    logger.debug("Dime import failed.")


class Streaming:
    """
    ANDES data streaming class to interface with CURENT LTB.
    """

    def __init__(self, system):
        self.system = system

        self.params_built = False
        self.SysParam = dict()
        self.SysName = dict()
        self.Idxvgs = dict()
        self.ModuleInfo = dict()
        self.Varheader = list()
        self.last_devices = list()
        self.has_pmu = False
        self.dimec = None

    def connect(self):
        """
        Connect to DiME 2 server.

        If ``dime_address`` is specified from the command-line,
        streaming will be automatically enabled.
        Otherwise, settings from the Config file will be used.
        """
        config = self.system.config
        options = self.system.options

        # enable only when both arguments are supplied
        if options.get("dime_address") is not None:

            config.dime_enabled = True
            config.dime_address = options.get("dime_address")

        if not config.dime_enabled:
            return False

        try:
            self.dimec = DimeClient(config.dime_address)
            self.dimec.join(config.dime_name)
            logger.info('Dime connection to "%s" was successful.', config.dime_address)
            return True
        except NameError:
            logger.error('Dime not installed. Set System config `dime_enabled` to `0` to suppress warning.')
            self.system.config.dime_enabled = False

        except FileNotFoundError:
            logger.error('Dime sever not found at "%s".', config.dime_address)
            self.system.config.dime_enabled = False

        return False

    def _build_SysParam(self):
        self.SysParam = self.system.as_dict(vin=True, skip_empty=True)
        self.params_built = True

    def _build_SysName(self):
        self.SysName['Bus'] = self.system.Bus.name.v
        if self.system.Area.n:
            self.SysName['Areas'] = self.system.Area.name.v

    def _build_Varheader(self):
        self.Varheader = self.system.dae.xy_name

    def _build_Idxvgs(self):
        m = self.system.dae.m
        n = self.system.dae.n
        mn = m + n  # NOQA

        self.Idxvgs['System'] = {
            'nBus': self.system.Bus.n,
            'nLine': self.system.Line.n,
        }
        self.Idxvgs['Bus'] = {
            'theta': 1 + n + self.system.Bus.a.a,
            'V': 1 + n + self.system.Bus.v.a,
            'w_Busfreq': 1 + n + self.system.BusFreq.f.a,
            # NO LONGER SUPPORTED
            # 'P': 1 + mn + array(range(self.system.Bus.n)),
            # 'Q': 1 + mn + self.system.Bus.n + array(range(self.system.Bus.n)),
        }
        self.Idxvgs['Pmu'] = {
            # NOT YET SUPPORTED
            'vm': 1 + self.system.PMU.vm.a,
            'am': 1 + self.system.PMU.am.a,
        }

        # NOT YET SUPPORTED
        # line0 = 1 + mn + 2 * self.system.Bus.n
        self.Idxvgs['Line'] = {
            # 'Pij': line0 + array(range(self.system.Line.n)),
            # 'Pji': line0 + self.system.Line.n + array(range(self.system.Line.n)),
            # 'Qij': line0 + 2 * self.system.Line.n + array(range(self.system.Line.n)),
            # 'Qji': line0 + 3 * self.system.Line.n + array(range(self.system.Line.n)),
        }

        self.Idxvgs['Syn'] = {
            'delta': 1 + np.append(self.system.GENCLS.delta.a, self.system.GENROU.delta.a),
            'omega': 1 + np.append(self.system.GENCLS.omega.a, self.system.GENROU.omega.a),
            'e1d': 1 + np.append([0] * self.system.GENCLS.n, self.system.GENROU.e1d.a),
            'e1q': 1 + np.append([0] * self.system.GENCLS.n, self.system.GENROU.e1q.a),
            'e2d': 1 + np.append([0] * self.system.GENCLS.n, self.system.GENROU.e2d.a),
            'e2q': 1 + np.append([0] * self.system.GENCLS.n, self.system.GENROU.e2q.a),
            'psid': 1 + np.append([0] * self.system.GENCLS.n, self.system.GENROU.psid.a),
            'psiq': 1 + np.append([0] * self.system.GENCLS.n, self.system.GENROU.psiq.a),
            # NOT SUPPORTED
            # 'p': 1 + n + array([0] * self.system.GENCLS.n + self.system.GENROU.p.a),
            # 'q': 1 + n + array([0] * self.system.GENCLS.n + self.system.GENROU.q.a),
        }
        self.Idxvgs['Tg'] = {
            'pm': 1 + n + self.system.TG2.pout.a,
            'wref': 1 + n + self.system.TG2.wref.a,
        }
        self.Idxvgs['Exc'] = {
            # NOT YET READY
            # 'vf':
            # 1 + n + array(self.system.AVR1.vfout + self.system.AVR2.vfout +
            #               self.system.AVR3.vfout),
            # 'vm':
            # 1 + array(self.system.AVR1.vm + self.system.AVR2.vm +
            #           self.system.AVR3.vm),
        }
        # NOT YET READY

        # if self.system.WTG3.n:
        #     self.Idxvgs['Dfig'] = {
        #         'omega_m': 1 + array(self.system.WTG3.omega_m),
        #         'theta_p': 1 + array(self.system.WTG3.theta_p),
        #         'idr': 1 + array(self.system.WTG3.ird),
        #         'iqr': 1 + array(self.system.WTG3.irq),
        #     }
        # if self.system.Node.n:
        #     self.Idxvgs['Node'] = {'v': 1 + n + array(self.system.Node.v)}
        #
        # dev_id = {
        #     1: 'R',
        #     2: 'C',
        #     3: 'L',
        #     4: 'RCp',
        #     5: 'RCs',
        #     6: 'RLCp',
        #     7: 'RLCs',
        #     8: 'RLs'
        # }
        # if 'DCLine' in self.SysParam:
        #     DCLine_types = set(self.SysParam['DCLine'][:, 2])
        #     idx = []
        #     for item in DCLine_types:
        #         item = int(item)
        #         idx.extend(self.system.__dict__[dev_id[item]].Idc)
        #     self.Idxvgs['DCLine'] = {'Idc': 1 + array(idx)}
        # else:
        #     DCLine_types = ()
        #     # self.Idxvgs['DCLine'] = {}

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
                val = list(self.system.__dict__[model].__dict__[p])
                # make sure val does not contain list
                if isinstance(val[0], list):
                    logger.warning('{}.{} contains list. Reset to zeros.'.format(model, p))
                    val = [0] * len(val)
                ret.append(val)

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
            ret.append(
                model_idx_list.index(item) if item in model_idx_list else 0)

        return ret

    def build_init(self):
        """
        Build `Varheader`, `Idxvgs` and `SysParam` after power flow routine

        """
        self._build_SysParam()
        self._build_SysName()
        self._build_Idxvgs()
        self._build_Varheader()

    def send_init(self, recepient='all'):
        """
        Broadcast `Varheader`, `Idxvgs` and `SysParam`
        to all DiME clients after power flow routine
        """
        if not self.system.config.dime_enabled:
            return
        if not self.params_built:
            self.build_init()

        if recepient == 'all':
            self.last_devices = self.dimec.devices()

            logger.debug('Connected modules are: ' +
                         ','.join(self.dimec.devices()))
            logger.debug(
                'Broadcasting Varheader, Idxvgs, SysParam and SysName...')

            sleep(0.05)
            self.dimec.broadcast_r(Varheader=self.Varheader)

            sleep(0.05)
            self.dimec.broadcast_r(Idxvgs=self.Idxvgs)

            sleep(0.05)
            try:
                self.dimec.broadcast_r(SysParam=self.SysParam)
                self.dimec.broadcast_r(SysName=self.SysName)
            except:  # NOQA
                logger.warning(
                    'SysParam or SysName broadcast error.'
                    ' Check bus coordinates.'
                )
            sleep(0.5)
        else:
            if type(recepient) != list:
                recepient = [recepient]
            for item in recepient:
                self.dimec.send_r(item, Varheader=self.Varheader)
                self.dimec.send_r(item, Idxvgs=self.Idxvgs)
                self.dimec.send_r(item, SysParam=self.SysParam)
                self.dimec.send_r(item, SysName=self.SysName)

    def record_module_init(self, name, init_var):
        """
        Record the variable requests from modules
        """
        ivar = dict(init_var)
        var_idx = ivar['vgsvaridx']
        ivar['lastk'] = 0

        if name not in self.ModuleInfo:
            self.ModuleInfo[name] = {}

        if isinstance(var_idx, int):
            var_idx = array(var_idx, dtype=int)
        elif isinstance(var_idx, ndarray):
            var_idx = var_idx.tolist()
            # unwrap if nested
            if isinstance(var_idx[0], list):
                var_idx = array(var_idx[0], dtype=int)
            else:
                var_idx = array(var_idx, dtype=int)

        ivar['vgsvaridx'] = (var_idx - 1).tolist()
        ivar['lastk'] = 0

        self.ModuleInfo[name].update(ivar)

        logger.debug('Module <%s> requests index %s', name, var_idx)

    @staticmethod
    def transpose_matlab_row(a):
        if type(a) is ndarray:
            if a.shape[0] == 1:
                a = a[0]
        return a

    def handle_alter(self, Alter):
        """Handle parameter altering"""
        pass

    def handle_event(self, Event):
        """Handle Fault, Breaker, Syn and Load Events"""
        fields = ('name', 'id', 'action', 'time', 'duration')
        for key in fields:
            if key not in Event:
                logger.warning(
                    'Event has missing key {}.'.format(key))
                return

        names = self.transpose_matlab_row(Event.get('name'))
        idxes = self.transpose_matlab_row(Event.get('id'))
        actions = self.transpose_matlab_row(Event.get('action'))
        times = self.transpose_matlab_row(Event.get('time'))
        durations = self.transpose_matlab_row(Event.get('duration'))

        n = len(names)
        for i in range(n):
            try:
                name = names[i]
                idx = idxes[i]
                action = actions[i]  # NOQA
                time = times[i]
                duration = durations[i]
            except IndexError:
                logger.warning(
                    'Event key values might have different lengths.')
                continue

            if time == -1:
                time = max(self.system.dae.t, 0) + self.system.tds.config.tstep

            tf = time + duration
            if duration == 0.:
                tf = 9999

            if name.lower() == 'bus':
                param = {'tf': time, 'tc': tf, 'bus': idx}
                self.system.Fault.insert(**param)
                logger.debug(
                    'Event <Fault> added for bus %s at t = %.6g and tf = %g',
                    idx, time, tf)
            elif name.lower() == 'line':
                bus = self.system.Line.get_field(
                    'bus1', ['Line_' + str(int(idx - 1))])[0]
                param = {
                    'line': 'Line_' + str(idx - 1),
                    'bus': bus,
                    't1': time,
                    't2': tf,
                    'u1': 1,
                    'u2': 1 if duration else 0,
                }
                self.system.Breaker.insert(**param)
                logger.debug(
                    'Event <Breaker> added for line %s at t = %.6g and tf = %g',
                    idx, time, tf)

            self.system.call.build_vec()
            self.system.call._compile_int()
            self.system.dae.rebuild = True

    def sync_and_handle(self):
        """
        Sync until the queue is empty. Handle sync'ed commands.
        """
        if not self.system.config.dime_enabled:
            return
        current_devices = self.dimec.devices()

        # record MiniPMU
        if not self.has_pmu:
            for item in current_devices:
                if item.startswith('PMU_'):
                    self.has_pmu = True

        # send Varheader, SysParam and Idxvgs to modules on the fly
        if set(current_devices) != set(self.last_devices):
            new_devices = list(current_devices)
            new_devices.remove(self.system.config.dime_name)
            for item in self.last_devices:
                if item in new_devices:
                    new_devices.remove(item)
            self.send_init(new_devices)

            self.last_devices = current_devices

        while True:
            var_names = self.dimec.sync(1)

            if not var_names:
                break

            workspace = self.dimec.workspace

            for var_name in var_names:

                var_value = workspace[var_name]

                if var_name in current_devices:
                    self.record_module_init(var_name, var_value)

                elif var_name == 'Event':
                    self.handle_event(var_value)

                else:
                    logger.warning(
                        'Synced variable {} not handled'.format(var_name))

    def vars_to_pmu(self):
        """
        Broadcast all PMU measurements and BusFreq measurements
        in the variable `pmudata`

        """
        if not self.system.config.dime_enabled:
            return
        if not self.has_pmu:
            return

        idx = np.concatenate((self.system.PMU.vm.a,
                              self.system.PMU.am.a,
                              self.system.dae.n + self.system.BusFreq.f.a,
                              ))

        t = self.system.dae.t.tolist()
        k = 0  # field `k` is not no use

        values = self.system.dae.xy[idx]   # a 1-d array as opposed to a N-by-1 2-d matrix

        pmudata = {
            't': t,
            'k': k,
            'vars': values,
        }
        self.dimec.broadcast_r(pmudata=pmudata)

    def vars_to_modules(self):
        """
        Stream the results from the last step to modules

        :return: None
        """
        if not self.system.config.dime_enabled:
            return

        for mod in self.ModuleInfo.keys():
            # skip PMU modules in this function. offload it to vars_to_pmu()
            if mod.startswith('PMU_'):
                continue

            limitsample = self.ModuleInfo[mod].get('limitsample', 0)

            idx = self.ModuleInfo[mod]['vgsvaridx']
            t = self.system.dae.t.tolist()
            k = 0

            lastk = self.ModuleInfo[mod]['lastk']
            if limitsample:
                every = 1 / self.system.tds.config.tstep / limitsample
                if (k - lastk) / every < 1:
                    continue
                else:
                    self.ModuleInfo[mod]['lastk'] = k

            values = self.system.dae.xy[idx]

            Varvgs = {
                't': t,
                'k': k,
                'vars': values,
                'accurate': values,
            }

            self.dimec.send_r(mod, Varvgs=Varvgs)
            logger.debug("Send Varvgs to module <%s>", mod)

    def finalize(self):
        """
        Send ``DONE`` signal when simulation completes

        :return: None
        """
        if not self.system.config.dime_enabled:
            return

        self.system.streaming.dimec.broadcast_r(DONE=1)
        self.system.streaming.dimec.close()
