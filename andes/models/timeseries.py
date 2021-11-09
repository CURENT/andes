"""
Model for metadata of timeseries.
"""

import logging

from collections import OrderedDict

from andes.core.model import ModelData, Model  # noaq
from andes.core.param import DataParam, IdxParam, NumParam  # noqa
from andes.core.discrete import Switcher
from andes.shared import pd, np, tqdm

logger = logging.getLogger(__name__)


def str_list_iconv(x):
    """
    Helper function to convert a string or a list of strings into a numpy array.
    """
    if isinstance(x, str):
        x = x.split(',')
        x = [item.strip() for item in x]

        return x

    raise NotImplementedError


def str_list_oconv(x):
    """
    Convert list into a list literal.
    """
    return ','.join(x)


class TimeSeriesData(ModelData):
    """
    Input data for metadata of timeseries.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.mode = NumParam(default='1',
                             info='Mode for applying timeseries. '
                                  '1: exact time, '
                                  '2: interpolated',
                             vrange=(1, 2),
                             )

        self.path = DataParam(mandatory=True, info='Path to timeseries xlsx file')
        self.sheet = DataParam(mandatory=True, info='Sheet name to use')
        self.fields = NumParam(mandatory=True,
                               info='comma-separated field names in timeseries data',
                               iconvert=str_list_iconv,
                               oconvert=str_list_oconv,
                               vtype=np.object,
                               )

        self.tkey = DataParam(default='t', info='Key for timestamps')

        self.model = DataParam(info='Model to link to', mandatory=True)
        self.dev = IdxParam(info='Idx of device to link to', mandatory=True)
        self.dests = NumParam(mandatory=True,
                              info='comma-separated device fields as destinations',
                              iconvert=str_list_iconv,
                              oconvert=str_list_oconv,
                              vtype=np.object,
                              )


class TimeSeriesModel(Model):
    """
    Implementation of TimeSeries.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.pflow = True
        self.flags.tds = True

        self.config.add(OrderedDict((('silent', 1),
                                     )))

        self.config.add_extra("_help",
                              silent="suppress output messages if is not zero",
                              )
        self.config.add_extra("_alt",
                              silent=(0, 1),
                              )

        self.SW = Switcher(self.mode, options=(0, 1, 2),
                           info='mode switcher', )

        self._data = OrderedDict()  # keys are the idx, and values are the dataframe

    def list2array(self):
        """
        Set internal storage for timeseries data.

        Open file and read data into internal storage.
        """

        Model.list2array(self)

        # read and store data
        for ii in range(self.n):
            idx = self.idx.v[ii]
            path = self.path.v[ii]
            sheet = self.sheet.v[ii]

            try:
                df = pd.read_excel(path, sheet_name=sheet)
            except FileNotFoundError as e:
                logger.error('<%s idx=%s>: File not found: "%s"',
                             self.class_name, idx, path)
                raise e
            except ValueError as e:
                logger.error('<%s idx=%s>: Sheet not found: "%s" in "%s"',
                             self.class_name, idx, sheet, path)
                raise e

            for field in self.fields.v[ii]:
                if field not in df.columns:
                    raise ValueError('Field {} not found in timeseries data'.format(field))

            self._data[idx] = df

    def get_times(self):
        """
        Gather simulation stop-at times for mode = 1.
        """

        Model.get_times(self)

        # collect all time stamps
        out = list()

        for ii in range(self.n):
            if self.SW.s1[ii] != 1:
                continue

            idx = self.idx.v[ii]
            df = self._data[idx]
            tkey = self.tkey.v[ii]

            out.append(df[tkey].to_numpy())

        return out

    def apply_exact(self, t):
        """
        Apply the timeseries data at the exact time.

        Parameters
        ----------
        t : float
            the current time
        """
        # convert from numpy scalar to float
        t = t.tolist()

        for ii in range(self.n):
            # skip offline devices
            if self.u.v[ii] == 0:
                continue

            # check mode
            if self.SW.s1[ii] != 1:
                continue

            idx = self.idx.v[ii]
            df = self._data[idx]
            tkey = self.tkey.v[ii]

            # check if current time is a valid time stamp
            if t not in df[tkey]:
                continue

            fields = self.fields.v[ii]
            dests = self.dests.v[ii]

            model = self.model.v[ii]
            dev_idx = self.dev.v[ii]

            # apply the value change
            for field, dest in zip(fields, dests):
                value = df.loc[df[tkey] == t, field].values[0]
                self.system.__dict__[model].set(dest, dev_idx, 'v', value)

                if not self.config.silent:
                    tqdm.write("<TimeSeries %s> set %s=%g for %s.%s at t=%g" %
                               (idx, dest, value, model, dev_idx, t))

    def apply_interpolate(self, t):
        """
        Apply timeseries data at the interpolated time.
        """

        raise NotImplementedError


class TimeSeries(TimeSeriesData, TimeSeriesModel):
    """
    Model for metadata of timeseries.

    TimeSeries will not overwrite values in power flow.
    """

    def __init__(self, system, config):
        TimeSeriesData.__init__(self)
        TimeSeriesModel.__init__(self, system, config)
