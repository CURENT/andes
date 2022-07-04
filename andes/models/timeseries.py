"""
Model for metadata of timeseries.
"""

import os
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

        self.path = DataParam(mandatory=True, info='Path to timeseries xlsx file.')
        self.sheet = DataParam(mandatory=True, info='Sheet name to use')
        self.fields = NumParam(mandatory=True,
                               info='comma-separated field names in timeseries data',
                               iconvert=str_list_iconv,
                               oconvert=str_list_oconv,
                               vtype=object,
                               )

        self.tkey = DataParam(default='t', info='Key for timestamps')

        self.model = DataParam(info='Model to link to', mandatory=True)
        self.dev = IdxParam(info='Idx of device to link to', mandatory=True)
        self.dests = NumParam(mandatory=True,
                              info='comma-separated device fields as destinations',
                              iconvert=str_list_iconv,
                              oconvert=str_list_oconv,
                              vtype=object,
                              )


class TimeSeriesModel(Model):
    """
    Implementation of TimeSeries.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        # Notes:
        # TimeSeries model is not used in power flow for now

        self.group = 'DataSeries'
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

        # TODO: timeseries file must exist for setup to pass. Consider moving
        # the file reading to a later stage so that adding sheets to xlsx file can work
        # without the file existing.

        Model.list2array(self)

        # read and store data
        for ii in range(self.n):
            idx = self.idx.v[ii]
            path = self.path.v[ii]
            sheet = self.sheet.v[ii]

            if not os.path.isabs(path):
                path = os.path.join(self.system.files.case_path, path)

            if not os.path.exists(path):
                raise FileNotFoundError('<%s idx=%s>: File not found: "%s"',
                                        self.class_name, idx, path)

            # --- read supported formats ---
            if path.endswith("xlsx") or path.endswith("xls"):
                df = self._read_excel(path, sheet, idx)
            elif path.endswith("csv"):
                df = pd.read_csv(path)

            for field in self.fields.v[ii]:
                if field not in df.columns:
                    raise ValueError('Field {} not found in timeseries data'.format(field))

            self._data[idx] = df
            logger.info('Read timeseries data from "%s"', path)

    def _read_excel(self, path, sheet, idx):
        """
        Helper function to read excel file.
        """

        try:
            df = pd.read_excel(path, sheet_name=sheet)
            return df
        except ValueError as e:
            logger.error('<%s idx=%s>: Sheet not found: "%s" in "%s"',
                         self.class_name, idx, sheet, path)
            raise e

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
            if t not in df[tkey].values:
                continue

            fields = self.fields.v[ii]
            dests = self.dests.v[ii]

            model = self.model.v[ii]
            dev_idx = self.dev.v[ii]

            # apply the value change
            for field, dest in zip(fields, dests):
                value = df.loc[df[tkey] == t, field].values
                if len(value) == 0:
                    continue
                value = value[0]
                self.system.__dict__[model].set(dest, dev_idx, 'v', value)

                if not self.config.silent:
                    tqdm.write("<TimeSeries %s> set %s=%g for %s.%s at t=%g" %
                               (idx, dest, value, model, dev_idx, t))

    def apply_interpolate(self, t):
        """
        Apply timeseries data at the interpolated time.
        """

        raise NotImplementedError

    def init(self, routine):
        """
        Set values for the very first time step.
        """

        Model.init(self, routine)

        self.apply_exact(np.array(self.system.TDS.config.t0))
        logger.debug('<%s>: Initialization done', self.class_name)


class TimeSeries(TimeSeriesData, TimeSeriesModel):
    """
    Model for applying time-series data.

    A TimeSeries device takes a `xlsx` data spreadsheet and applies the data to
    the specified device. The spreadsheet can contain multiple sheets, each with
    a column named ``t`` and multiple user-defined columns for the data. The
    values will be applied at the exact time instant.

    The ``xlsx`` data spreadsheet is assumed in the same folder as the case
    file.

    Regarding the parameters for the ``TimeSeries`` device:

    - The column names in the ``xlsx`` data file need to be specified through
      the ``fields`` parameter, separated by commas.
    - The parameter/service names of the device which is to be updated need to
      be specified through the ``dests`` parameter, separated by commas.

    There are a few caveats with the current TimeSeries implementation:

    - TimeSeries will not be applied power flow.
    - The interpolation mode has yet to be implemented.

    """

    def __init__(self, system, config):
        TimeSeriesData.__init__(self)
        TimeSeriesModel.__init__(self, system, config)
