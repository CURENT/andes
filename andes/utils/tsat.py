import andes
from andes.shared import pd

andes.config_logger(stream_level=30)


def tsat_to_df(file):
    """
    Convert a TSAT XLS export to DataFrame.
    """
    return excel_to_df(file, offset_row=6)


def psse_to_df(file):
    """
    Convert a PSS/E xlsx export to DataFrame.
    """
    return excel_to_df(file, offset_row=1, offset_col=0, )


def excel_to_df(file, offset_row=6, offset_col=0):
    """
    Convert a XLS export to DataFrame

    TSAT offset is 6, PSS/E offset is 1
    """
    wb = pd.read_excel(file)

    df = wb.iloc[offset_row:, offset_col:]
    df.transpose()
    df.columns = ['Time', *range(len(df.columns) - 1)]
    return df


def plot_comparison(system, variable, tsat_data, ylabel, a=None,
                    tsat_header=None, a_tsat=None, scale=1,
                    psse_data=None, psse_header=None, a_psse=None,
                    show=True, left=None, right=None,
                    legend=True, title=None):
    """
    Plot and compare ANDES, TSAT and PSS/E results.
    """
    plot = system.TDS.plotter.plot
    plot_data = system.TDS.plotter.plot_data
    a_tsat = list(a) if a_tsat is None else a_tsat
    a_psse = list(a) if a_psse is None else a_psse

    fig, ax = plot(variable,
                   a=a,
                   ycalc=lambda x: scale * x,
                   ylabel=ylabel,
                   show=False,
                   line_styles=['-', '-.'],
                   left=left,
                   right=right,
                   legend=legend,
                   title=title,
                   )

    fig, ax = plot_data(tsat_data['Time'].to_numpy(),
                        tsat_data[a_tsat].to_numpy(),
                        xheader=['Time [s]'],
                        yheader=[tsat_header[i] for i in a_tsat] if tsat_header is not None else None,
                        fig=fig,
                        ax=ax,
                        line_styles=['--', ':'],
                        left=left,
                        right=right,
                        legend=legend,
                        show=show,
                        title=title,
                        )
    if psse_data is not None:
        fig, ax = plot_data(
            psse_data['Time'].to_numpy(),
            psse_data[a_psse].to_numpy(),
            xheader=['Time [s]'],
            yheader=[psse_header[i] for i in a_psse] if psse_header is not None else None,
            fig=fig, ax=ax,
            line_styles=[':', '-.'],
            legend=legend,
            left=left,
            right=right,
            title=title,
        )
    return fig, ax


def run_cmp(raw, dyr, fault_line, t1=1, t2=1.1, tf=20, tstep=1/60, no_output=True,
            reconnect=True):
    """
    Run a case study with a line trip and reconnect event.
    """
    ss = andes.main.load(raw, addfile=dyr, routine='TDS',
                         tf=tf, setup=False, no_output=no_output)

    # configure simulation parameters
    ss.config.warn_limits = 0
    ss.config.warn_abnormal = 0

    ss.add('Toggler', {'model': 'Line', 'dev': fault_line, 't': t1})
    if reconnect:
        ss.add('Toggler', {'model': 'Line', 'dev': fault_line, 't': t2})

    ss.setup()
    ss.PFlow.run()
    ss.TDS.init()

    ss.TDS.config.tf = 20
    ss.TDS.config.tstep = tstep
    ss.TDS.run()

    return ss
