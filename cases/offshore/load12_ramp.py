"""Sample perturbation file"""


def pert(t, system):
    if 2.0 <= t <=2.2:
        if system.PQ.p[8] == 0.061:
            system.PQ.p[8] = 0.461
            system.Log.info('Load on Bus 12 ramped up by 0.4 pu at t = 2')
