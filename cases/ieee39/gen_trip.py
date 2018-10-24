
def pert(t, system):
    if abs(t - 2.0) < 1e-1:
        pass
        # if system.Syn6a.u[7] == 1:
        #     system.Syn6a.u[7] = 0
        #     system.log.info('Generator 8 tripped at t = 2')
        #     pass
