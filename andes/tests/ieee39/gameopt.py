
def pert(t, system):
    if abs(t - 2.0) < 1e-1:
        pass
        if system.Syn6a.u[1] == 1:
            system.Syn6a.u[1] = 0
            system.Log.info('Generator 2 tripped at t = 2')
            pass
