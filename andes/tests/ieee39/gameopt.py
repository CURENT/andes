
def pert(t, system):
    if abs(t - 2.0) < 1e-1:
        pass
        if system.Syn6a.u[2] == 1:
            system.Syn6a.u[2] = 0
            system.Log.info('Generator 3 tripped at t = 2')
            pass
