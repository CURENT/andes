def pert(t, system):
    """VSC converter station """
    if t >=2 and t <= 2.1:
        system.RLs.u[7] = 0
        system.dae.rebuild = True
        system.log.info('Tripped RL-7 at t=2 s')