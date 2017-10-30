def pert(t, system):
    """VSC converter station """
    if t >=2 and t <= 2.1:
        system.RLs.u[7] = 0
        system.DAE.rebuild = True