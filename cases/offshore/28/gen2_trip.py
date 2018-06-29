"""Sample perturbation file. It trips the 9th generator at t=2"""


def pert(t, system):
    if 2.0 <= t <=2.2:
        if system.Syn6a.u[1] == 1:
            system.Syn6a.u[1] = 0
            system.Log.info('Generator #2 tripped at t = 2')
