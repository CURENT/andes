"""Sample perturbation file. It trips the 9th generator at t=2"""


def pert(t, system):
    if 2.0 <= t <=2.2:
        if system.Syn6a.u[8] == 1:
            system.Syn6a.u[8] = 0
            system.log.info('Generator #9 tripped at t = 2')
