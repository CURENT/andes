import andes
from andes.main import run
from multiprocessing import Process


run(case='../tests/curent/NA_50_50_50_HVDC3.dm', routine='td', )



# jobs = []
# kwargs['verbose'] = ERROR
# for idx, casename in enumerate(cases):
#     kwargs['pid'] = idx
#     job = Process(name='Process {0:d}'.format(idx), target=run, args=(casename,), kwargs=kwargs)
#     jobs.append(job)
#     job.start()
#     print('Process {:d} <{:s}> started.'.format(idx, casename))
#
# sleep(0.1)
# for job in jobs:
#     job.join()
# t0, s0 = elapsed(t0)
# print('--> Multiple processing finished in {0:s}.'.format(s0))
# return
