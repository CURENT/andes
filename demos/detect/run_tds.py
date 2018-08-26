from andes import system, filters
from andes.routines import powerflow, timedomain
import os
from andes.plot import do_plot

os.chdir('../../cases/curent')
case = 'NA_50_50_50_HVDC3.dm'

sys = system.PowerSystem(case)
assert filters.guess(sys)
assert filters.parse(sys)

sys.setup()
sys.pf_init()
powerflow.run(sys)

sys.td_init()
sys.TDS.tf = 2
timedomain.run(sys)

Syn6a_omega = sys.Syn6a.omega
Syn6a_omega = [i + 1 for i in Syn6a_omega]

x, y = sys.VarOut.get_xy(Syn6a_omega)


do_plot(x, y)