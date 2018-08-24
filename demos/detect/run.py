from andes import system, filters
from andes.routines import powerflow, timedomain
import os

# os.chdir('/home/hcui7/repos/andes_github/cases')
# case = 'ieee14_syn.dm'

os.chdir('/home/hcui7/repos/andes_github/cases/curent')
case = 'NA_50_50_50_HVDC3.dm'

sys = system.PowerSystem(case)
assert filters.guess(sys)
assert filters.parse(sys)
sys.setup()
sys.pf_init()
powerflow.run(sys)

sys.td_init()
# timedomain.run(sys)

# bus list in NPCC
npcc_bus = list(range(100114, 100124)) + list(range(101001, 101120))

# In the test case ``NA_50_50_50_HVDC3.dm``, all NPCC generators are classical model
# ====== commented out ======
# 6-th order generator model on NPCC buses. All should be None.
# sys.Syn6a.on_bus(npcc_bus)
# ===========================

# get the classical generator idx on NPCC buses.
npcc_gen = sys.Syn2.on_bus(npcc_bus)

# save Andes cases to folder
save_path = '/home/hcui7/repos/andes_github/demos/detect/GT'

file_name_tpl = '{event}_{bus}_{element}.dm'

header = """# DOME format version 1.0

# This case implements generator idx=<{gen}> trip event on bus <{bus}> at t=1s

INCLUDE, NA_50_50_50_HVDC3.dm

"""
event_tpl = 'GenTrip, t1 = 1, gen = \"{}\"'


for bus_idx, gen_idx in zip(npcc_bus, npcc_gen):
    if gen_idx is None:
        continue

    file_name = file_name_tpl.format(event='GT', bus=bus_idx, element=gen_idx)

    out = ''
    if isinstance(gen_idx, list):
        for i in gen_idx:
            out = header.format(gen=i, bus=bus_idx)
            out += event_tpl.format(i)
            file_name = file_name_tpl.format(event='GT', bus=bus_idx, element=i)
    else:
        out = header.format(gen=gen_idx, bus=bus_idx)
        out += event_tpl.format(gen_idx)

    with open(os.path.join(save_path, file_name), 'w') as f:
        f.write(out)


# ==================== Load Shedding ========================

# get the load idx on NPCC buses.
npcc_load = sys.PQ.on_bus(npcc_bus)

# save Andes cases to folder
save_path = '/home/hcui7/repos/andes_github/demos/detect/LS'

file_name_tpl = '{event}_{bus}_{element}.dm'

header = """# DOME format version 1.0

# This case implements laod idx=<{load}> shedding event on bus <{bus}> at t=1s

INCLUDE, NA_50_50_50_HVDC3.dm

"""
event_tpl = 'LoadShed, group=\"StaticLoad\", t1 = 1, load = \"{}\"'


for bus_idx, load_idx in zip(npcc_bus, npcc_load):
    if load_idx is None:
        continue

    file_name = file_name_tpl.format(event='LS', bus=bus_idx, element=load_idx)

    out = ''
    if isinstance(load_idx, list):
        for i in load_idx:
            out = header.format(load=i, bus=bus_idx)
            out += event_tpl.format(i)
            file_name = file_name_tpl.format(event='LS', bus=bus_idx, element=i)
    else:
        out = header.format(load=load_idx, bus=bus_idx)
        out += event_tpl.format(load_idx)

    with open(os.path.join(save_path, file_name), 'w') as f:
        f.write(out)



sys.Line.link_bus(1)