# DOME format version 1.0

# This case implements laod idx=<PQ_EI_127> shedding event on bus <101109> at t=1s

INCLUDE, NA_50_50_50_HVDC3.dm

LoadShed, group="StaticLoad", t1 = 1, load = "PQ_EI_127"