# DOME format version 1.0

# WECC SYSTEM DATA
# WECC HVDC station buses: 2191, 2189, 2090, 2192
INCLUDE, WECC_PF_BASE.dm
INCLUDE, WECC_HVDC.dm
INCLUDE, WECC_DYN_BASE.dm
INCLUDE, WECC_MON.dm

Fault, Vn = 500, bus = 6, tf = 2.0, tc = 2.1, xf = 0.01
