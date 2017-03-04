# DOME format version 1.0

Bus, Vn = 69.0, idx = 1, name = "Bus 1", xcoord = [1.875; 2.925], ycoord = [3.45; 3.45]
Bus, Vn = 69.0, idx = 2, name = "Bus 2", xcoord = [3.975; 5.25], ycoord = [0.825; 0.825]
Bus, Vn = 69.0, idx = 3, name = "Bus 3", xcoord = [8.1; 8.775], ycoord = [0.0; 0.0]
Bus, Vn = 69.0, idx = 4, name = "Bus 4", xcoord = [7.875; 9.075], ycoord = [3.3; 3.3]
Bus, Vn = 69.0, idx = 5, name = "Bus 5", xcoord = [5.925; 6.525], ycoord = [2.625; 2.625]
Bus, Vn = 13.8, idx = 6, name = "Bus 6", xcoord = [5.925; 6.825], ycoord = [3.825; 3.825]
Bus, Vn = 13.8, idx = 7, name = "Bus 7", xcoord = [8.625; 9.075], ycoord = [4.05; 4.05]
Bus, Vn = 18.0, idx = 8, name = "Bus 8", xcoord = [9.6; 9.6], ycoord = [4.5; 4.05]
Bus, Vn = 13.8, idx = 9, name = "Bus 9", xcoord = [8.025; 9.075], ycoord = [4.5; 4.5]
Bus, Vn = 13.8, idx = 10, name = "Bus 10", xcoord = [7.875; 7.35], ycoord = [4.875; 4.875]
Bus, Vn = 13.8, idx = 11, name = "Bus 11", xcoord = [7.05; 6.525], ycoord = [5.025; 5.025]
Bus, Vn = 13.8, idx = 12, name = "Bus 12", xcoord = [4.875; 4.275], ycoord = [5.625; 5.625]
Bus, Vn = 13.8, idx = 13, name = "Bus 13", xcoord = [5.85; 6.6], ycoord = [6.075; 6.075]
Bus, Vn = 13.8, idx = 14, name = "Bus 14", xcoord = [7.875; 7.275], ycoord = [5.625; 5.625]

Area, idx = 1, name = "14-Bus"

Region, Ptol = 9.9999, idx = 1, name = "14Bus   14", slack = 1.0

# Parameter 'phi' of phase changer has unit 'Deg'
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0528, bus1 = 1, bus2 = 2,
      idx = "Line_1", name = "Line 1", r = 0.01938, x = 0.05917, xcoord = [2.025; 2.025; 4.2; 4.2],
      ycoord = [3.45; 3.15; 1.2; 0.825]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0492, bus1 = 1, bus2 = 5,
      idx = "Line_2", name = "Line 2", r = 0.05403, x = 0.22304, xcoord = [2.775; 2.775; 5.7; 6.075; 6.075],
      ycoord = [3.45; 3.15; 2.325; 2.325; 2.625]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0438, bus1 = 2, bus2 = 3,
      idx = "Line_3", name = "Line 3", r = 0.04699, x = 0.19797, xcoord = [5.1; 5.1; 8.25; 8.25],
      ycoord = [0.825; 0.6; 0.225; 0.0]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0374, bus1 = 2, bus2 = 4,
      idx = "Line_4", name = "Line 4", r = 0.05811, x = 0.17632, xcoord = [5.1; 5.1; 8.4; 8.4],
      ycoord = [0.825; 1.2; 3.15; 3.3]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.034, bus1 = 2, bus2 = 5,
      idx = "Line_5", name = "Line 5", r = 0.05695, x = 0.17388, xcoord = [4.725; 4.725; 6.225; 6.225],
      ycoord = [0.825; 1.2; 2.1; 2.625]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0346, bus1 = 3, bus2 = 4,
      idx = "Line_6", name = "Line 6", r = 0.06701, x = 0.17103, xcoord = [8.55; 8.55],
      ycoord = [0.0; 3.3]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0128, bus1 = 4, bus2 = 5,
      idx = "Line_7", name = "Line 7", r = 0.01335, x = 0.04211, xcoord = [6.375; 6.375; 8.025; 8.025],
      ycoord = [2.625; 2.175; 3.15; 3.3]
Line, Vn = 69.0, Vn2 = 13.8, bus1 = 4, bus2 = 7, idx = "Line_8",
      name = "Line 8", tap = 0.978, trasf = True, x = 0.20912,
      xcoord = [8.85; 8.85], ycoord = [3.3; 4.05]
Line, Vn = 69.0, Vn2 = 13.8, bus1 = 4, bus2 = 9, idx = "Line_9",
      name = "Line 9", tap = 0.969, trasf = True, x = 0.55618,
      xcoord = [8.25; 8.25], ycoord = [4.5; 3.3]
Line, Vn = 69.0, Vn2 = 13.8, bus1 = 5, bus2 = 6, idx = "Line_10",
      name = "Line 10", tap = 0.932, trasf = True, x = 0.25202,
      xcoord = [6.225; 6.225], ycoord = [2.625; 3.825]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 6, bus2 = 11, idx = "Line_11",
      name = "Line 11", r = 0.09498, x = 0.19890, xcoord = [6.45; 6.45; 6.675; 6.675], ycoord = [3.825; 4.05; 4.8; 5.025]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 6, bus2 = 12, idx = "Line_12",
      name = "Line 12", r = 0.12291, x = 0.25581, xcoord = [4.425; 4.425; 6.075; 6.075], ycoord = [5.625; 5.4; 4.05; 3.825]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 6, bus2 = 13, idx = "Line_13",
      name = "Line 13", r = 0.06615, x = 0.13027, xcoord = [6.225; 6.225], ycoord = [3.825; 6.075]
Line, Vn = 13.8, Vn2 = 18.0, bus1 = 7, bus2 = 8, idx = "Line_14",
      name = "Line 14", trasf = True, x = 0.17615, xcoord = [9.6; 8.925; 8.925],
      ycoord = [4.275; 4.275; 4.05]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 7, bus2 = 9, idx = "Line_15",
      name = "Line 15", x = 0.11001, xcoord = [8.775; 8.775], ycoord = [4.05; 4.5]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 9, bus2 = 10, idx = "Line_16",
      name = "Line 16", r = 0.03181, x = 0.08450, xcoord = [7.725; 7.725; 8.25; 8.25], ycoord = [4.875; 4.725; 4.65; 4.5]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 9, bus2 = 14, idx = "Line_17",
      name = "Line 17", r = 0.12711, x = 0.27038, xcoord = [8.55; 8.55; 7.65; 7.65], ycoord = [4.5; 4.8; 5.25; 5.625]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 10, bus2 = 11, idx = "Line_18",
      name = "Line 18", r = 0.08205, x = 0.19207, xcoord = [6.9; 6.9; 7.5; 7.5], ycoord = [5.025; 4.8; 4.725; 4.875]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 12, bus2 = 13, idx = "Line_19",
      name = "Line 19", r = 0.22092, x = 0.19988, xcoord = [6.0; 6.0; 5.025; 4.725; 4.725], ycoord = [6.075; 5.925; 5.4; 5.4; 5.625]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 13, bus2 = 14, idx = "Line_20",
      name = "Line 20", r = 0.17093, x = 0.34802, xcoord = [7.5; 7.5; 6.45; 6.45], ycoord = [5.625; 5.325; 5.925; 6.075]

#BusFreq, idx = 1, bus = 1
#BusFreq, idx = 2, bus = 2
#BusFreq, idx = 3, bus = 3
#BusFreq, idx = 4, bus = 4
#BusFreq, idx = 5, bus = 5
#BusFreq, idx = 6, bus = 6
#BusFreq, idx = 7, bus = 7
#BusFreq, idx = 8, bus = 8
#BusFreq, idx = 9, bus = 9
#BusFreq, idx = 10, bus = 10
#BusFreq, idx = 11, bus = 11
#BusFreq, idx = 12, bus = 12
#BusFreq, idx = 13, bus = 13
#BusFreq, idx = 14, bus = 14

#BusLine, line = "Line_1", busf = 1
#BusLine, line = "Line_2", busf = 1
#BusLine, line = "Line_3", busf = 2
##BusLine, line = "Line_4"
#BusLine, line = "Line_5", busf = 2
#BusLine, line = "Line_6", busf = 3
#BusLine, line = "Line_7", busf = 4
#BusLine, line = "Line_8"
#BusLine, line = "Line_9"
#BusLine, line = "Line_10"
#BusLine, line = "Line_11", busf = 6
#BusLine, line = "Line_12", busf = 6
#BusLine, line = "Line_13", busf = 6
#BusLine, line = "Line_14"
#BusLine, line = "Line_15", busf = 7
#BusLine, line = "Line_16", busf = 9
#BusLine, line = "Line_17", busf = 9
#BusLine, line = "Line_18", busf = 10
#BusLine, line = "Line_19", busf = 12
#BusLine, line = "Line_20", busf = 13

PQ, Vn = 69.0, bus = 2, idx = "PQ load_1", name = "PQ Bus 2", p = 0.217,
    q = 0.127
PQ, Vn = 69.0, bus = 3, idx = "PQ load_2", name = "PQ Bus 3", p = 0.942,
    q = 0.19
PQ, Vn = 69.0, bus = 4, idx = "PQ load_3", name = "PQ Bus 4", p = 0.478,
    q = -0.039
PQ, Vn = 69.0, bus = 5, idx = "PQ load_4", name = "PQ Bus 5", p = 0.076,
    q = 0.016
PQ, Vn = 13.8, bus = 6, idx = "PQ load_5", name = "PQ Bus 6", p = 0.112,
    q = 0.075
PQ, Vn = 13.8, bus = 9, idx = "PQ load_6", name = "PQ Bus 9", p = 0.295,
    q = 0.166
PQ, Vn = 13.8, bus = 10, idx = "PQ load_7", name = "PQ Bus 10", p = 0.09,
    q = 0.058
PQ, Vn = 13.8, bus = 11, idx = "PQ load_8", name = "PQ Bus 11", p = 0.035,
    q = 0.018
PQ, Vn = 13.8, bus = 12, idx = "PQ load_9", name = "PQ Bus 12", p = 0.061,
    q = 0.016
PQ, Vn = 13.8, bus = 13, idx = "PQ load_10", name = "PQ Bus 13", p = 0.135,
    q = 0.058
PQ, Vn = 13.8, bus = 14, idx = "PQ load_11", name = "PQ Bus 14", p = 0.149,
    q = 0.05

Fault, bus = 4, tf = 2.0, tc = 2.01, rf = 0.0, xf = 0.01

#Breaker, Vn = 69.0, bus = 2, fn = 60.0, idx = 1, line = "Line_4",
#         name = "Breaker 1", t1 = 1.0, t2 = 200.0, u1 = 1, u2 = 1

PV, Vn = 69.0, bus = 2, busr = 2, idx = 2, name = "PV Bus 2",
    pg = 0.4, pmax = 1.0, pmin = 0, qmax = 0.4, qmin = -0.4,
    v0 = 1.045
PV, Vn = 69.0, bus = 3, busr = 3, idx = 3, name = "PV Bus 3",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.4, v0 = 1.01
PV, Vn = 13.8, bus = 6, busr = 6, idx = 6, name = "PV Bus 6",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.24, qmin = -0.06, v0 = 1.07
PV, Vn = 18.0, bus = 8, busr = 8, idx = 8, name = "PV Bus 8",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.24, qmin = -0.06, v0 = 1.09

Shunt, Vn = 13.8, b = 0.19, bus = 9, idx = 1, name = "Shunt Bus 9"

SW, Vn = 69.0, bus = 1, busr = 1, idx = 1, name = "SW Bus 1",
    pmax = 999.9, pmin = -999.9, qmax = 9.9, qmin = -9.9,
    v0 = 1.06

#Syn2, D = 3.01, M = 6.02, Sn = 500, Vn = 69, xd1 = 0.302,
#      bus = 1, fn = 60.0, gen = 1, idx = 1, name = "Syn_1"
#Syn2, D = 3.01, M = 6.02, Sn = 500, Vn = 69, xd1 = 0.302,
#      bus = 2, fn = 60.0, gen = 2, idx = 2, name = "Syn_2"
#Syn2, D = 3.01, M = 6.02, Sn = 500, Vn = 69, xd1 = 0.302,
#      bus = 3, fn = 60.0, gen = 3, idx = 3, name = "Syn_3"
#Syn2, D = 3.01, M = 6.02, Sn = 500, Vn = 13.8, xd1 = 0.302,
#      bus = 6, fn = 60.0, gen = 6, idx = 4, name = "Syn_6"


#Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
#       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 1, fn = 60.0,
#       gen = 1, idx = 1, name = "Syn 1", ra = 0.0031, xd = 1.05,
#       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13

#Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
#       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 2, fn = 60.0,
#       gen = 2, idx = 2, name = "Syn 2", ra = 0.0031, xd = 1.05,
#       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13

#Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
#       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 3, fn = 60.0,
#       gen = 3, idx = 3, name = "Syn 3", ra = 0.0031, xd = 1.05,
#       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13

#Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
#       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 6, fn = 60.0,
#       gen = 6, idx = 6, name = "Syn 6", ra = 0.0031, xd = 1.05,
#       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13

#Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
#       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 8, fn = 60.0,
#       gen = 8, idx = 8, name = "Syn 8", ra = 0.0031, xd = 1.05,
#       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13

#Syn8a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
#       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 3, fn = 60.0,
#       gen = 3, idx = 3, name = "Syn 3", ra = 0.0031, xd = 1.05,
#       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13

#Avr1, Ka = 200.0, Kf = 0.0012, Ta = 0.02, Te = 0.19, Tf = 1.0,
#      bus = 1, idx = 1, name = "AVR 1", syn = 1, vmax = 9.99,
#      vmin = 0.0
#Avr1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 1.98, Tf = 1.0,
#      bus = 2, idx = 2, name = "AVR 2", syn = 2, vmax = 2.05,
#      vmin = 0.0
#Avr1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 1.98, Tf = 1.0,
#      bus = 3, idx = 3, name = "AVR 3", syn = 3, vmax = 1.7,
#      vmin = 0.0
#Avr1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 0.7, Tf = 1.0,
#      bus = 6, idx = 4, name = "AVR 4", syn = 4, vmax = 2.2,
#      vmin = 1.0
#Avr1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 0.7, Tf = 1.0,
#      bus = 8, idx = 5, name = "AVR 5", syn = 5, vmax = 2.2,
#      vmin = 1.0

