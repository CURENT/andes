# DOME format version 1.0

Bus, Vn = 345.0, idx = 1, name = "Bus 1", xcoord = [1; 1], ycoord = [1; 2], area = 1
Bus, Vn = 345.0, idx = 2, name = "Bus 2", xcoord = [4; 1], ycoord = [4; 2], area = 2
Bus, Vn = 345.0, idx = 3, name = "Bus 3", xcoord = [2; 1], ycoord = [2; 2], area = 1
Bus, Vn = 345.0, idx = 4, name = "Bus 4", xcoord = [3; 1], ycoord = [3; 2], area = 2
Bus, Vn = 345.0, idx = 5, name = "Bus 5", xcoord = [1; 3], ycoord = [2; 3], area = 1

Area, idx = 1, name = "Area 1"
Area, idx = 2, name = "Area 2"

BArea, idx=1, name = "BArea 1", area = 1, syn = [1], beta = 74.627
BArea, idx=2, name = "BArea 2", area = 2, syn = [2], beta = 74.627

# AGCTG, idx = 1, name = "AGCTG 1", BArea = 1, tg = [1], Ki = 0.5
# AGCTG, idx = 2, name = "AGCTG 2", BArea = 2, tg = [2], Ki = 0.5

# AGCSyn, idx = 1, name = "AGC 1", BArea = 1, syn = [1], Ki = 0.5
# AGCSyn, idx = 2, name = "AGC 2", BArea = 2, syn = [2], Ki = 0.5

# AGCSynVSC, idx = 1, name = "AGC 1", BArea = 1, syn = [1], Ki = 0.5, vsc = ["VSC1_1"], Rvsc = [0.02]
# AGCSynVSC, idx = 2, name = "AGC 2", BArea = 2, syn = [2], Ki = 0.5, vsc = ["VSC1_2"], Rvsc = [0.02]

AGCTGVSC, idx = 1, name = "AGC 1", BArea = 1, tg = [1], Ki = 0.5, vsc = ["VSC1_1"], Rvsc = [0.02]
AGCTGVSC, idx = 2, name = "AGC 2", BArea = 2, tg = [2], Ki = 0.5, vsc = ["VSC1_2"], Rvsc = [0.02]

# eAGC, idx=1, name = "EAGC 1", tl = 1.0, Pl = 1.0, BA = [1; 2], cl = [0.5; -0.5]
# eAGC, idx=2, name = "EAGC 2", tl = 1.0, Pl = 0.5, BA = [1; 2], cl = [0.5; -0.5]

Node, idx = 0, Vdcn = 100
Node, idx = 1, Vdcn = 100
Node, idx = 2, Vdcn = 100

Ground, idx = 0, node = 0, Vdcn = 100

R, idx = 1, node1 = 1, node2 = 2, Vdcn = 100, R = 10
C, idx = 1, node1 = 1, node2 = 0, Vdcn = 100, C = 1e-6
C, idx = 2, node1 = 2, node2 = 0, Vdcn = 100, C = 1e-6

VSC, Idcn = 10.0, Ishmax = 2, K = 0, Sn = 100.0, Vdcn = 100.0,
     Vn = 345.0, bus = 2, control = "vV", droop = 0, k0 = 0,
     k1 = 0, k2 = 0, node1 = 1, node2 = 0, pref0 = 0,
     qref0 = 0, rsh = 0.0025, u = 1, v0 = 1.0, vdcref0 = 1.0,
     vhigh = 9999, vlow = 0.0, vref0 = 1.05, vshmax = 1.1, vshmin = 0.9,
     xsh = 0.06, idx= "VSC_1"

VSC, Idcn = 10.0, Ishmax = 2, K = 0, Sn = 100.0, Vdcn = 100.0,
     Vn = 345.0, bus = 4, control = "PQ", droop = 0, k0 = 0,
     k1 = 0, k2 = 0, node1 = 2, node2 = 0, pref0 = -0.2,
     qref0 = 0, rsh = 0.0025, u = 1, v0 = 1.0, vdcref0 = 1.0,
     vhigh = 9999, vlow = 0.0, vref0 = 1.0, vshmax = 1.1, vshmin = 0.9,
     xsh = 0.06, idx = "VSC_2"

VSC1, idx = "VSC1_1", vsc = "VSC_1", name = "VSC 1", Kp1 = 0.2, Ki1 = 1,
      Kp2 = 4, Ki2 = 2, Kp3 = 1, Ki3 = 0.5

VSC1, idx = "VSC1_2", vsc = "VSC_2", name = "VSC 2", Kp1 = 0.2, Ki1 = 0.1,
      Kp2 = 0.2, Ki2 = 0.0, Kp3 = 0.01, Ki3 = 0.0

Region, Ptol = 9.9999, idx = 1, name = "5Bus   5", slack = 1.0

# Parameter 'phi' of phase changer has unit 'Deg'
Line, Vn = 345.0, Vn2 = 345.0, b = 0.0, bus1 = 1, bus2 = 3,
      idx = "Line_1", name = "Line 1", r = 0, x = 0.03, xcoord = [1; 2],
      ycoord = [1.5; 1.5]
Line, Vn = 345.0, Vn2 = 345.0, b = 0.0, bus1 = 1, bus2 = 5,
      idx = "Line_2", name = "Line 2", r = 0, x = 0.04, xcoord = [1; 1.5; 1.5],
      ycoord = [1.5; 1.5; 2]
Line, Vn = 345.0, Vn2 = 345.0, b = 0.0, bus1 = 3, bus2 = 4,
      idx = "Line_3", name = "Line 3", r = 0, x = 0.04, xcoord = [2; 3],
      ycoord = [1.5; 1.5]
Line, Vn = 345.0, Vn2 = 345.0, b = 0.0, bus1 = 4, bus2 = 2,
      idx = "Line_4", name = "Line 4", r = 0, x = 0.03, xcoord = [3; 4],
      ycoord = [1.5; 1.5]

PQ, Vn = 345.0, bus = 3, idx = "PQ load_1", name = "PQ Bus 3", p = 4.0,
    q = 0
PQ, Vn = 345., bus = 4, idx = "PQ load_2", name = "PQ Bus 4", p = 3.0,
    q = 0
PQ, Vn = 345.0, bus = 5, idx = "PQ load_3", name = "PQ Bus 5", p = 1.0,
    q = 0
    
PV, Vn = 345.0, bus = 2, busr = 2, idx = 2, name = "PV Bus 2",
    pg = 4.0, pmax = 6.0, pmin = 0, qmax = 100.0, qmin= -100,
    v0 = 1.05

SW, Vn=345.0, bus = 1, busr = 1, idx = 1, name = "SW Bus 1", pmax = 6.0,
     pmin = -6.0, qmax = 100.0, qmin = -100.0, v0 = 1.05

Breaker, Vn = 345.0, bus = 1, fn = 60.0, idx = 1, line = "Line_2",
         name = "Breaker 1", t1 = 1.0, u1 = 1, t2 = 200.0, u2=1
    
# Syn2, D = 10.0, M = 10.0, Sn = 500.0, Vn = 345.0, bus = 1, fn = 60.0,
#       gen = 1, idx = 1, name = "Syn 1", xd1 = 0.302
#
# Syn2, D = 10.0, M = 10.0, Sn = 500.0, Vn = 345.0, bus = 2, fn = 60.0,
#       gen = 2, idx = 2, name = "Syn 2", xd1 = 0.302


Syn6a, D = 10.0, M = 5.7512, Sn = 500.0, Td10 = 5.0, Td20 = 0.05,
       Tq10 = 0.12, Tq20 = 0.05, Vn = 345.0, bus = 1, fn = 60.0,
       gen = 1, idx = 1, name = "Syn 1", ra = 0, xd = 2.0,
       xd1 = 0.245, xd2 = 0.150, xq = 1.91, xq1 = 0.245, xq2 = 0.150

Syn6a, D = 10.0, M = 5.7512, Sn = 500.0, Td10 = 5.0, Td20 = 0.05,
       Tq10 = 0.12, Tq20 = 0.05, Vn = 345.0, bus = 2, fn = 60.0,
       gen = 2, idx = 2, name = "Syn 2", ra = 0, xd = 2.0,
       xd1 = 0.245, xd2 = 0.150, xq = 1.91, xq1 = 0.245, xq2 = 0.150

AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 0.19, Tf = 1.0,
      idx = 1, name = "AVR 1", syn = 1, vrmax = 10,
      vrmin = -0
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 0.19, Tf = 1.0,
      idx = 2, name = "AVR 2", syn = 2, vrmax = 10,
      vrmin = 0.0

TG1, idx = 1, gen = 1, pmax = 5, pmin = 0, R = 0.04, wref0 = 1.0,
     T3 = 0, T4 = 1.25, T5 = 5.0, Tc = 0.4, Ts = 0.1
TG1, idx = 2, gen = 2, pmax = 5, pmin = 0, R = 0.04, wref0 = 1.0,
     T3 = 0, T4 = 1.25, T5 = 5.0, Tc = 0.4, Ts = 0.1
     
BusFreq, idx = 1, bus = 1
BusFreq, idx = 2, bus = 2
BusFreq, idx = 3, bus = 3
BusFreq, idx = 4, bus = 4
BusFreq, idx = 5, bus = 5