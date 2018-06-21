# DOME format version 1.0

Bus, Vn = 345.0, idx = 1, name = "Bus 1", xcoord = [1; 1], ycoord = [1; 2]
Bus, Vn = 345.0, idx = 2, name = "Bus 2", xcoord = [4; 1], ycoord = [4; 2]
Bus, Vn = 345.0, idx = 3, name = "Bus 3", xcoord = [2; 1], ycoord = [2; 2]
Bus, Vn = 345.0, idx = 4, name = "Bus 4", xcoord = [3; 1], ycoord = [3; 2]
Bus, Vn = 345.0, idx = 5, name = "Bus 5", xcoord = [1; 3], ycoord = [2; 3]

Area, idx = 1, name = "5-Bus"

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

Fault, bus=5, tf=1.0, tc = 1.01, rf = 0, xf = 0.01

Breaker, Vn = 345.0, bus = 1, fn = 60.0, idx = 1, line = "Line_2",
         name = "Breaker 1", t1 = 1.0, u1 = 1, t2 = 200.0, u2=1
    
Syn6a, D = 0, M = 5.7512, Sn = 500.0, Td10 = 5.0, Td20 = 0.05,
       Tq10 = 0.12, Tq20 = 0.05, Vn = 345.0, bus = 1, fn = 60.0,
       gen = 1, idx = 1, name = "Syn 1", ra = 0, xd = 2.0,
       xd1 = 0.245, xd2 = 0.150, xq = 1.91, xq1 = 0.245, xq2 = 0.150

Syn6a, D = 0, M = 5.7512, Sn = 500.0, Td10 = 5.0, Td20 = 0.05,
       Tq10 = 0.12, Tq20 = 0.05, Vn = 345.0, bus = 2, fn = 60.0,
       gen = 2, idx = 2, name = "Syn 2", ra = 0, xd = 2.0,
       xd1 = 0.245, xd2 = 0.150, xq = 1.91, xq1 = 0.245, xq2 = 0.150



#Avr1, Ka = 200.0, Kf = 0.0012, Ta = 0.02, Te = 0.19, Tf = 1.0,
#      bus = 1, idx = 1, name = "AVR 1", syn = 1, vmax = 9.99,
#      vmin = 0.0
#Avr1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 1.98, Tf = 1.0,
#      bus = 2, idx = 2, name = "AVR 2", syn = 2, vmax = 2.05,
#      vmin = 0.0


