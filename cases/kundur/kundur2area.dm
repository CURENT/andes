# DOME format version 1.0

Syn6a, M = 13.0, Sn = 900.0, Taa = 0.002, Td20 = 0.03, Tq10 = 0.4, Td10 = 8,
       Tq20 = 0.05, Vn = 20.0, bus = 1, gen = 1, D = 2,
       idx = 1, name = "Syn6a 1", ra = 0.0025, xd = 1.8, xq = 1.8, xd1 = 0.3,
       xd2 = 0.25, xl = 0.2, xq1 = 0.55, xq2 = 0.25

Syn6a, M = 13.0, Sn = 900.0, Td20 = 0.03, Tq10 = 0.4, Tq20 = 0.05, Td10 = 8,
       Vn = 20.0, bus = 2, gen = 2, idx = 2, D = 2,
       name = "Syn6a 2", ra = 0.0025, xd = 1.8, xq = 1.8, xd1 = 0.3, xd2 = 0.25,
       xl = 0.2, xq1 = 0.55, xq2 = 0.25

#WTG3, bus = 1, gen = 1, wind = 1, Vn = 20, Sn = 400, qmax = 0.7, qmin = -0.7
#WTG3, bus = 2, gen = 2, wind = 2, Vn = 20, Sn = 400, qmax = 0.7, qmin = -0.7

#ConstWind, idx = 1, Vwn = 15
#ConstWind, idx = 2, Vwn = 15

Syn6a, M = 12.35, Sn = 900.0, Td20 = 0.03, Tq10 = 0.4, Tq20 = 0.05, Td10 = 8,
       Vn = 20.0, bus = 3, gen = 3, idx = 3, D = 2,
       name = "Syn6a 3", ra = 0.0025, xd = 1.8, xq = 1.8, xd1 = 0.3, xd2 = 0.25,
       xl = 0.2, xq1 = 0.55, xq2 = 0.25
Syn6a, M = 12.35, Sn = 900.0, Td20 = 0.03, Tq10 = 0.4, Tq20 = 0.05, Td10 = 8, 
       Vn = 20.0, bus = 4, gen = 4, idx = 4, D = 2,
       name = "Syn6a 4", ra = 0.0025, xd = 1.8, xq = 1.8, xd1 = 0.3, xd2 = 0.25,
       xl = 0.2, xq1 = 0.55, xq2 = 0.25

AVR1, Ae = 0.0056, Be = 1.075, Ka = 20.0, Kf = 0.125, Ta = 0.055,
      Te = 0.36, Tf = 1.8, Tr = 0.05, idx = 1, name = "AVR1 1",
      syn = 1
AVR1, Ae = 0.0056, Be = 1.075, Ka = 20.0, Kf = 0.125, Ta = 0.055,
      Te = 0.36, Tf = 1.8, Tr = 0.05, idx = 2, name = "AVR1 2",
      syn = 2
AVR1, Ae = 0.0056, Be = 1.075, Ka = 20.0, Kf = 0.125, Ta = 0.055,
      Te = 0.36, Tf = 1.8, Tr = 0.05, idx = 3, name = "AVR1 3",
      syn = 4
AVR1, Ae = 0.0056, Be = 1.075, Ka = 20.0, Kf = 0.125, Ta = 0.055,
      Te = 0.36, Tf = 1.8, Tr = 0.05, idx = 4, name = "AVR1 4",
      syn = 3

BusFreq, idx = 1, bus = 1
BusFreq, idx = 2, bus = 2
BusFreq, idx = 3, bus = 3
BusFreq, idx = 4, bus = 4
BusFreq, idx = 5, bus = 5
BusFreq, idx = 6, bus = 6
BusFreq, idx = 7, bus = 7
BusFreq, idx = 8, bus = 8
BusFreq, idx = 9, bus = 9
BusFreq, idx = 10, bus = 10
BusFreq, idx = 11, bus = 11

PSS1, idx = 1, avr = 3

#Breaker, bus = 1, idx = "Breaker_1", line = "Line_14", name = "Breaker 1", t1 = 1.1,
#         t2 = 999999.0, u1 = 1
Breaker, bus = 7, Vn = 230, idx = "Breaker_2", line = "Line_5", name = "Breaker 2", t1 = 2.0,
         t2 = 999999.0, u1 = 1

Bus, Vn = 20.0, angle = 0.35029, area = 1.0, idx = 1, name = "Bus 1 GEN G1", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.03, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 20.0, angle = 0.17987, area = 1.0, idx = 2, name = "Bus 2 GEN G2", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.01, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 20.0, angle = -0.12217, area = 2.0, idx = 3, name = "Bus 3 GEN G3", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.03, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 20.0, angle = -0.30006, area = 2.0, idx = 4, name = "Bus 4 GEN G4", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.01, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 230.0, angle = 0.23751, area = 1.0, idx = 5, name = "Bus 5 G1", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.00646, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 230.0, angle = 0.0615, area = 1.0, idx = 6, name = "Bus 6 G2", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 0.97813, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 230.0, angle = -0.08527, area = 1.0, idx = 7, name = "Bus 7 LOAD A", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 0.96102, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 230.0, angle = -0.32734, area = 1.0, idx = 8, name = "Bus 8 MID POINT", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 0.94862, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 230.0, angle = -0.56465, area = 2.0, idx = 9, name = "Bus 9 LOAD B", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 0.97137, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 230.0, angle = -0.41778, area = 2.0, idx = 10, name = "Bus 10 G4", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 0.98346, xcoord = 0, ycoord = 0, zone = 0
Bus, Vn = 230.0, angle = -0.23784, area = 2.0, idx = 11, name = "Bus 11 G3", owner = 0, region = 1.0,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.00826, xcoord = 0, ycoord = 0, zone = 0

Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.02187, b1 = 0.0, b2 = 0.0, bus1 = 5.0,
      bus2 = 6.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_0", name = "Line 1",
      owner = 0, phi = 0, r = 0.005, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.05, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.02187, b1 = 0.0, b2 = 0.0, bus1 = 5.0,
      bus2 = 6.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_1", name = "Line 2",
      owner = 0, phi = 0, r = 0.005, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.05, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.00583, b1 = 0.0, b2 = 0.0, bus1 = 6.0,
      bus2 = 7.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_2", name = "Line 3",
      owner = 0, phi = 0, r = 0.003, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.03, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.00583, b1 = 0.0, b2 = 0.0, bus1 = 6.0,
      bus2 = 7.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_3", name = "Line 4",
      owner = 0, phi = 0, r = 0.003, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.03, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.00583, b1 = 0.0, b2 = 0.0, bus1 = 6.0,
      bus2 = 7.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_4", name = "Line 5",
      owner = 0, phi = 0, r = 0.003, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.03, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.1925, b1 = 0.0, b2 = 0.0, bus1 = 7.0,
      bus2 = 8.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_5", name = "Line 6",
      owner = 0, phi = 0, r = 0.011, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.11, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.1925, b1 = 0.0, b2 = 0.0, bus1 = 7.0,
      bus2 = 8.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_6", name = "Line 7",
      owner = 0, phi = 0, r = 0.011, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.11, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.1925, b1 = 0.0, b2 = 0.0, bus1 = 8.0,
      bus2 = 9.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_7", name = "Line 8",
      owner = 0, phi = 0, r = 0.011, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.11, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.1925, b1 = 0.0, b2 = 0.0, bus1 = 8.0,
      bus2 = 9.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_8", name = "Line 9",
      owner = 0, phi = 0, r = 0.011, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.11, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.00583, b1 = 0.0, b2 = 0.0, bus1 = 9.0,
      bus2 = 10.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_9", name = "Line 10",
      owner = 0, phi = 0, r = 0.003, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.03, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.00583, b1 = 0.0, b2 = 0.0, bus1 = 9.0,
      bus2 = 10.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_10", name = "Line 11",
      owner = 0, phi = 0, r = 0.003, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.03, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.00583, b1 = 0.0, b2 = 0.0, bus1 = 9.0,
      bus2 = 10.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_11", name = "Line 12",
      owner = 0, phi = 0, r = 0.003, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.03, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.02187, b1 = 0.0, b2 = 0.0, bus1 = 10.0,
      bus2 = 11.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_12", name = "Line 13",
      owner = 0, phi = 0, r = 0.005, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.05, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.02187, b1 = 0.0, b2 = 0.0, bus1 = 10.0,
      bus2 = 11.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_13", name = "Line 14",
      owner = 0, phi = 0, r = 0.005, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.05, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 20.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 5.0,
      bus2 = 1.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_14", name = "Line 15",
      owner = 0, phi = 0, r = 0.0, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.01667, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 20.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 6.0,
      bus2 = 2.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_15", name = "Line 16",
      owner = 0, phi = 0, r = 0.0, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.01667, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 20.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 11.0,
      bus2 = 3.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_16", name = "Line 17",
      owner = 0, phi = 0, r = 0.0, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.01667, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 20.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 10.0,
      bus2 = 4.0, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_17", name = "Line 18",
      owner = 0, phi = 0, r = 0.0, rate_a = 0, tap = 1, trasf = False, u = 1,
      x = 0.01667, xcoord = 0, ycoord = 0

PQ, Sn = 100.0, Vn = 230.0, bus = 7, idx = "PQ_0", name = "PQ 7", owner = 0, p = 9.67,
    q = 1.0, u = 1, vmax = 1.1, vmin = 0.9
PQ, Sn = 100.0, Vn = 230.0, bus = 9, idx = "PQ_1", name = "PQ 9", owner = 0, p = 17.67,
    q = 1.0, u = 1, vmax = 1.1, vmin = 0.9

PV, Sn = 100.0, Vn = 20.0, bus = 1, busr = 1, idx = 1, name = "PV 1", pg = 7.0,
    pmax = 7.65, pmin = 0.0, qg = 1.85006, qmax = 4.74, qmin = -2.0, ra = 0.01, u = 1,
    v0 = 1.03, vmax = 1.4, vmin = 0.6, xs = 0.3
PV, Sn = 100.0, Vn = 20.0, bus = 2, busr = 2, idx = 2, name = "PV 2", pg = 7.0,
    pmax = 7.65, pmin = 0.0, qg = 2.34587, qmax = 4.74, qmin = -2.0, ra = 0.01, u = 1,
    v0 = 1.01, vmax = 1.4, vmin = 0.6, xs = 0.3
PV, Sn = 100.0, Vn = 20.0, bus = 4, busr = 4, idx = 4, name = "PV 4", pg = 7.0,
    pmax = 7.65, pmin = 0.0, qg = 2.02055, qmax = 4.74, qmin = -2.0, ra = 0.01, u = 1,
    v0 = 1.01, vmax = 1.4, vmin = 0.6, xs = 0.3

SW, Sn = 100.0, Vn = 20.0, a0 = 0.0, bus = 3, busr = 3, idx = 3, name = "SW 3",
    pg = 7.19092, pmax = 7.65, pmin = 0.0, qg = 1.76001, qmax = 4.74, qmin = -2.0, ra = 0.01,
    u = 1, v0 = 1.03, vmax = 1.4, vmin = 0.6, xs = 0.3

Shunt, Sn = 100.0, Vn = 230.0, b = 2.0, bus = 7, fn = 60.0, g = 0.0, idx = "Shunt_0",
       name = "Shunt 7", u = 1
Shunt, Sn = 100.0, Vn = 230.0, b = 3.5, bus = 9, fn = 60.0, g = 0.0, idx = "Shunt_1",
       name = "Shunt 9", u = 1

TG1, idx = 1, gen = 1, R = 0.02, T5 = 50, Tc = 0.56, Ts = 0.1
TG1, idx = 2, gen = 2, R = 0.02, T5 = 50, Tc = 0.56, Ts = 0.1
TG1, idx = 3, gen = 3, R = 0.02, T5 = 50, Tc = 0.56, Ts = 0.1
TG1, idx = 4, gen = 4, R = 0.02, T5 = 50, Tc = 0.56, Ts = 0.1
