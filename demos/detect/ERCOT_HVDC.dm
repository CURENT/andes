# DOME format version 1.0

# ERCOT HVDC STATION BUSES: 32002, 32004, 32007, 32008

Bus, Vn = 230.0, angle = 0.48021, area = 1, idx = 32001, name = "MRGNCRK_1", owner = 130, region = 163,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.03474, xcoord = 32.332383, ycoord = -100.915183
Bus, Vn = 230.0, angle = 0.48018, area = 1, idx = 32002, name = "MRGNCRK_2", owner = 130, region = 163,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.03491, xcoord = 32.133383, ycoord = -100.916183
Bus, Vn = 345.0, angle = 0.7803, area = 1, idx = 32003, name = "SULSP_SS345", owner = 130, region = 142,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.15906, xcoord = 33.083923, ycoord = -95.613726
Bus, Vn = 345.0, angle = 0.7785, area = 1, idx = 32004, name = "SULSP2_SS345", owner = 130, region = 142,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.1593, xcoord = 32.883923, ycoord = -95.614726
Bus, Vn = 230.0, angle = 0.56373, area = 6, idx = 32005, name = "OKLAUN1", owner = 169, region = 477,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.09229, xcoord = 34.082211, ycoord = -99.180769
Bus, Vn = 345.0, angle = 0.57538, area = 6, idx = 32006, name = "OKLAUN2", owner = 169, region = 477,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.02892, xcoord = 34.083211, ycoord = -98.981769
Bus, Vn = 230.0, angle = 0.56366, area = 6, idx = 32007, name = "OKLAUN3", owner = 169, region = 477,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.09311, xcoord = 34.084211, ycoord = -99.380769
Bus, Vn = 345.0, angle = 0.57531, area = 6, idx = 32008, name = "OKLAUN4", owner = 169, region = 477,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.0297, xcoord = 34.083211, ycoord = -99.580769

Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.076, b1 = 0.0, b2 = 0.0, bus1 = 32001,
      bus2 = 32002, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_ERCOT_868", name = "Line ERCOT 869",
      owner = 0, phi = 0, r = 0.00068, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.00446, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 32001,
      bus2 = 30017, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_ERCOT_998", name = "Line ERCOT 999",
      owner = 0, phi = -1.272, r = 0.00086, rate_a = 0.0, tap = 1.0, trasf = True, u = 1,
      x = 0.00103, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 138.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 30031,
      bus2 = 32003, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_ERCOT_1153", name = "Line ERCOT 1154",
      owner = 0, phi = -16.94, r = 0.00103, rate_a = 0.0, tap = 1.0, trasf = True, u = 1,
      x = 0.02, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0052, b1 = 0.0, b2 = 0.0, bus1 = 32003,
      bus2 = 32004, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_ERCOT_869", name = "Line ERCOT 870",
      owner = 0, phi = 0, r = 0.00179, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.0002, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.076, b1 = 0.0, b2 = 0.0, bus1 = 32005,
      bus2 = 32007, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_ERCOT_870", name = "Line ERCOT 871",
      owner = 0, phi = 0, r = 0.00179, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.01988, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.076, b1 = 0.0, b2 = 0.0, bus1 = 32006,
      bus2 = 32008, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_ERCOT_871", name = "Line ERCOT 872",
      owner = 0, phi = 0, r = 0.00179, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.01988, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 138.0, Vn2 = 230.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 30203,
      bus2 = 32005, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_ERCOT_1850", name = "Line ERCOT 1851",
      owner = 0, phi = -16.94, r = 0.01026, rate_a = 0.0, tap = 1.0, trasf = True, u = 1,
      x = 0.05433, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 138.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 30203,
      bus2 = 32006, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_ERCOT_1851", name = "Line ERCOT 1852",
      owner = 0, phi = -16.94, r = 0.00103, rate_a = 0.0, tap = 1.0, trasf = True, u = 1,
      x = 0.00543, xcoord = 0, ycoord = 0


Shunt, Sn = 100.0, Vn = 345.0, b = 1.5, bus = 32003, fn = 60.0, g = 0.0, idx = "Shunt_ERCOT_231",
       name = "Shunt ERCOT 232", u = 1
Shunt, Sn = 100.0, Vn = 345.0, b = 1.0, bus = 32004, fn = 60.0, g = 0.0, idx = "Shunt_ERCOT_232",
       name = "Shunt ERCOT 233", u = 1
Shunt, Sn = 100.0, Vn = 230.0, b = 1.0, bus = 32005, fn = 60.0, g = 0.0, idx = "Shunt_ERCOT_233",
       name = "Shunt ERCOT 234", u = 1
