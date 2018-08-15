# DOME format version 1.0

# EI HVDC STATION BUSES: 320002, 320004, 320005, 320006, 320007, 320009, 320011

Bus, Vn = 230.0, angle = -0.07708, area = 672, idx = 320001, name = "SASKACHEWA_I", owner = 1, region = 672,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.09461, xcoord = 50.68, ycoord = -109.94
Bus, Vn = 230.0, angle = -0.07713, area = 672, idx = 320002, name = "SASKACHEWA_R", owner = 1, region = 672,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.09476, xcoord = 50.68, ycoord = -109.94
Bus, Vn = 230.0, angle = -0.53842, area = 618, idx = 320003, name = "N_DAKOTA	_I", owner = 1, region = 624,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.0037, xcoord = 46.41, ycoord = -105.79
Bus, Vn = 230.0, angle = -0.53842, area = 618, idx = 320004, name = "N_DAKOTA	_R", owner = 1, region = 624,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.0037, xcoord = 46.41, ycoord = -105.79
Bus, Vn = 345.0, angle = -0.48375, area = 640, idx = 320005, name = "V_NEBRASKA_I", owner = 1, region = 659,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.019, xcoord = 41.82, ycoord = -103.95
Bus, Vn = 345.0, angle = -0.53407, area = 640, idx = 320006, name = "V_NEBRASKA_R", owner = 1, region = 659,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.02463, xcoord = 41.82, ycoord = -103.95
Bus, Vn = 230.0, angle = -0.52235, area = 640, idx = 320007, name = "S_NEBRASKA_I", owner = 1, region = 659,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 0.96003, xcoord = 41.12, ycoord = -103
Bus, Vn = 345.0, angle = -0.79358, area = 999, idx = 320008, name = "TX_OLKU_	I", owner = 1, region = 999,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 0.97077, xcoord = 34.13, ycoord = -99.15
Bus, Vn = 345.0, angle = -0.79358, area = 999, idx = 320009, name = "TX_OLKU_	R", owner = 1, region = 999,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 0.97082, xcoord = 34.13, ycoord = -99.15
Bus, Vn = 345.0, angle = -0.29217, area = 525, idx = 320010, name = "TX_WELSH	_I", owner = 1, region = 525,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.12853, xcoord = 33.06, ycoord = -94.84
Bus, Vn = 345.0, angle = -0.29217, area = 525, idx = 320011, name = "T459X_WELSH", owner = 1, region = 525,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.12853, xcoord = 33.06, ycoord = -94.84

Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 168601,
      bus2 = 320001, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_533", name = "Line EI 534",
      owner = 0, phi = 0, r = 0.05282, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.03737, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.02, b1 = 0.0, b2 = 0.0, bus1 = 320001,
      bus2 = 320002, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_540", name = "Line EI 541",
      owner = 0, phi = 0, r = 0.00528, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.01369, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.001, b1 = 0.0, b2 = 0.0, bus1 = 320003,
      bus2 = 320004, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_541", name = "Line EI 542",
      owner = 0, phi = 0, r = -0.0, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.0001, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.1, b1 = 0.0, b2 = 0.0, bus1 = 320008,
      bus2 = 320009, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_542", name = "Line EI 543",
      owner = 0, phi = 0, r = -0.0, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.001, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 320010,
      bus2 = 320011, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_543", name = "Line EI 544",
      owner = 0, phi = 0, r = 0.001, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.01, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.2, b1 = 0.0, b2 = 0.0, bus1 = 163056,
      bus2 = 320003, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_503", name = "Line EI 504",
      owner = 0, phi = 0, r = -0.0, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.0001, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 164786,
      bus2 = 320005, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_517", name = "Line EI 518",
      owner = 0, phi = 0, r = 0.00129, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.0103, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 164902,
      bus2 = 320006, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_521", name = "Line EI 522",
      owner = 0, phi = 0, r = 0.00129, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.0103, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 159995,
      bus2 = 320008, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_1533", name = "Line EI 1534",
      owner = 0, phi = -4.079, r = 0.00018, rate_a = 0.0, tap = 1.0, trasf = True, u = 1,
      x = 0.00144, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 230.0, Vn2 = 230.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 164759,
      bus2 = 320007, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_513", name = "Line EI 514",
      owner = 0, phi = 0, r = 0.00129, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.0103, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 138.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 155948,
      bus2 = 320010, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_EI_1471", name = "Line EI 1472",
      owner = 0, phi = -17.707, r = 0.00014, rate_a = 0.0, tap = 1.0, trasf = True, u = 1,
      x = 0.01232, xcoord = 0, ycoord = 0


Shunt, Sn = 100.0, Vn = 230.0, b = 1.0, bus = 320007, fn = 60.0, g = 0.0, idx = "Shunt_EI_396",
       name = "Shunt EI 397", u = 1
