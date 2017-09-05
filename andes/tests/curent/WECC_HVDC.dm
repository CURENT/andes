# DOME format version 1.0

# WECC SYSTEM HVDC CONVERTER STATION AC CONNECTIONS
# WECC HVDC station buses: 2191, 2189, 2190, 2192

Bus, Vn = 345.0, angle = -0.53362, area = 10, idx = 2183, name = "NEXMEXIC", owner = 1, region = 10,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.05366, xcoord = 33.133383, ycoord = -106.216183
Bus, Vn = 345.0, angle = -0.53362, area = 10, idx = 2184, name = "NEXMEXIC1", owner = 1, region = 10,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.05366, xcoord = 33.082211, ycoord = -106.180769
Bus, Vn = 138.0, angle = -0.39605, area = 60, idx = 2185, name = "WYOMING1", owner = 1, region = 63,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.08886, xcoord = 41.12, ycoord = -104.15
Bus, Vn = 138.0, angle = -0.39605, area = 60, idx = 2186, name = "WYOMING2", owner = 1, region = 63,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.08886, xcoord = 42.02, ycoord = -104.15
Bus, Vn = 138.0, angle = 0.00586, area = 50, idx = 2187, name = "ALBERTA2", owner = 1, region = 50,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.03538, xcoord = 50.68, ycoord = -110.14
Bus, Vn = 138.0, angle = 0.39447, area = 60, idx = 2188, name = "MONTANA1", owner = 1, region = 62,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.05793, xcoord = 46.41, ycoord = -105.99
Bus, Vn = 138.0, angle = -0.39605, area = 60, idx = 2189, name = "WYOMING3", owner = 1, region = 63,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.08886, xcoord = 41.82, ycoord = -104.15
Bus, Vn = 138.0, angle = 0.00586, area = 50, idx = 2190, name = "ALBERTA3", owner = 1, region = 50,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.03538, xcoord = 50.48, ycoord = -110.14
Bus, Vn = 345.0, angle = -0.53362, area = 10, idx = 2191, name = "NEXMEXIC2", owner = 1, region = 10,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.05366, xcoord = 33.332383, ycoord = -107.215183
Bus, Vn = 138.0, angle = 0.39447, area = 60, idx = 2192, name = "MONTANA2", owner = 1, region = 62,
     u = 1, vmax = 1.1, vmin = 0.9, voltage = 1.05793, xcoord = 46.61, ycoord = -105.99

Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0038, b1 = 0.0, b2 = 0.0, bus1 = 7,
      bus2 = 2183, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_1", name = "Line WECC AC HVDC 1",
      owner = 0, phi = 0, r = 0.00358, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.03976, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0038, b1 = 0.0, b2 = 0.0, bus1 = 7,
      bus2 = 2183, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_2", name = "Line WECC AC HVDC 2",
      owner = 0, phi = 0, r = 0.00358, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.03976, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0038, b1 = 0.0, b2 = 0.0, bus1 = 7,
      bus2 = 2184, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_3", name = "Line WECC AC HVDC 3",
      owner = 0, phi = 0, r = 0.00358, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.03976, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0038, b1 = 0.0, b2 = 0.0, bus1 = 7,
      bus2 = 2184, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_4", name = "Line WECC AC HVDC 4",
      owner = 0, phi = 0, r = 0.00358, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.03976, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 345.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 2183,
      bus2 = 2191, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_5", name = "Line WECC AC HVDC 5",
      owner = 0, phi = 0, r = 0.00036, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.00398, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 138.0, Vn2 = 138.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 2185,
      bus2 = 2186, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_6", name = "Line WECC AC HVDC 6",
      owner = 0, phi = 0, r = 0.00036, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.00398, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 138.0, Vn2 = 138.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 2186,
      bus2 = 2189, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_7", name = "Line WECC AC HVDC 7",
      owner = 0, phi = 0, r = 0.00036, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.00398, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 138.0, Vn2 = 138.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 2187,
      bus2 = 2190, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_8", name = "Line WECC AC HVDC 8",
      owner = 0, phi = 0, r = 0.00036, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.00398, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 138.0, Vn2 = 138.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 2188,
      bus2 = 2192, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_9", name = "Line WECC AC HVDC 9",
      owner = 0, phi = 0, r = 0.00036, rate_a = 0.0, tap = 1.0, trasf = False, u = 1,
      x = 0.00398, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 500.0, Vn2 = 138.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 30,
      bus2 = 2187, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_10", name = "Line WECC AC HVDC 10",
      owner = 0, phi = 0.0, r = -0.0, rate_a = 0.0, tap = 1.05, trasf = True, u = 1,
      x = 0.0015, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 500.0, Vn2 = 138.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 65,
      bus2 = 2188, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_11", name = "Line WECC AC HVDC 11",
      owner = 0, phi = 0.0, r = -0.0, rate_a = 0.0, tap = 1.09, trasf = True, u = 1,
      x = 0.0015, xcoord = 0, ycoord = 0
Line, Sn = 100.0, Vn = 345.0, Vn2 = 138.0, b = 0.0, b1 = 0.0, b2 = 0.0, bus1 = 165,
      bus2 = 2185, fn = 60, g = 0.0, g1 = 0.0, g2 = 0.0, idx = "Line_WECC_HVDC_12", name = "Line WECC AC HVDC 12",
      owner = 0, phi = 0.0, r = 0.0005, rate_a = 0.0, tap = 1.0588, trasf = True, u = 1,
      x = 0.0141, xcoord = 0, ycoord = 0
Shunt, Sn = 100.0, Vn = 138.0, b = 1.5, bus = 2185, fn = 60.0, g = 0.0, idx = "Shunt_WECC_HVDC_2185",
       name = "Shunt WECC HVDC 2185", u = 1

