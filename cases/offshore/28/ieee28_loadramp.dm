# DOME format version 1.0

Bus, Vn = 69.0, idx = 1, name = "Bus 1-1", xcoord = [1.875; 2.925], ycoord = [3.45; 3.45], area = 1
Bus, Vn = 69.0, idx = 2, name = "Bus 1-2", xcoord = [3.975; 5.25], ycoord = [0.825; 0.825], area = 1
Bus, Vn = 69.0, idx = 3, name = "Bus 1-3", xcoord = [8.1; 8.775], ycoord = [0.0; 0.0], area = 1
Bus, Vn = 69.0, idx = 4, name = "Bus 1-4", xcoord = [7.875; 9.075], ycoord = [3.3; 3.3], area = 1
Bus, Vn = 69.0, idx = 5, name = "Bus 1-5", xcoord = [5.925; 6.525], ycoord = [2.625; 2.625], area = 1
Bus, Vn = 13.8, idx = 6, name = "Bus 1-6", xcoord = [5.925; 6.825], ycoord = [3.825; 3.825], area = 1
Bus, Vn = 13.8, idx = 7, name = "Bus 1-7", xcoord = [8.625; 9.075], ycoord = [4.05; 4.05], area = 1
Bus, Vn = 18.0, idx = 8, name = "Bus 1-8", xcoord = [9.6; 9.6], ycoord = [4.5; 4.05], area = 2
Bus, Vn = 13.8, idx = 9, name = "Bus 1-9", xcoord = [8.025; 9.075], ycoord = [4.5; 4.5], area = 2
Bus, Vn = 13.8, idx = 10, name = "Bus 1-10", xcoord = [7.875; 7.35], ycoord = [4.875; 4.875], area = 2
Bus, Vn = 13.8, idx = 11, name = "Bus 1-11", xcoord = [7.05; 6.525], ycoord = [5.025; 5.025], area = 2
Bus, Vn = 13.8, idx = 12, name = "Bus 1-12", xcoord = [4.875; 4.275], ycoord = [5.625; 5.625], area = 2
Bus, Vn = 13.8, idx = 13, name = "Bus 1-13", xcoord = [5.85; 6.6], ycoord = [6.075; 6.075], area = 2
Bus, Vn = 13.8, idx = 14, name = "Bus 1-14", xcoord = [7.875; 7.275], ycoord = [5.625; 5.625], area = 2

Bus, Vn = 69.0, idx = 201, name = "Bus 2-1", xcoord = [1.875; 2.925], ycoord = [3.45; 3.45], area = 1
Bus, Vn = 69.0, idx = 202, name = "Bus 2-2", xcoord = [3.975; 5.25], ycoord = [0.825; 0.825], area = 1
Bus, Vn = 69.0, idx = 203, name = "Bus 2-3", xcoord = [8.1; 8.775], ycoord = [0.0; 0.0], area = 1
Bus, Vn = 69.0, idx = 204, name = "Bus 2-4", xcoord = [7.875; 9.075], ycoord = [3.3; 3.3], area = 1
Bus, Vn = 69.0, idx = 205, name = "Bus 2-5", xcoord = [5.925; 6.525], ycoord = [2.625; 2.625], area = 1
Bus, Vn = 13.8, idx = 206, name = "Bus 2-6", xcoord = [5.925; 6.825], ycoord = [3.825; 3.825], area = 1
Bus, Vn = 13.8, idx = 207, name = "Bus 2-7", xcoord = [8.625; 9.075], ycoord = [4.05; 4.05], area = 1
Bus, Vn = 18.0, idx = 208, name = "Bus 2-8", xcoord = [9.6; 9.6], ycoord = [4.5; 4.05], area = 2
Bus, Vn = 13.8, idx = 209, name = "Bus 2-9", xcoord = [8.025; 9.075], ycoord = [4.5; 4.5], area = 2
Bus, Vn = 13.8, idx = 2010, name = "Bus 2-10", xcoord = [7.875; 7.35], ycoord = [4.875; 4.875], area = 2
Bus, Vn = 13.8, idx = 2011, name = "Bus 2-11", xcoord = [7.05; 6.525], ycoord = [5.025; 5.025], area = 2
Bus, Vn = 13.8, idx = 2012, name = "Bus 2-12", xcoord = [4.875; 4.275], ycoord = [5.625; 5.625], area = 2
Bus, Vn = 13.8, idx = 2013, name = "Bus 2-13", xcoord = [5.85; 6.6], ycoord = [6.075; 6.075], area = 2
Bus, Vn = 13.8, idx = 2014, name = "Bus 2-14", xcoord = [7.875; 7.275], ycoord = [5.625; 5.625], area = 2

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

Line, Vn = 69.0, Vn2 = 69.0, b = 0.0528, bus1 = 201, bus2 = 202,
      idx = "Line_1", name = "Line 1", r = 0.01938, x = 0.05917, xcoord = [2.025; 2.025; 4.2; 4.2],
      ycoord = [3.45; 3.15; 1.2; 0.825]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0492, bus1 = 201, bus2 = 205,
      idx = "Line_2", name = "Line 2", r = 0.05403, x = 0.22304, xcoord = [2.775; 2.775; 5.7; 6.075; 6.075],
      ycoord = [3.45; 3.15; 2.325; 2.325; 2.625]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0438, bus1 = 202, bus2 = 203,
      idx = "Line_3", name = "Line 3", r = 0.04699, x = 0.19797, xcoord = [5.1; 5.1; 8.25; 8.25],
      ycoord = [0.825; 0.6; 0.225; 0.0]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0374, bus1 = 202, bus2 = 204,
      idx = "Line_4", name = "Line 4", r = 0.05811, x = 0.17632, xcoord = [5.1; 5.1; 8.4; 8.4],
      ycoord = [0.825; 1.2; 3.15; 3.3]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.034, bus1 = 202, bus2 = 205,
      idx = "Line_5", name = "Line 5", r = 0.05695, x = 0.17388, xcoord = [4.725; 4.725; 6.225; 6.225],
      ycoord = [0.825; 1.2; 2.1; 2.625]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0346, bus1 = 203, bus2 = 204,
      idx = "Line_6", name = "Line 6", r = 0.06701, x = 0.17103, xcoord = [8.55; 8.55],
      ycoord = [0.0; 3.3]
Line, Vn = 69.0, Vn2 = 69.0, b = 0.0128, bus1 = 204, bus2 = 205,
      idx = "Line_7", name = "Line 7", r = 0.01335, x = 0.04211, xcoord = [6.375; 6.375; 8.025; 8.025],
      ycoord = [2.625; 2.175; 3.15; 3.3]
Line, Vn = 69.0, Vn2 = 13.8, bus1 = 204, bus2 = 207, idx = "Line_8",
      name = "Line 8", tap = 0.978, trasf = True, x = 0.20912,
      xcoord = [8.85; 8.85], ycoord = [3.3; 4.05]
Line, Vn = 69.0, Vn2 = 13.8, bus1 = 204, bus2 = 209, idx = "Line_9",
      name = "Line 9", tap = 0.969, trasf = True, x = 0.55618,
      xcoord = [8.25; 8.25], ycoord = [4.5; 3.3]
Line, Vn = 69.0, Vn2 = 13.8, bus1 = 205, bus2 = 206, idx = "Line_10",
      name = "Line 10", tap = 0.932, trasf = True, x = 0.25202,
      xcoord = [6.225; 6.225], ycoord = [2.625; 3.825]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 206, bus2 = 2011, idx = "Line_11",
      name = "Line 11", r = 0.09498, x = 0.19890, xcoord = [6.45; 6.45; 6.675; 6.675], ycoord = [3.825; 4.05; 4.8; 5.025]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 206, bus2 = 2012, idx = "Line_12",
      name = "Line 12", r = 0.12291, x = 0.25581, xcoord = [4.425; 4.425; 6.075; 6.075], ycoord = [5.625; 5.4; 4.05; 3.825]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 206, bus2 = 2013, idx = "Line_13",
      name = "Line 13", r = 0.06615, x = 0.13027, xcoord = [6.225; 6.225], ycoord = [3.825; 6.075]
Line, Vn = 13.8, Vn2 = 18.0, bus1 = 207, bus2 = 208, idx = "Line_14",
      name = "Line 14", trasf = True, x = 0.17615, xcoord = [9.6; 8.925; 8.925],
      ycoord = [4.275; 4.275; 4.05]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 207, bus2 = 209, idx = "Line_15",
      name = "Line 15", x = 0.11001, xcoord = [8.775; 8.775], ycoord = [4.05; 4.5]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 209, bus2 = 2010, idx = "Line_16",
      name = "Line 16", r = 0.03181, x = 0.08450, xcoord = [7.725; 7.725; 8.25; 8.25], ycoord = [4.875; 4.725; 4.65; 4.5]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 209, bus2 = 2014, idx = "Line_17",
      name = "Line 17", r = 0.12711, x = 0.27038, xcoord = [8.55; 8.55; 7.65; 7.65], ycoord = [4.5; 4.8; 5.25; 5.625]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 2010, bus2 = 2011, idx = "Line_18",
      name = "Line 18", r = 0.08205, x = 0.19207, xcoord = [6.9; 6.9; 7.5; 7.5], ycoord = [5.025; 4.8; 4.725; 4.875]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 2012, bus2 = 2013, idx = "Line_19",
      name = "Line 19", r = 0.22092, x = 0.19988, xcoord = [6.0; 6.0; 5.025; 4.725; 4.725], ycoord = [6.075; 5.925; 5.4; 5.4; 5.625]
Line, Vn = 13.8, Vn2 = 13.8, bus1 = 2013, bus2 = 2014, idx = "Line_20",
      name = "Line 20", r = 0.17093, x = 0.34802, xcoord = [7.5; 7.5; 6.45; 6.45], ycoord = [5.625; 5.325; 5.925; 6.075]


BusFreq, idx = 1, bus = 1, name = "BusFreq 1-1"
BusFreq, idx = 2, bus = 2, name = "BusFreq 1-2"
BusFreq, idx = 3, bus = 3, name = "BusFreq 1-3"
BusFreq, idx = 4, bus = 4, name = "BusFreq 1-4"
BusFreq, idx = 5, bus = 5, name = "BusFreq 1-5"
BusFreq, idx = 6, bus = 6, name = "BusFreq 1-6"
BusFreq, idx = 7, bus = 7, name = "BusFreq 1-7"
BusFreq, idx = 8, bus = 8, name = "BusFreq 1-8"
BusFreq, idx = 9, bus = 9, name = "BusFreq 1-9"
BusFreq, idx = 10, bus = 10, name = "BusFreq 1-10"
BusFreq, idx = 11, bus = 11, name = "BusFreq 1-11"
BusFreq, idx = 12, bus = 12, name = "BusFreq 1-12"
BusFreq, idx = 13, bus = 13, name = "BusFreq 1-13"
BusFreq, idx = 14, bus = 14, name = "BusFreq 1-14"


BusFreq, idx = 201, bus = 201, name = "BusFreq 2-1"
BusFreq, idx = 202, bus = 202, name = "BusFreq 2-2"
BusFreq, idx = 203, bus = 203, name = "BusFreq 2-3"
BusFreq, idx = 204, bus = 204, name = "BusFreq 2-4"
BusFreq, idx = 205, bus = 205, name = "BusFreq 2-5"
BusFreq, idx = 206, bus = 206, name = "BusFreq 2-6"
BusFreq, idx = 207, bus = 207, name = "BusFreq 2-7"
BusFreq, idx = 208, bus = 208, name = "BusFreq 2-8"
BusFreq, idx = 209, bus = 209, name = "BusFreq 2-9"
BusFreq, idx = 2010, bus = 2010, name = "BusFreq 2-10"
BusFreq, idx = 2011, bus = 2011, name = "BusFreq 2-11"
BusFreq, idx = 2012, bus = 2012, name = "BusFreq 2-12"
BusFreq, idx = 2013, bus = 2013, name = "BusFreq 2-13"
BusFreq, idx = 2014, bus = 2014, name = "BusFreq 2-14"

PQ, Vn = 69.0, bus = 2, idx = "PQ load_1-1", name = "PQ_Bus_2", p = 0.217,
    q = 0.127
PQ, Vn = 69.0, bus = 3, idx = "PQ load_1-2", name = "PQ_Bus_3", p = 0.942,
    q = 0.19
PQ, Vn = 69.0, bus = 4, idx = "PQ load_1-3", name = "PQ Bus 4", p = 0.478,
    q = -0.039
PQ, Vn = 69.0, bus = 5, idx = "PQ load_1-4", name = "PQ Bus 5", p = 0.076,
    q = 0.016
PQ, Vn = 13.8, bus = 6, idx = "PQ load_1-5", name = "PQ Bus 6", p = 0.112,
    q = 0.075
PQ, Vn = 13.8, bus = 9, idx = "PQ load_1-6", name = "PQ Bus 9", p = 0.295,
    q = 0.166
PQ, Vn = 13.8, bus = 10, idx = "PQ load_1-7", name = "PQ Bus 10", p = 0.09,
    q = 0.058
PQ, Vn = 13.8, bus = 11, idx = "PQ load_1-8", name = "PQ Bus 11", p = 0.035,
    q = 0.018
PQ, Vn = 13.8, bus = 12, idx = "PQ load_1-9", name = "PQ Bus 12", p = 0.061,
    q = 0.016
PQ, Vn = 13.8, bus = 13, idx = "PQ load_1-10", name = "PQ Bus 13", p = 0.135,
    q = 0.058
PQ, Vn = 13.8, bus = 14, idx = "PQ load_1-11", name = "PQ Bus 14", p = 0.149,
    q = 0.05

PQ, Vn = 69.0, bus = 202, idx = "PQ load_2-1", name = "PQ_Bus_2", p = 0.217,
    q = 0.127
PQ, Vn = 69.0, bus = 203, idx = "PQ load_2-2", name = "PQ_Bus_3", p = 0.942,
    q = 0.19
PQ, Vn = 69.0, bus = 204, idx = "PQ load_2-3", name = "PQ Bus 4", p = 0.478,
    q = -0.039
PQ, Vn = 69.0, bus = 205, idx = "PQ load_2-4", name = "PQ Bus 5", p = 0.076,
    q = 0.016
PQ, Vn = 13.8, bus = 206, idx = "PQ load_2-5", name = "PQ Bus 6", p = 0.112,
    q = 0.075
PQ, Vn = 13.8, bus = 209, idx = "PQ load_2-6", name = "PQ Bus 9", p = 0.295,
    q = 0.166
PQ, Vn = 13.8, bus = 2010, idx = "PQ load_2-7", name = "PQ Bus 10", p = 0.09,
    q = 0.058
PQ, Vn = 13.8, bus = 2011, idx = "PQ load_2-8", name = "PQ Bus 11", p = 0.035,
    q = 0.018
PQ, Vn = 13.8, bus = 2012, idx = "PQ load_2-9", name = "PQ Bus 12", p = 0.061,
    q = 0.016
PQ, Vn = 13.8, bus = 2013, idx = "PQ load_2-10", name = "PQ Bus 13", p = 0.135,
    q = 0.058
PQ, Vn = 13.8, bus = 2014, idx = "PQ load_2-11", name = "PQ Bus 14", p = 0.149,
    q = 0.05


PV, Vn = 69.0, bus = 2, busr = 2, idx = 2, name = "PV Bus 2",
    pg = 0.7, pmax = 1.0, pmin = 0, qmax = 0.4, qmin = -0.4,
    v0 = 1.045
PV, Vn = 69.0, bus = 3, busr = 3, idx = 3, name = "PV Bus 3",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.4, v0 = 1.01
PV, Vn = 13.8, bus = 6, busr = 6, idx = 6, name = "PV Bus 6",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.24, qmin = -0.06, v0 = 1.07
PV, Vn = 18.0, bus = 8, busr = 8, idx = 8, name = "PV Bus 8",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.24, qmin = -0.06, v0 = 1.09

PV, Vn = 69.0, bus = 202, busr = 202, idx = 202, name = "PV Bus 2-2",
    pg = 0.7, pmax = 1.0, pmin = 0, qmax = 0.4, qmin = -0.4,
    v0 = 1.045
PV, Vn = 69.0, bus = 203, busr = 203, idx = 203, name = "PV Bus 2-3",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.4, v0 = 1.01
PV, Vn = 13.8, bus = 206, busr = 206, idx = 206, name = "PV Bus 2-6",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.24, qmin = -0.06, v0 = 1.07
PV, Vn = 18.0, bus = 208, busr = 208, idx = 208, name = "PV Bus 2-8",
    pg = 0, pmax = 1.0, pmin = 0, qmax = 0.24, qmin = -0.06, v0 = 1.09


Shunt, Vn = 13.8, b = 0.19, bus = 9, idx = 1, name = "Shunt_Bus_9"

Shunt, Vn = 13.8, b = 0.19, bus = 209, idx = 201, name = "Shunt_Bus_2-9"

SW, Vn = 69.0, bus = 1, busr = 1, idx = 1, name = "SW_Bus_1",
    pmax = 999.9, pmin = -999.9, qmax = 9.9, qmin = -9.9,
    v0 = 1.06

SW, Vn = 69.0, bus = 201, busr = 201, idx = 201, name = "SW_Bus_2-1",
    pmax = 999.9, pmin = -999.9, qmax = 9.9, qmin = -9.9,
    v0 = 1.06

Syn6a, D = 2.0, M = 10.296, Sn = 615.0, Td10 = 7.4, Td20 = 0.03,
       Tq10 = 1.8, Tq20 = 0.033, Vn = 69.0, bus = 1, fn = 60.0,
       gen = 1, idx = 1, name = "Syn 1-1", ra = 0.0031, xd = 0.8979,
       xd1 = 0.6, xd2 = 0.23, xl = 0.2396, xq = 1.10, xq1 = 0.646,
       xq2 = 0.4
Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 2, fn = 60.0,
       gen = 2, idx = 2, name = "Syn 1-2", ra = 0.0031, xd = 1.05,
       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13
Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 3, fn = 60.0,
       gen = 3, idx = 3, name = "Syn 1-3", ra = 0.0031, xd = 1.05,
       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13
Syn6a, D = 2.0, M = 10.12, Sn = 25.0, Td10 = 4.75, Td20 = 0.06,
       Tq10 = 1.5, Tq20 = 0.21, Vn = 13.8, bus = 6, fn = 60.0,
       gen = 6, idx = 4, name = "Syn 1-4", ra = 0.0041, xd = 1.25,
       xd1 = 0.232, xd2 = 0.12, xl = 0.134, xq = 1.22, xq1 = 0.715,
       xq2 = 0.12
Syn6a, D = 2.0, M = 10.12, Sn = 25.0, Td10 = 4.75, Td20 = 0.06,
       Tq10 = 1.5, Tq20 = 0.21, Vn = 18.0, bus = 8, fn = 60.0,
       gen = 8, idx = 5, name = "Syn 1-5", ra = 0.0041, xd = 1.25,
       xd1 = 0.232, xd2 = 0.12, xl = 0.134, xq = 1.22, xq1 = 0.715,
       xq2 = 0.12

Syn6a, D = 2.0, M = 10.296, Sn = 615.0, Td10 = 7.4, Td20 = 0.03,
       Tq10 = 1.8, Tq20 = 0.033, Vn = 69.0, bus = 201, fn = 60.0,
       gen = 201, idx = 201, name = "Syn 2-1", ra = 0.0031, xd = 0.8979,
       xd1 = 0.6, xd2 = 0.23, xl = 0.2396, xq = 1.10, xq1 = 0.646,
       xq2 = 0.4
Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 202, fn = 60.0,
       gen = 202, idx = 202, name = "Syn 2-2", ra = 0.0031, xd = 1.05,
       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13
Syn6a, D = 2.0, M = 13.08, Sn = 60.0, Td10 = 6.1, Td20 = 0.04,
       Tq10 = 0.3, Tq20 = 0.099, Vn = 69.0, bus = 203, fn = 60.0,
       gen = 203, idx = 203, name = "Syn 2-3", ra = 0.0031, xd = 1.05,
       xd1 = 0.185, xd2 = 0.13, xq = 0.98, xq1 = 0.36, xq2 = 0.13
Syn6a, D = 2.0, M = 10.12, Sn = 25.0, Td10 = 4.75, Td20 = 0.06,
       Tq10 = 1.5, Tq20 = 0.21, Vn = 13.8, bus = 206, fn = 60.0,
       gen = 206, idx = 204, name = "Syn 2-4", ra = 0.0041, xd = 1.25,
       xd1 = 0.232, xd2 = 0.12, xl = 0.134, xq = 1.22, xq1 = 0.715,
       xq2 = 0.12
Syn6a, D = 2.0, M = 10.12, Sn = 25.0, Td10 = 4.75, Td20 = 0.06,
       Tq10 = 1.5, Tq20 = 0.21, Vn = 18.0, bus = 208, fn = 60.0,
       gen = 208, idx = 205, name = "Syn 2-5", ra = 0.0041, xd = 1.25,
       xd1 = 0.232, xd2 = 0.12, xl = 0.134, xq = 1.22, xq1 = 0.715,
       xq2 = 0.12

TG1, gen = 1, pmax = 5, pmin = 0, R = 0.01, wref0 = 1.0,
     T3 = 0, T4 = 12.0, T5 = 50.0, Tc = 0.56, Ts = 0.1
TG1, gen = 2, pmax = 2, pmin = 0, R = 0.01, wref0 = 1.0,
     T3 = 0, T4 = 12.0, T5 = 50.0, Tc = 0.56, Ts = 0.1

TG1, gen = 201, pmax = 5, pmin = 0, R = 0.01, wref0 = 1.0,
     T3 = 0, T4 = 12.0, T5 = 50.0, Tc = 0.56, Ts = 0.1
TG1, gen = 202, pmax = 2, pmin = 0, R = 0.01, wref0 = 1.0,
     T3 = 0, T4 = 12.0, T5 = 50.0, Tc = 0.56, Ts = 0.1

AVR1, Ka = 100.0, Kf = 0.002, Ta = 0.02, Te = 0.2, Tf = 1.0, Tr = 0.001,
      idx = 1, name = "AVR 1-1", syn = 1, vrmax = 7.32,
      vrmin = 0
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 1.98, Tf = 1.0, Tr = 0.001,
      idx = 2, name = "AVR 1-2", syn = 2, vrmax = 4.38,
      vrmin = 0.0
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 1.98, Tf = 1.0, Tr = 0.001,
      idx = 3, name = "AVR 1-3", syn = 3, vrmax = 4.38,
      vrmin = 0.0
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 0.7, Tf = 1.0, Tr = 0.001,
      idx = 4, name = "AVR 1-4", syn = 4, vrmax = 6.81,
      vrmin = 1.395
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 0.7, Tf = 1.0, Tr = 0.001,
      idx = 5, name = "AVR 1-5", syn = 5, vrmax = 6.81,
      vrmin = 1.395

AVR1, Ka = 100.0, Kf = 0.002, Ta = 0.02, Te = 0.2, Tf = 1.0, Tr = 0.001,
      idx = 201, name = "AVR 2-1", syn = 201, vrmax = 7.32,
      vrmin = 0
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 1.98, Tf = 1.0, Tr = 0.001,
      idx = 202, name = "AVR 2-2", syn = 202, vrmax = 4.38,
      vrmin = 0.0
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 1.98, Tf = 1.0, Tr = 0.001,
      idx = 203, name = "AVR 2-3", syn = 203, vrmax = 4.38,
      vrmin = 0.0
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 0.7, Tf = 1.0, Tr = 0.001,
      idx = 204, name = "AVR 2-4", syn = 204, vrmax = 6.81,
      vrmin = 1.395
AVR1, Ka = 20.0, Kf = 0.001, Ta = 0.02, Te = 0.7, Tf = 1.0, Tr = 0.001,
      idx = 205, name = "AVR 2-5", syn = 205, vrmax = 6.81,
      vrmin = 1.395


Node, idx = 0, name = "Node 0", Vdcn = 100.0
Node, idx = 1, name = "Node 1", Vdcn = 100.0
Node, idx = 2, name = "Node 2", Vdcn = 100.0
Node, idx = 3, name = "Node 3", Vdcn = 100.0
Node, idx = 4, name = "Node 4", Vdcn = 100.0

Ground, idx = 0, name = "Ground 1", node = 0, Vdcn = 100.0, voltage = 0

VSC, idx = 1, node1 = 1, node2 = 0, bus = 1, Vn = 69, name = "VSC 1", rsh = 0.0025, xsh = 0.05,
     vshmax = 999, vshmin = 0, Ishmax = 999, vref0 = 1.06, vdcref0 = 1.0, control = "vV",
     Vdcn = 100, u = 1
VSC, idx = 2, node1 = 2, node2 = 0, bus = 202, Vn = 69, name = "VSC 2", rsh = 0.0025, xsh = 0.06,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = 0.4, qref0 = -0.00, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.99, Vdcn = 100, u = 1

VSC1_IE, vsc = 1, name = "VSC 1", Kp1 = 0.2, Ki1 = 0.1, Kp2 = 1, Ki2 = 0.2,
      Kp3 = 1, Ki3 = 0.5, Kp = 0, Kf = -0, Ki = -100
VSC1_IE, vsc = 2, name = "VSC 2", Kp1 = 0.2, Ki1 = 0.1, Kp2 = 0.2, Ki2 = 0.,
      Kp3 = 0.01, Ki3 = 0., Kdc = 10, Kf = 0, Ki = 100

RLs, idx = "RLs1", name = "RLs 1-2", node1 = 1, node2 = 2, Vdcn = 100, R = 0.1, L = 0.0001
RLs, idx = "RLs2", name = "RLs 2-4", node1 = 2, node2 = 4, Vdcn = 100, R = 0.1, L = 0.0001
RLs, idx = "RLs3", name = "RLs 3-4", node1 = 3, node2 = 4, Vdcn = 100, R = 0.1, L = 0.0001
RLs, idx = "RLs4", name = "RLs 1-3", node1 = 1, node2 = 3, Vdcn = 100, R = 0.1, L = 0.0001

C, idx = "C1", name = "C 1", node1 = 1, node2 = 0, Vdcn = 100, C = 0.00001
C, idx = "C2", name = "C 2", node1 = 2, node2 = 0, Vdcn = 100, C = 0.00001
C, idx = "C3", name = "C 3", node1 = 3, node2 = 0, Vdcn = 100, C = 0.00001
C, idx = "C4", name = "C 4", node1 = 4, node2 = 0, Vdcn = 100, C = 0.00001

DCgen, idx = "DCgen_3", Vdcn = 100, node1 = 3, node2 = 0, P = 0.4
DCgen, idx = "DCgen_4", Vdcn = 100, node1 = 4, node2 = 0, P = 0.4

ConstWind, idx = 3
ConstWind, idx = 4

# WTG4DC, node1 = 3, node2 = 0, dcgen = "DCgen_3", wind = 3
# WTG4DC, node1 = 4, node2 = 0, dcgen = "DCgen_4", wind = 4

WTG4DC, node1 = 3, node2 = 0, dcgen = "DCgen_3", wind = 3, busfreq = 1, coi = 1,
        Kdc = 0, Ki = 00, Kcoi = 0
WTG4DC, node1 = 4, node2 = 0, dcgen = "DCgen_4", wind = 4, busfreq = 2, coi = 1,
        Kdc = 0, Ki = 00, Kcoi = 0

COI, idx = 1, name = "COI 1", syn = [1; 2]