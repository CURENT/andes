# DOME format version 1.0

# DC NETWORK DEFINITION

Node, idx = 0, name = "Node 0", Vdcn = 100.0
Node, idx = 1, name = "Node 1", Vdcn = 100.0
Node, idx = 2, name = "Node 2", Vdcn = 100.0

Ground, idx = 0, name = "Ground 0", node = 0, Vdcn = 100.0, voltage = 0

VSC, idx = 1, node1 = 1, node2 = 0, bus = 2189, Vn = 138, name = "VSC 1", rsh = 0.0025, xsh = 0.06,
     vshmax = 999, vshmin = 0, Ishmax = 999, vref0 = 1.03, vdcref0 = 1.0, control = "vV",
     Vdcn = 100, u = 1
VSC, idx = 2, node1 = 2, node2 = 0, bus = 3, Vn = 20, name = "VSC 2", rsh = 0.0025, xsh = 0.06,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = 0.1, qref0 = 0.01, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.99, Vdcn = 100, u = 1

VSC1, vsc = 1, name = "VSC 1", Kp1 = 1, Ki1 = 5, Kp2 = 1, Ki2 = 5,
      Kp3 = 1, Ki3 = 0.5
VSC2B, vsc = 2, name = "VSC 2", Kp1 =0.5, Ki1 = 0.2, Kp2 = 0.5, Ki2 = 0.1,
      Kp3 = 0.2, Ki3 = 0, D = 3, M = 3

RLs, idx = "RLs1", name = "RLs 1-2", node1 = 1, node2 = 2, Vdcn = 100, R = 0.1, L = 0.01

C, idx = "C1", name = "C 1", node1 = 1, node2 = 0, Vdcn = 100, C = 1
C, idx = "C1", name = "C 2", node1 = 2, node2 = 0, Vdcn = 100, C = 1
