
Node, idx = 0, name = "Node 0", Vdcn = 100.0
Node, idx = 1, name = "Node 1", Vdcn = 100.0
Node, idx = 2, name = "Node 2", Vdcn = 100.0
Node, idx = 3, name = "Node 3", Vdcn = 100.0
Node, idx = 4, name = "Node 4", Vdcn = 100.0

Ground, idx = 0, name = "Ground 1", node = 0, Vdcn = 100.0, voltage = 0

VSC, idx = 1, node1 = 1, node2 = 0, bus = 22, Vn = 1, name = "VSC 1", rsh = 0.0025, xsh = 0.5,
     vshmax = 999, vshmin = 0, Ishmax = 999, vref0 = 1.05, vdcref0 = 1.0, control = "vV",
     Vdcn = 100, u = 1
VSC, idx = 2, node1 = 2, node2 = 0, bus = 29, Vn = 1, name = "VSC 2", rsh = 0.0025, xsh = 0.5,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = -1.5, qref0 = -0.00, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.99, Vdcn = 100, u = 1
VSC, idx = 3, node1 = 3, node2 = 0, bus = 6, Vn = 1, name = "VSC 3", rsh = 0.0025, xsh = 0.5,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = 1.5, qref0 = 0.00, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.995, Vdcn = 100, u = 1
VSC, idx = 4, node1 = 4, node2 = 0, bus = 26, Vn = 1, name = "VSC 4", rsh = 0.0025, xsh = 0.5,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = 1, qref0 = 0.00, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.995, Vdcn = 100, u = 1

VSC1, vsc = 1, name = "VSC 1", Kp1 = 0.4, Ki1 = 0.02, Kp2 = 3, Ki2 = 1,
      Kp3 = 0.5, Ki3 = 0.2
VSC2B, vsc = 2, name = "VSC 2", Kp1 = 0.1, Ki1 = 0.04, Kp2 = 0.5, Ki2 = 0.01,
      Kp3 = 0.6, Ki3 = 0.04, D = 4, M = 4, KQ = 0
VSC2B, vsc = 3, name = "VSC 3", Kp1 = 0.1, Ki1 = 0.01, Kp2 = 0.04, Ki2 = 0.00,
      Kp3 = 0.02, Ki3 = 0.00, D = 10, M = 6
VSC2B, vsc = 4, name = "VSC 4", Kp1 = 0.1, Ki1 = 0.04, Kp2 = 0.2, Ki2 = 0.001,
      Kp3 = 0.02, Ki3 = 0.00, D = 8, M = 6

RLs, idx = "RLs1", name = "RLs 1-2", node1 = 1, node2 = 2, Vdcn = 100, R = 0.1, L = 0.001
RLs, idx = "RLs2", name = "RLs 2-3", node1 = 2, node2 = 3, Vdcn = 100, R = 0.1, L = 0.001
RLs, idx = "RLs3", name = "RLs 3-4", node1 = 3, node2 = 4, Vdcn = 100, R = 0.1, L = 0.001
RLs, idx = "RLs4", name = "RLs 1-3", node1 = 1, node2 = 3, Vdcn = 100, R = 0.1, L = 0.001

C, idx = "C1", name = "C 1", node1 = 1, node2 = 0, Vdcn = 100, C = 0.001
C, idx = "C2", name = "C 2", node1 = 2, node2 = 0, Vdcn = 100, C = 0.001
C, idx = "C3", name = "C 3", node1 = 3, node2 = 0, Vdcn = 100, C = 0.001
C, idx = "C4", name = "C 4", node1 = 4, node2 = 0, Vdcn = 100, C = 0.001
