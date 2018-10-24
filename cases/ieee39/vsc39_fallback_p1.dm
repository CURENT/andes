
Node, idx = 0, name = "Node 0", Vdcn = 100.0
Node, idx = 1, name = "Node 1", Vdcn = 100.0
Node, idx = 2, name = "Node 2", Vdcn = 100.0
Node, idx = 3, name = "Node 3", Vdcn = 100.0
Node, idx = 4, name = "Node 4", Vdcn = 100.0

Ground, idx = 0, name = "Ground 1", node = 0, Vdcn = 100.0, voltage = 0

VSC, idx = 1, node1 = 1, node2 = 0, bus = 10, Vn = 1, name = "VSC 1", rsh = 0.01, xsh = 0.4,
     vshmax = 999, vshmin = 0, Ishmax = 999, vref0 = 1.04, vdcref0 = 1.0, control = "vV",
     Vdcn = 100, u = 1
VSC, idx = 2, node1 = 2, node2 = 0, bus = 30, Vn = 1, name = "VSC 2", rsh = 0.01, xsh = 0.4,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = -1, qref0 = -0.00, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.99, Vdcn = 100, u = 1
VSC, idx = 3, node1 = 3, node2 = 0, bus = 8, Vn = 1, name = "VSC 3", rsh = 0.01, xsh = 0.4,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = 0.6, qref0 = 0.00, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.995, Vdcn = 100, u = 1
VSC, idx = 4, node1 = 4, node2 = 0, bus = 16, Vn = 1, name = "VSC 4", rsh = 0.01, xsh = 0.4,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = 0.6, qref0 = 0.00, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.995, Vdcn = 100, u = 1

VSC1, vsc = 1, name = "VSC 1", Kp1 = 0.5, Ki1 = 0.0125, Kp2 = 1, Ki2 = 1.25,
      Kp3 = 0.25, Ki3 = 0.5
VSC2B, vsc = 2, name = "VSC 2", Kp1 = 0.03, Ki1 = 0.05, Kp2 = 0.5, Ki2 = 0.01,
      Kp3 = 0.6, Ki3 = 0.04, D = 12, M = 4, KQ = 0
VSC2B, vsc = 3, name = "VSC 3", Kp1 = 0.04, Ki1 = 0.04, Kp2 = 0.04, Ki2 = 0.00,
      Kp3 = 0.02, Ki3 = 0.001, D = 1, M = 4
VSC2B, vsc = 4, name = "VSC 4", Kp1 = 0.04, Ki1 = 0.04, Kp2 = 0.1, Ki2 = 0.00,
      Kp3 = 0.02, Ki3 = 0.001, D = 0.4, M = 4

RLs, idx = "RLs1", name = "RLs 1-2", node1 = 1, node2 = 2, Vdcn = 100, R = 0.1, L = 0.001
RLs, idx = "RLs2", name = "RLs 2-3", node1 = 2, node2 = 3, Vdcn = 100, R = 0.1, L = 0.001
RLs, idx = "RLs3", name = "RLs 3-4", node1 = 3, node2 = 4, Vdcn = 100, R = 0.1, L = 0.001
RLs, idx = "RLs4", name = "RLs 1-3", node1 = 1, node2 = 3, Vdcn = 100, R = 0.1, L = 0.002

C, idx = "C1", name = "C 1", node1 = 1, node2 = 0, Vdcn = 100, C = 0.0002
C, idx = "C2", name = "C 2", node1 = 2, node2 = 0, Vdcn = 100, C = 0.0002
C, idx = "C3", name = "C 3", node1 = 3, node2 = 0, Vdcn = 100, C = 0.0001
C, idx = "C4", name = "C 4", node1 = 4, node2 = 0, Vdcn = 100, C = 0.0001
