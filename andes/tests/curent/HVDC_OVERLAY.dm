# DOME format version 1.0

# 4-Node HVDC Overlay

Ground, idx = "Ground_COI", name = "Ground COI", node = "Node_0", Vdcn = 193.0, voltage = 0
Node, idx = "Node_0", name = "Node 0", Vdcn = 193.0

Node, idx = "Node_70_P", name = "Node 70 P", Vdcn = 193.0
Node, idx = "Node_59_P", name = "Node 59 P", Vdcn = 193.0
Node, idx = "Node_160002_P", name = "Node 160002 P", Vdcn = 193.0
Node, idx = "Node_30247_P", name = "Node 30247 P", Vdcn = 193.0

Node, idx = "Node_70_N", name = "Node 70 N", Vdcn = -193.0
Node, idx = "Node_59_N", name = "Node 59 N", Vdcn = -193.0
Node, idx = "Node_160002_N", name = "Node 160002 N", Vdcn = -193.0
Node, idx = "Node_30247_N", name = "Node 30247 N", Vdcn = -193.0

RLs, idx = "RLs_70_59_P", name = "RLs 70_59 P", node1 = "Node_70_P", node2 = "Node_59_P", Vdcn = 193.0, R = 5, L = 0.5
RLs, idx = "RLs_160002_70_P", name = "RLs 160002_70 P", node1 = "Node_160002_P", node2 = "Node_70_P", Vdcn = 193.0, R = 5, L = 0.5
RLs, idx = "RLs_160002_30247_P", name = "RLs 160002_30247 P", node1 = "Node_160002_P", node2 = "Node_30247_P", Vdcn = 193.0, R = 5, L = 0.5
RLs, idx = "RLs_30247_59_P", name = "RLs 30247_59 P", node1 = "Node_30247_P", node2 = "Node_59_P", Vdcn = 193.0, R = 5, L = 0.5

RLs, idx = "RLs_70_59_N", name = "RLs 70_59 N", node1 = "Node_70_N", node2 = "Node_59_N", Vdcn = -193.0, R = 5, L = 0.5
RLs, idx = "RLs_160002_70_N", name = "RLs 160002_70 N", node1 = "Node_160002_N", node2 = "Node_70_N", Vdcn = -193.0, R = 5, L = 0.5
RLs, idx = "RLs_160002_30247_N", name = "RLs 160002_30247 N", node1 = "Node_160002_N", node2 = "Node_30247_N", Vdcn = -193.0, R = 5, L = 0.5
RLs, idx = "RLs_30247_59_N", name = "RLs 30247_59 N", node1 = "Node_30247_N", node2 = "Node_59_N", Vdcn = -193.0, R = 5, L = 0.5

C, idx = "C_70_P", name = "C 70 P", node1 = "Node_70_P", node2 = "Node_0", Vdcn = 193.0, C = 100
C, idx = "C_59_P", name = "C 59 P", node1 = "Node_59_P", node2 = "Node_0", Vdcn = 193.0, C = 100
C, idx = "C_160002_P", name = "C 160002 P", node1 = "Node_160002_P", node2 = "Node_0", Vdcn = 193.0, C = 100
C, idx = "C_30247_P", name = "C 30247 P", node1 = "Node_30247_P", node2 = "Node_0", Vdcn = 193.0, C = 100

C, idx = "C_70_N", name = "C 70 N", node1 = "Node_70_N", node2 = "Node_0", Vdcn = 193.0, C = 100
C, idx = "C_59_N", name = "C 59 N", node1 = "Node_59_N", node2 = "Node_0", Vdcn = 193.0, C = 100
C, idx = "C_160002_N", name = "C 160002 N", node1 = "Node_160002_N", node2 = "Node_0", Vdcn = 193.0, C = 100
C, idx = "C_30247_N", name = "C 30247 N", node1 = "Node_30247_N", node2 = "Node_0", Vdcn = 193.0, C = 100

VSC, idx = "VSC_70_P", name = "VSC 70_P", bus = 70, node1 = "Node_70_P", node2 = "Node_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.025, control = "vV", Vdcn = 193, u = 1
VSC, idx = "VSC_59_P", name = "VSC 59_P", bus = 59, node1 = "Node_59_P", node2 = "Node_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, vref0 = 1.03, control = "PV", Vdcn = 193, u = 1
VSC, idx = "VSC_160002_P", name = "VSC 160002_P", bus = 160002, node1 = "Node_160002_P", node2 = "Node_0",
     Vn = 26,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = "VSC_30247_P", name = "VSC 30247_P", bus = 30247, node1 = "Node_30247_P", node2 = "Node_0",
     Vn = 20,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1

VSC, idx = "VSC_70_N", name = "VSC 70_N", bus = 70, node1 = "Node_70_N", node2 = "Node_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.025, control = "vV", Vdcn = 193, u = 1
VSC, idx = "VSC_59_N", name = "VSC 59_N", bus = 59, node1 = "Node_59_N", node2 = "Node_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, vref0 = 1.03, control = "PV", Vdcn = 193, u = 1
VSC, idx = "VSC_160002_N", name = "VSC 160002_N", bus = 160002, node1 = "Node_160002_N", node2 = "Node_0",
     Vn = 26,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = "VSC_30247_N", name = "VSC 30247_N", bus = 30247, node1 = "Node_30247_N", node2 = "Node_0",
     Vn = 20,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1

VSC1, vsc = "VSC_70_P", name = "VSC 70 P", Kp1 = 0.05, Ki1 = 0.01, Kp2 = 20, Ki2 = 1,
      Kp3 = 1, Ki3 = 0.2
VSC1, vsc = "VSC_59_P", name = "VSC 59 P", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1,
      Kp3 = 1, Ki3 = 1
VSC1, vsc = "VSC_160002_P", name = "VSC 160002 P", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1,
      Kp3 = 1, Ki3 = 1
VSC1, vsc = "VSC_30247_P", name = "VSC 30247 P", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1,
      Kp3 = 1, Ki3 = 1

VSC1, vsc = "VSC_70_N", name = "VSC 70 N", Kp1 = 0.05, Ki1 = 0.01, Kp2 = 20, Ki2 = 1,
      Kp3 = 1, Ki3 = 0.2
VSC1, vsc = "VSC_59_N", name = "VSC 59 N", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1,
      Kp3 = 1, Ki3 = 1
VSC1, vsc = "VSC_160002_N", name = "VSC 160002 N", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1,
      Kp3 = 1, Ki3 = 1
VSC1, vsc = "VSC_30247_N", name = "VSC 30247 N", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1,
      Kp3 = 1, Ki3 = 1
