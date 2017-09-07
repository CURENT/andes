# DOME format version 1.0

# 4-Node HVDC Overlay

Ground, idx = 0, name = "Ground COI", node = 0, Vdcn = 193.0, voltage = 0
Node, idx = 0, name = "Node 0", Vdcn = 193.0, xcoord = 45.596416, ycoord = -121.312202

Node, idx = 1, name = "Node 70 P", Vdcn = 193.0, xcoord = 45.596416, ycoord = -121.312202, area = 90, region = 90
Node, idx = 2, name = "Node 59 P", Vdcn = 193.0, xcoord = 34.312456, ycoord = -118.481217, area = 90, region = 90
Node, idx = 3, name = "Node 160002 P", Vdcn = 193.0, xcoord = 45.37, ycoord = -93.9, area = 600, region = 600
Node, idx = 4, name = "Node 30247 P", Vdcn = 193.0, xcoord = 29.7482, ycoord = -94.924983, area = 4, region = 305

Node, idx = 201, name = "Node 70 N", Vdcn = -193.0, xcoord = 45.596416, ycoord = -121.312202, area = 90, region = 90
Node, idx = 202, name = "Node 59 N", Vdcn = -193.0, xcoord = 34.312456, ycoord = -118.481217, area = 90, region = 90
Node, idx = 203, name = "Node 160002 N", Vdcn = -193.0, xcoord = 45.37, ycoord = -93.9, area = 600, region = 600
Node, idx = 204, name = "Node 30247 N", Vdcn = -193.0, xcoord = 29.7482, ycoord = -94.924983, area = 4, region = 305

RLs, idx = 1, name = "RLs 70_59 P", node1 = 1, node2 = 2, Vdcn = 193.0, R = 5, L = 0.5
RLs, idx = 2, name = "RLs 160002_70 P", node1 = 3, node2 = 1, Vdcn = 193.0, R = 5, L = 0.5
RLs, idx = 3, name = "RLs 160002_30247 P", node1 = 3, node2 = 4, Vdcn = 193.0, R = 5, L = 0.5
RLs, idx = 4, name = "RLs 30247_59 P", node1 = 4, node2 = 2, Vdcn = 193.0, R = 5, L = 0.5

RLs, idx = 5, name = "RLs 70_59 N", node1 = 201, node2 = 202, Vdcn = -193.0, R = 5, L = 0.5
RLs, idx = 6, name = "RLs 160002_70 N", node1 = 203, node2 = 201, Vdcn = -193.0, R = 5, L = 0.5
RLs, idx = 7, name = "RLs 160002_30247 N", node1 = 203, node2 = 204, Vdcn = -193.0, R = 5, L = 0.5
RLs, idx = 8, name = "RLs 30247_59 N", node1 = 204, node2 = 202, Vdcn = -193.0, R = 5, L = 0.5

C, idx = 1, name = "C 70 P", node1 = 1, node2 = 0, Vdcn = 193.0, C = 100
C, idx = 2, name = "C 59 P", node1 = 2, node2 = 0, Vdcn = 193.0, C = 100
C, idx = 3, name = "C 160002 P", node1 = 3, node2 = 0, Vdcn = 193.0, C = 100
C, idx = 4, name = "C 30247 P", node1 = 4, node2 = 0, Vdcn = 193.0, C = 100

C, idx = 5, name = "C 70 N", node1 = 201, node2 = 0, Vdcn = 193.0, C = 100
C, idx = 6, name = "C 59 N", node1 = 202, node2 = 0, Vdcn = 193.0, C = 100
C, idx = 7, name = "C 160002 N", node1 = 203, node2 = 0, Vdcn = 193.0, C = 100
C, idx = 8, name = "C 30247 N", node1 = 204, node2 = 0, Vdcn = 193.0, C = 100

VSC, idx = 1, name = "VSC 70_P", bus = 70, node1 = 1, node2 = 0,
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.025, control = "vV", Vdcn = 193, u = 1
VSC, idx = 2, name = "VSC 59_P", bus = 59, node1 = 2, node2 = 0,
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, vref0 = 1.03, control = "PV", Vdcn = 193, u = 1
VSC, idx = 3, name = "VSC 160002_P", bus = 160002, node1 = 3, node2 = 0,
     Vn = 26,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 4, name = "VSC 30247_P", bus = 30247, node1 = 4, node2 = 0,
     Vn = 20,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1

VSC, idx = 201, name = "VSC 70_N", bus = 70, node1 = 201, node2 = 0,
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.025, control = "vV", Vdcn = 193, u = 1
VSC, idx = 202, name = "VSC 59_N", bus = 59, node1 = 202, node2 = 0,
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, vref0 = 1.03, control = "PV", Vdcn = 193, u = 1
VSC, idx = 203, name = "VSC 160002_N", bus = 160002, node1 = 203, node2 = 0,
     Vn = 26,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 204, name = "VSC 30247_N", bus = 30247, node1 = 204, node2 = 0,
     Vn = 20,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1

VSC1, vsc = 1, name = "VSC 70 P", Kp1 = 0.05, Ki1 = 0.01, Kp2 = 20, Ki2 = 1, Kp3 = 1, Ki3 = 0.2
VSC1, vsc = 2, name = "VSC 59 P", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1, Kp3 = 1, Ki3 = 1
VSC1, vsc = 3, name = "VSC 160002 P", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1, Kp3 = 1, Ki3 = 1
VSC1, vsc = 4, name = "VSC 30247 P", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1, Kp3 = 1, Ki3 = 1

VSC1, vsc = 201, name = "VSC 70 N", Kp1 = 0.05, Ki1 = 0.01, Kp2 = 20, Ki2 = 1, Kp3 = 1, Ki3 = 0.2
VSC1, vsc = 202, name = "VSC 59 N", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1, Kp3 = 1, Ki3 = 1
VSC1, vsc = 203, name = "VSC 160002 N", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1, Kp3 = 1, Ki3 = 1
VSC1, vsc = 204, name = "VSC 30247 N", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 2, Ki2 = 1, Kp3 = 1, Ki3 = 1
