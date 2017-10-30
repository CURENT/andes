# DOME format version 1.0

# 9 - Node HVDC Overlay

Ground, idx = 0, name = "Ground COI", node = 0, Vdcn = 193.0, voltage = 0
Node, idx = 0, name = "Node 0", Vdcn = 193.0, xcoord = 45.596416, ycoord = -121.312202

Node, idx = 1, name = "Node 70 P", Vdcn = 193.0, xcoord = 45.59, ycoord = -121.31, area = 90, region = 90
Node, idx = 2, name = "Node 59 P", Vdcn = 193.0, xcoord = 34.31, ycoord = -118.48, area = 90, region = 90
Node, idx = 3, name = "Node 160002 P", Vdcn = 193.0, xcoord = 45.37, ycoord = -93.9, area = 600, region = 600
Node, idx = 4, name = "Node 137523 P", Vdcn = 193.0, xcoord = 41.25, ycoord = -88.23, area = 363, region = 937
Node, idx = 5, name = "Node 30035 P", Vdcn = 193.0, xcoord = 32.29, ycoord = -97.78, area = 151, region = 157
Node, idx = 6, name = "Node 15 P", Vdcn = 193.0, xcoord = 33.18, ycoord = -112.65, area = 10, region = 10
Node, idx = 7, name = "Node 72 P", Vdcn = 193.0, xcoord = 45.89, ycoord = -106.61, area = 60, region = 62
Node, idx = 8, name = "Node 157962 P", Vdcn = 193.0, xcoord = 39.45, ycoord = -94.98, area = 541, region = 541
Node, idx = 9, name = "Node 115250 P", Vdcn = 193.0, xcoord = 33.15, ycoord = -81.76, area = 146, region = 140

Node, idx = 201, name = "Node 70 N", Vdcn = -193.0, xcoord = 45.59, ycoord = -121.31, area = 90, region = 90
Node, idx = 202, name = "Node 59 N", Vdcn = -193.0, xcoord = 34.31, ycoord = -118.48, area = 90, region = 90
Node, idx = 203, name = "Node 160002 N", Vdcn = -193.0, xcoord = 45.37, ycoord = -93.9, area = 600, region = 600
Node, idx = 204, name = "Node 137523 N", Vdcn = -193.0, xcoord = 41.25, ycoord = -88.23, area = 363, region = 937
Node, idx = 205, name = "Node 30035 N", Vdcn = -193.0, xcoord = 32.29, ycoord = -97.78, area = 151, region = 157
Node, idx = 206, name = "Node 15 N", Vdcn = -193.0, xcoord = 33.18, ycoord = -112.65, area = 10, region = 10
Node, idx = 207, name = "Node 72 N", Vdcn = -193.0, xcoord = 45.89, ycoord = -106.61, area = 60, region = 62
Node, idx = 208, name = "Node 157962 N", Vdcn = -193.0, xcoord = 39.45, ycoord = -94.98, area = 541, region = 541
Node, idx = 209, name = "Node 115250 N", Vdcn = -193.0, xcoord = 33.15, ycoord = -81.76, area = 146, region = 140

RLs, idx = 1, name = "RLs 70_59 P", node1 = 1, node2 = 2, Vdcn = 193.0, R = 2.5, L = 0.625
RLs, idx = 2, name = "RLs 72_70 P", node1 = 7, node2 = 1, Vdcn = 193.0, R = 2, L = 0.5
RLs, idx = 3, name = "RLs 160002_72 P", node1 = 3, node2 = 7, Vdcn = 193.0, R = 2, L = 0.5
RLs, idx = 4, name = "RLs 160002_137523 P", node1 = 3, node2 = 4, Vdcn = 193.0, R = 1, L = 0.25
RLs, idx = 5, name = "RLs 137523_30035 P", node1 = 4, node2 = 5, Vdcn = 193.0, R = 2.5, L = 0.625
RLs, idx = 6, name = "RLs 30035_15 P", node1 = 5, node2 = 6, Vdcn = 193.0, R = 3, L = 0.75
RLs, idx = 7, name = "RLs 15_59 P", node1 = 6, node2 = 2, Vdcn = 193.0, R = 1, L = 0.25
RLs, idx = 8, name = "RLs 72_59 P", node1 = 7, node2 = 2, Vdcn = 193.0, R = 3, L = 0.75
RLs, idx = 9, name = "RLs 157962_137523 P", node1 = 8, node2 = 4, Vdcn = 193.0, R = 1, L = 0.25
RLs, idx = 10, name = "RLs 30035_115250 P", node1 = 5, node2 = 9, Vdcn = 193.0, R = 3, L = 0.75

RLs, idx = 201, name = "RLs 70_59 N", node1 = 201, node2 = 202, Vdcn = -193.0, R = 2.5, L = 0.625
RLs, idx = 202, name = "RLs 72_70 N", node1 = 207, node2 = 201, Vdcn = -193.0, R = 2, L = 0.5
RLs, idx = 203, name = "RLs 160002_72 N", node1 = 203, node2 = 207, Vdcn = -193.0, R = 2, L = 0.5
RLs, idx = 204, name = "RLs 160002_137523 N", node1 = 203, node2 = 204, Vdcn = -193.0, R = 1, L = 0.25
RLs, idx = 205, name = "RLs 137523_30035 N", node1 = 204, node2 = 205, Vdcn = -193.0, R = 2.5, L = 0.625
RLs, idx = 206, name = "RLs 30035_15 N", node1 = 205, node2 = 206, Vdcn = -193.0, R = 3, L = 0.75
RLs, idx = 207, name = "RLs 15_59 N", node1 = 206, node2 = 202, Vdcn = -193.0, R = 1, L = 0.25
RLs, idx = 208, name = "RLs 72_59 N", node1 = 207, node2 = 202, Vdcn = -193.0, R = 3, L = 0.75
RLs, idx = 209, name = "RLs 157962_137523 N", node1 = 208, node2 = 204, Vdcn = -193.0, R = 1, L = 0.25
RLs, idx = 210, name = "RLs 30035_115250 N", node1 = 205, node2 = 209, Vdcn = -193.0, R = 3, L = 0.75

C, idx = 1, name = "C 70 P", node1 = 1, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 2, name = "C 59 P", node1 = 2, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 3, name = "C 160002 P", node1 = 3, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 4, name = "C 137523 P", node1 = 4, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 5, name = "C 30035 P", node1 = 5, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 6, name = "C 15 P", node1 = 6, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 7, name = "C 72 P", node1 = 7, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 8, name = "C 157962 P", node1 = 8, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 9, name = "C 115250 P", node1 = 9, node2 = 0, Vdcn = 193.0, C = 0.02

C, idx = 201, name = "C 70 N", node1 = 201, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 202, name = "C 59 N", node1 = 202, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 203, name = "C 160002 N", node1 = 203, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 204, name = "C 137523 N", node1 = 204, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 205, name = "C 30035 N", node1 = 205, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 206, name = "C 15 N", node1 = 206, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 207, name = "C 72 N", node1 = 207, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 208, name = "C 157962 N", node1 = 208, node2 = 0, Vdcn = 193.0, C = 0.02
C, idx = 209, name = "C 115250 N", node1 = 209, node2 = 0, Vdcn = 193.0, C = 0.02

VSC, idx = 1, name = "VSC 70_P", bus = 70, node1 = 1, node2 = 0, Vn = 230,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.025, control = "vV", Vdcn = 193, u = 1
VSC, idx = 2, name = "VSC 59_P", bus = 59, node1 = 2, node2 = 0, Vn = 230,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = -9, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 3, name = "VSC 160002_P", bus = 160002, node1 = 3, node2 = 0, Vn = 26,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 4, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 4, name = "VSC 137523_P", bus = 137523, node1 = 4, node2 = 0, Vn = 25,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = -4, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 5, name = "VSC 30035_P", bus = 30035, node1 = 5, node2 = 0, Vn = 22,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2.5, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 6, name = "VSC 15_P", bus = 15, node1 = 6, node2 = 0, Vn = 500,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 1.5, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 7, name = "VSC 72_P", bus = 72, node1 = 7, node2 = 0, Vn = 500,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 8, name = "VSC 157962_P", bus = 157962, node1 = 8, node2 = 0, Vn = 25,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 3, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 9, name = "VSC 115250_P", bus = 115250, node1 = 9, node2 = 0, Vn = 25,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = -3, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1

VSC, idx = 201, name = "VSC 70_N", bus = 70, node1 = 201, node2 = 0, Vn = 230,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.025, control = "vV", Vdcn = 193, u = 1
VSC, idx = 202, name = "VSC 59_N", bus = 59, node1 = 202, node2 = 0, Vn = 230,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = -9, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 203, name = "VSC 160002_N", bus = 160002, node1 = 203, node2 = 0, Vn = 26,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 4, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 204, name = "VSC 137523_N", bus = 137523, node1 = 204, node2 = 0, Vn = 25,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = -4, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 205, name = "VSC 30035_N", bus = 30035, node1 = 205, node2 = 0, Vn = 22,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2.5, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 206, name = "VSC 15_N", bus = 15, node1 = 206, node2 = 0, Vn = 500,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 =1.5, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 207, name = "VSC 72_N", bus = 72, node1 = 207, node2 = 0,  Vn = 500,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 2, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 208, name = "VSC 157962_N", bus = 157962, node1 = 208, node2 = 0, Vn = 25,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 3, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1
VSC, idx = 209, name = "VSC 115250_N", bus = 115250, node1 = 209, node2 = 0, Vn = 25,  rsh = 0.001, xsh = 0.04, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = -3, qref0 = 0.01, control = "PQ", Vdcn = 193, u = 1

VSC1, vsc = 1, name = "VSC 70 P", Kp1 = 0.02, Ki1 = 0.5, Kp2 = 20, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 2, name = "VSC 59 P", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 3, name = "VSC 160002 P", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 4, name = "VSC 137523 P", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 5, name = "VSC 30035 P", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 6, name = "VSC 15 P", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 7, name = "VSC 72 P", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 8, name = "VSC 157962 P", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 9, name = "VSC 115250 P", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2

VSC1, vsc = 201, name = "VSC 70 N", Kp1 = 0.02, Ki1 = 0.5, Kp2 = 20, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 202, name = "VSC 59 N", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 203, name = "VSC 160002 N", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 204, name = "VSC 137523 N", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 205, name = "VSC 30035 N", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 206, name = "VSC 15 N", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 207, name = "VSC 72 N", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 208, name = "VSC 157962 N", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
VSC1, vsc = 209, name = "VSC 115250 N", Kp1 = 1, Ki1 = 2, Kp2 = 1, Ki2 = 2, Kp3 = 1, Ki3 = 2
