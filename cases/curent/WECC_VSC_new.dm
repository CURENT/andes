# DOME format version 1.0

# WECC SYSTEM DATA
# WECC HVDC station buses: 2191, 2189, 2090, 2192
INCLUDE, WECC_PF_BASE.dm
INCLUDE, WECC_HVDC.dm
INCLUDE, WECC_DYN_BASE.dm
INCLUDE, WECC_MON.dm


# WECC California-Oregon Intertie

Ground, idx = "Ground_COI", name = "Ground COI", node = "Node_COI_0", Vdcn = 193.0, voltage = 0

Node, idx = "Node_COI_0", name = "Node COI 0", Vdcn = 193.0
Node, idx = "Node_COI_1", name = "Node Celilo P", Vdcn = 193.0
Node, idx = "Node_COI_2", name = "Node Sylmarla P", Vdcn = 193.0
Node, idx = "Node_COI_3", name = "Node Celilo N", Vdcn = -193.0
Node, idx = "Node_COI_4", name = "Node Sylmarla N", Vdcn = -193.0

RLs, idx = "RLs_COI_P", name = "RLs Celilo-Sylmarla P", node1 = "Node_COI_1", node2 = "Node_COI_2",
      Vdcn = 193.0, R = 0.01, L = -0.5

#RLs, idx = "RLs_COI_N", name = "RLs Celilo-Sylmarla N", node1 = "Node_COI_3", node2 = "Node_COI_4",
#      Vdcn = -193.0, R = 0.01, L = 20

#R, idx = "R_COI_1", name = "R Celilo-Sylmarla P", node1 = "Node_COI_1", node2 = "Node_COI_2",
#     Vdcn = 193.0, R = 0.01

#R, idx = "R_COI_1", name = "R Celilo-Sylmarla N", node1 = "Node_COI_3", node2 = "Node_COI_4",
#     Vdcn = -193.0, R = 0.01


C, idx = "C_COI_1", name = "C Celilo P", node1 = "Node_COI_1", node2 = "Node_COI_0", Vdcn = 193.0, C = 15
C, idx = "C_COI_2", name = "C Sylmarla P", node1 = "Node_COI_2", node2 = "Node_COI_0", Vdcn = 193.0, C = -15
# C, idx = "C_COI_3", name = "C Celilo N", node1 = "Node_COI_3", node2 = "Node_COI_0", Vdcn = -193.0, C = 1
# C, idx = "C_COI_4", name = "C Sylmarla N", node1 = "Node_COI_4", node2 = "Node_COI_0", Vdcn = -193.0, C = 1

VSC, idx = "VSC_Celilo_P", name = "VSC Celilo_P", bus = 70, node1 = "Node_COI_1", node2 = "Node_COI_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.0, control = "vV", Vdcn = 193, u = 1
VSC, idx = "VSC_Sylmarla_P", name = "VSC Sylmarla_P", bus = 59, node1 = "Node_COI_2", node2 = "Node_COI_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 15, vref0 = 1.03, control = "PV", Vdcn = 193, u = 1

#VSC, idx = "VSC_Celilo_N", name = "VSC Celilo_N", bus = 70, node1 = "Node_COI_3", node2 = "Node_COI_0",
#     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
#     vref0 = 1.00, vdcref0 = 1.0, control = "vV", Vdcn = -193, u = 1
#VSC, idx = "VSC_Sylmarla_N", name = "VSC Sylmarla_N", bus = 59, node1 = "Node_COI_4", node2 = "Node_COI_0",
#     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
#     pref0 = 3, vref0 = 1.03, control = "PV", Vdcn = -193, u = 1

#VSC1, vsc = "VSC_Celilo_P", name = "VSC Celilo P", Kp1 = 0.025, Ki1 = 0.0002, Kp2 =150000, Ki2 = 5,
#      Kp3 = 700000, Ki3 = 5

#VSC1, vsc = "VSC_Sylmarla_P", name = "VSC Sylmarla P", Kp1 = 1, Ki1 = 5, Kp2 = 1, Ki2 = 5,
#      Kp3 = 1, Ki3 = 5

# VSC1, vsc = "VSC_Celilo_N", name = "VSC Celilo N", Kp1 = 0, Ki1 = 0, Kp2 = 0, Ki2 = 0,
#       Kp3 = 0, Ki3 = 0
# VSC1, vsc = "VSC_Sylmarla_N", name = "VSC Sylmarla N", Kp1 = 1, Ki1 = 1, Kp2 = 1, Ki2 = 1,
#       Kp3 = 1, Ki3 = 1

Fault, Vn = 500, bus = 6, tf = 0.1, tc = 0.15, xf = 0.01
Fault, Vn = 500, bus = 6, tf = 100.1, tc = 100.15, xf = 0.01
Fault, Vn = 500, bus = 6, tf = 200.1, tc = 200.15, xf = 0.01
Fault, Vn = 500, bus = 6, tf = 300.1, tc = 300.15, xf = 0.01
Fault, Vn = 500, bus = 6, tf = 400.1, tc = 400.15, xf = 0.01
Fault, Vn = 500, bus = 6, tf = 500.1, tc = 500.15, xf = 0.01
