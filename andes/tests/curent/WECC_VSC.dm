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
      Vdcn = 193.0, R = 0.01, L = 5

RLs, idx = "RLs_COI_N", name = "RLs Celilo-Sylmarla N", node1 = "Node_COI_3", node2 = "Node_COI_4",
      Vdcn = -193.0, R = 0.01, L = 5

C, idx = "C_COI_1", name = "C Celilo P", node1 = "Node_COI_1", node2 = "Node_COI_0", Vdcn = 193.0, C = 0.02
#C, idx = "C_COI_2", name = "C Sylmarla P", node1 = "Node_COI_2", node2 = "Node_COI_0", Vdcn = 193.0, C = -0.02
C, idx = "C_COI_3", name = "C Celilo N", node1 = "Node_COI_3", node2 = "Node_COI_0", Vdcn = -193.0, C = 0.02
#C, idx = "C_COI_4", name = "C Sylmarla N", node1 = "Node_COI_4", node2 = "Node_COI_0", Vdcn = -193.0, C = 0.02

VSC, idx = "VSC_Celilo_P", name = "VSC Celilo_P", bus = 70, node1 = "Node_COI_1", node2 = "Node_COI_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.0, control = "vV", Vdcn = 193, u = 1
VSC, idx = "VSC_Sylmarla_P", name = "VSC Sylmarla_P", bus = 59, node1 = "Node_COI_2", node2 = "Node_COI_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 13, vref0 = 1.03, control = "PV", Vdcn = 193, u = 1

VSC, idx = "VSC_Celilo_N", name = "VSC Celilo_N", bus = 70, node1 = "Node_COI_3", node2 = "Node_COI_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.00, vdcref0 = 1.0, control = "vV", Vdcn = -193, u = 1
VSC, idx = "VSC_Sylmarla_N", name = "VSC Sylmarla_N", bus = 59, node1 = "Node_COI_4", node2 = "Node_COI_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 13, vref0 = 1.03, control = "PV", Vdcn = -193, u = 1

VSC1, vsc = "VSC_Celilo_P", name = "VSC Celilo P", Kp1 = 0.049, Ki1 = 0, Kp2 = 20, Ki2 = 1,
      Kp3 = 1, Ki3 = 0.2
VSC1, vsc = "VSC_Sylmarla_P", name = "VSC Sylmarla P", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 1, Ki2 = 1,
      Kp3 = 1, Ki3 = 1
VSC1, vsc = "VSC_Celilo_N", name = "VSC Celilo N", Kp1 = 0.049, Ki1 = 0, Kp2 = 20, Ki2 = 1,
      Kp3 = 1, Ki3 = 0.2
VSC1, vsc = "VSC_Sylmarla_N", name = "VSC Sylmarla N", Kp1 = 0.2, Ki1 = 0.5, Kp2 = 1, Ki2 = 1,
       Kp3 = 1, Ki3 = 1
