# DOME format version 1.0

# WECC California-Oregon Intertie

Ground, idx = "Ground_COI", name = "Ground COI", node = "Node_COI_0", Vdcn = 193.0, voltage = 0

Node, idx = "Node_COI_0", name = "Node COI 0", Vdcn = 193.0
Node, idx = "Node_COI_1", name = "Node Celilo", Vdcn = 193.0
Node, idx = "Node_COI_2", name = "Node Sylmarla", Vdcn = 193.0

RLs, idx = "RLs_COI_1", name = "RLs Celilo-Sylmarla", node1 = "Node_COI_1", node2 = "Node_COI_2",
     Vdcn = 193.0, R = 0.1, L = 0.01

C, idx = "C_COI_1", name = "C Celilo", node1 = "Node_COI_1", node2 = "Node_COI_0", Vdcn = 193.0, C = 2
C, idx = "C_COI_2", name = "C Sylmarla", node1 = "Node_COI_2", node2 = "Node_COI_0", Vdcn = 193.0, C = 1

VSC, idx = "VSC_Celilo", name = "VSC Celilo", bus = 70, node1 = "Node_COI_1", node2 = "Node_COI_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     vref0 = 1.05, vdcref0 = 1.0, control = "vV", Vdcn = 193, u = 1
VSC, idx = "VSC_Sylmarla", name = "VSC Sylmarla", bus = 59, node1 = "Node_COI_2", node2 = "Node_COI_0",
     Vn = 230,  rsh = 0.0025, xsh = 0.06, vshmax = 999, vshmin = 0, Ishmax = 999,
     pref0 = 15, vref0 = 1.03, control = "PV", Vdcn = 193, u = 1

VSC1, vsc = "VSC_Celilo", name = "VSC Celilo", Kp1 = 2, Ki1 = 10, Kp2 = 4, Ki2 = 2,
      Kp3 = 1, Ki3 = 0.5
VSC1, vsc = "VSC_Sylmarla", name = "VSC Sylmarla", Kp1 = 0.1, Ki1 = 2, Kp2 = 2, Ki2 = 3,
      Kp3 = 0.2, Ki3 = 0, D = 3, M = 3


# DC NETWORK DEFINITION

Node, idx = "MT_HVDC_0", name = "Node 0", Vdcn = 100.0
Node, idx = "MT_HVDC_1", name = "Node 1", Vdcn = 100.0
Node, idx = "MT_HVDC_2", name = "Node 2", Vdcn = 100.0
Node, idx = "MT_HVDC_3", name = "Node 3", Vdcn = 100.0

Ground, idx = 0, name = "Ground 0", node = "MT_HVDC_0", Vdcn = 100.0, voltage = 0

RLs, idx = "RLs1", name = "RLs 1-2", node1 = "MT_HVDC_1", node2 = "MT_HVDC_2", Vdcn = 500, R = 0.1, L = 0.01
RLs, idx = "RLs2", name = "RLs 2-3", node1 = "MT_HVDC_2", node2 = "MT_HVDC_3", Vdcn = 500, R = 0.1, L = 0.01
RLs, idx = "RLs3", name = "RLs 1-3", node1 = "MT_HVDC_1", node2 = "MT_HVDC_3", Vdcn = 500, R = 0.1, L = 0.01

C, idx = "C1", name = "C 1", node1 = "MT_HVDC_1", node2 = "MT_HVDC_0", Vdcn = 500, C = 1
C, idx = "C2", name = "C 2", node1 = "MT_HVDC_2", node2 = "MT_HVDC_0", Vdcn = 500, C = 1
C, idx = "C3", name = "C 3", node1 = "MT_HVDC_3", node2 = "MT_HVDC_0", Vdcn = 500, C = 1

# WECC CONVERTERS
VSC, idx = "VSC_WECC_2190", node1 = "MT_HVDC_1", node2 = "MT_HVDC_0", bus = 2190, Vn = 138, name = "VSC WECC 2190", rsh = 0.0025, xsh = 0.06,
     vshmax = 999, vshmin = 0, Ishmax = 999, vref0 = 1.03, vdcref0 = 1.0, control = "vV",
     Vdcn = 100, u = 1

# ERCOT CONVERTERS
VSC, idx = "VSC_ERCOT_32002", node1 = "MT_HVDC_2", node2 = "MT_HVDC_0", bus = 32002, Vn = 230, name = "VSC ERCOT 32002", rsh = 0.0025, xsh = 0.06,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = 1.5, qref0 = 0.06, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.99, Vdcn = 100, u = 1

# EI VSC CONVERTERS
VSC, idx = "VSC_EI_320002", node1 = "MT_HVDC_3", node2 = "MT_HVDC_0", bus = 320002, Vn = 230, name = "VSC EI 320002", rsh = 0.0025, xsh = 0.06,
     vshmax = 999, vshmin = 0, Ishmax = 999, pref0 = -2, qref0 = -0.1, control = "PQ",
     droop = 0, K = -0.5, vhigh = 1.01, vlow = 0.99, Vdcn = 100, u = 1


# VSC DYNAMIC PARAMETERS
VSC1, vsc = "VSC_WECC_2190", name = "VSC 1", Kp1 = 1, Ki1 = 5, Kp2 = 1, Ki2 = 5,
      Kp3 = 1, Ki3 = 0.5
VSC2B, vsc = "VSC_ERCOT_32002", name = "VSC 2", Kp1 =0.5, Ki1 = 0.2, Kp2 = 0.5, Ki2 = 0.1,
      Kp3 = 0.2, Ki3 = 0, D = 3, M = 3
VSC2B, vsc = "VSC_EI_320002", name = "VSC 3", Kp1 =0.5, Ki1 = 0.2, Kp2 = 0.5, Ki2 = 0.1,
      Kp3 = 0.2, Ki3 = 0, D = 3, M = 3
