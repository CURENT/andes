from cvxopt import matrix, mul
from math import floor


p_ratio = [1, 1.3, 1.1, 0.95, 0.9, 1.4, 0.9, 0.8]
q_ratio = [1, 1.23, 1.1, 0.90, 0.85, 1.3, 0.9, 0.9]

# start_times = range(5, 170, 20)

interval = 20
start_times = range(5, 150, interval)

vref0 = [
[0.983210008, 1.008694725, 0.9911988, 0.977239909, 1.012624861, 1.005552435, 1.0393584, 0.99690634, 0.991380182, 1.010114262],
[0.983124687, 1.005231972, 0.990334177, 0.997894737, 1.003300397, 1.004771992, 1.03591672, 1.000821134, 0.994812161, 1.008494889],
[0.984500498, 1.004906166, 0.989791617, 1.005449598, 0.996471927, 1.005160103, 1.036213027, 1.004011079, 0.997335518, 1.008586438],
[0.983415303, 1.00138095, 0.991315781, 1.001258416, 1.001870276, 1.007257601, 1.035815102, 1.003046713, 0.997559333, 1.011262169],
[0.984701298, 1.003812931, 0.990918588, 0.994055759, 1.008400306, 1.003981995, 1.03745242, 0.999907461, 0.993368373, 1.007760932],
[1.02590655, 0.973774501, 1.027708065, 1.021972941, 0.975056634, 1.008257224, 1.025697231, 1.011955841, 1.018797621, 1.007160876],
[1.008459139, 0.963609686, 1.031473287, 1.016044914, 0.982291383, 1.020924857, 1.041252122, 0.996465578, 1.001519324, 1.00551817],
[0.989003184, 0.999975987, 0.994305232, 1.008134402, 0.998067111, 1.009206543, 1.034998465, 1.006715372, 0.996155777, 1.007745074],
]


load_idx = matrix([20, 25, 29]) - 1
gen_idx = matrix([34, 37, 38]) - 29 - 1


def pert(t, system):
    if t < start_times[0]:
        return
    elif t > start_times[-1] + interval:
        return

    k = int(floor( (t - start_times[0])  / (start_times[1] - start_times[0])))
    start = start_times[k]
    xp = p_ratio[k]
    xq = q_ratio[k]
    vref = vref0[k]

    set_load(t, start, system, load_idx, xp, xq)
    # set_avr(t, system, start, interval=0.1, values=vref)

    system.DAE.rebuild = True


def set_avr(t, system,  start, interval, values):
    n = len(values)
    if t < start:
        return
    elif t > start + n*interval:
        return

    i = floor((t - start) / interval)
    system.AVR3.vref0[i] = values[i]


def set_load(t, time, system, idx, p_ratio, q_ratio):
    if time <= t <= time + system.TDS.tstep:
        system.PQ.p[idx] = mul(system.PQ.p[idx], matrix(p_ratio))
        system.PQ.q[idx] = mul(system.PQ.q[idx], matrix(q_ratio))

