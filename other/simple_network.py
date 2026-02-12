from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

defaultclock.dt = 0.1 * ms

tfinal = 100 * ms
N = 10

V_spike = -40 * mV
V_rest = -65 * mV

tau_m = 10 * ms
R_m = 4 * Mohm
tau_I = 1 * ms
d_I = 0.1 * uA

eqs = '''
dV/dt = (-(V-V_rest) + I*R_m) / tau_m : volt (unless refractory)
dI/dt = -I/tau_I : amp
'''
G = NeuronGroup(N, eqs, threshold='V>=V_spike', reset='V=V_rest', method='euler', refractory=5*ms)
G.V = V_rest
G.I = 'rand() * 5 * uA'

spikemon = SpikeMonitor(G)
statemon = StateMonitor(G, ['V', 'I'], record=True)

S = Synapses(G, G, on_pre='I += d_I')
S.connect(condition='i!=j', p=0.2)

run(tfinal)

fig, (ax, ax_voltage) = plt.subplots(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (3, 1)})
ax.scatter(spikemon.t / ms, spikemon.i[:], marker="_", color="k", s=10)
ax.set_xlim(0, tfinal / ms)
ax.set_ylim(0, N)
ax.set_ylabel("neuron number")
ax.set_yticks(np.arange(0, N, 100))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax_voltage.plot(statemon.t / ms, statemon.I[0] / uA, color='k')
ax_voltage.set_xticks(np.arange(0, tfinal / ms, 100))
ax_voltage.spines['right'].set_visible(False)
ax_voltage.spines['top'].set_visible(False)
ax_voltage.set_xlabel("time, ms")

plt.show()