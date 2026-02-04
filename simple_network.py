from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

N = 10

V_spike = -40 * mV
V_rest = -70 * mV
V_reset = V_rest
E_L = -75 * mV
g_L = 10 * nS
tau_m = 10 * ms
# I = 10 * nA

eqs = '''
dV/dt = (-(V - E_L) + I/g_L) / tau_m : volt (unless refractory)
I : amp
'''
G = NeuronGroup(N, eqs, threshold='V>V_spike', reset='V=V_reset', method='exact', refractory=5*ms)
G.V = 'V_rest'
G.I = 'rand()*10*nA'

S = Synapses(G, G, on_pre='V_post += 10 * mV')
S.connect(condition='i!=j', p=0.2)

M = StateMonitor(G, 'V', record=True)

run(100*ms)

plt.imshow(M.V/mV, 'viridis', aspect='auto', interpolation='none')
plt.colorbar()
plt.show()