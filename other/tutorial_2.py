from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''

G = NeuronGroup(3, eqs, threshold='v>1', reset='v=0', method='exact')
G.I = [2, 0, 0]
G.tau = [10, 100, 100] * ms

S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
S.connect(i=[0, 0], j=[1, 2])
S.w = 'j*0.2'
S.delay = 'j*2*ms'

M = StateMonitor(G, 'v', record=True)

run(50*ms)

plt.plot(M.t/ms, M.v[0], label='Neuron 0')
plt.plot(M.t/ms, M.v[1], label='Neuron 1')
plt.plot(M.t/ms, M.v[2], label='Neuron 2')
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.legend()

plt.show()