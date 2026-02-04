from brian2 import *
import matplotlib.pyplot as plt

N = 100
tau = 10 * ms
eqs = '''
dv/dt = (1-v)/tau : 1 (unless refractory)
'''

G = NeuronGroup(N, eqs, method='exact', threshold='v>0.8', reset='v=0', refractory=5*ms)
M = StateMonitor(G, 'v', record=0)
G.v = 'rand()'

spikemon = SpikeMonitor(G)

run(60 * ms)

# print('Spike indices: %s' % spikemon.i[:])

plt.plot(M.t/ms, M.v[0])
# for t in spikemon.t:
    # axvline(t/ms, ls='--', c='C1', lw=3)
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.show()

plt.plot(spikemon.t/ms, spikemon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()