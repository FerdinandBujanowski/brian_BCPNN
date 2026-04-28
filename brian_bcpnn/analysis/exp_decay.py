from brian2 import *
import matplotlib.pyplot as plt

sim_t = 0.1*ms
defaultclock.dt = sim_t

tau_syn = 5 * ms

eqs = '''
dtime/dt = 0.1 * ms / sim_t : second
last_spike: second

syn = exp((-time-last_spike)/tau_syn) : 1
'''

model = NeuronGroup(N=1, model=eqs, method='euler')
mon = StateMonitor(model, variables=['time', 'last_spike', 'syn'], record=0)
model.time = 0*ms
model.last_spike = 0*ms

run(50*ms)

model.last_spike = model.time

run(50*ms)

plt.plot(mon.t/ms, mon.time[0]/ms, label='time')
plt.plot(mon.t/ms, mon.last_spike[0]/ms, label='last_spike')
plt.plot(mon.t/ms, mon.syn[0], label='syn')
plt.legend()
plt.show()