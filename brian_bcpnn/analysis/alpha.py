from brian2 import *
sim_t = 0.1*ms
defaultclock.dt = sim_t

peak = 5
tau_alpha = 10 * ms
alpha_system = '''
dH/dt = -H/tau_alpha : 1
dalpha/dt = (H-alpha)/tau_alpha : 1
peak_alpha = peak * alpha : 1
dx1/dt = H/sim_t : 1
dx2/dt = alpha/sim_t : 1
dx3/dt = -x3/sim_t : 1
dx4/dt = x3/tau_alpha : 1
'''

alpha_group = NeuronGroup(N=1, model=alpha_system, method='euler')
alpha_monitor = StateMonitor(alpha_group, variables=['H', 'alpha', 'peak_alpha', 'x1', 'x2', 'x3', 'x4'], record=True)

alpha_group.H = 1
alpha_group.x3 = 1

run(100*ms)

alpha_group.H = 1 
alpha_group.x3 = 1

run(100*ms)

plt.plot(alpha_monitor.t/ms, alpha_monitor.H[0]/1, label='H')
plt.plot(alpha_monitor.t/ms, alpha_monitor.alpha[0]/1, label='alpha')
plt.plot(alpha_monitor.t/ms, alpha_monitor.peak_alpha[0]/1, label='peak_alpha')
plt.show()

plt.plot(alpha_monitor.t/ms, alpha_monitor.x1[0], label='X1')
plt.plot(alpha_monitor.t/ms, alpha_monitor.x2[0], label='X2')
plt.plot(alpha_monitor.t/ms, alpha_monitor.x3[0], label='X3')
plt.plot(alpha_monitor.t/ms, alpha_monitor.x4[0], label='X4')
plt.legend()
plt.show()