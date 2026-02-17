from brian2 import *
import sys
sys.path.append("./")

import numpy as np
import matplotlib.pyplot as plt
from brian_bcpnn.models.tully_2014.parameters import *

print(epsilon)

V_space = np.linspace(-70, 0, 1000) * mV

def get_momentary_change(V, dt=sim_dt):
    return (g_L*(V-E_L)/-C_m)/(mV/ms)

plt.plot(V_space/mV, get_momentary_change(V_space))
plt.xlabel('V (mV)')
plt.ylabel('dV/dt (mV/ms)')
# plt.show()