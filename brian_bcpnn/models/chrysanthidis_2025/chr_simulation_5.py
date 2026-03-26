# SIMULATION PURPOSE: Test the weight init mechanism that allows for 
# minimizing the amplitude and time it takes for the network to converge to 
# stable dynamics of around 3 Hz without any additional input
# --> Plot Spiking Raster and Histogram of Frequencies
# --> Save all relevant network parameters (including connectivity) to file

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
from brian_bcpnn.stim_protocols.init_params import init_network_params


N_H = 5
N_M = 2
N_PYR = 30
model = ChrysanthidisNetwork(N_H, N_M, N_PYR, 4, namespace=chr_namespace)
p_c = 100/(N_H*N_PYR)
model.namespace['cp_PP'] = p_c
model.namespace['cp_PPL'] = p_c
init_network_params(model, 'data/chr/stable_init_eps_5.data')