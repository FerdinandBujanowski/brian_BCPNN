# SIMULATION PURPOSE: Test the weight init mechanism that allows for 
# minimizing the amplitude and time it takes for the network to converge to 
# stable dynamics of around 3 Hz without any additional input
# --> Plot Spiking Raster and Histogram of Frequencies
# --> Save all relevant network parameters (including connectivity) to file

from brian2 import *
# import brian2cuda
# set_device("cuda_standalone")
# brian2cuda.example_run()

sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.models.chrysanthidis_2025.chr_params import chr_namespace
from brian_bcpnn.stim_protocols.init_params import init_network_params

N_H = 1
N_M = 2
N_BA = 4
N_PYR = 30
model = ChrysanthidisNetwork(N_H, N_M, N_PYR, N_BA, namespace=chr_namespace)
init_network_params(model, f'./brian_bcpnn/data/chr/stable_init_{N_H}_{N_M}_{N_PYR}.data')