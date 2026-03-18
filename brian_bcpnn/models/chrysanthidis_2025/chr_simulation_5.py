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


model = ChrysanthidisNetwork(4, 2, 30, 4, namespace=chr_namespace)

init_network_params(model, 'data/chr/stable_init_test.data')