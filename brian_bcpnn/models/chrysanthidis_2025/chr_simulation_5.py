# SIMULATION PURPOSE: Initialise big (10x10x30) network by sampling
# from parameters saved from smaller network (10x2x30)

from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import ChrysanthidisNetwork
from brian_bcpnn.plot import trains, synapses, composite
from brian_bcpnn.stim_protocols.train_protocol import cue_n_epochs, get_total_time
import brian_bcpnn.utils.stim_utils as stils

N_H = 10
N_M = 10
N_pyr = 30
N_BA = 4
N_poisson = 1
N_batches = 1

sample_filepath = 'data/chr/stable_init_eps_10.data'
model = ChrysanthidisNetwork(
    N_H, N_M, N_pyr, N_BA, N_poisson
)
model.sample_params(sample_filepath)