from brian2 import *
sys.path.append("./")
from brian_bcpnn.networks import CorticalNetwork
from brian_bcpnn.plot import trains
import brian_bcpnn.utils.stim_utils as stils

def train_n_epochs(
        model:CorticalNetwork,
        t_init, t_stim, t_isi, t_end,
        patterns:stils.PatternList=None,
        n_batches=1,
        stim_ta_string='stim_ta'
):
    defaultclock.dt = model.namespace['t_sim']

    if patterns is None:
        patterns = stils.get_orthogonal_patterns(model.N_hyper, model.N_mini)
    
    stims, t_total = stils.train_patterns_protocol(
        patterns,
        t_init, t_stim, t_isi, t_end,
        n_batches
    )

    model.namespace[stim_ta_string] = stils.stim_times_to_timed_array(
        stims, t_total,
        model.N_hyper, model.N_mini
    )

    model.run(t_total)

    return t_total