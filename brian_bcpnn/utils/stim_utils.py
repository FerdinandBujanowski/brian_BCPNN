from brian2 import *
from math import gcd
from dataclasses import dataclass

def add_time(t_total, new_time):
    return t_total+new_time, new_time

def gcd_list(l, o=None):
    match len(l):
        case 0:
            if o is None:
                return 0
            return 0
        case 1:
            if o is None:
                return l[0]
            return gcd(l[0], o)
        case _:
            if o is None:
                return gcd_list(l[2:], gcd(l[0], l[1]))
            return gcd_list(l[1:], gcd(l[0], o))

@dataclass
class ColumnCoords:
    HC: int # attributes ? instances ?
    MC: int
    def __str__(self):
        return f'[{self.HC};{self.MC}]' # constructor? method?
@dataclass
class StimTime:
    t_start: Quantity # in ms
    t_end: Quantity # in ms
    def __eq__(self, value):
        if isinstance(value, StimTime):
            return self.t_start == value.t_start and self.t_end == value.t_end
        return False
    def __str__(self):
        return f'{self.t_start}-{self.t_end}'
@dataclass
class StimProtocol:
    coords: ColumnCoords
    stim_time: StimTime
    def __str__(self):
        return f'({str(self.coords)}, {str(self.stim_time)})'
        # returns either (0,0) or (1,0) for my model 
@dataclass
class Pattern:
    coord_list: list[ColumnCoords]
    def __str__(self):
        return '['+', '.join([str(coord) for coord in self.coord_list])+']'
    def contains(self, coord:ColumnCoords):
        for c in self.coord_list:
            if c.HC == coord.HC and c.MC == coord.MC:
                return True
        return False
@dataclass
class PatternProtocol:
    pattern: Pattern
    stim_time: StimTime
@dataclass
class PatternList:
    patterns: list[Pattern]
    def __str__(self):
        return '['+', '.join([str(pattern) for pattern in self.patterns])+']'
    def subset(self,i,j=None):
        if j is not None:
            return PatternList(self.patterns[i:j])
        return PatternList([self.patterns[i]])

def get_orthogonal_patterns(N_H, N_M) -> PatternList:
    return PatternList([Pattern([ColumnCoords(h, m) for h in range(N_H)]) for m in range(N_M)])

def get_incomplete_patterns(original_patterns: PatternList, n_MC) -> PatternList:
    new_list = []

    for pattern in original_patterns.patterns:
        chosen_subset = np.random.choice(pattern.coord_list, min(len(pattern.coord_list), n_MC), replace=False)
        new_list.append(Pattern(chosen_subset))

    return PatternList(new_list)

def train_patterns_protocol(
        pattern_list: PatternList, 
        t_init:Quantity, t_stim:Quantity, t_isi:Quantity, t_end:Quantity,
        n_batches:int=1
    ) -> tuple[list[StimProtocol], Quantity]:
    stims = []
    current_time = t_init
    for batch in range(n_batches):
        for i_pattern, pattern in enumerate(pattern_list.patterns):
            for coords in pattern.coord_list:
                stims.append(StimProtocol(coords, StimTime(current_time, current_time+t_stim)))
            current_time += t_stim
            if i_pattern < (len(pattern_list.patterns) - 1):
                current_time += t_isi

        if batch < (n_batches-1):
            current_time += t_isi

    current_time += t_end

    return stims, current_time

def pattern_protocol_to_stim_protocol(pattern_protocol:PatternProtocol) -> list[StimProtocol]:
    return [StimProtocol(coords, pattern_protocol.stim_time) for coords in pattern_protocol.pattern.coord_list]

def stim_times_to_timed_array(stims: list[StimProtocol], t_total:Quantity, N_H:int, N_M:int):
    times = {int(t_total/ms)}
    hc_list = []
    mc_list = []
    for stim in stims:
        stim_time = stim.stim_time
        times.add(int(stim_time.t_start/ms))
        times.add(int(stim_time.t_end/ms))
        hc_list.append(stim.coords.HC)
        mc_list.append(stim.coords.MC)

    stim_dt = gcd_list(list(times))*ms
    # print(stim_dt)
    n_time_steps = int(t_total/stim_dt)

    stim_array = np.zeros(shape=(N_H*N_M,n_time_steps),dtype=int32)
    for stim in stims:
        fr = int(round(stim.stim_time.t_start/stim_dt))
        to = int(round(stim.stim_time.t_end/stim_dt))
        stim_array[stim.coords.HC*N_M+stim.coords.MC, fr:to] = 1

    # print(stim_array.T)
    return TimedArray(stim_array.T, dt=stim_dt)

def get_pattern_time_dict(pl:PatternList, stims:list[StimProtocol]) -> dict[str,list[StimTime]]:
    pt_dict = dict()
    current_pattern = 1
    for pattern in pl.patterns:
        new_key = f'Pattern {current_pattern}'
        current_pattern += 1
        pt_dict[new_key] = []

        for stim in stims:
            st = stim.stim_time
            if pattern.contains(stim.coords) and st not in pt_dict[new_key]:
                pt_dict[new_key].append(st)

    return pt_dict

# test_protocol = [
#     StimProtocol(ColumnCoords(0, 0), 250*ms, 350*ms),
#     StimProtocol(ColumnCoords(1, 1), 300*ms, 400*ms)
#     ]
# ta = stim_times_to_timed_array(test_protocol, 5*second, 4, 2)