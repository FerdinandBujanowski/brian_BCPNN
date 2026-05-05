from brian2 import *
from math import gcd
from dataclasses import dataclass

def add_time(t_total, new_time):
    return t_total+new_time, new_time

def gcd_list(l, o=None):
    list_len = len(l)
    if list_len == 0:
        if o is None:
            return 0
        return 0
    elif list_len == 1:
        if o is None:
            return l[0]
        return gcd(l[0], o)
    if o is None:
        return gcd_list(l[2:], gcd(l[0], l[1]))
    return gcd_list(l[1:], gcd(l[0], o)) 
            

@dataclass
class ColumnCoords:
    HC: int
    MC: int
    def __str__(self):
        return f'[{self.HC};{self.MC}]'
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
    
def string_to_tuple(tuple_string):
    return (int(tuple_string[0]), int(tuple_string[1]))

def pattern_string_to_tuple_list(pattern_string):
    return sorted([string_to_tuple(s) for s in [c.replace('[', '').replace(';','').replace(']','').replace(',','') for c in pattern_string.split(' ')]])

def patterns_from_txt(filepath) -> PatternList:
    patterns = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tuple_list = pattern_string_to_tuple_list(line)
            patterns.append(Pattern([ColumnCoords(h, m) for (h,m) in tuple_list]))
    return PatternList(patterns)

def get_orthogonal_patterns(N_H, N_M) -> PatternList:
    return PatternList([Pattern([ColumnCoords(h, m) for h in range(N_H)]) for m in range(N_M)])

def get_incomplete_patterns(original_patterns: PatternList, n_MC) -> PatternList:
    new_list = []

    for pattern in original_patterns.patterns:
        chosen_subset = np.random.choice(pattern.coord_list, min(len(pattern.coord_list), n_MC), replace=False)
        new_list.append(Pattern(chosen_subset))

    return PatternList(new_list)

def distort_patterns(pattern_list:PatternList, N_M, n_dist=1) -> PatternList:
    new_patterns = []
    # n_dist: number of hypercolumns to be resampled
    for pattern in pattern_list.patterns:
        columns = [ColumnCoords(c.HC, c.MC) for c in pattern.coord_list]
        np.random.shuffle(columns)
        for i_dist in range(n_dist):
            new_mc = columns[i_dist].MC
            while new_mc == columns[i_dist].MC:
                new_mc = np.random.randint(N_M)
            columns[i_dist].MC = new_mc
        new_patterns.append(Pattern(columns))
    return PatternList(new_patterns)


def get_pattern_overlap_counts(pattern_list:PatternList) -> list[int]:
    # for each pattern, return the number of total minicolumn overlaps with all other patterns
    overlap_counts = []
    for i, i_pattern in enumerate(pattern_list.patterns):
        total_overlaps = 0
        for j, j_pattern in enumerate(pattern_list.patterns):
            if i != j:
                for i_coords in i_pattern.coord_list:
                    for j_coords in j_pattern.coord_list:
                        if i_coords == j_coords:
                            total_overlaps += 1
        overlap_counts.append(total_overlaps)
    return overlap_counts


def get_random_patterns(N_H, N_M, N_P) -> PatternList:
    patterns = []
    for _ in range(N_P):
        coord_list = []
        for h in range(N_H):
            coord_list.append(ColumnCoords(h, np.random.randint(N_M)))
        patterns.append(Pattern(coord_list))
    return PatternList(patterns)

def train_patterns_protocol(
        pattern_list: PatternList, 
        t_init:Quantity, t_stim:Quantity, t_isi:Quantity, t_end:Quantity,
        n_batches:int=1, shuffle_patterns=False
    ) -> tuple[list[StimProtocol], Quantity]:
    stims = []
    current_time = t_init
    for batch in range(n_batches):
        pattern_copy = np.array(pattern_list.patterns)
        if shuffle_patterns:
            np.random.shuffle(pattern_copy)
        for i_pattern, pattern in enumerate(pattern_copy):
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
    time_coord_dict = dict()

    for stim in stims:
        stim_key = str(stim.stim_time)
        if stim_key not in time_coord_dict.keys():
            time_coord_dict[stim_key] = [stim.coords]
        else:
            time_coord_dict[stim_key].append(stim.coords)
    
    for stim_time, coords in time_coord_dict.items():
        current_stim = None
        for stim in stims:
            if stim_time == str(stim.stim_time):
                current_stim = stim.stim_time
        for i_p, pattern in enumerate(pl.patterns):
            match_all_coords = True
            for coord in pattern.coord_list:
                if coord not in coords:
                    match_all_coords = False
            if match_all_coords and len(pattern.coord_list) == len(coords):
                new_key = f'Pattern {i_p + 1}'
                if new_key not in pt_dict.keys():
                    pt_dict[new_key] = [current_stim]
                else:
                    pt_dict[new_key].append(current_stim)

    return pt_dict

# test_protocol = [
#     StimProtocol(ColumnCoords(0, 0), 250*ms, 350*ms),
#     StimProtocol(ColumnCoords(1, 1), 300*ms, 400*ms)
#     ]
# ta = stim_times_to_timed_array(test_protocol, 5*second, 4, 2)