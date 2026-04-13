from brian2 import *
import sys
import numpy as np
import matplotlib.pyplot as plt

from brian_bcpnn.models.tully_2014.tully_params import *

test_array = np.zeros(shape=(3, 4))

test_array[0, 1:3] = 1
print(test_array)