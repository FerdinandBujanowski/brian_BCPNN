from brian2 import *
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./')
from brian_bcpnn.models.tully_2014.tully_params import *

test_array = np.zeros(shape=(3, 4))

test_array[0, 1:3] = 1
print(test_array)

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.set_xlabel("Test X")
    ax.set_ylabel("Test Y")
    ax.plot([0, 1, 2], [1, 2, 3])
    ax.set_title(f'test {i}')
plt.show()