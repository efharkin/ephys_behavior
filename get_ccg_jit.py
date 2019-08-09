from numba import njit, prange
import numpy as np

@njit
def get_ccg(spikes1, spikes2, width=0.1, bin_width=0.001, num_jitter=50, jitter_win=0.02):

    d=[]
    djit=[]             # Distance between any two spike times
    n_sp = len(spikes2)     # Number of spikes in the input spike train
    
    jitter = (np.random.random((num_jitter+1, spikes1.size))*(2*jitter_win) - jitter_win).astype(np.float32)
    jitter[0] = np.zeros(spikes1.size)
    
    for jit in prange(num_jitter):
        
        spikes1 = spikes1+jitter[jit]
        i, j = 0, 0
        for t in spikes1:
            # For each spike we only consider those spikes times that are at most
            # at a 'width' time lag. This requires finding the indices
            # associated with the limiting spikes.
            while i < n_sp and spikes2[i] < t - width:
                i += 1
            while j < n_sp and spikes2[j] < t + width:
                j += 1
    
            # Once the relevant spikes are found, add the time differences
            # to the list
            if jit==0:
                d.extend(spikes2[i:j] - t)
            else:
                djit.extend(spikes2[i:j] - t)

    return d, djit


if __name__ == "__main__":
    import timeit
    import numpy as np
    
    
    setup = '''
from __main__ import get_ccg
import numpy as np
spikes1  = np.random.random(10000)*3600
spikes2 = np.random.random(10000)*3600'''

    print(timeit.timeit("get_ccg(spikes1, spikes2)", setup=setup))

