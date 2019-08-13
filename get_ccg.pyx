def get_ccg(spikes1, spikes2, width, bin_width):

    d = []                   # Distance between any two spike times
    n_sp = len(spikes2)  # Number of spikes in the input spike train

    print('running')
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
        d.extend(spikes2[i:j] - t)

    return(d)