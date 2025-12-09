# show change of calcium level over time for one single neuron
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\monitors"
neuron_id = [10,50,100,500, 1000,5000, 10000, 50000]
for nid in neuron_id:
    neuron_file = f"0_{nid-1}.csv"
    full_path = os.path.join(file_path, neuron_file)
    example_neuron = pd.read_csv(full_path, header=None, sep=';')

    # plot the calcium level over time
    calcium_series = example_neuron.iloc[:, 5].to_numpy()
    steps = example_neuron.iloc[:, 0].to_numpy()
    # plot only points after 1000 time steps
    calcium_series = calcium_series[:501]
    plt.plot(calcium_series)
    plt.xlabel("Time step")
    plt.ylabel("Calcium level")
    plt.title(f"Calcium level over time for neuron 0_{nid-1}")
    plt.grid()
    plt.show()
    """This actually means that we need to skip the first 50000 time steps
    to see the change of calcium level more clearly.
    Or we could divide calcium level into two phases:
    - initial phase (0-50000 time steps): calcium level increases rapidly
    - later phase (after 50000 time steps): calcium level stabilizes with small fluctuations
    So at least two visualizations will be helpful to understand the calcium dynamics.
    The timelines need to be adjusted accordingly.
    """