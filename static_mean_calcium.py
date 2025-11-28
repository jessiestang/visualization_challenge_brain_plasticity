# This file will compute the static mean calcium activity for each neuron from the calcium data file
import numpy as np
import csv
import os

mean_calcium = {}
monitor_file_directory = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\monitors"
sub_set_size = 5000

# read the calcium data from the csv file
for i in range(sub_set_size):
    fname = os.path.join(monitor_file_directory, f"0_{i}.csv")
    print(f"Processing file: {fname}")

    # if data for a certain neuron is mssing, skip it
    if not os.path.exists(fname):
        print(f"[WARN] Missing monitor file for neuron {i}")
        mean_calcium[i] = 0.0
        continue

    ca_level = []
    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ca_value = float(row[5]) # current calcium is stored at the 6th column (index 5)
                ca_level.append(ca_value)
            except ValueError:
                continue
    # data is missing for neuron i
    if len(ca_level) == 0:
        mean_calcium[i] = 0.0
    else:
        mean_calcium[i] = np.mean(ca_level) # calculate the mean calcium activity (static)
