# This file will first read the positions of brai neurons from a txt file and plot a 3D graph of the neurons with ParaView
import numpy as np
import matplotlib.pyplot as plt
import vtk

positions = {}
areas = {}
file_directory = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\positions\\rank_0_positions.txt"
max_neurons = 5000 # for now only read those data

# read the data from the file, data format: neuron_id, x, y, z, area
with open(file_directory, "r") as f:
    count = 0
    for line in f:
        # limit the number of neurons read
        if count >= max_neurons:
            break

        line = line.strip()
        if not line or line.startswith("#"): # skip empty lines and comments
            continue
        parts = line.split()

        # read the correponded column
        neuron_id = int(parts[0]) - 1
        x, y, z = map(float, parts[1:4])
        area = parts[4].strip()

        # store the cleaned data to two dictionaries
        positions[neuron_id] = (x, y, z)
        areas[neuron_id] = area
        count += 1

# print the number of neurons read
num_neurons = len(positions)
print(f"Number of neurons read: {num_neurons}")

# try ploting with matplotlib first
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# read the the 3D positions of the neurons
xs = [pos[0] for pos in positions.values()]
ys = [pos[1] for pos in positions.values()]
zs = [pos[2] for pos in positions.values()]
ax.scatter(xs, ys, zs, c='b', marker='o') # only the position, not the area for now
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
