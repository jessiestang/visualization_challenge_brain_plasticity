import numpy as np
import matplotlib.pyplot as plt
import vtk
import os
import csv

positions = {}
areas = {}
file_directory = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\positions\\rank_0_positions.txt"
max_neurons = 25000 # for now only read those data
print("Reading neuron positions...")

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

# load the calcium data
mean_calcium = {}
mean_current = {}
mean_input = {}

monitor_file_directory = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\monitors"
sub_set_size = max_neurons
# read the calcium data from the csv file
for i in range(sub_set_size):
    fname = os.path.join(monitor_file_directory, f"0_{i}.csv")

    # if data for a certain neuron is mssing, skip it
    if not os.path.exists(fname):
        print(f"[WARN] Missing monitor file for neuron {i}")
        mean_calcium[i] = 0.0
        continue
    
    # selected features
    ca_level = []
    current_level = []
    input_level = []

    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ca_value = float(row[5]) # current calcium is stored at the 6th column (index 5)
                ca_level.append(ca_value)
                current_value = float(row[3]) # current value is stored at the 4th column (index 3)
                current_level.append(current_value)
                synaptic = float(row[7])
                background = float(row[8])
                input_level.append(synaptic + background) # total input current
            except ValueError:
                continue

    # data is missing for neuron i
    if len(ca_level) == 0:
        mean_calcium[i] = 0.0
    else:
        mean_calcium[i] = np.mean(ca_level) # calculate the mean calcium activity (static)
    if len(current_level) == 0:
        mean_current[i] = 0.0
    else:
        mean_current[i] = np.mean(current_level) # calculate the mean current level (static)   
    if len(input_level) == 0:
        mean_input[i] = 0.0
    else:
        mean_input[i] = np.mean(input_level) # calculate the mean input level (static)
print("Mean calcium activity/current/input computed for all neurons.")
    
# write to a vtk file for Paraview visualization
points = vtk.vtkPoints()
mean_ca_array = vtk.vtkFloatArray()
mean_ca_array.SetName("Mean_Calcium_Activity")
area_arr = vtk.vtkStringArray()
area_arr.SetName("area")
mean_current_array = vtk.vtkFloatArray()
mean_current_array.SetName("Mean_Current_Level")
mean_input_array = vtk.vtkFloatArray()
mean_input_array.SetName("Mean_Input_Level")

for neuron_id, (x, y, z) in positions.items():
    points.InsertNextPoint(x, y, z)
    mean_ca_array.InsertNextValue(mean_calcium.get(neuron_id, 0.0))
    area_arr.InsertNextValue(areas.get(neuron_id, "unknown"))
    mean_current_array.InsertNextValue(mean_current.get(neuron_id, 0.0))
    mean_input_array.InsertNextValue(mean_input.get(neuron_id, 0.0))

# create vtk vertices
vertices = vtk.vtkCellArray()
num_points = points.GetNumberOfPoints()
for i in range(num_points):
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(i)

# create a vtk polydata object
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetVerts(vertices)

# add data arrays to the polydata
polydata.GetPointData().AddArray(mean_ca_array) # add mean calcium activity as point data
polydata.GetPointData().AddArray(area_arr) # add area as point data
polydata.GetPointData().SetActiveScalars("Mean_Calcium_Activity") # set the active scalars to mean calcium activity
polydata.GetPointData().AddArray(mean_current_array) # add mean current level as point data
polydata.GetPointData().AddArray(mean_input_array) # add mean input level as point data

# write the polydata to a vtk file
writer = vtk.vtkPolyDataWriter()
writer.SetFileName("neuron_positions.vtk")
writer.SetInputData(polydata)
writer.Write()
print("VTK file 'neuron_positions.vtk' written successfully.")

