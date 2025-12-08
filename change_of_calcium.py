# change of calcium level
import os
import numpy as np
import pandas as pd
import vtk

# 1) read the positions of neurons
def load_positions(positions_file: str, max_neurons: int | None = None):
    """This function loads neuron positions and areas from the dataset.
    Input: file name of the dataset, the maximum number of neurons to read
    Return: positions (dict of nid -> (x,y,z)), areas (dict of nid -> area string), subset_ids (list of neuron ids)."""
    positions = {}
    areas = {}     

    with open(positions_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            nid = int(parts[0]) - 1   # zero-based
            x, y, z = map(float, parts[1:4]) # read coordinates
            area = parts[6] # read area
            positions[nid] = (x, y, z)
            areas[nid] = area

    all_ids_sorted = sorted(positions.keys())
    if max_neurons is not None:
        subset_ids = all_ids_sorted[:max_neurons] # only return a subset with maximum number of neurons
    else:
        subset_ids = all_ids_sorted

    return positions, areas, subset_ids

# 2) read the calcium data from monitors
def load_calcium_data(monitor_file: str, subset_ids: list[int], max_step: int = None):
    """This function loads the time-series calcium data for each coresponded neuron from the monitor file.
    Input: monitor file path, subset of neuron ids to read
    Return: time-series calcium data for each neuron"""
    calcium_data_dict = {}
    step = None
    for nid in subset_ids:
        path = os.path.join(monitor_file, f"0_{nid}.csv")
        if not os.path.exists(path):
            print(f"Warning: monitor file not found: {path}")
            continue
        monitor_data = pd.read_csv(path)
        calcium_data = monitor_data.iloc[5].to_numpy()  # assuming calcium data is in the 6th column (index 5)
        step = monitor_data.iloc[0].to_numpy()  # time steps in the first column
        if max_step is not None:
            calcium_data = calcium_data[:max_step]
        calcium_data_dict[nid] = calcium_data

    return calcium_data_dict, step

# 3) create vtkPolyData for neurons with calcium data at a specific time step
def create_calcium_polydata(positions: dict[int, tuple[float, float, float ]],
                             calcium_data: dict[int, np.ndarray],
                             subset_ids: list[int],
                             time_step: int) -> vtk.vtkPolyData:
    """This function creates a vtkPolyData object for neurons with calcium data at a specific time step.
    Input: positions dict, calcium data dict, subset of neuron ids, time step index
    Return: vtkPolyData object with neuron positions and calcium levels as point data."""
    points = vtk.vtkPoints()
    calcium_array = vtk.vtkFloatArray()
    calcium_array.SetName("CalciumLevel")

    for nid in subset_ids:
        if nid not in positions or nid not in calcium_data:
            continue
        x, y, z = positions[nid]
        points.InsertNextPoint(x, y, z)
        calcium_level = calcium_data[nid][time_step]
        calcium_array.InsertNextValue(calcium_level)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(calcium_array)
    polydata.GetPointData().SetActiveScalars("CalciumLevel")

    return polydata

# 4) write the calcium polydata to VTP files for multiple time steps
def write_calcium_time_series(polydata_by_step: dict[int, vtk.vtkPolyData],
                                time_steps: list[int],
                                output_dir: str,
                                base_name: str = "neuron_calcium_step_",
                                write_pvd: bool = True,
                                pvd_name: str = "neurons_calcium_time_series.pvd"):
        """This function writes the calcium polydata to VTP files for multiple time steps.
        Input: dict of time step -> vtkPolyData, list of time steps, output directory, base name for files, whether to write PVD file, PVD file name."""
        os.makedirs(output_dir, exist_ok=True)
    
        for step in time_steps:
            if step not in polydata_by_step:
                continue
            polydata = polydata_by_step[step]
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(os.path.join(output_dir, f"{base_name}{step:06d}.vtp"))
            writer.SetInputData(polydata)
            writer.Write()
    
        if write_pvd:
            pvd_path = os.path.join(output_dir, pvd_name)
            with open(pvd_path, 'w') as pvd_file:
                pvd_file.write('<?xml version="1.0"?>\n')
                pvd_file.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
                pvd_file.write('  <Collection>\n')
                for step in time_steps:
                    if step not in polydata_by_step:
                        continue
                    vtp_file = f"{base_name}{step:06d}.vtp"
                    pvd_file.write(f'    <DataSet timestep="{step}" group="" part="0" file="{vtp_file}"/>\n')
                pvd_file.write('  </Collection>\n')
                pvd_file.write('</VTKFile>\n')

def main_pipeline(positions_file: str,
                      monitor_dir: str,
                      output_dir: str,
                      max_neurons: int | None = None,
                      max_step: int | None = None):
    """Main pipeline to process calcium data and write to VTP files.
    Input: positions file path, monitor directory, output directory, list of time steps, maximum number of neurons, maximum time steps."""
    # Load positions and areas
    positions, areas, subset_ids = load_positions(positions_file, max_neurons)
    print(f"Loaded {len(subset_ids)} neurons for calcium data processing.")

    # Load calcium data
    calcium_data, step_array = load_calcium_data(monitor_dir, subset_ids, max_step)
    print(f"Loaded calcium data for {len(calcium_data)} neurons.")

    # Create polydata for each time step
    polydata_by_step = {}
    for step in step_array:
        polydata = create_calcium_polydata(positions, calcium_data, subset_ids, step)
        polydata_by_step[step] = polydata
    print(f"Created calcium polydata for {len(polydata_by_step)} time steps.")

    # Write to VTP files
    write_calcium_time_series(polydata_by_step, step_array, output_dir)
    print(f"Wrote calcium VTP files to {output_dir}.")


if __name__ == "__main__":
    POS_FILE = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-calcium\\positions\\rank_0_positions_filtered.txt"
    MON_DIR = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\monitors"
    OUT_DIR = "output_calcium_vtp"
    main_pipeline(
        positions_file=POS_FILE,
        monitor_dir=MON_DIR,
        output_dir=OUT_DIR,
    )
