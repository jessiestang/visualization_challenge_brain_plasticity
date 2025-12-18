# change of calcium level
import os
import numpy as np
import pandas as pd
import vtk
from typing import Dict, List, Optional,Sequence

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
    Input: monitor file path, subset of neuron ids to read, max_step: maximum number of rows to read
    Return: time-series calcium data for each neuron (dict of nid -> 1D numpy array)"""
    calcium_data_dict = {}
    
    for nid in subset_ids:
        path = os.path.join(monitor_file, f"0_{nid}.csv")
        if not os.path.exists(path):
            continue
        
        # Read only the first max_step rows from the CSV
        read_kwargs = {'sep': ';', 'header': None}
        if max_step is not None:
            read_kwargs['nrows'] = max_step
        
        monitor_data = pd.read_csv(path, **read_kwargs)
        
        # Extract calcium data from column 5 (6th column)
        try:
            calcium_series = monitor_data.iloc[:, 5].to_numpy()
            calcium_series = np.asarray(calcium_series).ravel().astype(float)
            calcium_data_dict[nid] = calcium_series
        except Exception:
            continue
    
    return calcium_data_dict
def read_calcium_at_indices(
    csv_path: str,
    wanted_indices: Sequence[int],
    sep: str = ";",
    calcium_col: int = 5,   # 0-based; change to 4 if your file differs
    comment_prefix: str = "#",
) -> List[Optional[float]]:
    """
    Returns calcium values at given row indices (0-based data rows, excluding comments).
    If an index is out of range, returns None for that position.
    This makes sure that the the timeline of the calcium data is aligned with the edge data
    """
    wanted_set = set(wanted_indices)
    max_wanted = max(wanted_indices) if wanted_indices else -1

    out_map: Dict[int, float] = {}
    data_row = -1  # counts only non-empty, non-comment rows

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(comment_prefix):
                continue
            data_row += 1

            if data_row > max_wanted:
                break

            if data_row in wanted_set:
                parts = [p.strip() for p in line.split(sep)]
                try:
                    out_map[data_row] = float(parts[calcium_col])
                except (IndexError, ValueError):
                    out_map[data_row] = float("nan")

    # preserve order of wanted_indices
    return [out_map.get(i, None) for i in wanted_indices]

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

        arr = calcium_data[nid]
        # Bounds check: if the requested index is out of range, insert 0.0
        if time_step < 0 or time_step >= arr.shape[0]:
            calcium_array.InsertNextValue(0.0)
            continue

        # Ensure we insert a Python float scalar into the VTK array
        try:
            calcium_level = float(arr[time_step])
        except Exception:
            # Fallback if the element is not convertible
            calcium_level = 0.0
        calcium_array.InsertNextValue(calcium_level)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(calcium_array)
    polydata.GetPointData().SetActiveScalars("CalciumLevel")

    return polydata

# 4) write the calcium polydata to VTP files for multiple time steps
def write_calcium_time_series(polydata_by_step: dict[int, vtk.vtkPolyData],
                                time_values: list[int],
                                output_dir: str,
                                base_name: str = "100000_neuron_calcium_step_",
                                write_pvd: bool = True,
                                pvd_name: str = "neurons_calcium_100000.pvd"):
    """This function writes the calcium polydata to VTP files for multiple selected time values.
    `polydata_by_step` keys are indices (0..N-1); `time_values` contains the actual timeline values.
    Filenames will include the actual time value. The PVD will reference those time values."""
    os.makedirs(output_dir, exist_ok=True)

    for idx, time_val in enumerate(time_values):
        if idx not in polydata_by_step:
            continue
        polydata = polydata_by_step[idx]
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(os.path.join(output_dir, f"{base_name}{time_val:06d}.vtp"))
        writer.SetInputData(polydata)
        writer.Write()

    if write_pvd:
        pvd_path = os.path.join(output_dir, pvd_name)
        with open(pvd_path, 'w') as pvd_file:
            pvd_file.write('<?xml version="1.0"?>\n')
            pvd_file.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            pvd_file.write('  <Collection>\n')
            for idx, time_val in enumerate(time_values):
                if idx not in polydata_by_step:
                    continue
                vtp_file = f"{base_name}{time_val:06d}.vtp"
                pvd_file.write(f'    <DataSet timestep="{time_val}" group="" part="0" file="{vtp_file}"/>\n')
            pvd_file.write('  </Collection>\n')
            pvd_file.write('</VTKFile>\n')

def main_pipeline(positions_file: str,
                      monitor_dir: str,
                      output_dir: str,
                      time_values: Sequence[int],
                      calcium_stride: int = 1,
                      max_neurons: int | None = None,
                      ):
    """Main pipeline to process calcium data and write to VTP files.
    Reads only the requested row indices from each neuron's CSV so timeline aligns with edge TIME_LINE.
    `time_values` is the list/array of actual timeline values (e.g., TIME_LINE); `calcium_stride` is the sampling interval used when CSVs were written."""
    # Load positions and areas
    positions, areas, subset_ids = load_positions(positions_file, max_neurons)
    print(f"Loaded {len(subset_ids)} neurons for calcium data processing.")

    # Determine which row indices in the CSV correspond to the requested timeline
    wanted_indices = [int(s // calcium_stride) for s in time_values]

    # Load calcium only at the wanted indices for every neuron
    calcium_data = {}
    for nid in subset_ids:
        csv_path = os.path.join(monitor_dir, f"0_{nid}.csv")
        if not os.path.exists(csv_path):
            continue
        vals = read_calcium_at_indices(csv_path, wanted_indices, sep=';', calcium_col=5)
        if not vals:
            continue
        arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
        calcium_data[nid] = arr

    print(f"Loaded calcium data (selected indices) for {len(calcium_data)} neurons.")

    if len(calcium_data) == 0:
        print("No calcium data found at requested indices; nothing to write.")
        return

    # Build polydata for each requested time index (indices 0..N-1 correspond to time_values)
    polydata_by_step = {}
    for idx in range(len(wanted_indices)):
        polydata = create_calcium_polydata(positions, calcium_data, subset_ids, idx)
        polydata_by_step[idx] = polydata

    print(f"Created calcium polydata for {len(polydata_by_step)} selected time steps.")

    # Write to VTP files (filename includes the actual time value) and write PVD with those time values
    write_calcium_time_series(polydata_by_step, list(time_values), output_dir)
    print(f"Wrote calcium VTP files and PVD to {output_dir}.")


if __name__ == "__main__":
    POS_FILE = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-calcium\\positions\\rank_0_positions_filtered.txt"
    MON_DIR = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\monitors"
    OUT_DIR = "output_calcium_initial_vtp"
    TIME_LINE = np.linspace(0, 100000, 11, dtype=int)
    calcium_stride = 100

    main_pipeline(
        positions_file=POS_FILE,
        monitor_dir=MON_DIR,
        output_dir=OUT_DIR,
        time_values=TIME_LINE,
        calcium_stride=calcium_stride,
    )
