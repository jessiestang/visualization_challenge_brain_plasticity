
import os
import csv
import vtk
import numpy as np


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


def compute_mean_from_monitors(monitor_dir: str, subset_ids):
    """
    This function computes mean calcium, mean current, mean input for each neuron
    Assumes:
      column 5 = current calcium
      column 3 = activity (used as 'current')
      column 7 + 8 = synaptic input ('input')
    Input: directory of correponded dataset, a subset of neuron ids to process
    Return: mean_calcium, mean_current, mean_input (each is a dict of nid -> mean value)
    """
    mean_calcium = {}
    mean_current = {}
    mean_input = {}

    for nid in subset_ids:
        path = os.path.join(monitor_dir, f"0_{nid}.csv")
        if not os.path.exists(path):
            print(f"Warning: monitor file not found: {path}")
            mean_calcium[nid] = 0.0
            mean_current[nid] = 0.0
            mean_input[nid] = 0.0
            continue

        ca_vals, cur_vals, inp_vals = [], [], []

        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    ca_vals.append(float(row[5]))
                    cur_vals.append(float(row[3]))
                    inp_vals.append(float(row[7]) + float(row[8]))
                except (ValueError, IndexError):
                    continue

        mean_calcium[nid] = np.mean(ca_vals)
        mean_current[nid] = np.mean(cur_vals)
        mean_input[nid] = np.mean(inp_vals)

    return mean_calcium, mean_current, mean_input


def build_points_and_pointdata(positions, areas, subset_ids):
    """
    This function creates vtkPoints and point-data arrays. 
    Return: points, nid_to_pid, mean_ca_array, area_arr, mean_current_array, mean_input_array
    """
    points = vtk.vtkPoints()
    nid_to_pid = {}

    # mean_ca_array = vtk.vtkFloatArray()
    # mean_ca_array.SetName("Mean_Calcium_Activity")

    area_arr = vtk.vtkStringArray()
    area_arr.SetName("area")

    # mean_current_array = vtk.vtkFloatArray()
    # mean_current_array.SetName("Mean_Current_Level")

    # mean_input_array = vtk.vtkFloatArray()
    # mean_input_array.SetName("Mean_Input_Level")

    for nid in subset_ids:
        x, y, z = positions[nid]
        pid = points.InsertNextPoint(x, y, z) # translate nid to pid
        nid_to_pid[nid] = pid

        # mean_ca_array.InsertNextValue(mean_calcium.get(nid, 0.0))
        area_arr.InsertNextValue(areas.get(nid, "unknown"))
        # mean_current_array.InsertNextValue(mean_current.get(nid, 0.0))
        # mean_input_array.InsertNextValue(mean_input.get(nid, 0.0))

    return (points,nid_to_pid,area_arr)


def load_edges_for_step(step: int, network_dir: str, subset_ids_set):
    """
    This function reads the edges from the given dataset.
    Input: step number, network directory, a set of subset neuron ids to filter
    Return: list of edges (src_id, tgt_id, weight)
    """
    fname = os.path.join(network_dir, f"rank_0_step_{step}_in_network.txt")
    edges = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tgt_id = int(parts[1]) - 1 # zero-based
            src_id = int(parts[3]) - 1 # zero-based
            w = float(parts[4])

            if src_id not in subset_ids_set or tgt_id not in subset_ids_set:
                continue

            edges.append((src_id, tgt_id, w))
    return edges

def build_polydata_for_step(
    step: int,
    points: vtk.vtkPoints,
    nid_to_pid: dict[int, int],
    area_arr: vtk.vtkStringArray,
    edges_this: list[tuple[int, int, float]],
    edge_keys_prev: set[tuple[int, int]],
    edge_keys_next: set[tuple[int, int]],
    is_first: bool = False,
    is_last: bool = False,
) -> vtk.vtkPolyData:
    """
    Create vtkPolyData for this simulation step:
      - points: neurons
      - lines: synapses
      - cell scalar: edge_state (1 new, 0 persist, -1 disappearing)
      - cell scalar: synapse_weight
    """
    lines = vtk.vtkCellArray()

    weights_arr = vtk.vtkFloatArray()
    weights_arr.SetName("synapse_weight")

    edge_state_arr = vtk.vtkIntArray()
    edge_state_arr.SetName("edge_state")

    

    for src_id, tgt_id, w in edges_this:
        key = (src_id, tgt_id)
        if is_first or is_last:
            state = 0   # at boundaries, treat all as persistent
        else:
            if key not in edge_keys_prev:
                state = 1   # newly created at this step
            elif key not in edge_keys_next:
                state = -1  # will disappear after this step
            else:
                state = 0   # persistent

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, nid_to_pid[src_id])
        line.GetPointIds().SetId(1, nid_to_pid[tgt_id])
        lines.InsertNextCell(line)

        weights_arr.InsertNextValue(w)
        edge_state_arr.InsertNextValue(state)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    # point data
    #poly.GetPointData().AddArray(mean_ca_array)
    poly.GetPointData().AddArray(area_arr)
    #poly.GetPointData().AddArray(mean_current_array)
    #poly.GetPointData().AddArray(mean_input_array)
    #poly.GetPointData().SetActiveScalars("Mean_Calcium_Activity")

    # cell data (edges)
    poly.GetCellData().AddArray(weights_arr)
    poly.GetCellData().AddArray(edge_state_arr)
    poly.GetCellData().SetActiveScalars("edge_state")

    return poly


def write_vtp_for_steps(
    time_steps,
    polydata_by_step: dict[int, vtk.vtkPolyData],
    output_dir: str,
    base_name: str = "initial_neuron_network_step_",
    write_pvd: bool = True,
    pvd_name: str = "initial_neurons_time_series.pvd",
):
    os.makedirs(output_dir, exist_ok=True)

    written_steps = []

    for step in time_steps:
        poly = polydata_by_step[step]
        out_path = os.path.join(output_dir, f"{base_name}{step}.vtp")

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(out_path)
        writer.SetInputData(poly)
        writer.Write()

        written_steps.append(step)
        print(f"[step {step}] wrote {out_path}")

    if write_pvd:
        pvd_path = os.path.join(output_dir, pvd_name)
        with open(pvd_path, "w") as f:
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write("  <Collection>\n")
            for step in written_steps:
                fname = f"{base_name}{step:05d}.vtp"
                f.write(f'    <DataSet timestep="{step}" file="{fname}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")
        print("Wrote PVD:", pvd_path)


def complete_pipeline(
    positions_file: str,
    network_dir: str,
    monitor_dir: str,
    output_dir: str,
    time_steps: list[int],
    max_neurons: int | None = 5000,
):
    # 1) positions + subset
    positions, areas, subset_ids = load_positions(positions_file, max_neurons)
    subset_ids_set = set(subset_ids)
    print(f"Loaded {len(positions)} neurons, using {len(subset_ids)}")

    # 2) monitor-based metrics (or replace with your own)
    #mean_ca, mean_curr, mean_inp = compute_mean_from_monitors(monitor_dir, subset_ids)
    #print("Computed mean calcium, current, input for subset neurons")

    # 3) shared VTK points + point data
    (
        points,
        nid_to_pid,
        area_arr,
    ) = build_points_and_pointdata(
        positions, areas, subset_ids
    )

    # 4) load edges for all steps
    edges_by_step = {}
    edge_keys_by_step = {}

    time_steps = sorted(time_steps)
    for step in time_steps:
        edges = load_edges_for_step(step, network_dir, subset_ids_set)
        edges_by_step[step] = edges
        edge_keys_by_step[step] = set((s, t) for s, t, _ in edges)
        print(f"step {step}: {len(edges)} edges (after subset filtering)")

    # 5) build polydata per step with edge_state
    poly_by_step = {}

    for i, step in enumerate(time_steps):
        prev_keys = edge_keys_by_step[time_steps[i - 1]] if i > 0 else set()
        next_keys = edge_keys_by_step[time_steps[i + 1]] if i < len(time_steps) - 1 else set()
        is_first = (i == 0) # handle boundary file
        is_last = (i == len(time_steps) - 1) # handle boundary file

        poly = build_polydata_for_step(
            step,
            points,
            nid_to_pid,
            area_arr,
            edges_by_step[step],
            prev_keys,
            next_keys,
            is_first = is_first,
            is_last = is_last,
        )
        poly_by_step[step] = poly
    print("Built polydata for all steps")

    # 6) write .vtp series (+ .pvd)
    write_vtp_for_steps(time_steps, poly_by_step, output_dir)
    print("Completed writing output VTP series.")


if __name__ == "__main__":
    POS_FILE = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-calcium\\positions\\rank_0_positions_filtered.txt"
    NET_DIR = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-calcium\\network"
    MON_DIR = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\monitors"
    OUT_DIR = "initial_edge_state_vtp"

    TIME_STEPS = np.linspace(0, 50000, 6, dtype=int)  # test steps
    print("Time steps to process:", TIME_STEPS)
    complete_pipeline(
        positions_file=POS_FILE,
        network_dir=NET_DIR,
        monitor_dir=MON_DIR,
        output_dir=OUT_DIR,
        time_steps=TIME_STEPS,
        max_neurons=None,  # or None for all
    )
