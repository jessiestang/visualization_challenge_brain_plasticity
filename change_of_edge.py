
import os
import csv
import vtk
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Optional,Sequence
from collections import defaultdict



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

def sample_neurons_per_area(
    positions_file: str,
    sample_size: int,
    max_neurons: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Randomly sample `sample_size` neurons from each area.

    Returns:
        dict: { area_name -> list of neuron_ids }
    """
    random.seed(seed)

    # 1. Group neurons by area
    neurons_by_area = defaultdict(list)

    with open(positions_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            nid = int(parts[0]) - 1      # zero-based neuron id
            area = parts[6]              # area label

            neurons_by_area[area].append(nid)

    # 2. Sample per area
    sampled = {}

    for area, neuron_ids in neurons_by_area.items():
        if max_neurons is not None:
            neuron_ids = neuron_ids[:max_neurons]

        if len(neuron_ids) <= sample_size:
            sampled[area] = neuron_ids
        else:
            sampled[area] = random.sample(neuron_ids, sample_size)

    return sampled

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

import math
import numpy as np

def mean_calcium_per_area_at_edge_steps(
    monitors_dir: str,
    sampled_by_area: Dict[str, List[int]],
    edge_steps: Sequence[int],
    calcium_stride: int = 100,
    calcium_col: int = 5,
) -> Dict[str, Dict[int, float]]:
    """
    Returns:
      { area_name: { edge_step: mean_calcium_over_sampled_neurons } }
    """
    wanted_indices = [s // calcium_stride for s in edge_steps]

    results: Dict[str, Dict[int, float]] = {}

    for area, neuron_ids in sampled_by_area.items():
        # accumulate per timestep
        sums = np.zeros(len(edge_steps), dtype=float)
        counts = np.zeros(len(edge_steps), dtype=int)

        for nid in neuron_ids:
            csv_path = os.path.join(monitors_dir, f"0_{nid}.csv")
            vals = read_calcium_at_indices(
                csv_path,
                wanted_indices=wanted_indices,
                sep=";",
                calcium_col=calcium_col,
            )

            for k, v in enumerate(vals):
                if v is None:
                    continue
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                sums[k] += v
                counts[k] += 1

        # finalize means
        area_series: Dict[int, float] = {}
        for k, step in enumerate(edge_steps):
            area_series[step] = (sums[k] / counts[k]) if counts[k] > 0 else float("nan")

        results[area] = area_series

    return results


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


def load_edges_for_step(step: int, network_dir: str, subset_ids_set, in_edge = True):
    """
    This function reads the edges from the given dataset.
    Input: step number, network directory, a set of subset neuron ids to filter
    Return: list of edges (src_id, tgt_id, weight)
    """
    if in_edge:  # open the in-connection file
        fname = os.path.join(network_dir, f"rank_0_step_{step}_in_network.txt")
    else:
        fname = os.path.join(network_dir, f"rank_0_step_{step}_out_network.txt")
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

def pipeine_counting_edge(positions_file: str,
    network_dir: str,
    time_steps: list[int],
    max_neurons: int | None = 5000):
    """
    This pipeline only count the number of edges at each simulation time step.
    It also counts the number of added edges and the number of deleted edges at each simulation time step.
    """
    # 1) positions + subset
    positions, areas, subset_ids = load_positions(positions_file, max_neurons)
    subset_ids_set = set(subset_ids)
    print(f"Loaded {len(positions)} neurons, using {len(subset_ids)}")

    # 2) load edges for all steps and record edge-key sets
    edge_keys_by_step = {}
    num_in_edge = []

    time_steps = sorted(time_steps)
    for step in time_steps:
        in_edges = load_edges_for_step(step, network_dir, subset_ids_set, in_edge=True)
        edge_keys = set((s, t) for s, t, _ in in_edges)
        edge_keys_by_step[step] = edge_keys
        num_in_edge.append(len(in_edges))
        print(f"step {step}: {len(in_edges)} in-edges (after subset filtering)")

    # 3) compute added / deleted counts per timestep
    added_counts = []
    deleted_counts = []
    prev_keys = None
    for step in time_steps:
        curr_keys = edge_keys_by_step[step]
        if prev_keys is None:
            added = 0
            deleted = 0
        else:
            added = len(curr_keys - prev_keys)
            deleted = len(prev_keys - curr_keys)

        added_counts.append(added)
        deleted_counts.append(deleted)
        prev_keys = curr_keys

    # 4) plot total, added and deleted edges per timestep
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, num_in_edge, linestyle="-", color="blue", label="total_in_edges")
    plt.title("Change of the number of links at each time step in brain area 1-10")
    plt.xlabel("Timestep (0-200000)")
    plt.ylabel("Number of links (area 1-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(time_steps, added_counts, linestyle="--", color="green", label="added_edges")
    plt.plot(time_steps, deleted_counts, linestyle="--", color="red", label="deleted_edges")
    plt.title("Addition/Deletion of the number of links at each time step in brain area 1-10")
    plt.xlabel("Timestep (0-200000)")
    plt.ylabel("Number of links (area 1-10)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def pipeline_calculating_calcium(positions_file: str,
    monitor_dir: str,
    time_steps: list[int],
    sample_size: int,
    max_neurons: int | None = 5000,):
    """
    This pipeline will sample sample_size neurons from each area (from 1 to 10) and calculate the mean calcium concentration at each simulation timestep.
    It will then try to compare the trend in changing of calcium level with that of the number of edges.
    """
    # 1) randomly sample neurons per area
    sampled_neurons = sample_neurons_per_area(
        positions_file=positions_file, sample_size=sample_size, max_neurons=max_neurons
    )

    # 2) compute mean calcium per sampled area at requested timesteps
    mean_by_area = mean_calcium_per_area_at_edge_steps(
        monitors_dir=monitor_dir, sampled_by_area=sampled_neurons, edge_steps=time_steps
    )

    # 3) aggregate overall mean across areas (ignore NaNs)
    overall_mean = []
    for step in time_steps:
        vals = [mean_by_area[a].get(step, float("nan")) for a in mean_by_area]
        overall_mean.append(float(np.nanmean(vals)) if len(vals) > 0 else float("nan"))

    
    # 5) plotting: per-area series (faint), overall mean (bold) and edge counts on twin axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # plot per-area traces
    for area, series in mean_by_area.items():
        ys = [series.get(step, float("nan")) for step in time_steps]
        ax1.plot(time_steps, ys, alpha=0.45, linewidth=1, label=f"area {area}")

    # overall mean
    ax1.plot(time_steps, overall_mean, color="black", linewidth=2.0, label="overall_mean")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Mean calcium (sampled neurons)")
    ax1.grid(True)
    ax1.set_title("Mean calcium (sampled neurons) vs timestep")

    # build combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc="upper right", fontsize="small")
    plt.show()



if __name__ == "__main__":
    POS_FILE = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-calcium\\positions\\rank_0_positions_filtered.txt"
    NET_DIR = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-calcium\\network"
    MON_DIR = "E:\\computational science\\scientific visualization\\SciVisContest23\\SciVisContest23\\viz-stimulus\\monitors"
    OUT_DIR = "initial_edge_state_vtp"

    TIME_STEPS = np.linspace(0, 200000, 21, dtype=int)  # test steps
    TIME_STEPS2 = np.linspace(500000, 700000, 21, dtype=int)
    TIME_STEPS3 = np.linspace(700000, 900000, 21, dtype=int)
    print("Time steps to process:", TIME_STEPS)
    """complete_pipeline(
        positions_file=POS_FILE,
        network_dir=NET_DIR,
        monitor_dir=MON_DIR,
        output_dir=OUT_DIR,
        time_steps=TIME_STEPS,
        max_neurons=None,  # or None for all
    )"""
    # plot for the in-link
    """pipeine_counting_edge(
    positions_file = POS_FILE,
    network_dir = NET_DIR,
    time_steps = TIME_STEPS3,
    max_neurons = None,
    )"""

    # plot for the mean calcium overtime
    pipeline_calculating_calcium(
        positions_file = POS_FILE,
        monitor_dir = MON_DIR,
        time_steps = TIME_STEPS3,
        sample_size= 100,
        max_neurons = None,
    )

