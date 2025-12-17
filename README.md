#  Scientific visualizations of the neuron network activity and plasticity 

Repository and project owned by: Xuening Tang, Luka Waronig (association with: University of Amsterdam)

This repository contains visualization tools and analysis scripts developed for exploring neuronal network simulations of plasticity changes in the human brain, inspired by the IEEE SciVis Contest 2023.

Brain plasticity, the brain's ability to reorganize and form new synaptic connections, is fundamental to learning, memory formation, and recovery from injury. This project provides interactive visualizations and analysis pipelines to understand the dynamics of neuronal networks, focusing on calcium signaling, synaptic connections, and temporal patterns of neural activity.

## Dataset

The visualizations work with data from neuronal network simulations based on:
- **Neuron positions**: Derived from the Human Connectome Project
- **Simulation model**: Based on structural plasticity models that simulate how synaptic connections form and dissolve over time
- **Parameters**: Each neuron tracks multiple properties including calcium levels, firing rates, axon/dendrite counts, and brain region assignments

## Repository contents:

### Analysis scripts:

- **`simple_pipeline.py`**: Core data processing pipeline for loading and preprocessing neuronal simulation data
- **`read_neuro-position.py`**: Utilities for reading and processing neuron spatial position data
- **`calcium_level_over_time.py`**: Temporal analysis of calcium signaling dynamics across neurons
- **`change_of_calcium.py`**: Analysis of calcium concentration changes and patterns
- **`change_of_edge.py`**: Tracking synaptic connection formation and deletion events
- **`static_mean_calcium.py`**: Static visualization of mean calcium levels across brain regions

### Jupyter notebooks:

- **`Ca_convergence_temporaldynamics.ipynb`**: Analysis of calcium convergence patterns and temporal dynamics
- **`co-activation_areas.ipynb`**: Identification and visualization of co-activated brain regions
- **`stimulus_protocol.ipynb`**: Analysis of neural responses to different stimulus protocols

To a large extent, to make notebooks fully functional to collaborators we integrated the necessary functions originally written in the .py scripts.

## Key visualization features:

### 1. Calcium dynamics
- Temporal evolution of calcium levels across neurons
- Identification of calcium signaling patterns
- Correlation between calcium activity and synaptic changes

### 2. Network dynamics
- Visualization of synaptic connection changes over time
- Edge creation and deletion tracking
- Network topology evolution

### 3. Spatial analysis
- 3D visualization of neuron positions
- Brain region-specific activity patterns
- Co-activation mapping across different areas

### 4. Temporal analysis
- Time-series analysis of neural properties
- Convergence patterns in calcium signaling
- Stimulus response protocols

## Getting started:

### Installation:

```bash
git clone https://github.com/jessiestang/visualization_challenge_brain_plasticity.git
cd visualization_challenge_brain_plasticity
```

### For example, run:

```bash
python simple_pipeline.py
```

```bash
python calcium_level_over_time.py
```

```bash
jupyter notebook Ca_convergence_temporaldynamics.ipynb
```

## Scientific contexts explored:

Brain plasticity involves continuous reorganization of neuronal connections. Key mechanisms include:

- **Synaptic plasticity**: Formation and elimination of synapses based on neural activity
- **Calcium signaling**: Calcium ions serve as crucial messengers triggering plasticity mechanisms
- **Hebbian learning**: "Neurons that fire together, wire together"
- **Homeostatic regulation**: Networks maintain stable activity levels while allowing structural changes

## References:

[DATASET] Gerrits, T., Czappa, F., Banesh, D., and Wolf, F. (2022). \textit{IEEE SciVis Contest 2023 - Dataset of Neuronal Network Simulations of the Human Brain} (1.0) Zenodo.

[NEURON POSITIONS] Marcus DS, Harwell J, Olsen T, Hodge M, Glasser MF, Prior F, Jenkinson M, Laumann T, Curtiss SW, and Van Essen DC. (2011). \textit{Informatics and data mining: Tools and strategies for the Human Connectome Project}. Frontiers in Neuroinformatics 5:4. 

[SIMULATION SETUP] Rinke, Sebastian and Butz-Ostendorf, Markus and Hermanns, Marc-André and Naveau, Mikael and Wolf, Felix. (2018): \textit{A scalable algorithm for simulating the structural plasticity of the brain}.


## Repository License:
MIT license.

---

*This project explores the fascinating dynamics of the brain's ability to rewire itself, making the invisible visible through scientific and large-simulation-data-based visualization.*
