# OmniCNN Infer Connectivity

This repository contains code for simulating point-neuron spike trains, computing pairwise cross-correlograms (CCGs), and training CNN models to infer synaptic connectivity and synaptic weight from CCGs. The code is organized as a reproducible pipeline for the manuscript and as a starting framework for applying CCG-based CNN inference to other datasets.

## Repository layout

```text
OmniCNN_Infer_Connectivity/
├── PointNeuron_Simulation/        # Custom point-neuron simulation package
├── Functions.py                   # Shared ML, dataloader, CCG, indicator, and plotting utilities
├── scripts/                       # Ordered pipeline entry points
├── examples/                      # Runnable shell-script examples / smoke tests
├── configs/                       # Optional readable example parameter settings
├── docs/                          # Reproducibility and figure-generation notes
├── data/                          # Generated or user-provided data; not tracked by Git
├── results/                       # Model outputs and analysis results; not tracked by Git
├── environment.yml                # Conda environment
├── requirements.txt               # pip environment alternative
└── README.md
```

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/XiaoqianSun0104/OmniCNN_Infer_Connectivity.git
cd OmniCNN_Infer_Connectivity
conda env create -f environment.yml
conda activate omnicnn-connectivity
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default `environment.yml` uses CPU PyTorch for portability. On GPU clusters, install the PyTorch build matching your CUDA version following the official PyTorch instructions, then install the remaining packages from `requirements.txt`.



## Pipeline overview

The scripts are numbered in the order they are typically used. Each script supports command-line arguments; use `--help` to inspect available options.

### Many-to-one-postsynaptic motif

| Script | Purpose |
|---|---|
| `scripts/01_simulate_many_to_one.py` | Simulate a controlled many-presynaptic-to-one-postsynaptic motif and compute CCGs. |
| `scripts/02_make_m2p_dataloaders.py` | Build dataloaders from many-to-one-postsynaptic datasets. |
| `scripts/08_compute_ccg_indicators_fmm.py` | Compute CCG indicators and feature-map-matching analysis. |
| `scripts/09_SHAP.py` | Train random-forest models using CCG indicators and run SHAP interpretation. |

Typical order:

```text
01_simulate_many_to_one.py
02_make_m2p_dataloaders.py
06_train_connectivity_cnn.py and/or 07_train_weight_cnn.py
08_compute_ccg_indicators_fmm.py
09_SHAP.py
```

### Static-network datasets

| Script | Purpose |
|---|---|
| `scripts/03_simulate_static_network.py` | Simulate static excitatory/inhibitory network datasets. |
| `scripts/04_make_network_dataloaders.py` | Build train/validation/test dataloaders from multiple simulated networks. |
| `scripts/05_make_one_dataloader.py` | Build one-network dataloaders for held-out evaluation. |

Typical order:

```text
03_simulate_static_network.py
04_make_network_dataloaders.py
05_make_one_dataloader.py
06_train_connectivity_cnn.py and/or 07_train_weight_cnn.py
```

### Shared CNN training scripts

| Script | Purpose |
|---|---|
| `scripts/06_train_connectivity_cnn.py` | Train ConnCNN for binary connectivity inference. |
| `scripts/07_train_weight_cnn.py` | Train WeightCNN for signed synaptic-weight inference. |


## Example commands

Run a static-network simulation:

```bash
python scripts/02_simulate_static_network.py \
  --dataFolderName data/network_smoke \
  --segFolderName static_network_1 \
  --simulationT 600000 \
  --num_Xe 10 --num_Xi 10 --num_Se 80 --num_Si 20
```

Create train/validation/test dataloaders from multiple simulated networks:

```bash
python scripts/04_make_network_dataloaders.py \
  --dataFolderName data/network_smoke \
  --segFolderNamePrefix static_network \
  --networkId_List 1 2 3 4
```

Train the connectivity CNN:

```bash
python scripts/06_train_connectivity_cnn.py \
  --dataloaderType Raw \
  --dataloaderName TTV_1_2_3_TTV.pkl \
  --unseen_dlName oneDL_4.pkl \
  --resultFolderName network_smoke \
  --numRuns 30 \
```

Train the weight CNN:

```bash
python scripts/07_train_weight_cnn.py \
  --dataloaderType Raw \
  --dataloaderName TTV_1_2_3_TTV.pkl \
  --unseen_dlName oneDL_4.pkl \
  --resultFolderName network_smoke \
  --numRuns 30 \
```

## Config files

The current entry-point scripts use command-line arguments. The YAML files in `configs/` are optional readable examples of parameter settings; they are included to document typical simulation configurations, not because the scripts require YAML input.

## Applying the framework to another dataset

To apply the CNN framework to another spike-train dataset, the expected workflow is:

1. Format spike times as neuron-wise spike-time tables.
2. Compute pairwise CCGs with the same binning convention used in the paper: ±50 ms with 1 ms bins
3. Build dataloaders containing CCG arrays, binary connectivity labels, signed/continuous synaptic weights when available, and sample identifiers.
4. Train or fine-tune the connectivity CNN and/or weight CNN using the scripts in `scripts/`.
5. Evaluate predictions and compute CCG indicators to interpret when the model succeeds or fails.

For real data without ground-truth connectivity labels, the trained CNN can generate candidate predictions, but quantitative validation requires experimentally confirmed connectivity or another trusted reference.

## Reproducibility

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for details on reproducing pipeline components and adapting the example shell scripts to larger manuscript-scale experiments.

This repository focuses on the computational framework: simulation, CCG generation, dataloader construction, CNN training, evaluation, indicator analysis, and SHAP interpretation. Manuscript figures were generated from aggregated summary tables produced by this pipeline. 

## Citation

If you use this code, please cite the associated manuscript once available.

## Contact

For questions, please open a GitHub issue or contact the repository maintainer.
