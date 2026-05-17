# Reproducibility guide

This repository is organized around the computational pipeline used in the manuscript:

1. simulate spike trains from point-neuron circuits,
2. compute cross-correlograms (CCGs),
3. build train/validation/test dataloaders,
4. train connectivity and weight CNNs,
5. evaluate generalization on held-out or perturbed datasets,
6. compute CCG indicators and SHAP-based interpretation analyses.

The full manuscript-scale experiments can be computationally expensive and may generate large datasets. For this reason, generated data, trained models, and result tables are not tracked in GitHub by default. The `data/` and `results/` folders are kept with `.gitkeep` placeholders.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate omnicnn-connectivity
```

Or use pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick smoke tests

The example shell scripts provide end-to-end commands for small test runs:

```bash
bash examples/smoke_test_m2p.sh
bash examples/smoke_test_network.sh
```

These scripts are intended to verify that the environment, imports, simulation modules, dataloader generation, and CNN training entry points work on a local machine.

## Many-to-one-postsynaptic pipeline

Typical order:

```text
scripts/01_simulate_many_to_one.py
scripts/02_make_m2p_dataloaders.py
scripts/06_train_connectivity_cnn.py
scripts/07_train_weight_cnn.py
scripts/08_compute_ccg_indicators_fmm.py
scripts/09_SHAP.py
```

## Static-network pipeline

Typical order:

```text
scripts/03_simulate_static_network.py
scripts/04_make_network_dataloaders.py
scripts/05_make_one_dataloader.py
scripts/06_train_connectivity_cnn.py
scripts/07_train_weight_cnn.py
```

## Notes on full paper reproduction

The paper-scale experiments require running multiple simulation regimes and repeated CNN training runs. The exact parameter settings used in the paper can be reproduced by adapting the shell scripts in `examples/` and the readable parameter notes in `configs/`.

The repository focuses on the reusable computational framework rather than storing large regenerated outputs. Users who want to reproduce the full manuscript figures should first regenerate the required simulation and evaluation tables, then use their preferred plotting scripts to reproduce the final panels.


# Figure generation notes

This repository primarily documents the reusable pipeline for simulation, CCG computation, dataloader construction, CNN training, evaluation, indicator analysis, and SHAP interpretation.

The main manuscript figures were generated from aggregated summary tables produced by the pipeline. Because many figure scripts only format already-summarized Excel/CSV tables, they are not required for applying the framework to new datasets.

Recommended practice for public use:

- keep the core pipeline scripts in `scripts/`,
- keep small runnable examples in `examples/`,
- keep large generated tables and trained models outside GitHub or provide them through a data archive,
- add figure plotting scripts only when they are necessary to reproduce a central quantitative result.

