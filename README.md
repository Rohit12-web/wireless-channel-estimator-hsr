# Wireless Channel Estimator for High-Speed Railway Communications

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22961234.svg)](https://doi.org/10.5281/zenodo.22961234)

Reproducibility repository for the research work **“Mobility-Aware Channel Estimation for IRS-Assisted High-Speed Railways.”**

This repository contains the publication-oriented implementation of a physics-guided and uncertainty-aware framework for future effective-channel prediction in a single-IRS-assisted high-speed railway (HSR) MIMO system. The framework combines pilot-based channel acquisition, mobility-aware physical modeling, residual temporal learning, predictive uncertainty, adaptive pilot evaluation, and prediction-aware IRS control.

> **Repository:** https://github.com/Rohit12-web/wireless-channel-estimator-hsr  
> **Archived release:** v1.0.0  
> **Zenodo DOI:** https://doi.org/10.5281/zenodo.22961234

## Repository Structure

```text
wireless-channel-estimator-hsr/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   └── publication_pipeline_v6_transactions_final.py
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── REPOSITORY_MANIFEST.md
└── results/
    ├── README.md
    ├── 01_nmse_vs_snr.png
    ├── 01b_nmse_temporal_fair.png
    ├── 02_nmse_vs_speed.png
    ├── 03_channel_aging.png
    ├── 04_pilot_efficiency_and_net_se.png
    ├── 06_ber_vs_snr_db.png
    ├── 07_se_vs_snr_db.png
    ├── 08_temporal_ablation.png
    ├── 09_k_factor_mismatch.png
    └── 10_phase_error_robustness.png
```

The repository intentionally retains only the publication-oriented source code and the selected result figures used for the archival research release. Older development versions, smoke-test outputs, generated model checkpoints, temporary plots, and exploratory implementations are excluded to keep the repository concise and traceable.

## Main Implementation

The principal implementation is:

```text
src/publication_pipeline_v6_transactions_final.py
```

The pipeline includes:

- single-IRS-assisted effective-channel modeling, `H_eff = H_d + H_r Phi G`;
- continuous HSR motion with geometry-dependent Doppler evolution;
- orthogonal MIMO pilot transmission and LS channel initialization;
- matched, mismatched, and temporal LMMSE references;
- spatial learning-based channel-estimation baselines;
- CNN-BiLSTM and GRU-ODE-inspired temporal prediction baselines;
- signed-Doppler/Jakes physics-guided future-channel prediction;
- condition-aware BiLSTM residual refinement;
- heteroscedastic predictive uncertainty estimation;
- adaptive pilot-selection evaluation;
- prediction-aware IRS-control evaluation; and
- NMSE, mobility, channel-aging, BER, spectral-efficiency, robustness,
  ablation, uncertainty, and computational-complexity studies.

## Requirements

Python 3.10 or later is recommended.

Install the required packages using:

```bash
pip install -r requirements.txt
```

A GPU-enabled TensorFlow environment is recommended for the complete training and evaluation pipeline.

## Quick Implementation Check

A lightweight implementation check can be executed using:

```bash
python src/publication_pipeline_v6_transactions_final.py --mode smoke
```

This mode is intended only to verify that the implementation and software dependencies execute correctly. Smoke-test outputs should not be treated as publication results.

## Publication Evaluation

The publication-oriented configuration can be executed using:

```bash
python src/publication_pipeline_v6_transactions_final.py \
    --mode publication \
    --output-dir results/publication_run
```

The complete publication evaluation is computationally intensive and is intended for a GPU-enabled TensorFlow environment.

The exact seed count, prediction horizon, training budget, and evaluation configuration used for a manuscript result should match the configuration reported for that particular experiment.

## Data Generation

No external measurement dataset is required by the main simulation pipeline.

All channel realizations are generated programmatically by the physical communication model implemented in:

```text
src/publication_pipeline_v6_transactions_final.py
```

The simulation includes:

- continuous train motion;
- direct BS--MCR and reflected BS--IRS--MCR propagation links;
- geometry-dependent path evolution;
- Rician fading;
- mobility-dependent Doppler evolution and temporal correlation;
- IRS phase configuration;
- orthogonal MIMO pilot observations; and
- additive complex Gaussian noise.

Because the experimental data are generated directly by the simulation code, no separate top-level dataset directory is required in this repository.

## Results

The `results/` directory contains the principal figures retained for the publication-oriented evaluation.

The included results cover:

- effective-channel NMSE versus SNR;
- fair temporal-estimator comparison;
- NMSE versus train speed;
- channel-aging performance;
- pilot-efficiency and net spectral-efficiency analysis;
- predictive BER versus SNR;
- predictive spectral efficiency versus SNR;
- temporal-model ablation analysis;
- Rician K-factor mismatch robustness; and
- IRS phase-error robustness.

The current figure files are:

```text
results/01_nmse_vs_snr.png
results/01b_nmse_temporal_fair.png
results/02_nmse_vs_speed.png
results/03_channel_aging.png
results/04_pilot_efficiency_and_net_se.png
results/06_ber_vs_snr_db.png
results/07_se_vs_snr_db.png
results/08_temporal_ablation.png
results/09_k_factor_mismatch.png
results/10_phase_error_robustness.png
```

Only figures corresponding to the manuscript-oriented evaluation should be retained in the archival release. Smoke-test outputs, reduced-budget experiments, obsolete figures, and historical development results should remain excluded.

Where available, the numerical CSV/JSON outputs used to generate the final figures may also be archived with a future release to further support reproducibility.

## Reproducibility

The repository is structured so that the simulation methodology and reported results can be traced to the publication-oriented source implementation.

For reproducible use:

1. use the source-code version associated with the relevant tagged release;
2. retain the seed count and evaluation settings reported for each experiment;
3. keep smoke-test and reduced-budget runs separate from publication results;
4. preserve manuscript figures together with their corresponding numerical outputs where available; and
5. cite the archived Zenodo release when referring to the exact software snapshot.

The first archived reproducibility release is:

```text
v1.0.0
DOI: 10.5281/zenodo.22961234
```

Further information is provided in:

```text
docs/REPRODUCIBILITY.md
```

## Citation

Citation metadata are provided in:

```text
CITATION.cff
```

The archived software release is available through Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.22961234

When using this implementation in academic work, please cite the Zenodo software release together with the associated research article, where appropriate.

## License

See the `LICENSE` file for the terms associated with this research code.
