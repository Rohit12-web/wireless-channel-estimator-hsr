Wireless Channel Estimator for High-Speed Railway Communications

Reproducibility repository for the research work “Mobility-Aware Channel Estimation for IRS-Assisted High-Speed Railways.”

The implementation investigates future effective-channel prediction for a single-IRS-assisted high-speed railway (HSR) MIMO link. The framework combines pilot-based channel acquisition, physics-guided mobility modeling, residual temporal learning, predictive uncertainty, adaptive pilot evaluation, and prediction-aware IRS control.

Repository: https://github.com/Rohit12-web/wireless-channel-estimator-hsr
Archival DOI: To be added after the GitHub release is archived through Zenodo.

Repository Structure

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
├── data/
│   └── README.md
└── results/
    └── README.md

The repository intentionally contains the publication-oriented implementation only. Older development versions, smoke-test outputs, generated model checkpoints, temporary figures, and exploratory scripts are excluded from the archival source repository.

Main Implementation

The principal implementation is:

src/publication_pipeline_v6_transactions_final.py

The pipeline includes:

single-IRS-assisted effective-channel modeling,
H_eff = H_d + H_r Phi G;

continuous HSR motion with geometry-dependent Doppler evolution;

orthogonal MIMO pilot transmission and LS channel initialization;

matched, mismatched, and temporal LMMSE references;

spatial learning-based channel-estimation baselines;

CNN-BiLSTM and GRU-ODE-inspired temporal prediction baselines;

signed-Doppler/Jakes physics-guided future-channel prediction;

condition-aware BiLSTM residual refinement;

heteroscedastic predictive uncertainty estimation;

adaptive pilot-selection evaluation;

prediction-aware IRS-control evaluation; and

NMSE, channel-aging, mobility, BER, spectral-efficiency, robustness,
ablation, uncertainty-calibration, and computational-complexity studies.

Requirements

Python 3.10 or later is recommended.

Install the required packages with:

pip install -r requirements.txt

A GPU-enabled TensorFlow environment is recommended for the complete publication run.

Quick Implementation Check

A lightweight smoke test can be executed using:

python src/publication_pipeline_v6_transactions_final.py --mode smoke

This mode is intended only to verify that the implementation and dependencies execute correctly. Smoke-test outputs should not be used as final manuscript results.

Publication Run

The full publication configuration can be executed using:

python src/publication_pipeline_v6_transactions_final.py \
    --mode publication \
    --output-dir results/publication_run

The publication profile uses five main-model seeds by default. The seed count can be changed explicitly, for example:

python src/publication_pipeline_v6_transactions_final.py \
    --mode publication \
    --seeds 5 \
    --output-dir results/publication_run

The headline nonzero prediction horizon can also be overridden when required:

python src/publication_pipeline_v6_transactions_final.py \
    --mode publication \
    --prediction-delay-ms 1.0 \
    --output-dir results/publication_run

The complete publication profile is computationally intensive and may require a GPU-enabled TensorFlow environment.

Data Generation

No external measurement dataset is required by the main simulation pipeline.

Channel realizations are generated from the implemented physical communication model, including:

continuous train motion;

direct BS--MCR and reflected BS--IRS--MCR links;

Rician fading;

geometry-dependent path evolution;

Doppler variation and temporal correlation;

IRS phase configuration;

orthogonal pilot observations; and

additive complex Gaussian noise.

Additional information is provided in:

data/README.md

Reproducibility

The repository is structured so that the simulation configuration used for a manuscript result can be traced to the corresponding source implementation.

For final archival use:

use the exact code version associated with the submitted manuscript;

retain the seed count and evaluation settings used for each reported result;

keep smoke-test and reduced-budget runs separate from publication results;

store only the final CSV, JSON, and figure outputs used in the paper under results/; and

create a tagged GitHub release before archiving the repository with Zenodo.

Further details are available in:

docs/REPRODUCIBILITY.md

Results

The results/ directory is reserved for outputs corresponding to the final frozen manuscript configuration.

Development results, old pipeline outputs, smoke-test runs, and reduced-budget experiments should not be mixed with the archival publication results.

Citation

Citation metadata are provided in:

CITATION.cff

After Zenodo archives the tagged GitHub release, the assigned DOI should be added to both CITATION.cff and this README.

A DOI badge can then be added at the top of this README using the Zenodo-generated badge code.

License

See the LICENSE file for the terms associated with this research code.