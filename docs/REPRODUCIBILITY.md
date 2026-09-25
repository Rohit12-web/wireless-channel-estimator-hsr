# Reproducibility Guide

## Authoritative source file

The DOI repository intentionally contains one authoritative research pipeline:

`src/publication_pipeline_v6_transactions_final.py`

Older V4/V5 pipelines, test scripts, exploratory DnCNN/U-Net implementations,
smoke-test output folders, manually generated BER/SE curves, and trained
checkpoint copies from the working directory are excluded.

## Recommended workflow

1. Create a clean Python environment.
2. Install `requirements.txt`.
3. Run the smoke profile and confirm it completes.
4. Run the publication profile on a GPU-enabled TensorFlow environment.
5. Record the TensorFlow/Python/GPU versions used for the final paper run.
6. Preserve the exact seed count and configuration used for each manuscript
   figure/table.
7. Copy only the final manuscript CSV/JSON result files and final figures into
   `results/` before creating the archival GitHub release.

## Publication run

```bash
python src/publication_pipeline_v6_transactions_final.py     --mode publication     --output-dir rohitash_ieee_access_results
```

## Scientific reporting

The implementation includes strong statistical references. The manuscript
should not imply universal superiority over condition-matched Temporal LMMSE
when the measured results do not support that claim.

Model inference latency reported by the profiler excludes pilot acquisition,
communication signaling, and physical IRS switching time.

The repository does not include generated neural checkpoints by default because
they are large derived artifacts. If pretrained models are later released,
attach them separately to a tagged GitHub/Zenodo release and document the exact
code version and seed that produced each checkpoint.
