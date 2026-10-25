# Calibration Count Bias, research code

This repository is a packaging of existing numerical experiments for investigating prediction set calibration, coverage statistics and finite sample effects. It contains the original GPU confirmation notebook and the Python source files embedded within that notebook.

## Contents

* `notebooks/Calibration_Count_One_GPU_Confirmation_v3.ipynb`, original confirmation workflow, with execution outputs cleared for safe publishing
* `src/confirm_once.py`, `src/confirmation_audit.py`, `src/confirmation_extra.py`, `src/confirmation_summary.py`, main runner and audit scripts extracted from the notebook
* `src/requirements.txt`, dependencies as provided by the original experiment
* `src/test_runner_local.py`, local test source
* `src/sources.txt`, original source and dataset information

## Running

Open the notebook on a machine with the GPUs and model access required by the original workflow, review its dependency and download cells, then configure the working directory. The notebook embeds a checksum-verified source archive that it extracts itself. The extracted files are provided under `src/` for easier inspection. No study runs were executed during repository packaging.

## Scope and provenance

These are recovered original notebook and source files, not newly fabricated experimental results. Data, model weights and large model output artifacts are not included. Review dataset permissions, verify dependencies, and test before making reproducibility claims. Authorship and degree of independent implementation must be verified by the applicant before sharing as admissions evidence.
