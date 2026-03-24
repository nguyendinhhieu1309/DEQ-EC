# DEQ-EC: Dynamic Exposure-Aware Quantification of Echo Chambers

## Authors

- **Nguyen Dinh Hieu** (FPT University, Hanoi, Vietnam)  
  Email: `hieundhe180318@fpt.edu.vn`
- **La Uyen Nhi** (Phuong Dong University, Hanoi, Vietnam)  
  Email: `alexander.launie@gmail.com`
- **Tran Tien Long** (FPT University, Hanoi, Vietnam)  
  Email: `longtthe176743@fpt.edu.vn`
- **Do Ngoc Bich** (FPT University, Hanoi, Vietnam)  
  Email: `tinhthanh719@gmail.com`

## Abstract

Structural polarization is commonly used as a proxy for echo chambers in social networks; however, densely clustered interaction graphs do not necessarily imply restricted information exposure. Existing metrics primarily emphasize community cohesion and separation, largely overlooking exposure dynamics and temporal reinforcement. This work introduces **DEQ-EC**, a dynamic exposure-aware framework that integrates structural polarization with user-level exposure diversity across time.  

DEQ-EC leverages temporal graph representations and exposure entropy derived from interaction-driven content distributions to produce a unified measure capable of distinguishing transient controversy from sustained echo chamber formation. Unlike prior approaches, the framework does not assume fixed ideological partitions and operates without labeled users while capturing evolving polarization patterns.  

Evaluation on a benchmark echo chamber dataset, a large-scale U.S. election Twitter corpus, and polarized Reddit communities demonstrates that DEQ-EC consistently improves discrimination between polarized and non-polarized environments and identifies event-driven polarization spikes that structural metrics alone fail to detect.


## Overview

This repository provides the implementation of:

- Static embedding-based echo chamber scoring (ECS baseline).
- Dynamic DEQ-EC modeling with temporal regularization.
- Exposure-aware modulation of structural separation/cohesion.
- Notebook-based experiments for quantification and ideology analysis.

The current codebase includes:

- `src/DEQ.py`: static GAE training and dynamic temporal training (`run_dynamic`).
- `src/EchoDEQ.py`: wrappers for static (`EchoGAE_algorithm`) and dynamic (`EchoDEQ_algorithm`) embedding learning.
- `src/echo_chamber_measure.py`: ECS metric and dynamic DEQ-EC metric (`DEQECMeasure`).
- `src/load_data.py`: graph, text embedding, and community preparation utilities.

## Method Summary

Given temporal interaction snapshots \(G^{(\tau)}\):

1. Learn user embeddings with graph autoencoding per snapshot.
2. Add temporal smoothness regularization across adjacent windows.
3. Compute structural terms:
   - cohesion \(c_u^{(\tau)}\)
   - separation \(\Delta_u^{(\tau)}\)
4. Compute exposure ratio \(\phi_u^{(\tau)}\) from cross-community outgoing interactions.
5. Compute:
   - \(s_u^{(\tau)} = \frac{\Delta_u^{(\tau)} - c_u^{(\tau)}}{\max(\Delta_u^{(\tau)}, c_u^{(\tau)})}\)
   - \(\tilde{s}_u^{(\tau)} = s_u^{(\tau)}(1-\phi_u^{(\tau)})\)
6. Aggregate to snapshot and full-horizon DEQ-EC scores.

## Project Structure

```text
DEQ-EC/
  data/                         # datasets (Twitter/Reddit/benchmark splits)
  src/
    DEQ.py
    EchoDEQ.py
    EchoGAE.py
    echo_chamber_measure.py
    load_data.py
    tweet_preprocessing.py
    baselines/
  exp_01_deqec_quantification.ipynb
  expt_2_ideology_detection.ipynb
  environment.yml
```

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate DEQ-EC
```

If your environment name differs, check the `name:` field in `environment.yml`.

## Quick Start

1. Prepare datasets under `data/` (same folder structure expected by `src/load_data.py`).
2. Run the quantification notebook:
   - `exp_01_deqec_quantification.ipynb`
3. Run ideology analysis notebook:
   - `expt_2_ideology_detection.ipynb`

## Reproducibility Notes

- Global seeds are set in notebooks for Python, NumPy, and PyTorch.
- GPU is used when available (`cuda`), otherwise CPU.
- Main dynamic training hyperparameters:
  - embedding dimension: `out_channels`
  - hidden dimension: `hidden_channels`
  - temporal regularization: `gamma`
  - optimizer: Adam

## Citation

If you use this codebase, please cite:

```bibtex
@article{deqec2026,
  title   = {DEQ-EC: Dynamic Exposure-Aware Quantification of Echo Chambers in Social Networks},
  author  = {Nguyen Dinh Hieu and La Uyen Nhi and Tran Tien Long and Do Ngoc Bich},
  year    = {2026},
  note    = {Manuscript in preparation / under submission}
}
```

## Contact

For questions, collaboration, or data/implementation details, please contact:

- `hieundhe180318@fpt.edu.vn`

