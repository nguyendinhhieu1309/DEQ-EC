# DEQ-EC: Dynamic Exposure-Aware Quantification of Echo Chambers in Social Networks

**Nguyen Dinh Hieu, La Uyen Nhi, Tran Tien Long, Do Ngoc Bich**  
FPT University, Hanoi, Vietnam; Phuong Dong University, Hanoi, Vietnam  
[hieundhe180318@fpt.edu.vn](mailto:hieundhe180318@fpt.edu.vn), [alexander.launie@gmail.com](mailto:alexander.launie@gmail.com), [longtthe176743@fpt.edu.vn](mailto:longtthe176743@fpt.edu.vn), [tinhthanh719@gmail.com](mailto:tinhthanh719@gmail.com)

---

This repository presents **DEQ-EC**, a dynamic exposure-aware framework for quantifying echo chambers in social networks. Unlike conventional structural metrics that mainly focus on graph clustering and community separation, DEQ-EC jointly models **structural polarization**, **user-level cross-community exposure**, and **temporal reinforcement**.  

By integrating temporally regularized graph representation learning with exposure-aware scoring, DEQ-EC distinguishes transient controversy from persistent information isolation. The framework is fully unsupervised (no ideology labels required during training) and is designed for scalable analysis on large social interaction graphs.

---

## Abstract

Structural polarization is commonly used as a proxy for echo chambers in social networks; however, densely clustered interaction graphs do not necessarily imply restricted information exposure. Existing metrics primarily emphasize community cohesion and separation, largely overlooking exposure dynamics and temporal reinforcement. This work introduces DEQ-EC, a dynamic exposure-aware framework that integrates structural polarization with user-level exposure diversity across time. DEQ-EC leverages temporal graph representations and exposure entropy derived from interaction-driven content distributions to produce a unified measure capable of distinguishing transient controversy from sustained echo chamber formation. Unlike prior approaches, the framework does not assume fixed ideological partitions and operates without labeled users while capturing evolving polarization patterns. Evaluation on benchmark and large-scale social media datasets demonstrates that DEQ-EC improves discrimination between polarized and non-polarized environments and identifies event-driven polarization spikes that structural metrics alone fail to detect.

**Keywords:** Echo Chambers, Polarization Measurement, Exposure Diversity, Dynamic Social Networks, Graph Representation Learning

---

## Pipeline

**DEQ-EC pipeline overview.** The framework includes:  
1) temporal snapshot construction from timestamped interactions,  
2) graph autoencoder-based representation learning,  
3) temporal smoothness regularization across adjacent windows,  
4) community-aware exposure modeling, and  
5) dynamic DEQ-EC scoring via structural separation/cohesion modulated by cross-community exposure.

![DEQ-EC Pipeline](<img width="721" height="249" alt="image" src="https://github.com/user-attachments/assets/67024d6b-97e1-457c-8646-319d57f8b158" />)

> Put your pipeline figure at `assets/deqec_pipeline.png` (you can export Figure 1 from your paper).

---

## ⚙️ Framework and Environment Setup

This project uses the following core libraries:

| Framework | Version |
|-----------|---------|
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-ee4c2c?logo=pytorch&logoColor=white) | 2.0.1 + cu118 |
| ![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3.1-0A66C2?logo=python&logoColor=white) | 2.3.1 |
| ![Python](https://img.shields.io/badge/Python-3.10-3776ab?logo=python&logoColor=white) | 3.10 |
| ![NetworkX](https://img.shields.io/badge/NetworkX-3.1-2C3E50?logo=python&logoColor=white) | 3.1 |
| ![NumPy](https://img.shields.io/badge/NumPy-1.23.5-013243?logo=numpy&logoColor=white) | 1.23.5 |
| ![Pandas](https://img.shields.io/badge/Pandas-2.0.3-150458?logo=pandas&logoColor=white) | 2.0.3 |
| ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?logo=scikitlearn&logoColor=white) | 1.3.0 |
| ![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.2.2-FFCA28?logo=huggingface&logoColor=black) | 2.2.2 |

---

## 🚀 Installation

```bash
# Clone repository
git clone <your-repo-url>
cd DEQ-EC

# Create environment from file
conda env create -f environment.yml
conda activate DEQ-EC
```

If your conda environment name is different, use the name defined in `environment.yml`.

---

## Dataset Preparation

Prepare datasets under `data/` with topic-specific folders (e.g., `gun`, `abortion`, `super_bowl`, `sxsw`) containing:

- `graph.gml`
- `tweets.feather`
- `allsides.feather`

Example layout:

```text
data/
  gun/
    graph.gml
    tweets.feather
    allsides.feather
  abortion/
  super_bowl/
  sxsw/
```

For dynamic experiments (e.g., election/reddit), prepare time-windowed snapshots in your preprocessing pipeline and feed them into `EchoDEQ_algorithm`.

---

## Running Experiments

### 1) Echo Chamber Quantification

Run:

- `exp_01_deqec_quantification.ipynb`

This notebook reports:
- ECS (embedding-only baseline)
- DEQ-EC (exposure-aware score)
- Optional baselines (RWC, Polarization Index)

### 2) Ideology Estimation / Representation Analysis

Run:

- `expt_2_ideology_detection.ipynb`

This notebook evaluates embedding quality for ideology-related downstream analysis.

---

## Code Structure

```text
src/
  DEQ.py                    # Static GAE + dynamic temporal training (run_dynamic)
  EchoDEQ.py                # Static and dynamic wrappers (EchoGAE_algorithm, EchoDEQ_algorithm)
  EchoGAE.py                # Compatibility export
  echo_chamber_measure.py   # ECS and DEQ-EC scoring
  load_data.py              # Data loading and preprocessing
  tweet_preprocessing.py
  baselines/
```

---

## Citation

If you use this repository, please cite:

```bibtex
@article{deqec2026,
  title   = {DEQ-EC: Dynamic Exposure-Aware Quantification of Echo Chambers in Social Networks},
  author  = {Nguyen, Dinh Hieu and La, Uyen Nhi and Tran, Tien Long and Do, Ngoc Bich},
  year    = {2026},
  note    = {Under submission}
}
```

---

## Contact

For questions or collaboration:

- `hieundhe180318@fpt.edu.vn`

