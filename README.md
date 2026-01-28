# compound-jamming-ftmlnet
# FT-MLNet (Full Open Source)

This repository provides the public implementation of **FT-MLNet** and the **dataset generation code** used in our manuscript:
“Few-Shot Compound Jamming Recognition via Multi-Task Feature Fusion and Feature Transfer”.

## 1) Files
- `matlab_generate_dataset.m`: MATLAB script to generate the dataset (time–frequency images + labels, and optional statistics/prior vectors).
- `train_eval_ftmlnet.py`: Python script for training and evaluation of FT-MLNet.
- `requirements.txt`: Python dependencies.

## 2) Environment
- Python: 3.11+
- TensorFlow: 2.10 (or compatible)
- GPU recommended (tested on NVIDIA RTX 4060)

Install:
```bash
pip install -r requirements.txt
