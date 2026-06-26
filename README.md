# MSP-GAD
We provide the code (in pytorch) and datasets for our paper: 

**Learning Transferable Structure-Aware Normality for Fully Unsupervised Cross-Domain Graph Anomaly Detection**

## Description


## Package Dependencies

- python 3.8.18
- pytorch 1.13.1+cu117
- cuda 11.7

## Running experiments
### Data Preparation
Download the datasets at 'https://drive.google.com/drive/folders/1SSWgFRdth3U44_IMRnW775B1l-bjQATW'

### Node GAD
To train and evaluate MSP-GAD on various domains, you may need to change the corresponding parameters in *main.py*.

Pre-training and adaptation:
```bash
 python main.py --force-reload --self-loop --dynamic-center --edition 111111111
```

