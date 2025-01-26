# EffoNAV
EffoNAV: an Effective Foundation-model-based Visual Navigation Approach for Challenging Environment.

# Setup
```
conda env create -f train/train_environment.yml
conda activate vint_train
pip install -e train/
```
# Data Preparation 
In this paper, we train on 4 publicly available datasets:
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing)
- [SACSoN](https://sites.google.com/view/sacson-review/huron-dataset)
You can use some sample scripts to process these datasets, either directly from a rosbag or from a custom format like HDF5s:
1. Run `process_bags.py` with the relevant args, or `process_recon.py` for processing RECON HDF5s.
2. Run `data_split.py` on your dataset folder with the relevant args.
