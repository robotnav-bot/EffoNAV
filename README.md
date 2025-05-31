# EffoNAV
EffoNAV: an Effective Foundation-model-based Visual Navigation Approach for Challenging Environment.
![image](https://github.com/robotnav-bot/EffoNAV/blob/main/showgif.gif)

EffoNAV can realize navigation under challenging situations such as large lighting variations and object variations.

## Setup
```
conda env create -f train/train_environment.yml
conda activate EffoNAV
pip install -e train/
```
## Data Preparation 
In this paper, we train on 4 publicly available datasets:
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing)
- [SACSoN](https://sites.google.com/view/sacson-review/huron-dataset)
  
You can use some sample scripts to process these datasets, either directly from a rosbag or from a custom format like HDF5s:
1. Run `process_bags.py` with the relevant args, or `process_recon.py` for processing RECON HDF5s.
2. Run `data_split.py` on your dataset folder with the relevant args.

## Training
Run this inside the `./train` directory:
```
python train.py -c config/EffoNAV.yaml
```
## Deployment
For the deployment procedures and details, please refer to [Deployment](https://github.com/robodhruv/visualnav-transformer/tree/main?tab=readme-ov-file#deployment)

## Acknowledgement
Our work references the training and deployment methods of [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer). We are grateful for their contributions to the field of robot navigation.

## Citing
```
@article{shen2025effonav,
  title={EffoNAV: An Effective Foundation-Model-Based Visual Navigation Approach in Challenging Environment},
  author={Shen, Wangtian and Gu, Pengfei and Qin, Haijian and Meng, Ziyang},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
