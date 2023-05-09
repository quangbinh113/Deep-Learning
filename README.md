# Single-frame Infrared Small Target (SIRST) Detection
Infrared small target detection has been a challenging task due to the weak radiation
intensity of targets and the complexity of the background. Traditional methods using hand-
designed features are usually effective for specific background but pose some problems in other
complex infrared scenes. Our work proposes some deep learning based approaches on single-
frame infrared small target (SIRST) detection in order to exploit the unexpected methods that
potentially lead to more adaptive and accurate solutions. Distinct artificial neural networks
are trained through thousands of infrared images in order to obtain the patterns of desired
tiny, disrupted targets and then suppress the other non-target regions. Extensive experiments
demonstrate that the proposed methods potentially handle effectively the variety and difficulty
of this problem, compared to common fixed algorithms, in terms of visual and quantitative
evaluation metrics.

<p align="center">
  <img src="https://user-images.githubusercontent.com/86721208/211051273-7f70a44f-e45d-40b9-8887-e567b086627d.png" />
</p>

## Deep Learning - DSAI K65: Group 16
1. Nguyễn Tống Minh (Email: minh.nt204885@sis.hust.edu.vn)
2. Hoàng Trần Nhật Minh (Email: minh.ht204883@sis.hust.edu.vn)
3. Hồ Minh Khôi (Email: khoi.hm204917@sis.hust.edu.vn)
4. Nguyễn Hoàng Phúc (Email: phuc.nh204923@sis.hust.edu.vn)
5. Trương Quang Bình (Email: binh.tq200068@sis.hust.edu.vn)

## Project Structure

```
datasets/               # datasets & torch data modules
logs/                   # checkpoints & training logs
models/                 # models
notebooks/              # execution show-off
report/                 # documents & slides
trainers/               # train-test runners
README.md
```
---

# Setup

Our project (notebooks and execution) is carried on Kaggle (Linux) with backends are the modules from this repository. Therefore, rerun is highly recommended to be on Kaggle with GPU P100 with the dataset [SIRST](https://www.kaggle.com/datasets/minhngt02/nudtsirst) (~32GB) and this repository attached to input and output folder, respectively.

## Requirements
- Quick installation on local environment (Anaconda required):
```
  # install all dependencies
  conda env create -f env.yml
  
  # activate conda env
  conda activate dl_sirst
```

- Installation on Kaggle environment:
```
  # install pycocotools for evaluation
  pip install pycocotools
  
  # install torchinfo for debug
  pip install torchinfo
  pip install torch-summary # old version of torchinfo
  
  # MAY install: segmentation libraries
  pip install segmentation_models_pytorch
```
