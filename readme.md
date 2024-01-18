# CALAMARI

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![CodeQL](https://github.com/MMintLab/VIRDO/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/MMintLab/VIRDO/actions/workflows/codeql-analysis.yml)


This is a github repository of a [CALAMARI: Contact-Aware and Language conditioned spatial Action MApping for contact-RIch manipulation](https://proceedings.mlr.press/v229/wi23a.html) (CoRL 2023).


We trained this code with GPU A6000, and ran inference on RTX 3080 and RTX 2070.

## 1. install project and Dependencies
```angular2html
conda create -n calamari python=3.8
conda activate calamari
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda env create -f environment.yml
```

```angular2html
git submodule add -f git@github.com:yswi/semantic-abstraction.git calamari/semantic_abstraction
```

## 2. Install Project
```
pip install -e .
```

## 3. Download Dataset
1. Download the dataset.zip from the link:
https://www.dropbox.com/scl/fo/6w3p35agbu89ojp1mux5t/h?rlkey=0dxqegorjzo45tlzzy06y0w2z&dl=0

2. Make 'dataset' folder and upzip the dataset.

```
── calamari
│   ├── calamari
│   ├── dataset
│   │   │── wipe
│   │   │── sweep
│   │   │── push
│   ├── script
...
```


## 4. Train Policy from Scratch
```
python script/train.py --task <TASK NAME> --logdir <FOLDER NAME> --gpu_id <GPU IDX>
```
Note: We use A6000 (48G) for training. You can decrease the batch size in config_multi_conv.py to match your GPU capacity, but a performance drop should be expected.


## (optionally) Train with Custom Data. 
Generate heatmap of the RLBench example.
```
 python script/dataprocessing/generate_heatmap.py --task <TASK>
```

## Notes
This repository trains the policy based on the RLbench dataset. RLbench code for inference and data collection will be released soon.