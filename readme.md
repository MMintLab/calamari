# CALAMARI

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![CodeQL](https://github.com/MMintLab/VIRDO/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/MMintLab/VIRDO/actions/workflows/codeql-analysis.yml)


This is a github repository of a [CALAMARI: Contact-Aware and Language conditioned spatial Action MApping for contact-RIch manipulation](https://proceedings.mlr.press/v229/wi23a.html) (CoRL 2023).


We trained with the GPU A6000 and ran inference on the RTX 3080 and RTX 2070.

## 1. install project and Dependencies
```angular2html
conda create -n calamari python=3.8
conda activate calamari
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda env create -f environment.yml
```
We utilize heatmap extraction from Semantic Abstraction (Huy et al., CoRL 2022)."
```angular2html
git submodule add -f git@github.com:yswi/semantic-abstraction.git calamari/semantic_abstraction
git submodule add -b ros -f git@github.com:UM-ARM-Lab/pytorch_mppi.git calamari/pytorch_mppi

```

## 2. Install Project
```
pip install -e .
```

## 3. Download Dataset & Pretrained
1. Download the dataset.zip from the [link](https://www.dropbox.com/scl/fo/6w3p35agbu89ojp1mux5t/h?rlkey=0dxqegorjzo45tlzzy06y0w2z&dl=0
). Unzip the folder under ``dataset/`` folder.

2. Download the pretrained .pth from the [link](https://www.dropbox.com/scl/fo/h53s8s108959q30vjmecb/h?rlkey=ubdmt9yumle313g4owjra1epe&dl=0
). Put them under ``script/model/`` folder.

3. As a result, the directory should be

```
── calamari
│   ├── calamari
│   ├── dataset
│   │   │── wipe_desk
│   │   │── sweep_to_dustpan
│   │   │── push_buttons
│   │   │── ...
│   ├── script
│   ├── ├── model
...
```


## 4. (optionally) Train Policy from Scratch

```
python script/train.py --task <TASK NAME> --logdir <FOLDER NAME> --gpu_id <GPU IDX>
```
Note: We use A6000 (48G) for training. You can decrease the batch size in config_multi_conv.py to match your GPU capacity, but a performance drop should be expected.



## 5. Inference
```commandline
python script/plan/mpc.py --task <task name> --txt_idx <txt idx> --ttm_idx <ttm idx> -v <task variation idx>  -s 0 --logdir <log dir>
```
Below are the combinations of parameters we used for the paper. You can find the inference code from 
<table>
    <thead>
        <tr>
            <th>task</th>
            <th>object</th>
            <th>ttm idx</th>
            <th>task variation idx</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>wipe</td>
            <td>train obj</td>
            <td>0</td>
            <td>0</td>
        </tr>
        <tr>
            <td>test obj1</td>
            <td>1</td>
            <td>0</td>
        </tr>
        <tr>
            <td>test obj2</td>
            <td>2</td>
            <td>0</td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <td rowspan=3>sweep</td>
            <td>train obj</td>
            <td>0</td>
            <td>0</td>
        </tr>
        <tr>
            <td>test obj1</td>
            <td>1</td>
            <td>0</td>
        </tr>
        <tr>
            <td>test obj2</td>
            <td>2</td>
            <td>0</td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <td rowspan=3>push</td>
            <td>train obj</td>
            <td>0</td>
            <td>0</td>
        </tr>
        <tr>
            <td>test obj1</td>
            <td>0</td>
            <td>1</td>
        </tr>
        <tr>
            <td>test obj2</td>
            <td>0</td>
            <td>2</td>
        </tr>
    </tbody>
</table>

## (optionally) Train with Custom Data. 
Generate heatmaps of the custom data.
```
 python script/dataprocessing/generate_heatmap.py --task <TASK>
```

## Notes
This repository trains the policy based on the RLbench dataset.
Please reach out to the author yswi@umich.edu for further questions.