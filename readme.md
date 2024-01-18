
We trained this code with GPU A6000, and ran inference on RTX 3080 and RTX 2070.

## 1. Dependencies
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


## (optionally) Train with Custom Data. 
Generate heatmap of the RLBench example.
```
 python script/dataprocessing/generate_heatmap.py --tas <TASK>
```