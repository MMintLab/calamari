
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

## 2. Install project
```
pip install -e .
```


## 3. (optionally) Generate Heatmap
```
 python script/dataprocessing/generate_heatmap.py
```

## 4. Train Policy
```
# temporal transformer
python script/train_temporal.py --gpu_id 0
```
