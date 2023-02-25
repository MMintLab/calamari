
## Dependencies
```angular2html
conda create -n lcc python=3.8
conda activate lcc
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

```
## Install project
```
pip install -e .
```

## Clone Package
```angular2html
git submodule add -f https://github.com/hila-chefer/Transformer-MM-Explainability.git Transformer_MM_Explainability
# git clone https://github.com/hila-chefer/Transformer-MM-Explainability.git
```



## Generetate Heatmap for training
```
 python script/dataprocessing/generate_heatmap_chefer.py
```

## Train
```
# temporal transformer
python script/train_temporal.py --gpu_id 0
```