# [NeurIPS 2025]DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving
### [Paper](https://arxiv.org/abs/2412.09043)  

> DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving

## Demo

<div class="video-container">
  <div class="row">
    <img src="./assets/s0.gif" alt="demo">
  </div>
  <div class="row">
    <img src="./assets/s6.gif" alt="demo">
    <img src="./assets/s7.gif" alt="demo">
    <img src="./assets/s8.gif" alt="demo">
  </div>
  <div class="row">
    <img src="./assets/s1.gif" alt="demo">
    <img src="./assets/s4.gif" alt="demo">
    <img src="./assets/s5.gif" alt="demo">
  </div>
</div>


## Getting Started

### Environmental Setups

```bash
git clone https://github.com/EnVision-Research/DriveRecon.git --recursive
cd DriveRecon

conda create -n drivingrecon python=3.9 pip -y
conda activate drivingrecon

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-build-isolation

pip install -e submodules/diff-surfel-rasterization --no-build-isolation
pip install -e submodules/simple-knn --no-build-isolation
```

### OmniScene 对比实验

静态六相机实验协议及实现细节见 [OmniScene 数据集实验文档](docs/OmniScene%20数据集实验文档.md)。

```bash
python train_omniscene.py --config configs/omniscene/112x200.py --print-config
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/112x200.py --mode train
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/224x400.py --mode train

CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/112x200.py --mode test --test-split total --checkpoint final
CUDA_VISIBLE_DEVICES=0 python train_omniscene.py --config configs/omniscene/224x400.py --mode test --test-split total --checkpoint final
```


### Preparing Dataset
Follow detailed instructions in [Prepare Dataset](docs/prepare_data.md). 


### Training

#### Single machines
```
accelerate  launch --config_file ./acc.yaml \
train.py  --port 6017 --expname 'waymo' --configs 'arguments/nvs.py'
```


#### Multiple machines and multiple gps
```
accelerate launch --config_file ./acc_config.yaml \
--machine_rank $MLP_ROLE_INDEX --num_machines 3 --num_processes 24 \
--main_process_ip $MLP_WORKER_0_HOST --main_process_port $MLP_WORKER_0_PORT  \
train.py --port 6017 --expname 'waymo' \
--configs 'arguments/nvs.py'
```

## Evaling 
```
# python eval.py --checkpoint_path "./checkpoint_10000.pth" --port 6017 --expname 'waymo' --configs 'arguments/nvs.py'
```


## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{Lu2024DrivingRecon,
        title={DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving},
        author={Hao LU, Tianshuo XU, Wenzhao ZHENG, Yunpeng ZHANG, Wei ZHAN, Dalong DU, Masayoshi Tomizuka, Kurt Keutzer, Yingcong CHEN},
        journal={NeurIPS},
        year={2025}
      }
```

