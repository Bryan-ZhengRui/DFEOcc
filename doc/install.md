## Environment Setup
step 1. Install environment for pytorch training (python=3.8.5, cuda=11.7)
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.7.0
pip install mmdet==2.27.0
pip install mmsegmentation==0.30.0
( If pip installation fails, it is recommended to install via mim:
pip install openmim
mim install mmcv-full==1.7.0
mim install mmdet==2.27.0
mim install mmsegmentation==0.30.0 )

sudo apt-get install python3-dev
sudo apt-get install libevent-dev
sudo apt-get install build-essential
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda
pip install pycuda

pip install lyft-dataset-sdk==0.0.8
pip install networkx==2.2
pip install numba==0.53.0
pip install numpy==1.23.5
pip install nuscenes-devkit==1.1.11
pip install plyfile
pip install scikit-image==0.19.3
pip install tensorboard==2.14.0
pip install trimesh==2.35.39
pip install setuptools==60.2.0
pip install yapf==0.40.1

cd Path_to_DFEOcc
git clone https://github.com/open-mmlab/mmdetection3d.git

cd Path_to_DFEOcc/mmdetection3d
git checkout v1.0.0rc4
pip install -v -e . 

cd Path_to_DFEOcc/projects
pip install -v -e . 
```
step 2. Install VMamba dependencies:
```
pip install packaging==24.2
pip install triton==3.1.0
pip install timm==0.4.12
pip install pytest==8.3.3
pip install yacs==0.1.8
pip install termcolor==2.4.0
pip install fvcore

cd tools/selective_scan && pip install .
```

