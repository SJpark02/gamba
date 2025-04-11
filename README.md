```bash
docker run -it -d -v /data/3dr/gamba:/workspace --gpus all --shm-size=10G --name=gamba nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

docker exec -it gamba bash

apt update && apt install ubuntu-drivers-common && apt install python3-pip && apt install git wget nano
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

#bashrc 수정 (alias)

pip install packaging
pip install causal-conv1d==1.2.0.post2 mamba-ssm==1.2.0 timm jaxtyping omegaconf
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118/xformers-0.0.22.post4%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl#sha256=acedcab631eecf9d943497726d036762f5a94e45a95abbd66a4b63d397354fe9

cd workspace
git clone https://github.com/SkyworkAI/Gamba.git
cd Gamba
rm -rf submodules/*

cd submodules

git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
git clone --recursive https://github.com/florinshen/rad-polygon-mask.git

cd ..

pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/rad-polygon-mask

pip install git+https://github.com/NVlabs/nvdiffrast

pip install -r requirements.txt
pip install numpy==1.24.1

mkdir checkpoint && cd checkpoint
wget https://huggingface.co/florinshen/Gamba/resolve/main/gamba_ep399.pth
cd ..



python gamba_infer.py --model-type gamba --resume ./checkpoint/gamba_ep399.pth --workspace workspace_test --test_path ./data_test


export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
