pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda install -c nvidia cuda-nvcc cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
pip install safetensors transformers tqdm xxhash numpy
pip install flash-attn-3 --extra-index-url https://download.pytorch.org/whl/cu126