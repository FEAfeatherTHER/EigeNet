pip install setuptools ruamel.yaml tqdm pyworld -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install librosa==0.11.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install torch==2.8.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install torchaudio==2.8.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install torchdiffeq==0.2.5 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install einops==0.8.1 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install matplotlib==3.10.1 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install torchcfm==1.0.7 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install bigvgan -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install audidata==0.0.5 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install accelerate==1.10.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install h5py==3.14.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install stable-audio-tools==0.0.19 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install json5 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install transformers===4.53.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

conda install conda-forge::pretty_midi
pip install pretty_midi
pip install --force-reinstall --no-cache-dir numpy pandas

# flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# acoustic degradation
pip install pedalboard resampy pyroomacoustics datasets -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
conda install -c conda-forge libstdcxx-ng -y
pip install trl==0.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple