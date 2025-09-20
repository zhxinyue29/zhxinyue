#!/bin/bash
# 激活虚拟环境
__conda_setup="$('/home/liyakun/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate llama_factory

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 设置环境变量
export MODEL_NAME=model2
export PYTHONPATH="/home/liyakun/twitter-stock-prediction:$PYTHONPATH"

# 运行 BM 实验
python -m BM.src.model2.train --config BM/configs/model2.yaml
