#!/bin/bash

# 激活虚拟环境（如果有）
__conda_setup="$('/home/liyakun/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate llama_factory

cd "$(dirname "$0")/.."


# 设置环境变量（如果需要）
export MODEL_NAME=model2

# 运行训练脚本
export PYTHONPATH="/home/liyakun/twitter-stock-prediction:$PYTHONPATH"
python -m src/model2/train.py  
