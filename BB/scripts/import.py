import os
from pathlib import Path

train_dir = '/home/liyakun/twitter-stock-prediction/data/splits/train/'

# 列出所有文件
files = list(Path(train_dir).iterdir())
print("目录文件：", [f.name for f in files])

# 检查是否有符合条件的x文件
for f in files:
    fname_lower = f.name.lower()
    print(f"检查文件：{f.name}，小写：{fname_lower}")
    if 'x' in fname_lower:
        print(f"找到匹配文件： {f.name}")