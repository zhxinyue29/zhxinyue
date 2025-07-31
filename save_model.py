import torch
from transformers import AutoModel, AutoTokenizer

# 模型目录路径
model_directory = "/home/liyakun/LLaMA-Factory-main/deepseek1.5B"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# 加载模型
model = AutoModel.from_pretrained(model_directory, trust_remote_code=True)

# 保存模型参数为 .pt 文件
save_path = "/home/liyakun/twitter-stock-prediction/models/model1/best_model.pt"
torch.save(model.state_dict(), save_path)

print(f"模型参数已保存为 {save_path}")