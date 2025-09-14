# inference_matrix.py
import torch
import os
from pathlib import Path

# ------------------------------
# 1️⃣ 定义模型类
# 你需要根据实际 DeepSeek 模型结构修改这里
# 这里给出一个示例 Transformer forward 网络
# ------------------------------
class DeepSeekMatrixModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=1536, output_dim=1, num_layers=30):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.ReLU(),
            *[torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()) for _ in range(num_layers-1)],
            torch.nn.Linear(hidden_size, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

# ------------------------------
# 2️⃣ 封装推理接口
# ------------------------------
class DeepSeekInfer:
    def __init__(self, device: str = "cuda"):
        # 固定模型权重路径
        model_path = "/home/liyakun/twitter-stock-prediction/models/model2/best_model.pt"
        lora_path = "/home/liyakun/twitter-stock-prediction/models/model2/lora_weights.pt"  # 如果没有可设置为 None

        # 自动选择设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ⚠️ 根据你的特征维度修改
        example_input_dim = 151
        self.model = DeepSeekMatrixModel(input_dim=example_input_dim).to(self.device)

        # 1️⃣ 加载基础模型权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"基础模型权重不存在: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"✅ 基础模型权重已加载: {model_path}")

        # 2️⃣ 加载 LoRA 权重（如果存在）
        if lora_path is not None and os.path.exists(lora_path):
            lora_checkpoint = torch.load(lora_path, map_location=self.device)
            lora_state_dict = self._extract_state_dict(lora_checkpoint)
            self.model.load_state_dict(lora_state_dict, strict=False)
            print(f"✅ LoRA 权重已叠加: {lora_path}")
        else:
            print("⚠️ 未找到 LoRA 权重，跳过叠加")

        self.model.eval()
        print(f"✅ 模型已加载到 {self.device}")

    def _extract_state_dict(self, checkpoint):
        """统一提取 state_dict 并去掉多 GPU 前缀"""
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        # 去掉多 GPU 前缀
        return {k.replace("module.", ""): v for k, v in state_dict.items()}

    @torch.no_grad()
    def predict(self, prompt: str):
        # 这里根据你原来的逻辑处理输出
        # 目前简单返回模型前向结果
        dummy_input = torch.randn(1, 1, 151).to(self.device)
        output = self.model(dummy_input)
        return output

# ------------------------------
# 3️⃣ 测试脚本（可直接运行）
# ------------------------------
if __name__ == "__main__":
    # 固定模型权重路径
    model_path = "/home/liyakun/twitter-stock-prediction/models/model2/best_model.pt"
    
    # 初始化推理器
    infer = DeepSeekInfer(model_path=model_path)
    
    # 模拟输入矩阵
    batch_size = 2
    seq_len = 30
    feature_dim = 151  # ⚠️ 这里要和模型实际特征维度一致
    dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    
    # 推理
    output = infer.predict(dummy_input)
    print("输出结果 shape:", output.shape)
    print("输出示例:", output)
