import yaml
import torch
import torch.nn as nn

# 读取配置文件
def load_model(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            print("配置为空或格式有误")
            return None
        print("配置内容：", config)
        return config
    except Exception as e:
        print("读取配置失败：", e)
        return None

config_path = '/home/liyakun/twitter-stock-prediction/configs/model2.yaml'
config = load_model(config_path)
input_tensor = torch.randn(16, 30, 773)   # batch size 1，长度10
print(input_tensor.dtype)


# 定义预测函数（你已经定义过，可直接使用）
def predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output



if config:
    try:
        hidden_size = config['env']['hidden_size']
        num_layers = config['env']['num_layers']
        num_heads = config['env']['num_heads']
        input_dim=config['env']['input_dim']
        print("加载成功：", hidden_size, num_layers, num_heads)
    except KeyError as e:
        print("配置中缺少字段：", e)
else:
    print("配置未加载成功，请检查配置文件和路径。")


    if config:
        print("配置内容：", config)
        try:
            hidden_size = config['env']['hidden_size']
            print("hidden_size:", hidden_size)
        except KeyError as e:
            print("配置缺少关键字段：", e)
    else:
        print("无法加载配置，请检查配置文件。")
        # 处理读取失败的情况
    input_dim=config['env']['input_dim']
    hidden_size = config['env']['hidden_size']
    num_layers = config['env']['num_layers']
    num_heads = config['env']['num_heads']

class DeepSeekModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_heads):
        super().__init__()
        # 取消Embedding，直接用输入特征
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4),
            num_layers=num_layers
        )
        # 输入是连续特征，需线性转换到hidden_size
        self.input_fc = nn.Linear(input_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)  # 根据你的预测目标确定输出维度（比如一个连续值）

    def forward(self, x):
        # x形状: [batch_size, seq_len, input_dim]
        # 先映射到hidden_size
        print("x的形状：", x.shape)
        x = self.input_fc(x)
        print("经过线性变换后x的形状：", x.shape)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # 转换回 [batch_size, seq_len, hidden_size]
        x = x.permute(1, 0, 2)
        # 根据任务需要，可能加池化或特定位置输出
        # 这里假设取序列的第一个输出
        out = self.output_layer(x)
        out = out.squeeze(2).squeeze(1) 
        return out

model = DeepSeekModel(input_dim, hidden_size, num_layers, num_heads)
print('实例化模型类型：', type(model))
if config:
    try:
        state_dict = torch.load(config['env']['prediction_model_path'])
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print("加载模型参数失败：", e)
        model = None

    # 调用预测
    print("模型类型：", type(model))
    if model:
        output = predict(model, input_tensor)
        print("预测结果：", output)
    else:
        print("模型未正确加载，无法进行预测！")
