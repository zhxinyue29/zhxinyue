import os
import json
import yaml
import torch
from collections import defaultdict
from transformers import AutoConfig

# 配置您的模型目录路径
MODEL_DIRECTORY = "/home/liyakun/LLaMA-Factory-main/deepseek1.5B"

def extract_model_parameters(model_dir):
    """从Hugging Face格式的模型目录中提取参数配置"""
    try:
        # 结果字典
        parameters = {
            'model_directory': os.path.abspath(model_dir),
            'parameters': {}
        }
        
        # 尝试加载配置文件
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            print(f"找到配置文件: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 提取基本配置
            parameters['parameters'] = config
            return parameters
        
        # 如果没有配置文件，尝试加载模型状态
        model_files = [
            f for f in os.listdir(model_dir) 
            if f.endswith(('.bin', '.pt', '.pth'))
        ]
        
        if model_files:
            # 尝试加载第一个模型文件
            model_path = os.path.join(model_dir, model_files[0])
            print(f"从模型文件提取参数: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 从状态字典提取参数
            parameters['parameters'] = extract_parameters_from_state_dict(state_dict)
            return parameters
        
        # 如果都没有，尝试使用transformers库
        try:
            print("尝试使用transformers.AutoConfig加载配置...")
            config = AutoConfig.from_pretrained(model_dir)
            parameters['parameters'] = config.to_dict()
            return parameters
        except:
            pass
        
        # 所有方法都失败
        raise ValueError("无法提取模型参数")
    
    except Exception as e:
        return {
            'error': f"提取参数失败: {str(e)}",
            'model_directory': model_dir
        }

def extract_parameters_from_state_dict(state_dict):
    """从状态字典中推断参数"""
    params = {}
    
    # 参数统计
    total_params = 0
    layer_types = defaultdict(int)
    
    # 分析所有参数
    for name, tensor in state_dict.items():
        total_params += tensor.numel()
        layer_type = name.split('.')[0]
        layer_types[layer_type] += tensor.numel()
    
    # 添加参数统计
    params['total_parameters'] = total_params
    params['parameter_distribution'] = dict(layer_types)
    
    # 尝试识别关键参数
    for name, tensor in state_dict.items():
        # 嵌入层
        if 'embed' in name and len(tensor.shape) == 2:
            params['vocab_size'] = tensor.shape[0]
            params['hidden_size'] = tensor.shape[1]
        
        # Transformer层数
        if 'layer' in name and 'weight' in name:
            # 尝试提取层索引
            parts = name.split('.')
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    if 'num_layers' not in params or layer_idx > params['num_layers']:
                        params['num_layers'] = layer_idx + 1
        
        # 注意力头
        if 'attention' in name and 'weight' in name and len(tensor.shape) == 2:
            params['num_attention_heads'] = tensor.shape[0] // 64  # 假设每个头64维
        
        # 隐藏层大小
        if 'dense' in name and 'weight' in name and len(tensor.shape) == 2:
            params['hidden_size'] = tensor.shape[0]  # 全连接层输出维度
        
        # 中间层大小
        if 'intermediate' in name and 'weight' in name and len(tensor.shape) == 2:
            params['intermediate_size'] = tensor.shape[0]
    
    return params

def save_parameters(parameters, output_path):
    """保存参数到文件"""
    try:
        with open(output_path, 'w') as f:
            if output_path.endswith('.json'):
                json.dump(parameters, f, indent=2)
            else:
                yaml.dump(parameters, f, indent=2)
        print(f"参数已保存到: {os.path.abspath(output_path)}")
        return True
    except Exception as e:
        print(f"保存失败: {str(e)}")
        return False

def print_parameters(parameters):
    """打印提取的参数"""
    if 'error' in parameters:
        print(f"错误: {parameters['error']}")
        return
    
    print("\n" + "="*70)
    print(f"模型目录: {parameters.get('model_directory', '未知路径')}")
    print("="*70)
    
    if 'parameters' not in parameters or not parameters['parameters']:
        print("未提取到参数信息")
        return
    
    params = parameters['parameters']
    
    # 打印基本信息
    print("\n[基本配置]")
    keys_to_print = [
        'vocab_size', 'hidden_size', 'num_hidden_layers', 
        'num_attention_heads', 'intermediate_size', 'max_position_embeddings',
        'model_type', 'architectures', 'torch_dtype'
    ]
    
    for key in keys_to_print:
        if key in params:
            print(f"{key}: {params[key]}")
    
    # 打印自定义提取的参数
    custom_keys = [
        'num_layers', 'action_space_size', 'input_dim', 'output_dim'
    ]
    for key in custom_keys:
        if key in params:
            print(f"{key}: {params[key]}")
    
    # 打印参数统计
    if 'total_parameters' in params:
        print(f"\n总参数量: {params['total_parameters']:,}")
    
    # 打印参数分布
    if 'parameter_distribution' in params:
        print("\n[参数分布]")
        total = params.get('total_parameters', 0)
        for layer, count in params['parameter_distribution'].items():
            if total > 0:
                percent = count / total
                print(f"{layer}: {count:,} 参数 ({percent:.1%})")
            else:
                print(f"{layer}: {count:,} 参数")
    
    # 打印其他参数
    print("\n[其他配置参数]")
    printed_keys = set(keys_to_print + custom_keys + ['total_parameters', 'parameter_distribution'])
    for key, value in params.items():
        if key not in printed_keys:
            print(f"{key}: {value}")

def get_model_size(directory):
    """计算模型目录总大小"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return f"{total_size / (1024**3):.2f} GB"

if __name__ == "__main__":
    print(f"开始提取模型参数: {MODEL_DIRECTORY}")
    
    # 计算模型大小
    model_size = get_model_size(MODEL_DIRECTORY)
    print(f"模型大小: {model_size}")
    
    # 提取参数
    parameters = extract_model_parameters(MODEL_DIRECTORY)
    
    # 添加模型大小信息
    if 'error' not in parameters:
        parameters['model_size'] = model_size
    
    # 打印结果
    print_parameters(parameters)
    
    # 保存到文件
    if 'error' not in parameters:
        output_file = os.path.join(MODEL_DIRECTORY, "model_parameters.yaml")
        save_parameters(parameters, output_file)
        
        # 同时保存JSON版本
        json_file = os.path.join(MODEL_DIRECTORY, "model_parameters.json")
        save_parameters(parameters, json_file)