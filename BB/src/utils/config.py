import yaml

def load_config(config_path):
    """
    从YAML文件加载配置。
    
    :param config_path: 配置文件路径。
    :return: 配置字典。
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 用法示例
config = load_config('configs/model1.yaml')
print(config)

# 访问配置中的各个部分
try:
    # 访问强化学习环境配置
    initial_balance = config['env']['initial_balance']
    transaction_fee = config['env']['transaction_fee']
    max_holding_days = config['env']['max_holding_days']
    reward_type = config['env']['reward_type']
    observation_space = config['env']['observation_space']
    
    # 访问策略网络架构
    network = config['policy']['network']
    layers = config['policy']['layers']
    activation = config['policy']['activation']
    gamma = config['policy']['gamma']
    ent_coef = config['policy']['ent_coef']
    
    # 访问训练参数
    algorithm = config['training']['algorithm']
    total_timesteps = config['training']['total_timesteps']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    save_interval = config['training']['save_interval']
    
    # 访问数据路径
    raw_tweets_path = config['paths']['raw_tweets']
    processed_data_path = config['paths']['processed_data']
    output_model_path = config['paths']['output_model']
    
    # 在这里可以使用这些配置参数来设置你的强化学习环境和训练过程

except KeyError as e:
    print(f"配置文件中缺少键: {e}")