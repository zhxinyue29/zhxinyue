import os
import gym
import pandas as pd
from stable_baselines3 import PPO
from ..utils import logger
from ..utils.logger import setup_logger
from ..utils.config import load_config
from ..model1.env import StockTradingEnv

def save_predictions(predictions, output_path):
    """Save predictions to a CSV file."""
    df = pd.DataFrame(predictions, columns=['action'])
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # 设置日志记录
    logger = setup_logger(log_file='logs/model1_predict.log')
    
    # 加载配置
    config = load_config('configs/model1.yaml')
    
    # 创建交易环境
    logger.info("Creating environment...")
    env = StockTradingEnv(config['env'])
    
    # 加载PPO模型
    logger.info("Loading model...")
    model = PPO.load(config['model']['load_path'])
    
    # 执行预测
    logger.info("Predicting...")
    obs = env.reset()
    predictions = []
    for _ in range(config['predict']['steps']):
        action, _states = model.predict(obs)
        predictions.append(action)  # 收集预测动作
        obs, rewards, done, info = env.step(action)
        if done:
            obs = env.reset()
    
    # 保存预测结果
    output_path = config['predict'].get('output_path', 'predictions.csv')
    save_predictions(predictions, output_path)