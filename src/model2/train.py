import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.model1.inference import load_model as load_model1, predict as predict_model1

def load_data(file_path, model1_output):
    """加载并拆分数据集，将模型1的输出作为额外特征"""
    data = pd.read_csv(file_path)
    data['model1_output'] = model1_output
    X = data.drop(columns=['target'])
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    logger = setup_logger(log_file='logs/model2_train.log')
    config = load_config('configs/model2.yaml')
    
    logger.info("Loading model1 for inference...")
    model1 = load_model1(config['data']['model1_path'])
    
    logger.info("Loading data...")
    # 使用模型1进行推理，得到输出作为特征
    model1_output = predict_model1(model1, X)
    X_train, X_test, y_train, y_test = load_data(config['data']['train_file'], model1_output)
    
    logger.info("Training model...")
    model2 = LinearRegression()
    model2.fit(X_train, y_train)
    
    logger.info("Evaluating model...")
    mse = mean_squared_error(y_test, model2.predict(X_test))
    logger.info(f"Model2 MSE: {mse}")
    
    model_path = config['paths']['output_model']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model2.save(model_path)
    logger.info(f"Model2 saved to {model_path}")