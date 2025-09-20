from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_mse(y_true, y_pred):
    """
    计算均方误差（MSE）。
    
    :param y_true: 真实值列表。
    :param y_pred: 预测值列表。
    :return: MSE值。
    """
    return mean_squared_error(y_true, y_pred)

def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差（MAE）。
    
    :param y_true: 真实值列表。
    :param y_pred: 预测值列表。
    :return: MAE值。
    """
    return mean_absolute_error(y_true, y_pred)
