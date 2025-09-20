import pandas as pd

class Dataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def get_features_and_labels(self):
        X = self.data.drop(columns=['target'])
        y = self.data['target']
        return X, y
