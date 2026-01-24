import torch
from .base import BaseDataPreprocessor


class MinMaxNormalization(BaseDataPreprocessor):
    
    def __init__(self):
        self.min_val = None
        self.max_val = None
    
    def fit(self, data):
        self.min_val = data.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        self.max_val = data.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        return self
    
    def transform(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Call fit() before transform()")
        return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
    
    def inverse_transform(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Call fit() before inverse_transform()")
        return data * (self.max_val - self.min_val) + self.min_val
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class StandardNormalization(BaseDataPreprocessor):
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()
        return self
    
    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Call fit() before transform()")
        return (data - self.mean) / (self.std + 1e-8)
    
    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Call fit() before inverse_transform()")
        return data * self.std + self.mean
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
