import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

"""
Baseline Model (TODO)

Fully Connected Neural Network for Speech Denoising
"""
class BaselineModel(BaseModel):
    def __init__(self):
        super(BaselineModel, self).__init__()
        

    def forward(self, x):
        return 0