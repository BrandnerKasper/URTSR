import torch
import torch.nn as nn

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel
    

class Stss(BaseModel):
    def __init__(self, scale: int, frame_number: int = 4):
        super(Stss, self).__init__(scale=scale, down_and_up=3)
        pass

    def forward(self, x):
        pass