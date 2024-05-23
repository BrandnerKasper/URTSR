import torch
import torch.nn as nn

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


class ExtraSS(BaseModel):
    def __init__(self, scale: int):
        super(ExtraSS, self).__init__(scale=scale)
        pass

    def forward(self, x):
        pass
