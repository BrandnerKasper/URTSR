import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, scale: int, down_and_up: int = 0):
        super(BaseModel, self).__init__()
        self.scale = scale
        self.down_and_up = down_and_up
