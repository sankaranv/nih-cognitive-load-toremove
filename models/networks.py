from nn import *
import torch.nn as nn


class OneLayerLinear(NNModel):
    def __init__(self, seq_len, pred_len, num_actors, num_features):
        super().__init__(seq_len, pred_len, num_actors, num_features)

    def create_model(self, seq_len, pred_len, num_actors, num_features):
        return nn.Sequential(
            nn.Linear(seq_len * num_features * num_actors, pred_len * num_actors),
        )
    

    

