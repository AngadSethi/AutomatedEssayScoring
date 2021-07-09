import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, num_models=2):
        super(Ensemble, self).__init__()

        self.projection = nn.Linear(
            in_features=num_models,
            out_features=1,
            bias=True
        )

        self.activation = nn.Sigmoid()

    def forward(self, essay_ids, essay_sets, x, y, scores, min_scores, max_scores, z=None):

        outputs = torch.stack([x, y], dim=-1)
        if z is not None:
            outputs = torch.stack([x, y, z], dim=-1)

        outputs = self.activation(self.projection(outputs))

        outputs = torch.squeeze(outputs, dim=-1)

        return outputs
