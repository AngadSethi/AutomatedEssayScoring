import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, num_models: int = 2):
        super(Ensemble, self).__init__()

        self.projection = nn.Linear(
            in_features=num_models,
            out_features=1,
            bias=True
        )

        self.activation = nn.Sigmoid()

    def forward(self, essay_ids: torch.LongTensor, essay_sets: torch.LongTensor, x: torch.FloatTensor,
                y: torch.FloatTensor, scores: torch.FloatTensor, min_scores: torch.LongTensor,
                max_scores: torch.LongTensor, z: torch.FloatTensor = None) -> torch.Tensor:

        outputs = torch.stack([x, y], dim=-1)
        if z is not None:
            outputs = torch.stack([x, y, z], dim=-1)

        outputs = self.activation(self.projection(outputs))

        return outputs
