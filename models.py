"""Top-level model classes.

Author:
    Angad Sethi
"""

import torch.nn as nn

import layers


class Ensemble(nn.Module):
    def __init__(self, model, max_seq_length: int, total_seq_length: int, bert=None, bidaf=None):
        super(Ensemble, self).__init__()
        self.bert = bert
        self.bidaf = bidaf
        self.out = layers.ModelOutput(model, max_seq_length, total_seq_length)
        self.model = model

    def forward(self, essays_bert, essays_bidaf, prompts, masks=None):
        output_1, output_2 = None, None
        if self.model == 'bert':
            output_1 = self.bert(essays_bert, masks)
        elif self.model == 'bidaf':
            output_1 = self.bidaf(essays_bidaf, prompts)
        else:
            output_1 = self.bert(essays_bert, masks)
            output_2 = self.bidaf(essays_bidaf, prompts)
        return self.out(output_1, output_2)
