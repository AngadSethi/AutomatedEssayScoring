"""Top-level model classes.

Author:
    Angad Sethi
"""
import torch
import torch.nn as nn
from transformers import BertModel

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


class OriginalModel(nn.Module):
    def __init__(self, hidden_size: int, model_checkpoint: str = 'bert-base-uncased', freeze: bool = True):
        super(OriginalModel, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(model_checkpoint)
        for param in self.bert_encoder.parameters():
            param.requires_grad = not freeze
        self.gru_encoder = nn.GRU(input_size=self.bert_encoder.config.hidden_size, hidden_size=hidden_size,
                                  batch_first=True)
        self.layer = nn.Linear(2 * hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, essay_ids: torch.LongTensor, essay_sets: torch.LongTensor, x: torch.LongTensor,
                masks: torch.BoolTensor, scores: torch.FloatTensor, min_scores: torch.LongTensor,
                max_scores: torch.LongTensor):
        output = self.bert_encoder(x).last_hidden_state
        _, output = self.gru_encoder(output)
        output = output.permute(1, 0, 2)
        output = torch.flatten(output, start_dim=1)
        output = self.layer(output)
        output = self.activation(output)
        return torch.squeeze(output, -1)
