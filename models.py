"""Top-level model classes.

Author:
    Angad Sethi
"""
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AutoConfig, AutoModelWithHeads

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
        config = AutoConfig.from_pretrained(model_checkpoint, num_labels=1)
        self.bert_encoder = AutoModelWithHeads.from_pretrained(model_checkpoint, config=config)

        self.bert_encoder.add_adapter("aes")
        self.bert_encoder.add_classification_head(
            "aes",
            use_pooler=True,
            num_labels=1
        )
        self.bert_encoder.train_adapter("aes")
        self.bert_encoder.set_active_adapters("aes")

        # self.rnn_encoder = layers.RNNEncoder(
        #     input_size=self.bert_encoder.config.hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=2,
        #     drop_prob=0.2
        # )
        # self.gru_encoder = nn.GRU(input_size=self.bert_encoder.config.hidden_size, hidden_size=hidden_size,
        #                           batch_first=True)
        # self.layer = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, essay_ids: torch.LongTensor, essay_sets: torch.LongTensor, x: torch.LongTensor,
                masks: torch.BoolTensor, scores: torch.FloatTensor, min_scores: torch.LongTensor,
                max_scores: torch.LongTensor):
        # output = self.bert_encoder(x).last_hidden_state
        # _, output = self.gru_encoder(output)
        # # output = self.rnn_encoder(output)
        # output = output.permute(1, 0, 2)
        # output = torch.flatten(output, start_dim=1)
        # # output = output[:, -1, :]
        # output = self.layer(output)
        # output = self.activation(output)
        return self.activation(self.bert_encoder(x).logits)
