"""Top-level model classes.

Author:
    Angad Sethi
"""
from json import dumps

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import optim
from transformers import BertModel, AutoModel, AutoConfig, AutoModelWithHeads

import layers
from util import quadratic_weighted_kappa, log_final_results
import torch.nn.functional as F
from ujson import load as json_load


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


class BertModelWithAdapters(LightningModule):
    def __init__(self, lr: float, prompts: str, bert_model: str, **kwargs):
        super().__init__()
        self.save_hyperparameters("lr", "bert_model")
        config = AutoConfig.from_pretrained(bert_model, num_labels=1)
        self.bert_encoder = AutoModelWithHeads.from_pretrained(bert_model, config=config)

        self.bert_encoder.add_adapter("aes")
        self.bert_encoder.add_classification_head(
            "aes",
            num_labels=1,
            activation_function="relu"
        )
        self.bert_encoder.train_adapter("aes")
        self.bert_encoder.set_active_adapters("aes")
        self.activation = nn.Sigmoid()

        with open(prompts, 'r', encoding='utf-8') as fh:
            self.prompts = json_load(fh)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BertAdapterModel")
        parser.add_argument('--bert_model',
                            type=str,
                            default='bert-base-uncased',
                            choices=('bert-base-uncased', 'bert-large-uncased'),
                            help='The type of BERT model used.')
        parser.add_argument('--prompts',
                            type=str,
                            default='./data/essay_prompts.json',
                            help='The JSON files with prompts')
        parser.add_argument('--lr',
                            type=float,
                            default=1e-04,
                            help='Learning rate.')
        parser.add_argument('--max_seq_length',
                            type=int,
                            default=384,
                            help='The maximum sequence length that is provided to the model.')
        return parent_parser

    def forward(self, x: torch.LongTensor, masks: torch.BoolTensor):
        output = self.bert_encoder(input_ids=x, attention_mask=masks).logits
        output = self.activation(output)
        return torch.squeeze(output, -1)

    def training_step(self, batch, batch_idx):
        essay_ids, essay_sets, x, masks, scores, min_scores, max_scores = batch
        predictions = self(x, masks)
        loss = F.mse_loss(predictions, scores)
        return loss

    def validation_step(self, batch, batch_idx):
        essay_ids, essay_sets, x, masks, scores, min_scores, max_scores = batch
        predictions = self(x, masks)
        loss = F.mse_loss(predictions, scores)

        scaled_predictions = min_scores + ((max_scores - min_scores) * predictions)
        scores_domain1 = min_scores + ((max_scores - min_scores) * scores)

        self.log('val_loss', loss)

        return {
            'essay_ids': essay_ids,
            'essay_sets': essay_sets,
            'loss': loss,
            'predictions': scaled_predictions,
            'scores': scores_domain1
        }

    def test_step(self, batch, batch_idx):
        essay_ids, essay_sets, x, masks, scores, min_scores, max_scores = batch
        predictions = self(x, masks)
        loss = F.mse_loss(predictions, scores)

        scaled_predictions = min_scores + ((max_scores - min_scores) * predictions)
        scores_domain1 = min_scores + ((max_scores - min_scores) * scores)

        self.log('test_loss', loss)

        return {
            'essay_ids': essay_ids,
            'essay_sets': essay_sets,
            'loss': loss,
            'predictions': scaled_predictions,
            'scores': scores_domain1
        }

    def validation_epoch_end(self, outputs):
        final_results = log_final_results(outputs, self.prompts)
        self.log_dict(final_results)

        print(f"Val Results: {dumps(final_results, indent=4)}")

    def test_epoch_end(self, outputs):
        final_results = log_final_results(outputs, self.prompts)
        self.log_dict(final_results)

        print(f"Test Results: {dumps(final_results, indent=4)}")
