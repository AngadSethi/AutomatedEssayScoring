"""Top-level model classes.

Author:
    Angad Sethi
"""
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import optim
from transformers import AutoConfig, AutoModelWithHeads

from util import log_final_results
import torch.nn.functional as F
from ujson import load as json_load


class BertModelWithAdapters(LightningModule):
    def __init__(self, lr: float, prompts: str, bert_model: str, drop_prob: float, use_scheduler: str,
                 max_seq_length: int, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        config = AutoConfig.from_pretrained(bert_model, num_labels=1)
        config.hidden_dropout_prob = drop_prob
        self.bert_encoder = AutoModelWithHeads.from_pretrained(bert_model, config=config)
        if kwargs['model'] == "original":
            self.bert_encoder.add_adapter("aes")
            self.bert_encoder.train_adapter("aes")
            self.bert_encoder.set_active_adapters("aes")
        self.bert_encoder.add_classification_head(
            "aes",
            num_labels=1,
            activation_function="relu"
        )
        self.activation = nn.Sigmoid()
        self.use_scheduler = use_scheduler

        with open(prompts, 'r', encoding='utf-8') as fh:
            self.prompts = json_load(fh)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        if self.use_scheduler == "no":
            return optimizer
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, 0.04 * self.hparams.lr, self.hparams.lr, cycle_momentum=False)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

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
        parser.add_argument('--drop_prob',
                            type=float,
                            default=0.2,
                            help='The dropout probability')
        parser.add_argument('--use_scheduler',
                            type=str,
                            default="no",
                            choices=("yes", "no"),
                            help='Should the model use a cyclic LR scheduler?')
        return parent_parser

    def forward(self, x: torch.LongTensor, masks: torch.BoolTensor):
        output = self.bert_encoder(input_ids=x, attention_mask=masks).logits
        output = self.activation(output)
        return torch.squeeze(output, -1)

    def training_step(self, batch, batch_idx):
        essay_ids, essay_sets, x, masks, scores, min_scores, max_scores = batch
        predictions = self(x, masks)
        loss = F.mse_loss(predictions, scores)
        self.log('train_loss', loss)
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

    def test_epoch_end(self, outputs):
        final_results = log_final_results(outputs, self.prompts)
        self.log_dict(final_results)
