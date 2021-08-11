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
    def __init__(self, lr: float, prompts_file: str, model_checkpoint: str = 'bert-base-uncased'):
        super().__init__()
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
        self.activation = nn.Sigmoid()
        self.lr = lr

        with open(prompts_file, 'r', encoding='utf-8') as fh:
            self.prompts = json_load(fh)

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=filter(lambda x: x.requires_grad, self.parameters()), lr=self.lr)
        return optimizer

    def forward(self, x: torch.LongTensor, masks: torch.BoolTensor):
        return torch.squeeze(self.activation(self.bert_encoder(x, attention_mask=masks).logits), -1)

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

        quadratic_kappa_overall = quadratic_weighted_kappa(
            torch.round(scaled_predictions).type(torch.IntTensor).tolist(),
            torch.round(scores_domain1).type(torch.IntTensor).tolist(),
            min_rating=0,
            max_rating=60
        )

        self.log_dict({'val_loss': loss, 'quadratic_kappa_overall_val': quadratic_kappa_overall})

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

        quadratic_kappa_overall = quadratic_weighted_kappa(
            torch.round(scaled_predictions).type(torch.IntTensor).tolist(),
            torch.round(scores_domain1).type(torch.IntTensor).tolist(),
            min_rating=0,
            max_rating=60
        )

        self.log_dict({'test_loss': loss, 'quadratic_kappa_overall_test': quadratic_kappa_overall})

        return {
            'essay_ids': essay_ids,
            'essay_sets': essay_sets,
            'loss': loss,
            'predictions': scaled_predictions,
            'scores': scores_domain1
        }

    def validation_epoch_end(self, outputs):
        print(f"Val Results: {dumps(log_final_results(outputs, self.prompts), indent=4)}")

    def test_epoch_end(self, outputs):
        print(f"Test Results: {dumps(log_final_results(outputs, self.prompts), indent=4)}")
