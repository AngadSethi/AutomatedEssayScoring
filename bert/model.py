import torch
from torch import optim
import torch.nn as nn
import transformers
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from ujson import load as json_load

from util import quadratic_weighted_kappa, log_final_results
from json import dumps


class BertModel(LightningModule):
    def __init__(self, checkpoint: str, prompts_file: str, lr: float):
        super().__init__()

        # Download the BERT model from Huggingface hub.
        self.bert_model = transformers.BertModel.from_pretrained(checkpoint, output_hidden_states=True)
        self.bert_config = self.bert_model.config
        self.lr = lr

        with open(prompts_file, 'r', encoding='utf-8') as fh:
            self.prompts = json_load(fh)

        # The linear layer for the pooler output.
        self.projection = nn.Linear(
            in_features=self.bert_config.hidden_size,
            out_features=1,
            bias=True
        )

        # The activation for the output of the pooler layer.
        self.activation = nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=filter(lambda x: x.requires_grad, self.parameters()), lr=self.lr)
        return optimizer

    def forward(self, x: torch.LongTensor, masks: torch.BoolTensor) -> torch.Tensor:
        """
        The method which "pushes" forward the input to produce a single score for each example in the batch.
        Args:
            x (torch.LongTensor): The token IDs
            masks (torch.BoolTensor): The tensor with 1s in position of importance

        Returns:

        """
        outputs = self.bert_model(x, attention_mask=masks)

        outputs = self.activation(self.projection(outputs.pooler_output))

        return torch.squeeze(outputs, -1)

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

        self.log_dict({'val_loss': loss, 'quadratic_kappa_overall_dev': quadratic_kappa_overall})
        
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
