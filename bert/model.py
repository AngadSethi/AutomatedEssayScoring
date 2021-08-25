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
    def __init__(self, lr: float, prompts: str, bert_model: str, **kwargs):
        super().__init__()
        self.save_hyperparameters("lr", "bert_model")
        # Download the BERT model from Huggingface hub.
        self.bert_model = transformers.BertModel.from_pretrained(bert_model, output_hidden_states=True)
        self.bert_config = self.bert_model.config

        with open(prompts, 'r', encoding='utf-8') as fh:
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
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BertModel")
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

    def forward(self, x: torch.LongTensor, masks: torch.BoolTensor) -> torch.Tensor:
        """
        The method which "pushes" forward the input to produce a single score for each example in the batch.
        Args:
            x (torch.LongTensor): The token IDs
            masks (torch.BoolTensor): The tensor with 1s in position of importance

        Returns:

        """
        outputs = self.bert_model(x, attention_mask=masks)

        outputs = self.activation(self.projection(outputs.last_hidden_state[:, 0]))

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
