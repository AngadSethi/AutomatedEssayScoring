import torch
import torch.nn as nn
import transformers


class BERT(nn.Module):
    def __init__(self, model: str = 'bert-base-uncased'):
        super(BERT, self).__init__()

        # Download the BERT model from Huggingface hub.
        self.bert_model = transformers.BertModel.from_pretrained(model, output_hidden_states=True)
        self.bert_config = self.bert_model.config

        # The linear layer for the pooler output.
        self.projection = nn.Linear(
            in_features=self.bert_config.hidden_size,
            out_features=1,
            bias=True
        )

        # The activation for the output of the pooler layer.
        self.activation = nn.Sigmoid()

    def forward(self, essay_ids: torch.LongTensor, essay_sets: torch.LongTensor, x: torch.LongTensor,
                masks: torch.BoolTensor, scores: torch.FloatTensor, min_scores: torch.LongTensor,
                max_scores: torch.LongTensor) -> torch.Tensor:
        """
        The method which "pushes" forward the input to produce a single score for each example in the batch.
        Args:
            essay_ids (torch.LongTensor): The IDs of the essay
            essay_sets (torch.LongTensor): The essay sets
            x (torch.LongTensor): The token IDs
            masks (torch.BoolTensor): The tensor with 1s in position of importance
            scores (torch.FloatTensor): The gold scores
            min_scores (torch.LongTensor): The minimum scores for scaling
            max_scores (torch.LongTensor): The maximum scores for scaling

        Returns:

        """
        outputs = self.bert_model(x, attention_mask=masks)

        outputs = self.activation(self.projection(outputs.pooler_output))

        outputs = torch.squeeze(outputs, dim=-1)

        return outputs
