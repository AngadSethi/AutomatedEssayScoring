from typing import Tuple, List, Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, PreTrainedTokenizer
from pytorch_lightning import LightningDataModule
from ujson import load as json_load


class BertDataset(Dataset):
    """
    The dataset for the BERT model. To be used only when training the BERT model, since BERT uses a different tokenizer.
    """
    def __init__(self, dataset: pd.DataFrame, max_seq_length: int, doc_stride: int, prompts: dict,
                 tokenizer: PreTrainedTokenizer):
        """
        Initiate the BERT dataset with the appropriate arguments. Appropriate for a PyTorch dataset.

        Args:
            dataset (pd.DataFrame): The pandas dataframe with the train examples.
            max_seq_length (int): The maximum length of the sequence allowed. Please keep in mind the memory limitations.
            doc_stride (int): The overlap between sentences of the same excerpt.
            prompts (dict): The dictionary with the prompts, the min and max scores.
            bert_model (str): The string denoting the size of the bert model to be used.
        """
        self.datalist = dataset.drop(['domain1_score', 'domain2_score'], axis=1)
        self.domain1_scores = dataset['domain1_score'].tolist()
        self.domain1_scores_raw = dataset['domain1_score'].tolist()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.prompts = prompts

        self.essay_ids = self.datalist['essay_id'].tolist()
        self.essay_sets = self.datalist['essay_set'].tolist()
        self.essays = self.datalist['essay'].tolist()

        self.x_encoded_bert = self.tokenizer(
            self.datalist['essay'].tolist(),
            max_length=max_seq_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
        )

        self.domain1_scores = [(score - prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) / (
                prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['max_score'] -
                prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) for i, score in
                               enumerate(self.domain1_scores)]

    def __getitem__(self, index: int) -> Tuple[int, int, List[int], List[int], float, int, int]:
        """
        Overriding the __getitem__ method of the PyTorch Dataset class.

        Args:
            index (int): The index of the record to be rendered in the dataset.

        Returns:
            dataitem (Tuple[int, int, List[int], List[int], float, int, int])
        """
        x = self.x_encoded_bert['input_ids'][index]
        mask = self.x_encoded_bert['attention_mask'][index]
        overflow_mapping = index
        essay_id = self.essay_ids[overflow_mapping]
        essay_set = self.essay_sets[overflow_mapping]
        prompt = self.prompts.get(str(essay_set))

        domain1_min_score = prompt['scoring']['domain1_score']['min_score']
        domain1_max_score = prompt['scoring']['domain1_score']['max_score']

        score1 = self.domain1_scores[overflow_mapping]

        dataitem = (
            essay_id,
            essay_set,
            x,
            mask,
            score1,
            domain1_min_score,
            domain1_max_score
        )

        return dataitem

    def __len__(self) -> int:
        return len(self.x_encoded_bert['input_ids'])


def collate_fn_lightning(examples: List[Tuple[int, int, List[int], List[int], float, int, int]]) -> Tuple:
    essay_ids = []
    essay_sets = []
    essays_bert = []
    scores_domain1 = []
    min_scores_domain1 = []
    max_scores_domain1 = []
    masks = []

    for essay_id, essay_set, bert, mask, score1, domain1_min_score, domain1_max_score in examples:
        essay_ids.append(essay_id)
        essay_sets.append(essay_set)
        masks.append(mask)
        essays_bert.append(bert)
        scores_domain1.append(score1)
        min_scores_domain1.append(domain1_min_score)
        max_scores_domain1.append(domain1_max_score)

    essay_ids = torch.LongTensor(essay_ids)
    essay_sets = torch.LongTensor(essay_sets)
    essays_bert = torch.LongTensor(essays_bert)
    masks = torch.BoolTensor(masks)
    scores_domain1 = torch.FloatTensor(scores_domain1)
    min_scores_domain1 = torch.LongTensor(min_scores_domain1)
    max_scores_domain1 = torch.LongTensor(max_scores_domain1)

    return (
        essay_ids,
        essay_sets,
        essays_bert,
        masks,
        scores_domain1,
        min_scores_domain1,
        max_scores_domain1
    )


class BertDataModule(LightningDataModule):
    def __init__(self, train_file: str, prompts_file: str, bert_model: str, seq_len: int, batch_size: int, essay_set: int, num_workers: int):
        super().__init__()
        self.train_file = train_file
        self.seq_len = seq_len
        self.prompts_file = prompts_file
        self.bert_model = bert_model
        self.batch_size = batch_size
        self.essay_set = essay_set
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        # Reading in essay prompts.
        with open(self.prompts_file, 'r', encoding='utf-8') as fh:
            self.prompts = json_load(fh)

        dataset = pd.read_csv(
            self.train_file,
            header=0,
            sep='\t',
            usecols=['essay_id', 'essay_set', 'essay', 'domain1_score', 'domain2_score'],
            encoding='latin-1'
        )
        self.train_dataset, self.dev_dataset, self.test_dataset = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        filtered_dataset = dataset[dataset['essay_set'] == self.essay_set] if self.essay_set != 0 else dataset
        self.train_dataset, t = train_test_split(filtered_dataset, test_size=0.2)
        self.dev_dataset, self.test_dataset = train_test_split(t, test_size=0.5)

    def train_dataloader(self):
        dataset = BertDataset(
            self.train_dataset,
            self.seq_len,
            doc_stride=0,
            prompts=self.prompts,
            tokenizer=self.tokenizer
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_lightning,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        dataset = BertDataset(
            self.dev_dataset,
            self.seq_len,
            doc_stride=0,
            prompts=self.prompts,
            tokenizer=self.tokenizer
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_lightning,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        dataset = BertDataset(
            self.test_dataset,
            self.seq_len,
            doc_stride=0,
            prompts=self.prompts,
            tokenizer=self.tokenizer
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_lightning,
            num_workers=self.num_workers,
            persistent_workers=True
        )
