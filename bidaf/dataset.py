import os
from typing import Tuple, List, Optional

import pandas as pd
import torch
import torchtext
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from ujson import load as json_load

from util import clean_text


class BiDAFDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, prompts: dict, seq_len: int, essay_set: int, mode: str = 'train'):
        """
        Initiate the BiDAF dataset with the appropriate arguments. Appropriate for a PyTorch dataset.

        Args:
            dataset (pd.DataFrame): The pandas dataframe with the train examples.
            prompts (dict): The dictionary with the prompts, the min and max scores.
            seq_len (int): The maximum length of the sequence allowed. Please keep in mind the memory limitations.
            mode (str): The mode of training (train, dev, or test).
        """
        self.datalist = dataset.drop(['domain1_score', 'domain2_score'], axis=1)
        self.domain1_scores = dataset['domain1_score'].tolist()
        self.domain1_scores_raw = dataset['domain1_score'].tolist()
        self.prompts = prompts

        self.essay_ids = self.datalist['essay_id'].tolist()
        self.essay_sets = self.datalist['essay_set'].tolist()
        self.essays = self.datalist['essay'].tolist()

        self.essay_set = essay_set

        # Download or retrieve the vocabulary.
        self.vocab = torchtext.vocab.GloVe()

        # Get the sPacy tokenizer.
        self.en_tokenizer = torchtext.data.get_tokenizer('spacy')

        # Scale the domain scores using the minimum and maximum scores.
        self.domain1_scores = [(score - prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) / (
                prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['max_score'] -
                prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) for i, score in
                               enumerate(self.domain1_scores)]

        self.x_encoded_bidaf_list = []

        self.prompt_encoded_bidaf_list = []

        # This is an intensive process, so we want to run it once. We do that and save the results as a tensor.
        prompt = self.prompts[str(essay_set)]
        self.prompt_encoded_bidaf_list.append(
            [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
             self.en_tokenizer(clean_text(prompt['prompt']))])

        # This is an intensive process, so we want to run it once. We do that and save the results as a tensor.
        for essay in tqdm(self.datalist['essay'], total=len(self.datalist['essay']), desc="Tokenizing Essays"):
            tokens = self.en_tokenizer(clean_text(essay))[:seq_len]
            if len(tokens) <= seq_len:
                tokens = tokens + ([0 for _ in range(seq_len - len(tokens))])
            self.x_encoded_bidaf_list.append(
                [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in tokens])

        self.prompt_encoded_bidaf = self.prompt_encoded_bidaf_list
        self.x_encoded_bidaf = self.x_encoded_bidaf_list

    def __getitem__(self, index: int) -> Tuple[int, int, List[int], List[int], float, int, int]:
        """
        Overriding the __getitem__ method of the PyTorch Dataset class.

        Args:
            index (int): The index of the record to be rendered in the dataset.

        Returns:
            dataitem (Tuple[int, int, List[int], List[int], float, int, int])
        """
        essay_id = self.essay_ids[index]
        essay_set = self.essay_set
        prompt = self.prompts.get(str(essay_set))

        domain1_min_score = prompt['scoring']['domain1_score']['min_score']
        domain1_max_score = prompt['scoring']['domain1_score']['max_score']

        encoded_prompt = self.prompt_encoded_bidaf[0]
        encoded_x = self.x_encoded_bidaf[index]

        score1 = self.domain1_scores[index]

        dataitem = (
            essay_id,
            essay_set,
            encoded_x,
            encoded_prompt,
            score1,
            domain1_min_score,
            domain1_max_score
        )

        return dataitem

    def __len__(self) -> int:
        """
        The number of examples in the dataset

        Returns:
            length (int): The number of examples.
        """
        return len(self.essay_ids)

def collate_fn(examples):
    """
    Collate the data items and return a dictionary with the parameters necessary to train or evaluate the model.
    Normally passed as a parameter to the DataLoader.

    Args:
        examples: The list of examples to be collated.

    """
    essay_ids = []
    essay_sets = []
    essays_bidaf = []
    prompts = []
    scores_domain1 = []
    min_scores_domain1 = []
    max_scores_domain1 = []

    for essay_id, essay_set, bidaf, prompt, score1, domain1_min_score, domain1_max_score in examples:
        essay_ids.append(essay_id)
        essay_sets.append(essay_set)
        essays_bidaf.append(bidaf)
        prompts.append(prompt)
        scores_domain1.append(score1)
        min_scores_domain1.append(domain1_min_score)
        max_scores_domain1.append(domain1_max_score)

    essay_ids = torch.LongTensor(essay_ids)
    essay_sets = torch.LongTensor(essay_sets)
    essays_bidaf = torch.LongTensor(essays_bidaf)
    prompts = torch.LongTensor(prompts)
    scores_domain1 = torch.FloatTensor(scores_domain1)
    min_scores_domain1 = torch.LongTensor(min_scores_domain1)
    max_scores_domain1 = torch.LongTensor(max_scores_domain1)

    return (
        essay_ids,
        essay_sets,
        essays_bidaf,
        prompts,
        scores_domain1,
        min_scores_domain1,
        max_scores_domain1
    )


class BidafDataModule(LightningDataModule):
    def __init__(self, train_file: str, prompts_file: str, seq_len: int, batch_size: int, essay_set: int, num_workers: int):
        super().__init__()
        self.train_file = train_file
        self.seq_len = seq_len
        self.prompts_file = prompts_file
        self.batch_size = batch_size
        self.essay_set = essay_set
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        torchtext.vocab.GloVe()
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

        filtered_dataset = dataset[dataset['essay_set'] == self.essay_set] if self.essay_set != 0 else dataset
        self.train_dataset, t = train_test_split(filtered_dataset, test_size=0.2)
        self.dev_dataset, self.test_dataset = train_test_split(t, test_size=0.5)

        self.train_dataset = BiDAFDataset(
            self.train_dataset,
            self.prompts,
            self.seq_len,
            essay_set=self.essay_set
        )
        self.dev_dataset = BiDAFDataset(
            self.dev_dataset,
            self.prompts,
            self.seq_len,
            mode='dev',
            essay_set=self.essay_set
        )
        self.test_dataset = BiDAFDataset(
            self.test_dataset,
            self.prompts,
            self.seq_len,
            mode='test',
            essay_set=self.essay_set
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )
