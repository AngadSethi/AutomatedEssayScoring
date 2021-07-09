from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset


class EnsembleDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, files: List, prompts: dict):
        self.datalist = dataset.drop(['domain1_score', 'domain2_score'], axis=1)
        self.domain1_scores = dataset['domain1_score'].tolist()
        self.domain1_scores_raw = dataset['domain1_score'].tolist()
        self.prompts = prompts

        self.essay_ids = self.datalist['essay_id'].tolist()
        self.essay_sets = self.datalist['essay_set'].tolist()

        self.domain1_scores = [(score - prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) / (
                    prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['max_score'] -
                    prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) for i, score in
                               enumerate(self.domain1_scores)]

        self.dataframes = [pd.read_csv(f, index_col='id') for f in files]
        self.num_models = len(files)

    def __getitem__(self, index: int):
        essay_id = self.essay_ids[index]
        essay_set = self.essay_sets[index]
        prompt = self.prompts.get(str(essay_set))

        domain1_min_score = prompt['scoring']['domain1_score']['min_score']
        domain1_max_score = prompt['scoring']['domain1_score']['max_score']

        x = [df.loc[essay_id]['result'] for df in self.dataframes]

        score1 = self.domain1_scores[index]

        dataitem = (
            essay_id,
            essay_set,
            x,
            score1,
            domain1_min_score,
            domain1_max_score
        )

        return dataitem

    def __len__(self) -> int:
        return len(self.essay_ids)


def collate_fn(examples):
    essay_ids = []
    essay_sets = []
    x = []
    y = []
    z = []
    scores_domain1 = []
    min_scores_domain1 = []
    max_scores_domain1 = []

    for essay_id, essay_set, data, score1, domain1_min_score, domain1_max_score in examples:
        essay_ids.append(essay_id)
        essay_sets.append(essay_set)
        x.append(data[0])
        y.append(data[1])
        if len(data) > 2:
            z.append(data[2])
        scores_domain1.append(score1)
        min_scores_domain1.append(domain1_min_score)
        max_scores_domain1.append(domain1_max_score)

    essay_ids = torch.LongTensor(essay_ids)
    essay_sets = torch.tensor(essay_sets, dtype=torch.long)
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    z = torch.FloatTensor(z) if len(z) > 0 else None
    scores_domain1 = torch.tensor(scores_domain1, dtype=torch.float)
    min_scores_domain1 = torch.tensor(min_scores_domain1, dtype=torch.long)
    max_scores_domain1 = torch.tensor(max_scores_domain1, dtype=torch.long)

    if z is None:
        return {
            'essay_ids': essay_ids,
            'essay_sets': essay_sets,
            'x': x,
            'y': y,
            'scores': scores_domain1,
            'min_scores': min_scores_domain1,
            'max_scores': max_scores_domain1
        }
    return {
        'essay_ids': essay_ids,
        'essay_sets': essay_sets,
        'x': x,
        'y': y,
        'z': z,
        'scores': scores_domain1,
        'min_scores': min_scores_domain1,
        'max_scores': max_scores_domain1
    }
