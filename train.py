"""Train a model on the HP automated essay scoring dataset.

Author:
    Angad Sethi
"""
import random
from collections import OrderedDict
from json import dumps

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torchtext.vocab import GloVe
from tqdm.auto import tqdm
from ujson import load as json_load

import util
from args import get_train_args
from bert.dataset import BertDataset, collate_fn as collate_fn_bert
from bert.model import BERT
from bidaf.dataset import BiDAFDataset, collate_fn as collate_fn_bidaf
from bidaf.model import BiDAF
from ensemble.dataset import EnsembleDataset, collate_fn as collate_fn_ensemble
from ensemble.model import Ensemble
from han.dataset import HANDataset, collate_fn as collate_fn_han
from han.model import HierarchicalAttentionNetwork
from util import quadratic_weighted_kappa


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    # device, args.gpu_ids = 'cpu', []
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    vocab = GloVe()

    # Get data loader
    log.info('Building dataset and model...')

    # Reading in essay prompts.
    with open('./data/essay_prompts.json', 'r', encoding='utf-8') as fh:
        prompts = json_load(fh)

    # Reading in the data from the TSV file
    dataset = pd.read_csv(
        './data/training_set_rel3.tsv',
        header=0,
        sep='\t',
        verbose=True,
        usecols=['essay_id', 'essay_set', 'essay', 'domain1_score', 'domain2_score'],
        encoding='latin-1'
    )

    if not args.train_split:
        train_dataset, dev_dataset = train_test_split(dataset, test_size=0.2, random_state=args.seed)
        dev_dataset, test_dataset = train_test_split(dev_dataset, test_size=0.5, random_state=args.seed)
    else:
        train_dataset, dev_dataset, test_dataset = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for x in range(1, 9):
            t, d = train_test_split(dataset[dataset['essay_set'] == x], test_size=0.2, random_state=args.seed)
            d, test = train_test_split(d, test_size=0.5, random_state=args.seed)
            train_dataset = pd.concat([train_dataset, t])
            dev_dataset = pd.concat([dev_dataset, d])
            test_dataset = pd.concat([test_dataset, test])

    if args.model == 'bert':
        train_dataset = BertDataset(
            train_dataset,
            args.max_seq_length,
            doc_stride=args.doc_stride,
            prompts=prompts,
            bert_model=args.bert_model
        )
        dev_dataset = BertDataset(
            dev_dataset,
            args.max_seq_length,
            doc_stride=args.doc_stride,
            prompts=prompts,
            bert_model=args.bert_model
        )
        test_dataset = BertDataset(
            test_dataset,
            args.max_seq_length,
            doc_stride=args.doc_stride,
            prompts=prompts,
            bert_model=args.bert_model
        )
        model = BERT(
            model=args.bert_model
        )
        collate_fn = collate_fn_bert
        optimizer = optim.AdamW(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                weight_decay=args.l2_wd)
    elif args.model == 'bidaf':
        train_dataset = BiDAFDataset(
            train_dataset,
            prompts,
            args.max_seq_length
        )
        dev_dataset = BiDAFDataset(
            dev_dataset,
            prompts,
            mode='dev',
            seq_len=args.max_seq_length
        )
        test_dataset = BiDAFDataset(
            test_dataset,
            prompts,
            mode='test',
            seq_len=args.max_seq_length
        )
        model = BiDAF(
            args.hidden_size,
            args.max_seq_length,
            vocab,
            args.drop_prob
        )
        collate_fn = collate_fn_bidaf
        optimizer = optim.Adadelta(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                   weight_decay=args.l2_wd)
    elif args.model == 'han':
        train_dataset = HANDataset(
            train_dataset,
            prompts,
            args.max_doc_length,
            args.max_sent_length
        )
        dev_dataset = HANDataset(
            dev_dataset,
            prompts,
            args.max_doc_length,
            args.max_sent_length
        )
        test_dataset = HANDataset(
            test_dataset,
            prompts,
            args.max_doc_length,
            args.max_sent_length
        )
        model = HierarchicalAttentionNetwork(
            args.word_hidden_size,
            args.sent_hidden_size,
            args.drop_prob
        )
        collate_fn = collate_fn_han
        optimizer = optim.SGD(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.l2_wd)
    else:
        train_dataset = EnsembleDataset(
            train_dataset,
            prompts=prompts,
            files=['han/best_train.csv', 'bidaf/best_train.csv', 'bert/best_train.csv']
        )
        dev_dataset = EnsembleDataset(
            dev_dataset,
            prompts=prompts,
            files=['han/best_val.csv', 'bidaf/best_val.csv', 'bert/best_val.csv']
        )
        test_dataset = EnsembleDataset(
            test_dataset,
            prompts=prompts,
            files=['han/best_test.csv', 'bidaf/best_test.csv', 'bert/best_test.csv']
        )
        model = Ensemble(
            num_models=3
        )
        collate_fn = collate_fn_ensemble
        optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                               weight_decay=args.l2_wd)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    dev_loader = data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    if len(args.gpu_ids) > 0:
        model = nn.DataParallel(model, args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0

    model = model.to(device)
    model.train()

    log.info("Model Summary: ")
    print(model)

    # Get saver
    saver = util.CheckpointSaver(
        args.save_dir,
        max_checkpoints=args.max_checkpoints,
        metric_name=args.metric_name,
        maximize_metric=args.maximize_metric,
        log=log
    )
    loss = nn.MSELoss()

    # Get optimizer and scheduler
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Train
    log.info('Training...')
    epoch = step // len(train_dataset)
    steps_till_eval = args.eval_steps
    while epoch != args.num_epochs:
        epoch += 1
        final_result = {
            'id': [],
            'result': []
        }
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}') as progress_bar:
            for inputs in train_loader:
                # Setup for forward
                inputs = {k: v.to(device) for k, v in inputs.items()}
                batch_size = inputs['essay_ids'].size(0)
                optimizer.zero_grad()

                # Forward
                predictions = model(**inputs)
                loss_op = loss(predictions, inputs['scores'])
                loss_val = loss_op.item()

                final_result['id'].extend(inputs['essay_ids'].tolist())
                final_result['result'].extend(predictions.tolist())

                # Backward
                loss_op.backward()
                nn.utils.clip_grad_norm_(filter(lambda x : x.requires_grad, model.parameters()), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Log info
                step += batch_size
                steps_till_eval -= batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, MSE=loss_val)
                tbx.add_scalar('train/MSE', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
        # Evaluate and save checkpoint
        log.info(f'Evaluating')
        results, pred_dict, final_result_val = evaluate(model, dev_loader, device)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'Dev {results_str}')

        if args.train_split:
            results_str = ', '.join(f'{k}: {v}' for k, v in pred_dict.items())
            log.info(f'Dev Stratified {results_str}')

        for k, v in results.items():
            tbx.add_scalar(f'dev/{k}', v, step)

        log.info(f'Testing')
        results_test, pred_dict_test, final_result_test = evaluate(model, test_loader, device)

        saver.save(step, model, results[args.metric_name], final_result, final_result_val, final_result_test, device)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results_test.items())
        log.info(f'Test {results_str}')

        if args.train_split:
            results_str = ', '.join(f'{k}: {v}' for k, v in pred_dict_test.items())
            log.info(f'Test Stratified {results_str}')

        for k, v in results_test.items():
            tbx.add_scalar(f'test/{k}', v, step)


def evaluate(model, data_loader, device):
    mse_meter = util.AverageMeter()
    qwk_meter_rater_1 = util.AverageMeter()
    model.eval()
    loss = nn.MSELoss()
    pred_dict = {}
    final_result = {
        'id': [],
        'result': []
    }
    with open('./data/essay_prompts.json', 'r', encoding='utf-8') as fh:
        prompt_dict = json_load(fh)
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for inputs in data_loader:
            # Setup for forward
            batch_size = inputs['essay_ids'].size(0)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward
            predictions = model(**inputs)
            predictions = torch.squeeze(predictions, dim=-1)
            loss_op = loss(predictions, inputs['scores'])
            loss_val = loss_op.item()
            mse_meter.update(loss_val, batch_size)

            final_result['id'].extend(inputs['essay_ids'].tolist())
            final_result['result'].extend(predictions.tolist())

            predictions = inputs['min_scores'] + ((inputs['max_scores'] - inputs['min_scores']) * predictions)
            scores_domain1 = inputs['min_scores'] + ((inputs['max_scores'] - inputs['min_scores']) * inputs['scores'])

            quadratic_kappa_1 = quadratic_weighted_kappa(
                torch.round(predictions).type(torch.LongTensor).tolist(),
                torch.round(scores_domain1).type(torch.LongTensor).tolist(),
                min_rating=0,
                max_rating=60
            )

            qwk_meter_rater_1.update(quadratic_kappa_1, batch_size)
            pred_dict.update(dict(zip(inputs['essay_ids'].tolist(), predictions.tolist())))

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(MSE=mse_meter.avg, QWK=qwk_meter_rater_1.avg)

    model.train()
    final_dict = {}

    true = data_loader.dataset.domain1_scores_raw
    for s in pred_dict.keys():
        index = data_loader.dataset.essay_ids.index(s)
        essay_s = data_loader.dataset.essay_sets[index]
        if str(essay_s) not in final_dict.keys():
            final_dict[str(essay_s)] = ([round(pred_dict[s])], [true[index]])
        else:
            final_dict[str(essay_s)][0].append(round(pred_dict[s]))
            final_dict[str(essay_s)][1].append(true[index])

    result_dict = {}
    m_sum = 0.0
    m_len = 0
    for i in range(1, 9):
        if str(i) in final_dict.keys():
            result_dict['essay_set_' + str(i)] = quadratic_weighted_kappa(final_dict[str(i)][0], final_dict[str(i)][1],
                                                                          min_rating=prompt_dict[str(i)]['scoring'][
                                                                              'domain1_score']['min_score'], max_rating=
                                                                          prompt_dict[str(i)]['scoring'][
                                                                              'domain1_score']['max_score'])
            m_sum += result_dict['essay_set_' + str(i)]
            m_len += 1

    result_dict['avg'] = m_sum / m_len

    results_list = [
        ('MSE', mse_meter.avg),
        ('QWK', result_dict['avg']),
        ('QWK_OVERALL', quadratic_kappa_1)
    ]
    results = OrderedDict(results_list)
    result_dict = OrderedDict(sorted(result_dict.items()))

    return results, result_dict, final_result


if __name__ == '__main__':
    args = get_train_args()
    args.name = 'ensemble-final'
    args.model = 'ensemble'
    args.bert_model = 'bert-base-uncased'
    args.batch_size = 32
    args.num_workers = 2
    args.train_split = True
    args.num_epochs = 100
    args.max_doc_length = 100
    args.max_sent_length = 400
    args.word_hidden_size = 100
    args.sent_hidden_size = 100
    args.max_seq_length = 1024
    args.lr = 0.001
    args.eval_steps = 5000
    main(args)
