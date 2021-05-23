"""Command-line arguments for setup.py, train.py, test.py.

Author:
    Angad Sethi
"""

import argparse


def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model to score essays in an automated pipeline.')

    add_train_test_args(parser)

    parser.add_argument('--lr',
                        type=float,
                        default=0.00001,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='QWK',
                        choices=('MSE', 'QWK'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')

    args, _ = parser.parse_known_args()

    if args.metric_name == 'MSE':
        # Best checkpoint is the one that minimizes mean squared error
        args.maximize_metric = False
    elif args.metric_name == 'QWK':
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--train_split',
                        type=bool,
                        default=False,
                        help='Should the essays be split according to essay sets')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=1024,
                        help='The maximum sequence length that is provided to the model.')
    parser.add_argument('--doc_stride',
                        type=int,
                        default=32,
                        help='The doc stride for the context')
    parser.add_argument('--max_query_length',
                        type=int,
                        default=128,
                        help='The maximum length of the query')
    parser.add_argument('--max_doc_length',
                        type=int,
                        default=1000,
                        help='The maximum number of the sentences involved in the document.')
    parser.add_argument('--max_sent_length',
                        type=int,
                        default=100,
                        help='The maximum length of the sentences involved in the document.')
    parser.add_argument('--bert_model',
                        type=str,
                        default='bert-base-uncased',
                        choices=('bert-base-uncased', 'bert-large-uncased'),
                        help='The type of BERT model used.')
    parser.add_argument('--model',
                        type=str,
                        choices=('bert', 'bidaf', 'han'),
                        help='The type of model used.')
    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')
    parser.add_argument('--word_hidden_size',
                        type=int,
                        default=50,
                        help='The hidden size of words, to be used in HAN')
    parser.add_argument('--sent_hidden_size',
                        type=int,
                        default=100,
                        help='The hidden size of sentences, to be used in HAN')
