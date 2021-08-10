"""Utility classes and methods.

Most of the functions here have been borrowed from https://github.com/chrischute/squad

Author:
    Angad Sethi (angadsethi_2k18co066@dtu.ac.in)
"""
import json
import logging
import math
import os
import queue
import shutil

import numpy as np
import pandas as pd
import torch
import tqdm
from prettytable import PrettyTable
import torch.nn.functional as F


class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, final_results, final_results_val, final_results_test, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        dataframe = pd.DataFrame(final_results)
        dataframe_val = pd.DataFrame(final_results_val)
        dataframe_test = pd.DataFrame(final_results_test)

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            best_result = os.path.join(self.save_dir, 'best_train.csv')
            best_result_val = os.path.join(self.save_dir, 'best_val.csv')
            best_result_test = os.path.join(self.save_dir, 'best_test.csv')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')
            dataframe.to_csv(best_result, index=False)
            dataframe_val.to_csv(best_result_val, index=False)
            dataframe_test.to_csv(best_result_test, index=False)

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    num_ratings = max_rating - min_rating + 1
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = max_rating - min_rating + 1
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b,
                             min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    scoreQuadraticWeightedKappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    scoreQuadraticWeightedKappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    score_quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    if denominator == 0:
        print(hist_rater_a, hist_rater_b)

    return 1.0 - numerator / denominator


def linear_weighted_kappa(rater_a, rater_b,
                          min_rating=None, max_rating=None):
    """
    Calculates the linear weighted kappa
    linear_weighted_kappa calculates the linear weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    linear_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    linear_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = abs(i - j) / float(num_ratings - 1)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def kappa(rater_a, rater_b,
          min_rating=None, max_rating=None):
    """
    Calculates the kappa
    kappa calculates the kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.

    kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.

    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.

    kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            if i == j:
                d = 0.0
            else:
                d = 1.0
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def mean_quadratic_weighted_kappa(kappas, weights=None):
    """
    Calculates the mean of the quadratic
    weighted kappas after applying Fisher's r-to-z transform, which is
    approximately a variance-stabilizing transformation.  This
    transformation is undefined if one of the kappas is 1.0, so all kappa
    values are capped in the range (-0.999, 0.999).  The reverse
    transformation is then applied before returning the result.

    mean_quadratic_weighted_kappa(kappas), where kappas is a vector of
    kappa values
    mean_quadratic_weighted_kappa(kappas, weights), where weights is a vector
    of weights that is the same size as kappas.  Weights are applied in the
    z-space
    """
    kappas = np.array(kappas, dtype=float)
    if weights is None:
        weights = np.ones(np.shape(kappas))
    else:
        weights = weights / np.mean(weights)

    # ensure that kappas are in the range [-.999, .999]
    kappas = np.array([min(x, .999) for x in kappas])
    kappas = np.array([max(x, -.999) for x in kappas])

    z = 0.5 * np.log((1 + kappas) / (1 - kappas)) * weights
    z = np.mean(z)
    kappa = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
    return kappa


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.
    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.
    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


def visualize(tbx, essay_ids, rater_1, rater_2, num_visuals, step, split='dev'):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        eval_path (str): Path to eval JSON file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return

    visual_ids = np.random.choice(torch.flatten(essay_ids).tolist(), size=num_visuals, replace=False)
    pred_dict = {}
    for index, id_ in enumerate(torch.flatten(essay_ids).tolist()):
        pred_dict[str(id_)] = (rater_1[index, 0], rater_2[index, 0])

    x = PrettyTable()
    x.field_names = ["Essay", "Gold Score", "Predicted Score", "Annual Rainfall"]

    dataset = pd.read_csv(
        './data/training_set_rel3.tsv',
        header=0,
        sep='\t',
        usecols=['essay_id', 'essay_set', 'essay', 'domain1_score', 'domain2_score'],
        encoding='latin-1'
    )

    for i, essay_id in enumerate(visual_ids):
        rater1, rater2 = pred_dict[str(essay_id)]
        row_index = dataset['essay_id'].tolist().index(essay_id)
        essay = dataset.iloc[row_index]['essay']
        score_1 = dataset.iloc[row_index]['domain1_score']
        score_2 = 0
        if not math.isnan(dataset.iloc[row_index]['domain2_score']):
            score_2 = dataset.iloc[row_index]['domain2_score']

        gold_rater1, gold_rater2 = score_1, score_2
        x.add_row([essay, gold_rater1, gold_rater2])

        tbl_fmt = (
                f'- **Essay:** {essay}\n'
                + f'- **Gold - Rater 1:** {gold_rater1}\n'
                + f'- **Predicted - Rater 1:** {rater1}\n'
                + f'- **Gold - Rater 2:** {gold_rater2}\n'
                + f'- **Predicted - Rater 2:** {rater2}'
        )
        tbx.add_text(tag=f'{split}/{i + 1}_of_{num_visuals}',
                     text_string=tbl_fmt,
                     global_step=step)
    print(x)


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


def visualize_table(essay_ids, rater_1, num_visuals):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        eval_path (str): Path to eval JSON file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return

    visual_ids = np.random.choice(essay_ids, size=num_visuals, replace=False)
    pred_dict = {}
    for index, id_ in enumerate(essay_ids):
        pred_dict[str(id_)] = rater_1[index]

    x = PrettyTable()
    x.field_names = ["Essay", "Gold Score", "Predicted Score"]

    dataset = pd.read_csv(
        './data/training_set_rel3.tsv',
        header=0,
        sep='\t',
        usecols=['essay_id', 'essay_set', 'essay', 'domain1_score', 'domain2_score'],
        encoding='latin-1'
    )

    for i, essay_id in enumerate(visual_ids):
        rater1 = pred_dict[str(essay_id)]
        row_index = dataset['essay_id'].tolist().index(essay_id)
        essay = dataset.iloc[row_index]['essay']
        score_1 = dataset.iloc[row_index]['domain1_score']
        score_2 = 0
        if not math.isnan(dataset.iloc[row_index]['domain2_score']):
            score_2 = dataset.iloc[row_index]['domain2_score']

        gold_rater1, gold_rater2 = score_1, score_2
        x.add_row([essay, gold_rater1, rater1])
    print(x)


def load_csv(args, mode='train'):
    file_groups = [
        os.path.join(args.ensemble_model_1, f'best_{mode}.csv'),
        os.path.join(args.ensemble_model_2, f'best_{mode}.csv')
    ]
    if len(args.ensemble_model_3) != 0:
        file_groups.append(os.path.join(args.ensemble_model_3, f'best_{mode}.csv'))
    return file_groups


def log_final_results(outputs, log_dict, prompts):
    essay_sets = torch.cat([o['essay_sets'] for o in outputs]).tolist()
    predictions = torch.round(torch.cat([o['predictions'] for o in outputs])).type(torch.IntTensor).tolist()
    scores = torch.cat([o['scores'] for o in outputs]).type(torch.IntTensor).tolist()

    result_sets = {}
    for index, essay_set in enumerate(essay_sets):
        essay_set = str(essay_set)
        if essay_set in result_sets.keys():
            result_sets[essay_set]['predictions'].append(predictions[index])
            result_sets[essay_set]['scores'].append(scores[index])
        else:
            result_sets[essay_set] = {
                'predictions': [predictions[index]],
                'scores': [scores[index]]
            }

    avg = 0.0
    l = 0
    final_results = {}

    for key, value in result_sets.items():
        qwk = quadratic_weighted_kappa(
            value['predictions'],
            value['scores'],
            min_rating=prompts[str(key)]['scoring']['domain1_score']['min_score'],
            max_rating=prompts[str(key)]['scoring']['domain1_score']['max_score']
        )
        final_results[f"essay_set_{key}"] = qwk
        avg += qwk
        l += 1

    final_results[f"essay_set_avg"] = avg / l
    log_dict(final_results)
