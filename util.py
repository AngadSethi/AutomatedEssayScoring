"""Utility classes and methods.

Most of the functions here have been borrowed from https://github.com/chrischute/squad

Author:
    Angad Sethi (angadsethi_2k18co066@dtu.ac.in)
"""
import math
import os

import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
import torch.nn.functional as F


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


def log_final_results(outputs, prompts):
    essay_sets = torch.cat([o['essay_sets'] for o in outputs]).tolist()
    predictions = torch.round(torch.cat([o['predictions'] for o in outputs])).type(torch.IntTensor).tolist()
    scores = torch.cat([o['scores'] for o in outputs]).type(torch.IntTensor).tolist()
    final_results = {'quadratic_kappa_overall': quadratic_weighted_kappa(
        predictions,
        scores,
        min_rating=0,
        max_rating=60
    )}

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
    return final_results
