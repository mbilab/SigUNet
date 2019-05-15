#!/usr/bin/env python3
import random

from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import matthews_corrcoef as MCC

def decision(sequence_proba, n, thr):
    """Decide sequence is signal peptide or not.

    Return true if continuous `n` positions have probability of signal peptide is upper than `thr`.

    Arguments:
        sequence_proba (numpy array): A sequence probability of a protein.
        n (int): The size of continuous range.
        thr (float): The threshold to decide a position is signal peptide residue or not.

    Returns:
        is_signal_peptide (bool): `True` for signal peptide, `False` for other.
    """

    # Calculate accumlate sum array
    accumlate = []
    accumlate.append(1 if sequence_proba[0] >= thr else 0)
    for i in range(1, len(sequence_proba)):
        accumlate.append(accumlate[-1] + (1 if sequence_proba[i] >= thr else 0))

    # append `0` for `accumlate[-1]`
    accumlate.append(0)

    for i in range(0, len(sequence_proba) - n + 1):
        if accumlate[i + n - 1] - accumlate[i - 1] == n:
            return True

    return False

def get_thr(label, proba):
    """Find the threshold that has maximum Matthews correlation coefficient.

    Arguments:
        label (numpy array): Class indices of each proteins. Its shape should be `(num_proteins,)`.
        proba (numpy array): Sequence probability of each protein. Its shape should be  `(num_proteins, num_classes)`.

    Returns:
        max_mcc (float): Maximum Matthews correlation coefficient.
        best_thr (float): The threshold whic has maximum Matthews correlation coefficient.
    """

    def _get_thr_subjob(i):
        thr = 1 - 0.01 * i
        pred = [1 if decision(p, n=4, thr=thr) else 0 for p in proba]

        return MCC(label, np.array(pred)), thr

    result = Parallel(n_jobs=-1)(delayed(_get_thr_subjob)(i) for i in range(70))

    return max(result, key=lambda el: el[0])

def balance_split(obj, tr_size):
    """Split data to two part.

    Split data to two part and make sure each label has same ratio in every part.
    First part has the ratio of `tr_size`, and second part has the ratio `1 - tr_size`

    Arguments:
        obj (numpy structured array): Data which need to be splited.
                                      See `./bin/build_train_data.py` to check the format of structured array.
        tr_size (float): A float number between 0 and 1. It mean the ratio of first part.

    Returns:
        first_part (numpy structured array): First part of `obj`.
        second_part (numpy structured array): Second part of `obj`.
    """

    num_classes = max(obj['label']) + 1

    each_idx = []
    each_tr_size = []

    # for each class
    for indice in range(num_classes):

        # find original index of current class
        class_idx = np.where(obj['label'] == indice)[0]

        # shuffle index order
        np.random.shuffle(class_idx)

        each_idx.append(class_idx)
        each_tr_size.append(int(class_idx.shape[0] * tr_size))

    train_idx = []
    valid_idx = []

    for size, class_idx in zip(each_tr_size, each_idx):
        train_idx.extend(class_idx[:size])
        valid_idx.extend(class_idx[size:])

    return obj[train_idx], obj[valid_idx]

def k_fold_balance_split(obj, folds):
    """Split data to `k` fold.

    Split data to `k` part and make sure each label has same ratio in every part.

    Arguments:
        obj (numpy structured array): Data which need to be splited.
                                      See `./bin/build_train_data.py` to check the format of structured array.
        folds (int): Number of folds,

    Returns:
        split_data (list): Each element is a fold of `obj`.
    """

    num_classes = max(obj['label']) + 1

    each_idx = []

    # for each class
    for indices in range(num_classes):

        # find original index of current class
        class_idx = np.where(obj['label'] == indices)[0]
        class_size = class_idx.shape[0] // folds

        # shuffle index order
        np.random.shuffle(class_idx)

        # calculate size of each fold
        segment = [class_size] * folds

        # check the size of data between `class_idx.shape[0] // folds * folds` and `class_idx.shape[0]`
        mod = class_idx.shape[0] - class_size * folds

        # random assign those data to each fold
        order = [i for i in range(folds)]
        random.shuffle(order)

        for i in range(mod):
            segment[order[i]] += 1

        for i in range(1, folds):
            segment[i] += segment[i - 1]

        each_idx.append(np.split(class_idx, segment)[:-1])

    split_data = []

    for i in range(folds):
        idx = []

        for j in range(num_classes):
            idx.extend(each_idx[j][i])

        split_data.append(obj[idx])

    return split_data

def repeat_sample(obj, sample_weights):
    """Repeat sample in dataset.

    Arguments:
        obj (numpy structured array): Data which need to be splited.
                                      See `./bin/build_train_data.py` to check the format of structured array.
        sample_weight (list): Ratio of each sample.

    Returns:
        new_obj (numpy structured array): New dataset. See `./bin/build_train_data.py`
                                          to check the format of structured array.
    """

    new_order = []
    for i, w in enumerate(sample_weights):
        class_idx = np.where(obj['label'] == i)[0]

        weighted_idx = np.repeat(class_idx, w)
        np.random.shuffle(weighted_idx)

        new_order.append(weighted_idx)

    new_order = np.concatenate(new_order, axis=0)

    return obj[new_order]

if __name__ == '__main__':

    data = np.load('data/features/train.npy')

    train, test = balance_split(data, tr_size=0.8)

    print('Balance split:')
    print('Count of class 0 on train data:', np.sum(np.where(train['label'] == 0, 1, 0)))
    print('Count of class 1 on train data:', np.sum(np.where(train['label'] == 1, 1, 0)))
    print('Count of class 2 on train data:', np.sum(np.where(train['label'] == 2, 1, 0)))
    print('')
    print('Count of class 0 on test data:', np.sum(np.where(test['label'] == 0, 1, 0)))
    print('Count of class 1 on test data:', np.sum(np.where(test['label'] == 1, 1, 0)))
    print('Count of class 2 on test data:', np.sum(np.where(test['label'] == 2, 1, 0)))

    print('====================================================')
    print('K-fold balance split:')

    data_split = k_fold_balance_split(data, folds=5)

    for i, s in enumerate(data_split):
        print('Count of class 0 on fold %d:' % i, np.sum(np.where(s['label'] == 0, 1, 0)))
        print('Count of class 1 on fold %d:' % i, np.sum(np.where(s['label'] == 1, 1, 0)))
        print('Count of class 2 on fold %d:' % i, np.sum(np.where(s['label'] == 2, 1, 0)))
        print('')

    print('====================================================')
    print('Repeat sample:')

    data_repeat = repeat_sample(data, sample_weights=[1, 2, 1])

    print('Count of class 0 before repeat:', np.sum(np.where(data['label'] == 0, 1, 0)))
    print('Count of class 1 before repeat:', np.sum(np.where(data['label'] == 1, 1, 0)))
    print('Count of class 2 before repeat:', np.sum(np.where(data['label'] == 2, 1, 0)))
    print('')
    print('Count of class 0 after repeat:', np.sum(np.where(data_repeat['label'] == 0, 1, 0)))
    print('Count of class 1 after repeat:', np.sum(np.where(data_repeat['label'] == 1, 1, 0)))
    print('Count of class 2 after repeat:', np.sum(np.where(data_repeat['label'] == 2, 1, 0)))
