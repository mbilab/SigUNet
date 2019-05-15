#!/usr/bin/env python3
import json
import sys
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.backend import get_session
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score

from sigunet.eval import SubModule

def main():
    with open(f'{sys.argv[1]}/metadata.json', 'r') as f:
        mode = json.load(f)['model']
    models = [SubModule(f'{sys.argv[1]}/{i}', mode) for i in range(5)]
    data = np.load('./data/features/test.npy')

    x = data['features']
    y_pred = np.array([model.predict(x) for model in models])
    y_pred = np.sum(y_pred, axis=0)
    y_pred = np.where(y_pred > 2.5, 1, 0)
    y_true = data['label']

    n = 0
    fp = 0

    for t, p in zip(y_true, y_pred):
        if t == 1:
            n += 1

            if p == 1:
                fp += 1

    y_true = np.where(y_true == 2, 1, 0)

    result = {
        'MCC': '%.6f' % matthews_corrcoef(y_true, y_pred),
        'Precision': '%.6f' % precision_score(y_true, y_pred),
        'Recall': '%.6f' % recall_score(y_true, y_pred),
        'F1': '%.6f' % f1_score(y_true, y_pred),
        'FPRtm': '%.6f' % (fp/n),
    }

    pprint(result)

if __name__ == '__main__':
    main()
