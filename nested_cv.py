#!/usr/bin/env python3
from glob import glob
import json
import shutil
import sys
import os
from pprint import pprint

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical

from sigunet.models import get_model, load_model
from sigunet.utils import k_fold_balance_split, get_thr, decision

if __name__ == '__main__':

    model_path = sys.argv[1]

    params = {
        'n': [16, 20, 24, 28],
        'm': [4, 8, 12, 16],
        'kernel_size': [7, 9, 11, 13],
    }

    data = np.load('./data/features/train.npy')
    data = k_fold_balance_split(data, folds=5)

    eval_prediction = []

    for i in range(5):

        eval_data = data[i][np.where(data[i]['meta'] == 'Evaluation')[0]]
        x_eval = eval_data['features']
        y_eval = eval_data['residue_label']
        y_eval_label = eval_data['label']

        best_va_loss = 10000

        for grid in ParameterGrid(params):

            valid_prediction = []
            cur_va_loss = []

            for j in range(5):

                if i == j:
                    continue

                if not os.path.isdir(f'{model_path}/tmp'):
                    os.makedirs(f'{model_path}/tmp')

                train_data = np.concatenate([data[el] for el in range(5) if el != i and el != j])
                valid_data = data[j]

                x_train = train_data['features']
                y_train = train_data['residue_label']

                x_valid = valid_data['features']
                y_valid = valid_data['residue_label']
                y_valid_label = valid_data['label']

                early_stop = EarlyStopping(patience=10, restore_best_weights=True)

                model = get_model(input_layer=None)(**grid)
                model.fit(x_train, y_train,
                          batch_size=32,
                          epochs=1000,
                          verbose=2,
                          validation_data=(x_valid, y_valid),
                          callbacks=[early_stop])

                model.save_weights(f'{model_path}/tmp/{j}.h5')

                va_loss = model.evaluate(x_valid, y_valid, batch_size=128)
                y_va_pred = model.predict(x_valid)

                cur_va_loss.append(va_loss)
                valid_prediction.append((y_valid_label, y_va_pred))

                clear_session()

            if sum(cur_va_loss) / 4 < best_va_loss:
                best_va_loss = sum(cur_va_loss) / 4

                if os.path.isdir(f'{model_path}/keep'):
                    shutil.rmtree(f'{model_path}/keep')

                os.rename(f'{model_path}/tmp', f'{model_path}/keep')

                va_pred = np.concatenate([el[1] for el in valid_prediction], axis=0)[:, :, 2]
                va_label = np.concatenate([el[0] for el in valid_prediction], axis=0)
                va_label = np.where(va_label == 2, 1, 0)

                va_mcc, va_thr = get_thr(va_label, va_pred)
                eval_thr = min(max(0.65, va_thr + 0.1), 0.85)

                with open(f'{model_path}/keep/config.json', 'w') as j:
                    json.dump({'val_loss': f'{best_va_loss:.6f}', 'val_thr': va_thr, 'eval_thr': eval_thr, 'params': grid},
                              j, indent=4)

            else:
                shutil.rmtree(f'{model_path}/tmp')

        os.rename(f'{model_path}/keep', f'{model_path}/{i}')
        np.save(f'{model_path}/{i}/eval.npy', eval_data)

        with open(f'{model_path}/{i}/config.json') as j:
            info = json.load(j)
            eval_thr = info['eval_thr']
            config = info['params']

        models = [load_model(config, path) for path in glob(f'{model_path}/{i}/*.h5')]
        y_eval_pred = sum([model.predict(x_eval) for model in models]) / 4
        eval_prediction.append((y_eval_label, y_eval_pred))

        clear_session()

    eval_pred = np.concatenate([el[1] for el in eval_prediction], axis=0)[:, :, 2]
    eval_label = np.concatenate([el[0] for el in eval_prediction], axis=0)

    pred = np.array([1 if decision(proba, 4, eval_thr) else 0 for proba in eval_pred])
    label = np.where(eval_label == 2, 1, 0)

    fp = 0
    n = 0

    for p, l in zip(pred, eval_label):
        if l == 1:
            n += 1

            if p == 1:
                fp += 1

    result = {
        'MCC': '%.6f' % matthews_corrcoef(label, pred),
        'Precision': '%.6f' % precision_score(label, pred),
        'Recall': '%.6f' % recall_score(label, pred),
        'F1': '%.6f' % f1_score(label, pred),
        'FPRtm': '%.6f' % (fp/n),
    }

    pprint(result)

    with open(f'{model_path}/result.json', 'w') as f:
        json.dump(result, f, indent=4)
