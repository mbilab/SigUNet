#!/usr/bin/env python3
from glob import glob
import json

import numpy as np

from .models import load_model
from .utils import decision

class SubModule:

    def __init__(self, model_path, mode):
        with open(f'{model_path}/grid_search_result.json', 'r') as j:
            self.config = json.load(j)

        self.models = [load_model(self.config['params'], weights, mode) for weights in glob(f'{model_path}/*.h5')]
        self.thr = self.config['eval_thr']

    def predict(self, x):
        y_pred = [model.predict(x) for model in self.models]
        y_pred = sum(y_pred) / len(self.models)
        y_pred = y_pred[:, :, 2]
        y_pred = np.array([decision(seq, n=4, thr=self.thr) for seq in y_pred])
        return y_pred
