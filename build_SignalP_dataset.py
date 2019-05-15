#!/usr/bin/env python3
import os
import sys

import numpy as np

from sigunet.constant import SEQUENCE_LENGTH, AMINO_ACID_LIST

data_path = './data/SignalP/'
output_dir = './data/features/'
sequence_length = SEQUENCE_LENGTH

amino_acid_list = AMINO_ACID_LIST


def read_raw_data(file_path, db):

    def _process_single_data(seq, header):
        assert len(seq) % 2 == 0

        length = len(seq) // 2
        data = seq[:length]
        label = seq[length:]

        feature_type = 'Reduction'

        if 'Train' in header:
            feature_type = 'Train'
        if 'Evaluation' in header:
            feature_type = 'Evaluation'

        return {
            'data': data,
            'label': label,
            'length': int(header[0]),
            'id': header[1],
            'type': feature_type,
            'db': db,
        }

    proteins = []

    with open(file_path, 'r') as f:

        lines = [el.rstrip('\n') for el in f.readlines()]

        seq = ''
        header = [el for el in lines[0].rstrip('\n').split(' ') if el != '']

        for line in lines[1:]:

            if line[0] is ' ':
                proteins.append(_process_single_data(seq, header))

                seq = ''
                header = [el for el in line.rstrip('\n').split(' ') if el != '']
            else:
                seq += line.rstrip('\n')

    proteins.append(_process_single_data(seq, header))

    return proteins

def _build_sequence_features(data, sequence_length):

    feature = []
    for c in data[:sequence_length]:
        if c in amino_acid_list:
            feature.append(amino_acid_list.index(c))
        else:
            feature.append(20)

    while len(feature) < sequence_length:
        feature.append(20)

    return feature

def _build_sequence_classes(label):
    if 'S' in label:
        return 2
    elif 't' in label[:70] or 'T' in label[:70]:
        return 1
    else:
        return 0

def _build_residue_classes(label):
    mapping = {'S': 2, 't': 1, 'T': 1, '.': 0}
    residue_label = [mapping[k] for k in label]

    while len(residue_label) < sequence_length:
        residue_label.append(0)

    site = -1
    for i, r in enumerate(residue_label):
        if i > 0 and residue_label[i] != 2 and residue_label[i - 1] == 2:
            site = i
            break

    return residue_label, site

def build_sequence_data(proteins, sequence_length):

    obj = []

    for p in proteins:
        features = _build_sequence_features(p['data'], sequence_length=sequence_length)
        label = _build_sequence_classes(p['label'])
        residue_label, cleavage_site = _build_residue_classes(p['label'][:sequence_length])
        meta_data = p['type']
        db = p['db']

        obj.append((features, label, meta_data, residue_label, cleavage_site, db))

    return np.array(obj, dtype=[
        ('features', np.int32, (sequence_length, )),
        ('label', np.int32),
        ('meta', np.unicode_, 10),
        ('residue_label', np.int32, (sequence_length, )),
        ('cleavage_site', np.int32),
        ('db', np.unicode_, 8),
    ])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: ./build_SignalP_dataset.py [dataset name]')
        print()
        print('Avaliable Dataset: euk, gram-, gram+, all')
        exit()

    dataset = sys.argv[1]
    proteins = []

    if dataset != 'all' and dataset != 'bacteria':
        proteins.extend(read_raw_data('%s/%s.2.how' % (data_path, dataset), dataset))

    elif dataset == 'all':
        proteins.extend(read_raw_data('%s/euk.2.how' % data_path, 'euk'))
        proteins.extend(read_raw_data('%s/gram-.2.how' % data_path, 'gram-'))
        proteins.extend(read_raw_data('%s/gram+.2.how' % data_path, 'gram+'))

    elif dataset == 'bacteria':
        proteins.extend(read_raw_data('%s/gram-.2.how' % data_path, 'gram-'))
        proteins.extend(read_raw_data('%s/gram+.2.how' % data_path, 'gram+'))
    else:
        print('Unknown dataset')
        exit()


    train_set = [p for p in proteins if p['type'] != 'Reduction']
    eval_set = [p for p in proteins if p['type'] == 'Evaluation']
    train_data = build_sequence_data(train_set, sequence_length = sequence_length)
    eval_data = build_sequence_data(eval_set, sequence_length = sequence_length)

    print('train features: ', train_data['features'].shape)
    print('train label   : ', train_data['label'].shape)

    print('Count of class 0 on train data:', np.sum(np.where(train_data['label'] == 0, 1, 0)))
    print('Count of class 1 on train data:', np.sum(np.where(train_data['label'] == 1, 1, 0)))
    print('Count of class 2 on train data:', np.sum(np.where(train_data['label'] == 2, 1, 0)))

    print('Count of class 0 on eval data:', np.sum(np.where(eval_data['label'] == 0, 1, 0)))
    print('Count of class 1 on eval data:', np.sum(np.where(eval_data['label'] == 1, 1, 0)))
    print('Count of class 2 on eval data:', np.sum(np.where(eval_data['label'] == 2, 1, 0)))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    np.save(output_dir + '/train.npy', train_data)

    print('============================================')
    print('Save npy file to %s' % output_dir)
