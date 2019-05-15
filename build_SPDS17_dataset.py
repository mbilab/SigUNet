#!/usr/bin/env python3
import os
import sys

import numpy as np

from sigunet.constant import SEQUENCE_LENGTH, AMINO_ACID_LIST

data_path = './data/SPDS17/'
output_dir = './data/features/'
amino_acid_list = AMINO_ACID_LIST
sequence_length = SEQUENCE_LENGTH

dataset_name = {
    'euk': 'Euk',
    'gram-': 'Gram-',
    'gram+': 'Gram+',
}

def read_raw_data(file):

    with open(file, 'r') as f:
        content = f.read()

    content = content.split('>')[1:]
    content = [line.split('\n')[1:-1] for line in content]
    content = [''.join(line) for line in content]
    content = [line for line in content]

    return content

def build_feature(data, sequence_length):

    feature = []
    for c in data[:sequence_length]:
        if c in amino_acid_list:
            feature.append(amino_acid_list.index(c))
        else:
            feature.append(20)

    while len(feature) < sequence_length:
        feature.append(20)

    return feature

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: ./build_SPDS17_dataset.py [dataset name]')
        print()
        print('Avaliable Dataset: euk, gram-, gram+')
        exit()

    dataset = sys.argv[1]

    files = [
        '%s/NC%s.nr.fasta' % (data_path, dataset_name[dataset]),
        '%s/TM%s.nr.fasta' % (data_path, dataset_name[dataset]),
        '%s/SP%s.nr.fasta' % (data_path, dataset_name[dataset]),
    ]

    obj = []

    for i, file in enumerate(files):
        content = read_raw_data(file)

        for line in content:
            features = build_feature(line, sequence_length)

            obj.append((features, i))

    test_data = np.array(obj, dtype=[('features', np.int32, (sequence_length, )), ('label', np.int32)])

    print('test features: ', test_data['features'].shape)
    print('test label   : ', test_data['label'].shape)

    print('Count of class 0 on test data:', np.sum(np.where(test_data['label'] == 0, 1, 0)))
    print('Count of class 1 on test data:', np.sum(np.where(test_data['label'] == 1, 1, 0)))
    print('Count of class 2 on test data:', np.sum(np.where(test_data['label'] == 2, 1, 0)))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    np.save('%s/test.npy' % output_dir, test_data)

    print('============================================')
    print('Save npy file to %s' % output_dir)
