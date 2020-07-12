# read the annotations from the CSV file
import csv
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dl_src.dataset import make_list_of_files

annotation_path = '/home/imt/dataset/dataset_for_multilabel_classification/person/results.csv'
images_dir = '/home/imt/dataset/dataset_for_multilabel_classification'
union_rule = {'corrupted': 'wrong', 'non_person': 'wrong'}

factor = .2
pv = int(1 / factor)

minw = 30 + 40
minh = 60 + 40

# data_name, attr_names, reader = None, None, None
if __name__ == '__main__':
    fnames, labels = [], []
    with open(annotation_path) as f:
        reader = csv.DictReader(f)
        data_name = reader.fieldnames[0]
        attr_names = reader.fieldnames[1]
        image2full = {a: os.path.join(b, a) for b, a in zip(*make_list_of_files(images_dir))}
        not_exists = 0
        for row in tqdm(reader):
            if (row[data_name] not in image2full) or (not os.path.exists(image2full[row[data_name]])):
                not_exists += 1
                continue
            img = cv2.imread(image2full[row[data_name]])
            if (img.shape[0] < minh) or (img.shape[1] < minw):
                print(f'[OUTSIZED] {img.shape[0]}x{img.shape[1]} {image2full[row[data_name]]}')
                # os.remove(image2full[row[data_name]])
                continue
            fnames.append(row[data_name])
            labels.append(union_rule.get(row[attr_names], row[attr_names]))

    print(f'total not found {not_exists}')
    lvalues, lcounts = np.unique(labels, return_counts=True)
    print('\n'.join([f'{a:.<15}{b}' for a, b in zip(lvalues, lcounts)]))
    counters = {x: 0 for x in lvalues}
    train, train_attr, test, test_attr = [], [], [], []
    for fn, atr in tqdm(zip(fnames, labels)):
        # print(atr)
        counters[atr] += 1
        if counters[atr] % pv == 0:
            test.append(fn)
            test_attr.append(atr)
        else:
            train.append(fn)
            train_attr.append(atr)

    trainfile = 'traindata/train.csv'
    df = pd.DataFrame(data={data_name: train, attr_names: train_attr}, dtype='uint8')
    df.to_csv(trainfile, index=False)
    testfile = 'traindata/test.csv'
    df = pd.DataFrame(data={data_name: test, attr_names: test_attr}, dtype='uint8')
    df.to_csv(testfile, index=False)
