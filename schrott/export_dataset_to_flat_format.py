
# read the annotations from the CSV file
import csv
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from easydict import EasyDict
from tqdm import tqdm
import scipy.io
import PIL
from scipy.io import loadmat
from dl_src.dataset import make_list_of_files

annotation_path = '/home/imt/dataset/dataset_for_multilabel_classification/person/results.csv'
images_dir = '/home/imt/dataset/dataset_for_multilabel_classification'
union_rule = {'corrupted': 'wrong', 'non_person': 'wrong'}
export_dir = '../dataset'
pattern = '{:0>5}.png'

factor = .125
pv = int(1 / factor)

minw = 30
minh = 60


if __name__ == '__main__':
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
    data_dir = os.path.join(export_dir, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fnames, labels = [], []
    with open(annotation_path) as f:
        reader = csv.DictReader(f)
        data_name = reader.fieldnames[0]
        attr_names = reader.fieldnames[1]
        image2full = {a: os.path.join(b, a) for b, a in zip(*make_list_of_files(images_dir))}
        not_exists = 0
        number = 0
        for row in tqdm(reader):
            if (row[data_name] not in image2full) or (not os.path.exists(image2full[row[data_name]])):
                not_exists += 1
                continue
            img = Image.open(image2full[row[data_name]])
            width, height = img.size
            img = img.crop((20, 20, width - 20, height - 20))
            if (width < minh) or (height < minw):
                print(f'[OUTSIZED] {width}x{height} {image2full[row[data_name]]}')
                continue
            number += 1
            file_name = f'{number:0>5}.png'
            fnames.append(file_name)
            labels.append(union_rule.get(row[attr_names], row[attr_names]))

            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(os.path.join(data_dir, file_name))

        df = pd.DataFrame(data={data_name: fnames, attr_names: labels}, dtype='uint8')
        df.to_csv(os.path.join(export_dir, 'data.csv'), index=False)

        print(f'total not found {not_exists}')
        lvalues, lcounts = np.unique(labels, return_counts=True)
        print('\n'.join([f'{a:.<15}{b}' for a, b in zip(lvalues, lcounts)]))
        counters = {x: 0 for x in lvalues}

        train, train_attr, test, test_attr = [], [], [], []
        val, val_attr = [], []
        for fn, atr in tqdm(zip(fnames, labels)):
            counters[atr] += 1
            pv_ = counters[atr] % pv
            if pv_ == 0:
                test.append(fn)
                test_attr.append(atr)
            elif pv_ == 1:
                val.append(fn)
                val_attr.append(atr)
            else:
                train.append(fn)
                train_attr.append(atr)

        df = pd.DataFrame(data={data_name: train, attr_names: train_attr}, dtype='uint8')
        df.to_csv(os.path.join(export_dir, 'train.csv'), index=False)
        df = pd.DataFrame(data={data_name: test, attr_names: test_attr}, dtype='uint8')
        df.to_csv(os.path.join(export_dir, 'test.csv'), index=False)
        df = pd.DataFrame(data={data_name: val, attr_names: val_attr}, dtype='uint8')
        df.to_csv(os.path.join(export_dir, 'val.csv'), index=False)

    #