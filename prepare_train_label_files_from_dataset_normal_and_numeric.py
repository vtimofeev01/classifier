# read the annotations from the CSV file
import argparse
import csv
import json
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
union_rule = {'corrupted': 'wrong'}  # , 'non_person': 'wrong', 'non_clear': 'clear'}

# "clear",
# "multy",
# "non-clear",
# "non-person",
# "wrong"

export_dir = './dataset'
# pattern = '{:0>5}.png'

factor = .125
pv = int(1 / factor)

minw = 30 + 40
minh = 60 + 40

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--images_dir', type=str, help="Folder containing images described in CSV file")
    parser.add_argument('--attributes_file', type=str, help="Path to the file with attributes")
    args = parser.parse_args()

    annotation_path = args.attributes_file
    images_dir = args.images_dir

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
    data_dir = os.path.join(export_dir, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fnames = []
    heights = []
    widths = []

    with open(annotation_path) as f:
        reader = csv.DictReader(f)
        data_name = reader.fieldnames[0]
        attr_names = reader.fieldnames[1:]
        labels = {attr_name: [] for attr_name in attr_names}
        print(reader.fieldnames)
        print(data_name)
        print(attr_names, labels.keys())
        image2full = {a: os.path.join(b, a) for b, a in zip(*make_list_of_files(images_dir))}
        not_exists = 0
        number = 0
        for row in tqdm(reader):
            if (row[data_name] not in image2full) or (not os.path.exists(image2full[row[data_name]])):
                not_exists += 1
                continue
            img = Image.open(image2full[row[data_name]])
            width, height = img.size
            # img = img.crop((20, 20, width - 20, height - 20))
            if (width < minw) or (height < minh):
                # print(f'[OUTSIZED] {width}x{height} {image2full[row[data_name]]}')
                continue
            number += 1
            # file_name = f'{number:0>6}.png'
            fnames.append(row[data_name])
            heights.append(height)
            widths.append(width)
            for attr_name in attr_names:
                labels[attr_name].append(union_rule.get(row[attr_name], row[attr_name]))

            if img.mode == 'RGBA':
                img = img.convert('RGB')
                img.save(image2full[row[data_name]])

    nth = 8
    nth5 = 5
    df = pd.DataFrame(data={data_name: fnames, 'height': heights, 'width': widths, **labels}, dtype='uint8')
    not_all_columns = [x for x in df.columns if x not in ('height', 'width')]
    f1 = df.index % nth == 0
    f2 = df.index % nth == 1

    df[not_all_columns].to_csv(os.path.join(export_dir, 'data.csv'), index=False)
    df[f1][not_all_columns].to_csv(os.path.join(export_dir, 'test.csv'), index=False)
    df[f1 | f2][not_all_columns].to_csv(os.path.join(export_dir, 'val.csv'), index=False)
    df[~(f1 | f2)][not_all_columns].to_csv(os.path.join(export_dir, 'train.csv'), index=False)
    print(f'Total after testing:{len(widths)}')

    ratio = 1.5
    h_size = df['height'].median()
    w_size = h_size / 1.2
    print(f'median = {df["height"].median()}  mean = {df["height"].mean()}  split-H = {h_size} split-W = {w_size}')
    f_heigh_val = (df['height'] < h_size) & (df['width'] < w_size)
    f_heigh_base = (df['height'] < h_size * ratio) & (df['width'] < w_size * ratio)
    f_heigh_base2 = (df['height'] > h_size / ratio) | (df['width'] > w_size / ratio)

    for prefix, (ftest, ftrain) in zip(('small_', 'big_'), (
            (f_heigh_val, f_heigh_base),
            (~f_heigh_val, f_heigh_base2)
    )):
        f_test_it = ftest & ftest[df.index % nth5 == 0]


        f1 = ftrain[df.index % nth5 == 0] & ftest
        f2 = ftrain & ~f1

        df[f1][not_all_columns].to_csv(os.path.join(export_dir, f'partial_{prefix}val.csv'), index=False)
        df[f2][not_all_columns].to_csv(os.path.join(export_dir, f'partial_{prefix}train.csv'),
                                                        index=False)
        print(f' {prefix[:-1]} set: val:{sum(ftest)}/train:{sum(ftrain)} tests {sum(ftrain)} % {nth5} & {sum(ftest)} = {sum(f1)} '
              f'trains={sum(ftrain)} -> {sum(f2)}')

    # bymeric labels

    label_to_numeric = {}
    for df_l in df:
        if df_l in ('image', "height", 'width'):
            continue
        s = {y: x for x, y in enumerate(df[df_l].unique())}
        print(f'{df_l}={s}')
        label_to_numeric[df_l] = s
        df[df_l] = df[df_l].replace(s)
    with open(os.path.join(export_dir, 'label_to_number.json'), 'w') as f:
        json.dump(label_to_numeric, f)
    df.to_csv(os.path.join(export_dir, 'numeric_full_data_n.csv'), index=False)
    df[f1].to_csv(os.path.join(export_dir, 'numeric_full_test_n.csv'), index=False)
    df[f1 | f2].to_csv(os.path.join(export_dir, 'numeric_full_val_n.csv'), index=False)
    df[~(f1 | f2)].to_csv(os.path.join(export_dir, 'numeric_full_train_n.csv'), index=False)

    #
    #     df.to_csv(os.path.join(export_dir, 'data.csv'), index=False)
    #
    #     print(f'total not found {not_exists}')
    #     lvalues, lcounts = np.unique(labels, return_counts=True)
    #     print('\n'.join([f'{a:.<15}{b}' for a, b in zip(lvalues, lcounts)]))
    #     counters = {x: 0 for x in lvalues}
    #
    #     train, train_attr, test, test_attr = [], [], [], []
    #     val, val_attr = [], []
    #     for fn, atr in tqdm(zip(fnames, labels)):
    #         counters[atr] += 1
    #         pv_ = counters[atr] % pv
    #         if pv_ == 0:
    #             test.append(fn)
    #             test_attr.append(atr)
    #         elif pv_ == 1:
    #             val.append(fn)
    #             val_attr.append(atr)
    #         else:
    #             train.append(fn)
    #             train_attr.append(atr)
    #
    #     df = pd.DataFrame(data={data_name: train, attr_names: train_attr}, dtype='uint8')
    #     df.to_csv(os.path.join(export_dir, 'train.csv'), index=False)
    #     df = pd.DataFrame(data={data_name: test, attr_names: test_attr}, dtype='uint8')
    #     df.to_csv(os.path.join(export_dir, 'test.csv'), index=False)
    #     df = pd.DataFrame(data={data_name: val, attr_names: val_attr}, dtype='uint8')
    #     df.to_csv(os.path.join(export_dir, 'val.csv'), index=False)
    #
    # #
