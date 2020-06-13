import csv
import os
from queue import Queue

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score

mean = [0.485, 0.456, 0.406]  # [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]  # [0.229, 0.224, 0.225]


def make_list_of_files(source, extensions=None):
    if extensions is None:
        extensions = ('.jpg', '.png')
    q = Queue()
    q.put(source)
    paths = []
    fnames = []
    while not q.empty():
        v = q.get()
        if os.path.isdir(v):
            for vs in sorted(os.listdir(v)):
                q.put(os.path.join(v, vs))
        elif os.path.splitext(v)[1] in extensions:
            path, name = os.path.split(v)
            paths.append(path)
            fnames.append(name)
    return paths, fnames


class AttributesDataset:
    def __init__(self, annotation_path, echo=True):

        datas = {}
        fld_names = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)

            fld_names = reader.fieldnames[1:]
            datas = {fn: list() for fn in fld_names}
            for row in reader:
                for fn in fld_names:
                    datas[fn].append(row[fn])

        self.fld_names = fld_names
        print(f"[ANNOTATION] attributes: {', '.join(self.fld_names)}")
        self.labels = {fn: np.unique(datas[fn]) for fn in fld_names}
        print(f"[ANNOTATION] labels: {self.labels}")
        self.num_labels = {fn: len(self.labels[fn]) for fn in fld_names}
        print(f"[ANNOTATION] len: {self.num_labels}")

        self.labels_id_to_name = {fn: dict(zip(range(len(self.labels[fn])), self.labels[fn])) for fn in fld_names}
        self.labels_name_to_id = {fn: dict(zip(self.labels[fn], range(len(self.labels[fn])))) for fn in fld_names}


class CSVDataset(Dataset):
    def __init__(self, annotation_path, images_dir, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.labels = {}

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            data_name = reader.fieldnames[0]
            attr_names = reader.fieldnames[1:]
            print(f'train data Attributes {annotation_path}: {", ".join(attr_names)}')
            self.attr_names = attr_names
            self.labels = {fn: list() for fn in self.attr_names}
            image2full = {a: os.path.join(b, a) for b, a in zip(*make_list_of_files(images_dir))}

            for row in reader:
                # print(row)
                # imfile = os.path.join(images_dir, row[data_name])
                imfile = image2full.get(row[data_name], False)
                if not imfile or not os.path.exists(imfile):
                    # print('zopa')
                    continue
                self.data.append(imfile)
                for attr in attr_names:
                    self.labels[attr].append(self.attr.labels_name_to_id[attr][row[attr]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]
        # print(img_path)

        # read image
        img = Image.open(img_path)

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'img_path': img_path,
            'labels': {an: self.labels[an][idx] for an in self.attr_names}
        }
        return dict_data
