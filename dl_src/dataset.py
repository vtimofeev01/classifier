import csv
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


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
        print(f"Annotated attributes: {', '.join(self.fld_names)}")
        self.labels = {fn: np.unique(datas[fn]) for fn in fld_names}
        self.num_labels = {fn: len(self.labels[fn]) for fn in fld_names}

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

        # df = pd.read_csv(annotation_path)
        # df = df.fillna('')
        # loaded_labels = df.to_dict('split')['data']


        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            data_name = reader.fieldnames[0]
            attr_names = reader.fieldnames[1:]
            print(f'train data Attributes {annotation_path}: {", ".join(attr_names)}')
            self.attr_names = attr_names
            self.labels = {fn: list() for fn in self.attr_names}
            for row in reader:
                # print(row)
                self.data.append(os.path.join(images_dir, row[data_name]))
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
