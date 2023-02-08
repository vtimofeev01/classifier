import csv
import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T


class AttributesDataset:
    def __init__(self, dataset, echo=True):

        annotation_path = os.path.join(dataset, 'data.csv')
        with open(annotation_path) as f:
            reader = csv.DictReader(f)

            fld_names = reader.fieldnames[1:]
            datas = {fn: list() for fn in fld_names}
            for row in reader:
                for fn in fld_names:
                    datas[fn].append(row[fn])

        self.fld_names = fld_names
        print(f"[ANNOTATION] dataset: {dataset}")
        print(f"[ANNOTATION] attributes: {', '.join(self.fld_names)}")
        self.labels = {fn: np.unique(datas[fn]) for fn in fld_names}
        print(f"[ANNOTATION] labels: {self.labels}")
        self.labels_list = {fn: len(self.labels[fn]) for fn in fld_names}
        self.attr_num = len(self.labels_list)
        print(f"[ANNOTATION] len: {self.attr_num}")

        self.labels_id_to_name = {fn: dict(zip(range(len(self.labels[fn])), self.labels[fn])) for fn in fld_names}
        self.labels_name_to_id = {fn: dict(zip(self.labels[fn], range(len(self.labels[fn])))) for fn in fld_names}



class AttrDataset(data.Dataset):

    def __init__(self, data_path, csv_filename, attributes, transform=None, target_transform=None):

        self.root_path = data_path
        self.attr = attributes
        self.file_names = []
        self.label = []
        self.transform = transform
        self.target_transform = target_transform
        with open(os.path.join(self.root_path, csv_filename)) as f:
            reader = csv.DictReader(f)
            data_name = reader.fieldnames[0]
            attr_names = reader.fieldnames[1:]
            print(f'train data Attributes {csv_filename}: {", ".join(attr_names)}')
            self.attr_names = attr_names
            for row in reader:
                imfile = os.path.join(self.root_path, 'data', row[data_name])
                if not imfile or not os.path.exists(imfile):
                    continue
                self.file_names.append(row[data_name])
                self.label.append(
                    [self.attr.labels_name_to_id[attr][row[attr]] for attr in attr_names]
                )
            self.label = np.array(self.label)

    def __getitem__(self, index):

        # imgname, gt_label, imgidx = self.file_names[index], self.label[index], self.img_idx[index]
        imgname, gt_label = self.file_names[index], self.label[index]
        imgpath = os.path.join(self.root_path, 'data', imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.file_names)





def get_transform(width, height):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
