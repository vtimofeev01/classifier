import datetime
import os
import shutil
from queue import Queue
from random import randint

import cv2
import pandas as pd
from flask import Response
from numpy import count_nonzero
import numpy as np
from tqdm import tqdm
from PIL import Image
opj = os.path.join
image_extensions = ['.png', '.jpg']
del_label = 'DELETE'


# TODO Make size output

def make_list_of_files_by_extension(source, extensions=None, escape_copies=True):
    if extensions is None:
        extensions = ('.jpg', '.png')
    ck = set()
    q = Queue()
    q.put(source)
    paths = []
    file_names = []
    while not q.empty():
        v = q.get()
        if os.path.isdir(v):
            for vs in sorted(os.listdir(v)):
                q.put(os.path.join(v, vs))
        elif os.path.splitext(v)[1] in extensions:
            path, name = os.path.split(v)
            paths.append(path)
            if name in ck:
                new_name = f'copy_{randint(0, 9999):0>4}_{name}'
                shutil.move(opj(path, name), opj(path, new_name))
                print(f'coping {name} -> {new_name}')
                ck.add(new_name)
            else:
                file_names.append(name)
                ck.add(name)

    return paths, file_names


def make_list_of_files_by_name(source, name):
    q = Queue()
    q.put(source)
    paths = []
    file_names = []
    while not q.empty():
        v = q.get()
        if os.path.isdir(v):
            for vs in sorted(os.listdir(v)):
                q.put(os.path.join(v, vs))
        else:
            path, n = os.path.split(v)
            if n == name:
                paths.append(path)
                file_names.append(n)
    return paths, file_names


class Dbs:

    def __init__(self):
        self.path = ''
        self.labels = {}
        self._log = []
        self.main = pd.DataFrame()
        self.filter = None
        self.l1, self.l2, self.l2 = 0, 0, 0
        self.delete_path = ''
        self.items_to_check = {}

    def log(self, s):
        self._log.append(s)
        print(s)

    def load(self, path, loadsize=True):
        self.path = path
        self.delete_path = opj(self.path, del_label)
        if not os.path.exists(self.delete_path):
            os.mkdir(self.delete_path)
        self.main['path'], self.main['name'] = make_list_of_files_by_extension(self.path)
        self.main = self.main.set_index('name')
        self.main['x'] = 0
        self.main['y'] = 0
        if loadsize:
            fl = [opj(p, f) for p, f in zip(self.main['path'], self.main.index)]
            fl_sz = np.array([Image.open(x).size for x in tqdm(fl)])
            self.main['x'] = fl_sz[:, 0]
            self.main['y'] = fl_sz[:, 1]
            self.l1 = np.percentile(fl_sz[:, 1], 40)
            self.l2 = np.percentile(fl_sz[:, 1], 60)
            self.l3 = np.percentile(fl_sz[:, 1], 80)
            print(f'percentiles {self.l1}, {self.l2}')
        for pth, nm in zip(*make_list_of_files_by_name(self.path, 'results.csv')):
            lbl = os.path.split(pth)[1]
            self.labels[lbl] = {'file': os.path.join(pth, nm), 'path': pth, 'nm': nm}
            self.log(f'loaded: {pth} {nm}')
            self.main[lbl] = ''
            l_df = pd.read_csv(self.labels[lbl]['file'])
            lbls = [x.strip() for x in list(l_df.label.unique()) if x not in (del_label)]
            lbls2 = []
            if 'wrong' in lbls:
                lbls2.append('wrong')
            lbls2 += sorted([x for x in sorted(lbls) if x != 'wrong'])
            print(f'lbl={lbl} lbls={lbls}  lbls2={lbls2}')
            self.labels[lbl]['values'] = lbls2.copy()
            l_df.set_index('image', inplace=True)
            l_df.rename(columns={'label': lbl}, inplace=True)
            self.main.update(l_df)
            print(self.main[lbl].unique())
            items_to_check = opj(pth, 'files_to_check.csv')
            if os.path.exists(items_to_check):
                itc = pd.read_csv(items_to_check)
                self.items_to_check[lbl] = {'file': itc['filename'].tolist(), 'text': itc['choice'].tolist()}
        self.filter = self.main.index

    def store_label(self, label):
        if label not in self.labels:
            return 'ok'
        shutil.copy(self.labels[label]['file'], opj(self.labels[label]['path'],
                                                    f'result_{datetime.datetime.now().isoformat()}.bak'))
        f = self.main[label] != ''
        try:
            pd.DataFrame({'image': self.main.index[f],
                          'label': self.main.loc[f, label]}).to_csv(self.labels[label]['file'], index=False)
            return 'ok'
        except Exception as e:
            return f'fail: {e}'

    def image(self, im):
        # print(f'image: {self.main.path[im]} {im}')
        if im in ('undefined', 'none', None):
            return None
        print(f'\npath:{self.main.path[im]}')
        print(f'file:{im}')
        return opj(self.main.path[im], im)

    def set_value(self, im, label, code):
        try:
            if code == del_label:
                f_file = opj(self.main.at[im, 'path'], im)
                to_file = opj(self.delete_path, im)
                print(f'removing {f_file} -> {to_file}')
                shutil.move(f_file, to_file)
                self.main.at[im, 'path'] = self.delete_path
                for _label in self.labels.keys():
                    self.main.at[im, _label] = del_label

            else:
                self.main.at[im, label] = code
            return {'res': 'ok'}
        except Exception as e:
            return {'res': f'fail: {e}'}

    def set_filter(self, label, value, seek_label, seek_only_clear='no', size='none', filter_text='none'):
        print(f'filter: label: "{label}" value: "{value}"  seek "{seek_label}"')

        is_label = label not in ('none', '', None, 'all')
        is_value = value not in ('none', '', None, 'all')
        if is_label:
            self.filter = self.main[label] != ''
        else:
            self.filter = self.main.index != ''
        print(f'label {is_label} {label} {count_nonzero(self.filter)}')

        if size != 'none':
            if size == 'up':
                fs = (self.main['y'] >= self.l2) & (self.main['y'] <= self.l3)
            elif size == 'height':
                fs = self.main['y'] >= self.l3
            elif size == 'small':
                fs = (self.main['y'] >= self.l1) & (self.main['y'] <= self.l2)
            else:
                fs = self.main['y'] <= self.l1
            self.filter = self.filter & fs
            print(f'filtered on size: {size} -> {count_nonzero(self.filter)}')

        itchk_text = []
        if is_label and is_value:
            if value == 'to_check':
                print(f'items to check={self.items_to_check.keys()}')
                print(f'{label} items to check={label in self.items_to_check.keys()}')
                itchk_files = self.items_to_check.get(label, {}).get('file', [])
                itchk_text = self.items_to_check.get(label, {}).get('text', [])
                print(f"cont of items={len(itchk_files)}")
                if filter_text == 'none':
                    f2 = self.main.index.isin(itchk_files)
                    # files_ = itchk_text
                else:
                    files_ = [fl for fl, txt in zip(itchk_files, itchk_text) if txt == filter_text]
                    itchk_text = [txt for fl, txt in zip(itchk_files, itchk_text) if txt == filter_text]
                    f2 = self.main.index.isin(files_)
            else:
                f2 = self.main[label] == value

            print(f'{label}={value} {count_nonzero(f2)}')
            self.filter = self.filter & f2
            print(f'label {is_label} {label}={value} {count_nonzero(self.filter)}')

        if (seek_label not in ('none', '', None)) and (seek_only_clear != 'no'):
            f3 = self.main[seek_label] == ''
            print(f'seek_label={seek_label} seek_only_clear={seek_only_clear} {count_nonzero(f3)}')
            self.filter = self.filter & f3
            print(f' self.filter={count_nonzero(self.filter)}')

        return {'images': self.main.index[self.filter].to_list(),
                'label': self.labels[label] if is_label else [],
                'labels': list(self.labels.keys()),
                'values': self.labels[label]['values'] if is_label else [],
                'seekvalues': [] if seek_label == 'none' else self.labels[seek_label]['values'],
                'counts': self.calc_counts(seeklabel=seek_label, filtervalue=value, filterlabel=label),
                'text': itchk_text}

    def get_label_value_on_image(self, label, im):
        print(f'get_label_value_on_image(label={label}, im={im})')
        if label in ('undefined', '', None, 'none'):
            return {'imlabel': ''}
        if im in ('undefined', '', None, 'none'):
            return {'imlabel': ''}
        return {'imlabel': self.main.at[im, label]}

    def calc_counts(self, seeklabel, filterlabel, filtervalue):
        filter_set = (filterlabel not in ('undefined', 'none', '', None, 'all')) and \
                     (filtervalue not in ('undefined', 'none', '', None, 'all'))
        if seeklabel in ('undefined', 'none', '', None, 'all'):
            return

        pd2 = pd.DataFrame({
            'groupes': self.main[seeklabel],
            seeklabel: [1] * self.main.shape[0]
        })
        # if filter_set:
        #     pd2[f'{filterlabel}=={filtervalue}'] = self.main[filterlabel] == filtervalue

        # print(pd2)
        grp = pd2.groupby(by='groupes').count()
        out = '<br>'.join([f'{"?" if k == "" else k:_<20} {v[seeklabel]}' for k, v in grp.iterrows()])
        return out

    def marked_image(self, im):

        # extracting frames
        im_name = opj(self.main.path[im], im)
        frame = cv2.imread(im_name)
        h, w = frame.shape[:2]
        lw = max(h // 150, 1)
        cv2.rectangle(frame, (+20, +20), (w - 20, h - 20), color=(0, 0, 255), thickness=lw)
        for hi in range(20, h - 20, 50):
            cv2.line(frame, (20, hi), (10, hi), color=(0, 0, 255), thickness=lw)
            cv2.line(frame, (w - 10, hi), (w - 20, hi), color=(0, 0, 255), thickness=lw)
        for wi in range(20, w - 20, 50):
            cv2.line(frame, (wi, 20), (wi, 10), color=(0, 0, 255), thickness=lw)
            cv2.line(frame, (wi, h - 10), (wi, h - 20), color=(0, 0, 255), thickness=lw)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


if __name__ == '__main__':
    d = Dbs()
    d.load('/home/imt/dataset/dataset_for_multilabel_classification')
    d.store_label('uniforme')
