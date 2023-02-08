import datetime
import os
import shutil
from queue import Queue
from random import randint
import yaml
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from numpy import count_nonzero, vstack, newaxis, argmax, where
from openvino.inference_engine.ie_api import IECore
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from dl_src.dnn import get_data_frame_from_folder, DNN
FAVORITES = 'favorites.txt'

opj = os.path.join
image_extensions = ['.png', '.jpg']
del_label = 'DELETE'
ie = IECore()
EMPTY = ''


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
        self.identifications = pd.DataFrame()
        self.path = ''
        self.labels = {}
        self._log = []
        self.main = pd.DataFrame()
        self.filter = None
        self.l1, self.l2, self.l2 = 0, 0, 0
        self.delete_path = ''
        self.items_to_check = {}
        self.images_folders = []
        self.dnn: DNN = None
        self.navigation = []


    def log(self, s):
        self._log.append(s)
        print(s)

    def load(self, path, loadsize=True, persons_reidentificator=None):
        self.path = path
        self.delete_path = opj(self.path, del_label)
        if not os.path.exists(self.delete_path):
            os.mkdir(self.delete_path)
        self.main['path'], self.main['name'] = make_list_of_files_by_extension(self.path)
        self.main = self.main.set_index('name')
        self.main['x'] = 0
        self.main['y'] = 0
        self.main['folder'] = [os.path.split(x)[1] for x in self.main['path']]
        self.images_folders = self.main['folder'].unique().tolist()
        self.dnn = DNN(ie_core=ie, xml=persons_reidentificator, device='CPU', num_requests=10, get_full=True)
        self.identifications = get_data_frame_from_folder(destination=path, dnn=self.dnn)
        print(self.identifications[self.dnn.xml_name])
        print(self.identifications.columns)

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
            self.main[lbl] = ''
            l_df = pd.read_csv(self.labels[lbl]['file'])

            lbls = [x.strip() for x in list(l_df.label.unique()) if str(x) not in (del_label)]
            lbls2 = []
            if 'wrong' in lbls:
                lbls2.append('wrong')
            lbls2 += sorted([x for x in sorted(lbls) if x != 'wrong'])
            self.log(f'loaded labels: {l_df.label.unique()}')
            print(f'[ {lbl} ] labels:{lbls}  labels2={lbls2}')
            self.log(f'         loaded: {pth} {nm}')
            self.labels[lbl]['values'] = lbls2.copy()
            l_df.set_index('image', inplace=True)
            l_df.rename(columns={'label': lbl}, inplace=True)
            f_delete = l_df[lbl] == 'DELETE'
            l_df.loc[f_delete, lbl] = ''
            self.main.update(l_df)
            idx_empty = self.main[lbl] == EMPTY
            print(f"          uniwu labels end {self.main[lbl].unique()} Ne={sum(idx_empty)}")
            items_to_check = opj(pth, 'files_to_check.csv')
            if os.path.exists(items_to_check):
                itc = pd.read_csv(items_to_check)
                self.items_to_check[lbl] = {'file': itc['filename'].tolist(), 'text': itc['choice'].tolist()}
            # store yaml (later read)
            settings_file = opj(pth, 'settings.yaml')
            if os.path.exists(settings_file):
                loaded_values = yaml.load(open(settings_file, 'r'), Loader=yaml.loader.SafeLoader)['values']
                # self.labels[lbl]['values'] += [x for x in loaded_values if x not in self.labels[lbl]['values']]
                self.labels[lbl]['values'] = loaded_values + [x for x in self.labels[lbl]['values'] if x not in loaded_values]
            yaml.dump(self.labels[lbl], open(settings_file, 'w'))
        self.filter = self.main.index
        # print(self.identifications.set_index('name'))
        self.main[self.dnn.xml_name] = None
        self.main.update(self.identifications.set_index('name'))
        self.main = self.main.sort_index()
        self.main_reid = vstack(self.main[self.dnn.xml_name].tolist())
        assert len(self.main_reid) == len(self.main)
        print(f'[integrity] main_reid={len(self.main_reid)} == main={len(self.main)}')
        self.main['favorites'] = False
        fav_name = os.path.join(path, FAVORITES)
        print(f'[{path}] look for {FAVORITES} file: {fav_name}')
        if os.path.exists(fav_name):
            with open(fav_name, 'r') as f:
                for name in f.readlines():
                    self.main.at[name, 'favorites'] = True
            print(f'[{path}] total favorites read {self.main["favorites"].sum()}')

        self.navigation = self.main.index.to_list()
        print(f'[integrity] main_reid={len(self.main_reid)} == main={len(self.main)}')

    def store_label(self, label):
        if label not in self.labels:
            return 'ok'
        shutil.copy(self.labels[label]['file'], opj(self.labels[label]['path'],
                                                    f'result_{datetime.datetime.now().isoformat()}.bak'))
        f = self.main[label] != ''
        try:
            pd.DataFrame({'image': self.main.index[f],
                          'label': self.main.loc[f, label]}).to_csv(self.labels[label]['file'], index=False)

        except Exception as e:
            return f'fail: {e}'

        fav_name = os.path.join(self.path, FAVORITES)
        with open(fav_name, 'w') as f:
            f.writelines( self.main.index[self.main['favorites'] == True].tolist())
            print(f'[{self.path}]stores favorites {(self.main["favorites"] == True).sum()}')

        return 'ok'


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

    def set_filter(self, label, value, seek_label, seek_only_clear='no', size='none', filter_text='none', fldr='all',
                   extraorder='True'):
        print(f'[FILTER] label="{label}" value:="{value}"  seek="{seek_label}" seek_only_clear"{seek_only_clear}" '
              f'size={size} filter_text={filter_text}')

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

        if fldr != 'all':
            f2 = self.main['folder'] == fldr
            self.filter = self.filter & f2
            print(f'[FILTER] selected folder="{fldr}" total folders = {len(self.images_folders)} '
                  f'files in folder = {count_nonzero(f2)}')

        if (seek_label not in ('none', '', None)) and (seek_only_clear != 'no'):
            f3 = self.main[seek_label] == ''

            self.filter = self.filter & f3
            print(f'[FILTER]  seek_label={seek_label} seek_only_clear={seek_only_clear} {count_nonzero(f3)} self.filter={count_nonzero(self.filter)}')

        return {'images': self.main.index[self.filter].to_list(),
                'label': self.labels[label] if is_label else [],
                'labels': list(self.labels.keys()),
                'values': self.labels[label]['values'] if is_label else [],
                'seekvalues': [] if seek_label == 'none' else self.labels[seek_label]['values'],
                'counts': self.calc_counts(seeklabel=seek_label, filtervalue=value, filterlabel=label),
                'text': itchk_text,
                'folders': self.images_folders}


    def return_filtered(self, label, seek_label, seek_value,
                        seek_only_clear, size, folder, favorites):
        print(f'[FILTER] label={label} seek_label:{seek_label}  seek_value={seek_value} seek_only_clear:{seek_only_clear} '
              f'size={size} folder={folder} favorites={favorites}')
        print(f'[integrity return_filtered] main_reid={len(self.main_reid)} == main={len(self.main)}')
        self.filter = self.main['x'] > 0
        print("[FILTER] init:", count_nonzero(self.filter))
        # print(f'label: {label} total:{count_nonzero(self.filter)}')

        if size:
            if size == 'up':
                self.filter &= (self.main['y'] >= self.l2) & (self.main['y'] <= self.l3)
            elif size == 'height':
                self.filter &= self.main['y'] >= self.l3
            elif size == 'small':
                self.filter &= (self.main['y'] >= self.l1) & (self.main['y'] <= self.l2)
            else:
                self.filter &= self.main['y'] <= self.l1
            print(f'[FILTER] size="{size}" total:{count_nonzero(self.filter)}')


        if seek_label and seek_value is not None:
            if seek_value == 'to_check':
                # print(f'items to check={self.items_to_check.keys()}')
                # print(f'{label} items to check={seek_label in self.items_to_check.keys()}')
                itchk_files = self.items_to_check.get(label, {}).get('file', [])
                itchk_files = self.items_to_check.get(label, {}).get('file', [])
                itchk_text = self.items_to_check.get(label, {}).get('text', [])
                # print(f"cont of items={len(itchk_files)}")
                f2 = self.main.index.isin(itchk_files)
            else:
                f2 = self.main[seek_label] == seek_value
            self.filter &= f2
            print(f'[FILTER] {seek_label}={seek_value} {count_nonzero(self.filter)}')


        if folder:
            self.filter &= self.main['folder'] == folder
            print(f'[FILTER] selected folder="{folder}" total folders = {len(self.images_folders)} '
                  f'files in folder = {count_nonzero(self.filter)}')

        if favorites == "yes":
            self.filter &= self.main['favorites'] == True
            print(f'[FILTER] favorites="{folder}" total:{count_nonzero(self.filter)}')
        elif favorites == "no":
            self.filter &= self.main['favorites'] == False
            print(f'[FILTER] favorites="{folder}" total:{count_nonzero(self.filter)}')

        if label is not None and seek_only_clear == 'yes':
            self.filter &= self.main[label] == ''
            print(f'[FILTER] only new="{seek_only_clear}" total:{count_nonzero(self.filter)}')

        # print(f"ZZZ={self.main.index[self.filter].to_list()}")
        # print(f"ZZZ={self.filter}")
        self.navigation = self.filter.index[self.filter].to_list()
        print(f'[FILTER] end count :{count_nonzero(self.filter)} = {len(self.navigation)}')

    def return_label_value_on_image(self, label, image_name):
        print(f'len main = {len(self.main)}  len reid = {len(self.main_reid)}')
        print(f'return_label_value_on_image(label={label}, im={image_name})')
        if (label in ('undefined', '', None, 'none')) or (image_name in ('undefined', '', None, 'none')):
            return {'label_value': '', 'icons': {'none': {'image': 'z', 'thr': -2}}}

        im_identificator = self.main.at[image_name, self.dnn.xml_name]
        image_ix = self.main.index.get_loc(image_name)
        print(f"image_ix:{image_ix}")
        cs = cosine_similarity(im_identificator[newaxis, :], self.main_reid).flatten()
        cs[image_ix] = -10
        out = []
        for iml in self.labels[label]['values']:
            f = self.main[label] == iml
            if np.any(f):
                print(f'len main = {len(self.main)}  len reid = {len(self.main_reid)}')
                # print(f"f={sum(f)} iml={iml} where f = {where(f)} argmax(cs[f])={argmax(cs[f])}")
                cs_v = where(f)[0][argmax(cs[f])]
                out += [{'lbl': iml, 'image': self.main.index[cs_v], 'thr': f"{float(cs[cs_v]):.3f}"}]
            else:
                out += [{'lbl': iml, 'image': 'none', 'thr': -1}]
        print(f"OUT:{out}")
        return out



    def get_label_value_on_image(self, label, im):
        print(f'get_label_value_on_image(label={label}, im={im})')
        if (label in ('undefined', '', None, 'none')) or (im in ('undefined', '', None, 'none')):
            return {'imlabel': '', 'icons': {'none': {'image': 'z', 'thr': -2}}}

        imlabel = self.main.at[im, label]
        out = {'imlabel': imlabel, 'icons': {}}

        im_reis = self.main.at[im, self.dnn.xml_name]
        cs = cosine_similarity(im_reis[newaxis, :], self.main_reid).flatten()
        out['icons'] = {}
        for iml in self.labels[label]['values']:
            f = self.main[label] == iml
            if np.any(f):
                cs_v = where(f)[0][argmax(cs[f])]
                out['icons'][iml] = {'image': self.main.index[cs_v], 'thr': float(cs[cs_v])}
            else:
                out['icons'][iml] = {'image': 'none', 'thr': -1}
        return out

    def calc_counts(self, seeklabel, filterlabel, filtervalue):
        filter_set = (filterlabel not in ('undefined', 'none', '', None, 'all')) and \
                     (filtervalue not in ('undefined', 'none', '', None, 'all'))
        if seeklabel in ('undefined', 'none', '', None, 'all'):
            return

        pd2 = pd.DataFrame({
            'groupes': self.main[seeklabel],
            seeklabel: [1] * self.main.shape[0]
        })
        grp = pd2.groupby(by='groupes').count()
        out = '<br>'.join([f'{"?" if k == "" else k:_<20} {v[seeklabel]}' for k, v in grp.iterrows()])
        return out

    def marked_image(self, im):

        # extracting frames
        im_name = opj(self.main.path[im], im)
        frame_row = cv2.imread(im_name)
        print(f'im_name:{frame_row.shape} {im_name} exists:{os.path.exists(im_name)}')
        h, w = frame_row.shape[:2]
        lw = max(h // 150, 1)
        # frame[:, 0:20] += 25
        # frame[:, w - 20:w] += 25
        # frame[0:20, :] += 25
        # frame[h - 20:h, :] += 25
        # frame_blanc = frame.copy()
        # frame = np.clip(frame_row[:, :] + 40, 0, 250)
        hsv = cv2.cvtColor(frame_row, cv2.COLOR_BGR2HSV)
        value = 100
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255 - value - 10) + value
        # hsv[:, :, 1] = np.clip(hsv[:, :, 1], 15, 255) - 10
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame[20: h-20, 20:w-20] = frame_row[20: h-20, 20:w-20]

        cv2.rectangle(frame, (+20, +20), (w - 20, h - 20), color=(0, 0, 255), thickness=lw)
        for hi in range(20, h - 20, 50):
            cv2.line(frame, (20, hi), (10, hi), color=(0, 0, 255), thickness=lw)
            cv2.line(frame, (w - 10, hi), (w - 20, hi), color=(0, 0, 255), thickness=lw)
        for wi in range(20, w - 20, 50):
            cv2.line(frame, (wi, 20), (wi, 10), color=(0, 0, 255), thickness=lw)
            cv2.line(frame, (wi, h - 10), (wi, h - 20), color=(0, 0, 255), thickness=lw)
        ret, jpeg = cv2.imencode('.jpg', frame)
        print(f"out:{frame.shape}")
        return jpeg.tobytes()


if __name__ == '__main__':
    d = Dbs()
    d.load('~/dataset/dataset_for_multilabel_classification')
    d.store_label('uniforme')
