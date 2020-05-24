import datetime
import logging
import os
import shutil
from collections import OrderedDict
from queue import Queue

import cv2
import pandas as pd

from math import log10
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from numpy import unique

from .view import CViewer
import collections

LOGGER = logging.getLogger(__name__)
opj = os.path.join
IMEXTS = ['.png', '.jpg']

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


def parse_dataset_to_navigation_dict(datasetname, datasetfile, patternfiles):
    out = {}
    if datasetfile and os.path.isfile(datasetfile):
        df = pd.read_csv(datasetfile)
        df = df.fillna('')
        ds_imgs, ds_lbls = [], []
        for im, lbl in df.to_dict('split')['data']:
            if im not in patternfiles:
                continue
            ds_imgs.append(im)
            ds_lbls.append(lbl)
        out[f'{datasetname}: All'] = {'data':[patternfiles.index(x) for x in ds_imgs], 'pos': 0}
        for ulbl in unique(ds_lbls):
            out[f'{datasetname}: {ulbl}'] = {'data':[patternfiles.index(x) for ix, x in enumerate(ds_imgs) if ds_lbls[ix] == ulbl], 'pos': 0}
    return out

class CApp(CViewer):
    def __init__(self, imgdir, label: str = None, label_values: str = None):
        super().__init__(title=f'classifier for {label}')
        outfile = "results.csv"
        history = "results.csv"

        assert label is not None

        self.fv_bv = collections.deque([], 10)
        self.fv_bv_index = -1

        self.label = label
        self.label_dir = os.path.join(imgdir, self.label)
        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)
        self.outfile = opj(self.label_dir, outfile)

        self.history_file = opj(self.label_dir, history)
        self.loaded_labels = {}

        # self.images = [x for x in os.listdir(os.path.abspath(imgdir)) if os.path.splitext(x)[1] in IMEXTS]
        # self.path = [imgdir] * len(self.images)
        pth, imgs = make_list_of_files(imgdir)
        self.path, self.images = [], []
        for pt, img in zip(pth, imgs):
            if img in self.images:
                continue
            self.path.append(pt)
            self.images.append(img)
        print(f'[LOAD] loaded images {len(self.images)}')
        # doubles = [a for a, b in zip(*unique(self.images, return_counts=True)) if b > 1]
        # print(f' len of doubles = {len(doubles)}')
        self.label_values = label_values.split(',') if label_values is not None else ['True', 'False']
        # self.image_index = 0
        self.current_labels = {img: None for img in self.images}
        print(f'[LOAD] current_labels init {len(self.current_labels)}')

        self._load_history()
        self.label_values = self.sort_labels(self.label_values)

        self.len_of_label_values = len(self.label_values)

        self.cur_time = datetime.datetime.now()
        self.av_sec = None
        self.av_count = 10

        self.file_to_check = opj(self.label_dir, 'files_to_check.csv')
        self.has_to_check = os.path.exists(self.file_to_check)
        self.item_to_check_i = 0
        if self.has_to_check:
            ftck = pd.read_csv(self.file_to_check)
            ftck = ftck.fillna('')
            self.list_items_to_check = [x for x in ftck.to_dict('split')['data'] if x[0] in self.images]
            self.help_label_text += f'\n\nLoaded items to check:{len(self.list_items_to_check)}' \
                                    '\n< q   e > previous/next item'
            self.Help_label.setText(self.help_label_text)
            self.images_2_check = [x[0] for x in self.list_items_to_check]
        else:
            self.list_items_to_check = []
            self.images_2_check = []

        all = list(range(len(self.images)))
        self.navigations = {'ALL': {'data': all, 'pos': 0},
                            'items 2 check': {'data': [self.images.index(x) for x in self.images_2_check], 'pos': 0}
                            }
        for outer_label in [x for x in os.listdir(imgdir) if os.path.isdir(opj(imgdir, x))]:
            outer_label_dataset = opj(imgdir, outer_label, 'results.csv')
            if os.path.exists(outer_label_dataset):
                self.navigations.update(
                    parse_dataset_to_navigation_dict(outer_label,
                                                     outer_label_dataset,
                                                     self.images))
        self.navigationtextes = [f'{a:.<35} {len(b["data"])}' for a, b in self.navigations.items()]
        self.navi_labels = sorted(self.navigations.keys())
        self.filter_list.addItems(self.navi_labels)

        self.filter_list.activated[str].connect(self.set_other_items_filter)
        self.main_index = 'ALL'

        print('\n'.join(self.navigationtextes))
        self._render_window()
        self._goto_next_unlabeled_image()


    def set_other_items_filter(self, name):
        self.main_index = name
        print(f'Navigation on dataset is set on {self.main_index}')
        # self.filter_list.clearFocus()
        self.fv_bv_index = 0
        self.fv_bv = []
        self._render_window()

    def sort_labels(self, l):
        il = sorted(l)
        if 'wrong' in il:
            ix = il.index('wrong')
            il.pop(ix)
            il = ['wrong'] + il
        return il

    def image_file_name(self, ix):
        return opj(self.path[ix], self.images[ix])

    def _load_history(self):
        if self.history_file and os.path.isfile(self.history_file):
            df = pd.read_csv(self.history_file)
            df = df.fillna('')
            loaded_labels = df.to_dict('split')['data']
            self.loaded_labels = {img: label for img, label in loaded_labels if (label not in [None, ''])
                                  and (img in self.images)}
            print(f'[history] history loaded {len(self.loaded_labels)} from {len(loaded_labels)} - {self.history_file}')
            loaded_label_values = {label: 0 for img, label in self.loaded_labels.items()}.keys()
            print('[history] labels: ', list(loaded_label_values))
            if len(loaded_labels) > 0:
                if len(self.label_values) == 2 and set(self.label_values) == {'False', 'True'}:
                    self.label_values = []

            self.label_values += [x for x in loaded_label_values if x not in self.label_values + [None, '']]
            print('[history] updated labels: ', list(self.label_values))
            self.current_labels.update(self.loaded_labels)
            print(f'[history] current_labels after update: {len(self.current_labels)}')
            # self._goto_next_unlabeled_image()

    def export(self):
        orderdict = OrderedDict(sorted([[a, b] for a, b in self.current_labels.items() if b is not None],
                                       key=lambda x: x[0]))
        df = pd.DataFrame(data={'image': list(orderdict.keys()), 'label': list(orderdict.values())}, dtype='uint8')
        df.to_csv(self.outfile, index=False)
        QMessageBox.information(self, 'Information', 'Export label result {}'.format(self.outfile))

    def _render_window(self, t=None):
        # assert 0 <= self.image_index < len(self.images)
        pos = self.navigations[self.main_index]['pos']
        # print(f'filter={self.main_index} pos = {pos} len={len(self.navigations[self.main_index]["data"])}')
        cv_im = cv2.imread(self.image_file_name(self.navigations[self.main_index]['data'][pos]))
        h, w = cv_im.shape[:2]
        BORDER = 20
        cv2.rectangle(cv_im, (BORDER, BORDER), (w - BORDER, h - BORDER), (0, 0, 255), 1)

        image = QImage(cv_im, cv_im.shape[1], cv_im.shape[0], cv_im.shape[1] * 3, QImage.Format_RGB888)
        image = QPixmap(image)
        # image = QPixmap(self.image_file_name(self.image_index))

        image = image.scaled(self.imW, self.imH, Qt.KeepAspectRatio)
        self.label_image.setPixmap(image)
        self.label_image.resize(self.imW, self.imH)
        self._render_status(t=t)
        #
        self.show()

    def _render_status(self, t=None):
        if t is None:
            pos = self.navigations[self.main_index]['pos']
            image_name = self.image_file_name(self.navigations[self.main_index]['data'][pos])
            labeled = self.current_labels[self.images[self.navigations[self.main_index]['data'][pos]]]
            pad_zero = int(log10(len(self.images))) + 1
            if labeled is not None:
                self.label_status.setText('({}/{}) {} => Labeled as {}'.format(
                    str(self.navigations[self.main_index]['data'][pos] + 1).zfill(pad_zero), len(self.images), image_name, labeled
                ))
            else:
                self.label_status.setText('({}/{}) {}'.format(
                    str(self.navigations[self.main_index]['data'][pos] + 1).zfill(pad_zero), len(self.images), image_name
                ))
        else:
            self.label_status.setText(t)

        self._redraw_labels_selection_list()

    def _prev(self):

        pos = self.navigations[self.main_index]['pos']
        pos -= 1
        if pos == 0:
            pos = len(self.navigations[self.main_index]['data']) - 1
        self.navigations[self.main_index]['pos'] = pos
        if pos not in self.fv_bv:
            self.fv_bv.append(self.navigations[self.main_index]['pos'])
        self.red_mark.setText('')
        # print(f'pos={pos} #={self.navigations[self.main_index]["pos"]} self.main_index={self.main_index}')

    def _next(self):

        pos = self.navigations[self.main_index]['pos']
        pos += 1
        if pos == len(self.navigations[self.main_index]['data']):
            pos = 0
        self.navigations[self.main_index]['pos'] = pos
        if pos not in self.fv_bv:
            self.fv_bv.append(self.navigations[self.main_index]['pos'])
        self.red_mark.setText('')
        print(f'pos={pos} #={self.navigations[self.main_index]["pos"]} self.main_index={self.main_index}')


    # def _goto_prev_unlabeled_image(self):
    #     self.red_mark.setText('')
    #     if self.image_index == 0:
    #         QMessageBox.warning(self, 'Warning', 'Reach the top of images')
    #     else:
    #         prev_image_index = self.image_index
    #         for idx in range(self.image_index - 1, 0, -1):
    #             if self.current_labels[self.images[idx]] is None:
    #                 prev_image_index = idx
    #                 break
    #         if prev_image_index == self.image_index:
    #             QMessageBox.information(self, 'Information', 'No more prev unlabeled image')
    #         else:
    #             self.image_index = prev_image_index
    #             self._render_window()
    #     if self.image_index not in self.fv_bv:
    #         self.fv_bv.append(self.image_index)

    def _goto_next_unlabeled_image(self):
        self.red_mark.setText('')
        pos = self.navigations[self.main_index]['pos']
        imix = pos
        for p in self.navigations[self.main_index]['data'][pos:] + self.navigations[self.main_index]['data'][:pos]:
            imix = self.navigations[self.main_index]['data'][p]
            imnm = self.images[imix]
            if self.current_labels[imnm] is None:
                break
        if imix == pos:
            QMessageBox.information(self, 'Information', 'No more next unlabeled image')
        self.navigations[self.main_index]['pos'] = imix
        if imix not in self.fv_bv:
            self.fv_bv.append(self.navigations[self.main_index]['pos'])

    def keyPressEvent(self, event):

        new_time = datetime.datetime.now()
        td = new_time - self.cur_time
        self.cur_time = new_time
        if self.av_sec is None:
            self.av_sec = td.seconds
        else:
            self.av_sec = (self.av_sec * self.av_count + td.seconds) / (self.av_count + 1)
            t = f'\n\naverage tempoo {self.av_sec:.1f} sec / image'
            self.Help_label.setText(self.help_label_text + t)

        kval = event.key()
        if kval in [Qt.Key_Left]:
            self.label_selection_list.setStyleSheet("color: black;")
            # self.fv_bv_index = 0
            # self.fv_bv = []
            self._prev()

        elif kval == Qt.Key_Down and len(self.fv_bv) > 0:
            self.fv_bv_index = len(self.fv_bv) - 1 if self.fv_bv_index < 1 else self.fv_bv_index - 1
            fv_bv_val = self.fv_bv[self.fv_bv_index]
            if type(fv_bv_val) == int and fv_bv_val < len(self.images):
                self.navigations[self.main_index]['pos'] = fv_bv_val
                self._render_window()
                self.red_mark.setText('')

        elif kval == Qt.Key_Up and len(self.fv_bv) > 0:
            self.fv_bv_index = 0 if self.fv_bv_index > len(self.fv_bv) - 2 else self.fv_bv_index + 1
            fv_bv_val = self.fv_bv[self.fv_bv_index]
            if type(fv_bv_val) == int and fv_bv_val < len(self.images):
                self.navigations[self.main_index]['pos'] = fv_bv_val
                self._render_window()
                self.red_mark.setText('')

        elif kval in [Qt.Key_Right]:
            self.label_selection_list.setStyleSheet("color: black;")
            # self.fv_bv_index = 0
            # self.fv_bv = []
            self._next()
        elif kval == Qt.Key_Space:
            self.label_selection_list.setStyleSheet("color: black;")
            self._goto_next_unlabeled_image()
        elif kval == Qt.Key_S:
            self.export()
        elif kval in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8,
                      Qt.Key_9]:
            if kval - 48 >= self.len_of_label_values:
                self.label_selection_list.setStyleSheet("color: red;")
            else:
                self.label_selection_list.setStyleSheet("color: black;")
                lv_index = min(kval - 48, self.len_of_label_values - 1)
                pos = self.navigations[self.main_index]['pos']
                imname = self.images[self.navigations[self.main_index]['data'][pos]]
                self.current_labels[imname] = self.label_values[lv_index]
            self._render_window()

        elif kval == Qt.Key_Delete:
            pos = self.navigations[self.main_index]['pos']
            oldname = self.image_file_name(self.navigations[self.main_index]['data'][pos])
            self.images[self.navigations[self.main_index]['data'][pos]] = 'DELETED_' + self.images[self.navigations[self.main_index]['data'][pos]]
            shutil.move(oldname, self.image_file_name(self.navigations[self.main_index]['data'][pos]))
        else:
            print('You Clicked {} but nothing happened...'.format(event.key()))
        self._render_window()

    def _redraw_labels_selection_list(self, selected=-1):
        lst = []
        imcode = ''
        t_len = 0
        for ii, lv in enumerate(self.label_values):
            pos = self.navigations[self.main_index]['pos']
            imcode = self.current_labels[self.images[self.navigations[self.main_index]['data'][pos]]]
            mark = ' ' if imcode is None or ii != self.label_values.index(imcode) else '*'
            lv_n = len([v for z, v in self.current_labels.items() if v == lv])
            t_len += lv_n
            lst.append(f'{mark}{ii}  {lv:.<15} {lv_n}')
        lst.append(f'\n{"-"*25}\n    total ......... {t_len}    \n    left .......... {len(self.images) - t_len}')
        t = '\n'.join(lst)
        self.red_mark.setText(imcode)
        self.label_selection_list.setText(t)
