import datetime
import logging
import os
from collections import OrderedDict

import pandas as pd

from math import log10
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from .view import CViewer

LOGGER = logging.getLogger(__name__)
opj = os.path.join
IMEXTS = ['.pmg', '.jpg']


class CApp(CViewer):
    def __init__(self, imgdir, label: str = None, label_values: str = None):
        super().__init__(title=f'classifier for {label}')
        outfile = "results.csv"
        history = "results.csv"

        assert label is not None
        self.label = label
        self.label_dir = os.path.join(imgdir, self.label)
        if not os.path.exists(self.label_dir):
            os.mkdir(self.label_dir)
        self.outfile = opj(self.label_dir, outfile)

        self.history_file = opj(self.label_dir, history)
        self.loaded_labels = {}

        self.images = [x for x in os.listdir(os.path.abspath(imgdir)) if os.path.splitext(x)[1] in IMEXTS]
        self.path = [imgdir] * len(self.images)

        self.label_values = label_values.split(',') if label_values is not None else ['True, False']
        self.image_index = 0
        self.current_labels = {img: None for img in self.images}

        self._load_history()
        self._render_window()
        self.len_of_label_values = len(self.label_values)

        self.cur_time = datetime.datetime.now()
        self.av_sec = None
        self.av_count = 30

    def image_file_name(self, ix):
        return opj(self.path[ix], self.images[ix])

    def _load_history(self):
        if self.history_file and os.path.isfile(self.history_file):
            df = pd.read_csv(self.history_file)
            df = df.fillna('')
            loaded_labels = df.to_dict('split')['data']
            self.loaded_labels = {img: label for img, label in loaded_labels if label not in [None, '']}
            print(f'Load history file N={len(self.loaded_labels)} - {self.history_file}')
            loaded_label_values = {label: 0 for img, label in self.loaded_labels.items()}.keys()
            print('Loaded labels: ', list(loaded_label_values))
            self.label_values += [x for x in loaded_label_values if x not in self.label_values + [None, '']]
            self.current_labels.update(loaded_labels)
            self._goto_next_unlabeled_image()

    def export(self):
        orderdict = OrderedDict(sorted([[a, b] for a, b in self.current_labels.items() if b is not None],
                                       key=lambda x: x[0]))
        df = pd.DataFrame(data={'image': list(orderdict.keys()), 'label': list(orderdict.values())}, dtype='uint8')
        df.to_csv(self.outfile, index=False)
        QMessageBox.information(self, 'Information', 'Export label result {}'.format(self.outfile))

    def _render_window(self):
        assert 0 <= self.image_index < len(self.images)
        image = QPixmap(self.image_file_name(self.image_index))
        image = image.scaled(self.imW, self.imH, Qt.KeepAspectRatio)
        self.label_image.setPixmap(image)
        self.label_image.resize(self.imW, self.imH)
        self._render_status()
        self.show()

    def _render_status(self):
        image_name = self.image_file_name(self.image_index)
        labeled = self.current_labels[self.images[self.image_index]]
        pad_zero = int(log10(len(self.images))) + 1
        if labeled is not None:
            self.label_status.setText('({}/{}) {} => Labeled as {}'.format(
                str(self.image_index + 1).zfill(pad_zero), len(self.images), image_name, labeled
            ))
        else:
            self.label_status.setText('({}/{}) {}'.format(
                str(self.image_index + 1).zfill(pad_zero), len(self.images), image_name
            ))
        self._redraw_labels_selection_list()

    def _prev(self):
        self.image_index = max(0, self.image_index - 1)
        self._render_window()
        self.red_mark.setText('')

    def _next(self):
        self.image_index = min(self.image_index + 1, len(self.images) - 1)
        self._render_window()
        self.red_mark.setText('')

    def _goto_prev_unlabeled_image(self):
        self.red_mark.setText('')
        if self.image_index == 0:
            QMessageBox.warning(self, 'Warning', 'Reach the top of images')
        else:
            prev_image_index = self.image_index
            for idx in range(self.image_index - 1, 0, -1):
                if self.current_labels[self.images[idx]] is None:
                    prev_image_index = idx
                    break
            if prev_image_index == self.image_index:
                QMessageBox.information(self, 'Information', 'No more prev unlabeled image')
            else:
                self.image_index = prev_image_index
                self._render_window()

    def _goto_next_unlabeled_image(self):
        self.red_mark.setText('')
        if self.image_index == len(self.images) - 1:
            QMessageBox.warning(self, 'Warning', 'Reach the end of images')
        else:
            next_image_index = self.image_index
            for idx in range(self.image_index + 1, len(self.images)):
                if self.current_labels[self.images[idx]] is None:
                    next_image_index = idx
                    break
            if next_image_index == self.image_index:
                QMessageBox.information(self, 'Information', 'No more next unlabeled image')
            else:
                self.image_index = next_image_index
                self._render_window()

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
        if kval in [Qt.Key_Left, Qt.Key_A]:
            self.label_selection_list.setStyleSheet("color: black;")
            self._prev()
        elif kval in [Qt.Key_Right, Qt.Key_D]:
            self.label_selection_list.setStyleSheet("color: black;")
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
                imname = self.images[self.image_index]
                self.current_labels[imname] = self.label_values[lv_index]
        else:
            LOGGER.debug('You Clicked {} but nothing happened...'.format(event.key()))
        self._render_window()

    def _redraw_labels_selection_list(self, selected=-1):
        lst = []
        imcode = ''
        for ii, lv in enumerate(self.label_values):
            imcode = self.current_labels[self.images[self.image_index]]
            mark = ' ' if imcode is None or ii != self.label_values.index(imcode) else '*'
            lv_n = len([v for z, v in self.current_labels.items() if v == lv])
            lst.append(f'{mark}{ii}  {lv:.<15} {lv_n}')
        t = '\n'.join(lst)
        self.red_mark.setText(imcode)
        self.label_selection_list.setText(t)
