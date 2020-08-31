import datetime
import os
import shutil
from queue import Queue
import pandas as pd

opj = os.path.join
image_extensions = ['.png', '.jpg']


def make_list_of_files_by_extension(source, extensions=None):
    if extensions is None:
        extensions = ('.jpg', '.png')
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
            file_names.append(name)
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

    def log(self, s):
        self._log.append(s)
        print(s)

    def load(self, path):
        self.path = path
        self.main['path'], self.main['name'] = make_list_of_files_by_extension(self.path)
        self.main = self.main.set_index('name')

        for pth, nm in zip(*make_list_of_files_by_name(self.path, 'results.csv')):
            lbl = os.path.split(pth)[1]
            self.labels[lbl] = {'file': os.path.join(pth, nm), 'path': pth, 'nm': nm}
            self.log(f'loaded: {pth} {nm}')
            self.main[lbl] = ''
            l_df = pd.read_csv(self.labels[lbl]['file'])
            self.labels[lbl]['labels'] = l_df.label.unique()
            l_df.set_index('image', inplace=True)
            l_df.rename(columns={'label': lbl}, inplace=True)
            self.main.update(l_df)
            print(self.main[lbl].unique())
        self.filter = self.main.index()

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
        return opj(self.main.path[im], im)

    def set_value(self, im, label, code):
        try:
            self.main.at[im, label] = code
            return 'ok'
        except Exception as e:
            return f'fail: {e}'

    def set_filter(self, label, value, seek_label='none'):
        if label not in ('none', '', None):
            self.filter = self.main.index()
        self.filter = self.main[label] == value
        if seek_label not in ('none', '', None):
            self.filter = self.filter & self.main[seek_label] != ''
        return self.main.index[self.filter].to_list()


if __name__ == '__main__':
    d = Dbs()
    d.load('/home/imt/dataset/dataset_for_multilabel_classification')
    d.store_label('uniforme')
