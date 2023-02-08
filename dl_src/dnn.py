import os
from queue import Queue

import cv2
from numpy import zeros, count_nonzero
import numpy as np
import pandas as pd
import shutil

empty_image = zeros((30, 30, 3), dtype=np.uint8)
FAVORITES = 'favorites.txt'

class DNN:

    def __init__(self, ie_core, xml, device, num_requests, get_full=False, show_tqdm=False):
        net = ie_core.read_network(model=xml, weights=os.path.splitext(xml)[0] + ".bin")
        self.input_blob = next(iter(net.input_info))
        self.input_shape = net.input_info[self.input_blob].input_data.shape
        self.n, self.c, self.h, self.w = net.input_info[self.input_blob].input_data.shape
        self.out_blob = next(iter(net.outputs))
        self.out_shape = net.outputs[self.out_blob].shape
        self.n_requests = num_requests
        self.exec_net = ie_core.load_network(network=net, device_name=device, num_requests=num_requests)
        self.xml_name = os.path.splitext(os.path.split(xml)[1])[0]
        print('')
        print(f"[{self.xml_name}] XML: {xml}")
        print(f'n:{self.n} c:{self.c} w:{self.w} h:{self.h}')
        self.idents = [None] * num_requests
        self.i = -1
        self.get_full = get_full
        del net
        self._next = None
        self.show_tqdm = show_tqdm

    def drop(self):
        self.idents = [None] * self.n_requests
        self.i = -1
        if self._next is not None:
            self._next.drop()

    def go(self, frames):

        fframes = frames if self._next is None else self._next.go(frames)

        for frame in fframes:
            self.i += 1
            index = self.i % self.n_requests
            if self.i >= self.n_requests:
                self.exec_net.requests[index].wait()
                out_frame = self.idents[index].copy()
                res = self.exec_net.requests[index].output_blobs[self.out_blob].buffer.flatten()
                out_frame[self.xml_name] = res if self.get_full else res.argmax()
                yield out_frame
            in_frame = cv2.resize(frame['image'], (self.w, self.h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
            self.exec_net.start_async(request_id=index, inputs={self.input_blob: in_frame.reshape(self.input_shape)})
            self.idents[index] = frame
        self.i += 1
        for ii in list(range(self.i % self.n_requests, self.n_requests)) + list(range(0, self.i % self.n_requests)):
            if self.idents[ii] is None:
                continue
            self.exec_net.requests[ii].wait()
            res = self.exec_net.requests[ii].output_blobs[self.out_blob].buffer.flatten()
            self.idents[ii][self.xml_name] = res if self.get_full else res.argmax()
            yield self.idents[ii]
            self.idents[ii] = None

    def __rshift__(self, other):
        if type(other) == type(self):
            self._next = other
            print(f'[DNN] {self.xml_name} >> {other.xml_name}')
        return other

    def __repr__(self):
        o = [self.xml_name]
        if self._next is not None:
            o.append(self._next.__repr__())
        return '_'.join(o)

    def get_attr(self):
        o = []
        if not self.get_full:
            o.append(self.xml_name)
        if self._next:
            o += self._next.get_attr()
        return o


def img_iterator(images_names, excludes=None):
    for fnm in images_names:
        if excludes is not None and fnm['file'] in excludes:
            continue
        try:
            fnm['image'] = cv2.imread(fnm['file'])[20:-20, 20:-20]
            # print(f"fnm['file']:{fnm['file']}")
        except Exception as e:
            fnm['image'] = zeros((30, 30, 3), dtype=np.uint8)
        yield fnm


def iter_unimagined(recs):
    for rec in recs:
        yield {a: b for a, b in rec.items() if a != 'image'}


def make_list_of_files_by_extension(source, extensions=None):
    if extensions is None:
        extensions = ('.jpg', '.png')
    q = Queue()
    q.put(source)
    while not q.empty():
        v = q.get()
        if os.path.isdir(v):
            for vs in sorted(os.listdir(v)):
                q.put(os.path.join(v, vs))
        elif os.path.splitext(v)[1] in extensions:
            # print(f'File {v}')
            yield {'file': v}


def get_data_frame_from_folder(destination, dnn: DNN) -> pd.DataFrame:
    file_name = os.path.join(f'{dnn}.pkl')
    full_file_name = os.path.join(destination, file_name)
    print(f'[{destination}] Reading files')
    all_files = list(make_list_of_files_by_extension(source=destination))
    file_names = [x['file'] for x in all_files]
    print(f'[{destination}] num of images: {len(all_files)}')
    if not all_files:
        return pd.DataFrame()

    out_pd = pd.DataFrame()
    print(f'[{destination}] {file_name} exists: <<{os.path.exists(full_file_name)}>>')
    preload_files = []
    if os.path.exists(full_file_name):
        out_pd = pd.read_pickle(full_file_name)
        print(f'[{destination}] loaded records from file: {out_pd.shape}. Checking ...')
        out_pd = out_pd.drop_duplicates(subset=["file"])
        f_exists = out_pd['file'].isin(file_names)
        print(f'[{destination}] check of preloaded {out_pd.shape[0]} '
              f'-> existing: {len(file_names)} '
              f'-> filtered: {count_nonzero(f_exists)}')
        out_pd = out_pd[f_exists]
        preload_files = set(out_pd['file'].tolist())
        if os.path.exists(full_file_name):
            shutil.copy(full_file_name, full_file_name + '.bak')
        out_pd.to_pickle(full_file_name)

    not_loaded_files = [x for x in all_files if x['file'] not in preload_files]
    print(f'[{destination}] not loaded files: {len(not_loaded_files)} loaded files: {len(preload_files)}')
    len_of_set = len(not_loaded_files)
    if len_of_set > 0:
        print(f'[{destination}] added {len_of_set} files')
        data_frame_len = 1000
        for st in range(0, len_of_set, data_frame_len):
            fsh = min(st + data_frame_len, len_of_set)
            dnn.drop()
            out_pd = out_pd.append(pd.DataFrame(list(iter_unimagined(dnn.go(img_iterator(not_loaded_files[st:fsh]))))),
                                   ignore_index=True)
            if os.path.exists(full_file_name):
                shutil.copy(full_file_name, full_file_name + '.bak')
            out_pd.to_pickle(full_file_name)
            print(f'[{destination}] stored {st} ... {fsh}')
    print(f'stored: {out_pd.shape[0]} records to: {full_file_name}')
    print(f'Data main record filled [{" ".join(out_pd.keys())}]')
    out_pd['name'] = [os.path.split(x)[1] for x in out_pd['file']]
    print(f'[{destination}] look for {FAVORITES} file')
    return out_pd.set_index(np.arange(0, out_pd.shape[0]))
