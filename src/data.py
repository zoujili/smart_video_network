import json
import os
import random
import threading

import numpy as np

import src.utils as util


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""

    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))

    return gen


class DataSet:
    def __init__(self, src=util.SCRIPT_EXTRACT_SEQ_SPLIT_PATH):
        self.data = self.load_data(src)

    @staticmethod
    def load_data(src):
        with open(src) as file:
            data = json.load(file)
        return data

    @staticmethod
    def get_extracted_seq(data):
        path = data['seq_path']
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def len_data(self, data_type='train'):
        return len(self.data[data_type])

    # @threadsafe_generator
    def frame_generator(self, batch_size, data_type='train'):
        data = self.data[data_type]
        while 1:
            random_list = data
            random.shuffle(random_list)
            X, Y = [], []
            Z = []
            # Generate batch_size samples.
            for item in random_list:
                seq = self.get_extracted_seq(item)
                if seq is None:
                    raise ValueError("Can't find sequence. Did you generate them?")
                X.append(seq)
                Y.append(util.categories_to_np(item['categories']))
                # Z.append(item['seq_path'])
                if len(X) == batch_size:
                    # print(Z)
                    X_NP = np.array(X)
                    Y_NP = np.array(Y)
                    X = []
                    Y = []
                    Z = []
                    yield X_NP, Y_NP

    def get_test_frames(self, data_type='test'):
        data = self.data[data_type]
        random.shuffle(data)
        X, Y = [], []
        for item in data:
            seq = self.get_extracted_seq(item)
            if seq is None:
                raise ValueError("Can't find sequence. Did you generate them?")
            X.append(seq)
            Y.append(util.categories_to_np(item['categories']))
        return np.array(X), np.array(Y), data


if __name__ == '__main__':
    data = DataSet(util.SCRIPT_EXTRACT_SEQ_SPLIT_PATH)
    print("success")
