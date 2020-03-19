import os

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from sklearn import metrics

import src.utils as util
from src.data import DataSet
import time

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
K.set_session(tf_sess)
K.set_learning_phase(0)
print("tf version: {}".format(tf.__version__))


def test_rnn(src, model):
    data = DataSet(src)
    x, y, data_list = data.get_test_frames('train')
    s = time.clock()
    y_pred = model.predict(x)
    e = time.clock()
    print(e - s)
    y_pred[y_pred < 0.7] = 0
    y_pred[y_pred >= 0.7] = 1

    print(metrics.precision_score(y, y_pred, average='micro', zero_division=0))
    print(metrics.precision_score(y, y_pred, average='macro', zero_division=0))
    print(metrics.recall_score(y, y_pred, average='micro', zero_division=0))
    print(metrics.recall_score(y, y_pred, average='macro', zero_division=0))
    print(metrics.f1_score(y, y_pred, average='weighted', zero_division=0))


if __name__ == '__main__':
    os.chdir('./..')
    saved_rnn_model_path = './output/checkpoint/v1.hdf5'
    # saved_rnn_model_path = './output/history/vgg-19.hdf5'
    # saved_rnn_model_path = './output/history/vgg-16.hdf5'
    s = time.clock()
    saved_rnn_model = load_model(saved_rnn_model_path)
    e = time.clock()
    print(e - s)
    test_rnn(util.SCRIPT_EXTRACT_SEQ_SPLIT_PATH, saved_rnn_model)
