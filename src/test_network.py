from src.data import DataSet
from keras.models import load_model
import numpy as np
from sklearn import metrics


def test_rnn(model):
    data = DataSet()
    x, y , data_list= data.get_test_frames()
    y_pred = model.predict(x)

    y_pred[y_pred < 0.7] = 0
    y_pred[y_pred >= 0.7] = 1

    print(metrics.precision_score(y,y_pred,average='micro',zero_division=0))
    print(metrics.precision_score(y,y_pred,average='macro',zero_division=0))
    print(metrics.recall_score(y,y_pred,average='micro',zero_division=0))
    print(metrics.recall_score(y, y_pred, average='macro',zero_division=0))
    print(metrics.f1_score(y, y_pred, average='weighted',zero_division=0))



if __name__ == '__main__':
    saved_rnn_model_path = '/home/ubuntu/jili/ai_project/video_to_seq_rnn/output/checkpoint/lrcn.v1.hdf5'
    saved_rnn_model = load_model(saved_rnn_model_path)
    test_rnn(saved_rnn_model)

