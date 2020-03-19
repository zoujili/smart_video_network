from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from src.data import DataSet
from keras.models import load_model
import numpy as np
from sklearn import metrics

def mds_and_plot(model):

    data = DataSet()
    x, y , data_list= data.get_test_frames()
    y_pred = model.predict(x)


    mds = MDS()
    mds.fit(y_pred)
    a = mds.embedding_

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    color = 0
    j = 0
    for item in y:
        index = 0
        for i in item:
            if i == 1:
                break
            index = index + 1

        plt.plot([a[j:j + 1, 0]], [a[j:j + 1, 1]], mark[index], markersize=5)
        print(index)
        j += 1
    plt.show()



saved_rnn_model_path = '/home/ubuntu/jili/ai_project/video_to_seq_rnn/output/checkpoint/lrcn.v1.hdf5'
model = load_model(saved_rnn_model_path)
mds_and_plot(model)

































