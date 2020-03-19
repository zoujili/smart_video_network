from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from src.data import DataSet
from keras.models import load_model
import os


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


if __name__ == '__main__':
    os.chdir('./../../')
    saved_rnn_model_path = './output/checkpoint/v1.hdf5'
    model = load_model(saved_rnn_model_path)
    mds_and_plot(model)


































