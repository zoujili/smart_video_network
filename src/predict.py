import src.preprocess as process
import numpy as np
from keras.models import load_model
import os
import shutil


def predict_video(video_path, rnn_model):
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')

    data_list = []
    data_item = {'seq_path': './tmp/video_test/test.npy',
                 'image_folder': './tmp/video_test',
                 'seq_folder': './tmp/video_test'}

    process.extract_images_from_video(video_path, './tmp/video_test')
    data_list.append(data_item)
    process.build_sequence(data_list, 7)

    seq = np.load(data_item['seq_path'])
    seq = np.expand_dims(seq, axis=0)

    r = rnn_model.predict(seq)
    print(r)

    shutil.rmtree('tmp')


if __name__ == '__main__':
    os.chdir('./..')

    rnn_model_path = './output/checkpoint/v1.hdf5'
    rnn_model = load_model(rnn_model_path)

    # gun
    video_path = './dataset/annotated_video/2019-12-20-13-55-17_jili/00046.mp4'
    # [[1.6030058e-04 1.3164644e-03 9.9730730e-01 1.8988950e-04 1.0260656e-03]]

    # shooting
    # video_path = './dataset/annotated_video/2020-01-06-14-21-05_jili/00679.mp4'
    # [[0.00803162 0.00396992 0.00793385 0.00167812 0.97838646]]

    # run
    # video_path = './dataset/annotated_video/2019-12-20-13-55-17_jili/00044.mp4'
    # [[1.8742312e-05 9.9940276e-01 3.1481043e-04 1.8490202e-04 7.8919897e-05]]

    # chests
    # video_path = './dataset/annotated_video/2020-01-06-14-21-04_jili/00182.mp4'
    # [[6.3970168e-03 1.8032858e-02 1.0630875e-03 9.7370130e-01 8.0572651e-04]]

    # chop
    # video_path = './dataset/annotated_video/2019-12-20-13-55-17_jili/00008.mp4'
    # #[[9.9747247e-01 3.3262811e-04 8.0785598e-04 6.2148774e-04 7.6553330e-04]]

    predict_video(video_path, rnn_model)

    print('sucess')
