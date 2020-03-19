from src.network.cnn_extractor import Extractor
import src.preprocess as preprocess
import numpy as np
from keras.models import load_model


def predict_video(video_path,rnn_model,cnn_extractor):
    X = []
    seq = preprocess.video_to_features(video_path,cnn_extractor)
    X.append(seq)
    x_np=np.array(X)

    r = rnn_model.predict(x_np)
    print(r)

if __name__ == '__main__':
    rnn_model_path = '/home/kris/ai_project/smart_light_seq_v2/output/checkpoint/lstm.v1.hdf5'
    video_path = '/home/kris/ai_project/smart_light_seq_v2/data/annotated_video/2019-12-20-13-55-17_jili/00295.mp4'

    cnn_extractor = Extractor()
    rnn_model = load_model(rnn_model_path)
    predict_video(video_path,rnn_model,cnn_extractor)

    print('sucess')

