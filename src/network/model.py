import sys

import tensorflow as tf
from keras import backend as K
from keras.applications import MobileNetV2, VGG19, MobileNet
from keras.layers import CuDNNLSTM
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
                                        MaxPooling2D)
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import src.utils as util


tf.config.experimental_run_functions_eagerly(True)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
K.set_session(tf_sess)
K.set_learning_phase(0)
print("tf version: {}".format(tf.__version__))


# Reference https://github.com/peachman05/RGB_action_recognition


class MLModel:
    def __init__(self, nb_classes, data_type, seq_length,
                 saved_model=None, features_length=1024):
        """
        `model` = lstm (only one for this case)
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """
        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.features_length = features_length
        self.nb_classes = nb_classes
        self.input_shape = (seq_length, 224, 224, 3)

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        if data_type == 'feature':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
        elif data_type == 'image':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 224, 224, 3)
        else:
            print("Unknown network.")
            sys.exit()

    def compile(self, model):
        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')
        # Now compile the network.
        # optimizer = Adam(lr=1e-5, decay=1e-6)
        optimizer = Adam(lr=1e-4,decay=1e-5)
        # optimizer = Adam(lr=1e-6,decay=1e-7)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=metrics)

    def create_custom_model(self):
        model = Sequential()
        # after having Conv2D...
        model.add(
            TimeDistributed(
                Conv2D(64, (3, 3), activation='relu'),
                input_shape=self.input_shape  # 5 images...
            )
        )
        model.add(
            TimeDistributed(
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2))  # Or Flatten()
            )
        )
        ###
        model.add(
            TimeDistributed(
                Conv2D(64, (3, 3), activation='relu')
            )
        )
        # We need to have only one dimension per output
        # to insert them to the LSTM layer - Flatten or use Pooling
        model.add(
            TimeDistributed(
                GlobalAveragePooling2D()  # Or Flatten()
            )
        )
        # previous layer gives 5 outputs, Keras will make the job
        # to configure LSTM inputs shape (5, ...)
        model.add(
            CuDNNLSTM(32, return_sequences=False)
        )
        # and then, common Dense layers... Dropout...
        # up to you
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(.5))

        model.add(Dense(16, activation='relu'))
        model.add(Dropout(.5))
        # For example, for 3 outputs classes
        model.add(Dense(self.nb_classes, activation='softmax'))
        # sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
        # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        self.compile(model)

        return model

    def create_pre_train_model(self):
        model = Sequential()
        # after having Conv2D...
        # if pretrain_name == 'ResNet152V2':
        #     model.add(
        #         TimeDistributed(
        #             ResNet152V2(weights='imagenet',include_top=False),
        #             input_shape=(n_sequence, *dim, n_channels) # 5 images...
        #         )
        #     )
        # elif pretrain_name == 'Xception':
        #     model.add(
        #         TimeDistributed(
        #             Xception(weights='imagenet',include_top=False),
        #             input_shape=(n_sequence, *dim, n_channels) # 5 images...
        #         )
        #     )
        # elif pretrain_name == 'MobileNetV2':
        #     model.add(
        #         TimeDistributed(
        #             # MobileNetV2(weights='imagenet',include_top=False),
        #             MobileNetV2(weights='imagenet',include_top=False, alpha= alpha),
        #             input_shape=(n_sequence, *dim, n_channels) # 5 images...
        #         )
        #     )
        # else:
        #     raise ValueError('pretrain_name is incorrect')

        # MobileNetV2, MobileNet, ResNet152V2, Xception, VGG19, VGG16, DenseNet201
        # print_cnn = VGG19(weights='imagenet', include_top=False)
        # print_cnn.summary()

        model.add(
            TimeDistributed(
                MobileNet(weights='imagenet', include_top=False),
                input_shape=self.input_shape  # 5 images...
            )
        )

        model.add(
            TimeDistributed(
                GlobalAveragePooling2D()  # Or Flatten()
            )
        )
        model.add(
            CuDNNLSTM(256, return_sequences=False)
        )
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(.5))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(.5))
        # model.add(Dense(n_output))
        model.add(Dense(self.nb_classes, activation='softmax'))

        # model.compile(optimizer='sgd', loss=my_loss, metrics=['sparse_categorical_accuracy'])

        model.summary()
        # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        self.compile(model)

        return model

    def create_conv3D_model(self, set_pretrain=False):
        model = Sequential()
        n_first_filter = 16
        model.add(Conv3D(n_first_filter, kernel_size=(3, 3, 3), activation='relu',
                         kernel_initializer='he_uniform',
                         input_shape=self.input_shape)
                  )
        model.add(MaxPooling3D(pool_size=(1, 1, 1)))
        model.add(Conv3D(8, kernel_size=(3, 3, 3), activation='relu',
                         kernel_initializer='he_uniform')
                  )
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Flatten())
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(.4))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        if set_pretrain:
            mobile_model = Sequential()
            mobile_model.add(MobileNetV2(weights='imagenet', include_top=False))

            weights = model.layers[0].get_weights()  # first layer
            weight_mobile = mobile_model.layers[0].get_weights()[0]  # first layer, first weight

            for i in range(3):
                weights[0][:, :, i, :, :] = weight_mobile[:, :, :, :n_first_filter]

            model.layers[0].set_weights(weights)

        model.summary()
        self.compile(model)

        return model

    def create_lstm_model(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(32,
                       return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5,
                       recurrent_regularizer=tf.keras.regularizers.l2(l=0.001),
                       kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                       bias_regularizer=tf.keras.regularizers.l2(l=0.001)))
        # model.add(GaussianNoise(0.5))
        model.add(Dense(64, activation='relu'))
        # model.add(GaussianNoise(0.1))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.summary()

        self.compile(model)
        return model

    # def create_model_skeleton(self, n_sequence, n_joint, n_output):
    #     skeleton_stream = Input(shape=(n_sequence, n_joint * 3), name='skleton_stream')
    #     skeleton_lstm = CuDNNLSTM(50, return_sequences=False)(skeleton_stream)
    #     skeleton_lstm = Dropout(0.5)(skeleton_lstm)
    #     fc_1 = Dense(units=60, activation='relu')(skeleton_lstm)
    #     fc_1 = Dropout(0.5)(fc_1)
    #     fc_2 = Dense(units=n_output, activation='softmax', use_bias=True, name='main_output')(fc_1)
    #     model = Model(inputs=skeleton_stream, outputs=fc_2)
    #     # print(model.summary())
    #
    #
    #     sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
    #     model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     return model
    #
    # def create_2stream_model(self, dim, n_sequence, n_channels, n_joint, n_output):
    #     rgb_stream = Input(shape=(n_sequence, *dim, n_channels), name='rgb_stream')
    #     skeleton_stream = Input(shape=(n_sequence, n_joint * 3), name='skleton_stream')
    #
    #     mobileNet = TimeDistributed(MobileNetV2(weights='imagenet', include_top=False))(rgb_stream)
    #     rgb_feature = TimeDistributed(GlobalAveragePooling2D())(mobileNet)
    #
    #     rgb_lstm = CuDNNLSTM(64, return_sequences=False)(rgb_feature)
    #     skeleton_lstm = CuDNNLSTM(64, return_sequences=False)(skeleton_stream)
    #
    #     combine = concatenate([rgb_lstm, skeleton_lstm])
    #
    #     fc_1 = Dense(units=64, activation='relu')(combine)
    #     fc_1 = Dropout(0.5)(fc_1)
    #     fc_2 = Dense(units=24, activation='relu')(fc_1)
    #     fc_2 = Dropout(0.3)(fc_2)
    #     fc_3 = Dense(units=n_output, activation='softmax', use_bias=True, name='main_output')(fc_2)
    #     model = Model(inputs=[rgb_stream, skeleton_stream], outputs=fc_3)
    #     sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
    #     model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     return model


if __name__ == '__main__':
    saved_model = None

    em = MLModel(len(util.VIDEO_CATEGORIES),
                 util.SEQ_DATA_TYPE,
                 util.SEQ_LEN,
                 saved_model,
                 util.CNN_FEATURE_LEN)
    rm = em.create_model_pretrain()
