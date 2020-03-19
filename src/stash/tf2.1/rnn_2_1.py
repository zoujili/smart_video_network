"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D,BatchNormalization,Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
                                        MaxPooling2D)
from collections import deque
import tensorflow as tf
import sys, settings
from keras.layers import GaussianNoise
from keras.models import Model
from network.cnn import DP_model
import keras.backend as K




class RNNModels_2_1():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=1024):
        """
        `model` = lstm (only one for this case)
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        #optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        #optimizer =  tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        # optimizer = Adam(lr=1e-4, decay=1e-5)
        # optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

       # K.set_floatx('float16')




        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        # K.set_session(sess)


        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.features_length = features_length
        self.nb_classes = nb_classes

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 224, 224, 3)
            self.model = self.lrcn()
        else:
            print("Unknown network.")
            sys.exit()



        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=metrics)

        #print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        # model.add(LSTM(64,
        #                return_sequences=False,
        #                input_shape=self.input_shape,
        #                dropout=0.5,
        #                recurrent_regularizer=tf.keras.regularizers.l2(l=0.001),
        #                kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
        #                bias_regularizer=tf.keras.regularizers.l2(l=0.001)))

        # model.add(LSTM(32,
        #                return_sequences= True,
        #                input_shape=self.input_shape,
        #                dropout=0.5,
        #                recurrent_regularizer=tf.keras.regularizers.l2(l=0.001),
        #                kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
        #                bias_regularizer=tf.keras.regularizers.l2(l=0.001)))

        model.add(LSTM(256,
                       return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5,
                       recurrent_regularizer=tf.keras.regularizers.l2(l=0.001),
                       kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                       bias_regularizer=tf.keras.regularizers.l2(l=0.001)))
        #model.add(GaussianNoise(0.5))
        model.add(Dense(64, activation='relu'))
        #model.add(GaussianNoise(0.1))
        model.add(Dropout(0.5))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def ResNet_sigmoid(class_num=6):
        img_input = tf.keras.layers.Input(shape=(None, None, 3))
        base_model = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet',
                                                               input_tensor=img_input, pooling='max')
        x = base_model.output
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        class_out = tf.keras.layers.Dense(class_num, activation='sigmoid', dtype='float32')(x)
        model = tf.keras.models.Model(inputs=img_input, outputs=class_out)
        model.summary()
        return model

    def lrcn(self):
        initialiser = 'glorot_uniform'
        reg_lambda = 0.001
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                         kernel_initializer=initialiser,
                                         kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
        model.add(tf.keras.layers.LSTM(16, dropout=0.5))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(self.nb_classes, activation='softmax',dtype='float32'))
        model.summary()

        return model


    def lrcn_orgin(self):
        initialiser = 'glorot_uniform'
        reg_lambda = 0.001


        model = Sequential()

       # model = tf.keras.models.Sequential


        # first (non-default) block
        model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                         kernel_initializer=initialiser,
                                         kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                         kernel_initializer=initialiser,
                                         kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))


        model.add(TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                                         kernel_initializer=initialiser,
                                         kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))




        # model.add(TimeDistributed(Conv2D(8, (5, 5), strides=(2, 2), padding='same',
        #                                  kernel_initializer=initialiser,
        #                                  kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda)),
        #                           input_shape=self.input_shape))
        # model.add(TimeDistributed(BatchNormalization()))
        # model.add(TimeDistributed(Activation('relu')))
        #
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))


        model.add(TimeDistributed(Flatten()))

        # model.add(GaussianNoise(0.1))
        # model.add(Dropout(0.5))

        model.add(LSTM(16, return_sequences=False, dropout=0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.summary()
        #
        # model.add(TimeDistributed(Flatten()))
        # model.add(LSTM(256, return_sequences=False, dropout=0.5))

        # model.add(Dense(self.nb_classes, activation='softmax'))

        # x =TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
        #                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
        #                           input_shape=self.input_shape)
        #
        # x = TimeDistributed(Flatten())
        # x = LSTM(256, return_sequences=False, dropout=0.5)(x)
        # x = Dense(self.nb_classes, activation='softmax')(x)
        # model = Model(x.input, x, name='AAA')

        # dp = DP_model()
        # resUnet = dp.ResNet_sigmoid(class_num=len(settings.VIDEO_CATEGORIES))
        # x = resUnet.output
        # x = LSTM(256, return_sequences=False, dropout=0.5)(x)
        # x = Dense(self.nb_classes, activation='softmax')(x)
        # model = Model(resUnet.input, x, name='AAA')

        # m = Model(
        #     inputs=resUnet.input,
        #     outputs = resUnet.output,
        # )
        #
        # # resUnet.load_weights(settings.CNN_NETWORK_PATH)
        # m.add(TimeDistributed(Flatten()))
        # m.add(LSTM(256, return_sequences=False, dropout=0.5))
        #
        return model


    def lrcn_backup(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        # def add_default_block(model, kernel_filters, init, reg_lambda):
        #
        #     # conv
        #     model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
        #                                      kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda))))
        #     model.add(TimeDistributed(BatchNormalization()))
        #     model.add(TimeDistributed(Activation('relu')))
        #     # conv
        #     model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
        #                                      kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda))))
        #     model.add(TimeDistributed(BatchNormalization()))
        #     model.add(TimeDistributed(Activation('relu')))
        #     # max pool
        #     model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        #
        #     return model

        initialiser = 'glorot_uniform'
        reg_lambda  = 0.001

        model = Sequential()

        # first (non-default) block
        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                         kernel_initializer=initialiser, kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser, kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # 2nd-5th (default) blocks
        # model = add_default_block(model, 64,  init=initialiser, reg_lambda=reg_lambda)
        # model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
        # model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
        # model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head

        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def lrcn_customer_backup(self):
        dp = DP_model()
        dp_model = dp.ResNet_sigmoid_v2(class_num=len(settings.VIDEO_CATEGORIES_ADD))
        # dp_model.load_weights(settings.CNN_NETWORK_PATH)
        dp_model_output = Model(
            inputs=dp_model.input,
            # outputs = resUnet.output
            outputs=dp_model.get_layer('global_max_pooling2d_1').output
            # outputs=resUnet.get_layer('dense_1').output
        )
        dp.freeze_top_layers(dp_model_output, 3)

        cnn_model = Sequential(layers=dp_model_output.layers)

        model = Sequential()
        model.add(TimeDistributed(cnn_model, input_shape=self.input_shape))
        # model.add(TimeDistributed(Flatten()))
        model.add(LSTM(64, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.summary()
        #
        # model.add(TimeDistributed(Flatten()))
        # model.add(LSTM(256, return_sequences=False, dropout=0.5))

        # model.add(Dense(self.nb_classes, activation='softmax'))

        # x =TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
        #                                kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
        #                           input_shape=self.input_shape)
        #
        # x = TimeDistributed(Flatten())
        # x = LSTM(256, return_sequences=False, dropout=0.5)(x)
        # x = Dense(self.nb_classes, activation='softmax')(x)
        # model = Model(x.input, x, name='AAA')

        # dp = DP_model()
        # resUnet = dp.ResNet_sigmoid(class_num=len(settings.VIDEO_CATEGORIES))
        # x = resUnet.output
        # x = LSTM(256, return_sequences=False, dropout=0.5)(x)
        # x = Dense(self.nb_classes, activation='softmax')(x)
        # model = Model(resUnet.input, x, name='AAA')

        # m = Model(
        #     inputs=resUnet.input,
        #     outputs = resUnet.output,
        # )
        #
        # # resUnet.load_weights(settings.CNN_NETWORK_PATH)
        # m.add(TimeDistributed(Flatten()))
        # m.add(LSTM(256, return_sequences=False, dropout=0.5))
        #
        return model
