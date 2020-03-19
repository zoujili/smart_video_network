from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Dense, Dropout, \
    Activation
from keras.models import Model
import src.utils as util


def mobile_net_V2(class_num=3):
    img_input = Input(shape=(None, None, 3), name="image")
    base_model = MobileNet(include_top=False, input_tensor=img_input, pooling='max', weights='imagenet')
    x = base_model.output
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    class_out = Dense(class_num, activation='softmax')(x)
    model = Model(img_input, class_out, name='ResNet')
    model.summary()

    return model


def mobile_net(class_num=6):
    img_input = Input(shape=(None, None, 3), name="image")
    base_model = MobileNet(input_tensor=img_input, pooling='max', include_top=False)
    x = base_model.output
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    class_out = Dense(class_num, activation='sigmoid')(x)
    model = Model(img_input, class_out, name='ResNet')
    return model


def vgg_19(self):
    return VGG19(weights="imagenet")


def freeze_top_layers(model, num=0):
    print("total layers: {}".format(len(model.layers)))
    if num and num < len(model.layers):
        print("Freezing {} top layers".format(num))
        for layer in model.layers[:-num]:
            layer.trainable = False
        for layer in model.layers[-num:]:
            layer.trainable = True


class Extractor():
    def __init__(self, model_path,nb_class,image_shape=(224, 224, 3)):
        input_tensor = Input(image_shape)
        model = mobile_net(class_num=nb_class)
        # model = mobile_net(class_num=8)
        model.load_weights(model_path)
        model.summary()
        # We'll extract features at the final pool layer.
        self.model = Model(
            inputs=model.input,
            outputs=model.get_layer('global_max_pooling2d_1').output
        )

    def extract(self, np_data):
        features = self.model.predict(np_data)
        return features[0]


if __name__ == '__main__':
    e = Extractor()
    feature = e.extract(
        "/home/kris/ai_project/smart_light_seq_v2/data/extracted_image/2019-12-20-13-55-16_jili/00001/0001.png")

    print(feature)
