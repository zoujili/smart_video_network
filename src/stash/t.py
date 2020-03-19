from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D,Flatten

nb_lstm_outputs = 30
nb_time_steps = 28
nb_input_vector = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)


# model = Sequential()
# model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
# model.add(Dense(10, activation='softmax'))

model = Sequential()
model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
#odel.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu',input_shape = (28, 28, 1)))
#model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))


# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# print(x_train.shape)
# print(x_test.shape)


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=128, epochs = 1, verbose=1, validation_data=(x_test, y_test))
# model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=1)


model.summary()

score = model.evaluate(x_test, y_test,batch_size=1, verbose=1)
print(score)