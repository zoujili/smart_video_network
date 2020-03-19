from network.rnn_2_1 import RNNModels_2_1
from data import DataSet
import time
import os.path
import settings
import tensorflow as tf

def train (batch_size, nb_epoch,saved_model=None):
    # model can be only 'lstm'
    if settings.DATA_TYPE is "images":
        model = "lrcn"
    else:
        model = 'lstm'


    checkpointer =  tf.keras.callbacks.ModelCheckpoint(
        #filepath=os.path.join(settings.OUTPUT_CHECKPOINT_FOLDER, model + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        filepath=os.path.join(settings.OUTPUT_CHECKPOINT_FOLDER, model + '.v1.hdf5'),
        verbose=1,)
       #)

    # Helper: TensorBoard
    tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(settings.OUTPUT_LOG, model))

    # Helper: Stop when we stop learning.
    # early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger =tf.keras.callbacks.CSVLogger(os.path.join(settings.OUTPUT_LOG, model + '-' + 'training-' + str(timestamp) + '.log'))

    data = DataSet(settings.SCRIPT_EXTRACT_SEQ_SPLIT_PATH)

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = data.len_train_data()/batch_size

    generator = data.frame_generator(batch_size, 'train')
    val_generator = data.frame_generator(batch_size, 'valid')

    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    # Get the model.
    rm = RNNModels_2_1(len(settings.VIDEO_CATEGORIES_ADD), model, settings.SEQ_LEN, saved_model,settings.CNN_FEATURE_LEN)

    rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=1,
        verbose=1,
        callbacks=[tb, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=100)
        # workers=4)


def main():
    saved_model = None
    train(settings.BATCH_SIZE,settings.NB_EPOCH,saved_model)

if __name__ == '__main__':
    main()

