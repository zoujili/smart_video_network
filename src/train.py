import os.path
import time

from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

import src.utils as util
from src.data import DataSet
from src.network.model import MLModel


def train(batch_size, nb_epoch, data_type, seq_len, categories, feature_len, saved_model=None):
    checkpointer = ModelCheckpoint(
        # filepath=os.path.join(settings.OUTPUT_CHECKPOINT_FOLDER, model + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        filepath=os.path.join(util.OUTPUT_CHECKPOINT_FOLDER, 'v1.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=util.OUTPUT_LOG)

    # Helper: Stop when we stop learning.
    # early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(util.OUTPUT_LOG, 'training-' + str(timestamp) + '.log'))

    data = DataSet(util.SCRIPT_EXTRACT_SEQ_SPLIT_PATH)

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = data.len_data() / batch_size

    generator = data.frame_generator(batch_size, 'train')
    val_generator = data.frame_generator(batch_size, 'valid')

    # Get the model.
    model = MLModel(len(categories), data_type, seq_len, saved_model,
                    feature_len)
    rm = model.create_pre_train_model()
    # rm = em.create_model()

    rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=200 / batch_size)


def main():
    os.chdir('./..')
    saved_model = None
    train(util.BATCH_SIZE, util.NB_EPOCH, util.SEQ_DATA_TYPE, util.SEQ_LEN, util.VIDEO_CATEGORIES, util.CNN_FEATURE_LEN,
          saved_model)


if __name__ == '__main__':
    main()
