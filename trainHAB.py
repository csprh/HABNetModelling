"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from dataHAB import DataSet
import time
import os.path
import xml.etree.ElementTree as ET
import sys

def train(data_type, seq_length, model, image_shape=None,
          batch_size=32, nb_epoch=100):
    #import pdb; pdb.set_trace()
    #import pudb; pu.db
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=100)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    generator = data.frame_generator(batch_size, 'train', data_type)
    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(2, model, seq_length, None)

    rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=40,
        workers=4)

def main(argv):
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, mlp
    #import pudb; pu.db
    model = 'mlp'
    seq_length = 5
    batch_size = 128
    nb_epoch = 1000
    #argv[1] = 'classifyHAB1.xml'
    if (len(argv)==0):
        xmlName = 'classifyHAB1.xml'
    else:
        xmlName = argv[0]
    tree = ET.parse(xmlName)
    root = tree.getroot()

    for child in root:
        print(child.tag, child.attrib)

    train('features', seq_length, model, image_shape=None,
          batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main(sys.argv[1:])
