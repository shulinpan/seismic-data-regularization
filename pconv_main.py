import os
import gc
import datetime
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
import tensorflow as tf
from keras import backend as K
from keras.utils import Sequence
from keras_tqdm import TQDMCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from pconv_model import PConvUnet
from util import MaskGenerator
from data_generator import AugmentingDataGenerator




def parse_args():
    parser = ArgumentParser(description='Training script for PConv interpolation')
    parser.add_argument('--stage',type=str, default='train',help='Which stage of training to run',choices=['train', 'finetune'])
    parser.add_argument('--data_dir', type=str, default='data', help='Folder with data')
    parser.add_argument('--train',type=str, default='train',help='Folder with training data')
    parser.add_argument('--validation',type=str,default='val', help='Folder with validation data')
    parser.add_argument( '--test',type=str,default='test',help='Folder with testing data')
    parser.add_argument('--name',type=str, default='myDataset_pconv',help='Dataset name')
    parser.add_argument('--batch_size',type=int, default=4,help='What batch-size should we use')
    parser.add_argument('--test_path', type=str, default='./data/test_samples/',help='Where to output test data during training')
    parser.add_argument('--weight_path',type=str, default='./data/logs/',help='Where to output weights during training')
    parser.add_argument('--log_path',type=str, default='./data/logs/',help='Where to output tensorboard logs during training')
    parser.add_argument('--checkpoint',type=str,help='Previous weights to be loaded onto model')
    return parser.parse_args()






if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    args = parse_args()

    if args.stage == 'finetune' and not args.checkpoint:
        raise AttributeError('If you are finetuning your model, you must supply a checkpoint file')

    # Create training generator
    train_datagen = AugmentingDataGenerator(os.path.join(os.getcwd(),args.data_dir,args.train,'npy'))
    train_generator = train_datagen.flow_from_directory(MaskGenerator(128,128))

    # # Create validation generator
    # val_datagen = AugmentingDataGenerator(os.path.join(os.getcwd(),args.data_dir,args.validation,'npy'))
    # val_generator = val_datagen.flow_from_directory(MaskGenerator(128,128),seed =40)

    # Create testing generator
    test_datagen = AugmentingDataGenerator(os.path.join(os.getcwd(),args.data_dir,args.test,'npy'),shuffle=False)
    test_generator = test_datagen.flow_from_directory(MaskGenerator(128,128),seed=40)

    # Pick out an example to be send to test samples folder
    test_data = next(test_generator)
    (masked, mask), ori = test_data



    def plot_callback(model, path):
        """Called at the end of each epoch, displaying our previous test datas,
        as well as their masked predictions and saving them to disk"""

        pred_data = model.predict([masked, mask])
        pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


        _, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes[0][0].imshow(masked[0, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[0][1].imshow(pred_data[0, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[0][2].imshow(ori[0, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[0][3].imshow((pred_data[0, :, :, 0]-ori[0, :, :, 0]).T, cmap='gray',vmin=0, vmax=1)
        axes[1][0].imshow(masked[1, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[1][1].imshow(pred_data[1, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[1][2].imshow(ori[1, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[1][3].imshow((pred_data[1, :, :, 0] - ori[1, :, :, 0]).T, cmap='gray',vmin=0, vmax=1)
        axes[2][0].imshow(masked[2, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[2][1].imshow(pred_data[2, :, :,0].T, cmap='gray',vmin=0, vmax=1)
        axes[2][2].imshow(ori[2, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[2][3].imshow((pred_data[2, :, :, 0] - ori[2, :, :, 0]).T, cmap='gray',vmin=0, vmax=1)
        axes[3][0].imshow(masked[3, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[3][1].imshow(pred_data[3, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[3][2].imshow(ori[3, :, :, 0].T, cmap='gray',vmin=0, vmax=1)
        axes[3][3].imshow((pred_data[3, :, :, 0] - ori[3, :, :, 0]).T, cmap='gray',vmin=0, vmax=1)
        axes[0][0].set_title('Masked Data')
        axes[0][1].set_title('Predicted Data')
        axes[0][2].set_title('Original Data')
        axes[0][3].set_title('Residual')
        axes[1][0].set_title('Masked Data')
        axes[1][1].set_title('Predicted Data')
        axes[1][2].set_title('Original Data')
        axes[1][3].set_title('Residual')
        axes[2][0].set_title('Masked Data')
        axes[2][1].set_title('Predicted Data')
        axes[2][2].set_title('Original Data')
        axes[2][3].set_title('Residual')
        axes[3][0].set_title('Masked Data')
        axes[3][1].set_title('Predicted Data')
        axes[3][2].set_title('Original Data')
        axes[3][3].set_title('Residual')

        if not os.path.exists(path + args.name + '_phase1'):
            os.makedirs(path + args.name + '_phase1')
        plt.savefig(path + args.name + '_phase1/'+'img_{}.png'.format( pred_time))

        plt.close()

    # Load the model
    model = PConvUnet()


    # Loading of checkpoint
    if args.checkpoint:
        if args.stage == 'train':
            model.load(args.checkpoint)
        elif args.stage == 'finetune':
            model.load(args.checkpoint, train_bn=False, lr=0.00005)
    else:
        print('no checkpoint file')
    # Fit model
    model.fit(
        train_generator,
        steps_per_epoch=5000,
        # validation_data=val_generator,
        # validation_steps=1000,
        epochs=2000,
        verbose=0,
        callbacks=[
            TensorBoard(
                log_dir=os.path.join(args.log_path, args.name + '_phase1'),
                write_graph=False
            ),

            ModelCheckpoint(
                os.path.join(args.log_path, args.name + '_phase1', 'weights.{epoch:003d}-{loss:.2f}.h5'),
                # monitor='val_loss',
                # save_best_only=True,
                save_weights_only=True
            ),

            LambdaCallback(
                on_epoch_end=lambda epoch, logs: plot_callback(model, args.test_path)
            ),

            TQDMCallback()
        ]
    )

