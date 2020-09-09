import keras
from keras import backend as K


###########################################################################################################


# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
# # Dice similarity coefficient loss

def dice_coef(y_true, y_pred, smooth=1) :
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


###########################################################################################################


import os
import numpy as np

from segmentation_models import get_preprocessing

# SINCE WE LOAD U-NET WITH PRETRAINING FROM IMAGENET =>
preprocess = get_preprocessing('resnet34') # for resnet, img = (img-110.0)/1.0


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class custom_data_generator(keras.utils.Sequence) :
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(256, 1600), n_channels=3,
                 shuffle=True,
                 base_path=os.path.join('data', 'train_images', 'preprocessed')
                ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.base_path = base_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.list_IDs.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs.iloc[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = preprocess( cv2.imread( os.path.join(self.base_path, ID['ImageId']) ) )
            img = cv2.resize( img, (800, 128), interpolation = cv2.INTER_AREA)
            img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY ) / 255.
            X[i,] = np.resize(img, (*img.shape, 1))

            # Store outcome
            # (force-convert grayscale to 0/1 backwhite [to conuter-balance any cv2 inner-conversions])
            mask = cv2.threshold(
                cv2.imread( os.path.join(self.base_path, ID['MaskId']), cv2.IMREAD_GRAYSCALE )
                , .5, 1, cv2.THRESH_BINARY)[1]
            mask = cv2.resize(mask, (800, 128), interpolation = cv2.INTER_AREA)
            y[i] = mask

        return X, y.reshape((*y.shape, 1))


###########################################################################################################


def get_callback(patient) :
    ES = EarlyStopping(
        monitor='loss', 
        patience=patient, 
        mode='max', 
        verbose=1)
    RR = ReduceLROnPlateau(
        monitor = 'loss', 
        factor = 0.5, 
        patience = patient / 2, 
        min_lr=0.000001, 
        verbose=1, 
        mode='max')
    CP = ModelCheckpoint(
        os.path.join('model', 'model.h5'), 
        monitor='val_dice_coef', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False,
        mode='max')
    #return [tensorboard_callback, ES, RR, CP]
    return [RR, CP]


###########################################################################################################


import cv2
from timeit import default_timer as timer
from datetime import timedelta


def get_predicted_defects_mask(
    model
    , img
    , img_is_lightweight = False
    , verbose = 0
) :
    '''
    returns the predicted mask from an input image

    Parameter:
    model (keras.model): the steel sheet defect segmentation (U-Net34) model
    img (numpy.array): RGB input image
    img_is_lightweight (boolean):
        - True if the input image is grayscale and of height and width that are expected by the model
        - False if the input image is in color and of height and width twice as large as the ones accepted by the model as its input.
    verbose (int): 0 or 1, wether or not the prediction time is to be printed on stdout

    Returns: 
    numpy.array: numpy array of the mask of same height and width as the input
    '''

    # perform intended preprocessing (incl. downgrading img to grayscale and making its height and width 2 times smaller)
    if img_is_lightweight :
        prediction_shape = (img.shape[0], img.shape[1])
        input = preprocess( img ) / 255.
    else :
        prediction_shape = [int(x/2) for x in (img.shape[0], img.shape[1])]
        input = cv2.cvtColor(
            preprocess(
                cv2.resize( img, (prediction_shape[1], prediction_shape[0]), interpolation = cv2.INTER_AREA)
            )
            , cv2.COLOR_BGR2GRAY
        ) / 255.

    # make the input a 4D tensor
    input = np.resize(
        input, (1, prediction_shape[0], prediction_shape[1], 1)
    )

    start = timer()
    predicted_mask = model.predict( input )
    if verbose == 1 : print( "prediction made in " + "{:.3f}".format( timedelta(seconds=timer()-start).total_seconds() ) + " seconds" )

    # from 4D (1, x, y, 1) tensor to 2D (x, y) grayscale image
    predicted_mask = np.reshape( predicted_mask, prediction_shape )

    # bring predicted mask height and width up to twice what they were
    if not img_is_lightweight :
        predicted_mask = cv2.resize( predicted_mask, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
    else :
        predicted_mask = cv2.resize( predicted_mask, (img.shape[1]*2, img.shape[0]*2), interpolation = cv2.INTER_AREA)

    return predicted_mask














































