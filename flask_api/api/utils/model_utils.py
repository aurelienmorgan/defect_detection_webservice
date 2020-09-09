

import cv2
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import sys
import traceback


from segmentation_models import get_preprocessing

# SINCE WE LOAD U-NET WITH PRETRAINING FROM IMAGENET =>
preprocess = get_preprocessing('resnet34') # for resnet, img = (img-110.0)/1.0


IMG_HEIGHT = 256
IMG_WIDTH = 1600

from api import views

def get_predicted_defects_mask(
    model
    , img
    , img_is_lightweight = False
    , enlarge_by = 1
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
    enlarge_by (int): factor by which the height and width of the output must be enlarged
                      compared to those of the input 'img'.
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
    #print( "views.tf_session : " + str( views.tf_session is None ) )
    #print( "views.tf_graph : " + str( views.tf_graph is None ) )
    with views.tf_session.as_default() :
        with views.tf_graph.as_default() :
            try :
                predicted_mask = model.predict( input )
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print( exc_type )
                print( exc_value )
                if (str(exc_value).startswith(
                    "Failed to get convolution algorithm. " +
                    "This is probably because cuDNN failed to initialize")) :
                    print()
                    print( "##################################################################################################" )
                    print( "## DO YOU HAVE ANY OTHER RUNNING TENSORFLOW SESSION USING GPU                                   ##" )
                    print( "## AND NO MEMORY ALLOCATION SAFEGUARD ON IT                                                     ##" )
                    print( "## SUCH AS 'config.gpu_options.allow_growth = True' OR                                          ##" )
                    print( "## 'config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))' ?? ##" )
                    print( "## ANY RUNNING TENSORFLOW SESSION AT ALL ??? NO JUPYTER NOTEBOOK ??                             ##" )
                    print( "##################################################################################################" )
                    print()
                # Extract unformatter stack traces as tuples
                trace_back = traceback.extract_tb(exc_traceback)
                trace = trace_back[0]
                print( "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]) )

                return None
    if verbose == 1 : print( "prediction made in " + "{:.3f}".format( timedelta(seconds=timer()-start).total_seconds() ) + " seconds [" + '{0:2.2%}'.format( np.amax(predicted_mask) // 0.0001 / 10000 ) + "]" )

    # from 4D (1, x, y, 1) tensor to 2D (x, y) grayscale image
    predicted_mask = np.reshape( predicted_mask, prediction_shape )


    if not img_is_lightweight :
        enlarge_by *= 2
    if enlarge_by != 1 :
        predicted_mask = cv2.resize(
           predicted_mask, (prediction_shape[1]*enlarge_by, prediction_shape[0]*enlarge_by)
           , interpolation = cv2.INTER_AREA)


    return predicted_mask












































