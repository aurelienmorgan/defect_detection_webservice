
import numpy as np
import json
import cv2
import matplotlib ; matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename

def image_is_grayscale( img ) :
    '''
    indicats whether or not the image is grayscale
    
    Parameter:
    img (numpy.ndarray) : an image

    Returns: a boolean indicating whether or not the image is grayscale
    '''

    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all():
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        return True

    return False



def mask_add_pad( mask, pad = 2 ) :
    '''
    enlarge mask to add space around reported defects
    '''

    w = mask.shape[1]
    h = mask.shape[0]

    # upper bound
    for k in range( 1, pad, 2 ) :
        temp = np.concatenate(
            [mask[k:,:],np.zeros((k,w))],
            axis = 0
        )
        mask = np.logical_or(mask,temp)

    # lower bound
    for k in range( 1, pad, 2 ) :
        temp = np.concatenate(
            [np.zeros((k,w)),mask[:-k,:]],
            axis = 0
        )
        mask = np.logical_or(mask,temp)

    # left bound
    for k in range( 1, pad, 2 ) :
        temp = np.concatenate(
            [mask[:,k:],np.zeros((h,k))],
            axis = 1
        )
        mask = np.logical_or(mask,temp)

    # right bound
    for k in range( 1, pad, 2 ) :
        temp = np.concatenate(
            [np.zeros((h,k)),mask[:,:-k]],
            axis = 1
        )
        mask = np.logical_or(mask,temp)


    return mask



def mask_to_contour( mask, width = 3 ):
    '''
    convert mask to its contour
    '''

    w = mask.shape[1]
    h = mask.shape[0]

    mask2 = np.concatenate(
        [mask[ : , width : ], np.zeros( (h,width) )],
        axis = 1
    )
    mask2 = np.logical_xor( mask, mask2 )

    mask3 = np.concatenate(
        [mask[ width : , : ], np.zeros( (width,w) )],
        axis = 0
    )
    mask3 = np.logical_xor( mask, mask3 )

    return np.logical_or( mask2, mask3 )



def contour_to_coordinates( contour ) :
    '''
    convert contour to pairs of coordinates
    '''

    result = []
    for point, value in np.ndenumerate( contour ):
        if( value ) :
            result.append( point )


    return result



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def prediction_to_json( predicted_mask, pad = 2, img = None, filename = 'temp.png' ) :
    '''
    returns a jsonified predicted mask

    Parameter:
    predicted_mask (numpy.array): the steel sheet defect predicted mask; pixels are each assigned a defect probability
                                  (thus are each in the [0-1] range)
    pad (int): width of the padding added to the predicted mask
    img (np.ndarray) : a color image for defect overlay. The result will be saved locally.
    filename (string) the filename assigned to the saved overlayed image (ignored if img = None).

    Returns: 
    a json-serialized string. The represented object has three (or four) attributes:
       - max_prob: highest value of all the pixel probabilities to be a defect from the input predicted_mask
       - defect_thresh: the threshold value as returned by the OTSU method for image binarization (pixel == defect y/n)
       - contour_pixels: a list of coordinates of pixels belonging to the defect contour
       - filename: the relative path to the saved overlayed image.
    
    '''


    predicted_mask_otsu = (predicted_mask*255).astype('uint8')
    (thresh, predicted_mask_otsu) = cv2.threshold(
        predicted_mask_otsu, np.amax(predicted_mask_otsu)/2.,
        np.amax(predicted_mask_otsu),
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    predicted_contour_otsu = mask_to_contour( mask_add_pad( predicted_mask_otsu, pad = pad ) , width = 1 )


    data = {}
    data['max_prob'] = np.amax(predicted_mask)
    data['defect_thresh'] = thresh / 255.
    data['contour_pixels'] = contour_to_coordinates( predicted_contour_otsu )


    if not img is None :
        enlarge_by = int(predicted_contour_otsu.shape[0] / img.shape[0])
        if enlarge_by != 1 :
            # need to make the contaour thicker (or will look discontinuous)
            predicted_contour_otsu = mask_to_contour( mask_add_pad( predicted_mask_otsu, pad = pad )
                                                     , width = enlarge_by )
            # resize the input image for rendering
            img = cv2.resize( img, (predicted_contour_otsu.shape[1], predicted_contour_otsu.shape[0])
                 , interpolation = cv2.INTER_AREA)
        img[ predicted_contour_otsu==1, 1 ] = 255 # green channel
        fig = plt.figure(frameon=False, figsize=(predicted_contour_otsu.shape[1], predicted_contour_otsu.shape[0]), dpi=1)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, aspect='auto')
        filename = secure_filename(filename)
        #print( filename + " - " + str(img.shape) )
        fig.savefig(os.path.join(os.getenv("MEDIA_FOLDER"), filename), dpi=fig.dpi)

        data['filename'] = filename


    return json.dumps(data, cls=MyEncoder)
















































