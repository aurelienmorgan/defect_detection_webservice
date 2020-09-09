import numpy as np
import json
import cv2

# from https://www.kaggle.com/robertkag/rle-to-mask-converter
def rle_to_mask( rle_string, height, width ) :
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters:
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    numpy.array: numpy array of the mask
    '''
    
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 1
        img = img.reshape(cols,rows)
        img = img.T
        return img



def mask_to_rle( img ) :
    '''
    convert an RGB image to an RLE(run length encoding) string

    Parameters: 
    img: an RGB image

    Returns: 
    an RLE(run length encoding) string
    '''
    tmp = np.rot90( np.flipud( img ), k=3 )
    rle = []
    lastColor = 0;
    startpos = 0
    endpos = 0

    tmp = tmp.reshape(-1,1)   
    for i in range( len(tmp) ):
        if (lastColor==0) and tmp[i]>0:
            startpos = i
            lastColor = 1
        elif (lastColor==1) and (tmp[i]==0):
            endpos = i-1
            lastColor = 0
            rle.append( str(startpos)+' '+str(endpos-startpos+1) )
    return " ".join( rle )


##########################################################################################################################################


# this is an awesome little function to remove small spots in our predictions
#from skimage import morphology
#def remove_small_regions(img, size):
#    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
#    img = morphology.remove_small_objects(img, size)
#    img = morphology.remove_small_holes(img, size)
#    return img


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

def prediction_to_json( predicted_mask, pad = 2 ) :
    '''
    returns a jsonified predicted mask

    Parameter:
    predicted_mask (numpy.array): the steel sheet defect predicted mask; pixels are each assigned a defect probability
                                  (thus are each in the [0-1] range)
    pad (int): width of the padding added to the predicted mask

    Returns: 
    a json-serialized string. The represented object has three attributes:
       - max_prob: highest value of all the pixel probabilities to be a defect from the input predicted_mask
       - defect_thresh: the threshold value as returned by the OTSU method for image binarization (pixel == defect y/n)
       - contour_pixels: a list of coordinates of pixels belonging to the defect contour
    
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


    return json.dumps(data, cls=MyEncoder)
















































