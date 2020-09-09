
from api import api

import os
from flask import request, jsonify

tf_session = None
tf_graph = None
model_reconstructed = None

from api.utils.model_utils import get_predicted_defects_mask, IMG_HEIGHT, IMG_WIDTH
from .utils.mask_utils import image_is_grayscale, prediction_to_json


# web api

import tensorflow as tf
import cv2
import imghdr

import json

@api.route('/')
@api.route('/index')
def index():
    global model_reconstructed
    print( os.path.abspath(__file__) )
    print( ' - Tensorflow ' + tf.__version__ )
    print( model_reconstructed.layers[2].name )

    return "Hello"


@api.route("/get_defect_contour", methods = [ "GET", "POST" ])
def predict() :
    global model_reconstructed

    if request.method == 'POST' :
        # curl -d "filename=0000f269f.jpg&enlarge_by=2" http://127.0.0.1:5000/get_defect_contour
        filename = request.form.get('filename')
        enlarge_by = request.form.get('enlarge_by')
    else :
        # curl "http://127.0.0.1:5000/get_defect_contour?filename=000ccc2ac.jpg&enlarge_by=2"
        filename = request.args.get('filename')
        enlarge_by = request.args.get('enlarge_by')


    if filename is None :
        f = request.form
        for key in f.keys():
            for value in f.getlist(key):
                print( "'",key,"':'",value,"'" )
        return "filename missing", 403
    if enlarge_by is None :
        enlarge_by = 1
    else :
        try :
            enlarge_by = int(enlarge_by)
        except :
            enlarge_by = 1

    fileFullname = os.path.join(os.getenv("MEDIA_FOLDER"), filename)
    #print( fileFullname + " - fileexists : " + str( os.path.isfile(fileFullname) ) )
    if not os.path.isfile(fileFullname) :
        return "file not found '" + fileFullname + "'", 403
    if imghdr.what( fileFullname ) is None :
        return "not an image file '" + fileFullname + "'", 403


    #########################################################################
    ## Ensuring that the image is gray-scaled.                             ##
    #########################################################################
    img = cv2.imread( fileFullname )
    img_clone = img.copy()
    if( image_is_grayscale(img_clone) ) :
        if len(img_clone.shape) == 3 :
            if img_clone.shape[2]  == 1 :
                img_clone = np.resize(img_clone, (img_clone.shape[0], img_clone.shape[1]))
                img = np.squeeze(np.stack((img,)*3, axis=-1))
            else :
                # grayscale image encoded on 3 channels => address that
                img_clone = cv2.cvtColor( img_clone, cv2.COLOR_BGR2GRAY )
        else :
            img = np.stack((img,)*3, axis=-1)
    else :
        img_clone = cv2.cvtColor( img_clone, cv2.COLOR_BGR2GRAY )


    #########################################################################
    ## REMINDER: the input images are of dimension (IMG_HEIGHT, IMG_WIDTH) ##
    ## and the model takes images of dimension (IMG_HEIGHT/2, IMG_WIDTH/2) ##
    ## as inputs.                                                          ##
    #########################################################################
    img_width = img_clone.shape[1] ; img_height = img_clone.shape[0]
    if (img_height, img_width) != (IMG_HEIGHT, IMG_WIDTH) :
        if (img_height, img_width) != (IMG_HEIGHT/2, IMG_WIDTH/2) :
            return "image dimensions incorrect '" + str((img_height, img_width)) + "'", 403
    else :
        img_clone = \
            cv2.resize( img_clone, (int(IMG_WIDTH/2), int(IMG_HEIGHT/2))
                 , interpolation = cv2.INTER_AREA)



    #########################################################################
    ## Generating the predicted defect mask.                               ##
    #########################################################################
    predicted_mask = get_predicted_defects_mask(
        model_reconstructed
        , img_clone
        , img_is_lightweight = True
        , enlarge_by = 2*enlarge_by #since the input/output model sizes are halfed when 'img_is_lightweight'
        , verbose = 1
    )
    
    filename = filename.split(".")[0]
    json_prediction = prediction_to_json( predicted_mask
                                         , img=img, filename = filename + "_mask.png"  )
    with open(os.path.join(os.getenv("MEDIA_FOLDER"), filename + "_" + str(enlarge_by) + '.json')
              , 'w') as f:
        json.dump(json_prediction, f)



    return json_prediction, 200

