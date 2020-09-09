
from api import (
    api
    , views # to register the routes
)

application = api # to let Gunicorn discover it (looking for the callable named "application")


import os

#from waitress import serve


debug_=False # True # 


import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session



def load_model() :
    path = os.path.abspath(os.path.dirname(__file__))

    # model reconstruction from JSON :
    with open(os.path.abspath( os.path.join( path, 'api', 'model', 'my_model_archi.json') ), 'r') as f:
        json_string = f.read()
        
    views.model_reconstructed = model_from_json(json_string)

    # model trained weights loading from h5df :
    views.model_reconstructed.load_weights(
        os.path.abspath( os.path.join( path, 'api', 'model', 'my_model_weights.h5') ) )

    views.tf_graph = tf.get_default_graph()

    print( "MODEL LOADED" )
    #print( views.model_reconstructed.summary() )



if views.model_reconstructed is None :
    config = tf.ConfigProto()

    if debug_ :
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.abspath( os.path.join( path, os.pardir, 'flask_app' ) )
        os.environ[ "MEDIA_FOLDER" ] = \
            os.path.abspath( os.path.join( path, "app", "media-offline" ) )

        config.gpu_options.allow_growth = True # still ok in an envirnoment with no GPU, no worries
    
    views.tf_session = tf.Session(config=config)
    #print( "tf.test.is_gpu_available() : " + str(tf.test.is_gpu_available()) ) # WARNING do not call this prior to setting the session with GPU memory allocation constraints !!!

    #print( "views.tf_session is None : " + str( views.tf_session is None ) )
    set_session(views.tf_session)


    load_model()
    print( "views.model_reconstructed is None : " + str( views.model_reconstructed is None ) )




if __name__ == "__main__" :

    api.run(debug=debug_)
    #serve(api, host='0.0.0.0', port=80)