from app import app

import os

#from waitress import serve

debug_=False # True # 



if __name__ == "__main__":
    path = os.path.abspath(os.path.dirname(__file__))
    os.environ[ "STATIC_FOLDER" ] = \
        os.path.abspath( path + "/app/static" )
    os.environ[ "MEDIA_FOLDER" ] = \
        os.path.abspath( path + "/app/media-offline" )

    app.run(debug=debug_)
    #serve(app, host='0.0.0.0', port=80)