from app import app

from run import debug_

import os
from subprocess import call
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join
from flask import (
    Flask,
    send_from_directory,
    render_template,
    url_for,
    redirect,
    request
)

import requests
from urllib.parse import quote_plus
import json

# web app

@app.route('/about')
def about():
    return render_template('about.html', page_title = 'About')

@app.route('/')
@app.route('/home')
@app.route('/index')
def index():
    # print( os.path.abspath(__file__) )
    return render_template( 'home.html' )

#@app.route("/static/<path:filename>")
#def staticfiles(filename):
#    return send_from_directory(os.getenv("STATIC_FOLDER"), filename)

@app.route("/media/<path:filename>")
def mediafiles(filename):
    path = os.path.join( os.getenv("MEDIA_FOLDER"), filename )
    #print( "mediafiles - " + os.path.basename(path) + " -- " + os.path.dirname(path))
    path_split = os.path.split(path)

    return send_from_directory( path_split[:-1][0], path_split[-1:][0])

@app.route("/upload", methods = [ "GET", "POST" ])
def upload_file():
    print( os.getenv("MEDIA_FOLDER") + " - debug : " + str(debug_) )
    if request.method == "POST" :
        file = request.files[ "file" ]
        filename = secure_filename( file.filename )
        file.save( os.path.join( os.getenv( "MEDIA_FOLDER" ), filename ))
        
        return redirect(url_for('prediction', filename=filename))
    
    # case "redirected from failed prediction" BEGIN #
    status_code = request.args.get('status_code')
    info_message = request.args.get('info_message')
    if not status_code is None and not info_message is None :
        return render_template( 'upload.html', page_title = 'Upload - ' + str(status_code)
                                , info_class = 'class=error'
                                , info_message = \
                                    str(status_code) + " - " + info_message
                                )
    # case "redirected from failed prediction" END #

    return render_template( 'upload.html', page_title = 'Upload' )


@app.route('/prediction')
def prediction():
    filename = request.args.get('filename')

    if filename is None :
        return redirect(url_for('upload_file', filename=filename), code = 303)

    #print( os.path.join( os.getenv( "MEDIA_FOLDER" ), filename ) + " - " + str( request.content_length ) )
    if not debug_ :
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"}
        url = \
            "http://defect-api-service:9090/get_defect_contour?" + \
            "filename=" + quote_plus(filename) + "&enlarge_by=2"
        try :
            response = requests.get(url=url, headers=headers, timeout=None)
        except requests.exceptions.Timeout as tEx :
            print( tEx )
            return "requests (defect-api-service) timeout", 500

        if response.status_code == 200 :
            prediction = response.json()
            #print(prediction)
            return render_template( 'prediction.html', page_title = 'Prediction'
                                    , prediction = prediction )
        else :
            return redirect(url_for('upload_file', filename=filename) +
                            "&status_code=" + str(response.status_code) +
                            "&info_message=" + quote_plus(response.text)
                            , code = 303)
    else :
        prediction = json.loads('{"filename": \"' + filename +
                               '\", "max_prob": 0.9999974966049194, "defect_thresh": 0.4392156862745098, "contour_pixels": [[0, 571], [0, 572], [0, 594]]}')
             
        print(prediction)
        return render_template( 'prediction.html', page_title = 'Prediction'
                                , prediction = prediction )




