from flask import Flask, request, send_from_directory, render_template
import analyzer as core_analysis
import json
import base64
from PIL import Image
from log import load_model
from keras.optimizers import SGD
from keras.models import model_from_json
import numpy as np

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='/static')
# ana = core_analysis.HPAnalysis();

@app.route('/')
def get_home():
    return render_template("index.html")

@app.route('/api/evaluate', methods=['POST'])
def hello():
    try:
        json_request = request.get_json(silent=True)
        image_str = json_request['image']
        image_str = image_str[image_str.find(",")+1:]
        byte_string = bytes(image_str, 'cp1252')
        #with open("test-64.txt", 'w') as f:
            #f.write(image_str)
        image_64_decode = base64.decodestring(byte_string)
        with open("test.jpg","wb") as f:
            f.write(image_64_decode)
        #jpgfile = Image.open("test.jpg").convert('LA');
        img = Image.open('test.jpg').convert('L')
        img.thumbnail((48,48), Image.ANTIALIAS)
        data = np.asarray(img.getdata()).reshape(img.size)
        print(data.shape)
        data = data.reshape(-1, 1, data.shape[0], data.shape[1])

        model = load_model('bestModel.json','bestModel.h5');
        #sgd
        lrate = 0.01
        decay = lrate/200
        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #adam
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        y = model.predict(data)
        print(y);

    except Exception as error:
        return json.dumps({'success':False, 'message' : repr(error)}), 500, {'ContentType':'application/json'}
    return json.dumps({'success':True, 'prediction' : y.tolist()}), 200, {'ContentType':'application/json'}
@app.route('/pics/<path:path>')
def get_pics(path):
    return send_from_directory('pics', path)

@app.errorhandler(Exception)
def exception_handler(error):
    return repr(error);

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(port=9009)
