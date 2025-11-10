from flask import Flask, render_template, request, flash, redirect
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,session,flash,redirect, url_for, session,flash
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras



app = Flask(__name__)
app.secret_key = '1a2b3c4d5e'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'accounts'

# Intialize MySQL
mysql = MySQL(app)

@app.route('/login', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' :
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        print(username)
        print(password)
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s AND password = %s', (username, password))
        print('SELECT * FROM accounts WHERE email = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        print(account)
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            #session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return render_template('home.html')#redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            flash("Incorrect username/password!", "danger")
    return render_template('login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    print("test result")
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' :
        print("test result resultrss")
        # Create variables for easy access
        username = request.form['un']
        password = request.form['pwd']
        email = request.form['em']
        ph = request.form['cn']
        br = request.form['br']
        clg = request.form['clg']
        adr = request.form['adr']
        cpwd = request.form['cpwd']
        
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM accounts WHERE username = %s', (username))
        cursor.execute( "SELECT * FROM accounts WHERE username LIKE %s", [username] )
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        elif cpwd != password :
            flash("Password Should Match", "danger")
        else:
        # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (%s, %s, %s,%s, %s, %s, %s)', (username,email, password,ph,br,clg,adr))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
            return render_template('login.html',title="Login")

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    return render_template('register.html',title="Register")

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


def predictimg(fname):
    model = keras.models.load_model("newsofttissuetumor.h5")
    class_names = ['Benign', 'Malignant', 'No Tumor']# <-- update with your real classes
    img_path = fname   # <-- change to your test image
    img_size = (224, 224)   # must match training
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension
    img_array = keras.applications.efficientnet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    confidence=round(confidence, 2)
    return(predicted_class,confidence)


class SafeGroupNormalization(keras.layers.Layer):
    def __init__(self, groups=8, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.init_groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.groups = min(self.init_groups, dim)
        self.gamma = self.add_weight(name="gamma", shape=(dim,), initializer="ones", trainable=True)
        self.beta  = self.add_weight(name="beta",  shape=(dim,), initializer="zeros", trainable=True)

    def call(self, x):
        input_shape = tf.shape(x)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        G = tf.minimum(self.groups, C)
        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, axes=[1,2,4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [N, H, W, C])
        return self.gamma * x + self.beta

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def iou_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    total = tf.reduce_sum(y_true + y_pred)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + (1.0 - dice_coef(y_true, y_pred))


# ===============================
# Load pretrained GM-UNet model
# ===============================
model = keras.models.load_model(
    "gmunet_final.h5",   # <-- your trained model path
    custom_objects={
        "SafeGroupNormalization": SafeGroupNormalization,
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "iou_coef": iou_coef
    }
)

# ===============================
# Preprocessing function
# ===============================
IMG_SIZE = (256, 256)

def read_image(path, size=IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def imgseg(fname):
    test_img_path = fname
    img = read_image(test_img_path)
    inp = tf.expand_dims(img, axis=0)

    # Predict mask
    pred_mask = model.predict(inp)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    #cv2.imwrite('static/imgseg.png',tf.squeeze(pred_mask))
    mask = tf.squeeze(pred_mask).numpy()
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_box = (img.numpy() * 255).astype(np.uint8).copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # ignore small noise
            cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)

    mask_save_path = os.path.join('static', "pred_mask.png")
    tumor_save_path = os.path.join('static', "tumor_marked.png")

    cv2.imwrite(mask_save_path, mask_uint8)
    cv2.imwrite(tumor_save_path, cv2.cvtColor(img_with_box, cv2.COLOR_RGB2BGR))
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("MRI Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(pred_mask), cmap="gray")

    plt.title("Predicted Mask")
    plt.axis("off")
    plt.savefig("static/imgseg.png", bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    #plt.show()
    

@app.route("/result1")
def result1():
    pred=session.get('cname')
    print(pred)

    return render_template("result2.html",pred=pred)

@app.route("/result2")
def result2():
    

    return render_template("result1.html")


@app.route("/predict", methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)
        cname,confi=predictimg(file_path)
        session['cname']=cname
        
        imgseg(file_path)
        return render_template('result.html', pred = cname,desc=confi,fname=f.filename)
    return render_template('home.html')


if __name__ == '__main__':
	app.run(debug = True,host='0.0.0.0')
