#from jinja2 import Template
from tensorflow.python.keras.saving.saved_model.load import load
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
#from tensorflow.keras.preprocessing import image
import numpy as np
import os
import keras
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)


img_size = (299,299)
preprocess_input = keras.applications.xception.preprocess_input
#last_conv_layer_name = "block14_sepconv2_act"

# Loading our trained model for Atelectasis
atelectasis_model = load_model('static/all models/Atelectasis.h5' , compile=True)
cardiomegaly_model = load_model('static/all models/Cardiomegaly.h5', compile=True)
consolidation_model = load_model('static/all models/Consolidation.h5' , compile=True)
edema_model = load_model('static/all models/Edema.h5' , compile=True)
effusion_model = load_model('static/all models/Effusion.h5' , compile=True)
emphysema_model = load_model('static/all models/Emphysema.h5' , compile=True) 
fibrosis_model= load_model('static/all models/Fibrosis.h5',compile=True)
hernia_model = load_model('static/all models/Hernia.h5',compile=True)
infiltration_model = load_model('static/all models/Infiltration.h5' , compile=True)
lungopacity_model = load_model('static/all models/lung_opacity.h5' , compile=True)


# picfolder = os.path.join('static','upload')
# app.config['UPLOAD_FOLDER'] = picfolder

picfolder1 = os.path.join('static','uploadseparate')
#app.config['SEPARATE_UPLOAD_FOLDER'] = picfolder1

# path = "static/upload"
path1 = "static/uploadseparate"
# start of gradcam


def model_predict_for_disease(img_path, model):
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    preds = model.predict(img_array)
    return preds

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array
'''
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
'''
#endof gradcam



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/aboutus')
def About():
    return render_template('aboutus.html')


@app.route('/dev')
def dev():
    return render_template('devTeam.html')


@app.route('/upload')
def upload():
    return render_template('separatetest.html')


@app.route('/separatetest')
def separatetest():
    return render_template('separatetest.html')







# @app.route('/uploadingimage' ,  methods=['GET','POST'])
# def uploadingimage():
#     if request.method == 'POST':
#         f = request.files['chest-x-ray']
#         file1_path = os.path.join(path,secure_filename(f.filename))
#         f.save(file1_path)
#         pic1 = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
#         # Make Prediction
#         atelectasis_result = model_predict_for_atelactasis(file1_path, atelectasis_model)
#         return render_template('upload.html' , atelectasis_result = atelectasis_result
         
#          )

@app.route('/separateupload' ,  methods=['GET','POST'])
def separateupload():
    final_result = ""
    disease_name = ""
    probabilityofdisease = 0
    if request.method == 'POST':
        selectopt = request.form.get('selectopt')
        f = request.files['chest-x-ray']
        file1_path = os.path.join(path1,secure_filename(f.filename))
        f.save(file1_path)
        if selectopt == '1':
            disease_name = "Atelectasis"
            #Make Prediction
            atelectasis_result = model_predict_for_disease(file1_path, atelectasis_model)
            i = np.argmax(atelectasis_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = atelectasis_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = atelectasis_result[0][1]*100
        
        
        elif selectopt == '2':
            disease_name = "Cardiomegaly"
            #Make Prediction
            cardiomegaly_result = model_predict_for_disease(file1_path, cardiomegaly_model)
            i = np.argmax(cardiomegaly_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = cardiomegaly_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = cardiomegaly_result[0][1]*100
        
        
        elif selectopt == '3':
            disease_name = "Consolidation"
            #Make Prediction
            consolidation_result = model_predict_for_disease(file1_path, consolidation_model)
            i = np.argmax(consolidation_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = consolidation_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = consolidation_result[0][1]*100
        
        
        elif selectopt == '4':
            disease_name = "Edema"
            #Make Prediction
            edema_result = model_predict_for_disease(file1_path, edema_model)
            i = np.argmax(edema_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = edema_result[0][0]*100
            else:
                final_result = "Normal"
                probabilityofdisease = edema_result[0][1]*100
        
        
        elif selectopt == '5':
            disease_name = "Effusion"
            #Make Prediction
            effusion_result = model_predict_for_disease(file1_path, effusion_model)
            i = np.argmax(effusion_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = effusion_result[0][0]*100
            else:
                final_result = "Normal"
                probabilityofdisease = effusion_result[0][1]*100
        
        
        elif selectopt == '6':
            disease_name = "Emphysema"
            #Make Prediction
            emphysema_result = model_predict_for_disease(file1_path, emphysema_model)
            i = np.argmax(emphysema_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = emphysema_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = emphysema_result[0][1]*100
        
        
        elif selectopt == '7':
            disease_name = "Fibrosis"
            #Make Prediction
            fibrosis_result = model_predict_for_disease(file1_path, fibrosis_model)
            i = np.argmax(fibrosis_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = fibrosis_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = fibrosis_result[0][1]*100
        
        
        elif selectopt == '8':
            disease_name = "Hernia"
            #Make Prediction
            hernia_result = model_predict_for_disease(file1_path, hernia_model)
            i = np.argmax(hernia_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = hernia_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = hernia_result[0][1]*100
        
        
        elif selectopt == '9':
            disease_name = "Infiltration"
            #Make Prediction
            infiltration_result = model_predict_for_disease(file1_path, infiltration_model)
            i = np.argmax(infiltration_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = infiltration_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = infiltration_result[0][1]*100
        
        
        elif selectopt == '10':
            disease_name = "Lung Opacity"
            #Make Prediction
            lungopacity_result = model_predict_for_disease(file1_path, lungopacity_model)
            i = np.argmax(lungopacity_result[0])
            if i==0:
                final_result = disease_name
                probabilityofdisease = lungopacity_result[0][0]*100
            else:
                final_result = "Normal" 
                probabilityofdisease = lungopacity_result[0][1]*100



    return render_template('separatetest.html' , final_result = final_result , probabilityofdisease = probabilityofdisease ,
    disease_name = disease_name)


if __name__ == '__main__':
    app.run(debug=True)
