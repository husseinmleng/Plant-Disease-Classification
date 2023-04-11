#Stage 1: Import all project dependencies
import numpy as np
import keras
import cv2
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array
from flask import Flask, jsonify 

#Stage 2: Load the pretrained model
model = keras.models.load_model('/home/hussein/Jupyter NoteBooks/Graduation Project/best_weights_96.hdf5')
#Stage 3: Creating the Flask API
#Starting the Flask application
app = Flask(__name__)

#Defining the classify_image function
@app.route('/api/v1/<string:img_name>/<label>',methods=['POST','GET'])
def classify_img(img_name,label):
    upload_dir = '/home/hussein/Jupyter NoteBooks/Graduation Project/Flask+API/uploads/'
    img = load_img(upload_dir + img_name,target_size=(224,224))
    img = img_to_array(img)
    img = cv2.resize(img,(224,224))     # resize image to match model's expected sizing
    img = img.reshape(1,224,224,3)
    class_names = []
    with open('/home/hussein/Jupyter NoteBooks/Graduation Project/Flask+API/labels.txt','r') as names:
        for line in names.readlines():
            class_names.append(line.strip())
    prediction = model.predict(img)
    prediction = class_names[np.argmax(prediction[0])]
    response =  {'True Label' : label, 'Prediction ': prediction}
    return jsonify(response)
    
    
#Start the Flask application
app.run(port=5000,debug=False)