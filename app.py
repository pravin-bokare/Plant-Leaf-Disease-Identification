import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename



app = Flask(__name__)

MODEL = tf.keras.models.load_model("models/1")

CLASS_NAMES = ['Grape Black Rot', 'Grape Esca (Black_Measles)', 'Grape Leaf_blight (Isariopsis_Leaf_Spot)',
               'Grape Healthy', 'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy', 'Tomato Bacterial Spot',
               'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
               'Tomato Spider Mites', 'Tomato Target Spot', 'Tomato Tomato Yellow Leaf', 'Tomato Mosaic Virus',
               'Tomato Healthy']


def model_predict(img_path, MODEL):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    predictions = MODEL.predict(x)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])*100
    return f'Disease : {predicted_class} ,  Accuracy : {round(confidence, 2)}'


@app.route('/', methods=['GET'])
def index():
    return render_template('/index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, MODEL)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
