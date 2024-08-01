from flask import Flask, request, render_template
import pickle
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog


app = Flask(__name__)

# Charger le modèle
model_path = 'random_forest_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Vérifier si le fichier est présent dans la requête
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message='No selected file')
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            # Prédire à partir de l'image
            image = Image.open(file_path)
            bounding_box = (0.0, 0.515625, 0.6015625, 0.3359375, 0.43984375) 
            # image = prepare_image(image)  # Supposons que cette fonction prépare vos données
            image = extract_features(file_path)
            print(image)
            prediction = model.predict([image])
            print(prediction)
            result = 'Accident' if prediction[0] == 0 else 'No Accident'
            return render_template('upload.html', message=f'Prediction: {result}')
    return render_template('upload.html')

def prepare_image(img):
    # Cette fonction doit être ajustée en fonction de la façon dont le modèle a été entraîné
    img = img.resize((128, 128))
    img = np.array(img)
    img = img.flatten()
    return img

def extract_features(image_path):
    image = imread(image_path)
    image_gray = rgb2gray(image)
    image_resized = resize(image_gray, (128, 64), anti_aliasing=True)

    # Extraction des caractéristiques HOG
    fd_hog = hog(image_resized, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), feature_vector=True)

    return fd_hog


if __name__ == '__main__':
    app.run(debug=True)
