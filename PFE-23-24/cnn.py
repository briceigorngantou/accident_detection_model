from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the image and predict
            img = keras_image.load_img(file_path, target_size=(64, 64))
            img_array = keras_image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            prediction = model.predict(img_array)
            result = 'Accident Detected' if prediction[0] > 0.5 else 'No Accident Detected'
            
            # Generate output file
            output_path = file_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            with open(output_path, 'w') as f:
                if prediction[0] > 0.5:
                    f.write("0.0, 0.515625, 0.6015625, 0.3359375, 0.43984375")
            return render_template('result.html', result=result)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
