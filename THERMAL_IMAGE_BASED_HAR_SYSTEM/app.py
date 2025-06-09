from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from src.predict import HAR_Predictor

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize prediction pipeline
label_map = {
    0: "walking",
    1: "LOOKING_STRAIGHT",
    2: "STANDING",
    3: "jumping_climbing",
    4: "suspicious_look",
    5: "EXERCISE_BODY_SWING",
    6: "SITTING_STANDING",
    7: "fighting",
    8: "gesturing",
    9: "LOOKING_UP"
}

pipeline = HAR_Predictor(
    yolo_model_path="artifacts/weights/best.pt",
    classifier_path="artifacts/classifiers/mlp.pkl",
    label_map=label_map
)

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="❌ No file part provided.")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="❌ No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            prediction = pipeline.predict(image_path)
            return render_template('index.html', prediction=prediction, image_url=image_path)

    return render_template('index.html', prediction=None)

# Main
if __name__ == '__main__':
    app.run(debug=True)
