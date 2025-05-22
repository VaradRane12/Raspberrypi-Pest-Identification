from flask import Flask, Response
import cv2
import numpy as np
from picamera2 import Picamera2
import tensorflow as tf

# ==== CONFIG ====
MODEL_PATH = "insect_identification.keras"
IMG_SIZE = (224, 224)

SELECTED_CLASSES = [ '62', '61', '56', '73', '80', '75', '65', '43', '72', '98',
    '79', '15', '81', '63', '25', '35', '96', '31', '74', '82',
    '53', '78', '94', '30', '67', '85', '36', '58', '48', '14' ]

PEST_NAMES = {
    '62': "Brevipoalpus lewisi McGregor", '61': "Colomerus vitis", '56': "alfalfa seed chalcid",
    '73': "Erythroneura apicalis", '80': "Chrysomphalus aonidum", '75': "Panonchus citri McGregor",
    '65': "Pseudococcus comstocki Kuwana", '43': "beet weevil", '72': "Trialeurodes vaporariorum",
    '98': "Chlumetia transversa", '79': "Ceroplastes rubens", '15': "grub",
    '81': "Parlatoria zizyphus Lucus", '63': "oides decempunctata", '25': "aphids",
    '35': "wheat sawfly", '96': "Salurnis marginella Guerr", '31': "bird cherry-oataphid",
    '74': "Papilio xuthus", '82': "Nipaecoccus vastalor", '53': "therioaphis maculata Buckton",
    '78': "Unaspis yanonensis", '94': "Dasineura sp", '30': "green bug",
    '67': "Ampelophaga", '85': "Dacus dorsalis(Hendel)", '36': "cerodonta denticornis",
    '58': "Apolygus lucorum", '48': "tarnished plant bug", '14': "rice shell pest"
}

# ==== Load Model ====
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ==== Camera Setup ====
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# ==== Flask App ====
app = Flask(__name__)

def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def generate_frames():
    while True:
        frame = picam2.capture_array()

        input_tensor = preprocess(frame)
        prediction = model.predict(input_tensor)[0]
        idx = np.argmax(prediction)
        folder_num = SELECTED_CLASSES[idx]
        pest_name = PEST_NAMES.get(folder_num, "Unknown Pest")
        confidence = prediction[idx]

        label = f"{pest_name} ({confidence*100:.2f}%)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in multipart response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "<h2>Raspberry Pi Pest Detection</h2><img src='/video'>"

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
