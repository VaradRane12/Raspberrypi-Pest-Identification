from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from picamera2 import Picamera2
import tensorflow as tf
import RPi.GPIO as GPIO
import time
import threading

def blink_led(duration=10, interval=0.5):
    end_time = time.time() + duration
    while time.time() < end_time:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(interval)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(interval)

MODEL_PATH = "model.tflite"
IMG_SIZE = (224, 224)
model_active = True  # Controls pause/resume

SELECTED_CLASSES = [
    '62', '61', '56', '73', '80', '75', '65', '43', '72', '98',
    '79', '15', '81', '63', '25', '35', '96', '31', '74', '82',
    '53', '78', '94', '30', '67', '85', '36', '58', '48', '14'
]

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

# Load TFLite Model 
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded.")

# Camera Setup 
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Flask App 
app = Flask(__name__)
last_label = "Loading..."

LED_PIN = 17  # or whichever GPIO pin you're using
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Start with LED off

def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def generate_frames():
    global model_active, last_label
    last_prediction = None
    prediction_count = 0
    stable_label = "Detecting..."
    min_confidence = 0.70
    required_stability = 2

    while True:
        GPIO.output(LED_PIN, GPIO.HIGH if model_active else GPIO.LOW)

        frame = picam2.capture_array()

        if model_active:
            input_tensor = preprocess(frame)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

            idx = np.argmax(prediction)
            confidence = prediction[idx]
            folder_num = SELECTED_CLASSES[idx]
            pest_name = PEST_NAMES.get(folder_num, "Unknown Pest")

            if confidence > min_confidence and folder_num == last_prediction:
                prediction_count += 1
            else:
                prediction_count = 1
                last_prediction = folder_num

            if prediction_count >= required_stability:
                last_label = f"{pest_name} ({confidence * 100:.2f}%)"
                stable_label = last_label
        else:
            stable_label = "Paused"

        # Display label on frame
        color = (0, 255, 0) if model_active else (0, 0, 255)
        cv2.putText(frame, stable_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Raspberry Pi Pest Detection</title>
        <style>
            body { background: #f0f0f0; font-family: sans-serif; text-align: center; padding: 20px; }
            h1 { color: #333; }
            .frame { border: 5px solid #4CAF50; border-radius: 8px; display: inline-block; margin: 10px; }
            .label { font-size: 18px; font-weight: bold; margin-top: 10px; color: #444; }
            button { padding: 10px 20px; margin: 10px; font-size: 16px; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Raspberry Pi Pest Detection</h1>
        <div class="frame"><img src="{{ url_for('video') }}" width="640" height="480"></div>
        <div class="label">Current Prediction: <span id="label">{{ label }}</span></div>
        <div>
            <button onclick="fetch('/pause')">Pause</button>
            <button onclick="fetch('/resume')">Resume</button>
        </div>

        <script>
            setInterval(async () => {
                const response = await fetch('/label');
                const data = await response.text();
                document.getElementById('label').innerText = data;
            }, 1000);
        </script>
    </body>
    </html>
    """, label=last_label)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/label')
def label():
    return last_label

@app.route('/pause')
def pause():
    global model_active
    model_active = False
    return "Model Paused"

@app.route('/resume')
def resume():
    global model_active
    model_active = True
    return "Model Resumed"

if __name__ == '__main__':
    GPIO.output(LED_PIN, GPIO.LOW)

    # Start LED blinking in a background thread
    threading.Thread(target=blink_led, args=(10,), daemon=True).start()
    app.run(host='0.0.0.0', port=8000)

