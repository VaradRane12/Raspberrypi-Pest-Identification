from flask import Flask, Response
import cv2
import numpy as np
from picamera2 import Picamera2
import tensorflow as tf

# ==== CONFIG ====
MODEL_PATH = "model.tflite"  # Changed to TFLite model
IMG_SIZE = (224, 224)

SELECTED_CLASSES = [ 
    # ... (same as original) ...
]

PEST_NAMES = {
    # ... (same as original) ...
}

# ==== Load TFLite Model ====
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded.")

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
    img = np.array(img, dtype=np.float32) / 255.0  # Maintain float32 conversion
    return np.expand_dims(img, axis=0)

def generate_frames():
    while True:
        frame = picam2.capture_array()

        # Preprocess and make prediction
        input_data = preprocess(frame)
        
        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

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

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ... (rest of the Flask routes remain the same) ...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)