import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

TFLITE_MODEL_PATH = "model.tflite"
IMG_SIZE = (224, 224)

SELECTED_CLASSES = [
    '62', '61', '56', '73', '80', '75', '65', '43', '72', '98',
    '79', '15', '81', '63', '25', '35', '96', '31', '74', '82',
    '53', '78', '94', '30', '67', '85', '36', '58', '48', '14'
]

PEST_NAMES = {
    '62': "Brevipoalpus lewisi McGregor",
    '61': "Colomerus vitis",
    '56': "alfalfa seed chalcid",
    '73': "Erythroneura apicalis",
    '80': "Chrysomphalus aonidum",
    '75': "Panonchus citri McGregor",
    '65': "Pseudococcus comstocki Kuwana",
    '43': "beet weevil",
    '72': "Trialeurodes vaporariorum",
    '98': "Chlumetia transversa",
    '79': "Ceroplastes rubens",
    '15': "grub",
    '81': "Parlatoria zizyphus Lucus",
    '63': "oides decempunctata",
    '25': "aphids",
    '35': "wheat sawfly",
    '96': "Salurnis marginella Guerr",
    '31': "bird cherry-oataphid",
    '74': "Papilio xuthus",
    '82': "Nipaecoccus vastalor",
    '53': "therioaphis maculata Buckton",
    '78': "Unaspis yanonensis",
    '94': "Dasineura sp",
    '30': "green bug",
    '67': "Ampelophaga",
    '85': "Dacus dorsalis(Hendel)",
    '36': "cerodonta denticornis",
    '58': "Apolygus lucorum",
    '48': "tarnished plant bug",
    '14': "rice shell pest"
}

# Load TFLite Model 
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded.")

# Preprocessing Function 
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(input_details[0]['dtype'])

#Webcam Prediction Loop
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    input_tensor = preprocess(frame)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = np.argmax(prediction)
    folder_num = SELECTED_CLASSES[idx]
    pest_name = PEST_NAMES.get(folder_num, "Unknown Pest")
    confidence = prediction[idx]

    # Overlay prediction
    label = f"{pest_name} ({confidence * 100:.2f}%)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("Live Pest Detection (TFLite)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
