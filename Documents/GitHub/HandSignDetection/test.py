import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import tensorflow as tf
import math
import os

custom_objects = {"DepthwiseConv2D": DepthwiseConv2D}

try:
    with custom_object_scope(custom_objects):
        model = tf.keras.models.load_model("Model/keras_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define custom objects for model loading
custom_objects = {"DepthwiseConv2D": DepthwiseConv2D}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = HandDetector(maxHands=1)

# Load the model with custom objects
try:
    with custom_object_scope(custom_objects):
        model = tf.keras.models.load_model("Model/keras_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/A"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P","Q","R", "S", "T", "U", "W", "X", "Y"]

if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    hands, img = detector.findHands(img)
    print(f"Number of hands detected: {len(hands)}")

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Debug: Check imgWhite shape and type
        print("imgWhite shape:", imgWhite.shape)
        print("imgWhite dtype:", imgWhite.dtype)

        # Get prediction using the processed image
        try:
            prediction, index = classifier.getPrediction(imgWhite)
            print(f"Prediction: {prediction}, Index: {index}")
        except Exception as e:
            print(f"Error during prediction: {e}")

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()