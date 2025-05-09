import cv2
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
from PIL import Image
import keras
from pyfirmata import Arduino
from time import sleep
import os

board = Arduino('COM3')

greenLed = board.get_pin('d:12:o')
redLed = board.get_pin('d:11:o')
imgsToAnalyse = []
fruits = ['orange', 'mango', 'banana', 'grape', 'apple']
yolo = YOLO('yolov8s.pt')

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def runModel(inputImg):
    modelPath = "./rottenPredictionModel.tflite"
    interpreter = tf.lite.Interpreter(model_path=modelPath)

    img_width = 180
    img_height = 180
    class_names = ['rotten', 'fresh']

    img = keras.utils.load_img(inputImg, target_size=(img_height, img_width))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)


    classify_lite = interpreter.get_signature_runner('serving_default')

    predictions_lite = classify_lite(keras_tensor=img_array)['output_0']
    score_lite = tf.nn.softmax(predictions_lite)

    return class_names[np.argmax(score_lite)]


camera = cv2.VideoCapture(0)
while camera.isOpened():
    ret, img = camera.read()
    results = yolo.track(img, stream=True)
    for result in results:
        # get the classes names
        classes_names = result.names
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                if class_name in fruits:
                    colour = getColours(cls)
                    # draw the rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
                    cropped_img = img[y1:y2, x1:x2] # crop frame to fruit
                    imgsToAnalyse.append(cropped_img) # add cropped frame to model array
                    cv2.putText(img, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

count = 1
results = []
for img in imgsToAnalyse: # iterate over captured imgs
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(imageRGB)
    img.save(f"./imgs/fruit{count}.png")
    print(runModel(f"./imgs/fruit{count}.png")) # run model on img
    os.remove(f"./imgs/fruit{count}.png")
    count += 1

freshNum = results.count("fresh")
rottenNum = results.count("rotten")
if freshNum > rottenNum:
    greenLed.write(1)
    sleep(5)
    greenLed.write(0)
else:
    redLed.write(1)
    sleep(5)
    redLed.write(0)


