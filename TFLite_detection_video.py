######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import time

#set the dependencies to the buzzer
#Libraries
import RPi.GPIO as GPIO
from time import sleep
#Disable warnings (optional)
GPIO.setwarnings(False)
#Select GPIO mode
GPIO.setmode(GPIO.BCM)
#Set buzzer - pin 23 as output
buzzer=23
GPIO.setup(buzzer,GPIO.OUT)


MODEL_NAME = "/home/pi/tflite1/trained_models"
GRAPH_NAME = "model_mobinetv2_optimsize.tflite"
LABELMAP_NAME = "/home/pi/tflite1/trained_models/labels.txt"
VIDEO_NAME = "/home/pi/tflite1/video/test.mp4"
min_conf_threshold = float(0.5)
use_TPU = False

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = [0.485, 0.456, 0.406] 
input_std = [0.229, 0.224, 0.225]

def class_map(prediction):
    if(prediction==0):
        return 'safe driving'
    if(prediction==1):
        return 'texting - right'
    if(prediction==2):
        return 'talking on the phone - right'
    if(prediction==3):
        return 'texting - left'
    if(prediction==4):
        return 'talking on the phone - left'
    if(prediction==5):
        return 'operating the radio'
    if(prediction==6):
        return 'drinking'
    if(prediction==7):
        return 'reaching behind'
    if(prediction==8):
        return 'hair and makeup'
    if(prediction==9):
        return 'talking to passenger'
    
# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

while(video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = input_data / 255.0
        input_data = (input_data - input_mean) / input_std
        input_data = np.float32(input_data)
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    tf_results = interpreter.get_tensor(output_details[0]['index'])    
    max_class = np.argmax(tf_results)
    class_name = class_map(max_class)

    # Draw framerate in corner of frame
    cv2.putText(frame,'Predicted Class: {}'.format(str(class_name)),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    if(class_name !='safe driving'):
        cv2.imshow('Object detector', frame)
        GPIO.output(buzzer,GPIO.HIGH)
        print ("Detected Driver Distraction")
        sleep(2.0) # Delay in seconds
        GPIO.output(buzzer,GPIO.LOW)
        print ("Turn off the Buzzer")

    #All the results have been drawn on the frame, so it's time to display it.
    sleep(3.0)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

