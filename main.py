import os
import sys
import time
import argparse
from collections import deque
from pathlib import Path

import cv2
import easyocr
import pygame
import torch
import numpy as np
from ultralytics import YOLO

CONF_THRES = 0.5        # Confidence threshold for displaying bounding boxes
NUM_CLASSES = 13        # Total number of classes 
COOLDOWN = 8            # Cooldown period between same class being announced

CONS_DETECT_INTERVAL = 0.5      # Max time interval for consecutive detections
CONS_DETECT_COUNT = 3           # Number of consecutive frames to consider a valid prediction.

PRIORITY_CLASSES = {'green', 'red', 'yellow', 'left green', 'stop', 'yield'}
LIGHTS = {'green', 'red', 'yellow', 'left green'}

# Audio Files
GREEN = "audio_files/green_light.mp3"
RED = "audio_files/red_light.mp3"
YELLOW = "audio_files/yellow_light.mp3"
LEFT_GREEN = "audio_files/left_turn_green.mp3"
STOP = "audio_files/stop.mp3"
CONSTRUCTION = "audio_files/construction.mp3"
YIELD = "audio_files/yield.mp3"
FLASH_40 = "audio_files/flashing_40.mp3"
SPEED_40 = "audio_files/speed_40.mp3"
SPEED_50 = "audio_files/speed_50.mp3"
SPEED_60 = "audio_files/speed_60.mp3"
SPEED_70 = "audio_files/speed_70.mp3"
SPEED_80 = "audio_files/speed_80.mp3"

AUDIO_FILES = {
    'green': GREEN,
    'red': RED,
    'yellow': YELLOW, 
    'left green': LEFT_GREEN, 
    'stop': STOP,
    'construction': CONSTRUCTION, 
    'yield': YIELD,
    'flashing 40': FLASH_40, 
    'speed 40': SPEED_40, 
    'speed 50': SPEED_50, 
    'speed 60': SPEED_60, 
    'speed 70': SPEED_70, 
    'speed 80': SPEED_80, 
}

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)

# Function to play audio file
def play_music(mp3_file):
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()

# Initialize the easyocr reader
reader = easyocr.Reader(['en'])

# Initialize the pygame mixer
pygame.mixer.init()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Path to YOLO model file", required=True)
parser.add_argument('--source', help='Image source, can be image file, \
                    image folder, video file, index of USB camera ("usb0")', 
                    required=True)
args = parser.parse_args()

model_path = args.model
source = args.source

# Load model
model = YOLO(model_path, task='detect')
model.to(device)
labels = model.names

img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(source):
    source_type = 'folder'
elif os.path.isfile(source):
    root, ext = os.path.splitext(source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
elif 'usb' in source:
    source_type = 'usb'
    usb_idx = int(source[3:])
else:
    print(f'Input {source} is invalid. Please try again.')
    sys.exit(0)

if source_type == 'image':
    images = [source]
elif source_type == 'folder':
    images = []
    for file in os.listdir(source):
        root, ext = os.path.splitext(file)
        if ext in img_ext_list:
            images.append(Path(source) / file)
elif source_type == 'video':
    stream = cv2.VideoCapture(source)
else:
    stream = cv2.VideoCapture(usb_idx)

colours = [
    (0, 0, 255),       # Red
    (0, 255, 0),       # Green
    (255, 0, 0),       # Blue
    (0, 255, 255),     # Yellow (Cyan in BGR)
    (255, 0, 255),     # Magenta
    (255, 255, 0),     # Aqua (Yellow in BGR)
    (128, 0, 128),     # Purple
    (0, 128, 128),     # Teal
    (128, 128, 0),     # Olive
    (0, 165, 255),     # Orange
    (203, 192, 255),   # Pink
    (42, 42, 165),     # Dark Red
    (180, 105, 255),   # Light Pink / Orchid
]

average_fps = 0
frame_rate_buffer = deque()
buffer_len = 200
img_idx = 0

curr_speed = ""     # variable to store current speed

audio_queue = deque()   # A queue for storing the classes to be announced.
audio_playing = None    # A string storing the class currently being announced if any. 
last_announced = {}     # A dictionary to store when each class was last announced.
detection_counts = {}   # A dictionary to hold a count of consecutive frames a class has been predicted.

# Inference Loop
while True:
    
    # Timer for calculating frame rate
    t_start = time.perf_counter()

    # Get input image/frame
    if source_type == 'image' or source_type == 'folder':
        if img_idx == len(images):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        image = images[img_idx]
        print(image)
        frame = cv2.imread(image)
        img_idx += 1
    
    elif source_type == 'video':
        success, frame = stream.read()
        if not success:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb':
        success, frame = stream.read()
        if (frame is None) or (not success):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Get model predictions
    results = model(frame, verbose=False)
    detections = results[0].boxes

    for detection in detections:

        # Extract bounding box coordinates
        xyxy_tensor = detection.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detection.cls.item())
        classname = labels[classidx]

        # Ignore lights near the left or right edges of the frame, as they may
        # belong to cross traffic.
        if classname in LIGHTS:
            x = frame.shape[1]
            if xmax <= 0.1 * x or xmin >= 0.9 * x:
                continue

        conf = detection.conf.item()

        if conf >= CONF_THRES:

            # Draw bounding box
            colour = colours[classidx]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colour, thickness=2)

            # Draw box label
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)   # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), colour, cv2.FILLED)   # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)    # Draw label text


            # Add the class to the audio queue given that it is not already in the queue and 
            # was not recently announced and has been seen for CONS_DETECT_COUNT consecutive frames.
            # Two frames are considered consecutive if they are within CONS_DETECT_INTERVAL of each other.
            if (classname not in audio_queue and classname != audio_playing):
                if (classname not in last_announced) or (time.time() - last_announced[classname] > COOLDOWN):

                    if (classname not in detection_counts) or (time.time() - detection_counts[classname][1] > CONS_DETECT_INTERVAL) :
                        detection_counts[classname] = [1, time.time()]

                    elif detection_counts[classname][0] == CONS_DETECT_COUNT - 1:
                        detection_counts.pop(classname)

                        if (classname in PRIORITY_CLASSES):
                            audio_queue.appendleft(classname)

                        elif classname == "speed": 
                            # Get speed limit using OCR
                            ocr_results = reader.readtext(frame[ymin:ymax+1, xmin:xmax+1])
                            if not ocr_results:
                                continue
                            curr_speed = ""
                            ocr_text = "".join([text for _, text, _ in ocr_results])
                            if "40" in ocr_text:
                                curr_speed = "speed 40"
                            elif "50" in ocr_text:
                                curr_speed = "speed 50"
                            elif "60" in ocr_text:
                                curr_speed = "speed 60"
                            elif "70" in ocr_text:
                                curr_speed = "speed 70"
                            elif "80" in ocr_text:
                                curr_speed = "speed 80"
                            else:   # False positive
                                continue

                            audio_queue.append(classname)
                                
                        else:
                            audio_queue.append(classname)
                    else:
                        detection_counts[classname][0] += 1
                        detection_counts[classname][1] = time.time()

    

    # Handle audio
    if not pygame.mixer.music.get_busy():       # if not audio is playing
        audio_playing = None
    if audio_queue and not audio_playing:
        class_name = audio_queue.popleft()
        if class_name == 'speed':
            play_music(AUDIO_FILES[curr_speed])
        else:
            play_music(AUDIO_FILES[class_name])
        audio_playing = class_name
        last_announced[class_name] = time.time()

    # Draw framerate (if using video or USB)
    if source_type == 'video' or source_type == 'usb':
        cv2.putText(frame, f'FPS: {average_fps:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    # Display the image
    cv2.imshow('YOLO detection results',frame)

    # If inferencing on individual images, wait for user keypress before moving 
    # to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer to calculate average FPS
    if len(frame_rate_buffer) == buffer_len:
        frame_rate_buffer.popleft()
    frame_rate_buffer.append(frame_rate)

    # Calculate average FPS
    average_fps = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {average_fps:.2f}')
if source_type == 'video' or source_type == 'usb':
    stream.release()
cv2.destroyAllWindows()