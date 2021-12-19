# -*- coding: utf-8 -*-
import base64
import numpy as np
import socketio
import eventlet
# import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO
# ------------- Add library ------------#
# import matplotlib as plt
# import math
import time
from data_processing.Augmentation import *
from data_processing.helper import *
from data_processing.trained_model import *
import tensorflow as tf
import os
import glob
import json
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# --------------------------------------#
error_arr = np.zeros(5)
annot_dir = '../Unity_dataset/Dataset/annotation_definitions.json'
f = open(annot_dir)
data = json.load(f)
new_dict = {}
df = pd.DataFrame(columns=['label','r','g','b'])
for i in data[list(data.keys())[1]][1]['spec']:
    new_dict['label'] = i['label_name']
    new_dict['r'] = [int(i['pixel_value']['r']*255)]
    new_dict['g'] = [int(i['pixel_value']['g']*255)]
    new_dict['b'] = [int(i['pixel_value']['b']*255)]
    df = pd.concat([df,pd.DataFrame(new_dict)])
df = df.reset_index()[df.reset_index().columns[1:]]

labels = df['label'].to_list()
index_label = [0, 7, 11, 20, 22, 26]
classes = df.label[index_label].to_list()

preprocess_input, model= get_trained_model('efficientnetb3', 'FPN', classes, labels)
model.load_weights('./checkpoint/Unity_efficientnetb3_FPN.h5')

dt = time.time()
# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = 0  # Góc lái hiện tại của xe
        speed_callback = 0  # Vận tốc hiện tại của xe
        image = 0  # Ảnh gốc
        # Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        cv2.imshow('',image)
        #cv2.imshow('Before RGB2BGR',image)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = cv2.resize(image,(320,320))
        cv2.imshow("RGB_view",img)
        img = preprocess_input(img)
        
        mask  = model.predict(np.array([img]))
        #mask = cv2.resize(mask,(320,180))
        #image = cv2.resize(image,(320,180))

        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe

        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 50
        # ------------------------------------------  Work space  ----------------------------------------------#
        cv2.imshow("Car_view", process_mask(mask[..., 5].squeeze(),0.999))
        cv2.imshow("Road_view", process_mask(mask[..., 1].squeeze(),0.6))
        cv2.imshow("Trafficsign_view", mask[..., 3].squeeze())
        cv2.waitKey(1)


        print('toc do nap len {} : {}'.format(sendBack_angle, sendBack_Speed))
        print('van toc tra ve {} : {}'.format(steering_angle, speed_callback))
        send_control(sendBack_angle, sendBack_Speed)
        print(sendBack_Speed)

    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    print('on connect')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)

# def drawRec_withObject(name_list,mask,img,colorRGB_list,threshold=0.9):
#     #CLASSES = ['car', 'pedestrian','road','roadmark','trafficsign']
#     CLASSES = ['car','pedestrian','road','roadcone','roadline','trafficsign']
#     tmp_img = img.copy()
#     mask = cv2.resize(mask[0],(320,180))
#     temp = np.zeros((tmp_img.shape[0],tmp_img.shape[1],tmp_img.shape[2]),dtype='uint8')
#     for j in range(len(name_list)):
#         for i in range(len(CLASSES)):
#             if CLASSES[i] == name_list[j]:
#                 mask_cvt = mask[..., i].squeeze().copy()
#                 temp[:,:,0] = np.array(mask_cvt>=0.9,dtype='uint8')*255
#                 temp[:,:,1] = np.array(mask_cvt>=0.9,dtype='uint8')*255
#                 temp[:,:,2] = np.array(mask_cvt>=0.9,dtype='uint8')*255
#                 canny= cv2.Canny(cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY),200,255)
#                 countours, hierachy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#                 if len(countours) > 0:
#                     x_max,x_min = countours[0][:,0,0].max(),countours[0][:,0,0].min()
#                     y_max,y_min = countours[0][:,0,1].max(),countours[0][:,0,1].min()
#                     if x_max - x_min > 5:
#                         cv2.rectangle(tmp_img,(x_min-5,y_min-5),(x_max+5,y_max+5),colorRGB_list[j],2)
#                         cv2.putText(tmp_img,name_list[j],(x_min,y_min-10),cv2.FONT_HERSHEY_COMPLEX,1/2,colorRGB_list[j],1)
#     return tmp_img

def process_mask(mask,threshold=0.9):
    processed_mask = np.zeros((180,320))
    mask = cv2.resize(mask,(320,180))
    processed_mask[:,:] = np.array(mask>=threshold,dtype='uint8')
    return processed_mask

def drawInterested_Sign(mask,img,color,threshold=0.8):
    tmp_img = img.copy()
    mask = cv2.resize(mask,(320,180))
    temp = np.zeros((tmp_img.shape[0],tmp_img.shape[1],tmp_img.shape[2]),dtype='uint8')
    temp[:,:,0] = np.array(mask>=threshold,dtype='uint8')*255
    temp[:,:,1] = np.array(mask>=threshold,dtype='uint8')*255
    temp[:,:,2] = np.array(mask>=threshold,dtype='uint8')*255
    canny= cv2.Canny(cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY),200,255)
    countours, hierachy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(countours) > 1:
        sorted_countour = sorted(countours,key=cv2.contourArea)
        x_max,x_min = countours[-1][:,0,0].max(),countours[-1][:,0,0].min()
        y_max,y_min = countours[-1][:,0,1].max(),countours[-1][:,0,1].min()
        if x_min >=5:
            if y_min >= 5:
                cv2.rectangle(tmp_img,(x_min-5,y_min-5),(x_max+5,y_max+5),(255,0,0),2)
            else:
                cv2.rectangle(tmp_img,(x_min-5,0),(x_max+5,y_max+5),(255,0,0),2)
        else:
            if y_min >= 5:
                cv2.rectangle(tmp_img,(0,y_min-5),(x_max+5,y_max+5),(255,0,0),2)
            else:
                cv2.rectangle(tmp_img,(0,0),(x_max+5,y_max+5),(255,0,0),2)   
        # Model predict name there

        cv2.putText(tmp_img,'Sign',(x_min,y_min),cv2.FONT_HERSHEY_COMPLEX,2/5,(0,0,255),1)
    return tmp_img

def Can(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 7)
    #blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 25, 255)
    return canny


def bird_view(image):

    width, height = 300, 320
    pts1 = np.float32([[0, 100], [300, 100], [0, 200], [300, 200]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    birdview = cv2.warpPerspective(image, matrix, (height, width))
    '''
    pts1 = np.float32([[0, 100], [300, 100], [0, 200], [300, 200]])
    pts2 = np.float32([[0, 0], [200, 0], [200 - 140, 300], [170, 300]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    birdview = cv2.warpPerspective(image, matrix, (200, 350))
    '''
    return birdview

def ROI(image):
    height = image.shape[0]
    shape = np.array([[10, height], [450, height], [450, 100], [50, 100]])
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[1]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def PID(error, p=0.45, i=0.05, d=0.02):
    global dt
    global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error * p
    delta_t = time.time() - dt
    dt = time.time()
    D = (error - error_arr[1]) / delta_t * d
    I = np.sum(error_arr) * delta_t * i
    angle = P + I + D
    if abs(angle) > 45:
        angle = np.sign(angle) * 60
    return -int(angle)


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10, 8)
        xa = int((x1 + x2) / 2 - 130)  # red 110 160
        ya = int((y1 + y2) / 2)
        cv2.circle(line_image, (xa - 20, ya), 5, (0, 255, 0), -1)
        x_d = xa - 150  # green 150
        cv2.circle(line_image, (x_d, ya), 5, (0, 0, 255), -1)
        angle_PID = PID(x_d)
        if angle_PID > 20:
            angle_PID = 15
        elif angle_PID < -20:
            angle_PID = -15
    return line_image, angle_PID


def average_slope_intercept(image, lines):
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]  # he so a
        intercept = parameters[1]  # he so b
        if slope > 0:
            right_fit.append((slope, intercept))
    # sap xep right_fit theo chieu tang dan cua intercept
    leng = len(right_fit)
    right_fit = np.array(sorted(right_fit, key=lambda a_entry: a_entry[0]))
    right_fit = right_fit[::-1]
    right_fit_focus = right_fit
    if leng > 2:
        right_fit_focus = right_fit[:1]
    right_fit_average = np.average(right_fit_focus, axis=0)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([right_line])


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


if __name__ == '__main__':
    # -----------------------------------  Setup  ------------------------------------------#

    # --------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

