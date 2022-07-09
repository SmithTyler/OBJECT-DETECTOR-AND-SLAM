import cv2
import numpy as np
import pyrealsense2 as rs 
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def wrapped_detection(in_image):
    
    #Loading the model 
    net = cv2.dnn.readNet('7_class_5k.onnx')
    
    #Using cuda cores
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    
    #FORMATTING THE IMAGE TO YOLOv5's 640X640 input
    row, col = in_image.shape[:2]
    expected = 640
    aspect = col / row
    resized_image  = cv2.resize(in_image, (round(expected * aspect), expected))
    crop_start = round(expected * (aspect - 1) / 2)
    crop_img = resized_image[0:expected, crop_start:crop_start+expected]
    
    #NEW RESIZE
    dim = (640, 640)
    resize = cv2.resize(in_image, dim)
    
    #Creating a blob from img to pass into net
    blob = cv2.dnn.blobFromImage(crop_img,1/255.0,(640, 640), swapRB=True)
    
    #Passing the image through net
    net.setInput(blob)
    predictions = net.forward()
    
    #Extracting the resulting matrix
    output_data = predictions[0]
    
    return resize, crop_img, output_data
    
###############################################################
#This function unwraps the detection: gets boxes, conf, classes 
###############################################################

def unwrap_detection(input_image, output_data):
    
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
                
                
            
                #Removing the overlapping/duplicated detections
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.15, 0.25)
                
                #Remaking the detections by only keeping true indexes
                result_class_ids = []
                result_confidences = []
                result_boxes = []
                
                for i in indexes:
                    result_class_ids.append(class_ids[i])
                    result_confidences.append(confidences[i])
                    result_boxes.append(boxes[i])
    if len(class_ids) == 0:
        result_class_ids = 0
        result_confidences = 0
        result_boxes = 0
        return result_class_ids, result_confidences, result_boxes
    else: 
        return result_class_ids, result_confidences, result_boxes

####################
# PRINTING THE BOXES
####################

def print_box(image,result_class_ids, result_confidences, result_boxes):
    classNames = ("APPLE",
                  "BELL PEPPER",
                  "CORN",
                  "PEACH",
                  "POTATO",
                  "RASBERRY",
                  "TOMATO")
    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]
        CONF = result_confidences
        
        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, classNames[class_id]+' '+str(round(CONF[i],3)), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
    
    #plt.figure(figsize=(30,30))
    #plt.imshow(image)
    #plt.show()
    return image
    
def detection_main(filename):
    img = cv2.imread(filename)

    #Calling Wrapped Detection
    resize,crop_img, output_data = wrapped_detection(img)
    #print(output_data)
    #Calling Unwrapped Detection
    IDS, CONF, BOXES = unwrap_detection(img, output_data)
    #print(IDS)
    if (IDS !=0)&(CONF !=0)&(BOXES!=0):
        print('DETECTION !!')
        image = print_box(img, IDS, CONF, BOXES)
        os.remove('DETECTION.jpg')
        cv2.imwrite('DETECTION.jpg',image)
        return IDS, CONF
    else:
        print('NO DETECTION')
        
        return 0,0
        
        
IDS, CONF =detection_main('BELL_PEPPER.jpg')
print(IDS)
print(CONF)