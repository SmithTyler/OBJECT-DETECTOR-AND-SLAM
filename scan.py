import cv2
import numpy as np
import pyrealsense2 as rs 
import sys
import pandas as pd

############################################################
#This runs the yolov5 model for detections in an image frame
############################################################

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
    
    #Creating a blob from img to pass into net
    blob = cv2.dnn.blobFromImage(crop_img,1/255.0,(640, 640), swapRB=True)
    
    #Passing the image through net
    net.setInput(blob)
    predictions = net.forward()
    
    #Extracting the resulting matrix
    output_data = predictions[0]
    
    return crop_img, output_data

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
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
                
                #Remaking the detections by only keeping true indexes
                result_class_ids = []
                result_confidences = []
                result_boxes = []
                
                for i in indexes:
                    result_class_ids.append(class_ids[i])
                    result_confidences.append(confidences[i])
                    result_boxes.append(boxes[i])
                
    return result_class_ids, result_confidences, result_boxes
      
#################################################################
#Returns the coordinates of the bounding box over the depth image
#################################################################

def detect_depth(box, color, depth):

    height, width = color.shape[:2]
    expected = 640
    scale = height / expected
    aspect = width / height
    crop_start = round(expected * (aspect - 1) / 2)
    #Getting the borders of the box
    xmin, ymin, w, h = box[0]
    xmax = xmin + w
    ymax = ymin + h
    
    #Shifting box coordinates to depth scale
    xmin_depth = xmin
    ymin_depth = round(ymin / aspect)
    xmax_depth = xmax
    ymax_depth = round(ymax / aspect)
    depth_box = [xmin_depth,ymin_depth,xmax_depth,ymax_depth]
    #Overlaying the rectangle
    cv2.rectangle(depth, (xmin_depth, ymin_depth), 
                 (xmax_depth, ymax_depth), (255, 255, 255), 2)
    return depth_box

############################################################
#This runs the yolov5 model for detections in an image frame
############################################################

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
    
    #Creating a blob from img to pass into net
    blob = cv2.dnn.blobFromImage(crop_img,1/255.0,(640, 640), swapRB=True)
    
    #Passing the image through net
    net.setInput(blob)
    predictions = net.forward()
    
    #Extracting the resulting matrix
    output_data = predictions[0]
    
    return crop_img, output_data

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
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
                
                #Remaking the detections by only keeping true indexes
                result_class_ids = []
                result_confidences = []
                result_boxes = []
                
                for i in indexes:
                    result_class_ids.append(class_ids[i])
                    result_confidences.append(confidences[i])
                    result_boxes.append(boxes[i])
                
    return result_class_ids, result_confidences, result_boxes

#######################################
#Prints the bounding box over RGB image
#######################################

def print_box(image,result_class_ids, result_confidences, result_boxes):
    for i in range(len(ids)):
        for i in range(len(result_class_ids)):

            box = result_boxes[i]
            class_id = result_class_ids[i]

            cv2.rectangle(image, box, (0, 255, 255), 1)
            #cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
            #cv2.putText(image, classNames[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

#########################################
#Prints the bounding box over Depth image
#########################################

def detect_depth(box, color, depth):

    height, width = color.shape[:2]
    expected = 640
    scale = height / expected
    aspect = width / height
    crop_start = round(expected * (aspect - 1) / 2)
    #Getting the borders of the box
    xmin, ymin, w, h = box[0]
    xmax = xmin + w
    ymax = ymin + h
    
    #Shifting box coordinates to depth scale
    xmin_depth = xmin
    ymin_depth = round(ymin / aspect)
    xmax_depth = xmax
    ymax_depth = round(ymax / aspect)
    depth_box = [xmin_depth,ymin_depth,xmax_depth,ymax_depth]
    #Overlaying the rectangle
    cv2.rectangle(depth, (xmin_depth, ymin_depth), 
                 (xmax_depth, ymax_depth), (255, 255, 255), 2)
                 

    x_center = (xmax_depth + xmin_depth) / 2
    y_center = (ymax_depth + ymin_depth) / 2
    object_center = [x_center, y_center]
    
    return depth_box, object_center

###################################
#Calculating the distance to object
###################################

def distance_to_object(depth_box, aligned_depth, profile):
    
    #extracting the box coordinates for depth frame
    xmin_depth,ymin_depth,xmax_depth,ymax_depth = depth_box
    
    #Getting depth data
    depth = np.asanyarray(aligned_depth.get_data())
    depth = depth[xmin_depth:xmax_depth,ymin_depth:ymax_depth].astype(float)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth = depth * depth_scale
    dist,_,_,_ = cv2.mean(depth)
    return dist



if __name__ == "__main__":
    #######
    #inputs
    #######
    robot_x = sys.argv[1]
    robot_y = sys.argv[2]
    robot_yaw = sys.argv[3]
    robot_pitch = sys.argv[4] 
    video_name = sys.argv[5]
    #######
    
    classNames =("Apple Scab Leaf",
                 "Apple leaf",
                 "Apple rust leaf",
                 "Bell_pepper leaf spot",
                 "Bell_pepper leaf",
                 "Blueberry leaf",
                 "Cherry leaf",
                 "Corn Gray leaf spot",
                 "Corn leaf blight",
                 "Corn rust leaf",
                 "Peach leaf",
                 "Potato leaf early blight",
                 "Potato leaf late blight",
                 "Potato leaf",
                 "Raspberry leaf",
                 "Soyabean leaf",
                 "Soybean leaf"\
                 
                 "Squash Powdery mildew leaf",
                 "Strawberry leaf",
                 "Tomato Early blight leaf",
                 "Tomato Septoria leaf spot",
                 "Tomato leaf bacterial spot",
                 "Tomato leaf late blight",
                 "Tomato leaf mosaic virus",
                 "Tomato leaf yellow virus",
                 "Tomato leaf",
                 "Tomato mold leaf",
                 "Tomato two spotted spider mites leaf",
                 "grape leaf black rot",
                 "grape leaf")
                 

    ######################################################
    # REALSENSE FRAMES 
    ######################################################

    #################
    #RECORDING FRAMES
    #################    
       
    # Setup:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(video_name)
    profile = pipe.start(cfg)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
      pipe.wait_for_frames()
      
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    pipe.stop()
    print("Frames Captured")

    #########
    #RGB DATA
    #########

    color = np.asanyarray(color_frame.get_data())

    ###########
    #DEPTH DATA
    ###########

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    #################
    #ALLIGNING FRAMES
    #################

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())


    ####################################################
    ####################################################

    ###########
    #TESTING
    ###########

    #Calling deteciton with RGB image
    crop_img, out, = wrapped_detection(color)
    ids,conf,box = unwrap_detection(crop_img, out)
    print_box(crop_img, ids, conf, box)

    ###########################################################################
    #COMPUTING THE DISTANCES WHEN THERE IS MORE THAN ONE DETECTION IN THE FRAME
    ###########################################################################
    
    if len(ids) > 1:
        
        for i in range(0,len(ids)):
        
            #############################################
            # COMPUTING THE XYX RESPECT FROM CAMERA LENSE 
            #############################################
            
            depth_box, center = detect_depth(box[i],color,colorized_depth)
            x = round(center[0])
            y = round(center[1])
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            dist = depth_frame.get_distance(x,y)
            
            result = rs.rs2_deproject_pixel_to_point(depth_intrin, center, dist)

            right = result[0]
            down = result[1]
            deep = result[2]
         
            #########################################
            # COMPUTING TRUE XYZ WITHIN WIREBOT SPACE
            #########################################
            
            #x = robot_x - right
            #y = robot_y - down
            #z = 1.2 - deep
            x = right
            y = down
            z = deep
            ####################
            # WRITING TO THE CSV
            ####################
            plant_class = ids[i]
            confidence = conf[i]
            data = {'CLASS':[plant_class],
                    'CONFIDENCE':[confidence],
                    'X':[x],
                    'Y':[y],
                    'Z':[z]}
                    
            DF = pd.DataFrame(data)
            DF.to_csv('detected_locations.csv', mode = 'a', index = False, header = False) 
    
    ############################################################
    #COMPUTING THE DISTANCE WHEN THERE IS ONE DETECTION IN FRAME
    ############################################################
    
    else:
    
        #############################################
        # COMPUTING THE XYX RESPECT FROM CAMERA LENSE 
        #############################################
        
        depth_box, center = detect_depth(box, color, colorized_depth)
        x = round(center[0])
        y = round(center[1])
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        dist = depth_frame.get_distance(x,y)
        
        result = rs.rs2_deproject_pixel_to_point(depth_intrin, center, dist)
     
        right = result[0]
        down = result[1]
        deep = result[2]
        
        #########################################
        # COMPUTING TRUE XYZ WITHIN WIREBOT SPACE
        #########################################
        

        
        x = right
        y = down
        z = deep
        ######################
        # WRITING TO THE CSV
        ######################
        plant_class = ids[0]
        confidence = conf[0]
        data = {'CLASS':[plant_class],
                'CONFIDENCE':[confidence],
                'X':[x],
                'Y':[y],
                'Z':[z]}
        
        DF = pd.DataFrame(data)
        DF.to_csv('detected_locations.csv', mode = 'a', index = False, header = False)
        
        

