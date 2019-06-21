import cv2
import numpy as np
import pudb
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.models import *

from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from utils import *
from copy import copy

input_shape = (224, 224, 3)

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
folder = "models/"
classesFile = folder+"/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = folder+"/yolov3.cfg"
modelWeights = folder+"/yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def create_base_network(in_dim):
    """ Base network to be shared (eq. to feature extraction).
    """
    model = ResNet50(weights="imagenet")
    print(model.summary())
    return Model(inputs=model.input, outputs=model.get_layer('fc1000').output)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    # label = '%.2f' % conf
        
    # # Get the label for the class name and its confidence
    # if classes:
    #     assert(classId < len(classes))
    #     label = '%s:%s' % (classes[classId], label)

    # #Display the label at the top of the bounding box
    # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # top = max(top, labelSize[1])
    # cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    # cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    frame_to_return = copy(frame)

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # print (confidence)
            # pu.db
            if confidence >= 0.0:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    boxes_to_return = []
    # pu.db
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if classIds[i] == 2:
            boxes_to_return.append(box)
            drawPred(frame_to_return, classIds[i], confidences[i], left, top, left + width, top + height)
    
    # pu.db
    return boxes_to_return, frame_to_return


cap1 = cv2.VideoCapture('../Videos/ferst_atlantic_day.avi')
cap2 = cv2.VideoCapture('../Videos/ferst_state_day.avi')

# my_training_batch_generator = My_Generator(files_perm, GT_training, batch_size)
# print("Would you like to load an existing model? (y/n)")
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
# if choice != 'n' and choice != 'N':
model.load_weights("models/check_resnet_distance_weights.h5")
# model_gpu = multi_gpu_model(model, gpus=4)


if (cap1.isOpened()== False or cap2.isOpened()== False): 
  print("Error opening video stream or file")

first = 1
saved_box = []
saved_frame = []

while(cap1.isOpened() and cap2.isOpened()):
  # Capture frame-by-frame
  ret1, frame1 = cap1.read()
  ret2, frame2 = cap2.read()
  if ret1 == True and ret2 == True:
 
    blob1 = cv2.dnn.blobFromImage(frame1, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    blob2 = cv2.dnn.blobFromImage(frame2, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    net.setInput(blob1)
    outs = net.forward(getOutputsNames(net))
    box_here, frame_here = postprocess(frame1, outs)

    if first == 7:
        saved_box = box_here[0]
        saved_frame = frame1[saved_box[1]:saved_box[1]+saved_box[3], saved_box[0]:saved_box[0]+saved_box[2], :]
    # else:

    net.setInput(blob2)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame2, outs)

    # Display the resulting frame
    cv2.imshow('Frame1', frame_here)
    # pu.db
    if first != 7:
        cv2.imshow('car in frame 1', frame1[box_here[0][1]:box_here[0][1]+box_here[0][3], box_here[0][0]:box_here[0][0]+box_here[0][2], :])
    else:
        # pu.db
        cv2.imshow('car in frame 1', saved_frame)
    cv2.imshow('Frame2', frame2)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
    first += 1
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap1.release()
cap2.release()
