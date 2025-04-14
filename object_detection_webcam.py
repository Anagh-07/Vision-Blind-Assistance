
import numpy as np
import os
import urllib.request
import tarfile
import tensorflow as tf
import cv2
import sys

sys.path.append(r"C:\Users\dell\Downloads\Blind-Assistance-Object-Detection-and-Navigation-master\models\research\object_detection")

# Import object detection utilities
from utils import label_map_util
from utils import visualization_utils as vis_util

# Define video capture
cap = cv2.VideoCapture(0) 

# Model selection
MODEL_NAME = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'  

# Paths to model and label map
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

#This file maps COCO dataset class IDs to their respective names (e.g., person, car, chair).
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

#  Download and extract the model 
if not os.path.exists(PATH_TO_CKPT):
    print("Downloading model...")
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    with tarfile.open(MODEL_FILE) as tar:
        tar.extractall()
    os.remove(MODEL_FILE)
    print("Model downloaded and extracted.")

#  Load TensorFlow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#  Load label map correctly
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories) 

# Function to convert image to numpy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#  Object Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            # Read frame from camera
            ret, image_np = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Expand dimensions for model input
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Extract model tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Perform object detection
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

            # Visualization
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8
            )

            # Show detection output
            cv2.imshow('Object Detection', cv2.resize(image_np, (800, 600)))

            # Exit on 'q' key press
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
