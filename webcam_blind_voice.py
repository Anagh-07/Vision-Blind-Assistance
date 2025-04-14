import os
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from object_detection.utils import label_map_util, visualization_utils as vis_util
import pyttsx3
import pytesseract
import urllib.request
import tarfile

# Disable TensorFlow v2 behavior for compatibility
tf.disable_v2_behavior()

# Download and extract TensorFlow model 
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(os.getcwd(), 'models', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

if not os.path.exists(PATH_TO_CKPT):
    print("Downloading model...")
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    with tarfile.open(MODEL_FILE) as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar.extractall(path, members, numeric_owner=numeric_owner)
        
        safe_extract(tar, ".")
    os.remove(MODEL_FILE)
    print("Model downloaded and extracted.")

# Load frozen model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load Places365 model for scene recognition
scene_model = models.resnet18(num_classes=365)
checkpoint = torch.load("whole_resnet18_places365.pth.tar", map_location=torch.device('cpu'))

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    scene_model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    scene_model.load_state_dict(checkpoint.state_dict(), strict=False)

scene_model.eval()

centre_crop = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Text-to-Speech initialization
engine = pyttsx3.init()

def recognize_scene(image):
    image = centre_crop(image).unsqueeze(0)
    with torch.no_grad():
        logits = scene_model(image)
    _, predicted_class = logits.max(1)
    return predicted_class.item()

def detect_objects(image_np, sess, detection_graph):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded}
    )

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8
    )
    return classes, scores, boxes

def read_text(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Start video capture
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while cap.isOpened():
            ret, image_np = cap.read()
            if not ret:
                print("Error reading from webcam")
                break

            classes, scores, boxes = detect_objects(image_np, sess, detection_graph)
            detected_objects = []
            detected_text = "Objects detected: "
            danger_zone = False

            for cls, score, box in zip(np.squeeze(classes), np.squeeze(scores), np.squeeze(boxes)):
                if score > 0.5:
                    obj_name = category_index.get(int(cls), {'name': 'Unknown'})['name']
                    ymin, xmin, ymax, xmax = box  # Bounding box coordinates (normalized)
                    height = ymax - ymin  # Calculate object size (height)

                    # Approx 4 steps -caution
                    #Apprx- above 5 steps safe distance
                    #approx below 3-4 step dnager


                    # Determine distance based on bounding box height
                    if height > 0.9:
                        distance_status = "very close! DANGER!"
                        danger_zone = True
                    elif height > 0.7:
                        distance_status = "nearby, be cautious"
                    else:
                        distance_status = "at a safe distance"

                    detected_text += f"{obj_name} is {distance_status}. "

            print(detected_text)
            engine.say(detected_text)
            engine.runAndWait()

            cv2.imshow('Blind Assistance System', image_np)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
#
cap.release()
cv2.destroyAllWindows()