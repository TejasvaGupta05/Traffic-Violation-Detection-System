# coding: utf-8

# # Object Detection Demo
# Traffic Violation Detection using SSD MobileNet + TensorFlow

# # Imports
import numpy as np
import os
import sys
import tarfile

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import cv2
from collections import defaultdict
import datetime
import pickle
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import label_map_util, dataFileGlobal
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


def download_model():
    """Downloads the model tar.gz if not already present."""
    if not os.path.exists(MODEL_FILE):
        print(f"[INFO] Downloading model: {MODEL_FILE} ...")
        url = DOWNLOAD_BASE + MODEL_FILE
        urllib.request.urlretrieve(url, MODEL_FILE)
        print("[INFO] Download complete.")
    else:
        print(f"[INFO] Model file already exists: {MODEL_FILE}")


def extract_model():
    """Extracts frozen_inference_graph.pb from the tar if not already extracted."""
    if not os.path.exists(PATH_TO_CKPT):
        print("[INFO] Extracting model...")
        tar_file = tarfile.open(MODEL_FILE)
        for member in tar_file.getmembers():
            if 'frozen_inference_graph.pb' in os.path.basename(member.name):
                tar_file.extract(member, os.getcwd())
        tar_file.close()
        print("[INFO] Extraction complete.")
    else:
        print(f"[INFO] Frozen graph already exists: {PATH_TO_CKPT}")


def load_graph():
    """Loads the frozen TensorFlow graph."""
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


if __name__ == '__main__':

    # --- Step 1: Download & extract model ---
    download_model()
    extract_model()

    # --- Step 2: Load TF graph ---
    print("[INFO] Loading detection graph...")
    detection_graph = load_graph()
    print("[INFO] Graph loaded.")

    # --- Step 3: Load label map ---
    label_map    = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories   = label_map_util.convert_label_map_to_categories(
                       label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # --- Step 4: Open video ---
    cam = cv2.VideoCapture('test1.mp4')
    if not cam.isOpened():
        print("[ERROR] Could not open test1.mp4 — check that the file exists in this folder.")
        sys.exit(1)

    ok, frame = cam.read()
    if not ok or frame is None:
        print("[ERROR] Could not read first frame from video.")
        cam.release()
        sys.exit(1)

    frame = cv2.resize(frame, (1280, 720))

    # --- Step 5: Select ROI (the lane/zone to monitor) ---
    print("[INFO] Draw the violation zone on the video frame, then press ENTER or SPACE to confirm.")
    bbox_tmp = cv2.selectROI('ROI', frame, False)
    print(bbox_tmp, 'ROI selected')
    cv2.destroyWindow('ROI')

    # --- Step 6: Init global state ---
    dataFileGlobal.init()
    dataFileGlobal.myList.append(bbox_tmp)

    # --- Step 7: Run detection loop ---
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor      = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes   = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores  = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections    = detection_graph.get_tensor_by_name('num_detections:0')

            print("[INFO] Detection running. Press 'q' to quit.")
            while True:
                ret, image_np = cam.read()
                if not ret or image_np is None:
                    print("[INFO] End of video.")
                    break

                image_np_expanded = np.expand_dims(image_np, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Draw the ROI zone in green
                cv2.rectangle(
                    image_np,
                    (bbox_tmp[0], bbox_tmp[1]),
                    (bbox_tmp[0] + bbox_tmp[2], bbox_tmp[1] + bbox_tmp[3]),
                    (0, 255, 0), 2)

                cv2.imshow('Traffic Violation Detection', image_np)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print(f"[INFO] Total violations tracked: {dataFileGlobal.numTrack}")
                    break

    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")