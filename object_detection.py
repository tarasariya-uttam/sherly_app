import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path, cfg_path):
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608
        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.colors = np.random.uniform(0, 255, size=(80, 3))

    def load_detection_model(self, image_size=None, nmsThreshold=None, confThreshold=None):
        if image_size:
            self.image_size = image_size
        if nmsThreshold:
            self.nmsThreshold = nmsThreshold
        if confThreshold:
            self.confThreshold = confThreshold

        print("Loading Object Detection with params:")
        print("image_size=({}, {})".format(self.image_size, self.image_size))
        print("nmsThreshold={}".format(self.nmsThreshold))
        print("confThreshold={}".format(self.confThreshold))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path):
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

