import numpy as np
import deep_sort.nn_matching
import deep_sort.preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import deep_sort.generate_detections as gdet


class Deep(object):
    def __init__(self, model="deep_sort/mars-small128.pb", max_distance=0.7, nn_budget=100,
                 nms_max_overlap=1.0, n_init=3, max_age=15, max_iou_distance=0.7):
        self.model = model
        self.distance_metric = "cosine"
        self.max_distance = max_distance
        self.nn_budget = nn_budget
        self.nms_max_overlap = nms_max_overlap
        self.batch_size = 1
        self.n_init = n_init
        self.max_age = max_age
        self.max_iou_distance = max_iou_distance
        self.encoder = gdet.create_box_encoder(self.model, batch_size=self.batch_size)
        self.metric = deep_sort.nn_matching.NearestNeighborDistanceMetric(self.distance_metric, self.max_distance, self.nn_budget)

    def sort_tracker(self):
        return Tracker(metric=self.metric, n_init=self.n_init, max_age=self.max_age, max_iou_distance=self.max_iou_distance)

    def Detection(self, boxes, scores, classes, features):
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(boxes, scores, classes, features)]
        # run non-maxima suppression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = deep_sort.preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        return detections