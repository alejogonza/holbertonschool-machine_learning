#!/usr/bin/env python3
"""contains the Yolo class"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor
        :param self:
        :param model_path: path to where a Darknet Keras model is stored
        :param classes_path: path to where the list of class names used
            for the Darknet model, listed in order of index, can be found
        :param class_t: float representing the box score threshold
            for the initial filtering step
        :param nms_t: float representing the IOU threshold
            for non-max suppression
        :param anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            outputs is the number of outputs (predictions)
                made by the Darknet model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """ calculates sigmoid function """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """

        :param self:
        :param outputs: list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image:
        :param image_size: numpy.ndarray containing the image’s original size
        :return: tuple of (boxes, box_confidences, box_class_probs)
            boxes: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)
                containing the processed boundary boxes for each output
                    4 => (x1, y1, x2, y2)
                    (x1, y1, x2, y2) should represent the boundary box
                    relative to original image
            box_confidences: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1)
                containing the box confidences for each output
            box_class_probs: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)
                containing the box’s class probabilities for each output
        """

        image_height, image_width = image_size[0], image_size[1]

        boxes = [output[..., 0:4] for output in outputs]

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            # BONDING BOX CENTER COORDINATES (x,y)
            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            # height
            indexes_y = np.arange(grid_height)
            indexes_y = indexes_y.reshape(grid_height, 1, 1)
            cy = c + indexes_y

            # width
            indexes_x = np.arange(grid_width)
            indexes_x = indexes_x.reshape(1, grid_width, 1)
            cx = c + indexes_x

            # darknet center coordinates output
            tx = (box[..., 0])
            ty = (box[..., 1])

            # normalized output
            tx_n = self.sigmoid(tx)
            ty_n = self.sigmoid(ty)

            # placement within grid
            bx = tx_n + cx
            by = ty_n + cy

            # normalize to grid
            bx /= grid_width
            by /= grid_height

            # BONDING BOX WIDTH AND HEIGHT (w, h)
            # darknet output
            tw = (box[..., 2])
            th = (box[..., 3])

            # log-space transformation
            tw_t = np.exp(tw)
            th_t = np.exp(th)

            # anchors box dimensions [anchor_box_width, anchor_box_height]
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # scale to anchors box dimensions
            bw = pw * tw_t
            bh = ph * th_t

            # normalizing to model input size
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height

            # BOUNDING BOX CORNER COORDINATES
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            # scaling to image size
            box[..., 0] = x1 * image_width
            box[..., 1] = y1 * image_height
            box[..., 2] = x2 * image_width
            box[..., 3] = y2 * image_height

        box_confidences = \
            [self.sigmoid(output[..., 4, np.newaxis]) for output in outputs]
        box_class_probs = \
            [self.sigmoid(output[..., 5:]) for output in outputs]

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """

        :param boxes: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output
        :param box_confidences: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing the processed box confidences for each output
        :param box_class_probs: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing the processed box class probabilities for each output
        :return: tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4)
                containing all of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing
                 the class number that each box in filtered_boxes predicts
            box_scores: a numpy.ndarray of shape (?)
                containing the box scores for each box in filtered_boxes
        """
        obj_thresh = self.class_t

        # box_scores
        box_scores_full = []
        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            box_scores_full.append(box_conf * box_class_prob)

        box_scores_list = [score.max(axis=3) for score in box_scores_full]
        box_scores_list = [score.reshape(-1) for score in box_scores_list]
        box_scores = np.concatenate(box_scores_list)

        index_to_delete = np.where(box_scores < obj_thresh)

        box_scores = np.delete(box_scores, index_to_delete)

        # box_classes
        box_classes_list = [box.argmax(axis=3) for box in box_scores_full]
        box_classes_list = [box.reshape(-1) for box in box_classes_list]
        box_classes = np.concatenate(box_classes_list)
        box_classes = np.delete(box_classes, index_to_delete)

        # filtered_boxes
        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(boxes_list, axis=0)
        filtered_boxes = np.delete(boxes, index_to_delete, axis=0)

        return filtered_boxes, box_classes, box_scores
