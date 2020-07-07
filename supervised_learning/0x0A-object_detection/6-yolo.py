#!/usr/bin/env python3
"""contains the Yolo class"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2
import os


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

    @staticmethod
    def iou(box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area

        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """

        :param filtered_boxes: umpy.ndarray of shape (?, 4)
            containing all of the filtered bounding boxes
        :param box_classes: numpy.ndarray of shape (?,)
            containing the class number for the class that
            filtered_boxes predicts
        :param box_scores: numpy.ndarray of shape (?)
            containing the box scores for each box in filtered_boxes
        :return: tuple of
            (box_predictions, predicted_box_classes, predicted_box_scores)
            box_predictions: a numpy.ndarray of shape (?, 4)
                    containing all of the predicted bounding boxes ordered by
                    class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,)
                containing the class number for box_predictions ordered by
                class and box score
            predicted_box_scores: a numpy.ndarray of shape (?)
                containing the box scores for box_predictions ordered by
                class and box score
        """
        index = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in index])
        predicted_box_classes = np.array([box_classes[i] for i in index])
        predicted_box_scores = np.array([box_scores[i] for i in index])

        _, class_counts = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        accumulated_count = 0

        for class_count in class_counts:
            while i < accumulated_count + class_count:
                j = i + 1
                while j < accumulated_count + class_count:
                    if self.iou(box_predictions[i],
                                box_predictions[j]) >= self.nms_t:
                        box_predictions = np.delete(box_predictions, j,
                                                    axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        predicted_box_classes = (np.delete
                                                 (predicted_box_classes,
                                                  j, axis=0))
                        class_count -= 1
                    else:
                        j += 1
                i += 1
            accumulated_count += class_count

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        :param folder_path: string representing the path to the folder
            holding all the images to load
        :return: Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(image) for image in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """
        :param images: list of images as numpy.ndarray
        :return: (pimages, image_shapes):
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model
                input_w: the input width for the Darknet model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2)
                containing the original height and width of the images
                2 => (image_height, image_width)
        """
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value

        pimages_list = []
        image_shapes_list = []

        for img in images:
            # save original image size
            img_shape = img.shape[0], img.shape[1]
            image_shapes_list.append(img_shape)

            # Resize the images with inter-cubic interpolation
            dim = (input_w, input_h)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

            # Rescale all images to have pixel values in the range [0, 1]
            pimage = resized / 255
            pimages_list.append(pimage)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and box scores
        :param image: numpy.ndarray containing an unprocessed image
        :param boxes: numpy.ndarray containing the boundary boxes for the image
        :param box_classes: numpy.ndarray containing the class indices
            for each box
        :param box_scores: numpy.ndarray containing the box scores for each box
        :param file_name: file path where the original image is stored
        :return:
        """
        for i in range(len(boxes)):
            # SCORES
            score = "{:.2f}".format(box_scores[i])

            # BOXES
            # Start coordinate, top left corner of image
            start_point = (int(boxes[i, 0]), int(boxes[i, 1]))
            # End coordinate bottom right corner of image
            end_point = (int(boxes[i, 2]), int(boxes[i, 3]))
            # Blue color in BGR
            color = (255, 0, 0)

            thickness = 2

            # Using cv2.line() method
            # Draw a diagonal green line with thickness of 9 px
            image = cv2.rectangle(image,
                                  start_point, end_point,
                                  color, thickness)

            # TEXT
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(boxes[i, 0]), int(boxes[i, 1] - 5))
            fontScale = 0.5

            # Red color in BGR
            color = (0, 0, 255)

            thickness = 1
            image = cv2.putText(image,
                                self.class_names[box_classes[i]] + score,
                                org, font, fontScale, color, thickness,
                                cv2.LINE_AA)

        # Displaying the image
        cv2.imshow(file_name, image)

        # cv2.waitKey(0)
        key = cv2.waitKey(0)
        if key == ord('s'):
            # If detections does not exist, create it
            os.mkdir('detections') if not os.path.isdir('detections') else None

            # Change the current directory
            os.chdir('detections')

            # Using cv2.imwrite() method to save the image
            cv2.imwrite(file_name, image)

            # Change back to working directory
            os.chdir('../')
        cv2.destroyAllWindows()
