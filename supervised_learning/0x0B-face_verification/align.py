#!/usr/bin/env python3
"""contians the FaceAlign class"""

import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
import dlib


class FaceAlign:
    """
    Face Align class
    """

    def __init__(self, shape_predictor_path):
        """
        class constructor
        :param shape_predictor_path: path to the dlib shape predictor model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        detects a face in an image
        :param image: numpy.ndarray of rank 3 containing an image
            from which to detect a face
        :return: dlib.rectangle containing the boundary box
            for the face in the image,
            or None on failure
        """
        height, width, c = image.shape
        rectangle = dlib.rectangle(left=0, top=0, right=width, bottom=height)
        area = 0
        try:
            rects = self.detector(image, 1)

            for rect in rects:
                new_area = rect.area()
                if new_area > area:
                    area = new_area
                    rectangle = rect

            return rectangle

        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """
        finds facial landmarks
        :param image: numpy.ndarray of an image from which to
            find facial landmarks
        :param detection: dlib.rectangle containing the boundary box
            of the face in the image
        :return: Returns: a numpy.ndarray of shape (p, 2)
            containing the landmark points, or None on failure
            p is the number of landmark points
            2 is the x and y coordinates of the point
        """
        try:
            shape = self.shape_predictor(image, detection)

            # initialize the list of (x, y)-coordinates
            coords = np.zeros((68, 2))
            # loop over the 68 facial landmarks and convert them
            # to a 2-tuple of (x, y)-coordinates
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            # return the list of (x, y)-coordinates
            return coords

        except RuntimeError:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        aligns an image for face verification
        :param image: numpy.ndarray of rank 3 containing the
            image to be aligned
        :param landmark_indices: numpy.ndarray of shape (3,)
            containing the indices of the three landmark points
            that should be used for the affine transformation
        :param anchor_points: numpy.ndarray of shape (3, 2)
            containing the destination points for the affine transformation,
            scaled to the range [0, 1]
        :param size: desired size of the aligned image
        :return: numpy.ndarray of shape (size, size, 3)
            containing the aligned image,
            or None if no face is detected
        """
        rect = self.detect(image)
        coords = self.find_landmarks(image, rect)
        input_points = coords[landmark_indices]
        input_points = input_points.astype('float32')

        output_points = anchor_points * size

        warp_mat = cv2.getAffineTransform(input_points, output_points)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

        return warp_dst
