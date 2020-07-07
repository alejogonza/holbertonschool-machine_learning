#!/usr/bin/env python3
"""utilities module"""

import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt


def load_images(images_path, as_array=True):
    """

    :param images_path: path to a directory from which to load images
    :param as_array: boolean indicating whether the images
        should be loaded as one numpy.ndarray
        If True, the images should be loaded as a numpy.ndarray
            of shape (m, h, w, c) where:
        m is the number of images
        h, w, and c are the height, width,
        and number of channels of all images, respectively
        If False, the images should be loaded as a list
        of individual numpy.ndarrays
    :return: images, filenames
        images is either a list/numpy.ndarray of all images
        filenames is a list of the filenames associated
        with each image in images
    """
    image_paths = glob.glob(images_path + '/*')
    image_paths = sorted([i for i in image_paths])

    filenames = [path.split('/')[-1] for path in image_paths]

    images_bgr = [cv2.imread(image) for image in image_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]

    if as_array:
        images = np.array(images)

    return images, filenames


def load_csv(csv_path, params={}):
    """
    loads the contents of a csv file as a list of lists
    :param csv_path: path to the csv to load
    :param params: parameters to load the csv with
    :return: list of lists representing the contents found in csv_path
    """
    with open(csv_path, 'r') as f:
        csv_lines = [line.strip() for line in f]

    csv_list = [line.split(',') for line in csv_lines]

    return csv_list


def save_images(path, images, filenames):
    """
    saves images to a specific path
    :param path: path to the directory in which the images should be saved
    :param images: list/numpy.ndarray of images to save
    :param filenames: list of filenames of the images to save
    :return: True on success and False on failure
    """

    try:
        # Change the current directory
        os.chdir(path)

        for i, image in enumerate(images):
            # Using cv2.imwrite() method to save the image
            cv2.imwrite(filenames[i], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Change back to working directory
        os.chdir('../')
        return True
    except FileNotFoundError:
        return False
