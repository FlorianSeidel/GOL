__author__ = 'florian'

import matplotlib.pyplot as plt
import cv2

class Matplotlib_pROST_Visualizer:

    def __init__(self):
        fig = plt.figure()
        self.original = fig.add_subplot(221)
        self.background = fig.add_subplot(222)
        self.foreground = fig.add_subplot(223)
        self.segmentation = fig.add_subplot(224)
        plt.ion()
        plt.show()


    def show_original(self,image):
        self.original.imshow(image)

    def show_background(self,image):
        self.background.imshow(image)

    def show_foreground(self,image):
        self.foreground.imshow(image)

    def show_segmentation(self,image):
        self.segmentation.imshow(image)

    def update(self):
        plt.pause(0.001)

class OpenCV_pROST_Visualizer:

    def __init__(self):
        pass

    def show_original(self,image):
        cv2.imshow("Original",image)

    def show_background(self,image):
        cv2.imshow("Background",image)

    def show_foreground(self,image):
        cv2.imshow("Foreground",image)

    def show_segmentation(self,image):
        cv2.imshow("Segmentation",image)

    def update(self):
        cv2.waitKey(1)